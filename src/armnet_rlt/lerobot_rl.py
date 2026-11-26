"""RL token, DSRL noise steering and RLT reference chunks for LeRobot base policies
(SmolVLA, multi-task DiT), behind the same policy-server interface as ``OpenPIRLTPolicy``.

The RL token follows Pi0RL (envs/openpi/src/openpi/models/pi0_rl.py): a small
transformer encoder compresses the frozen policy's observation embeddings into one
vector through a learned query, and a causal decoder reconstructs the embeddings from
it. The embeddings are SmolVLA's VLM prefix output, or the DiT's CLIP image features
(one per camera and observation step). They are projected to ``RL_TOKEN_DIM`` so the
actor and learner networks are the same as for π0.5.

Chunks are ``CHUNK_LEN`` absolute joint targets. DSRL noise is ``(CHUNK_LEN, 6)``; it
fills the policy's noise tensor with its last row repeated over the remaining steps
and zeros in padded action dimensions. A diffusion DiT samples with DDIM (eta 0) on
its training schedule, so the initial noise determines the chunk.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

RL_TOKEN_DIM = 2048
CHUNK_LEN = 30
ACTION_DIM = 6
CAMERAS = ("front", "top", "wrist")


class _Block(nn.Module):
    """Pre-norm transformer block with a SwiGLU feed-forward."""

    def __init__(self, dim: int, heads: int, mlp_dim: int) -> None:
        super().__init__()
        self.heads = heads
        self.attn_norm = nn.RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.ffn_norm = nn.RMSNorm(dim)
        self.gate = nn.Linear(dim, mlp_dim)
        self.up = nn.Linear(dim, mlp_dim)
        self.down = nn.Linear(mlp_dim, dim)

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        b, s, d = x.shape
        q, k, v = self.qkv(self.attn_norm(x)).reshape(b, s, 3, self.heads, d // self.heads).unbind(2)
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=None if mask is None else mask[:, None],
        )
        x = x + self.out(attn.transpose(1, 2).reshape(b, s, d))
        h = self.ffn_norm(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class RLTokenModel(nn.Module):
    """Encoder-decoder over ``(b, M, in_dim)`` embeddings; the token is ``(b, width)``."""

    def __init__(
        self, in_dim: int, width: int = RL_TOKEN_DIM, heads: int = 8, mlp_dim: int = 4096, layers: int = 2
    ) -> None:
        super().__init__()
        self.config = {"in_dim": in_dim, "width": width, "heads": heads, "mlp_dim": mlp_dim, "layers": layers}
        self.in_proj = nn.Linear(in_dim, width)
        self.query = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.encoder = nn.ModuleList(_Block(width, heads, mlp_dim) for _ in range(layers))
        self.decoder = nn.ModuleList(_Block(width, heads, mlp_dim) for _ in range(layers))
        self.out_proj = nn.Linear(width, in_dim)

    def encode(self, embeddings: Tensor, mask: Tensor | None = None) -> Tensor:
        b = embeddings.shape[0]
        x = torch.cat([self.in_proj(embeddings), self.query.expand(b, 1, -1)], dim=1)
        attn = None
        if mask is not None:
            valid = torch.cat([mask.bool(), mask.new_ones(b, 1, dtype=torch.bool)], dim=1)
            attn = valid[:, None, :] & valid[:, :, None]
        for block in self.encoder:
            x = block(x, attn)
        return x[:, -1]

    def reconstruction_loss(self, embeddings: Tensor, mask: Tensor | None = None) -> Tensor:
        """Mean squared L2 error of the teacher-forced reconstruction over valid tokens."""
        b, m, _ = embeddings.shape
        token = self.encode(embeddings, mask)
        x = torch.cat([token[:, None], self.in_proj(embeddings[:, :-1])], dim=1)
        attn = torch.tril(torch.ones(m, m, dtype=torch.bool, device=x.device))[None].expand(b, m, m)
        if mask is not None:
            keys = torch.cat([mask.new_ones(b, 1, dtype=torch.bool), mask[:, :-1].bool()], dim=1)
            attn = attn & keys[:, None, :]
        for block in self.decoder:
            x = block(x, attn)
        error = (self.out_proj(x) - embeddings).square().sum(-1)
        if mask is None:
            return error.mean()
        weight = mask.float()
        return (error * weight).sum() / weight.sum().clamp_min(1.0)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "rl_token_config.json").write_text(json.dumps(self.config))
        torch.save(self.state_dict(), directory / "rl_token.pt")

    @classmethod
    def load(cls, directory: str | Path, device: str = "cpu") -> "RLTokenModel":
        directory = Path(directory)
        model = cls(**json.loads((directory / "rl_token_config.json").read_text()))
        model.load_state_dict(torch.load(directory / "rl_token.pt", map_location=device))
        return model.to(device).eval()


def _policy_kind(policy: Any) -> str:
    kind = str(policy.config.type)
    if kind not in ("smolvla", "multi_task_dit"):
        raise ValueError(f"no RL token support for LeRobot policy type {kind!r}")
    return kind


def prepare_inference_batch(policy: Any, batch: dict[str, Tensor]) -> dict[str, Tensor]:
    """One preprocessed frame stacked with the policy's observation history."""
    from lerobot.policies.utils import populate_queues
    from lerobot.utils.constants import ACTION

    batch = policy._prepare_batch(dict(batch))
    policy._queues = populate_queues(policy._queues, batch, exclude_keys=[ACTION])
    for key in list(batch):
        if key in policy._queues and key != ACTION:
            batch[key] = torch.stack(list(policy._queues[key]), dim=1)
    return batch


def _dit_image_features(policy: Any, batch: dict[str, Tensor]) -> Tensor:
    """CLIP features ``(b, steps * cameras, F)`` of a DiT batch with stacked images."""
    import einops
    from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

    encoder = policy.observation_encoder
    b, steps = batch[OBS_STATE].shape[:2]
    images = batch[OBS_IMAGES]
    if images.ndim == 5:
        images = images.unsqueeze(1)
    cameras = images.shape[2]
    if policy.config.use_separate_rgb_encoder_per_camera:
        per_camera = []
        for index in range(cameras):
            flat = einops.rearrange(images[:, :, index], "b s c h w -> (b s) c h w")
            feats = encoder.vision_encoders[index](encoder._apply_preprocessing(flat)).flatten(1)
            per_camera.append(einops.rearrange(feats, "(b s) f -> b s f", b=b, s=steps))
        feats = torch.stack(per_camera, dim=2)
    else:
        flat = einops.rearrange(images, "b s n c h w -> (b s n) c h w")
        feats = encoder.vision_encoder(encoder._apply_preprocessing(flat)).flatten(1)
        feats = einops.rearrange(feats, "(b s n) f -> b s n f", b=b, s=steps, n=cameras)
    return feats.reshape(b, steps * cameras, -1).float()


def _smolvla_prefix(policy: Any, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Any]:
    """SmolVLA's VLM prefix output, its padding mask and the key-value cache."""
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    model = policy.model
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    prefix_embs, pad_masks, att_masks = model.embed_prefix(
        images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK], state=state
    )
    outputs, cache = model.vlm_with_expert.forward(
        attention_mask=make_att_2d_masks(pad_masks, att_masks),
        position_ids=torch.cumsum(pad_masks, dim=1) - 1,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
        fill_kv_cache=True,
    )
    return outputs[0], pad_masks, cache


def embeddings(policy: Any, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor | None]:
    """The embeddings the RL token compresses, and their validity mask (None: all valid)."""
    if _policy_kind(policy) == "smolvla":
        prefix, pad, _ = _smolvla_prefix(policy, batch)
        return prefix.float(), pad
    return _dit_image_features(policy, batch), None


def _full_noise(noise: Tensor, rows: int, columns: int) -> Tensor:
    """``(b, CHUNK_LEN, 6)`` noise to ``(b, rows, columns)``: last row repeated, zero columns."""
    b, given, dim = noise.shape
    if rows > given:
        noise = torch.cat([noise, noise[:, -1:].expand(b, rows - given, dim)], dim=1)
    full = noise.new_zeros(b, rows, columns)
    full[:, :, :dim] = noise[:, :rows]
    return full


def _ddim(objective: Any) -> Any:
    from diffusers import DDIMScheduler

    config = objective.noise_scheduler.config
    return DDIMScheduler(
        num_train_timesteps=config.num_train_timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        beta_schedule=config.beta_schedule,
        clip_sample=config.clip_sample,
        clip_sample_range=getattr(config, "clip_sample_range", 1.0),
        prediction_type=config.prediction_type,
    )


def sample_chunk(
    policy: Any, batch: dict[str, Tensor], noise: Tensor, *, prefix: tuple[Tensor, Tensor, Any] | None = None
) -> Tensor:
    """Normalized actions ``(b, CHUNK_LEN, 6)`` from the given ``(b, CHUNK_LEN, 6)`` noise."""
    if _policy_kind(policy) == "smolvla":
        model = policy.model
        _, pad_masks, cache = prefix if prefix is not None else _smolvla_prefix(policy, batch)
        x = _full_noise(noise, model.config.chunk_size, model.config.max_action_dim).to(pad_masks.device)
        steps = model.config.num_steps
        for step in range(steps):
            time = torch.full((x.shape[0],), 1.0 - step / steps, dtype=torch.float32, device=x.device)
            v = model.denoise_step(x_t=x, prefix_pad_masks=pad_masks, past_key_values=cache, timestep=time)
            x = x - v / steps
        return x[:, :CHUNK_LEN, :ACTION_DIM]
    from lerobot.utils.constants import OBS_STATE

    conditioning = policy.observation_encoder.encode(batch)
    objective = policy.objective
    model = policy.noise_predictor
    x = _full_noise(noise, objective.horizon, objective.action_dim).to(conditioning)
    if hasattr(objective, "noise_scheduler"):
        scheduler = _ddim(objective)
        scheduler.set_timesteps(objective.num_inference_steps)
        for t in scheduler.timesteps:
            out = model(x, torch.full(x.shape[:1], t, dtype=torch.long, device=x.device), conditioning_vec=conditioning)
            x = scheduler.step(out, t, x, eta=0.0).prev_sample
    else:
        steps = objective.config.num_integration_steps
        grid = torch.linspace(0, 1, steps + 1, device=x.device)
        for i in range(steps):
            t = grid[i].expand(x.shape[0])
            x = x + (grid[i + 1] - grid[i]) * model(x, t, conditioning_vec=conditioning)
    start = batch[OBS_STATE].shape[1] - 1
    chunk = x[:, start : start + CHUNK_LEN]
    if chunk.shape[1] < CHUNK_LEN:
        raise ValueError(f"DiT horizon gives {chunk.shape[1]} actions after the history, need {CHUNK_LEN}")
    return chunk


class LeRobotRLPolicy:
    """A frozen LeRobot policy with an RL token, served like ``OpenPIRLTPolicy``.

    ``norm_stats``, ``use_quantile_norm`` and ``uses_delta_actions`` are the π0.5
    task's, so the actor normalizes chunks the same way for every base policy.
    """

    rl_token_dim = RL_TOKEN_DIM
    noise_action_dim = ACTION_DIM

    def __init__(
        self,
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        rl_token: RLTokenModel,
        *,
        norm_stats: dict,
        use_quantile_norm: bool,
        uses_delta_actions: bool,
        rename_map: dict[str, str] | None = None,
        noise_seed: int | None = None,
    ) -> None:
        from armnet_rlt.dsrl import actor_from_steering_payload  # noqa: F401

        _policy_kind(policy)
        self._policy = policy.eval()
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._rl_token = rl_token.eval()
        self.norm_stats = norm_stats
        self.use_quantile_norm = use_quantile_norm
        self.uses_delta_actions = uses_delta_actions
        self._rename = rename_map or {}
        self._noise_rng = np.random.default_rng(noise_seed)
        self._steering: Any = None
        self.steering_step: int | None = None
        self._device = next(policy.parameters()).device

    @classmethod
    def from_checkpoint(
        cls,
        policy_path: str,
        rl_token_dir: str,
        norm_stats_file: str,
        *,
        device: str = "cuda",
        noise_seed: int | None = None,
    ) -> "LeRobotRLPolicy":
        """``norm_stats_file`` is what ``export_norm_stats`` wrote for the π0.5 task."""
        from armnet_rlt.lerobot_actor import load_lerobot_policy, policy_rename_map
        from armnet_rlt.remote_policy import unpack_norm_stats

        policy, pre, post = load_lerobot_policy(policy_path, device)
        with np.load(norm_stats_file) as arrays:
            stats = {key: arrays[key] for key in arrays.files}
        meta = json.loads(str(stats.pop("meta")))
        return cls(
            policy,
            pre,
            post,
            RLTokenModel.load(rl_token_dir, device),
            norm_stats=unpack_norm_stats(stats),
            use_quantile_norm=bool(meta["use_quantile_norm"]),
            uses_delta_actions=bool(meta["uses_delta_actions"]),
            rename_map=policy_rename_map(policy_path),
            noise_seed=noise_seed,
        )

    def _batch(self, observation: Any) -> dict[str, Tensor]:
        from armnet_rlt.lerobot_actor import policy_frame

        images = {f"observation.images.{name}": image for name, image in observation.images.items()}
        frame = policy_frame(np.asarray(observation.state, dtype=np.float32), images, observation.prompt)
        return prepare_inference_batch(self._policy, self._preprocessor(frame))

    def set_steering(self, payload: dict[str, Any]) -> int | None:
        from armnet_rlt.dsrl import actor_from_steering_payload

        actor, step = actor_from_steering_payload(payload)
        if (actor.noise_len, actor.per_step_action_dim) != self.noise_shape():
            raise ValueError(
                f"steering actor emits {(actor.noise_len, actor.per_step_action_dim)} noise, "
                f"the base policy takes {self.noise_shape()}"
            )
        self._steering = actor
        self.steering_step = step
        return step

    def noise_shape(self) -> tuple[int, int]:
        return CHUNK_LEN, ACTION_DIM

    @torch.inference_mode()
    def infer(self, observation: Any, *, exploration_std: float | None = None, steer: bool = True) -> Any:
        from armnet_rlt.dsrl import choose_noise, random_noise
        from armnet_rlt.openpi_rlt import PolicyOutput

        batch = self._batch(observation)
        prefix = None
        if _policy_kind(self._policy) == "smolvla":
            prefix = _smolvla_prefix(self._policy, batch)
            emb, mask = prefix[0].float(), prefix[1]
        else:
            emb, mask = embeddings(self._policy, batch)
        token = self._rl_token.encode(emb, mask)[0].float().cpu().numpy()
        state = np.asarray(observation.state, dtype=np.float32)
        chosen = None
        if exploration_std is None:
            raw = torch.randn(1, CHUNK_LEN, ACTION_DIM, device=self._device)
        else:
            if steer and self._steering is None:
                raise RuntimeError("no steering actor has been set on this policy")
            if steer:
                chosen = choose_noise(
                    self._steering, token, state, exploration_std=exploration_std,
                    noise_bound=float(self._steering.noise_bound), rng=self._noise_rng,
                )
            else:
                bound = float(self._steering.noise_bound) if self._steering is not None else math.inf
                chosen = random_noise(
                    self.noise_shape(), exploration_std=exploration_std, noise_bound=bound, rng=self._noise_rng
                )
            raw = torch.as_tensor(chosen, device=self._device)[None]
        normalized = sample_chunk(self._policy, batch, raw, prefix=prefix)
        chunk = self._postprocessor(normalized.reshape(-1, ACTION_DIM)).reshape(CHUNK_LEN, ACTION_DIM)
        return PolicyOutput(
            action_chunk=chunk.float().cpu().numpy(),
            rl_token=token.astype(np.float32),
            noise=chosen,
            steering_step=self.steering_step if (chosen is not None and steer) else None,
        )

    def reset(self) -> None:
        self._policy.reset()


def train_rl_token(
    policy_path: str,
    dataset: str,
    out_dir: str,
    *,
    steps: int = 5000,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cuda",
    seed: int = 42,
    log_every: int = 100,
) -> dict[str, float]:
    """Train an RL token on the frozen policy's embeddings of every frame of ``dataset``;
    5% of the episodes are held out. Returns the final train and held-out losses and the
    held-out loss with the token zeroed and shuffled (``zero_gap``, ``shuffle_gap``)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    from armnet_rlt.lerobot_actor import load_lerobot_policy

    torch.manual_seed(seed)
    policy, pre, _ = load_lerobot_policy(policy_path, device)
    policy.eval()
    meta = LeRobotDatasetMetadata(dataset)
    indices = policy.config.observation_delta_indices or [0]
    deltas = {
        key: [i / meta.fps for i in indices]
        for key in meta.features
        if key.startswith("observation.")
    }
    data = LeRobotDataset(dataset, delta_timestamps=deltas)
    episodes = np.asarray(data.hf_dataset["episode_index"])
    held = set(np.random.default_rng(seed).choice(np.unique(episodes), max(1, len(np.unique(episodes)) // 20), replace=False).tolist())
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=4)
    train_emb: list[tuple[Tensor, Tensor | None]] = []
    held_emb: list[tuple[Tensor, Tensor | None]] = []
    with torch.inference_mode():
        for raw in loader:
            episode = raw["episode_index"]
            raw = {k: v for k, v in raw.items() if not k.startswith("action")}
            batch = pre(raw)
            if _policy_kind(policy) == "multi_task_dit":
                batch = policy._prepare_batch(batch)
            emb, mask = embeddings(policy, batch)
            for i in range(emb.shape[0]):
                item = (emb[i].half().cpu(), None if mask is None else mask[i].cpu())
                (held_emb if int(episode[i]) in held else train_emb).append(item)
    del policy
    torch.cuda.empty_cache()
    model = RLTokenModel(in_dim=train_emb[0][0].shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)

    def stack(items: list[tuple[Tensor, Tensor | None]]) -> tuple[Tensor, Tensor | None]:
        emb = torch.stack([e for e, _ in items]).float().to(device)
        masks = [m for _, m in items]
        return emb, None if masks[0] is None else torch.stack(masks).to(device)

    loss = torch.tensor(0.0)
    for step in range(1, steps + 1):
        emb, mask = stack([train_emb[i] for i in rng.integers(0, len(train_emb), batch_size)])
        loss = model.reconstruction_loss(emb, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % log_every == 0:
            print(f"[rl-token] step {step} loss {float(loss):.4f}", flush=True)
    model.eval()
    results = {"train_loss": float(loss)}
    with torch.inference_mode():
        emb, mask = stack(held_emb)
        real = float(model.reconstruction_loss(emb, mask))
        token_encode = model.encode

        def with_token(fn: Any) -> float:
            model.encode = lambda e, m=None: fn(token_encode(e, m))
            try:
                return float(model.reconstruction_loss(emb, mask))
            finally:
                model.encode = token_encode

        results.update(
            held_out_loss=real,
            zero_gap=with_token(torch.zeros_like) - real,
            shuffle_gap=with_token(lambda t: t.roll(1, 0)) - real,
            train_frames=len(train_emb),
            held_out_frames=len(held_emb),
        )
    model.save(out_dir)
    (Path(out_dir) / "rl_token_metrics.json").write_text(json.dumps(results, indent=2))
    print(f"[rl-token] saved {out_dir}: {results}", flush=True)
    return results
