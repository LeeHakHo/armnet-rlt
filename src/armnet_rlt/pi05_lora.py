"""π0.5 with LoRA adapters: loading, serving to LeRobot actors, and BC training.

The base is a Pi0RL checkpoint (``run.sh``'s π0.5); LoRA adapters of rank 16 (PaliGemma)
and 32 (action expert) are added to every attention and feed-forward layer and are the
only trained parameters. A LoRA checkpoint is a ``pretrained_model`` directory with
``lora.npz`` (the adapters) and ``pi05_lora.json`` (base checkpoint and OpenPI config).
``lora_b`` starts at zero, so a fresh adapter reproduces the base policy.

``Pi05LoRABackend`` serves the policy over the LeRobot policy-server protocol, so
``--policy-type lerobot`` actors run it and stream their episodes to
``Pi05LoRATrainer`` (``lerobot_learner.run_streaming_bc``).
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from armnet_rlt.openpi_rlt import (
    OpenPIRLTPolicy,
    PolicyObservation,
    _install_training_stubs,
    load_norm_stats,
)
from armnet_rlt.remote_policy import PolicyBackend, decode_images

LORA_FILE = "lora.npz"
META_FILE = "pi05_lora.json"
LORA_PATTERN = ".*lora.*"
CAMERAS = ("front", "top", "wrist")
IMAGE_PREFIX = "observation.images."
# Height and width of the images actors send (the demo dataset's size).
IMAGE_SHAPE = (240, 426)
ACTIONS_PER_CALL = 10


def lora_train_config(config_name: str) -> Any:
    """``config_name`` with LoRA PaliGemma and action-expert variants."""
    import dataclasses

    _install_training_stubs()
    from openpi.training.config import get_config

    config = get_config(config_name)
    model = dataclasses.replace(
        config.model,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    return dataclasses.replace(config, model=model)


def resolve_checkpoint(path: str | Path) -> tuple[Path, Path | None, str | None]:
    """(base checkpoint, LoRA file or None, config name or None) for a base checkpoint
    or a LoRA ``pretrained_model`` directory."""
    path = Path(path)
    meta = path / META_FILE
    if meta.is_file():
        info = json.loads(meta.read_text())
        return Path(info["base_checkpoint"]), path / LORA_FILE, str(info["config_name"])
    return path, None, None


def _flat(tree: Mapping[Any, Any], prefix: tuple = ()) -> dict[tuple, Any]:
    out: dict[tuple, Any] = {}
    for key, value in tree.items():
        if isinstance(value, Mapping):
            out.update(_flat(value, prefix + (key,)))
        else:
            out[prefix + (key,)] = value
    return out


def _nest(flat: Mapping[tuple, Any]) -> dict[Any, Any]:
    tree: dict[Any, Any] = {}
    for path, value in flat.items():
        node = tree
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value
    return tree


def path_name(path: tuple) -> str:
    return "/".join(str(key) for key in path)


def _lora_init(name: str, shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """``lora_b`` zero, ``lora_a`` normal(0, 0.01) as in OpenPI's LoRAConfig."""
    if name.endswith("lora_b"):
        return np.zeros(shape, dtype=np.float32)
    return (0.01 * rng.standard_normal(shape)).astype(np.float32)


class LoRAModel:
    """A Pi0RL model with LoRA adapters, split into adapter and frozen state."""

    def __init__(self, model: Any) -> None:
        from flax import nnx
        from openpi.shared import nnx_utils

        self.graphdef, self.lora, self.rest = nnx.split(
            model, nnx_utils.PathRegex(LORA_PATTERN), ...
        )
        self._template = {
            path_name(path): path for path in _flat(self.lora.to_pure_dict())
        }

    def model(self, lora: Any = None) -> Any:
        from flax import nnx

        return nnx.merge(self.graphdef, self.lora if lora is None else lora, self.rest)

    def names(self) -> list[str]:
        return sorted(self._template)

    def arrays(self, lora: Any = None) -> dict[str, np.ndarray]:
        """Adapter values by ``/``-joined path."""
        flat = _flat((self.lora if lora is None else lora).to_pure_dict())
        return {path_name(path): np.asarray(value) for path, value in flat.items()}

    def replaced(self, arrays: Mapping[str, Any], *, partial: bool = False) -> Any:
        """Adapter state with ``arrays`` installed; ``partial`` keeps unnamed ones."""
        import jax.numpy as jnp

        current = _flat(self.lora.to_pure_dict())
        missing = set(self._template) - set(arrays)
        if missing and not partial:
            raise ValueError(f"{len(missing)} LoRA arrays missing, e.g. {sorted(missing)[0]}")
        unknown = set(arrays) - set(self._template)
        if unknown:
            raise ValueError(f"unknown LoRA arrays, e.g. {sorted(unknown)[0]}")
        for name, value in arrays.items():
            path = self._template[name]
            old = current[path]
            new = jnp.asarray(np.asarray(value), dtype=old.dtype)
            if new.shape != old.shape:
                raise ValueError(f"{name}: shape {new.shape}, expected {old.shape}")
            current[path] = new
        return _replace_pure(self.lora, _nest(current))


def _replace_pure(state: Any, pure: dict[Any, Any]) -> Any:
    """A copy of nnx ``state`` holding the values of the pure dict ``pure``."""
    import jax

    copy = jax.tree.map(lambda value: value, state)
    copy.replace_by_pure_dict(pure)
    return copy


def load_lora_model(
    config: Any, base_params: Mapping[Any, Any], lora: Mapping[str, Any] | None, *, seed: int = 0
) -> Any:
    """``config.model`` with ``base_params`` and the adapters in ``lora`` (fresh ones
    when None)."""
    import jax
    from flax import nnx

    shapes = nnx.eval_shape(config.model.create, jax.random.key(0))
    _, state = nnx.split(shapes)
    expected = _flat(state.to_pure_dict())
    base = _flat(base_params)
    rng = np.random.default_rng(seed)
    params: dict[tuple, Any] = {}
    for path, spec in expected.items():
        name = path_name(path)
        if "lora" in str(path[-1]):
            if lora is not None:
                if name not in lora:
                    raise ValueError(f"LoRA checkpoint has no {name}")
                params[path] = np.asarray(lora[name], dtype=np.float32)
            else:
                params[path] = _lora_init(str(path[-1]), tuple(spec.shape), rng)
        elif path in base:
            params[path] = base[path]
        else:
            raise ValueError(f"base checkpoint has no {name}")
    return config.model.load(_nest(params))


class Pi05LoRAPolicy(OpenPIRLTPolicy):
    """``OpenPIRLTPolicy`` whose LoRA adapters can be replaced while serving."""

    @classmethod
    def load(
        cls,
        checkpoint: str | Path,
        config_name: str,
        *,
        default_prompt: str | None = None,
        seed: int = 0,
    ) -> "Pi05LoRAPolicy":
        """A base checkpoint (fresh adapters) or a LoRA ``pretrained_model`` directory."""
        import jax
        from flax import nnx
        from openpi.policies.policy_config import create_trained_policy

        base_dir, lora_file, saved_config = resolve_checkpoint(checkpoint)
        from openpi.training.config import get_config

        config_name = saved_config or config_name
        base_config = lora_train_config(config_name)

        plain = get_config(config_name)
        data_config = plain.data.create(plain.assets_dirs, plain.model)
        norm_stats = load_norm_stats(base_dir, data_config)
        policy = create_trained_policy(
            plain, base_dir, default_prompt=default_prompt, norm_stats=norm_stats
        )
        _, base_state = nnx.split(policy._model)
        lora = dict(np.load(lora_file)) if lora_file is not None else None
        model = load_lora_model(base_config, base_state.to_pure_dict(), lora, seed=seed)
        policy._model = model
        policy._sample_actions = None
        instance = cls(
            policy,
            None,
            norm_stats=norm_stats,
            use_quantile_norm=bool(getattr(data_config, "use_quantile_norm", True)),
            uses_delta_actions=_delta(plain, data_config),
            model=model,
        )
        instance.config = base_config
        instance.config_name = config_name
        instance.base_checkpoint = base_dir
        instance.split = LoRAModel(model)
        graphdef = instance.split.graphdef

        def sample(lora: Any, rest: Any, rng: Any, observation: Any) -> Any:
            return nnx.merge(graphdef, lora, rest).sample_actions_with_rl_token(rng, observation)

        jitted = jax.jit(sample)
        instance._sample = lambda rng, observation: jitted(
            instance.split.lora, instance.split.rest, rng, observation
        )
        return instance

    def install(self, arrays: Mapping[str, Any], *, partial: bool = False) -> None:
        self.split.lora = self.split.replaced(arrays, partial=partial)


def _delta(config: Any, data_config: Any) -> bool:
    from armnet_rlt.openpi_rlt import uses_delta_actions

    return uses_delta_actions(config, data_config)


def save_lora(
    arrays: Mapping[str, np.ndarray], target: Path, *, base_checkpoint: Path, config_name: str
) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    np.savez(target / LORA_FILE, **{name: np.asarray(v, np.float32) for name, v in arrays.items()})
    (target / META_FILE).write_text(
        json.dumps({"base_checkpoint": str(base_checkpoint), "config_name": config_name}) + "\n"
    )
    return target


class Pi05LoRABackend(PolicyBackend):
    """A ``Pi05LoRAPolicy`` behind the policy server, in the LeRobot protocol: each
    request returns the first ``ACTIONS_PER_CALL`` actions of the chunk."""

    def __init__(
        self,
        policy: Pi05LoRAPolicy,
        *,
        prompt: str,
        learner_step: int | None = None,
        served_from_learner: bool = False,
    ) -> None:
        self._policy = policy
        self._prompt = prompt
        self.learner_step = learner_step
        self.served_from_learner = served_from_learner

    def info(self) -> dict[str, Any]:
        meta = {
            "policy_type": "lerobot",
            "lerobot_type": "pi05_lora",
            "chunked": True,
            "n_action_steps": ACTIONS_PER_CALL,
            "uses_task": True,
            "shapes": {IMAGE_PREFIX + camera: list(IMAGE_SHAPE) for camera in CAMERAS},
            "learner_step": -1 if self.learner_step is None else int(self.learner_step),
            "steering_from_learner": bool(self.served_from_learner),
        }
        return {"meta": json.dumps(meta)}

    def infer(self, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
        images = decode_images(arrays, str(arrays["codec"]))
        missing = [camera for camera in CAMERAS if IMAGE_PREFIX + camera not in images]
        if missing:
            raise ValueError(f"request has no image for {missing}")
        observation = PolicyObservation(
            state=np.asarray(arrays["state"], dtype=np.float32),
            images={camera: images[IMAGE_PREFIX + camera] for camera in CAMERAS},
            prompt=str(arrays["task"]) if "task" in arrays else self._prompt,
        )
        chunk = self._policy.infer(observation).action_chunk
        return {
            "actions": np.asarray(chunk[:ACTIONS_PER_CALL], dtype=np.float32),
            "policy_step": -1 if self.learner_step is None else int(self.learner_step),
        }

    def apply_weights(
        self, weights: Mapping[str, torch.Tensor], learner_step: int | None, partial: bool
    ) -> None:
        """Install adapters pushed by ``Pi05LoRATrainer``."""
        arrays = {name: value.float().numpy() for name, value in weights.items()}
        if arrays:
            self._policy.install(arrays, partial=partial)
        if learner_step is not None:
            self.learner_step = int(learner_step)


def training_sample(
    state: np.ndarray,
    images: Mapping[str, np.ndarray],
    actions: np.ndarray,
    prompt: str,
) -> dict[str, Any]:
    """One raw sample in the keys the policy's input transforms take."""
    sample: dict[str, Any] = {
        "observation.state": np.asarray(state, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "prompt": prompt,
    }
    for camera in CAMERAS:
        sample[IMAGE_PREFIX + camera] = np.asarray(images[camera], dtype=np.uint8)
    return sample


def episode_sample(
    episode: Mapping[str, Any], index: int, horizon: int, prompt: str
) -> dict[str, Any]:
    """Frame ``index`` of an actor episode with the next ``horizon`` actions (the last
    action repeats past the end)."""
    from armnet_rlt.lerobot_actor import decode_jpeg

    length = len(episode["action"])
    rows = np.clip(np.arange(index, index + horizon), 0, length - 1)
    images = {
        camera: decode_jpeg(episode["images"][IMAGE_PREFIX + camera][index]) for camera in CAMERAS
    }
    return training_sample(episode["state"][index], images, episode["action"][rows], prompt)


class DemoSamples:
    """Random frames of a LeRobot dataset as raw training samples."""

    def __init__(self, repo_id: str, horizon: int, prompt: str) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

        fps = LeRobotDatasetMetadata(repo_id).fps
        self.dataset = LeRobotDataset(
            repo_id,
            delta_timestamps={"action": [i / fps for i in range(horizon)]},
            video_backend="pyav",
        )
        self.prompt = prompt
        print(
            f"[lerobot sbc] base dataset {repo_id}: {self.dataset.num_episodes} episodes, "
            f"{self.dataset.num_frames} frames",
            flush=True,
        )

    def sample(self) -> dict[str, Any]:
        item = self.dataset[random.randrange(len(self.dataset))]
        images = {
            camera: (item[IMAGE_PREFIX + camera].permute(1, 2, 0).numpy() * 255.0)
            .round()
            .clip(0, 255)
            .astype(np.uint8)
            for camera in CAMERAS
        }
        return training_sample(
            item["observation.state"].numpy(), images, item["action"].numpy(), self.prompt
        )


class _LoRAWeights:
    """Every adapter, for ``_push_weights``; a push always carries all of them."""

    def __init__(self, trainer: "Pi05LoRATrainer") -> None:
        self._trainer = trainer
        self.last_push_tensors = 0

    def rebase(self) -> None:
        pass

    def mark_checkpoint(self) -> None:
        pass

    def actor_state_bytes(self) -> dict[str, torch.Tensor]:
        arrays = self._trainer.split.arrays(self._trainer.lora)
        self.last_push_tensors = len(arrays)
        return {name: torch.from_numpy(np.array(v, np.float32)) for name, v in arrays.items()}


class Pi05LoRATrainer:
    """BC on π0.5's LoRA adapters: flow-matching loss of OpenPI's Pi0 on batches of
    success frames and, with ``base_dataset``, demonstration frames."""

    def __init__(
        self,
        checkpoint: str | Path,
        config_name: str,
        *,
        task: str,
        batch_size: int,
        base_dataset: str = "",
        base_batch_fraction: float = 0.0,
        lr: float = 1e-4,
        grad_clip_norm: float = 1.0,
        seed: int = 42,
    ) -> None:
        import jax
        import optax

        self.policy = Pi05LoRAPolicy.load(checkpoint, config_name, default_prompt=task, seed=seed)
        self.split = self.policy.split
        self.lora = self.split.lora
        self.task = task
        self.horizon = int(self.policy.config.model.action_horizon)
        self.base_size = round(batch_size * base_batch_fraction) if base_batch_fraction else 0
        self.online_size = batch_size - self.base_size
        self.demos = DemoSamples(base_dataset, self.horizon, task) if self.base_size else None
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(grad_clip_norm), optax.adamw(lr, weight_decay=0.0)
        )
        self.opt_state = self.optimizer.init(self.lora)
        self.rng = jax.random.key(seed)
        self.pool = ThreadPoolExecutor(max_workers=8)
        self.weights = _LoRAWeights(self)
        self.kind = "pi05_lora"
        self.description = (
            f"pi05_lora from {checkpoint} ({len(self.split.names())} LoRA arrays, "
            f"{sum(v.size for v in self.split.arrays().values()):,} parameters)"
        )
        self.settings = f"lr={lr:g}, action horizon {self.horizon}"
        self._step = self._make_step()

    def close(self) -> None:
        self.pool.shutdown(wait=False)

    def _make_step(self) -> Any:
        import jax
        import jax.numpy as jnp
        import optax
        from flax import nnx
        from openpi.models import pi0 as _pi0

        graphdef, optimizer = self.split.graphdef, self.optimizer

        def loss_fn(lora: Any, rest: Any, rng: Any, observation: Any, actions: Any) -> Any:
            model = nnx.merge(graphdef, lora, rest)
            return jnp.mean(_pi0.Pi0.compute_loss(model, rng, observation, actions, train=True))

        def step(lora: Any, rest: Any, opt_state: Any, rng: Any, observation: Any, actions: Any):
            loss, grads = jax.value_and_grad(loss_fn)(lora, rest, rng, observation, actions)
            updates, opt_state = optimizer.update(grads, opt_state, lora)
            lora = optax.apply_updates(lora, updates)
            return lora, opt_state, loss, optax.global_norm(grads)

        return jax.jit(step)

    def _batch(self, successes: list[dict[str, Any]]) -> tuple[Any, Any]:
        import jax
        import jax.numpy as jnp
        from openpi.models import model as model_module

        lengths = [len(episode["state"]) for episode in successes]
        picks = [
            (episode, random.randrange(len(episode["state"])))
            for episode in random.choices(successes, weights=lengths, k=self.online_size)
        ]
        raw = list(
            self.pool.map(
                lambda pick: episode_sample(pick[0], pick[1], self.horizon, self.task), picks
            )
        )
        if self.demos is not None:
            raw += list(self.pool.map(lambda _: self.demos.sample(), range(self.base_size)))
        transform = self.policy._policy._input_transform
        items = list(self.pool.map(transform, raw))
        batch = jax.tree.map(lambda *values: jnp.stack([jnp.asarray(v) for v in values]), *items)
        actions = batch.pop("actions")
        return model_module.Observation.from_dict(batch), actions

    def train_step(self, successes: list[dict[str, Any]]) -> tuple[float, dict[str, float], float]:
        import jax

        observation, actions = self._batch(successes)
        self.rng, step_rng = jax.random.split(self.rng)
        self.lora, self.opt_state, loss, grad_norm = self._step(
            self.lora, self.split.rest, self.opt_state, step_rng, observation, actions
        )
        return float(loss), {}, float(grad_norm)

    def save(self, run_dir: Path, step: int) -> Path:
        target = run_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
        save_lora(
            self.split.arrays(self.lora),
            target,
            base_checkpoint=self.policy.base_checkpoint,
            config_name=self.policy.config_name,
        )
        last = run_dir / "checkpoints" / "last"
        last.unlink(missing_ok=True)
        last.symlink_to(target.parent.name)
        return target
