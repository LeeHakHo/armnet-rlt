"""ACT with a mixture-of-Gaussians action head (policy type ``act_gmm``).

The transformer, VAE and processors are LeRobot's ACT; only the action head
and the loss change. Training maximises the chunk likelihood under
``num_modes`` chunk trajectories (see :mod:`armnet_rlt.gmm`) plus ACT's KL
term; inference executes the likeliest trajectory rather than the mean.
Importing this module registers the type with LeRobot's config registry.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES

from armnet_rlt.gmm import GMMHead, chunk_log_prob, chunk_weights, pick_chunk


POLICY_TYPE = "act_gmm"


@PreTrainedConfig.register_subclass(POLICY_TYPE)
@dataclass
class ACTGMMConfig(ACTConfig):
    """ACT's configuration plus the mixture head.

    ``num_modes`` chunk trajectories; ``init_std`` is each mode's starting
    standard deviation in normalized action units; ``sample_mode`` draws the
    executed mode by its weight instead of taking the likeliest, and
    ``sample_noise`` adds the mode's Gaussian noise to it.
    """

    num_modes: int = 5
    init_std: float = 0.2
    min_log_std: float = -5.0
    max_log_std: float = 2.0
    sample_mode: bool = False
    sample_noise: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_modes < 1:
            raise ValueError("num_modes must be positive")
        if self.init_std <= 0:
            raise ValueError("init_std must be positive")


class ACTGMMPolicy(ACTPolicy):
    """ACT whose decoder output feeds a :class:`GMMHead`."""

    config_class = ACTGMMConfig
    name = POLICY_TYPE

    def __init__(self, config: ACTGMMConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        action_dim = int(config.action_feature.shape[0])
        self.model.action_head = GMMHead(
            config.dim_model,
            action_dim,
            config.num_modes,
            min_log_std=config.min_log_std,
            max_log_std=config.max_log_std,
        )

    @property
    def head(self) -> GMMHead:
        return self.model.action_head

    def _mixture(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor, tuple[Tensor, Tensor]]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        raw, latent = self.model(batch)
        means, log_std, logits = self.head.unpack(raw)
        return means, log_std, logits, latent

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Negative chunk log-likelihood per valid action element, plus ACT's KL."""
        means, log_std, logits, (mu_hat, log_sigma_x2_hat) = self._mixture(batch)
        valid = ~batch["action_is_pad"]
        mixture_ll, mode_ll = chunk_log_prob(means, log_std, logits, batch[ACTION], valid)
        elements = valid.sum(dim=1).clamp_min(1).to(mixture_ll.dtype) * means.shape[-1]
        nll = (-mixture_ll / elements).mean()
        with torch.no_grad():
            best = mode_ll.argmax(dim=-1)
            index = best.view(-1, 1, 1, 1).expand(-1, means.shape[1], 1, means.shape[-1])
            best_chunk = means.gather(2, index).squeeze(2)
            l1 = (
                (batch[ACTION] - best_chunk).abs() * valid.unsqueeze(-1)
            ).sum() / elements.sum()
            weights = chunk_weights(logits)
            entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1).mean()
        loss_dict = {
            "nll_loss": nll.item(),
            "l1_loss": l1.item(),
            "mode_entropy": entropy.item(),
            "std_mean": log_std.exp().mean().item(),
        }
        if self.config.use_vae:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp()))
                .sum(-1)
                .mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = nll + mean_kld * self.config.kl_weight
        else:
            loss = nll
        return loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """The likeliest mode's chunk (or a drawn one; see the config)."""
        self.eval()
        means, log_std, logits, _ = self._mixture(batch)
        chunk, _chosen = pick_chunk(
            means,
            log_std,
            logits,
            sample_mode=bool(self.config.sample_mode),
            sample_noise=bool(self.config.sample_noise),
        )
        return chunk

    @torch.no_grad()
    def mode_weights(self, batch: dict[str, Tensor]) -> Tensor:
        """Mixture weights ``(B, K)`` for a batch of observations."""
        self.eval()
        _means, _log_std, logits, _ = self._mixture(batch)
        return chunk_weights(logits)


def config_from_act(config: ACTConfig, **overrides: Any) -> ACTGMMConfig:
    """An ``ACTGMMConfig`` carrying every field of an ACT configuration."""
    values = {f.name: getattr(config, f.name) for f in fields(ACTConfig) if f.init}
    values.update(overrides)
    return ACTGMMConfig(**values)


def from_act_checkpoint(path: str | Path, *, device: str = "cpu", **overrides: Any) -> ACTGMMPolicy:
    """Build an ``act_gmm`` policy warm-started from a trained ACT checkpoint:
    its backbone, transformer and VAE weights, and every mode of the new head
    initialised from the old action head."""
    source = ACTPolicy.from_pretrained(str(path))
    config = config_from_act(source.config, **overrides)
    config.device = device
    policy = ACTGMMPolicy(config)
    state = {
        key: value
        for key, value in source.state_dict().items()
        if not key.startswith("model.action_head.")
    }
    missing, unexpected = policy.load_state_dict(state, strict=False)
    stray = [key for key in missing if not key.startswith("model.action_head.")]
    if stray or unexpected:
        raise RuntimeError(f"ACT weights do not fit: missing {stray[:5]}, unexpected {unexpected[:5]}")
    policy.head.init_from_linear(source.model.action_head, init_std=config.init_std)
    return policy.to(device)
