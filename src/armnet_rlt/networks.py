from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from armnet_rlt.artifacts import NormStats
from armnet_rlt.config import RLTNetworkConfig


def _norm(
    value: Tensor,
    mean: Tensor,
    std: Tensor,
    q01: Tensor,
    q99: Tensor,
    use_quantiles: Tensor,
) -> Tensor:
    if bool(use_quantiles.item()):
        return (value - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    return (value - mean) / (std + 1e-6)


def _unnorm(
    value: Tensor,
    mean: Tensor,
    std: Tensor,
    q01: Tensor,
    q99: Tensor,
    use_quantiles: Tensor,
) -> Tensor:
    if bool(use_quantiles.item()):
        return (value + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    return value * (std + 1e-6) + mean


def _register_norm_buffers(module: nn.Module, prefix: str, size: int) -> None:
    module.register_buffer(f"{prefix}_mean", torch.zeros(size))
    module.register_buffer(f"{prefix}_std", torch.ones(size))
    module.register_buffer(f"{prefix}_q01", torch.full((size,), -1.0))
    module.register_buffer(f"{prefix}_q99", torch.ones(size))


def _expand_action_stat(
    value: np.ndarray | None,
    *,
    chunk_len: int,
    action_dim: int,
    fill: float,
) -> np.ndarray:
    if value is None:
        return np.full(chunk_len * action_dim, fill, dtype=np.float32)
    array = np.squeeze(value)
    if array.ndim == 1:
        return np.tile(array[:action_dim], chunk_len).astype(np.float32)
    return array[:chunk_len, :action_dim].reshape(-1).astype(np.float32)


def _set_state_buffers(
    module: nn.Module, prefix: str, stats: NormStats, size: int
) -> None:
    for name, value in (("mean", stats.mean), ("std", stats.std)):
        target = getattr(module, f"{prefix}_{name}")
        target.copy_(torch.from_numpy(np.squeeze(value).astype(np.float32)[:size].copy()))
    for name, value in (("q01", stats.q01), ("q99", stats.q99)):
        if value is not None:
            target = getattr(module, f"{prefix}_{name}")
            target.copy_(
                torch.from_numpy(np.squeeze(value).astype(np.float32)[:size].copy())
            )


def _set_action_buffers(
    module: nn.Module,
    prefix: str,
    stats: NormStats,
    *,
    chunk_len: int,
    action_dim: int,
) -> None:
    values = (
        ("mean", stats.mean, 0.0),
        ("std", stats.std, 1.0),
        ("q01", stats.q01, -1.0),
        ("q99", stats.q99, 1.0),
    )
    for name, value, fill in values:
        target = getattr(module, f"{prefix}_{name}")
        expanded = _expand_action_stat(
            value, chunk_len=chunk_len, action_dim=action_dim, fill=fill
        )
        target.copy_(torch.from_numpy(expanded.copy()))


class RLTActor(nn.Module):
    def __init__(self, config: RLTNetworkConfig, *, delta_actions: bool = False):
        super().__init__()
        self.predicted_action_len = config.predicted_action_len
        self.reference_action_len = config.reference_action_len
        self.per_step_action_dim = config.action_dim
        self.predicted_action_dim = config.predicted_action_dim
        self.reference_action_dim = config.reference_action_dim
        self.proprioception_dim = config.proprioception_dim
        self._delta_actions = delta_actions

        self.register_buffer(
            "action_std",
            torch.full((self.predicted_action_dim,), config.fixed_action_std),
        )
        mask = torch.tensor(config.delta_action_mask, dtype=torch.float32)
        self.register_buffer(
            "delta_prop_mask", mask.repeat(self.predicted_action_len)
        )
        self.register_buffer(
            "delta_prop_mask_ref", mask.repeat(self.reference_action_len)
        )

        input_dim = (
            config.rl_token_dim
            + config.proprioception_dim
            + self.reference_action_dim
        )
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        previous = input_dim
        for hidden in config.hidden_dims:
            layers.extend((nn.Linear(previous, hidden), nn.SiLU()))
            previous = hidden
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(previous, self.predicted_action_dim)
        if config.init_final is not None:
            nn.init.uniform_(
                self.mean_head.weight, -config.init_final, config.init_final
            )
            nn.init.uniform_(
                self.mean_head.bias, -config.init_final, config.init_final
            )

        self.register_buffer("_use_quantiles", torch.tensor(False))
        _register_norm_buffers(self, "ns", config.proprioception_dim)
        _register_norm_buffers(self, "na", self.predicted_action_dim)
        _register_norm_buffers(self, "na_ref", self.reference_action_dim)

    def set_norm_stats(
        self, norm_stats: dict[str, NormStats], *, use_quantiles: bool = False
    ) -> None:
        self._use_quantiles.fill_(use_quantiles)
        _set_state_buffers(
            self, "ns", norm_stats["state"], self.proprioception_dim
        )
        _set_action_buffers(
            self,
            "na",
            norm_stats["actions"],
            chunk_len=self.predicted_action_len,
            action_dim=self.per_step_action_dim,
        )
        _set_action_buffers(
            self,
            "na_ref",
            norm_stats["actions"],
            chunk_len=self.reference_action_len,
            action_dim=self.per_step_action_dim,
        )

    def _proprioception_chunk(
        self, proprioception: Tensor, *, reference: bool
    ) -> Tensor:
        length = self.reference_action_len if reference else self.predicted_action_len
        mask = self.delta_prop_mask_ref if reference else self.delta_prop_mask
        return (
            proprioception[:, : self.per_step_action_dim].repeat(1, length) * mask
        )

    def normalize_action(
        self, action: Tensor, proprioception: Tensor | None = None
    ) -> Tensor:
        if self._delta_actions:
            if proprioception is None:
                raise ValueError("proprioception is required for delta actions")
            action = action - self._proprioception_chunk(
                proprioception, reference=False
            )
        return _norm(
            action,
            self.na_mean,
            self.na_std,
            self.na_q01,
            self.na_q99,
            self._use_quantiles,
        )

    def unnormalize_action(
        self, action: Tensor, proprioception: Tensor | None = None
    ) -> Tensor:
        raw = _unnorm(
            action,
            self.na_mean,
            self.na_std,
            self.na_q01,
            self.na_q99,
            self._use_quantiles,
        )
        if self._delta_actions:
            if proprioception is None:
                raise ValueError("proprioception is required for delta actions")
            raw = raw + self._proprioception_chunk(
                proprioception, reference=False
            )
        return raw

    def _normalize_reference(
        self, reference_action: Tensor, proprioception: Tensor
    ) -> Tensor:
        if self._delta_actions:
            reference_action = reference_action - self._proprioception_chunk(
                proprioception, reference=True
            )
        return _norm(
            reference_action,
            self.na_ref_mean,
            self.na_ref_std,
            self.na_ref_q01,
            self.na_ref_q99,
            self._use_quantiles,
        )

    def forward(
        self,
        rl_token: Tensor,
        proprioception: Tensor,
        reference_action: Tensor,
        *,
        sample: bool = True,
    ) -> tuple[Tensor, Tensor]:
        if reference_action.shape[-1] != self.reference_action_dim:
            raise ValueError(
                "reference_action has last dimension "
                f"{reference_action.shape[-1]}, expected {self.reference_action_dim}"
            )
        state = _norm(
            proprioception,
            self.ns_mean,
            self.ns_std,
            self.ns_q01,
            self.ns_q99,
            self._use_quantiles,
        )
        reference = self._normalize_reference(reference_action, proprioception)
        hidden = self.trunk(torch.cat((rl_token, state, reference), dim=-1))
        mean = torch.tanh(self.mean_head(hidden))
        if sample and self.training:
            action = (mean + torch.randn_like(mean) * self.action_std).clamp(
                -1.0, 1.0
            )
        else:
            action = mean
        return action, mean


class RLTCritic(nn.Module):
    def __init__(self, config: RLTNetworkConfig):
        super().__init__()
        self.predicted_action_dim = config.predicted_action_dim
        input_dim = (
            config.rl_token_dim
            + config.proprioception_dim
            + config.predicted_action_dim
        )
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        previous = input_dim
        for hidden in config.hidden_dims:
            layers.extend((nn.Linear(previous, hidden), nn.SiLU()))
            previous = hidden
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)
        self.register_buffer("_use_quantiles", torch.tensor(False))
        _register_norm_buffers(self, "ns", config.proprioception_dim)

    def set_norm_stats(
        self, norm_stats: dict[str, NormStats], *, use_quantiles: bool = False
    ) -> None:
        self._use_quantiles.fill_(use_quantiles)
        _set_state_buffers(
            self, "ns", norm_stats["state"], self.ns_mean.shape[0]
        )

    def forward(
        self, rl_token: Tensor, proprioception: Tensor, action: Tensor
    ) -> Tensor:
        if action.shape[-1] != self.predicted_action_dim:
            raise ValueError(
                f"action has last dimension {action.shape[-1]}, "
                f"expected {self.predicted_action_dim}"
            )
        state = _norm(
            proprioception,
            self.ns_mean,
            self.ns_std,
            self.ns_q01,
            self.ns_q99,
            self._use_quantiles,
        )
        return self.net(torch.cat((rl_token, state, action), dim=-1)).squeeze(-1)


class RLTCriticEnsemble(nn.Module):
    def __init__(self, config: RLTNetworkConfig, num_critics: int = 2):
        super().__init__()
        self.critics = nn.ModuleList(
            RLTCritic(config) for _ in range(num_critics)
        )

    def set_norm_stats(
        self, norm_stats: dict[str, NormStats], *, use_quantiles: bool = False
    ) -> None:
        for critic in self.critics:
            critic.set_norm_stats(norm_stats, use_quantiles=use_quantiles)

    def forward(
        self, rl_token: Tensor, proprioception: Tensor, action: Tensor
    ) -> Tensor:
        return torch.stack(
            [
                critic(rl_token, proprioception, action)
                for critic in self.critics
            ]
        )
