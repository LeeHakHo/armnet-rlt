"""Diffusion steering via RL (DSRL) on top of the frozen Pi0.5 flow policy.

The actor maps the RL token and proprioception to the initial noise of the
base policy's flow; the base policy turns that noise into the action chunk the
robot executes. Transitions therefore carry the noise as ``action`` and the
chunk the base policy produced from it as ``reference_action``.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor

from armnet_rlt.artifacts import NormStats
from armnet_rlt.config import RLTConfig, RLTNetworkConfig
from armnet_rlt.networks import (
    RLTCriticEnsemble,
    _norm,
    _register_norm_buffers,
    _set_state_buffers,
)

STEERING_SCHEMA_VERSION = 1


class DSRLActor(nn.Module):
    """``(rl_token, proprioception)`` to noise in ``[-bound, bound]^noise_dim``.

    ``forward`` returns the noise normalized to ``[-1, 1]`` like ``RLTActor``;
    ``unnormalize_action`` scales it by the bound. The reference chunk argument
    is accepted for interface parity and ignored.
    """

    def __init__(self, config: RLTNetworkConfig, *, noise_bound: float) -> None:
        super().__init__()
        if noise_bound <= 0:
            raise ValueError("noise_bound must be positive")
        self.noise_len = config.predicted_action_len
        self.per_step_action_dim = config.action_dim
        self.predicted_action_dim = config.predicted_action_dim
        self.reference_action_dim = config.reference_action_dim
        self.proprioception_dim = config.proprioception_dim
        self.register_buffer("noise_bound", torch.tensor(float(noise_bound)))

        input_dim = config.rl_token_dim + config.proprioception_dim
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

    def set_norm_stats(
        self, norm_stats: Mapping[str, NormStats], *, use_quantiles: bool = False
    ) -> None:
        self._use_quantiles.fill_(use_quantiles)
        _set_state_buffers(
            self, "ns", norm_stats["state"], self.proprioception_dim
        )

    def normalize_action(
        self, action: Tensor, proprioception: Tensor | None = None
    ) -> Tensor:
        return action / self.noise_bound

    def unnormalize_action(
        self, action: Tensor, proprioception: Tensor | None = None
    ) -> Tensor:
        return action * self.noise_bound

    def forward(
        self,
        rl_token: Tensor,
        proprioception: Tensor,
        reference_action: Tensor | None = None,
        *,
        sample: bool = False,
    ) -> tuple[Tensor, Tensor]:
        state = _norm(
            proprioception,
            self.ns_mean,
            self.ns_std,
            self.ns_q01,
            self.ns_q99,
            self._use_quantiles,
        )
        mean = torch.tanh(self.mean_head(self.trunk(torch.cat((rl_token, state), dim=-1))))
        return mean, mean


class DSRLPolicy(nn.Module):
    """TD3 over the base policy's noise; mirrors ``RLTPolicy``'s learner API."""

    def __init__(self, config: RLTConfig) -> None:
        super().__init__()
        if config.algorithm != "dsrl":
            raise ValueError("DSRLPolicy requires config.algorithm == 'dsrl'")
        self.config = config
        self.actor = DSRLActor(config.network, noise_bound=config.dsrl_noise_bound)
        self.critic_ensemble = RLTCriticEnsemble(
            config.network, num_critics=config.num_critics
        )
        self.critic_target = copy.deepcopy(self.critic_ensemble)
        self.critic_target.requires_grad_(False)
        self.predicted_action_dim = config.network.predicted_action_dim
        self.reference_action_dim = config.network.reference_action_dim
        self.chunk_discount = config.discount**config.actions_to_execute
        self.target_policy_noise = config.target_policy_noise
        self.target_noise_clip = config.target_noise_clip
        self.q_loss_normalization = config.q_loss_normalization
        self.q_loss_alpha = config.q_loss_alpha

    def set_norm_stats(
        self, norm_stats: Mapping[str, NormStats], *, use_quantiles: bool = False
    ) -> None:
        self.actor.set_norm_stats(norm_stats, use_quantiles=use_quantiles)
        self.critic_ensemble.set_norm_stats(norm_stats, use_quantiles=use_quantiles)
        self.critic_target.set_norm_stats(norm_stats, use_quantiles=use_quantiles)

    def compute_loss_bc(self, *args: Any, **kwargs: Any) -> Tensor:
        raise NotImplementedError("DSRL has no demonstration behavior cloning")

    @torch.no_grad()
    def select_action(
        self,
        rl_token: Tensor,
        proprioception: Tensor,
        reference_action: Tensor | None = None,
        *,
        deterministic: bool = False,
        exploration_std: float = 0.0,
    ) -> Tensor:
        """Noise for the base policy, in its own units."""
        _, mean = self.actor(rl_token, proprioception)
        noise = self.actor.unnormalize_action(mean)
        if not deterministic and exploration_std > 0:
            noise = noise + torch.randn_like(noise) * exploration_std
        bound = float(self.actor.noise_bound)
        return noise.clamp(-bound, bound)

    def compute_loss_critic(
        self,
        rl_token: Tensor,
        proprioception: Tensor,
        action: Tensor,
        reward: Tensor,
        next_rl_token: Tensor,
        next_proprioception: Tensor,
        next_reference_action: Tensor,
        done: Tensor,
    ) -> Tensor:
        if action.shape[-1] != self.predicted_action_dim:
            raise ValueError(
                f"action has last dimension {action.shape[-1]}, "
                f"expected {self.predicted_action_dim}"
            )
        normalized_action = self.actor.normalize_action(action).clamp(-1.0, 1.0)
        with torch.no_grad():
            _, next_mean = self.actor(next_rl_token, next_proprioception)
            smoothing = (
                torch.randn_like(next_mean) * self.target_policy_noise
            ).clamp(-self.target_noise_clip, self.target_noise_clip)
            next_action = (next_mean + smoothing).clamp(-1.0, 1.0)
            target_values = self.critic_target(
                next_rl_token, next_proprioception, next_action
            )
            target = reward + (1.0 - done) * self.chunk_discount * target_values.min(
                dim=0
            ).values
            if self.config.max_q is not None:
                target = target.clamp(-self.config.max_q, self.config.max_q)
        predictions = self.critic_ensemble(rl_token, proprioception, normalized_action)
        return functional.mse_loss(
            predictions, target.unsqueeze(0).expand_as(predictions), reduction="none"
        ).mean()

    def compute_loss_actor(
        self,
        rl_token: Tensor,
        proprioception: Tensor,
        reference_action: Tensor | None = None,
    ) -> Tensor:
        _, mean = self.actor(rl_token, proprioception)
        minimum_q = self.critic_ensemble(rl_token, proprioception, mean).min(dim=0).values
        if self.q_loss_normalization:
            scale = self.q_loss_alpha / minimum_q.detach().abs().mean().clamp_min(1e-6)
            return -(scale * minimum_q).mean()
        return -minimum_q.mean()

    @torch.no_grad()
    def update_target_networks(self) -> None:
        tau = self.config.critic_target_update_weight
        for target, source in zip(
            self.critic_target.parameters(),
            self.critic_ensemble.parameters(),
            strict=True,
        ):
            target.lerp_(source, tau)

    def actor_state_bytes(self) -> dict[str, Tensor]:
        return {
            name: value.detach().cpu()
            for name, value in self.actor.state_dict().items()
        }


def steering_payload(
    actor_state: Mapping[str, Any],
    *,
    network: RLTNetworkConfig,
    noise_bound: float,
    learner_step: int | None,
) -> dict[str, Any]:
    """Flat ``np.savez``-able description of a DSRL actor for a policy server."""
    meta = {
        "schema_version": STEERING_SCHEMA_VERSION,
        "network": asdict(network),
        "noise_bound": float(noise_bound),
        "learner_step": learner_step,
    }
    payload: dict[str, Any] = {"meta": json.dumps(meta)}
    for name, value in actor_state.items():
        array = value.detach().cpu().numpy() if isinstance(value, Tensor) else np.asarray(value)
        payload[f"param/{name}"] = array
    return payload


def actor_from_steering_payload(payload: Mapping[str, Any]) -> tuple[DSRLActor, int | None]:
    """Rebuild the actor a ``steering_payload`` describes, in eval mode on CPU."""
    meta = json.loads(str(payload["meta"]))
    if meta.get("schema_version") != STEERING_SCHEMA_VERSION:
        raise ValueError(f"unsupported steering schema {meta.get('schema_version')!r}")
    values = dict(meta["network"])
    for name in ("hidden_dims", "delta_action_mask"):
        if name in values:
            values[name] = tuple(values[name])
    actor = DSRLActor(RLTNetworkConfig(**values), noise_bound=float(meta["noise_bound"]))
    state = {
        key.removeprefix("param/"): torch.as_tensor(np.asarray(value))
        for key, value in payload.items()
        if key.startswith("param/")
    }
    actor.load_state_dict(state)
    actor.eval()
    step = meta.get("learner_step")
    return actor, (int(step) if step is not None else None)


def choose_noise(
    actor: DSRLActor,
    rl_token: np.ndarray,
    proprioception: np.ndarray,
    *,
    exploration_std: float,
    noise_bound: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """One ``(noise_len, action_dim)`` noise sample: the actor's mean plus
    Gaussian exploration, clipped to the bound."""
    with torch.inference_mode():
        _, mean = actor(
            torch.as_tensor(rl_token, dtype=torch.float32).reshape(1, -1),
            torch.as_tensor(proprioception, dtype=torch.float32).reshape(1, -1),
        )
    noise = actor.unnormalize_action(mean).numpy().reshape(
        actor.noise_len, actor.per_step_action_dim
    ).astype(np.float64)
    if exploration_std > 0:
        noise = noise + rng.normal(0.0, exploration_std, size=noise.shape)
    return np.clip(noise, -noise_bound, noise_bound).astype(np.float32)


def random_noise(
    shape: tuple[int, int],
    *,
    exploration_std: float,
    noise_bound: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Unsteered noise: zero mean plus Gaussian exploration, clipped to the bound."""
    noise = np.zeros(shape, dtype=np.float64)
    if exploration_std > 0:
        noise = noise + rng.normal(0.0, exploration_std, size=shape)
    return np.clip(noise, -noise_bound, noise_bound).astype(np.float32)
