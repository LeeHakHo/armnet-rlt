"""Diffusion steering via RL (DSRL) on top of the frozen Pi0.5 flow policy.

The actor maps the RL token and proprioception to the initial noise of the
base policy's flow; the base policy turns that noise into the action chunk the
robot executes. Transitions therefore carry the noise as ``action`` and the
chunk the base policy produced from it as ``reference_action``.

The actor is a soft actor-critic policy: a diagonal Gaussian in pre-tanh
space, squashed by tanh to ``[-1, 1]`` and scaled by the noise bound, with an
entropy bonus whose temperature follows a target entropy.
"""

from __future__ import annotations

import copy
import json
import math
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

# Schema 1 carried a deterministic actor (mean head only).
STEERING_SCHEMA_VERSION = 2

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
# Pre-tanh standard deviation at initialization: with a zero mean and a bound
# of 2 the squashed samples have roughly unit variance, like the base policy's
# own noise.
LOG_STD_INIT = math.log(0.55)


def squash_gaussian(
    mean: Tensor, log_std: Tensor, epsilon: Tensor
) -> tuple[Tensor, Tensor]:
    """tanh of a reparameterized Gaussian sample and the sample's log density.

    The density is the Gaussian log density of the pre-tanh sample minus the
    tanh Jacobian ``sum(log(1 - tanh(u)^2))``, written as
    ``2 (log 2 - u - softplus(-2u))`` for numerical stability.
    """
    pre_tanh = mean + log_std.exp() * epsilon
    action = torch.tanh(pre_tanh)
    gaussian = -0.5 * epsilon.pow(2) - log_std - 0.5 * math.log(2.0 * math.pi)
    jacobian = 2.0 * (math.log(2.0) - pre_tanh - functional.softplus(-2.0 * pre_tanh))
    return action, (gaussian - jacobian).sum(dim=-1)


class DSRLActor(nn.Module):
    """``(rl_token, proprioception)`` to a squashed Gaussian over noise in
    ``[-bound, bound]^noise_dim``.

    ``forward`` returns the squashed mean normalized to ``[-1, 1]`` like
    ``RLTActor``; ``sample`` draws a reparameterized action with its log
    probability; ``unnormalize_action`` scales by the bound. The reference
    chunk argument is accepted for interface parity and ignored.
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
        self.log_std_head = nn.Linear(previous, self.predicted_action_dim)
        if config.init_final is not None:
            for head in (self.mean_head, self.log_std_head):
                nn.init.uniform_(head.weight, -config.init_final, config.init_final)
            nn.init.uniform_(
                self.mean_head.bias, -config.init_final, config.init_final
            )
        nn.init.constant_(self.log_std_head.bias, LOG_STD_INIT)
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

    def distribution(
        self, rl_token: Tensor, proprioception: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Pre-tanh mean and log standard deviation."""
        state = _norm(
            proprioception,
            self.ns_mean,
            self.ns_std,
            self.ns_q01,
            self.ns_q99,
            self._use_quantiles,
        )
        hidden = self.trunk(torch.cat((rl_token, state), dim=-1))
        log_std = self.log_std_head(hidden).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return self.mean_head(hidden), log_std

    def sample(
        self,
        rl_token: Tensor,
        proprioception: Tensor,
        *,
        std_scale: float = 1.0,
        epsilon: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Reparameterized normalized noise and its log probability.

        ``std_scale`` multiplies the standard deviation; the log probability
        is that of the scaled distribution.
        """
        if not math.isfinite(std_scale) or std_scale <= 0:
            raise ValueError("std_scale must be finite and positive")
        mean, log_std = self.distribution(rl_token, proprioception)
        if std_scale != 1.0:
            log_std = log_std + math.log(std_scale)
        if epsilon is None:
            epsilon = torch.randn_like(mean)
        return squash_gaussian(mean, log_std, epsilon)

    def forward(
        self,
        rl_token: Tensor,
        proprioception: Tensor,
        reference_action: Tensor | None = None,
        *,
        sample: bool = False,
    ) -> tuple[Tensor, Tensor]:
        mean, log_std = self.distribution(rl_token, proprioception)
        squashed_mean = torch.tanh(mean)
        if not sample:
            return squashed_mean, squashed_mean
        action, _ = squash_gaussian(mean, log_std, torch.randn_like(mean))
        return action, squashed_mean


class DSRLPolicy(nn.Module):
    """Soft actor-critic over the base policy's noise; mirrors ``RLTPolicy``'s
    learner API."""

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
        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(config.dsrl_alpha_init), dtype=torch.float32)
        )
        self.target_entropy = float(config.dsrl_target_entropy)
        self.predicted_action_dim = config.network.predicted_action_dim
        self.reference_action_dim = config.network.reference_action_dim
        self.chunk_discount = config.discount**config.actions_to_execute
        self.q_loss_normalization = config.q_loss_normalization
        self.q_loss_alpha = config.q_loss_alpha
        # Statistics of the last actor update, for the learner's logs.
        self.last_actor_stats: dict[str, float] = {}

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    def temperature_parameters(self) -> list[nn.Parameter]:
        """Parameters the actor optimizer trains besides the actor's own."""
        return [self.log_alpha]

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
        """Noise for the base policy, in its own units: the squashed mean, or
        with ``exploration_std > 0`` a sample whose standard deviation is
        scaled by it."""
        if deterministic or exploration_std <= 0:
            _, normalized = self.actor(rl_token, proprioception)
        else:
            normalized, _ = self.actor.sample(
                rl_token, proprioception, std_scale=exploration_std
            )
        bound = float(self.actor.noise_bound)
        return self.actor.unnormalize_action(normalized).clamp(-bound, bound)

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
            next_action, next_log_prob = self.actor.sample(
                next_rl_token, next_proprioception
            )
            target_values = self.critic_target(
                next_rl_token, next_proprioception, next_action
            )
            soft_value = target_values.min(dim=0).values - self.alpha * next_log_prob
            target = reward + (1.0 - done) * self.chunk_discount * soft_value
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
        """Actor loss plus the temperature loss; the two share no parameters."""
        action, log_prob = self.actor.sample(rl_token, proprioception)
        minimum_q = self.critic_ensemble(rl_token, proprioception, action).min(dim=0).values
        if self.q_loss_normalization:
            scale = self.q_loss_alpha / minimum_q.detach().abs().mean().clamp_min(1e-6)
            minimum_q = scale * minimum_q
        actor_loss = (self.alpha.detach() * log_prob - minimum_q).mean()
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        with torch.no_grad():
            _, log_std = self.actor.distribution(rl_token, proprioception)
            self.last_actor_stats = {
                "alpha": float(self.alpha),
                "entropy": float(-log_prob.mean()),
                "log_std": float(log_std.mean()),
                "noise_saturation": float((action.abs() > 0.95).float().mean()),
            }
        return actor_loss + alpha_loss

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
    """One ``(noise_len, action_dim)`` noise: the actor's squashed mean when
    ``exploration_std`` is 0, otherwise a sample from the actor's Gaussian with
    its standard deviation scaled by ``exploration_std``, clipped to the bound."""
    token = torch.as_tensor(rl_token, dtype=torch.float32).reshape(1, -1)
    state = torch.as_tensor(proprioception, dtype=torch.float32).reshape(1, -1)
    with torch.inference_mode():
        if exploration_std > 0:
            mean, log_std = actor.distribution(token, state)
            epsilon = torch.as_tensor(
                rng.standard_normal(tuple(mean.shape)), dtype=torch.float32
            )
            normalized, _ = squash_gaussian(
                mean, log_std + math.log(exploration_std), epsilon
            )
        else:
            _, normalized = actor(token, state)
    noise = actor.unnormalize_action(normalized).numpy().reshape(
        actor.noise_len, actor.per_step_action_dim
    )
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
