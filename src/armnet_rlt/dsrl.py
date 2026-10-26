"""Diffusion steering via RL (DSRL) on top of the frozen Pi0.5 flow policy.

The actor maps the RL token and proprioception to the initial noise of the
base policy's flow; the base policy turns that noise into the action chunk the
robot executes. Transitions carry the full chunk noise as ``action`` and the
chunk the base policy produced from it as ``reference_action``.

The learner is the soft actor-critic of the official pi0 implementation
(nakamotoo/dsrl_pi0, ``jaxrl2/agents/pixel_sac``): a tanh-Gaussian actor whose
``noise_steps`` rows of noise are repeated over the chunk, two critics with
layer normalization, a target without an entropy term, and a temperature
trained toward a target entropy. Rewards are -1 per decision step and 0 at
success; a failed episode is truncated, not terminal.
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
from torch import Tensor

from armnet_rlt.artifacts import NormStats
from armnet_rlt.config import RLTConfig, RLTNetworkConfig
from armnet_rlt.networks import _norm, _register_norm_buffers, _set_state_buffers

# Schema 1 carried a deterministic actor, schema 2 a full-chunk Gaussian actor.
STEERING_SCHEMA_VERSION = 3

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
# Orthogonal gain of the mean and log-std heads (jaxrl2 ``default_init(1e-2)``).
HEAD_INIT_SCALE = 1e-2
# Flax LayerNorm epsilon.
LAYER_NORM_EPS = 1e-6


def _orthogonal_linear(in_features: int, out_features: int, gain: float) -> nn.Linear:
    layer = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


def _mlp(
    input_dim: int, hidden_dims: tuple[int, ...], *, layer_norm: bool
) -> tuple[nn.Sequential, int]:
    """jaxrl2 ``MLP`` with ``activate_final``: Dense, [LayerNorm], ReLU per layer."""
    layers: list[nn.Module] = []
    previous = input_dim
    for hidden in hidden_dims:
        layers.append(_orthogonal_linear(previous, hidden, 1.0))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden, eps=LAYER_NORM_EPS))
        layers.append(nn.ReLU())
        previous = hidden
    return nn.Sequential(*layers), previous


def squash_gaussian(
    mean: Tensor, log_std: Tensor, epsilon: Tensor, *, bound: float = 1.0
) -> tuple[Tensor, Tensor]:
    """tanh of a reparameterized Gaussian sample and the log density of the
    sample scaled to ``[-bound, bound]``.

    The density is the Gaussian log density of the pre-tanh sample minus the
    tanh Jacobian ``sum(log(1 - tanh(u)^2))``, written as
    ``2 (log 2 - u - softplus(-2u))``, minus ``log(bound)`` per dimension for
    the rescaling. The returned action is the tanh value in ``[-1, 1]``.
    """
    pre_tanh = mean + log_std.exp() * epsilon
    action = torch.tanh(pre_tanh)
    gaussian = -0.5 * epsilon.pow(2) - log_std - 0.5 * math.log(2.0 * math.pi)
    jacobian = 2.0 * (
        math.log(2.0) - pre_tanh - nn.functional.softplus(-2.0 * pre_tanh)
    )
    log_prob = (gaussian - jacobian).sum(dim=-1) - mean.shape[-1] * math.log(bound)
    return action, log_prob


class DSRLActor(nn.Module):
    """``(rl_token, proprioception)`` to a tanh-Gaussian over ``noise_steps``
    rows of noise in ``[-bound, bound]``; the rows are repeated to fill the
    base policy's chunk.

    ``forward`` returns the squashed mean normalized to ``[-1, 1]`` like
    ``RLTActor``; ``sample`` draws a reparameterized action with its log
    probability; ``chunk_noise`` repeats the actor's rows over the chunk;
    ``unnormalize_action`` scales by the bound. The reference chunk argument
    is accepted for interface parity and ignored.
    """

    def __init__(
        self,
        config: RLTNetworkConfig,
        *,
        noise_bound: float,
        noise_steps: int = 1,
    ) -> None:
        super().__init__()
        if noise_bound <= 0:
            raise ValueError("noise_bound must be positive")
        if not 1 <= noise_steps <= config.predicted_action_len:
            raise ValueError(
                "noise_steps must be within 1..predicted_action_len "
                f"({noise_steps} vs {config.predicted_action_len})"
            )
        self.noise_len = config.predicted_action_len
        self.noise_steps = int(noise_steps)
        self.per_step_action_dim = config.action_dim
        self.noise_dim = self.noise_steps * config.action_dim
        self.predicted_action_dim = config.predicted_action_dim
        self.reference_action_dim = config.reference_action_dim
        self.proprioception_dim = config.proprioception_dim
        self.register_buffer("noise_bound", torch.tensor(float(noise_bound)))

        input_dim = config.rl_token_dim + config.proprioception_dim
        self.trunk, width = _mlp(input_dim, config.hidden_dims, layer_norm=False)
        self.mean_head = _orthogonal_linear(width, self.noise_dim, HEAD_INIT_SCALE)
        self.log_std_head = _orthogonal_linear(width, self.noise_dim, HEAD_INIT_SCALE)
        self.register_buffer("_use_quantiles", torch.tensor(False))
        _register_norm_buffers(self, "ns", config.proprioception_dim)

    def set_norm_stats(
        self, norm_stats: Mapping[str, NormStats], *, use_quantiles: bool = False
    ) -> None:
        self._use_quantiles.fill_(use_quantiles)
        _set_state_buffers(self, "ns", norm_stats["state"], self.proprioception_dim)

    def normalize_action(
        self, action: Tensor, proprioception: Tensor | None = None
    ) -> Tensor:
        return action / self.noise_bound

    def unnormalize_action(
        self, action: Tensor, proprioception: Tensor | None = None
    ) -> Tensor:
        return action * self.noise_bound

    def chunk_noise(self, noise: Tensor) -> Tensor:
        """``(..., noise_dim)`` to ``(..., predicted_action_dim)``: the actor's
        rows followed by its last row repeated over the rest of the chunk."""
        rows = noise.reshape(*noise.shape[:-1], self.noise_steps, self.per_step_action_dim)
        tail = rows[..., -1:, :].expand(
            *rows.shape[:-2], self.noise_len - self.noise_steps, self.per_step_action_dim
        )
        return torch.cat((rows, tail), dim=-2).reshape(
            *noise.shape[:-1], self.predicted_action_dim
        )

    def actor_noise(self, chunk_noise: Tensor) -> Tensor:
        """The actor's rows of a ``(..., predicted_action_dim)`` chunk noise."""
        if chunk_noise.shape[-1] != self.predicted_action_dim:
            raise ValueError(
                f"action has last dimension {chunk_noise.shape[-1]}, "
                f"expected {self.predicted_action_dim}"
            )
        rows = chunk_noise.reshape(
            *chunk_noise.shape[:-1], self.noise_len, self.per_step_action_dim
        )
        return rows[..., : self.noise_steps, :].reshape(
            *chunk_noise.shape[:-1], self.noise_dim
        )

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
        """Reparameterized normalized noise ``(..., noise_dim)`` and the log
        probability of the noise in its own units.

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
        return squash_gaussian(mean, log_std, epsilon, bound=float(self.noise_bound))

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


class DSRLCritic(nn.Module):
    """``Q(rl_token, proprioception, noise)`` with the noise in its own units;
    jaxrl2 ``StateActionValue`` (LayerNorm MLP)."""

    def __init__(self, config: RLTNetworkConfig, *, noise_dim: int) -> None:
        super().__init__()
        self.noise_dim = int(noise_dim)
        input_dim = config.rl_token_dim + config.proprioception_dim + self.noise_dim
        self.trunk, width = _mlp(input_dim, config.hidden_dims, layer_norm=True)
        self.head = _orthogonal_linear(width, 1, 1.0)
        self.register_buffer("_use_quantiles", torch.tensor(False))
        _register_norm_buffers(self, "ns", config.proprioception_dim)

    def set_norm_stats(
        self, norm_stats: Mapping[str, NormStats], *, use_quantiles: bool = False
    ) -> None:
        self._use_quantiles.fill_(use_quantiles)
        _set_state_buffers(self, "ns", norm_stats["state"], self.ns_mean.shape[0])

    def forward(self, rl_token: Tensor, proprioception: Tensor, noise: Tensor) -> Tensor:
        if noise.shape[-1] != self.noise_dim:
            raise ValueError(
                f"noise has last dimension {noise.shape[-1]}, expected {self.noise_dim}"
            )
        state = _norm(
            proprioception,
            self.ns_mean,
            self.ns_std,
            self.ns_q01,
            self.ns_q99,
            self._use_quantiles,
        )
        hidden = self.trunk(torch.cat((rl_token, state, noise), dim=-1))
        return self.head(hidden).squeeze(-1)


class DSRLCriticEnsemble(nn.Module):
    def __init__(
        self, config: RLTNetworkConfig, *, noise_dim: int, num_critics: int = 2
    ) -> None:
        super().__init__()
        self.critics = nn.ModuleList(
            DSRLCritic(config, noise_dim=noise_dim) for _ in range(num_critics)
        )

    def set_norm_stats(
        self, norm_stats: Mapping[str, NormStats], *, use_quantiles: bool = False
    ) -> None:
        for critic in self.critics:
            critic.set_norm_stats(norm_stats, use_quantiles=use_quantiles)

    def forward(self, rl_token: Tensor, proprioception: Tensor, noise: Tensor) -> Tensor:
        return torch.stack(
            [critic(rl_token, proprioception, noise) for critic in self.critics]
        )


class DSRLPolicy(nn.Module):
    """Soft actor-critic over the base policy's noise; mirrors ``RLTPolicy``'s
    learner API. One gradient step is ``compute_loss_critic``,
    ``update_target_networks``, ``compute_loss_actor``, then
    ``compute_loss_temperature`` on the actor's entropy."""

    def __init__(self, config: RLTConfig) -> None:
        super().__init__()
        if config.algorithm != "dsrl":
            raise ValueError("DSRLPolicy requires config.algorithm == 'dsrl'")
        self.config = config
        self.actor = DSRLActor(
            config.network,
            noise_bound=config.dsrl_noise_bound,
            noise_steps=config.dsrl_noise_steps,
        )
        self.critic_ensemble = DSRLCriticEnsemble(
            config.network,
            noise_dim=self.actor.noise_dim,
            num_critics=config.num_critics,
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
        # Entropy of the last actor update, for the temperature loss.
        self.last_entropy: Tensor = torch.tensor(0.0)
        # Statistics of the last actor update, for the learner's logs.
        self.last_actor_stats: dict[str, float] = {}

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    def temperature_parameters(self) -> list[nn.Parameter]:
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
        """Chunk noise for the base policy, in its own units: the squashed mean
        repeated over the chunk, or with ``exploration_std > 0`` a sample whose
        standard deviation is scaled by it."""
        if deterministic or exploration_std <= 0:
            _, normalized = self.actor(rl_token, proprioception)
        else:
            normalized, _ = self.actor.sample(
                rl_token, proprioception, std_scale=exploration_std
            )
        bound = float(self.actor.noise_bound)
        chunk = self.actor.chunk_noise(normalized)
        return self.actor.unnormalize_action(chunk).clamp(-bound, bound)

    def min_q(self, rl_token: Tensor, proprioception: Tensor, action: Tensor) -> Tensor:
        """Minimum over critics for a ``(..., predicted_action_dim)`` chunk noise."""
        noise = self.actor.actor_noise(action)
        return self.critic_ensemble(rl_token, proprioception, noise).min(dim=0).values

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
        """``(Q - (r + gamma^k (1 - done) min Q_target(s', a')))^2`` with
        ``a'`` sampled from the current actor; no entropy term in the target."""
        noise = self.actor.actor_noise(action)
        with torch.no_grad():
            next_normalized, _ = self.actor.sample(next_rl_token, next_proprioception)
            next_noise = self.actor.unnormalize_action(next_normalized)
            next_q = self.critic_target(
                next_rl_token, next_proprioception, next_noise
            ).min(dim=0).values
            target = reward + self.chunk_discount * (1.0 - done) * next_q
        predictions = self.critic_ensemble(rl_token, proprioception, noise)
        return (predictions - target.unsqueeze(0)).pow(2).mean()

    def compute_loss_actor(
        self,
        rl_token: Tensor,
        proprioception: Tensor,
        reference_action: Tensor | None = None,
    ) -> Tensor:
        """``alpha log pi(a|s) - min Q(s, a)`` with ``a`` reparameterized."""
        normalized, log_prob = self.actor.sample(rl_token, proprioception)
        noise = self.actor.unnormalize_action(normalized)
        minimum_q = self.critic_ensemble(rl_token, proprioception, noise).min(dim=0).values
        actor_loss = (self.alpha.detach() * log_prob - minimum_q).mean()
        self.last_entropy = (-log_prob).mean().detach()
        with torch.no_grad():
            mean, log_std = self.actor.distribution(rl_token, proprioception)
            self.last_actor_stats = {
                "alpha": float(self.alpha),
                "entropy": float(self.last_entropy),
                "log_std": float(log_std.mean()),
                "mean_pi_norm": float(mean.norm(dim=-1).mean()),
                "q_pi": float(minimum_q.mean()),
                "noise_saturation": float((normalized.abs() > 0.95).float().mean()),
            }
        return actor_loss

    def compute_loss_temperature(self, entropy: Tensor | None = None) -> Tensor:
        """``alpha (H - H_target)`` for the entropy of the last actor update."""
        if entropy is None:
            entropy = self.last_entropy
        return self.alpha * (entropy.detach() - self.target_entropy)

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


def relabel_episode(transitions: list[dict[str, Tensor]]) -> None:
    """Rewrite one episode's learner transitions in place to the DSRL reward:
    -1 per decision step, 0 at the success that ends the episode (``done``),
    and -1 with ``done`` 0 at the end of a failed episode, which bootstraps.

    The actor labels only the last transition with the episode's shaped reward
    and marks it ``done``; a success is a positive reward there.
    """
    for fields in transitions:
        success = bool(fields["done"]) and float(fields["reward"]) > 0
        fields["reward"] = torch.tensor(0.0 if success else -1.0, dtype=torch.float32)
        fields["done"] = torch.tensor(1.0 if success else 0.0, dtype=torch.float32)


def steering_payload(
    actor_state: Mapping[str, Any],
    *,
    network: RLTNetworkConfig,
    noise_bound: float,
    learner_step: int | None,
    noise_steps: int = 1,
) -> dict[str, Any]:
    """Flat ``np.savez``-able description of a DSRL actor for a policy server."""
    meta = {
        "schema_version": STEERING_SCHEMA_VERSION,
        "network": asdict(network),
        "noise_bound": float(noise_bound),
        "noise_steps": int(noise_steps),
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
    actor = DSRLActor(
        RLTNetworkConfig(**values),
        noise_bound=float(meta["noise_bound"]),
        noise_steps=int(meta["noise_steps"]),
    )
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
    """One ``(noise_len, action_dim)`` chunk noise: the actor's squashed mean
    when ``exploration_std`` is 0, otherwise a sample from the actor's Gaussian
    with its standard deviation scaled by ``exploration_std``; the actor's rows
    are repeated over the chunk and clipped to the bound."""
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
        chunk = actor.unnormalize_action(actor.chunk_noise(normalized))
    noise = chunk.numpy().reshape(actor.noise_len, actor.per_step_action_dim)
    return np.clip(noise, -noise_bound, noise_bound).astype(np.float32)


def random_noise(
    shape: tuple[int, int],
    *,
    exploration_std: float,
    noise_bound: float,
    rng: np.random.Generator,
    noise_steps: int | None = None,
) -> np.ndarray:
    """Unsteered noise: zero mean plus Gaussian exploration, clipped to the
    bound. With ``noise_steps`` only that many rows are drawn and the last one
    is repeated over the rest of the chunk, like the actor's noise."""
    rows, columns = shape
    noise = np.zeros(shape, dtype=np.float64)
    if exploration_std > 0:
        drawn = rows if noise_steps is None else int(noise_steps)
        if not 1 <= drawn <= rows:
            raise ValueError("noise_steps must be within 1..rows")
        head = rng.normal(0.0, exploration_std, size=(drawn, columns))
        noise = np.concatenate(
            (head, np.repeat(head[-1:], rows - drawn, axis=0)), axis=0
        )
    return np.clip(noise, -noise_bound, noise_bound).astype(np.float32)
