"""Copies of the rlt and dsrl code paths before the algorithm interface, kept
as the reference the ported algorithms are compared against."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor

from armnet_rlt.dsrl import squash_gaussian


def _train_step(
    policy: Any,
    batch: dict[str, Any],
    *,
    step: int,
    config: Any,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
) -> tuple[float, float | None]:
    state = batch["state"]
    next_state = batch["next_state"]
    critic_loss: Tensor | None = None
    for _ in range(config.utd_ratio):
        critic_loss = policy.compute_loss_critic(
            state["rl_token"],
            state["proprioception"],
            batch["action"],
            batch["reward"],
            next_state["rl_token"],
            next_state["proprioception"],
            next_state["reference_action"],
            batch["done"],
            reference_action=state["reference_action"],
        )
        critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            policy.critic_ensemble.parameters(), config.grad_clip_norm
        )
        critic_optimizer.step()
        policy.update_target_networks()

    actor_value: float | None = None
    if step % config.policy_update_freq == 0:
        actor_loss = policy.compute_loss_actor(
            state["rl_token"],
            state["proprioception"],
            state["reference_action"],
        )
        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            policy.actor.parameters(), config.grad_clip_norm
        )
        actor_optimizer.step()
        actor_value = float(actor_loss.detach())
    assert critic_loss is not None
    return float(critic_loss.detach()), actor_value


def _train_step_dsrl(
    policy: Any,
    batch: dict[str, Any],
    *,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    temperature_optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    """One SAC gradient step on one batch: critics, target critics, actor,
    temperature."""
    state = batch["state"]
    next_state = batch["next_state"]
    critic_loss = policy.compute_loss_critic(
        state["rl_token"],
        state["proprioception"],
        batch["action"],
        batch["reward"],
        next_state["rl_token"],
        next_state["proprioception"],
        next_state["reference_action"],
        batch["done"],
    )
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_optimizer.step()
    policy.update_target_networks()

    actor_loss = policy.compute_loss_actor(
        state["rl_token"], state["proprioception"], state["reference_action"]
    )
    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    actor_optimizer.step()

    temperature_loss = policy.compute_loss_temperature()
    temperature_optimizer.zero_grad(set_to_none=True)
    temperature_loss.backward()
    temperature_optimizer.step()
    return float(critic_loss.detach()), float(actor_loss.detach())


def relabel_rlt_safety_trip(
    transitions: list[dict[str, Tensor]], penalty: float
) -> None:
    """Set the terminal reward of a safety-tripped RLT episode to ``-penalty``
    (success 1, failure 0)."""
    if transitions and bool(float(transitions[-1].get("safety_trip", 0.0))):
        transitions[-1]["reward"] = torch.tensor(-float(penalty), dtype=torch.float32)


def _correlated_noise(
    predicted_len: int,
    action_dim: int,
    correlation: float,
    *,
    device: Any,
    dtype: Any,
) -> torch.Tensor:
    """``(1, predicted_len, action_dim)`` AR(1) Gaussian noise with unit
    marginal std and lag-one ``correlation``."""
    innovations = torch.randn((1, predicted_len, action_dim), device=device, dtype=dtype)
    noise = torch.empty_like(innovations)
    noise[:, 0] = innovations[:, 0]
    innovation_scale = math.sqrt(1.0 - correlation**2)
    for index in range(1, predicted_len):
        noise[:, index] = (
            correlation * noise[:, index - 1] + innovation_scale * innovations[:, index]
        )
    return noise


def _jitter_chunk(
    actor: Any,
    chunk: torch.Tensor,
    proprioception: torch.Tensor,
    *,
    predicted_len: int,
    action_dim: int,
    exploration_correlation: float,
    exploration_scale: float,
) -> torch.Tensor:
    """``chunk`` plus correlated exploration noise added in normalized action space."""
    with torch.inference_mode():
        normalized = actor.normalize_action(
            chunk.flatten().unsqueeze(0), proprioception.unsqueeze(0)
        )
        noise = _correlated_noise(
            predicted_len,
            action_dim,
            exploration_correlation,
            device=normalized.device,
            dtype=normalized.dtype,
        )
        jittered = (
            normalized
            + noise.flatten(1) * actor.action_std.reshape(1, -1) * exploration_scale
        ).clamp(-1.0, 1.0)
        action = actor.unnormalize_action(jittered, proprioception.unsqueeze(0))
    return action.reshape(predicted_len, action_dim).detach()


def _refine_chunk(
    actor: Any,
    token: torch.Tensor,
    proprioception: torch.Tensor,
    reference: torch.Tensor,
    *,
    predicted_len: int,
    action_dim: int,
    exploration_correlation: float,
    exploration_scale: float = 1.0,
) -> torch.Tensor:
    with torch.inference_mode():
        _, mean = actor(
            token.unsqueeze(0),
            proprioception.unsqueeze(0),
            reference.flatten().unsqueeze(0),
            sample=False,
        )
        std = actor.action_std.reshape(1, -1)
        noise = _correlated_noise(
            predicted_len,
            action_dim,
            exploration_correlation,
            device=mean.device,
            dtype=mean.dtype,
        )
        normalized = (
            mean + noise.flatten(1) * std * exploration_scale
        ).clamp(-1.0, 1.0)
        action = actor.unnormalize_action(
            normalized, proprioception.unsqueeze(0)
        )
    return action.reshape(predicted_len, action_dim).detach()


def choose_noise(
    actor: Any,
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
