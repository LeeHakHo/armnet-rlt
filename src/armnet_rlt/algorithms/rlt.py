"""RLT: an actor on the RL token that refines π0.5's chunk, trained with TD3
plus a behaviour-cloning term, pretrained on the demonstration cache."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import Tensor

from armnet_rlt.algorithm import Actor, ActorInput, ActorOutput, Algorithm, AlgorithmSpec
from armnet_rlt.networks import RLTActor
from armnet_rlt.policy import RLTPolicy


def _correlated_noise(
    predicted_len: int,
    action_dim: int,
    correlation: float,
    *,
    device: Any,
    dtype: Any,
) -> Tensor:
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
    chunk: Tensor,
    proprioception: Tensor,
    *,
    predicted_len: int,
    action_dim: int,
    exploration_correlation: float,
    exploration_scale: float,
) -> Tensor:
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
    token: Tensor,
    proprioception: Tensor,
    reference: Tensor,
    *,
    predicted_len: int,
    action_dim: int,
    exploration_correlation: float,
    exploration_scale: float = 1.0,
) -> Tensor:
    """The actor's chunk for one observation plus correlated exploration noise."""
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


@torch.no_grad()
def q_stats(policy: Any, batch: dict[str, Any], action: Tensor | None = None) -> dict[str, float]:
    """Mean and std of min-Q over the batch, plus per-reward means when both
    rewarded and unrewarded transitions are present."""
    state = batch["state"]
    action = batch["action"] if action is None else action
    if hasattr(policy, "min_q"):
        q = policy.min_q(state["rl_token"], state["proprioception"], action).reshape(-1)
    else:
        normalized = policy.actor.normalize_action(
            action, state["proprioception"]
        ).clamp(-1.0, 1.0)
        reference_chunk = (
            policy.actor.normalize_action(
                state["reference_action"][:, : policy.predicted_action_dim],
                state["proprioception"],
            ).clamp(-1.0, 1.0)
            if policy.critic_ensemble.action_residual
            else None
        )
        q = policy.critic_ensemble(
            state["rl_token"], state["proprioception"], normalized, reference_chunk
        ).min(dim=0).values.reshape(-1)
    reward = batch["reward"].reshape(-1)
    stats = {"q_mean": float(q.mean()), "q_std": float(q.std())}
    rewarded = reward > 0
    if bool(rewarded.any()) and bool((~rewarded).any()):
        high = float(q[rewarded].mean())
        low = float(q[~rewarded].mean())
        stats["q_reward1_mean"] = high
        stats["q_reward0_mean"] = low
        stats["q_gap"] = high - low
    return stats


class RLTChunkActor(RLTActor, Actor):
    """``RLTActor`` in the policy server: the refined chunk, or π0.5's chunk
    for a base episode, with correlated exploration noise."""

    def noise(self, obs: ActorInput, explore: bool) -> Tensor | None:
        return None

    def act(self, obs: ActorInput, base_chunk: Tensor | None, explore: bool) -> ActorOutput:
        if base_chunk is None or obs.rl_token is None:
            raise ValueError("the rlt actor needs π0.5's chunk and RL token")
        reference = base_chunk[: self.reference_action_len]
        scale = float(obs.exploration_scale) if explore else 0.0
        options = {
            "predicted_len": self.predicted_action_len,
            "action_dim": self.per_step_action_dim,
            "exploration_correlation": float(obs.exploration_correlation),
            "exploration_scale": scale,
        }
        if obs.base:
            chunk = reference[: self.predicted_action_len].detach().clone()
            if scale > 0:
                chunk = _jitter_chunk(self, chunk, obs.proprioception, **options)
        else:
            chunk = _refine_chunk(self, obs.rl_token, obs.proprioception, reference, **options)
        return ActorOutput(chunk=chunk)


class RLTAlgorithm(Algorithm):
    name = "rlt"
    uses_demos = True

    def __init__(self, spec: AlgorithmSpec, config: Any) -> None:
        super().__init__(spec, config)
        self.policy = RLTPolicy(config)
        norm_stats = spec.norm_stats_objects()
        if norm_stats is not None:
            self.policy.set_norm_stats(norm_stats, use_quantiles=config.use_quantile_norm)
        self.policy.to(torch.device(config.device)).train()
        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.policy.critic_ensemble.parameters(), lr=config.critic_lr
        )
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer,
            T_max=max(1, config.max_demo_pretrain_steps),
            eta_min=config.actor_lr_min,
        )
        self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer,
            T_max=max(1, config.max_demo_pretrain_steps),
            eta_min=config.critic_lr_min,
        )
        # Learner steps taken; the actor updates every policy_update_freq of them.
        self.step = 0

    @classmethod
    def build_actor(cls, spec: AlgorithmSpec, config: Any) -> RLTChunkActor:
        return RLTChunkActor(config.network, delta_actions=config.policy_uses_delta_actions)

    def make_actor(self) -> RLTChunkActor:
        actor = self.build_actor(self.spec, self.config)
        actor.load_state_dict(self.actor_state())
        return actor

    def _step(self, batch: dict[str, Any]) -> dict[str, float]:
        """``utd_ratio`` critic updates, then an actor update every
        ``policy_update_freq`` steps."""
        config = self.config
        policy = self.policy
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
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.critic_ensemble.parameters(), config.grad_clip_norm
            )
            self.critic_optimizer.step()
            policy.update_target_networks()
        assert critic_loss is not None
        metrics = {"critic_loss": float(critic_loss.detach())}
        if self.step % config.policy_update_freq == 0:
            actor_loss = policy.compute_loss_actor(
                state["rl_token"], state["proprioception"], state["reference_action"]
            )
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), config.grad_clip_norm)
            self.actor_optimizer.step()
            metrics["actor_loss"] = float(actor_loss.detach())
        self.step += 1
        return metrics

    def _learning_rates(self) -> dict[str, float]:
        return {
            "actor_lr": self.actor_optimizer.param_groups[0]["lr"],
            "critic_lr": self.critic_optimizer.param_groups[0]["lr"],
        }

    def pretrain(self, batch: dict[str, Any]) -> dict[str, float]:
        metrics = self._step(batch)
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        return {**metrics, **self._learning_rates()}

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        return {**self._step(batch), **self._learning_rates()}

    def diagnostics(self, batch: dict[str, Any]) -> dict[str, float]:
        return q_stats(self.policy, batch)

    def actor_state(self) -> dict[str, Tensor]:
        return self.policy.actor_state_bytes()

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "policy_state_dict": self.policy.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_scheduler": self.actor_scheduler.state_dict(),
            "critic_scheduler": self.critic_scheduler.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.policy.load_state_dict(state["policy_state_dict"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.actor_scheduler.load_state_dict(state["actor_scheduler"])
        self.critic_scheduler.load_state_dict(state["critic_scheduler"])
        self.step = int(state.get("step", 0))
