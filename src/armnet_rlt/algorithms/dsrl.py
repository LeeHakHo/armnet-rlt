"""DSRL: a SAC actor on the RL token that picks π0.5's initial flow noise;
the cell executes the chunk π0.5 makes from it."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from armnet_rlt.algorithm import Actor, ActorInput, ActorOutput, Algorithm, AlgorithmSpec
from armnet_rlt.algorithms.rlt import q_stats
from armnet_rlt.dsrl import DSRLActor, DSRLPolicy, choose_noise, random_noise, relabel_episode


class DSRLNoiseActor(DSRLActor, Actor):
    """``DSRLActor`` in the policy server. ``noise`` is the actor's chunk
    noise (unit Gaussian noise for a base episode); ``act`` returns the first
    ``execute_len`` rows of π0.5's chunk and the noise as the ``noise`` extra."""

    def __init__(self, *args: Any, execute_len: int, seed: int | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.execute_len = int(execute_len)
        self.rng = np.random.default_rng(seed)
        self._last_noise: np.ndarray | None = None

    def noise(self, obs: ActorInput, explore: bool) -> Tensor:
        bound = float(self.noise_bound)
        if obs.base:
            noise = random_noise(
                (self.noise_len, self.per_step_action_dim),
                exploration_std=1.0,
                noise_bound=bound,
                rng=self.rng,
            )
        else:
            if obs.rl_token is None:
                raise ValueError("the dsrl actor needs the RL token")
            noise = choose_noise(
                self,
                obs.rl_token.detach().cpu().numpy(),
                obs.proprioception.detach().cpu().numpy(),
                exploration_std=float(obs.exploration_scale) if explore else 0.0,
                noise_bound=bound,
                rng=self.rng,
            )
        self._last_noise = noise
        return torch.from_numpy(noise)

    def act(self, obs: ActorInput, base_chunk: Tensor | None, explore: bool) -> ActorOutput:
        if base_chunk is None or self._last_noise is None:
            raise ValueError("the dsrl actor acts on π0.5's chunk for its noise")
        noise, self._last_noise = self._last_noise, None
        return ActorOutput(
            chunk=base_chunk[: self.execute_len].detach().clone(),
            extras={"noise": torch.from_numpy(noise)},
        )


class DSRLAlgorithm(Algorithm):
    name = "dsrl"
    balance_curriculum = False
    keeps_partial_chunks = True

    def __init__(self, spec: AlgorithmSpec, config: Any) -> None:
        super().__init__(spec, config)
        self.policy = DSRLPolicy(config)
        norm_stats = spec.norm_stats_objects()
        if norm_stats is not None:
            self.policy.set_norm_stats(norm_stats, use_quantiles=config.use_quantile_norm)
        self.policy.to(torch.device(config.device)).train()
        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.policy.critic_ensemble.parameters(), lr=config.critic_lr
        )
        self.temperature_optimizer = torch.optim.Adam(
            self.policy.temperature_parameters(), lr=config.dsrl_temperature_lr
        )
        # Kept for the checkpoint layout; DSRL never steps them.
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
        # utd_ratio gradient steps per transition; dsrl_initial_steps after the
        # first episode.
        self.updates_per_transition = int(config.utd_ratio)
        self.initial_updates = int(config.dsrl_initial_steps)

    @classmethod
    def build_actor(cls, spec: AlgorithmSpec, config: Any) -> DSRLNoiseActor:
        return DSRLNoiseActor(
            config.network,
            noise_bound=config.dsrl_noise_bound,
            noise_steps=config.dsrl_noise_steps,
            execute_len=config.actions_to_execute,
        )

    def make_actor(self) -> DSRLNoiseActor:
        actor = self.build_actor(self.spec, self.config)
        actor.load_state_dict(self.actor_state())
        return actor

    @staticmethod
    def _noise(batch: dict[str, Any]) -> Tensor:
        """The chunk noise of each transition, ``[B, horizon * action_dim]``."""
        return batch["extras"]["noise"].flatten(1)

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        """One SAC gradient step: critics, target critics, actor, temperature."""
        policy = self.policy
        state = batch["state"]
        next_state = batch["next_state"]
        critic_loss = policy.compute_loss_critic(
            state["rl_token"],
            state["proprioception"],
            self._noise(batch),
            batch["reward"],
            next_state["rl_token"],
            next_state["proprioception"],
            next_state["reference_action"],
            batch["done"],
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        policy.update_target_networks()

        actor_loss = policy.compute_loss_actor(
            state["rl_token"], state["proprioception"], state["reference_action"]
        )
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        temperature_loss = policy.compute_loss_temperature()
        self.temperature_optimizer.zero_grad(set_to_none=True)
        temperature_loss.backward()
        self.temperature_optimizer.step()
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "actor_lr": self.actor_optimizer.param_groups[0]["lr"],
            "critic_lr": self.critic_optimizer.param_groups[0]["lr"],
            **policy.last_actor_stats,
        }

    def relabel(self, episode: list[dict[str, Tensor]]) -> None:
        relabel_episode(
            episode,
            chunk_discount=self.policy.chunk_discount,
            safety_trip_penalty=self.config.safety_trip_penalty,
        )

    def diagnostics(self, batch: dict[str, Any]) -> dict[str, float]:
        return q_stats(self.policy, batch, action=self._noise(batch))

    def actor_state(self) -> dict[str, Tensor]:
        return self.policy.actor_state_bytes()

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy_state_dict": self.policy.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_scheduler": self.actor_scheduler.state_dict(),
            "critic_scheduler": self.critic_scheduler.state_dict(),
            "temperature_optimizer": self.temperature_optimizer.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.policy.load_state_dict(state["policy_state_dict"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.actor_scheduler.load_state_dict(state["actor_scheduler"])
        self.critic_scheduler.load_state_dict(state["critic_scheduler"])
        if "temperature_optimizer" in state:
            self.temperature_optimizer.load_state_dict(state["temperature_optimizer"])
