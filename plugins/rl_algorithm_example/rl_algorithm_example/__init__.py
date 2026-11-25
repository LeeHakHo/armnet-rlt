"""Example RL algorithm on top of π0.5: a residual MLP trained with TD3.

The actor sees π0.5's RL token, the joint positions and π0.5's chunk, and adds
a bounded residual to the first ``chunk_len`` rows of that chunk. Twin critics
score the residual; the actor loss penalises its size.

To start your own: copy this package, rename it, change the entry point in
pyproject.toml, replace the networks and ``update``.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from armnet_rlt.algorithm import Actor, ActorInput, ActorOutput, Algorithm, AlgorithmSpec

# Options read from config.algorithm_options (--algorithm-option KEY=VALUE).
DEFAULTS = {
    "hidden": 256,
    # Largest residual per joint, in state standard deviations.
    "residual_scale": 0.1,
    "lr": 3e-4,
    "tau": 0.005,
    # Actor update every this many critic updates.
    "policy_delay": 2,
    # TD3 target smoothing noise and its clip, in residual units ([-1, 1]).
    "target_noise": 0.2,
    "target_noise_clip": 0.5,
    # Weight of the squared residual in the actor loss.
    "residual_penalty": 0.1,
    # Exploration noise std in residual units at --exploration-scale 1.
    "explore_std": 0.3,
}


def _mlp(inputs: int, hidden: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inputs, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, outputs),
    )


def _state_stats(spec: AlgorithmSpec) -> tuple[Tensor, Tensor]:
    """Mean and std of the joint positions, from the base checkpoint's stats."""
    stats = (spec.norm_stats or {}).get("state")
    if stats is None:
        return torch.zeros(spec.proprioception_dim), torch.ones(spec.proprioception_dim)
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32).reshape(-1)[: spec.proprioception_dim]
    std = torch.as_tensor(stats["std"], dtype=torch.float32).reshape(-1)[: spec.proprioception_dim]
    return mean, std.clamp_min(1e-3)


class ResidualActor(Actor):
    """Runs in the policy server; ``forward`` gives a residual in [-1, 1]."""

    def __init__(self, spec: AlgorithmSpec, options: Mapping[str, Any]) -> None:
        super().__init__()
        self.chunk_len = spec.chunk_len
        self.action_dim = spec.action_dim
        inputs = spec.rl_token_dim + spec.proprioception_dim + spec.chunk_len * spec.action_dim
        self.net = _mlp(inputs, int(options["hidden"]), spec.chunk_len * spec.action_dim)
        self.explore_std = float(options["explore_std"])
        mean, std = _state_stats(spec)
        # Buffers are part of the state dict sent to the policy server.
        self.register_buffer("state_mean", mean)
        self.register_buffer("state_std", std)
        self.register_buffer("residual_scale", std[: spec.action_dim] * float(options["residual_scale"]))

    def features(self, rl_token: Tensor, proprioception: Tensor, base_chunk: Tensor) -> Tensor:
        """``[B, 2048]``, ``[B, 6]``, ``[B, chunk_len * 6]`` (absolute targets)."""
        state = (proprioception - self.state_mean) / self.state_std
        base = (base_chunk - self.state_mean.repeat(self.chunk_len)) / self.state_std.repeat(self.chunk_len)
        return torch.cat((rl_token, state, base), dim=-1)

    def forward(self, rl_token: Tensor, proprioception: Tensor, base_chunk: Tensor) -> Tensor:
        return torch.tanh(self.net(self.features(rl_token, proprioception, base_chunk)))

    def to_targets(self, residual: Tensor, base_chunk: Tensor) -> Tensor:
        """Residual in [-1, 1] to absolute joint targets."""
        return base_chunk + residual * self.residual_scale.repeat(self.chunk_len)

    def act(self, obs: ActorInput, base_chunk: Tensor | None, explore: bool) -> ActorOutput:
        # base_chunk is π0.5's chunk [base_chunk_len, 6]; the cell executes chunk_len rows.
        base = base_chunk[: self.chunk_len].reshape(1, -1)
        residual = self(obs.rl_token[None], obs.proprioception[None], base)
        if explore:
            noise = torch.randn_like(residual) * self.explore_std * obs.exploration_scale
            residual = (residual + noise).clamp(-1.0, 1.0)
        chunk = self.to_targets(residual, base).reshape(self.chunk_len, self.action_dim)
        # Extras come back in the learner's batches.
        return ActorOutput(chunk=chunk, extras={"residual": residual[0]})


class Critic(nn.Module):
    """Q(features, residual)."""

    def __init__(self, spec: AlgorithmSpec, hidden: int) -> None:
        super().__init__()
        chunk = spec.chunk_len * spec.action_dim
        self.net = _mlp(spec.rl_token_dim + spec.proprioception_dim + 2 * chunk, hidden, 1)

    def forward(self, features: Tensor, residual: Tensor) -> Tensor:
        return self.net(torch.cat((features, residual), dim=-1)).squeeze(-1)


class ResidualTD3(Algorithm):
    name = "residual_td3"
    uses_base = True
    needs_images = False

    def __init__(self, spec: AlgorithmSpec, config: Any) -> None:
        super().__init__(spec, config)
        self.options = {**DEFAULTS, **dict(getattr(config, "algorithm_options", {}) or {})}
        self.device = torch.device(config.device)
        hidden = int(self.options["hidden"])
        self.actor = ResidualActor(spec, self.options).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(False)
        self.critics = nn.ModuleList(Critic(spec, hidden) for _ in range(2)).to(self.device)
        self.critics_target = copy.deepcopy(self.critics).requires_grad_(False)
        lr = float(self.options["lr"])
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=lr)
        # One transition is one chunk, so the discount covers chunk_len steps.
        self.chunk_discount = config.discount ** spec.chunk_len
        self.steps = 0

    @classmethod
    def build_actor(cls, spec: AlgorithmSpec, config: Any) -> ResidualActor:
        # Actor only, for the policy server.
        options = {**DEFAULTS, **dict(getattr(config, "algorithm_options", {}) or {})}
        return ResidualActor(spec, options)

    def make_actor(self) -> ResidualActor:
        actor = self.build_actor(self.spec, self.config)
        actor.load_state_dict(self.actor_state())
        return actor

    def _residual(self, action: Tensor, base: Tensor) -> Tensor:
        """The executed chunk as a residual on π0.5's chunk, in [-1, 1]."""
        scale = self.actor.residual_scale.repeat(self.spec.chunk_len)
        return ((action - base) / scale).clamp(-1.0, 1.0)

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        o = self.options
        chunk = self.spec.chunk_len * self.spec.action_dim
        state, next_state = batch["state"], batch["next_state"]
        # reference_action is π0.5's whole chunk; the executed part is its first rows.
        base = state["reference_action"][:, :chunk]
        next_base = next_state["reference_action"][:, :chunk]
        features = self.actor.features(state["rl_token"], state["proprioception"], base)
        next_features = self.actor.features(
            next_state["rl_token"], next_state["proprioception"], next_base
        )
        residual = self._residual(batch["action"], base)

        with torch.no_grad():
            next_residual = self.actor_target(
                next_state["rl_token"], next_state["proprioception"], next_base
            )
            noise = (torch.randn_like(next_residual) * o["target_noise"]).clamp(
                -o["target_noise_clip"], o["target_noise_clip"]
            )
            next_residual = (next_residual + noise).clamp(-1.0, 1.0)
            next_q = torch.min(
                *(critic(next_features, next_residual) for critic in self.critics_target)
            )
            target = batch["reward"] + self.chunk_discount * (1.0 - batch["done"]) * next_q
        critic_loss = sum(F.mse_loss(critic(features, residual), target) for critic in self.critics)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        metrics = {"critic_loss": float(critic_loss.detach()), "q_mean": float(target.mean())}
        self.steps += 1
        if self.steps % int(o["policy_delay"]) == 0:
            new_residual = self.actor(state["rl_token"], state["proprioception"], base)
            q = self.critics[0](features, new_residual)
            actor_loss = -q.mean() + o["residual_penalty"] * new_residual.square().mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            metrics["actor_loss"] = float(actor_loss.detach())
            with torch.no_grad():
                for target_net, net in ((self.actor_target, self.actor), (self.critics_target, self.critics)):
                    for target_parameter, parameter in zip(target_net.parameters(), net.parameters()):
                        target_parameter.lerp_(parameter, float(o["tau"]))
        return metrics

    def actor_state(self) -> dict[str, Tensor]:
        return {name: value.detach().cpu() for name, value in self.actor.state_dict().items()}

    def state_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critics": self.critics.state_dict(),
            "critics_target": self.critics_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.steps = int(state["steps"])
        self.actor.load_state_dict(state["actor"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.critics.load_state_dict(state["critics"])
        self.critics_target.load_state_dict(state["critics_target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
