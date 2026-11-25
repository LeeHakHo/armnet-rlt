"""Example RL algorithm without π0.5: a CNN on the three cameras and SAC.

``uses_base = False``: the policy server skips π0.5; the actor gets no RL token
and no base chunk. ``needs_images = True``: the actor and the batches get the
top, front and wrist images (``images``, ``next_images``).

The actor outputs bounded offsets from the current joint positions; two critics
with their own encoder score them; the entropy temperature is learned.

To start your own: copy this package, rename it, change the entry point in
pyproject.toml, replace the networks and ``update``.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from armnet_rlt.algorithm import Actor, ActorInput, ActorOutput, Algorithm, AlgorithmSpec

# Options read from config.algorithm_options (--algorithm-option KEY=VALUE).
DEFAULTS = {
    "hidden": 256,
    "features": 64,
    # Largest offset of a target from the current position, in state stds.
    "action_scale": 0.3,
    "lr": 3e-4,
    "tau": 0.01,
    "alpha_init": 0.1,
}
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class Encoder(nn.Module):
    """A small CNN applied to each view; the view features are concatenated."""

    def __init__(self, views: int, features: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, features), nn.LayerNorm(features),
        )
        self.out_dim = views * features

    def forward(self, images: dict[str, Tensor], views: tuple[str, ...]) -> Tensor:
        # uint8 [B, H, W, 3] to float [B, 3, H, W] in [0, 1].
        encoded = [
            self.conv(images[view].permute(0, 3, 1, 2).float() / 255.0) for view in views
        ]
        return torch.cat(encoded, dim=-1)


def _state_stats(spec: AlgorithmSpec) -> tuple[Tensor, Tensor]:
    stats = (spec.norm_stats or {}).get("state")
    if stats is None:
        return torch.zeros(spec.proprioception_dim), torch.ones(spec.proprioception_dim)
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32).reshape(-1)[: spec.proprioception_dim]
    std = torch.as_tensor(stats["std"], dtype=torch.float32).reshape(-1)[: spec.proprioception_dim]
    return mean, std.clamp_min(1e-3)


class PixelActor(Actor):
    """Tanh-Gaussian over the chunk's offsets; runs in the policy server."""

    def __init__(self, spec: AlgorithmSpec, options: Mapping[str, Any]) -> None:
        super().__init__()
        self.views = tuple(spec.image_views)
        self.chunk_len = spec.chunk_len
        self.action_dim = spec.action_dim
        self.encoder = Encoder(len(self.views), int(options["features"]))
        hidden = int(options["hidden"])
        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.out_dim + spec.proprioception_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, spec.chunk_len * spec.action_dim)
        self.log_std = nn.Linear(hidden, spec.chunk_len * spec.action_dim)
        mean, std = _state_stats(spec)
        self.register_buffer("state_mean", mean)
        self.register_buffer("state_std", std)
        self.register_buffer("action_scale", std[: spec.action_dim] * float(options["action_scale"]))

    def normalized_state(self, proprioception: Tensor) -> Tensor:
        return (proprioception - self.state_mean) / self.state_std

    def forward(
        self, images: dict[str, Tensor], proprioception: Tensor, *, sample: bool
    ) -> tuple[Tensor, Tensor]:
        """Offsets in [-1, 1] ``[B, chunk_len * 6]`` and their log probability."""
        hidden = self.trunk(
            torch.cat((self.encoder(images, self.views), self.normalized_state(proprioception)), -1)
        )
        mean = self.mean(hidden)
        log_std = self.log_std(hidden).clamp(LOG_STD_MIN, LOG_STD_MAX)
        if not sample:
            return torch.tanh(mean), torch.zeros(mean.shape[0], device=mean.device)
        noise = torch.randn_like(mean)
        pre_tanh = mean + log_std.exp() * noise
        offset = torch.tanh(pre_tanh)
        log_prob = (
            -0.5 * noise.square() - log_std - 0.5 * math.log(2 * math.pi)
            - torch.log1p(-offset.square() + 1e-6)
        ).sum(-1)
        return offset, log_prob

    def to_targets(self, offset: Tensor, proprioception: Tensor) -> Tensor:
        """Offsets to absolute joint targets ``[B, chunk_len * 6]``."""
        current = proprioception[:, : self.action_dim].repeat(1, self.chunk_len)
        return current + offset * self.action_scale.repeat(self.chunk_len)

    def to_offsets(self, targets: Tensor, proprioception: Tensor) -> Tensor:
        current = proprioception[:, : self.action_dim].repeat(1, self.chunk_len)
        return ((targets - current) / self.action_scale.repeat(self.chunk_len)).clamp(-1.0, 1.0)

    def act(self, obs: ActorInput, base_chunk: Tensor | None, explore: bool) -> ActorOutput:
        # base_chunk and obs.rl_token are None: this algorithm does not use π0.5.
        images = {view: image[None] for view, image in obs.images.items()}
        state = obs.proprioception[None]
        offset, _ = self(images, state, sample=explore)
        chunk = self.to_targets(offset, state).reshape(self.chunk_len, self.action_dim)
        return ActorOutput(chunk=chunk)


class Critic(nn.Module):
    def __init__(self, spec: AlgorithmSpec, options: Mapping[str, Any]) -> None:
        super().__init__()
        self.views = tuple(spec.image_views)
        self.encoder = Encoder(len(self.views), int(options["features"]))
        hidden = int(options["hidden"])
        inputs = self.encoder.out_dim + spec.proprioception_dim + spec.chunk_len * spec.action_dim
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(inputs, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )
            for _ in range(2)
        )

    def forward(self, images: dict[str, Tensor], state: Tensor, offset: Tensor) -> Tensor:
        """Q of both heads, ``[2, B]``; ``state`` is normalized."""
        inputs = torch.cat((self.encoder(images, self.views), state, offset), dim=-1)
        return torch.stack([head(inputs).squeeze(-1) for head in self.heads])


class PixelSAC(Algorithm):
    name = "pixel_sac"
    uses_base = False
    needs_images = True
    # (height, width) the server resizes the cameras to.
    image_size = (96, 96)

    def __init__(self, spec: AlgorithmSpec, config: Any) -> None:
        super().__init__(spec, config)
        self.options = {**DEFAULTS, **dict(getattr(config, "algorithm_options", {}) or {})}
        self.device = torch.device(config.device)
        lr = float(self.options["lr"])
        self.actor = PixelActor(spec, self.options).to(self.device)
        self.critic = Critic(spec, self.options).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_alpha = torch.tensor(
            math.log(float(self.options["alpha_init"])), device=self.device, requires_grad=True
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -float(spec.chunk_len * spec.action_dim)
        self.chunk_discount = config.discount ** spec.chunk_len

    @classmethod
    def build_actor(cls, spec: AlgorithmSpec, config: Any) -> PixelActor:
        options = {**DEFAULTS, **dict(getattr(config, "algorithm_options", {}) or {})}
        return PixelActor(spec, options)

    def make_actor(self) -> PixelActor:
        actor = self.build_actor(self.spec, self.config)
        actor.load_state_dict(self.actor_state())
        return actor

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        state = batch["state"]["proprioception"]
        next_state = batch["next_state"]["proprioception"]
        images, next_images = batch["images"], batch["next_images"]
        # batch["action"] is the chunk as executed: absolute targets, flattened.
        offset = self.actor.to_offsets(batch["action"], state)
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            next_offset, next_log_prob = self.actor(next_images, next_state, sample=True)
            next_q = self.critic_target(
                next_images, self.actor.normalized_state(next_state), next_offset
            ).min(dim=0).values
            target = batch["reward"] + self.chunk_discount * (1.0 - batch["done"]) * (
                next_q - alpha * next_log_prob
            )
        q = self.critic(images, self.actor.normalized_state(state), offset)
        critic_loss = (q - target).square().mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        new_offset, log_prob = self.actor(images, state, sample=True)
        new_q = self.critic(images, self.actor.normalized_state(state), new_offset).min(dim=0).values
        actor_loss = (alpha * log_prob - new_q).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.critic_target.parameters(), self.critic.parameters()
            ):
                target_parameter.lerp_(parameter, float(self.options["tau"]))
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha": float(alpha),
            "entropy": float(-log_prob.detach().mean()),
        }

    def actor_state(self) -> dict[str, Tensor]:
        return {name: value.detach().cpu() for name, value in self.actor.state_dict().items()}

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(torch.as_tensor(state["log_alpha"]))
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
