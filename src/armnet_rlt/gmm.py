"""A mixture-of-Gaussians head over action chunks.

Behaviour cloning with a squared or absolute error regresses to the mean of
the demonstrated actions. Where the demonstrations hold several ways of
acting (press now, or line up first), that mean is a motion nobody
demonstrated. The head here keeps ``num_modes`` chunk trajectories, each a
diagonal Gaussian per step, and one weight per trajectory for the whole
chunk, so a mode is a complete chunk and the policy executes the likeliest
one instead of the average.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class GMMHead(nn.Module):
    """``(..., dim_model)`` features to a flat ``(..., num_modes * (2 * action_dim + 1))``
    tensor holding each mode's mean, log std and logit for every step."""

    def __init__(
        self,
        dim_model: int,
        action_dim: int,
        num_modes: int,
        *,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
    ) -> None:
        super().__init__()
        if num_modes < 1 or action_dim < 1:
            raise ValueError("num_modes and action_dim must be positive")
        if min_log_std >= max_log_std:
            raise ValueError("min_log_std must be below max_log_std")
        self.action_dim = int(action_dim)
        self.num_modes = int(num_modes)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.linear = nn.Linear(dim_model, self.num_modes * (2 * self.action_dim + 1))

    @property
    def output_dim(self) -> int:
        return self.num_modes * (2 * self.action_dim + 1)

    def forward(self, features: Tensor) -> Tensor:
        return self.linear(features)

    def unpack(self, raw: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Split a head output into means ``(..., K, A)``, log stds ``(..., K, A)``
        clamped to the configured range, and per-step logits ``(..., K)``."""
        k, a = self.num_modes, self.action_dim
        if raw.shape[-1] != self.output_dim:
            raise ValueError(f"expected {self.output_dim} head outputs, got {raw.shape[-1]}")
        means = raw[..., : k * a].reshape(*raw.shape[:-1], k, a)
        log_std = raw[..., k * a : 2 * k * a].reshape(*raw.shape[:-1], k, a)
        log_std = log_std.clamp(self.min_log_std, self.max_log_std)
        logits = raw[..., 2 * k * a :]
        return means, log_std, logits

    @torch.no_grad()
    def init_from_linear(self, linear: nn.Linear, *, init_std: float, jitter: float = 0.01) -> None:
        """Start every mode at a trained single-output head, each nudged by a
        different small perturbation so the modes can separate, with the
        given standard deviation and equal weights."""
        if linear.out_features != self.action_dim or linear.in_features != self.linear.in_features:
            raise ValueError("the linear head's shape does not match this mixture head")
        k, a = self.num_modes, self.action_dim
        weight = self.linear.weight
        bias = self.linear.bias
        weight.zero_()
        bias.zero_()
        for mode in range(k):
            rows = slice(mode * a, (mode + 1) * a)
            weight[rows] = linear.weight + jitter * torch.randn_like(linear.weight)
            bias[rows] = linear.bias + jitter * torch.randn_like(linear.bias)
        bias[k * a : 2 * k * a] = math.log(init_std)


def chunk_log_prob(
    means: Tensor,
    log_std: Tensor,
    logits: Tensor,
    actions: Tensor,
    valid: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Log-likelihood of each action chunk under the mixture.

    ``means``/``log_std`` are ``(B, T, K, A)``, ``logits`` ``(B, T, K)``, ``actions``
    ``(B, T, A)`` and ``valid`` ``(B, T)`` marks the steps that count (padding
    past the episode end does not). One weight per chunk comes from the mean
    of the per-step logits. Returns the mixture log-likelihood ``(B,)`` and the
    per-mode log-likelihoods ``(B, K)``, both summed over valid steps and
    action dimensions.
    """
    if valid is None:
        valid = torch.ones(actions.shape[:2], dtype=torch.bool, device=actions.device)
    diff = (actions.unsqueeze(2) - means) / log_std.exp()
    step_ll = -0.5 * diff.square() - log_std - 0.5 * math.log(2 * math.pi)
    step_ll = step_ll.sum(dim=-1) * valid.unsqueeze(-1).to(step_ll.dtype)
    mode_ll = step_ll.sum(dim=1)
    log_weights = torch.log_softmax(logits.mean(dim=1), dim=-1)
    return torch.logsumexp(log_weights + mode_ll, dim=-1), mode_ll


def chunk_weights(logits: Tensor) -> Tensor:
    """Mixture weights ``(B, K)`` from per-step logits ``(B, T, K)``."""
    return torch.softmax(logits.mean(dim=1), dim=-1)


def pick_chunk(
    means: Tensor,
    log_std: Tensor,
    logits: Tensor,
    *,
    sample_mode: bool = False,
    sample_noise: bool = False,
) -> tuple[Tensor, Tensor]:
    """One chunk ``(B, T, A)`` per batch element: the likeliest mode's means, or
    a mode drawn by its weight and, with ``sample_noise``, Gaussian noise added.
    Also returns the chosen mode indices ``(B,)``."""
    weights = chunk_weights(logits)
    if sample_mode:
        chosen = torch.multinomial(weights, 1).squeeze(-1)
    else:
        chosen = weights.argmax(dim=-1)
    index = chosen.view(-1, 1, 1, 1).expand(-1, means.shape[1], 1, means.shape[-1])
    chunk = means.gather(2, index).squeeze(2)
    if sample_noise:
        std = log_std.gather(2, index).squeeze(2).exp()
        chunk = chunk + std * torch.randn_like(chunk)
    return chunk, chosen
