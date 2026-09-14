from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor

from armnet_rlt.artifacts import NormStats
from armnet_rlt.config import RLTConfig
from armnet_rlt.networks import RLTActor, RLTCriticEnsemble


class RLTPolicy(nn.Module):
    def __init__(self, config: RLTConfig):
        super().__init__()
        self.config = config
        self.cfg = config
        self.actor = RLTActor(
            config.network, delta_actions=config.policy_uses_delta_actions
        )
        self.critic_ensemble = RLTCriticEnsemble(
            config.network, num_critics=config.num_critics
        )
        self.critic_target = copy.deepcopy(self.critic_ensemble)
        self.critic_target.requires_grad_(False)

        self.predicted_action_len = config.network.predicted_action_len
        self.reference_action_len = config.network.reference_action_len
        self.per_step_action_dim = config.network.action_dim
        self.predicted_action_dim = config.network.predicted_action_dim
        self.reference_action_dim = config.network.reference_action_dim
        self.chunk_discount = config.discount**self.predicted_action_len
        self._chunk_discount = self.chunk_discount
        self.ref_dropout = config.network.ref_action_dropout
        self.bc_beta = config.bc_beta
        self.jerk_lambda = config.jerk_lambda
        self.target_policy_noise = config.target_policy_noise
        self.target_noise_clip = config.target_noise_clip

        self.register_buffer(
            "_jerk_joint_weights",
            torch.tensor(config.jerk_joint_weights, dtype=torch.float32),
        )
        bc_mask = torch.ones(
            self.predicted_action_len, self.per_step_action_dim
        )
        bc_mask[:, -1] = config.bc_gripper_weight
        self.register_buffer("_bc_mask", bc_mask.flatten())

    def set_norm_stats(
        self, norm_stats: dict[str, NormStats], *, use_quantiles: bool = False
    ) -> None:
        self.actor.set_norm_stats(norm_stats, use_quantiles=use_quantiles)
        self.critic_ensemble.set_norm_stats(
            norm_stats, use_quantiles=use_quantiles
        )
        self.critic_target.set_norm_stats(
            norm_stats, use_quantiles=use_quantiles
        )

    @torch.no_grad()
    def select_action(
        self,
        rl_token: Tensor,
        proprioception: Tensor,
        reference_action: Tensor,
        *,
        deterministic: bool = False,
    ) -> Tensor:
        was_training = self.training
        self.eval()
        try:
            _, mean = self.actor(
                rl_token, proprioception, reference_action, sample=False
            )
            action = mean
            if not deterministic:
                action = (
                    mean + torch.randn_like(mean) * self.actor.action_std
                ).clamp(-1.0, 1.0)
            return self.actor.unnormalize_action(action, proprioception)
        finally:
            self.train(was_training)

    def _apply_ref_dropout(self, reference_action: Tensor) -> Tensor:
        if self.ref_dropout <= 0:
            return reference_action
        keep = (
            torch.rand(
                reference_action.shape[0],
                1,
                device=reference_action.device,
            )
            >= self.ref_dropout
        )
        return reference_action * keep

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
        self._validate_predicted(action, "action")
        self._validate_reference(
            next_reference_action, "next_reference_action"
        )
        normalized_action = self.actor.normalize_action(
            action, proprioception
        ).clamp(-1.0, 1.0)
        with torch.no_grad():
            next_reference = self._apply_ref_dropout(next_reference_action)
            _, next_mean = self.actor(
                next_rl_token,
                next_proprioception,
                next_reference,
                sample=False,
            )
            noise = (
                torch.randn_like(next_mean) * self.target_policy_noise
            ).clamp(-self.target_noise_clip, self.target_noise_clip)
            next_action = (next_mean + noise).clamp(-1.0, 1.0)
            target_values = self.critic_target(
                next_rl_token, next_proprioception, next_action
            )
            minimum_target = target_values.min(dim=0).values
            target = (
                reward
                + (1.0 - done) * self.chunk_discount * minimum_target
            )
            if self.config.max_q is not None:
                target = target.clamp(
                    -self.config.max_q, self.config.max_q
                )
        predictions = self.critic_ensemble(
            rl_token, proprioception, normalized_action
        )
        return functional.mse_loss(
            predictions,
            target.unsqueeze(0).expand_as(predictions),
            reduction="none",
        ).mean()

    def jerk_penalty(
        self,
        action: Tensor,
        proprioception: Tensor | None = None,
    ) -> Tensor:
        chunks = action.view(
            action.shape[0],
            self.predicted_action_len,
            self.per_step_action_dim,
        )
        if proprioception is not None:
            # A newly inferred chunk starts from the robot's observed pose, not
            # from an arbitrary point on the chunk. Prepending that stationary
            # target twice extends the existing second-difference loss across
            # the chunk boundary: the first term penalizes an initial command
            # jump and the second penalizes the initial acceleration.
            stationary = proprioception[
                :, : self.per_step_action_dim
            ].repeat(1, self.predicted_action_len)
            normalized_stationary = self.actor.normalize_action(
                stationary, proprioception
            )
            boundary = normalized_stationary.view_as(chunks)[:, :1].clamp(
                -1.0, 1.0
            )
            chunks = torch.cat((boundary, boundary, chunks), dim=1)
        elif self.predicted_action_len < 3:
            return action.new_zeros(action.shape[0])
        second_difference = (
            chunks[:, 2:] - 2 * chunks[:, 1:-1] + chunks[:, :-2]
        )
        return (
            second_difference.square() * self._jerk_joint_weights
        ).sum(dim=(-2, -1))

    def _jerk_penalty(
        self,
        action: Tensor,
        proprioception: Tensor | None = None,
    ) -> Tensor:
        return self.jerk_penalty(action, proprioception)

    def compute_loss_actor(
        self,
        rl_token: Tensor,
        proprioception: Tensor,
        reference_action: Tensor,
    ) -> Tensor:
        self._validate_reference(reference_action, "reference_action")
        actor_input = self._apply_ref_dropout(reference_action)
        action, _ = self.actor(
            rl_token, proprioception, actor_input, sample=True
        )
        q_values = self.critic_ensemble(rl_token, proprioception, action)
        minimum_q = q_values.min(dim=0).values
        reference = reference_action[:, : self.predicted_action_dim]
        normalized_reference = self.actor.normalize_action(
            reference, proprioception
        ).clamp(-1.0, 1.0)
        bc_penalty = (
            (action - normalized_reference).square() * self._bc_mask
        ).sum(dim=-1)
        loss = -minimum_q + self.bc_beta * bc_penalty
        if self.jerk_lambda > 0:
            loss = loss + self.jerk_lambda * self.jerk_penalty(
                action, proprioception
            )
        return loss.mean()

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

    def _validate_predicted(self, value: Tensor, name: str) -> None:
        if value.shape[-1] != self.predicted_action_dim:
            raise ValueError(
                f"{name} has last dimension {value.shape[-1]}, "
                f"expected {self.predicted_action_dim}"
            )

    def _validate_reference(self, value: Tensor, name: str) -> None:
        if value.shape[-1] != self.reference_action_dim:
            raise ValueError(
                f"{name} has last dimension {value.shape[-1]}, "
                f"expected {self.reference_action_dim}"
            )
