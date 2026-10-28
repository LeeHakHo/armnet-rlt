from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor

from armnet_rlt.artifacts import NormStats
from armnet_rlt.config import RLTConfig
from armnet_rlt.gmm import chunk_log_prob
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
        self.q_loss_normalization = config.q_loss_normalization
        self.q_loss_alpha = config.q_loss_alpha
        self.jerk_lambda = config.jerk_lambda
        self.bc_deviation_max = config.bc_deviation_max
        self.bc_deviation_weight = config.bc_deviation_weight
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

    def compute_loss_bc(
        self,
        rl_token: Tensor,
        proprioception: Tensor,
        reference_action: Tensor,
        action: Tensor,
    ) -> Tensor:
        """Squared error between the sampled action and the recorded ``action``,
        or with a mixture head the negative log-likelihood of ``action`` under
        the mixture (``compute_loss_bc_gmm``).

        Uses no critic. ``bc_deviation_max`` adds a hinge penalty on the distance
        from ``reference_action``; the gripper weighting and jerk penalty are
        shared with ``compute_loss_actor``.
        """
        if self.actor.head == "gmm":
            return self.compute_loss_bc_gmm(
                rl_token, proprioception, reference_action, action
            )
        self._validate_reference(reference_action, "reference_action")
        sampled, _ = self.actor(
            rl_token, proprioception, self._apply_ref_dropout(reference_action)
        )
        target = self.actor.normalize_action(
            action[:, : self.predicted_action_dim], proprioception
        ).clamp(-1.0, 1.0)
        loss = ((sampled - target).square() * self._bc_mask).sum(dim=-1)
        if self.bc_deviation_max > 0:
            # Only the part above bc_deviation_max is penalised.
            normalized_reference = self.actor.normalize_action(
                reference_action[:, : self.predicted_action_dim], proprioception
            ).clamp(-1.0, 1.0)
            spread = (sampled - normalized_reference).norm(dim=-1)
            overshoot = (spread - self.bc_deviation_max).clamp_min(0.0)
            loss = loss + self.bc_deviation_weight * overshoot.square()
        if self.jerk_lambda > 0:
            loss = loss + self.jerk_lambda * self.jerk_penalty(
                sampled, proprioception
            )
        return loss.mean()

    def compute_loss_bc_gmm(
        self,
        rl_token: Tensor,
        proprioception: Tensor,
        reference_action: Tensor,
        action: Tensor,
    ) -> Tensor:
        """Negative log-likelihood of the recorded ``action`` under the actor's
        mixture, summed over the chunk and weighted like the squared error
        (``bc_gripper_weight`` scales the gripper's log-density). The jerk
        penalty and the deviation hinge apply to every mode's mean."""
        self._validate_reference(reference_action, "reference_action")
        means, log_std, logits = self.actor.distribution(
            rl_token, proprioception, self._apply_ref_dropout(reference_action)
        )
        target = self.actor.normalize_action(
            action[:, : self.predicted_action_dim], proprioception
        ).clamp(-1.0, 1.0)
        log_likelihood, _ = chunk_log_prob(
            means.unsqueeze(1),
            log_std.unsqueeze(1),
            logits.unsqueeze(1),
            target.unsqueeze(1),
        )
        loss = -log_likelihood
        batch, modes, dim = means.shape
        flat_means = means.reshape(batch * modes, dim)
        flat_proprioception = proprioception.repeat_interleave(modes, dim=0)
        if self.bc_deviation_max > 0:
            normalized_reference = self.actor.normalize_action(
                reference_action[:, : self.predicted_action_dim], proprioception
            ).clamp(-1.0, 1.0)
            spread = (means - normalized_reference.unsqueeze(1)).norm(dim=-1)
            overshoot = (spread - self.bc_deviation_max).clamp_min(0.0)
            loss = loss + self.bc_deviation_weight * overshoot.square().mean(dim=-1)
        if self.jerk_lambda > 0:
            jerk = self.jerk_penalty(flat_means, flat_proprioception)
            loss = loss + self.jerk_lambda * jerk.view(batch, modes).mean(dim=-1)
        return loss.mean()

    @torch.no_grad()
    def mixture_stats(
        self, rl_token: Tensor, proprioception: Tensor, reference_action: Tensor
    ) -> dict[str, float]:
        """Mode entropy and mean std of the actor's mixture on a batch."""
        was_training = self.training
        self.eval()
        try:
            _, log_std, logits = self.actor.distribution(
                rl_token, proprioception, reference_action
            )
            weights = torch.softmax(logits, dim=-1)
            entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1)
            return {
                "mode_entropy": float(entropy.mean()),
                "std_mean": float(log_std.exp().mean()),
                "top_weight": float(weights.max(dim=-1).values.mean()),
            }
        finally:
            self.train(was_training)

    @torch.no_grad()
    def select_action(
        self,
        rl_token: Tensor,
        proprioception: Tensor,
        reference_action: Tensor,
        *,
        deterministic: bool = False,
    ) -> Tensor:
        """The deterministic chunk (the mean, or the likeliest mode's mean) or,
        without ``deterministic``, that chunk plus its exploration noise (with a
        mixture head: a mode drawn by its weight plus that mode's noise)."""
        was_training = self.training
        self.eval()
        try:
            if self.actor.head == "gmm":
                mean, std, _ = self.actor.pick_mode(
                    rl_token,
                    proprioception,
                    reference_action,
                    sample_mode=not deterministic,
                )
                action = mean
                if not deterministic:
                    action = (mean + torch.randn_like(mean) * std).clamp(-1.0, 1.0)
                return self.actor.unnormalize_action(action, proprioception)
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

    def load_policy_state(self, state_dict: dict[str, Tensor]) -> bool:
        """Load a learner ``policy_state_dict``. A mean-head checkpoint loaded
        into a mixture-head policy keeps its trunk and critics and starts every
        mode at the trained mean head; returns True when that conversion ran."""
        mean_weight = state_dict.get("actor.mean_head.weight")
        if self.actor.head == "gmm" and mean_weight is not None:
            mean_head = nn.Linear(mean_weight.shape[1], mean_weight.shape[0])
            with torch.no_grad():
                mean_head.weight.copy_(mean_weight)
                mean_head.bias.copy_(state_dict["actor.mean_head.bias"])
            remaining = {
                name: value
                for name, value in state_dict.items()
                if not name.startswith("actor.mean_head.")
            }
            result = self.load_state_dict(remaining, strict=False)
            unexpected = list(result.unexpected_keys)
            missing = [
                name
                for name in result.missing_keys
                if not name.startswith("actor.gmm_head.")
            ]
            if unexpected or missing:
                raise ValueError(
                    f"checkpoint does not match the policy: unexpected {unexpected[:5]}, "
                    f"missing {missing[:5]}"
                )
            self.actor.init_gmm_from_mean_head(
                mean_head.to(self.actor.gmm_head.linear.weight.device)
            )
            return True
        self.load_state_dict(state_dict)
        return False

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
        if self.q_loss_normalization:
            # Scale the Q term by alpha / mean|Q|.
            scale = self.q_loss_alpha / minimum_q.detach().abs().mean().clamp_min(1e-6)
            loss = -(scale * minimum_q) + self.bc_beta * bc_penalty
        else:
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
