from __future__ import annotations

import numpy as np
import torch

from armnet_rlt.artifacts import NormStats
from armnet_rlt.config import RLTConfig, so101_network_config
from armnet_rlt.policy import RLTPolicy


def _policy() -> RLTPolicy:
    network = so101_network_config(
        hidden_dims=(16,),
        reference_action_len=3,
        predicted_action_len=3,
        fixed_action_std=0.0,
        ref_action_dropout=0.0,
    )
    config = RLTConfig(
        network=network,
        actions_to_execute=3,
        use_subsampling=False,
        num_critics=2,
        utd_ratio=1,
    )
    policy = RLTPolicy(config)
    policy.set_norm_stats(
        {
            "state": NormStats(np.zeros(6, np.float32), np.ones(6, np.float32)),
            "actions": NormStats(
                np.zeros(6, np.float32), np.ones(6, np.float32)
            ),
        }
    )
    return policy


def test_policy_action_and_losses_have_expected_shapes() -> None:
    policy = _policy()
    batch = 2
    token = torch.randn(batch, 2048)
    state = torch.randn(batch, 6)
    reference = torch.randn(batch, 18)
    action = torch.randn(batch, 18)

    selected = policy.select_action(
        token, state, reference, deterministic=True
    )
    assert selected.shape == (batch, 18)

    critic_loss = policy.compute_loss_critic(
        token,
        state,
        action,
        torch.zeros(batch),
        token,
        state,
        reference,
        torch.zeros(batch),
    )
    actor_loss = policy.compute_loss_actor(token, state, reference)
    assert critic_loss.ndim == 0 and torch.isfinite(critic_loss)
    assert actor_loss.ndim == 0 and torch.isfinite(actor_loss)


def test_jerk_penalty_is_zero_for_constant_velocity() -> None:
    policy = _policy()
    timesteps = torch.tensor([0.0, 1.0, 2.0]).view(1, 3, 1)
    action = timesteps.repeat(1, 1, 6).flatten(1)
    assert torch.equal(policy.jerk_penalty(action), torch.zeros(1))


def test_delta_mask_leaves_gripper_absolute() -> None:
    network = so101_network_config(
        hidden_dims=(8,), reference_action_len=3, predicted_action_len=3
    )
    config = RLTConfig(
        network=network,
        actions_to_execute=3,
        use_subsampling=False,
        policy_uses_delta_actions=True,
    )
    policy = RLTPolicy(config)
    stats = NormStats(np.zeros(6, np.float32), np.ones(6, np.float32))
    policy.set_norm_stats({"state": stats, "actions": stats})
    state = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.float32)
    raw = state.repeat(1, 3)
    normalized = policy.actor.normalize_action(raw, state)
    normalized = normalized.view(1, 3, 6)
    assert torch.equal(normalized[..., :5], torch.zeros(1, 3, 5))
    assert torch.allclose(normalized[..., 5], torch.full((1, 3), 6.0))


def test_openpi_padded_action_stats_are_sliced_to_native_six_dims() -> None:
    policy = _policy()
    padded = NormStats(
        np.arange(32, dtype=np.float32),
        np.ones(32, dtype=np.float32),
    )
    state = NormStats(np.zeros(6, np.float32), np.ones(6, np.float32))

    policy.set_norm_stats({"state": state, "actions": padded})

    assert policy.actor.na_mean.shape == (18,)
    assert torch.equal(
        policy.actor.na_mean[:6], torch.arange(6, dtype=torch.float32)
    )
