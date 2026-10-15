from __future__ import annotations

import numpy as np
import pytest
import torch

from armnet_rlt.artifacts import NormStats
from armnet_rlt.config import RLTConfig, RLTNetworkConfig, so101_network_config
from armnet_rlt.networks import RLTActor, RLTCritic
from armnet_rlt.policy import RLTPolicy


def _network(**overrides) -> RLTNetworkConfig:
    values = dict(
        hidden_dims=(16,),
        reference_action_len=3,
        predicted_action_len=3,
        fixed_action_std=0.0,
        ref_action_dropout=0.25,
    )
    values.update(overrides)
    return so101_network_config(**values)


def _policy(**network_overrides) -> RLTPolicy:
    config = RLTConfig(
        network=_network(**network_overrides),
        actions_to_execute=3,
        use_subsampling=False,
        num_critics=2,
        utd_ratio=1,
    )
    policy = RLTPolicy(config)
    identity = NormStats(np.zeros(6, np.float32), np.ones(6, np.float32))
    policy.set_norm_stats({"state": identity, "actions": identity})
    return policy


def _batch(count: int = 4):
    torch.manual_seed(0)
    return (
        torch.randn(count, 2048) * 0.1,
        torch.randn(count, 6) * 0.1,
        torch.randn(count, 9 * 2) * 0.1,  # reference: 3 steps x 6 dims
    )


def test_residual_actor_starts_exactly_at_the_reference() -> None:
    policy = _policy(residual_actor=True)
    token, state, reference = _batch()

    action = policy.select_action(token, state, reference, deterministic=True)

    torch.testing.assert_close(
        action, reference[:, :18], atol=1e-4, rtol=0
    )


def test_default_actor_is_not_residual() -> None:
    policy = _policy()
    token, state, reference = _batch()

    action = policy.select_action(token, state, reference, deterministic=True)

    assert not bool(policy.actor.residual_mode)
    assert (action - reference[:, :18]).abs().max() > 1e-3


def test_reference_dropout_is_disabled_for_the_residual_actor() -> None:
    assert _policy(residual_actor=True).ref_dropout == 0.0
    assert _policy().ref_dropout == 0.25


def test_residual_head_receives_gradient_from_the_actor_loss() -> None:
    policy = _policy(residual_actor=True)
    token, state, reference = _batch()

    policy.compute_loss_actor(token, state, reference).backward()

    assert policy.actor.mean_head.weight.grad.abs().sum() > 0


def test_residual_mode_travels_with_the_state_dict() -> None:
    learner = _policy(residual_actor=True, residual_scale=0.5).actor
    actor = RLTActor(_network())  # built as a plain actor, like the cell job
    identity = NormStats(np.zeros(6, np.float32), np.ones(6, np.float32))
    actor.set_norm_stats({"state": identity, "actions": identity})
    assert not bool(actor.residual_mode)

    actor.load_state_dict(learner.state_dict())

    assert bool(actor.residual_mode)
    assert float(actor.residual_scale) == 0.5


def test_old_state_dicts_without_the_new_buffers_still_load() -> None:
    actor = RLTActor(_network())
    old = {
        key: value
        for key, value in actor.state_dict().items()
        if key not in ("residual_mode", "residual_scale")
    }

    result = actor.load_state_dict(old)

    assert result.missing_keys == []
    assert not bool(actor.residual_mode)


def test_critic_action_residual_depends_only_on_action_minus_reference() -> None:
    critic = RLTCritic(_network(critic_action_residual=True))
    token, state, _ = _batch()
    action = torch.randn(4, 18) * 0.1
    reference = torch.randn(4, 18) * 0.1
    shift = torch.full((4, 18), 0.3)

    first = critic(token, state, action, reference)
    shifted = critic(token, state, action + shift, reference + shift)

    torch.testing.assert_close(first, shifted, atol=1e-5, rtol=0)


def test_critic_action_residual_requires_the_reference() -> None:
    critic = RLTCritic(_network(critic_action_residual=True))
    token, state, _ = _batch()

    with pytest.raises(ValueError, match="reference"):
        critic(token, state, torch.zeros(4, 18))


def test_plain_critic_ignores_the_reference_argument() -> None:
    critic = RLTCritic(_network())
    token, state, _ = _batch()
    action = torch.randn(4, 18) * 0.1

    with_reference = critic(token, state, action, torch.ones(4, 18))
    without = critic(token, state, action)

    torch.testing.assert_close(with_reference, without)


@pytest.mark.parametrize(
    "overrides",
    [
        {"residual_actor": True},
        {"critic_action_residual": True},
        {"residual_actor": True, "critic_action_residual": True},
    ],
)
def test_losses_run_with_the_structural_variants(overrides) -> None:
    policy = _policy(**overrides)
    token, state, reference = _batch()
    action = torch.randn(4, 18) * 0.1

    critic_loss = policy.compute_loss_critic(
        token, state, action, torch.zeros(4), token, state, reference,
        torch.zeros(4), reference_action=reference,
    )
    actor_loss = policy.compute_loss_actor(token, state, reference)

    assert torch.isfinite(critic_loss) and torch.isfinite(actor_loss)


def test_critic_loss_needs_the_current_reference_in_critic_residual_mode() -> None:
    policy = _policy(critic_action_residual=True)
    token, state, reference = _batch()

    with pytest.raises(ValueError, match="reference_action"):
        policy.compute_loss_critic(
            token, state, torch.zeros(4, 18), torch.zeros(4), token, state,
            reference, torch.zeros(4),
        )


def test_invalid_residual_settings_are_rejected() -> None:
    with pytest.raises(ValueError, match="residual_scale"):
        _network(residual_scale=0.0)
    with pytest.raises(ValueError, match="critic_action_gain"):
        _network(critic_action_gain=-1.0)
