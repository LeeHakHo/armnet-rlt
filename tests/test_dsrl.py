from __future__ import annotations

import numpy as np
import pytest
import torch

from armnet_rlt.artifacts import NormStats
from armnet_rlt.config import RLTConfig, so101_network_config
from armnet_rlt.dsrl import (
    DSRLActor,
    DSRLPolicy,
    actor_from_steering_payload,
    choose_noise,
    random_noise,
    steering_payload,
)


def _network(**overrides):
    values = {
        "hidden_dims": (16,),
        "reference_action_len": 3,
        "predicted_action_len": 3,
    }
    values.update(overrides)
    return so101_network_config(**values)


def _config(**overrides) -> RLTConfig:
    values = {
        "network": _network(),
        "algorithm": "dsrl",
        "actions_to_execute": 2,
        "use_subsampling": False,
        "num_critics": 2,
        "utd_ratio": 1,
        "max_demo_pretrain_steps": 0,
        "online_batch_fraction": 1.0,
        "dsrl_noise_bound": 2.0,
    }
    values.update(overrides)
    return RLTConfig(**values)


def _norm_stats() -> dict[str, NormStats]:
    return {
        "state": NormStats(np.zeros(6, np.float32), np.ones(6, np.float32)),
        "actions": NormStats(np.zeros(6, np.float32), np.ones(6, np.float32)),
    }


def test_dsrl_config_requires_full_chunk_noise_and_no_demos() -> None:
    _config()
    with pytest.raises(ValueError, match="reference_action_len"):
        _config(network=_network(predicted_action_len=2))
    with pytest.raises(ValueError, match="actions_to_execute"):
        _config(actions_to_execute=4)
    with pytest.raises(ValueError, match="pretraining"):
        _config(max_demo_pretrain_steps=5)
    with pytest.raises(ValueError, match="online transitions only"):
        _config(online_batch_fraction=0.5)
    with pytest.raises(ValueError, match="noise_bound"):
        _config(dsrl_noise_bound=0.0)
    with pytest.raises(ValueError, match="algorithm"):
        _config(algorithm="sac")


def test_rlt_config_still_ties_execution_to_the_predicted_chunk() -> None:
    with pytest.raises(ValueError, match="actions_to_execute"):
        RLTConfig(network=_network(), actions_to_execute=2, use_subsampling=False)


def test_policy_losses_and_noise_have_expected_shapes() -> None:
    policy = DSRLPolicy(_config())
    policy.set_norm_stats(_norm_stats())
    batch, noise_dim = 4, 18
    token = torch.randn(batch, 2048)
    proprio = torch.randn(batch, 6)
    reference = torch.randn(batch, noise_dim)
    noise = torch.rand(batch, noise_dim) * 4 - 2

    critic_loss = policy.compute_loss_critic(
        token,
        proprio,
        noise,
        torch.tensor([0.0, 1.0, 0.0, 1.0]),
        token,
        proprio,
        reference,
        torch.tensor([0.0, 1.0, 0.0, 0.0]),
    )
    actor_loss = policy.compute_loss_actor(token, proprio, reference)
    assert critic_loss.ndim == 0 and torch.isfinite(critic_loss)
    assert actor_loss.ndim == 0 and torch.isfinite(actor_loss)
    with pytest.raises(ValueError, match="last dimension"):
        policy.compute_loss_critic(
            token, proprio, noise[:, :6], torch.zeros(batch), token, proprio, reference,
            torch.zeros(batch),
        )

    deterministic = policy.select_action(token, proprio, deterministic=True)
    assert deterministic.shape == (batch, noise_dim)
    assert float(deterministic.abs().max()) <= 2.0
    explored = policy.select_action(token, proprio, exploration_std=10.0)
    assert float(explored.abs().max()) <= 2.0
    assert not torch.allclose(explored, deterministic)
    with pytest.raises(NotImplementedError):
        policy.compute_loss_bc()


def test_actor_normalizes_noise_by_its_bound_and_ignores_the_reference() -> None:
    actor = DSRLActor(_network(), noise_bound=2.0)
    noise = torch.tensor([[2.0, -1.0, 0.5] + [0.0] * 15])
    assert torch.allclose(actor.normalize_action(noise), noise / 2)
    assert torch.allclose(actor.unnormalize_action(noise / 2), noise)
    token, proprio = torch.randn(1, 2048), torch.randn(1, 6)
    _, without = actor(token, proprio)
    _, with_reference = actor(token, proprio, torch.randn(1, 18))
    assert torch.equal(without, with_reference)
    assert float(without.detach().abs().max()) <= 1.0


def test_steering_payload_round_trips_through_numpy_arrays() -> None:
    actor = DSRLActor(_network(), noise_bound=1.5)
    actor.set_norm_stats(_norm_stats(), use_quantiles=True)
    payload = steering_payload(
        actor.state_dict(), network=_network(), noise_bound=1.5, learner_step=42
    )
    assert isinstance(payload["meta"], str)
    assert all(
        isinstance(value, np.ndarray)
        for key, value in payload.items()
        if key != "meta"
    )
    restored, step = actor_from_steering_payload(payload)
    assert step == 42
    assert not restored.training
    assert float(restored.noise_bound) == 1.5
    token, proprio = torch.randn(1, 2048), torch.randn(1, 6)
    assert torch.allclose(actor(token, proprio)[1], restored(token, proprio)[1])
    bad = dict(payload)
    bad["meta"] = '{"schema_version": 99}'
    with pytest.raises(ValueError, match="schema"):
        actor_from_steering_payload(bad)


def test_chosen_noise_follows_the_actor_and_stays_within_the_bound() -> None:
    actor = DSRLActor(_network(), noise_bound=2.0)
    actor.eval()
    token, proprio = np.random.randn(2048).astype(np.float32), np.zeros(6, np.float32)
    rng = np.random.default_rng(0)
    exact = choose_noise(
        actor, token, proprio, exploration_std=0.0, noise_bound=2.0, rng=rng
    )
    assert exact.shape == (3, 6) and exact.dtype == np.float32
    with torch.inference_mode():
        _, mean = actor(torch.from_numpy(token)[None], torch.from_numpy(proprio)[None])
    assert np.allclose(exact, mean.numpy().reshape(3, 6) * 2.0, atol=1e-6)
    noisy = choose_noise(
        actor, token, proprio, exploration_std=50.0, noise_bound=2.0, rng=rng
    )
    assert np.abs(noisy).max() <= 2.0
    assert not np.allclose(noisy, exact)

    unsteered = random_noise((3, 6), exploration_std=1.0, noise_bound=2.0, rng=rng)
    assert unsteered.shape == (3, 6) and np.abs(unsteered).max() <= 2.0
    assert np.array_equal(
        random_noise((3, 6), exploration_std=0.0, noise_bound=2.0, rng=rng),
        np.zeros((3, 6), np.float32),
    )
