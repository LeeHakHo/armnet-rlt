from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from armnet_rlt.artifacts import NormStats
from armnet_rlt.config import RLTConfig, so101_network_config
from armnet_rlt.dsrl import (
    LOG_STD_INIT,
    DSRLActor,
    DSRLPolicy,
    actor_from_steering_payload,
    choose_noise,
    random_noise,
    squash_gaussian,
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
    with pytest.raises(ValueError, match="target_entropy"):
        _config(dsrl_target_entropy=float("nan"))
    with pytest.raises(ValueError, match="alpha_init"):
        _config(dsrl_alpha_init=0.0)
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


def test_squashed_log_prob_matches_the_change_of_variables() -> None:
    torch.manual_seed(0)
    mean, log_std, epsilon = torch.randn(5, 7), torch.randn(5, 7) * 0.5, torch.randn(5, 7)
    action, log_prob = squash_gaussian(mean, log_std, epsilon)
    pre_tanh = mean + log_std.exp() * epsilon
    normal = torch.distributions.Normal(mean, log_std.exp())
    expected = (normal.log_prob(pre_tanh) - torch.log(1 - action.pow(2))).sum(-1)
    assert torch.allclose(action, torch.tanh(pre_tanh))
    assert torch.allclose(log_prob, expected, atol=1e-3, rtol=1e-4)
    # Saturating the squash concentrates the density: a mean of 6 pre-tanh
    # gives a far higher log probability than a mean of 0 at the same std.
    _, centered = squash_gaussian(torch.zeros(1, 7), torch.zeros(1, 7), torch.zeros(1, 7))
    _, saturated = squash_gaussian(torch.full((1, 7), 6.0), torch.zeros(1, 7), torch.zeros(1, 7))
    assert float(saturated) > float(centered) + 7 * 5


def test_actor_starts_near_unit_noise_and_samples_with_the_scaled_std() -> None:
    torch.manual_seed(0)
    actor = DSRLActor(_network(), noise_bound=2.0)
    assert torch.allclose(actor.log_std_head.bias, torch.full((18,), LOG_STD_INIT))
    token, proprio = torch.randn(1, 2048), torch.randn(1, 6)
    _, log_std = actor.distribution(token, proprio)
    assert torch.allclose(log_std, torch.full_like(log_std, LOG_STD_INIT), atol=0.2)
    token = token.expand(4096, -1).contiguous()
    proprio = proprio.expand(4096, -1).contiguous()
    with torch.no_grad():
        normalized, log_prob = actor.sample(token, proprio)
    noise = actor.unnormalize_action(normalized)
    assert normalized.shape == (4096, 18) and log_prob.shape == (4096,)
    assert float(noise.abs().max()) <= 2.0
    assert 0.8 < float(noise.std()) < 1.2
    with torch.no_grad():
        sampled, squashed_mean = actor(token[:2], proprio[:2], sample=True)
    assert not torch.allclose(sampled, squashed_mean)
    assert torch.allclose(squashed_mean, actor(token[:2], proprio[:2])[0])
    with torch.no_grad():
        wide, _ = actor.sample(token, proprio, std_scale=3.0)
        narrow, _ = actor.sample(token, proprio, std_scale=0.1)
    assert float(wide.std()) > float(normalized.std()) > float(narrow.std())
    with pytest.raises(ValueError, match="std_scale"):
        actor.sample(token[:1], proprio[:1], std_scale=0.0)
    epsilon = torch.zeros(1, 18)
    at_mean, _ = actor.sample(token[:1], proprio[:1], epsilon=epsilon)
    assert torch.allclose(at_mean, actor(token[:1], proprio[:1])[1])


def test_policy_trains_the_temperature_toward_the_target_entropy() -> None:
    torch.manual_seed(0)
    token, proprio = torch.randn(8, 2048), torch.randn(8, 6)
    for target, expected_sign in ((1e6, 1.0), (-1e6, -1.0)):
        policy = DSRLPolicy(_config(dsrl_target_entropy=target, dsrl_alpha_init=0.5))
        policy.set_norm_stats(_norm_stats())
        assert policy.temperature_parameters() == [policy.log_alpha]
        assert float(policy.alpha.detach()) == pytest.approx(0.5)
        loss = policy.compute_loss_actor(token, proprio)
        loss.backward()
        # Entropy below the target raises alpha (negative gradient on log_alpha).
        assert math.copysign(1.0, -float(policy.log_alpha.grad)) == expected_sign
        assert all(
            parameter.grad is not None for parameter in policy.actor.parameters()
        )
        stats = policy.last_actor_stats
        assert set(stats) == {"alpha", "entropy", "log_std", "noise_saturation"}
        assert 0.0 <= stats["noise_saturation"] <= 1.0
    # The soft target subtracts alpha * log pi of the next action.
    policy = DSRLPolicy(_config(dsrl_alpha_init=1e-8))
    policy.set_norm_stats(_norm_stats())
    noise = torch.rand(8, 18) * 4 - 2
    reward, done = torch.zeros(8), torch.zeros(8)
    torch.manual_seed(1)
    cold = policy.compute_loss_critic(token, proprio, noise, reward, token, proprio, noise, done)
    policy.log_alpha.data.fill_(math.log(10.0))
    torch.manual_seed(1)
    hot = policy.compute_loss_critic(token, proprio, noise, reward, token, proprio, noise, done)
    assert not torch.isclose(cold, hot)
    assert torch.isfinite(hot)


def test_sampled_noise_from_the_payload_actor_matches_the_learner_actor() -> None:
    torch.manual_seed(0)
    actor = DSRLActor(_network(), noise_bound=2.0)
    actor.eval()
    restored, _ = actor_from_steering_payload(
        steering_payload(actor.state_dict(), network=_network(), noise_bound=2.0, learner_step=1)
    )
    token, proprio = np.random.randn(2048).astype(np.float32), np.zeros(6, np.float32)
    first = choose_noise(
        actor, token, proprio, exploration_std=1.0, noise_bound=2.0, rng=np.random.default_rng(3)
    )
    second = choose_noise(
        restored, token, proprio, exploration_std=1.0, noise_bound=2.0, rng=np.random.default_rng(3)
    )
    assert np.allclose(first, second, atol=1e-6)
    assert np.abs(first).max() <= 2.0
    draws = np.stack(
        [
            choose_noise(actor, token, proprio, exploration_std=1.0, noise_bound=2.0, rng=rng)
            for rng in (np.random.default_rng(seed) for seed in range(200))
        ]
    )
    exact = choose_noise(actor, token, proprio, exploration_std=0.0, noise_bound=2.0, rng=np.random.default_rng(0))
    # Samples scatter around the squashed mean with roughly the initial unit spread.
    assert 0.6 < float(draws.std()) < 1.4
    assert np.abs(draws.mean(axis=0) - exact).max() < 0.5
    old_schema = steering_payload(actor.state_dict(), network=_network(), noise_bound=2.0, learner_step=1)
    old_schema["meta"] = old_schema["meta"].replace('"schema_version": 2', '"schema_version": 1')
    with pytest.raises(ValueError, match="schema"):
        actor_from_steering_payload(old_schema)
