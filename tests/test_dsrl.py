from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from armnet_rlt.artifacts import NormStats
from armnet_rlt.config import RLTConfig, so101_network_config
from armnet_rlt.dsrl import (
    HEAD_INIT_SCALE,
    LOG_STD_MAX,
    LOG_STD_MIN,
    DSRLActor,
    DSRLCritic,
    DSRLPolicy,
    actor_from_steering_payload,
    choose_noise,
    random_noise,
    relabel_episode,
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
    config = _config()
    assert config.dsrl_noise_steps == 1
    _config(dsrl_noise_steps=3)
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
    with pytest.raises(ValueError, match="noise_steps"):
        _config(dsrl_noise_steps=4)
    with pytest.raises(ValueError, match="noise_steps"):
        _config(dsrl_noise_steps=0)
    with pytest.raises(ValueError, match="target_entropy"):
        _config(dsrl_target_entropy=float("nan"))
    with pytest.raises(ValueError, match="alpha_init"):
        _config(dsrl_alpha_init=0.0)
    with pytest.raises(ValueError, match="temperature_lr"):
        _config(dsrl_temperature_lr=0.0)
    with pytest.raises(ValueError, match="initial_steps"):
        _config(dsrl_initial_steps=-1)
    with pytest.raises(ValueError, match="algorithm"):
        _config(algorithm="sac")


def test_rlt_config_still_ties_execution_to_the_predicted_chunk() -> None:
    with pytest.raises(ValueError, match="actions_to_execute"):
        RLTConfig(network=_network(), actions_to_execute=2, use_subsampling=False)


def test_networks_follow_the_official_layout() -> None:
    actor = DSRLActor(_network(hidden_dims=(16, 8)), noise_bound=2.5)
    assert actor.noise_dim == 6 and actor.noise_len == 3
    # Actor: Dense, ReLU per layer, no normalization; heads with a 1e-2 gain.
    assert [type(m).__name__ for m in actor.trunk] == ["Linear", "ReLU", "Linear", "ReLU"]
    assert actor.mean_head.out_features == 6 and actor.log_std_head.out_features == 6
    assert float(actor.mean_head.weight.detach().abs().max()) <= HEAD_INIT_SCALE + 1e-6
    assert float(actor.log_std_head.bias.detach().abs().max()) == 0.0
    assert (LOG_STD_MIN, LOG_STD_MAX) == (-20.0, 2.0)
    # Critic: Dense, LayerNorm, ReLU per layer on (token, state, noise).
    critic = DSRLCritic(_network(hidden_dims=(16, 8)), noise_dim=6)
    assert [type(m).__name__ for m in critic.trunk] == [
        "Linear", "LayerNorm", "ReLU", "Linear", "LayerNorm", "ReLU",
    ]
    assert critic.trunk[0].in_features == 2048 + 6 + 6
    token, proprio = torch.randn(4, 2048), torch.randn(4, 6)
    assert critic(token, proprio, torch.randn(4, 6)).shape == (4,)
    with pytest.raises(ValueError, match="noise has last dimension"):
        critic(token, proprio, torch.randn(4, 18))
    with pytest.raises(ValueError, match="noise_steps"):
        DSRLActor(_network(), noise_bound=2.5, noise_steps=4)


def test_actor_repeats_its_rows_over_the_chunk_and_recovers_them() -> None:
    actor = DSRLActor(_network(), noise_bound=2.0)
    rows = torch.arange(6.0).reshape(1, 6)
    chunk = actor.chunk_noise(rows)
    assert chunk.shape == (1, 18)
    assert torch.equal(chunk.reshape(3, 6), rows.expand(3, 6))
    assert torch.equal(actor.actor_noise(chunk), rows)
    two = DSRLActor(_network(), noise_bound=2.0, noise_steps=2)
    rows = torch.arange(12.0).reshape(1, 12)
    chunk = two.chunk_noise(rows)
    assert torch.equal(chunk.reshape(3, 6)[:2], rows.reshape(2, 6))
    assert torch.equal(chunk.reshape(3, 6)[2], rows.reshape(2, 6)[1])
    assert torch.equal(two.actor_noise(chunk), rows)
    with pytest.raises(ValueError, match="last dimension"):
        actor.actor_noise(torch.zeros(1, 6))


def test_policy_losses_and_noise_have_expected_shapes() -> None:
    policy = DSRLPolicy(_config())
    policy.set_norm_stats(_norm_stats())
    batch, chunk_dim = 4, 18
    token = torch.randn(batch, 2048)
    proprio = torch.randn(batch, 6)
    reference = torch.randn(batch, chunk_dim)
    noise = torch.rand(batch, chunk_dim) * 4 - 2

    critic_loss = policy.compute_loss_critic(
        token,
        proprio,
        noise,
        torch.tensor([-1.0, 0.0, -1.0, -1.0]),
        token,
        proprio,
        reference,
        torch.tensor([0.0, 1.0, 0.0, 0.0]),
    )
    actor_loss = policy.compute_loss_actor(token, proprio, reference)
    temperature_loss = policy.compute_loss_temperature()
    assert critic_loss.ndim == 0 and torch.isfinite(critic_loss)
    assert actor_loss.ndim == 0 and torch.isfinite(actor_loss)
    assert temperature_loss.ndim == 0 and torch.isfinite(temperature_loss)
    assert policy.min_q(token, proprio, noise).shape == (batch,)
    with pytest.raises(ValueError, match="last dimension"):
        policy.compute_loss_critic(
            token, proprio, noise[:, :6], torch.zeros(batch), token, proprio, reference,
            torch.zeros(batch),
        )

    deterministic = policy.select_action(token, proprio, deterministic=True)
    assert deterministic.shape == (batch, chunk_dim)
    assert float(deterministic.abs().max()) <= 2.0
    rows = deterministic.reshape(batch, 3, 6)
    assert torch.allclose(rows[:, 0], rows[:, 1]) and torch.allclose(rows[:, 0], rows[:, 2])
    explored = policy.select_action(token, proprio, exploration_std=10.0)
    assert float(explored.abs().max()) <= 2.0
    assert not torch.allclose(explored, deterministic)
    with pytest.raises(NotImplementedError):
        policy.compute_loss_bc()


def test_critic_target_has_no_entropy_term_and_bootstraps_through_truncation() -> None:
    torch.manual_seed(0)
    policy = DSRLPolicy(_config(discount=0.99, actions_to_execute=2))
    policy.set_norm_stats(_norm_stats())
    token, proprio = torch.randn(3, 2048), torch.randn(3, 6)
    noise = torch.rand(3, 18) * 4 - 2
    reward = torch.tensor([-1.0, 0.0, -1.0])
    done = torch.tensor([0.0, 1.0, 0.0])
    torch.manual_seed(1)
    with torch.no_grad():
        next_normalized, _ = policy.actor.sample(token, proprio)
        next_q = policy.critic_target(
            token, proprio, policy.actor.unnormalize_action(next_normalized)
        ).min(dim=0).values
        expected_target = reward + 0.99**2 * (1 - done) * next_q
        predictions = policy.critic_ensemble(token, proprio, policy.actor.actor_noise(noise))
        expected = (predictions - expected_target.unsqueeze(0)).pow(2).mean()
    torch.manual_seed(1)
    loss = policy.compute_loss_critic(token, proprio, noise, reward, token, proprio, noise, done)
    assert torch.allclose(loss, expected)
    # Changing the temperature leaves the critic loss untouched.
    policy.log_alpha.data.fill_(math.log(50.0))
    torch.manual_seed(1)
    hot = policy.compute_loss_critic(token, proprio, noise, reward, token, proprio, noise, done)
    assert torch.allclose(hot, expected)


def test_actor_loss_and_temperature_match_the_official_sac() -> None:
    torch.manual_seed(0)
    token, proprio = torch.randn(8, 2048), torch.randn(8, 6)
    policy = DSRLPolicy(_config(dsrl_alpha_init=0.5, dsrl_target_entropy=1.0))
    policy.set_norm_stats(_norm_stats())
    assert policy.temperature_parameters() == [policy.log_alpha]
    assert float(policy.alpha.detach()) == pytest.approx(0.5)
    torch.manual_seed(2)
    loss = policy.compute_loss_actor(token, proprio)
    torch.manual_seed(2)
    with torch.no_grad():
        normalized, log_prob = policy.actor.sample(token, proprio)
        q = policy.critic_ensemble(
            token, proprio, policy.actor.unnormalize_action(normalized)
        ).min(dim=0).values
        expected = (0.5 * log_prob - q).mean()
    assert torch.allclose(loss, expected)
    loss.backward()
    assert policy.log_alpha.grad is None
    assert all(parameter.grad is not None for parameter in policy.actor.parameters())
    entropy = float(policy.last_entropy)
    assert entropy == pytest.approx(float(-log_prob.mean()))
    assert set(policy.last_actor_stats) == {
        "alpha", "entropy", "log_std", "mean_pi_norm", "q_pi", "noise_saturation",
    }
    # Temperature loss alpha (H - H_target): its gradient on log alpha is
    # alpha (H - H_target), so entropy below the target raises alpha.
    temperature_loss = policy.compute_loss_temperature()
    assert float(temperature_loss) == pytest.approx(0.5 * (entropy - 1.0), rel=1e-5)
    temperature_loss.backward()
    assert float(policy.log_alpha.grad) == pytest.approx(0.5 * (entropy - 1.0), rel=1e-4)


def test_squashed_log_prob_matches_the_change_of_variables() -> None:
    torch.manual_seed(0)
    mean, log_std, epsilon = torch.randn(5, 7), torch.randn(5, 7) * 0.5, torch.randn(5, 7)
    action, log_prob = squash_gaussian(mean, log_std, epsilon)
    pre_tanh = mean + log_std.exp() * epsilon
    normal = torch.distributions.Normal(mean, log_std.exp())
    expected = (normal.log_prob(pre_tanh) - torch.log(1 - action.pow(2))).sum(-1)
    assert torch.allclose(action, torch.tanh(pre_tanh))
    assert torch.allclose(log_prob, expected, atol=1e-3, rtol=1e-4)
    # Rescaling to [-b, b] lowers the density by log(b) per dimension.
    _, scaled = squash_gaussian(mean, log_std, epsilon, bound=2.5)
    assert torch.allclose(scaled, log_prob - 7 * math.log(2.5), atol=1e-4)
    actor = DSRLActor(_network(), noise_bound=2.5)
    token, proprio = torch.randn(2, 2048), torch.randn(2, 6)
    normalized, actor_log_prob = actor.sample(token, proprio, epsilon=torch.zeros(2, 6))
    mean, log_std = actor.distribution(token, proprio)
    _, reference = squash_gaussian(mean, log_std, torch.zeros(2, 6), bound=2.5)
    assert torch.allclose(actor_log_prob, reference)
    assert torch.allclose(normalized, torch.tanh(mean))


def test_actor_starts_with_unit_pre_tanh_std_and_scales_it() -> None:
    torch.manual_seed(0)
    actor = DSRLActor(_network(), noise_bound=2.5)
    token, proprio = torch.randn(1, 2048), torch.randn(1, 6)
    mean, log_std = actor.distribution(token, proprio)
    assert float(mean.abs().max()) < 0.5
    assert torch.allclose(log_std, torch.zeros_like(log_std), atol=0.5)
    token = token.expand(4096, -1).contiguous()
    proprio = proprio.expand(4096, -1).contiguous()
    with torch.no_grad():
        normalized, log_prob = actor.sample(token, proprio)
    assert normalized.shape == (4096, 6) and log_prob.shape == (4096,)
    noise = actor.unnormalize_action(normalized)
    assert float(noise.abs().max()) <= 2.5
    # tanh of a unit Gaussian has a standard deviation near 0.63.
    assert 0.5 < float(normalized.std()) < 0.75
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


def test_relabel_episode_uses_the_official_reward_and_truncation() -> None:
    def episode(final_reward: float) -> list[dict[str, torch.Tensor]]:
        return [
            {"reward": torch.tensor(0.0), "done": torch.tensor(0.0)},
            {"reward": torch.tensor(0.0), "done": torch.tensor(0.0)},
            {"reward": torch.tensor(final_reward), "done": torch.tensor(1.0)},
        ]

    success = episode(0.8)
    relabel_episode(success)
    assert [float(t["reward"]) for t in success] == [-1.0, -1.0, 0.0]
    assert [float(t["done"]) for t in success] == [0.0, 0.0, 1.0]
    failure = episode(0.0)
    relabel_episode(failure)
    assert [float(t["reward"]) for t in failure] == [-1.0, -1.0, -1.0]
    assert [float(t["done"]) for t in failure] == [0.0, 0.0, 0.0]
    penalized = episode(-0.3)
    relabel_episode(penalized)
    assert float(penalized[-1]["done"]) == 0.0


def test_steering_payload_round_trips_through_numpy_arrays() -> None:
    actor = DSRLActor(_network(), noise_bound=1.5, noise_steps=2)
    actor.set_norm_stats(_norm_stats(), use_quantiles=True)
    payload = steering_payload(
        actor.state_dict(), network=_network(), noise_bound=1.5, noise_steps=2, learner_step=42
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
    assert restored.noise_steps == 2 and restored.noise_dim == 12
    token, proprio = torch.randn(1, 2048), torch.randn(1, 6)
    assert torch.allclose(actor(token, proprio)[1], restored(token, proprio)[1])
    for version in (1, 2, 99):
        bad = dict(payload)
        bad["meta"] = payload["meta"].replace('"schema_version": 3', f'"schema_version": {version}')
        with pytest.raises(ValueError, match="schema"):
            actor_from_steering_payload(bad)


def test_chosen_noise_follows_the_actor_and_repeats_over_the_chunk() -> None:
    actor = DSRLActor(_network(), noise_bound=2.5)
    actor.eval()
    token, proprio = np.random.randn(2048).astype(np.float32), np.zeros(6, np.float32)
    rng = np.random.default_rng(0)
    exact = choose_noise(
        actor, token, proprio, exploration_std=0.0, noise_bound=2.5, rng=rng
    )
    assert exact.shape == (3, 6) and exact.dtype == np.float32
    with torch.inference_mode():
        _, mean = actor(torch.from_numpy(token)[None], torch.from_numpy(proprio)[None])
    assert np.allclose(exact, np.repeat(mean.numpy().reshape(1, 6) * 2.5, 3, axis=0), atol=1e-6)
    noisy = choose_noise(
        actor, token, proprio, exploration_std=1.0, noise_bound=2.5, rng=rng
    )
    assert np.abs(noisy).max() <= 2.5
    assert not np.allclose(noisy, exact)
    assert np.array_equal(noisy[0], noisy[1]) and np.array_equal(noisy[0], noisy[2])

    unsteered = random_noise((3, 6), exploration_std=1.0, noise_bound=2.5, rng=rng)
    assert unsteered.shape == (3, 6) and np.abs(unsteered).max() <= 2.5
    assert not np.array_equal(unsteered[0], unsteered[1])
    repeated = random_noise(
        (3, 6), exploration_std=1.0, noise_bound=2.5, rng=rng, noise_steps=1
    )
    assert np.array_equal(repeated[0], repeated[1]) and np.array_equal(repeated[0], repeated[2])
    assert np.array_equal(
        random_noise((3, 6), exploration_std=0.0, noise_bound=2.5, rng=rng),
        np.zeros((3, 6), np.float32),
    )


def test_sampled_noise_from_the_payload_actor_matches_the_learner_actor() -> None:
    torch.manual_seed(0)
    actor = DSRLActor(_network(), noise_bound=2.5)
    actor.eval()
    restored, _ = actor_from_steering_payload(
        steering_payload(actor.state_dict(), network=_network(), noise_bound=2.5, learner_step=1)
    )
    token, proprio = np.random.randn(2048).astype(np.float32), np.zeros(6, np.float32)
    first = choose_noise(
        actor, token, proprio, exploration_std=1.0, noise_bound=2.5, rng=np.random.default_rng(3)
    )
    second = choose_noise(
        restored, token, proprio, exploration_std=1.0, noise_bound=2.5, rng=np.random.default_rng(3)
    )
    assert np.allclose(first, second, atol=1e-6)
    draws = np.stack(
        [
            choose_noise(actor, token, proprio, exploration_std=1.0, noise_bound=2.5, rng=rng)[0]
            for rng in (np.random.default_rng(seed) for seed in range(300))
        ]
    )
    exact = choose_noise(
        actor, token, proprio, exploration_std=0.0, noise_bound=2.5, rng=np.random.default_rng(0)
    )[0]
    # 2.5 tanh(N(0, 1)) has a standard deviation near 1.6 around the mean.
    assert 1.2 < float(draws.std()) < 2.0
    assert np.abs(draws.mean(axis=0) - exact).max() < 0.5


def test_learner_config_applies_the_official_dsrl_settings(tmp_path) -> None:
    from armnet_rlt.learner_config import learner_config

    config = learner_config(
        data_root=tmp_path,
        embodiment="so101",
        config_name="cfg",
        run_id="run",
        algorithm="dsrl",
        online_steps=500_000,
    )
    assert config.network.hidden_dims == (1024, 1024, 1024)
    assert config.network.predicted_action_len == 30
    assert config.dsrl_noise_steps == 1 and config.dsrl_noise_bound == 2.5
    assert config.dsrl_alpha_init == 1.0 and config.dsrl_target_entropy == 0.0
    assert config.dsrl_temperature_lr == 3e-4 and config.dsrl_initial_steps == 5000
    assert config.num_critics == 2 and config.utd_ratio == 30
    assert config.discount == 0.99 and config.actions_to_execute == 10
    assert config.actor_lr == config.actor_lr_min == 1e-4
    assert config.critic_lr == config.critic_lr_min == 3e-4
    assert config.policy_update_freq == 1 and config.max_q is None
    assert config.online_step_before_learning == 1
    assert config.online_buffer_capacity == 2 * 500_000 // 30
    assert config.batch_size == 256 and config.max_demo_pretrain_steps == 0
    rlt = learner_config(
        data_root=tmp_path, embodiment="so101", config_name="cfg", run_id="run"
    )
    assert rlt.network.hidden_dims == (512, 512, 512, 512)
    assert rlt.num_critics == 4 and rlt.max_q == 2.0


def test_dsrl_gradient_step_updates_every_module_and_schedules_utd_steps() -> None:
    from armnet_rlt.learner import _schedule_online_updates, _train_step_dsrl

    torch.manual_seed(0)
    policy = DSRLPolicy(_config(dsrl_alpha_init=0.5))
    policy.set_norm_stats(_norm_stats())
    batch = {
        "state": {
            "rl_token": torch.randn(4, 2048),
            "proprioception": torch.randn(4, 6),
            "reference_action": torch.randn(4, 18),
        },
        "action": torch.rand(4, 18) * 4 - 2,
        "reward": torch.tensor([-1.0, -1.0, 0.0, -1.0]),
        "next_state": {
            "rl_token": torch.randn(4, 2048),
            "proprioception": torch.randn(4, 6),
            "reference_action": torch.randn(4, 18),
        },
        "done": torch.tensor([0.0, 0.0, 1.0, 0.0]),
    }
    before = {
        name: value.clone() for name, value in policy.state_dict().items()
        if value.dtype.is_floating_point
    }
    critic_loss, actor_loss = _train_step_dsrl(
        policy,
        batch,
        actor_optimizer=torch.optim.Adam(policy.actor.parameters(), lr=1e-3),
        critic_optimizer=torch.optim.Adam(policy.critic_ensemble.parameters(), lr=1e-3),
        temperature_optimizer=torch.optim.Adam(policy.temperature_parameters(), lr=1e-3),
    )
    assert math.isfinite(critic_loss) and math.isfinite(actor_loss)
    after = policy.state_dict()
    changed = {name for name in before if not torch.equal(before[name], after[name])}
    assert any(name.startswith("actor.trunk") for name in changed)
    assert any(name.startswith("critic_ensemble") for name in changed)
    assert any(name.startswith("critic_target") for name in changed)
    assert "log_alpha" in changed
    assert _schedule_online_updates(0, 17, steps_per_transition=30) == 510
    assert _schedule_online_updates(5, 2) == 7
    with pytest.raises(ValueError):
        _schedule_online_updates(0, 1, steps_per_transition=0)
