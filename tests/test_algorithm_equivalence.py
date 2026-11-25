"""The ported rlt and dsrl algorithms give the same chunks and losses as the
code paths they replace (copied in legacy_reference.py)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import legacy_reference as legacy
from armnet_rlt.algorithm import ActorInput, AlgorithmSpec, norm_stats_lists
from armnet_rlt.algorithms.dsrl import DSRLAlgorithm
from armnet_rlt.algorithms.rlt import RLTAlgorithm
from armnet_rlt.artifacts import NormStats
from armnet_rlt.config import RLTConfig, so101_network_config
from armnet_rlt.dsrl import DSRLPolicy
from armnet_rlt.policy import RLTPolicy


def _norm_stats(chunk: int) -> dict[str, NormStats]:
    generator = np.random.default_rng(7)
    return {
        "state": NormStats(
            generator.normal(size=6).astype(np.float32),
            generator.uniform(0.5, 2.0, size=6).astype(np.float32),
            generator.uniform(-3, -1, size=6).astype(np.float32),
            generator.uniform(1, 3, size=6).astype(np.float32),
        ),
        "actions": NormStats(
            generator.normal(size=(chunk, 6)).astype(np.float32),
            generator.uniform(0.5, 2.0, size=(chunk, 6)).astype(np.float32),
            generator.uniform(-3, -1, size=(chunk, 6)).astype(np.float32),
            generator.uniform(1, 3, size=(chunk, 6)).astype(np.float32),
        ),
    }


def _rlt_config() -> RLTConfig:
    network = so101_network_config(
        hidden_dims=(32, 32),
        reference_action_len=6,
        predicted_action_len=3,
        residual_actor=True,
        critic_action_residual=True,
    )
    return RLTConfig(
        network=network,
        actions_to_execute=3,
        use_subsampling=False,
        num_critics=2,
        utd_ratio=2,
        policy_update_freq=2,
        max_demo_pretrain_steps=4,
        policy_uses_delta_actions=True,
        device="cpu",
    )


def _dsrl_config() -> RLTConfig:
    network = so101_network_config(
        hidden_dims=(32, 32), reference_action_len=6, predicted_action_len=6
    )
    return RLTConfig(
        network=network,
        algorithm="dsrl",
        actions_to_execute=3,
        use_subsampling=False,
        num_critics=2,
        max_demo_pretrain_steps=0,
        online_batch_fraction=1.0,
        dsrl_noise_bound=1.5,
        dsrl_noise_steps=2,
        dsrl_alpha_init=0.7,
        device="cpu",
    )


def _batch(config: RLTConfig, generator: torch.Generator) -> dict:
    network = config.network
    size = 8

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    return {
        "state": {
            "rl_token": randn(size, 2048),
            "proprioception": randn(size, 6),
            "reference_action": randn(size, network.reference_action_dim),
        },
        "action": randn(size, network.predicted_action_dim),
        "reward": (torch.rand(size, generator=generator) > 0.5).float(),
        "next_state": {
            "rl_token": randn(size, 2048),
            "proprioception": randn(size, 6),
            "reference_action": randn(size, network.reference_action_dim),
        },
        "done": (torch.rand(size, generator=generator) > 0.7).float(),
    }


def _assert_same_state(first: dict, second: dict) -> None:
    assert first.keys() == second.keys()
    for name in first:
        assert torch.equal(first[name], second[name]), name


def test_rlt_pretrain_and_update_match_the_old_train_step() -> None:
    config = _rlt_config()
    norm_stats = _norm_stats(config.network.reference_action_len)

    torch.manual_seed(0)
    old = RLTPolicy(config)
    old.set_norm_stats(norm_stats, use_quantiles=config.use_quantile_norm)
    old.to("cpu").train()
    actor_optimizer = torch.optim.Adam(old.actor.parameters(), lr=config.actor_lr)
    critic_optimizer = torch.optim.Adam(old.critic_ensemble.parameters(), lr=config.critic_lr)
    actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        actor_optimizer, T_max=config.max_demo_pretrain_steps, eta_min=config.actor_lr_min
    )
    critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        critic_optimizer, T_max=config.max_demo_pretrain_steps, eta_min=config.critic_lr_min
    )

    torch.manual_seed(0)
    new = RLTAlgorithm(AlgorithmSpec(norm_stats=norm_stats_lists(norm_stats)), config)

    generator = torch.Generator().manual_seed(1)
    for step in range(7):
        batch = _batch(config, generator)
        torch.manual_seed(100 + step)
        old_critic, old_actor = legacy._train_step(
            old,
            batch,
            step=step,
            config=config,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
        )
        pretraining = step < config.max_demo_pretrain_steps
        if pretraining:
            actor_scheduler.step()
            critic_scheduler.step()
        torch.manual_seed(100 + step)
        metrics = new.pretrain(batch) if pretraining else new.update(batch)
        assert metrics["critic_loss"] == old_critic
        assert metrics.get("actor_loss") == old_actor
        assert metrics["actor_lr"] == actor_optimizer.param_groups[0]["lr"]
    _assert_same_state(old.state_dict(), new.policy.state_dict())
    _assert_same_state(old.actor_state_bytes(), new.actor_state())


def test_dsrl_update_matches_the_old_sac_step() -> None:
    config = _dsrl_config()
    norm_stats = _norm_stats(config.network.reference_action_len)

    torch.manual_seed(0)
    old = DSRLPolicy(config)
    old.set_norm_stats(norm_stats, use_quantiles=config.use_quantile_norm)
    old.to("cpu").train()
    actor_optimizer = torch.optim.Adam(old.actor.parameters(), lr=config.actor_lr)
    critic_optimizer = torch.optim.Adam(old.critic_ensemble.parameters(), lr=config.critic_lr)
    temperature_optimizer = torch.optim.Adam(
        old.temperature_parameters(), lr=config.dsrl_temperature_lr
    )

    torch.manual_seed(0)
    new = DSRLAlgorithm(AlgorithmSpec(norm_stats=norm_stats_lists(norm_stats)), config)

    generator = torch.Generator().manual_seed(1)
    network = config.network
    for step in range(5):
        batch = _batch(config, generator)
        # The old transition action was the chunk noise; it is now an extra
        # next to the executed chunk.
        noise = torch.rand(8, network.predicted_action_dim, generator=generator) * 3 - 1.5
        old_batch = {**batch, "action": noise}
        new_batch = {
            **batch,
            "action": torch.randn(8, config.actions_to_execute * 6, generator=generator),
            "extras": {"noise": noise.reshape(8, network.predicted_action_len, 6)},
        }
        torch.manual_seed(200 + step)
        old_critic, old_actor = legacy._train_step_dsrl(
            old,
            old_batch,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            temperature_optimizer=temperature_optimizer,
        )
        torch.manual_seed(200 + step)
        metrics = new.update(new_batch)
        assert metrics["critic_loss"] == old_critic
        assert metrics["actor_loss"] == old_actor
        assert metrics["alpha"] == old.last_actor_stats["alpha"]
    _assert_same_state(old.state_dict(), new.policy.state_dict())


def _rlt_actor_pair(config: RLTConfig):
    norm_stats = _norm_stats(config.network.reference_action_len)
    torch.manual_seed(3)
    algorithm = RLTAlgorithm(AlgorithmSpec(norm_stats=norm_stats_lists(norm_stats)), config)
    with torch.no_grad():
        # A nonzero residual head, so the refined chunk differs from the base.
        algorithm.policy.actor.mean_head.weight.normal_(0.0, 0.05)
    old = algorithm.policy.actor
    new = RLTAlgorithm.build_actor(algorithm.spec, config)
    new.load_state_dict(algorithm.actor_state())
    new.eval()
    return old, new


@pytest.mark.parametrize("base", [False, True])
@pytest.mark.parametrize("scale", [0.0, 0.5])
def test_rlt_actor_gives_the_old_cell_chunk(base: bool, scale: float) -> None:
    config = _rlt_config()
    old, new = _rlt_actor_pair(config)
    old.train()
    generator = torch.Generator().manual_seed(5)
    token = torch.randn(2048, generator=generator)
    proprioception = torch.randn(6, generator=generator)
    # π0.5's chunk is longer than the reference; the cell kept the first rows.
    base_chunk = torch.randn(9, 6, generator=generator)
    reference = base_chunk[: config.network.reference_action_len]

    torch.manual_seed(11)
    if base:
        expected = reference[: config.network.predicted_action_len].clone()
        if scale > 0:
            expected = legacy._jitter_chunk(
                old,
                expected,
                proprioception,
                predicted_len=3,
                action_dim=6,
                exploration_correlation=0.85,
                exploration_scale=scale,
            )
    else:
        expected = legacy._refine_chunk(
            old,
            token,
            proprioception,
            reference,
            predicted_len=3,
            action_dim=6,
            exploration_correlation=0.85,
            exploration_scale=scale,
        )
    torch.manual_seed(11)
    output = new.act(
        ActorInput(
            proprioception=proprioception,
            rl_token=token,
            exploration_scale=scale,
            exploration_correlation=0.85,
            base=base,
        ),
        reference,
        explore=scale > 0,
    )
    assert new.noise(ActorInput(proprioception=proprioception), explore=True) is None
    assert torch.equal(output.chunk, expected)
    assert output.extras == {}


@pytest.mark.parametrize("base", [False, True])
@pytest.mark.parametrize("scale", [0.0, 0.5])
def test_dsrl_actor_gives_the_old_server_noise_and_chunk(base: bool, scale: float) -> None:
    config = _dsrl_config()
    norm_stats = _norm_stats(config.network.reference_action_len)
    torch.manual_seed(3)
    algorithm = DSRLAlgorithm(AlgorithmSpec(norm_stats=norm_stats_lists(norm_stats)), config)
    old = algorithm.policy.actor.eval()
    new = DSRLAlgorithm.build_actor(algorithm.spec, config)
    new.load_state_dict(algorithm.actor_state())
    new.eval()
    generator = torch.Generator().manual_seed(5)
    token = torch.randn(2048, generator=generator)
    proprioception = torch.randn(6, generator=generator)
    bound = config.dsrl_noise_bound
    rows = config.network.predicted_action_len

    rng = np.random.default_rng(9)
    if base:
        expected = legacy.random_noise((rows, 6), exploration_std=1.0, noise_bound=bound, rng=rng)
    else:
        expected = legacy.choose_noise(
            old, token.numpy(), proprioception.numpy(),
            exploration_std=scale, noise_bound=bound, rng=rng,
        )
    new.rng = np.random.default_rng(9)
    obs = ActorInput(
        proprioception=proprioception, rl_token=token, exploration_scale=scale, base=base
    )
    noise = new.noise(obs, explore=scale > 0)
    assert np.array_equal(noise.numpy(), expected)

    base_chunk = torch.randn(rows, 6, generator=generator)
    output = new.act(obs, base_chunk, explore=scale > 0)
    assert torch.equal(output.chunk, base_chunk[: config.actions_to_execute])
    assert np.array_equal(output.extras["noise"].numpy(), expected)
