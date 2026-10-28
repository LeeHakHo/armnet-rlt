from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict

import numpy as np
import pytest
import torch

from armnet_rlt.artifacts import NormStats
from armnet_rlt.config import RLTConfig, RLTNetworkConfig, so101_network_config
from armnet_rlt.frozen import (
    config_from_frozen_checkpoint,
    freeze_learner_checkpoint,
    load_frozen_checkpoint,
)
from armnet_rlt.networks import RLTActor
from armnet_rlt.policy import RLTPolicy
from armnet_rlt.so101_actor import _refine_chunk


def _norm_stats() -> dict[str, NormStats]:
    return {
        "state": NormStats(np.zeros(6, np.float32), np.ones(6, np.float32)),
        "actions": NormStats(np.zeros(6, np.float32), np.ones(6, np.float32)),
    }


def _policy(head: str = "gmm", num_modes: int = 3) -> RLTPolicy:
    network = so101_network_config(
        hidden_dims=(16,),
        reference_action_len=3,
        predicted_action_len=3,
        fixed_action_std=0.0,
        ref_action_dropout=0.0,
        head=head,
        num_modes=num_modes,
    )
    config = RLTConfig(
        network=network,
        actions_to_execute=3,
        use_subsampling=False,
        num_critics=2,
        utd_ratio=1,
    )
    policy = RLTPolicy(config)
    policy.set_norm_stats(_norm_stats())
    return policy


def test_network_config_validates_the_head() -> None:
    with pytest.raises(ValueError, match="head"):
        so101_network_config(head="mixture")
    with pytest.raises(ValueError, match="num_modes"):
        so101_network_config(head="gmm", num_modes=0)
    with pytest.raises(ValueError, match="gmm_min_log_std"):
        so101_network_config(head="gmm", gmm_min_log_std=0.0, gmm_max_log_std=0.0)
    assert RLTNetworkConfig().head == "mean"


def test_gmm_actor_distribution_and_modes() -> None:
    policy = _policy(num_modes=3)
    token = torch.randn(2, 2048)
    state = torch.randn(2, 6)
    reference = torch.randn(2, 18)

    means, log_std, logits = policy.actor.distribution(token, state, reference)
    assert means.shape == (2, 3, 18)
    assert log_std.shape == (2, 3, 18)
    assert logits.shape == (2, 3)
    assert means.abs().max() <= 1.0
    assert torch.allclose(log_std.exp().mean(), torch.tensor(0.1), atol=0.02)

    mean, std, chosen = policy.actor.pick_mode(token, state, reference)
    weights = torch.softmax(logits, dim=-1)
    assert torch.equal(chosen, weights.argmax(dim=-1))
    rows = torch.arange(2)
    assert torch.allclose(mean, means[rows, chosen])
    assert torch.allclose(std, log_std[rows, chosen].exp())

    selected = policy.select_action(token, state, reference, deterministic=True)
    assert torch.allclose(selected, policy.actor.unnormalize_action(mean, state))
    noisy = policy.select_action(token, state, reference, deterministic=False)
    assert noisy.shape == (2, 18)


def test_mean_head_distribution_is_one_mode() -> None:
    policy = _policy(head="mean")
    token = torch.randn(2, 2048)
    state = torch.randn(2, 6)
    reference = torch.randn(2, 18)
    means, log_std, logits = policy.actor.distribution(token, state, reference)
    assert means.shape == (2, 1, 18) and logits.shape == (2, 1)
    _, mean = policy.actor(token, state, reference, sample=False)
    assert torch.allclose(means[:, 0], mean)
    assert policy.actor.num_modes == 1


def test_gmm_bc_loss_is_a_likelihood_and_trains() -> None:
    policy = _policy(num_modes=2)
    token = torch.randn(4, 2048)
    state = torch.randn(4, 6)
    reference = torch.randn(4, 18)
    action = torch.randn(4, 18).clamp(-0.5, 0.5)

    loss = policy.compute_loss_bc(token, state, reference, action)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert policy.actor.gmm_head.linear.weight.grad is not None
    assert policy.actor.gmm_head.linear.weight.grad.abs().sum() > 0
    stats = policy.mixture_stats(token, state, reference)
    assert set(stats) == {"mode_entropy", "std_mean", "top_weight"}
    assert 0.0 <= stats["mode_entropy"] <= np.log(2) + 1e-6

    # Fitting one chunk pulls the likelihood up.
    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=1e-2)
    before = float(policy.compute_loss_bc(token, state, reference, action).detach())
    for _ in range(30):
        optimizer.zero_grad()
        policy.compute_loss_bc(token, state, reference, action).backward()
        optimizer.step()
    after = float(policy.compute_loss_bc(token, state, reference, action).detach())
    assert after < before


def test_mean_checkpoint_seeds_every_mode_of_a_gmm_policy() -> None:
    source = _policy(head="mean")
    with torch.no_grad():
        source.actor.mean_head.weight.normal_()
        source.actor.mean_head.bias.normal_()
    target = _policy(num_modes=3)
    converted = target.load_policy_state(source.state_dict())
    assert converted is True

    token = torch.randn(2, 2048)
    state = torch.randn(2, 6)
    reference = torch.randn(2, 18)
    _, source_mean = source.actor(token, state, reference, sample=False)
    means, log_std, logits = target.actor.distribution(token, state, reference)
    for mode in range(3):
        assert torch.allclose(means[:, mode], source_mean, atol=0.15)
    assert torch.allclose(log_std.exp(), torch.full_like(log_std, 0.1), atol=1e-4)
    assert torch.allclose(logits, torch.zeros_like(logits), atol=1e-6)
    # Trunk and critics are copied as they are.
    assert torch.equal(
        target.actor.trunk[1].weight, source.actor.trunk[1].weight
    )
    assert torch.equal(
        target.critic_ensemble.critics[0].net[1].weight,
        source.critic_ensemble.critics[0].net[1].weight,
    )
    # A matching checkpoint loads strictly.
    assert target.load_policy_state(target.state_dict()) is False


def test_frozen_gmm_checkpoint_rebuilds_the_mixture_actor(tmp_path) -> None:
    policy = _policy(num_modes=4)
    artifact = freeze_learner_checkpoint(
        {
            "step": 7,
            "policy_state_dict": policy.state_dict(),
            "config": asdict(policy.config),
        }
    )
    path = tmp_path / "frozen_actor.pt"
    torch.save(artifact, path)
    config = config_from_frozen_checkpoint(load_frozen_checkpoint(path))
    assert config.network.head == "gmm"
    assert config.network.num_modes == 4
    actor = RLTActor(config.network, delta_actions=config.policy_uses_delta_actions)
    actor.load_state_dict(artifact["actor_state_dict"])
    assert actor.gmm_head.num_modes == 4


def test_refine_chunk_executes_the_likeliest_mode_without_exploration() -> None:
    policy = _policy(num_modes=3)
    actor = policy.actor.eval()
    token = torch.randn(2048)
    state = torch.randn(6)
    reference = torch.randn(3, 6)

    chunk = _refine_chunk(
        actor,
        token,
        state,
        reference,
        predicted_len=3,
        action_dim=6,
        exploration_correlation=0.85,
        exploration_scale=0.0,
    )
    mean, _, _ = actor.pick_mode(
        token.unsqueeze(0), state.unsqueeze(0), reference.flatten().unsqueeze(0)
    )
    expected = actor.unnormalize_action(mean, state.unsqueeze(0)).reshape(3, 6)
    assert chunk.shape == (3, 6)
    assert torch.allclose(chunk, expected)

    noisy = _refine_chunk(
        actor,
        token,
        state,
        reference,
        predicted_len=3,
        action_dim=6,
        exploration_correlation=0.85,
        exploration_scale=1.0,
    )
    assert noisy.shape == (3, 6)


def test_actor_submit_and_job_carry_the_head() -> None:
    from armnet_rlt.submit_actor import _parser, build_job_args

    entry = {
        "host": "example.modal.run",
        "port": 443,
        "use_tls": True,
        "config_name": "pi05_rlt_so101_busybox",
        "run_id": "run-1",
    }
    base = [
        "--task",
        "push_green_button",
        "--config-name",
        "pi05_rlt_so101_busybox",
        "--hf-checkpoint-repo",
        "org/checkpoint",
        "--language-instruction",
        "Press the green button",
        "--learner-key",
        "pi05_rlt_so101_busybox:run-1",
    ]
    parsed = _parser().parse_args([*base, "--head", "gmm", "--num-modes", "4"])
    assert parsed.head == "gmm" and parsed.num_modes == 4
    job_args = build_job_args(parsed, entry)
    assert job_args["head"] == "gmm" and job_args["num_modes"] == 4
    assert build_job_args(_parser().parse_args(base), entry)["head"] == "mean"


def test_local_app_builds_a_gmm_learner_config(tmp_path, monkeypatch) -> None:
    from armnet_rlt import local_app

    monkeypatch.setattr(local_app, "local_data_root", lambda: tmp_path)
    args = Namespace(
        embodiment="so101",
        config_name="pi05_rlt_busybox_multitask_singlearm_minmax",
        run_id="run-1",
        head="gmm",
        num_modes=3,
        gmm_init_std=0.2,
    )
    config = local_app._config(args)
    assert config.network.head == "gmm"
    assert config.network.num_modes == 3
    assert config.network.gmm_init_std == 0.2
    plain = local_app._config(Namespace(**{**vars(args), "head": "mean"}))
    assert plain.network.head == "mean"
    parsed = local_app._parser().parse_args(
        ["streaming-bc", "--head", "gmm", "--num-modes", "3", "--cache-name", "x.pt"]
    )
    assert parsed.head == "gmm" and parsed.cache_name == "x.pt"
    parsed = local_app._parser().parse_args(
        ["bc", "--head", "gmm", "--init-run-id", "r", "--init-step", "5000"]
    )
    assert parsed.init_run_id == "r" and parsed.init_step == 5000
