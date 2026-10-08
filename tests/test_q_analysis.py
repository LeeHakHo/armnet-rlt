from __future__ import annotations

import json
import math
from dataclasses import asdict

import numpy as np
import pytest
import torch

from armnet_rlt.artifacts import NormStats
from armnet_rlt.config import RLTConfig, so101_network_config
from armnet_rlt.policy import RLTPolicy
from armnet_rlt.q_analysis import (
    analyze_transitions,
    config_from_checkpoint,
    main,
    segment_episodes,
)


def _config() -> RLTConfig:
    network = so101_network_config(
        hidden_dims=(8,),
        reference_action_len=3,
        predicted_action_len=3,
        ref_action_dropout=0.0,
    )
    return RLTConfig(
        network=network,
        actions_to_execute=3,
        use_subsampling=False,
        num_critics=2,
        utd_ratio=1,
        policy_uses_delta_actions=True,
        discount=0.9,
    )


def _policy(config: RLTConfig) -> RLTPolicy:
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


def _episode(length: int, reward: float, network) -> list[dict]:
    items = []
    for index in range(length):
        terminal = index == length - 1
        items.append(
            {
                "rl_token": torch.randn(2048) * 0.01,
                "proprioception": torch.zeros(6),
                "reference_action": torch.zeros(network.reference_action_dim),
                "action": torch.zeros(network.predicted_action_dim),
                "reward": torch.tensor(reward if terminal else 0.0),
                "next_rl_token": torch.randn(2048) * 0.01,
                "next_proprioception": torch.zeros(6),
                "next_reference_action": torch.zeros(
                    network.reference_action_dim
                ),
                "done": torch.tensor(float(terminal)),
            }
        )
    return items


def test_segment_episodes_drops_unterminated_tail() -> None:
    network = _config().network
    transitions = (
        _episode(3, 1.0, network)
        + _episode(2, 0.0, network)
        + _episode(4, 1.0, network)[:-1]
    )

    assert segment_episodes(transitions) == [[0, 1, 2], [3, 4]]


def test_analyze_reports_ideal_values_by_distance_to_end() -> None:
    config = _config()
    network = config.network
    transitions = (
        _episode(4, 1.0, network)
        + _episode(3, 0.0, network)
        + _episode(2, 0.5, network)
    )
    policy = _policy(config)

    result = analyze_transitions(policy, transitions, batch_size=4, max_k=10)

    summary = result["summary"]
    assert summary["episodes"] == 3
    assert summary["successful_episodes"] == 2
    assert summary["transitions"] == 9
    discount = 0.9**3
    success = {row["k"]: row for row in result["by_outcome"]["success"]}
    assert success[0]["n"] == 2
    assert success[0]["ideal"] == pytest.approx((1.0 + 0.5) / 2)
    assert success[1]["ideal"] == pytest.approx((discount + 0.5 * discount) / 2)
    assert success[3]["n"] == 1
    assert success[3]["ideal"] == pytest.approx(discount**3)
    failure = result["by_outcome"]["failure"]
    assert {row["k"] for row in failure} == {0, 1, 2}
    assert all(row["ideal"] == 0.0 for row in failure)
    assert math.isfinite(result["aggregate_q_stats"]["learner/pull_ratio"])


def test_analyze_pools_distances_past_max_k() -> None:
    config = _config()
    transitions = _episode(5, 1.0, config.network)

    result = analyze_transitions(
        _policy(config), transitions, batch_size=8, max_k=2
    )

    rows = result["by_outcome"]["success"]
    assert [row["k"] for row in rows] == [0, 1, 2]
    assert rows[-1]["n"] == 3


def test_analyze_rejects_data_without_a_complete_episode() -> None:
    config = _config()
    transitions = _episode(3, 1.0, config.network)[:-1]

    with pytest.raises(ValueError, match="complete episode"):
        analyze_transitions(_policy(config), transitions)


def test_config_round_trips_through_checkpoint_dict() -> None:
    config = _config()

    rebuilt = config_from_checkpoint(asdict(config))

    assert rebuilt.network == config.network
    assert rebuilt.discount == 0.9
    assert rebuilt.policy_uses_delta_actions is True
    assert rebuilt.jerk_joint_weights == config.jerk_joint_weights


def test_cli_reads_checkpoint_cache_and_replay_buffer(tmp_path, capsys) -> None:
    config = _config()
    policy = _policy(config)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "step": 7,
            "config": asdict(config),
            "policy_state_dict": policy.state_dict(),
        },
        checkpoint,
    )
    assets = tmp_path / "assets"
    assets.mkdir()
    stats = {
        key: {
            "mean": [0.0] * 6,
            "std": [1.0] * 6,
            "q01": [-1.0] * 6,
            "q99": [1.0] * 6,
        }
        for key in ("state", "actions")
    }
    (assets / "norm_stats.json").write_text(json.dumps(stats))
    (assets / "norm_stats_actions_per_timestep.json").write_text(
        json.dumps({"actions": stats["actions"]})
    )
    cache = tmp_path / "demo.pt"
    torch.save(_episode(3, 1.0, config.network), cache)
    replay = tmp_path / "online_replay_buffer.pt"
    torch.save(
        {
            "schema_version": 1,
            "learner_step": 7,
            "transitions": _episode(2, 0.0, config.network),
        },
        replay,
    )
    output = tmp_path / "q.json"

    main(
        [
            "--checkpoint", str(checkpoint),
            "--assets-dir", str(assets),
            "--demo-cache", str(cache),
            "--replay-buffer", str(replay),
            "--output", str(output),
        ]
    )

    written = json.loads(output.read_text())
    assert set(written) == {"demo_cache", "online_replay_buffer"}
    assert written["demo_cache"]["summary"]["successful_episodes"] == 1
    assert written["online_replay_buffer"]["summary"]["successful_episodes"] == 0
    assert "checkpoint step: 7" in capsys.readouterr().out
