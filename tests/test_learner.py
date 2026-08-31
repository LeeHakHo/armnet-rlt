from __future__ import annotations

import json
import math
import pickle
import sys
import types
from pathlib import Path
from queue import Queue

import torch

from armnet_rlt.config import RLTConfig, so101_network_config
from armnet_rlt.jsonl_log import read_jsonl
from armnet_rlt.learner import (
    _drain_interactions,
    _push_weights,
    _schedule_online_updates,
    run_learner,
)
from armnet_rlt.metrics import RollingMetrics


def _write_artifacts(root: Path, network) -> tuple[Path, Path]:
    assets = root / "assets"
    assets.mkdir()
    stats = {
        "state": {
            "mean": [0.0] * 6,
            "std": [1.0] * 6,
            "q01": [-1.0] * 6,
            "q99": [1.0] * 6,
        },
        "actions": {
            "mean": [0.0] * 6,
            "std": [1.0] * 6,
            "q01": [-1.0] * 6,
            "q99": [1.0] * 6,
        },
    }
    (assets / "norm_stats.json").write_text(json.dumps(stats))
    transitions = []
    for index in range(3):
        transitions.append(
            {
                "rl_token": torch.randn(2048) * 0.01,
                "proprioception": torch.zeros(6),
                "reference_action": torch.zeros(
                    network.reference_action_dim
                ),
                "action": torch.zeros(network.predicted_action_dim),
                "reward": torch.tensor(float(index == 2)),
                "next_rl_token": torch.randn(2048) * 0.01,
                "next_proprioception": torch.zeros(6),
                "next_reference_action": torch.zeros(
                    network.reference_action_dim
                ),
                "done": torch.tensor(float(index == 2)),
            }
        )
    cache = root / "demo.pt"
    torch.save(transitions, cache)
    return assets, cache


def test_offline_smoke_finishes_without_lerobot(tmp_path) -> None:
    network = so101_network_config(
        hidden_dims=(8,),
        reference_action_len=2,
        predicted_action_len=2,
        ref_action_dropout=0.0,
    )
    assets, cache = _write_artifacts(tmp_path, network)
    config = RLTConfig(
        network=network,
        assets_dir=assets,
        demo_cache_path=cache,
        output_dir=tmp_path / "runs",
        run_id="smoke",
        actions_to_execute=2,
        use_subsampling=False,
        num_critics=2,
        utd_ratio=1,
        batch_size=2,
        max_demo_pretrain_steps=1,
        online_steps=1,
        device="cpu",
        save_freq=0,
    )

    summary = run_learner(config)

    assert summary["transition_count"] == 3
    assert summary["positive_rate"] == 1 / 3
    assert summary["action_dim"] == 6
    assert summary["completed_steps"] == 1
    assert math.isfinite(summary["final_actor_loss"])
    assert math.isfinite(summary["final_critic_loss"])
    checkpoint = Path(summary["checkpoint_path"])
    assert checkpoint.is_file()
    assert checkpoint.is_relative_to(tmp_path / "runs" / "smoke")


def test_interactions_are_persisted_with_learner_step(
    monkeypatch, tmp_path: Path
) -> None:
    lerobot = types.ModuleType("lerobot")
    transport = types.ModuleType("lerobot.transport")
    utils = types.ModuleType("lerobot.transport.utils")
    utils.bytes_to_python_object = pickle.loads
    monkeypatch.setitem(sys.modules, "lerobot", lerobot)
    monkeypatch.setitem(sys.modules, "lerobot.transport", transport)
    monkeypatch.setitem(sys.modules, "lerobot.transport.utils", utils)

    queue = Queue()
    queue.put(
        pickle.dumps(
            {
                "session_id": "actor-1",
                "session_rollout_index": 1,
                "success": True,
                "duration_s": 4.5,
                "action_deviation_by_joint": [0.1] * 6,
            }
        )
    )
    path = tmp_path / "rollout_metrics.jsonl"
    count = _drain_interactions(
        queue,
        RollingMetrics(window=10),
        rollout_log=path,
        episode_count=2,
        learner_step=123,
    )

    assert count == 1
    [record] = read_jsonl(path)
    assert record["global_rollout_index"] == 3
    assert record["learner_step_at_log"] == 123
    assert record["rolling/success_rate_10"] == 1.0


def test_parameter_queue_keeps_only_latest_snapshot(monkeypatch) -> None:
    lerobot = types.ModuleType("lerobot")
    transport = types.ModuleType("lerobot.transport")
    utils = types.ModuleType("lerobot.transport.utils")
    utils.state_to_bytes = pickle.dumps
    monkeypatch.setitem(sys.modules, "lerobot", lerobot)
    monkeypatch.setitem(sys.modules, "lerobot.transport", transport)
    monkeypatch.setitem(sys.modules, "lerobot.transport.utils", utils)

    policy = types.SimpleNamespace(
        actor_state_bytes=lambda: {"weight": torch.tensor([1.0])}
    )
    queue = Queue(maxsize=1)
    _push_weights(queue, policy, learner_step=10)
    _push_weights(queue, policy, learner_step=20)

    payload = pickle.loads(queue.get_nowait())
    assert payload["learner_step"].item() == 20
    assert queue.empty()


def test_online_utd_ratio_is_not_scheduled_twice() -> None:
    assert _schedule_online_updates(3, 52) == 55
