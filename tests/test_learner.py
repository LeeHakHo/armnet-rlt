from __future__ import annotations

from collections import deque
import json
import math
import pickle
import sys
import types
from pathlib import Path
from queue import Queue

import pytest
import torch

from armnet_rlt.config import (
    RLTActorLearnerConfig,
    RLTConfig,
    so101_network_config,
)
from armnet_rlt.jsonl_log import read_jsonl
from armnet_rlt.learner import (
    _batch_from_transitions,
    _drain_interactions,
    _logged_episode_keys,
    _push_weights,
    _restore_replay_buffer,
    _save_replay_buffer,
    _schedule_online_updates,
    run_bc_training,
    run_learner,
    run_streaming_bc,
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


def _config_with_auth_token(tmp_path: Path) -> RLTConfig:
    network = so101_network_config(
        hidden_dims=(8,),
        reference_action_len=2,
        predicted_action_len=2,
        ref_action_dropout=0.0,
    )
    assets, cache = _write_artifacts(tmp_path, network)
    return RLTConfig(
        network=network,
        assets_dir=assets,
        demo_cache_path=cache,
        output_dir=tmp_path / "runs",
        run_id="token",
        actions_to_execute=2,
        use_subsampling=False,
        num_critics=2,
        utd_ratio=1,
        batch_size=2,
        max_demo_pretrain_steps=1,
        online_steps=1,
        device="cpu",
        save_freq=0,
        actor_learner=RLTActorLearnerConfig(auth_token="learner-secret"),
    )


def test_auth_token_stays_out_of_checkpoints(tmp_path) -> None:
    config = _config_with_auth_token(tmp_path)

    summary = run_learner(config, offline_only=True)

    checkpoint = Path(summary["checkpoint_path"])
    assert b"learner-secret" not in checkpoint.read_bytes()
    assert config.actor_learner.auth_token == "learner-secret"


class _StopAtWandb(Exception):
    pass


@pytest.mark.parametrize(
    "start",
    [
        lambda config: run_learner(config, wandb_project="test"),
        lambda config: run_bc_training(config, steps=1, wandb_project="test"),
        lambda config: run_streaming_bc(config, wandb_project="test"),
    ],
    ids=["online", "bc", "streaming-bc"],
)
def test_auth_token_stays_out_of_wandb(tmp_path, monkeypatch, start) -> None:
    logged_configs = []

    def init(**kwargs):
        logged_configs.append(kwargs["config"])
        raise _StopAtWandb

    monkeypatch.setitem(
        sys.modules, "wandb", types.SimpleNamespace(init=init)
    )
    config = _config_with_auth_token(tmp_path)

    # run_learner reports a failed wandb.init as a RuntimeError.
    with pytest.raises((_StopAtWandb, RuntimeError)):
        start(config)

    assert logged_configs[0]["actor_learner"]["auth_token"] == ""
    assert config.actor_learner.auth_token == "learner-secret"


def test_offline_smoke_trains_the_residual_structure(tmp_path) -> None:
    network = so101_network_config(
        hidden_dims=(8,),
        reference_action_len=2,
        predicted_action_len=2,
        residual_actor=True,
        critic_action_residual=True,
    )
    assets, cache = _write_artifacts(tmp_path, network)
    config = RLTConfig(
        network=network,
        assets_dir=assets,
        demo_cache_path=cache,
        output_dir=tmp_path / "runs",
        run_id="residual",
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

    summary = run_learner(config, offline_only=True)

    assert math.isfinite(summary["final_actor_loss"])
    assert math.isfinite(summary["final_critic_loss"])
    checkpoint = torch.load(summary["checkpoint_path"], weights_only=False)
    assert bool(checkpoint["policy_state_dict"]["actor.residual_mode"])


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


def test_dsrl_learner_needs_no_demonstration_cache(tmp_path) -> None:
    network = so101_network_config(
        hidden_dims=(8,),
        reference_action_len=2,
        predicted_action_len=2,
    )
    assets, _cache = _write_artifacts(tmp_path, network)
    config = RLTConfig(
        network=network,
        algorithm="dsrl",
        assets_dir=assets,
        output_dir=tmp_path / "runs",
        run_id="dsrl-smoke",
        dsrl_noise_steps=1,
        actions_to_execute=1,
        use_subsampling=False,
        num_critics=2,
        utd_ratio=1,
        batch_size=2,
        max_demo_pretrain_steps=0,
        online_batch_fraction=1.0,
        online_steps=1,
        device="cpu",
        save_freq=0,
    )

    summary = run_learner(config, offline_only=True)

    assert summary["transition_count"] == 0
    assert summary["predicted_action_dim"] == 12
    assert summary["completed_steps"] == 0
    checkpoint = torch.load(summary["checkpoint_path"], weights_only=False)
    assert checkpoint["config"]["algorithm"] == "dsrl"
    assert "actor.noise_bound" in checkpoint["policy_state_dict"]


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


def test_online_replay_buffer_round_trips_with_learner_step(
    tmp_path: Path,
) -> None:
    config = RLTConfig(
        output_dir=tmp_path,
        run_id="replay",
    )
    config.checkpoint_dir.mkdir(parents=True)
    replay = deque(
        [{"reward": torch.tensor(0.5), "done": torch.tensor(1.0)}],
        maxlen=10,
    )

    path = _save_replay_buffer(config, replay, step=123)
    restored = _restore_replay_buffer(
        config,
        checkpoint_step=123,
    )

    assert path.is_file()
    assert len(restored) == 1
    assert restored[0]["reward"].item() == 0.5


def test_online_batches_balance_curriculum_levels(tmp_path: Path) -> None:
    network = so101_network_config()
    _assets, cache = _write_artifacts(tmp_path, network)
    [transition, *_] = torch.load(
        cache,
        map_location="cpu",
        weights_only=False,
    )
    transitions = [
        {
            **transition,
            "curriculum_scale": torch.tensor(0.1),
        }
        for _ in range(100)
    ]
    transitions.append(
        {
            **transition,
            "curriculum_scale": torch.tensor(0.5),
        }
    )
    torch.manual_seed(42)

    batch = _batch_from_transitions(
        transitions,
        1_000,
        torch.device("cpu"),
        balance_key="curriculum_scale",
    )

    high_fraction = (
        batch["curriculum_scale"].eq(0.5).float().mean().item()
    )
    assert 0.4 < high_fraction < 0.6


def test_transition_fields_read_the_safety_trip_flag_with_a_default() -> None:
    from armnet_rlt.learner import _transition_fields

    def transition(state_extra: dict[str, torch.Tensor]) -> dict[str, object]:
        state = {
            "rl_token": torch.zeros(1, 4),
            "proprioception": torch.zeros(1, 6),
            "reference_action": torch.zeros(1, 12),
            **state_extra,
        }
        return {
            "state": state,
            "action": torch.zeros(1, 6),
            "reward": -1.0,
            "next_state": dict(state),
            "done": True,
        }

    plain = _transition_fields(transition({}))
    assert float(plain["safety_trip"]) == 0.0
    tripped = _transition_fields(transition({"safety_trip": torch.tensor([1.0])}))
    assert float(tripped["safety_trip"]) == 1.0 and tripped["safety_trip"].ndim == 0


def test_drain_interactions_ignores_an_episode_delivered_twice(
    monkeypatch, tmp_path, capsys
) -> None:
    lerobot = types.ModuleType("lerobot")
    transport = types.ModuleType("lerobot.transport")
    utils = types.ModuleType("lerobot.transport.utils")
    utils.bytes_to_python_object = pickle.loads
    monkeypatch.setitem(sys.modules, "lerobot", lerobot)
    monkeypatch.setitem(sys.modules, "lerobot.transport", transport)
    monkeypatch.setitem(sys.modules, "lerobot.transport.utils", utils)

    def record(index: int) -> bytes:
        return pickle.dumps(
            {
                "session_id": "actor-1",
                "session_rollout_index": index,
                "success": index % 2 == 0,
                "duration_s": 4.5,
                "action_deviation_by_joint": [0.1] * 6,
            }
        )

    path = tmp_path / "rollout_metrics.jsonl"
    rolling = RollingMetrics(window=10)
    seen: set[tuple[str, int]] = set()
    queue = Queue()
    for index in (1, 2):
        queue.put(record(index))
    assert (
        _drain_interactions(
            queue, rolling, rollout_log=path, episode_count=0, learner_step=1, seen=seen
        )
        == 2
    )
    # A reconnecting stream replays both episodes and brings one new one.
    for index in (1, 2, 3):
        queue.put(record(index))
    count = _drain_interactions(
        queue, rolling, rollout_log=path, episode_count=2, learner_step=2, seen=seen
    )
    assert count == 1
    assert [r["session_rollout_index"] for r in read_jsonl(path)] == [1, 2, 3]
    assert read_jsonl(path)[-1]["global_rollout_index"] == 3
    assert seen == {("actor-1", 1), ("actor-1", 2), ("actor-1", 3)}
    assert capsys.readouterr().out.count("duplicate episode") == 2
    # The keys of a log on disk seed the set when a learner resumes.
    assert _logged_episode_keys(path) == seen
    # Records without a session id are never treated as duplicates.
    queue.put(pickle.dumps({"success": True, "duration_s": 1.0, "action_deviation_by_joint": [0.0] * 6}))
    queue.put(pickle.dumps({"success": True, "duration_s": 1.0, "action_deviation_by_joint": [0.0] * 6}))
    assert (
        _drain_interactions(
            queue, rolling, rollout_log=path, episode_count=3, learner_step=3, seen=seen
        )
        == 2
    )


def test_learner_channel_options_disable_retries_and_keep_alive() -> None:
    from armnet_rlt.grpc_util import grpc_channel_options

    default = dict(grpc_channel_options(1024))
    assert default["grpc.enable_retries"] == 1 and "grpc.service_config" in default
    learner = dict(grpc_channel_options(1024, retries=False, keepalive=True))
    assert learner["grpc.enable_retries"] == 0
    assert "grpc.service_config" not in learner
    assert learner["grpc.keepalive_time_ms"] == 30_000
    assert learner["grpc.keepalive_permit_without_calls"] == 1


def _fake_transition(step: int, *, done: bool) -> dict:
    return {
        "state": {
            "rl_token": torch.full((1, 4), float(step)),
            "proprioception": torch.zeros(1, 6),
            "reference_action": torch.zeros(1, 18),
        },
        "action": torch.zeros(1, 18),
        "reward": 1.0 if done else 0.0,
        "next_state": {
            "rl_token": torch.full((1, 4), float(step) + 0.5),
            "proprioception": torch.zeros(1, 6),
            "reference_action": torch.zeros(1, 18),
        },
        "done": done,
    }


def _fake_episode_bytes(first_step: int, length: int) -> bytes:
    return pickle.dumps(
        [
            _fake_transition(first_step + i, done=i == length - 1)
            for i in range(length)
        ]
    )


def _patch_transition_codec(monkeypatch) -> None:
    lerobot = types.ModuleType("lerobot")
    transport = types.ModuleType("lerobot.transport")
    utils = types.ModuleType("lerobot.transport.utils")
    utils.bytes_to_transitions = pickle.loads
    monkeypatch.setitem(sys.modules, "lerobot", lerobot)
    monkeypatch.setitem(sys.modules, "lerobot.transport", transport)
    monkeypatch.setitem(sys.modules, "lerobot.transport.utils", utils)


def test_drain_episode_segments_ignores_a_replayed_episode(monkeypatch, capsys) -> None:
    from threading import Event

    from armnet_rlt.learner import _drain_episode_segments

    _patch_transition_codec(monkeypatch)
    queue = Queue()
    first, second, third = (
        _fake_episode_bytes(0, 3),
        _fake_episode_bytes(10, 2),
        _fake_episode_bytes(20, 4),
    )
    for payload in (first, second):
        queue.put(payload)
    pending: list = []
    completed: deque = deque()
    seen: set[bytes] = set()
    assert _drain_episode_segments(queue, pending, completed, Event(), seen=seen) == 2
    # The stream reconnects and replays both episodes before a new one.
    for payload in (first, second, third):
        queue.put(payload)
    assert _drain_episode_segments(queue, pending, completed, Event(), seen=seen) == 1
    assert [len(segment) for segment in completed] == [3, 2, 4]
    assert capsys.readouterr().out.count("duplicate transitions ignored") == 2
    # Without a digest set every message counts.
    queue.put(first)
    assert _drain_episode_segments(queue, pending, completed, Event()) == 1


def test_drain_online_transitions_ignores_a_replayed_episode(monkeypatch) -> None:
    from threading import Event

    from armnet_rlt.learner import _drain_online_transitions

    _patch_transition_codec(monkeypatch)
    queue = Queue()
    buffer: deque = deque()
    seen: set[bytes] = set()
    first = _fake_episode_bytes(0, 3)
    queue.put(first)
    assert (
        _drain_online_transitions(queue, buffer, torch.device("cpu"), Event(), seen=seen)
        == 3
    )
    # A reconnecting stream re-sends the same bytes.
    queue.put(first)
    queue.put(_fake_episode_bytes(10, 2))
    assert (
        _drain_online_transitions(queue, buffer, torch.device("cpu"), Event(), seen=seen)
        == 2
    )
    assert len(buffer) == 5


def test_rlt_learner_runs_without_a_demo_cache_when_it_uses_none(tmp_path) -> None:
    import pytest

    network = so101_network_config(
        hidden_dims=(8,),
        reference_action_len=2,
        predicted_action_len=2,
    )
    assets, _cache = _write_artifacts(tmp_path, network)
    common = dict(
        network=network,
        assets_dir=assets,
        demo_cache_path=tmp_path / "missing.pt",
        output_dir=tmp_path / "runs",
        actions_to_execute=2,
        use_subsampling=False,
        num_critics=2,
        utd_ratio=1,
        batch_size=2,
        online_steps=1,
        device="cpu",
        save_freq=0,
    )
    summary = run_learner(
        RLTConfig(
            run_id="no-cache",
            max_demo_pretrain_steps=0,
            online_batch_fraction=1.0,
            **common,
        ),
        offline_only=True,
    )
    assert summary["transition_count"] == 0 and summary["completed_steps"] == 0

    with pytest.raises(ValueError, match="no demo cache"):
        run_learner(
            RLTConfig(
                run_id="needs-cache",
                max_demo_pretrain_steps=1,
                online_batch_fraction=1.0,
                **common,
            ),
            offline_only=True,
        )


def test_learner_accepts_actor_keepalive_pings_on_an_idle_stream() -> None:
    import inspect

    from armnet_rlt import learner

    source = inspect.getsource(learner._start_online_transport)
    assert '"grpc.http2.min_ping_interval_without_data_ms", 20_000' in source
    assert '"grpc.http2.max_ping_strikes", 0' in source


def test_relabel_rlt_safety_trip_sets_the_terminal_reward_to_minus_the_penalty() -> None:
    from armnet_rlt.learner import relabel_rlt_safety_trip

    def episode(trip: float) -> list[dict[str, torch.Tensor]]:
        return [
            {"reward": torch.tensor(0.0), "safety_trip": torch.tensor(trip)},
            {"reward": torch.tensor(-1.0), "safety_trip": torch.tensor(trip)},
        ]

    tripped = episode(1.0)
    relabel_rlt_safety_trip(tripped, 2.0)
    assert [float(t["reward"]) for t in tripped] == [0.0, -2.0]
    clean = episode(0.0)
    relabel_rlt_safety_trip(clean, 2.0)
    assert [float(t["reward"]) for t in clean] == [0.0, -1.0]


def test_safety_trip_penalty_must_be_nonnegative() -> None:
    from armnet_rlt.config import RLTConfig

    with pytest.raises(ValueError, match="safety_trip_penalty"):
        RLTConfig(safety_trip_penalty=-1.0)
