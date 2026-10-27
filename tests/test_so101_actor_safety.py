"""An unsafe command fails one rollout; it must not end the whole actor job.

A latched trip (a current gate) still ends the job, after that rollout's
episode has gone to the learner as a safety_trip failure."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from armnet_rlt import networks, so101_actor
from armnet_rlt.openpi_rlt import MOTOR_NAMES
from armnet_rlt.operator_control import CellOperatorControl

PREDICTED_LEN = 2
ACTION_DIM = 6


class _Robot:
    is_connected = True

    def __init__(self, failures: list[BaseException]) -> None:
        self._failures = failures
        self.sent: list[dict[str, float]] = []

    def get_observation(self) -> dict[str, object]:
        observation: dict[str, object] = {
            f"{name}.pos": 0.0 for name in MOTOR_NAMES
        }
        observation["front"] = np.zeros((8, 8, 3), dtype=np.uint8)
        return observation

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if self._failures:
            raise self._failures.pop(0)
        self.sent.append(action)
        return action

    def disconnect(self) -> None:
        pass


class _Cell:
    safety_limit = None
    language_instruction = "push the green button"
    cell_id = "cell-test"

    def __init__(self, robot: _Robot) -> None:
        self._robot = robot
        self.resets: list[bool] = []
        self.ended: list[dict[str, object]] = []

    def instrument(self, robot):
        return robot

    def reset(self, *, unlock: bool = False) -> None:
        self.resets.append(unlock)

    def rollout_begin(self, **_kwargs):
        return ()

    def is_complete(self):
        # The only way to succeed is to have driven the arm.
        return SimpleNamespace(
            complete=bool(self._robot.sent), success=True, scored_by="monitor"
        )

    def rollout_end(self, **kwargs) -> None:
        self.ended.append(kwargs)

    def should_stop(self) -> bool:
        return False

    def is_shutting_down(self) -> bool:
        return False

    def close(self) -> None:
        pass


class _Actor:
    action_std = torch.ones(PREDICTED_LEN * ACTION_DIM)

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def set_norm_stats(self, *_args, **_kwargs) -> None:
        pass

    def to(self, _device):
        return self

    def train(self):
        return self

    def __call__(self, *_args, **_kwargs):
        return None, torch.zeros(1, PREDICTED_LEN * ACTION_DIM)

    def unnormalize_action(self, value, _proprioception):
        return value


class _Policy:
    norm_stats = None
    use_quantile_norm = False
    uses_delta_actions = False

    def infer(self, _observation):
        return SimpleNamespace(
            rl_token=np.zeros(2048, dtype=np.float32),
            action_chunk=np.zeros((PREDICTED_LEN, ACTION_DIM), dtype=np.float32),
        )

    def reset(self) -> None:
        pass


class _Transport:
    def __init__(self) -> None:
        self.episodes: list[object] = []

    def start(self) -> None:
        pass

    def wait_for_initial_parameters(self):
        return b""

    def latest_parameters(self):
        return None

    def send_episode(self, _transitions, interaction) -> None:
        self.episodes.append(interaction)

    def close(self) -> None:
        pass


def _is_command_safety_violation(exc: BaseException) -> bool:
    return "unsafe commanded motion" in str(exc)


@pytest.fixture
def run(monkeypatch):
    runtime = types.ModuleType("armnet_runtime")
    context = types.ModuleType("armnet_runtime.context")
    context.is_command_safety_violation = _is_command_safety_violation
    monkeypatch.setitem(sys.modules, "armnet_runtime", runtime)
    monkeypatch.setitem(sys.modules, "armnet_runtime.context", context)
    monkeypatch.setattr(networks, "RLTActor", _Actor)
    monkeypatch.setattr(so101_actor, "_load_parameters", lambda _actor, _payload: 0)
    monkeypatch.setattr(so101_actor, "_serialize_episode", lambda *_a, **_k: None)
    monkeypatch.setattr(so101_actor, "_serialize_interaction", lambda record: record)
    monkeypatch.setattr(so101_actor.time, "sleep", lambda _seconds: None)

    def run(robot: _Robot):
        cell = _Cell(robot)
        progress: list[str] = []
        ctx = SimpleNamespace(
            cell=cell,
            args={"record_dataset": False},
            report_progress=progress.append,
        )
        cfg = SimpleNamespace(
            network=SimpleNamespace(
                action_dim=ACTION_DIM,
                proprioception_dim=6,
                rl_token_dim=2048,
                predicted_action_len=PREDICTED_LEN,
                reference_action_len=PREDICTED_LEN,
            ),
            device="cpu",
            exploration_scale=0.0,
            actor_learner=None,
        )
        transport = _Transport()
        run.last = (cell, transport, progress)
        result = so101_actor.run_actor(
            ctx,
            cfg,
            operator=CellOperatorControl(cell, total_rollouts=2),
            robot=robot,
            policy=_Policy(),
            transport=transport,
        )
        return result, cell, transport, progress

    return run


def test_an_unsafe_command_fails_the_rollout_and_the_next_one_runs(run) -> None:
    robot = _Robot(
        [RuntimeError("unsafe commanded motion: command speed on shoulder_lift")]
    )

    result, cell, transport, progress = run(robot)

    assert result["rollouts"] == 2
    assert result["successes"] == 1
    first, second = result["per_rollout"]
    assert (first["outcome"], first["scored_by"]) == ("safety_trip", "safety")
    assert first["shaped_reward"] == -1.0
    assert second["success"] is True
    # The edge's lock is released before the second rollout, and only then.
    assert cell.resets == [False, True]
    assert cell.ended == [{"success": False}, {"success": True}]
    assert len(transport.episodes) == 2
    assert robot.sent
    assert any("unsafe command" in message for message in progress)


def test_any_other_robot_failure_still_ends_the_job(run) -> None:
    robot = _Robot([RuntimeError("robot connector is unavailable")])

    with pytest.raises(RuntimeError, match="robot connector is unavailable"):
        run(robot)


def test_a_current_gate_trip_sends_the_episode_then_ends_the_job(run) -> None:
    class RemoteCallError(Exception):
        pass

    robot = _Robot(
        [
            RemoteCallError(
                "current gate tripped on elbow_flex: current=507 counts "
                "(3.2955 A nominal), trip limit=275.0 counts for at least 500 ms"
            )
        ]
    )

    with pytest.raises(RemoteCallError, match="current gate tripped"):
        run(robot)
    cell, transport, _progress = run.last
    assert len(transport.episodes) == 1
    assert transport.episodes[0].outcome == "safety_trip"
    assert cell.resets == [False] and cell.ended == [{"success": False}]


def test_a_lost_learner_link_ends_the_job_instead_of_blind_rollouts(run) -> None:
    class _FailingTransport(_Transport):
        failed = False

        def send_episode(self, transitions, interaction) -> None:
            super().send_episode(transitions, interaction)
            self.failed = True  # the stream died while this episode was queued

    import armnet_rlt.so101_actor as module

    original = module.run_actor

    with pytest.raises(ConnectionError, match="learner connection is gone"):
        robot = _Robot([])
        cell = _Cell(robot)
        ctx = SimpleNamespace(cell=cell, args={"record_dataset": False}, report_progress=lambda m: None)
        cfg = SimpleNamespace(
            network=SimpleNamespace(
                action_dim=ACTION_DIM, proprioception_dim=6, rl_token_dim=2048,
                predicted_action_len=PREDICTED_LEN, reference_action_len=PREDICTED_LEN,
            ),
            device="cpu", exploration_scale=0.0, actor_learner=None,
        )
        original(
            ctx, cfg,
            operator=CellOperatorControl(cell, total_rollouts=3),
            robot=robot, policy=_Policy(), transport=_FailingTransport(),
        )
    assert cell.ended == [{"success": True}]
