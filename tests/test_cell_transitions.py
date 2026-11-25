"""The cell records one transition per chunk the policy server returns, with
the server's extras and, for an algorithm that needs images, JPEG views; the
learner turns the episode back into batch fields."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from armnet_rlt import so101_actor
from armnet_rlt.learner import (
    _batch_from_transitions,
    _decode_episode_images,
    _drop_partial_chunks,
    _transition_fields,
)
from armnet_rlt.operator_control import CellOperatorControl
from armnet_rlt.policy_actor import ChunkResult
from armnet_rlt.remote_policy import encode_image
from test_so101_actor_safety import _Cell, _Robot, _Transport, _is_command_safety_violation

CHUNK = 2


class _Policy:
    """A policy server whose chunk and noise extra count the requests."""

    actor_source = "learner"

    def __init__(self, *, base: bool = True, images: bool = False) -> None:
        self.base = base
        self.images = images
        self.calls = 0
        self.options: list[dict] = []

    def act(self, observation, **options):
        self.options.append(options)
        index = self.calls
        self.calls += 1
        frames = None
        if self.images:
            frames = {
                view: encode_image(np.full((8, 8, 3), 10 * index, dtype=np.uint8), "jpeg")
                for view in ("top", "front", "wrist")
            }
        return ChunkResult(
            chunk=np.full((CHUNK, 6), float(index), dtype=np.float32),
            base_chunk=(
                np.full((4, 6), 100.0 + index, dtype=np.float32) if self.base else None
            ),
            rl_token=np.full(2048, index, dtype=np.float32) if self.base else None,
            extras={"noise": np.full((3, 6), index, dtype=np.float32)},
            actor_step=5,
            frames=frames,
            needs_images=self.images,
        )

    def reset(self) -> None:
        pass


class _Cell5(_Cell):
    """Completes after five commanded steps."""

    def is_complete(self):
        return SimpleNamespace(
            complete=len(self._robot.sent) >= 5, success=True, scored_by="monitor"
        )


@pytest.fixture
def run(monkeypatch):
    runtime = types.ModuleType("armnet_runtime")
    context = types.ModuleType("armnet_runtime.context")
    context.is_command_safety_violation = _is_command_safety_violation
    lerobot = types.ModuleType("lerobot")
    transport = types.ModuleType("lerobot.transport")
    utils = types.ModuleType("lerobot.transport.utils")
    utils.transitions_to_bytes = lambda transitions: transitions
    utils_package = types.ModuleType("lerobot.utils")
    transition = types.ModuleType("lerobot.utils.transition")
    transition.Transition = lambda **fields: dict(fields)
    for name, module in {
        "armnet_runtime": runtime,
        "armnet_runtime.context": context,
        "lerobot": lerobot,
        "lerobot.transport": transport,
        "lerobot.transport.utils": utils,
        "lerobot.utils": utils_package,
        "lerobot.utils.transition": transition,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(so101_actor, "_load_parameters", lambda _actor, _payload: 0)
    monkeypatch.setattr(so101_actor, "_serialize_interaction", lambda record: record)
    monkeypatch.setattr(so101_actor.time, "sleep", lambda _seconds: None)

    def run(policy: _Policy, robot: _Robot, rollouts: int = 1):
        cell = _Cell5(robot)
        sent: list = []

        class Transport(_Transport):
            def send_episode(self, transitions, interaction) -> None:
                sent.append(transitions)
                super().send_episode(transitions, interaction)

        cfg = SimpleNamespace(
            network=SimpleNamespace(
                action_dim=6, proprioception_dim=6, rl_token_dim=2048,
                predicted_action_len=CHUNK, reference_action_len=4,
            ),
            device="cpu",
            exploration_scale=0.3,
            exploration_correlation=0.6,
            actor_learner=None,
        )
        ctx = SimpleNamespace(cell=cell, args={"record_dataset": False}, report_progress=lambda m: None)
        try:
            so101_actor.run_actor(
                ctx, cfg,
                operator=CellOperatorControl(cell, total_rollouts=rollouts),
                robot=robot, policy=policy, transport=Transport(),
            )
        finally:
            run.sent = sent
        return sent

    return run


def test_transitions_carry_the_executed_chunk_and_the_server_extras(run) -> None:
    policy = _Policy()
    sent = run(policy, _Robot([]))

    # The warmup request is deterministic; rollout requests explore.
    assert policy.options[0]["explore"] is False
    assert policy.options[1] == {
        "explore": True, "exploration_scale": 0.3, "exploration_correlation": 0.6, "base": False,
    }
    (episode,) = sent
    # Requests 1..3 during the episode: two transitions with a next state.
    assert len(episode) == 2
    first, second = episode
    assert torch.equal(first["state"]["extra/noise"][0], torch.full((3, 6), 1.0))
    assert torch.equal(second["state"]["extra/noise"][0], torch.full((3, 6), 2.0))
    assert torch.equal(first["action"][0], torch.full((CHUNK * 6,), 1.0))
    assert torch.equal(first["state"]["reference_action"][0], torch.full((24,), 101.0))
    assert torch.equal(first["next_state"]["rl_token"][0], torch.full((2048,), 2.0))
    assert not any(key.startswith("image/") for key in first["state"])

    fields = [_transition_fields(item) for item in episode]
    batch = _batch_from_transitions(fields, 4, torch.device("cpu"))
    assert batch["extras"]["noise"].shape == (4, 3, 6)
    assert batch["action"].shape == (4, CHUNK * 6)
    assert "images" not in batch


def test_an_algorithm_without_the_base_gets_images_and_empty_tokens(run) -> None:
    policy = _Policy(base=False, images=True)
    (episode,) = run(policy, _Robot([]))

    first, second = episode
    assert first["state"]["rl_token"].shape == (1, 0)
    assert first["state"]["reference_action"].shape == (1, 0)
    assert set(key for key in first["state"] if key.startswith("image/")) == {
        "image/top", "image/front", "image/wrist"
    }
    assert not any(key.startswith("next_image/") for key in first["state"])
    assert {key for key in second["state"] if key.startswith("next_image/")} == {
        "next_image/top", "next_image/front", "next_image/wrist"
    }

    fields = [_transition_fields(item) for item in episode]
    _decode_episode_images(fields)
    # The first transition's next images are the second one's images.
    assert fields[0]["next_images"] is fields[1]["images"]
    assert int(fields[0]["images"]["top"].float().mean()) == pytest.approx(10, abs=2)
    assert int(fields[1]["next_images"]["top"].float().mean()) == pytest.approx(30, abs=2)
    batch = _batch_from_transitions(fields, 3, torch.device("cpu"))
    assert batch["images"]["front"].shape == (3, 8, 8, 3)
    assert batch["images"]["front"].dtype == torch.uint8
    assert batch["next_images"]["wrist"].shape == (3, 8, 8, 3)
    assert batch["state"]["rl_token"].shape == (3, 0)


def test_a_safety_trip_sends_the_in_flight_chunk_as_partial(run) -> None:
    class RemoteCallError(Exception):
        pass

    trip = RemoteCallError("current gate tripped on elbow_flex")
    # Three steps run (chunk 1 and half of chunk 2), then the gate trips.
    robot = _Robot([])
    original = robot.send_action

    def send(action):
        if len(robot.sent) == 3:
            raise trip
        return original(action)

    robot.send_action = send
    with pytest.raises(RemoteCallError):
        run(_Policy(), robot, rollouts=2)
    (episode,) = run.sent
    assert len(episode) == 2
    complete, partial = episode
    assert "partial" not in complete["state"]
    assert float(partial["state"]["partial"][0]) == 1.0
    assert float(partial["state"]["safety_trip"][0]) == 1.0
    assert torch.equal(partial["state"]["extra/noise"][0], torch.full((3, 6), 2.0))
    # One row ran; the other holds the commanded target.
    assert torch.equal(partial["action"][0], torch.full((CHUNK * 6,), 2.0))
    assert partial["reward"] == -1.0 and partial["done"] is True

    fields = [_transition_fields(item) for item in episode]
    _drop_partial_chunks(fields)
    assert len(fields) == 1
    assert float(fields[0]["reward"]) == -1.0 and float(fields[0]["done"]) == 1.0
    assert float(fields[0]["safety_trip"]) == 1.0
