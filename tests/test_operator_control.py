from __future__ import annotations

import sys
import types
from types import SimpleNamespace

from armnet_rlt.operator_control import CellOperatorControl
from armnet_rlt.state_machine import EpisodeEvent, EpisodeOutcome, EpisodeState


class FakeCell:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.status = SimpleNamespace(complete=False, success=False)

    def rollout_begin(self, **kwargs):
        self.calls.append(("begin", kwargs))

    def is_complete(self):
        return self.status

    def rollout_end(self, **kwargs):
        self.calls.append(("end", kwargs))

    def reset(self):
        self.calls.append(("reset",))

    def should_stop(self):
        return False

    def is_shutting_down(self):
        return False


def test_cell_operator_runs_bounded_lifecycle() -> None:
    cell = FakeCell()
    control = CellOperatorControl(cell, total_rollouts=1)
    control.start()

    assert control.poll_events(EpisodeState.END_EPISODE) == [
        (EpisodeEvent.START, None)
    ]
    assert cell.calls[:2] == [
        ("reset",),
        (
            "begin",
            {"index": 1, "total": 1, "outcome_controls": True},
        ),
    ]
    cell.status = SimpleNamespace(complete=True, success=True)
    assert control.poll_events(EpisodeState.IN_EPISODE) == [
        (EpisodeEvent.SUCCESS, None)
    ]

    control.on_episode_end(EpisodeOutcome.SUCCESS)
    assert cell.calls[-1] == ("end", {"success": True})
    assert control.should_stop()
    assert control.completed_rollouts == 1


def test_cell_failure_maps_to_binary_fail() -> None:
    cell = FakeCell()
    control = CellOperatorControl(cell)
    control.start()
    control.poll_events(EpisodeState.END_EPISODE)
    cell.status = SimpleNamespace(complete=True, success=False)
    assert control.poll_events(EpisodeState.IN_EPISODE) == [
        (EpisodeEvent.FAIL, None)
    ]


def test_cell_operator_varies_after_reset_before_rollout(monkeypatch) -> None:
    cell = FakeCell()
    ctx = SimpleNamespace(cell=cell)

    def apply(ctx_arg, episode):
        assert ctx_arg is ctx
        cell.calls.append(("variation", episode))
        return {"seed": 42, "episode": episode}

    runtime = types.ModuleType("armnet_runtime")
    variation = types.ModuleType("armnet_runtime.variation")
    variation.apply_episode_variation = apply
    runtime.variation = variation
    monkeypatch.setitem(sys.modules, "armnet_runtime", runtime)
    monkeypatch.setitem(sys.modules, "armnet_runtime.variation", variation)
    control = CellOperatorControl(
        cell, total_rollouts=2, ctx=ctx, variation_episode_offset=5
    )
    control.start()

    control.poll_events(EpisodeState.END_EPISODE)
    assert cell.calls[:3] == [
        ("reset",),
        ("variation", 5),
        (
            "begin",
            {"index": 1, "total": 2, "outcome_controls": True},
        ),
    ]
    assert control.variation == {"seed": 42, "episode": 5}

    cell.status = SimpleNamespace(
        complete=True, success=False, scored_by="busybox"
    )
    control.poll_events(EpisodeState.IN_EPISODE)
    assert control.scored_by == "busybox"
    control.on_episode_end(EpisodeOutcome.FAIL)

    control.poll_events(EpisodeState.END_EPISODE)
    assert cell.calls[-3:] == [
        ("reset",),
        ("variation", 6),
        (
            "begin",
            {"index": 2, "total": 2, "outcome_controls": True},
        ),
    ]

