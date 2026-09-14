from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from armnet_rlt.operator_control import (
    CellOperatorControl,
    VariationCurriculum,
    VariationCurriculumConfig,
    _apply_curriculum_variation,
    _variation_setting_enabled,
)
from armnet_rlt.state_machine import EpisodeEvent, EpisodeOutcome, EpisodeState


def test_curriculum_promotes_and_demotes_with_hysteresis() -> None:
    curriculum = VariationCurriculum(
        VariationCurriculumConfig(
            enabled=True,
            start_scale=0.5,
            min_scale=0.2,
            max_scale=0.8,
            step_up=0.2,
            step_down=0.1,
            window=4,
            promote_threshold=0.75,
            demote_threshold=0.25,
        )
    )

    for success in (True, True, True):
        assert curriculum.observe(success) is None
    assert curriculum.observe(False) == ("promoted", 0.5, 0.75)
    assert curriculum.scale == pytest.approx(0.7)

    for success in (False, False, False):
        assert curriculum.observe(success) is None
    assert curriculum.observe(True) == ("demoted", 0.7, 0.25)
    assert curriculum.scale == pytest.approx(0.6)


def test_curriculum_rotates_deterministic_frontier_axes() -> None:
    curriculum = VariationCurriculum(
        VariationCurriculumConfig(
            enabled=True,
            frontier_fraction=0.25,
        )
    )
    axes = ["rail", "brightness"]

    assert curriculum.frontier_axis(axes, episode=2) is None
    assert curriculum.frontier_axis(axes, episode=3) == "brightness"
    assert curriculum.frontier_axis(axes, episode=7) == "rail"


def test_variation_family_selectors_can_disable_cameras() -> None:
    args = {
        "variation": True,
        "variation_rail": True,
        "variation_cameras": False,
        "variation_lighting": True,
    }

    assert _variation_setting_enabled("rail", args)
    assert _variation_setting_enabled("brightness", args)
    assert not _variation_setting_enabled("camera.front.pan", args)


@dataclass(frozen=True)
class _Setting:
    name: str
    kind: str
    target: str | None
    value: float
    default: float
    clamped: bool = False


@dataclass(frozen=True)
class _Scenario:
    seed: int
    episode: int
    cell_id: str
    settings: tuple[_Setting, ...]

    def describe(self) -> str:
        return "test scenario"

    def apply(self, cell, *, notify=None):
        cell.applied = self
        return ()

    def as_dict(self):
        return {
            "seed": self.seed,
            "episode": self.episode,
            "cell_id": self.cell_id,
            "values": {
                setting.name: setting.value for setting in self.settings
            },
        }


def test_curriculum_scales_variation_around_calibrated_default(
    monkeypatch,
) -> None:
    armnet_core = types.ModuleType("armnet_core")
    armnet_core.VARIATION_CLIP_SIGMA = 2.5
    variation = types.ModuleType("armnet_runtime.variation")
    scenario = _Scenario(
        seed=42,
        episode=0,
        cell_id="cell-01",
        settings=(
            _Setting(
                name="rail",
                kind="rail",
                target=None,
                value=0.9,
                default=0.5,
            ),
        ),
    )
    variation.variation_request = lambda _args: (True, 42)
    variation.sample_scenario = (
        lambda _cell, *, seed, episode: scenario
    )
    monkeypatch.setitem(sys.modules, "armnet_core", armnet_core)
    monkeypatch.setitem(
        sys.modules, "armnet_runtime.variation", variation
    )

    axis = SimpleNamespace(default=0.5, std=0.2, minimum=0.0, maximum=1.0)
    cell = SimpleNamespace(
        device_config=SimpleNamespace(
            rails=[SimpleNamespace(arm=None, position=axis)],
            camera_mounts={},
            lightbox_brightness=None,
        )
    )
    ctx = SimpleNamespace(
        args={"variation": True},
        cell=cell,
        report_progress=lambda _message: None,
    )
    curriculum = VariationCurriculum(
        VariationCurriculumConfig(
            enabled=True,
            start_scale=0.25,
            frontier_fraction=0.0,
        )
    )

    record = _apply_curriculum_variation(ctx, 0, curriculum)

    assert record is not None
    assert record["values"]["rail"] == pytest.approx(0.6)
    assert record["curriculum"]["scale"] == pytest.approx(0.25)
    assert cell.applied.settings[0].value == pytest.approx(0.6)


@pytest.mark.parametrize(
    "updates",
    [
        {"start_scale": 1.1},
        {"step_up": 0.0},
        {"window": 0},
        {"promote_threshold": 0.5, "demote_threshold": 0.5},
        {"frontier_fraction": -0.1},
    ],
)
def test_curriculum_rejects_invalid_configuration(updates) -> None:
    with pytest.raises(ValueError):
        VariationCurriculumConfig(**updates)

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


def test_variation_repeat_pairs_two_policies_on_same_scene(
    monkeypatch,
) -> None:
    cell = FakeCell()
    ctx = SimpleNamespace(cell=cell)
    runtime = types.ModuleType("armnet_runtime")
    variation = types.ModuleType("armnet_runtime.variation")
    variation.apply_episode_variation = (
        lambda _ctx, episode: {"episode": episode}
    )
    runtime.variation = variation
    monkeypatch.setitem(sys.modules, "armnet_runtime", runtime)
    monkeypatch.setitem(
        sys.modules, "armnet_runtime.variation", variation
    )
    control = CellOperatorControl(
        cell,
        total_rollouts=2,
        ctx=ctx,
        variation_episode_offset=7,
        variation_repeat=2,
    )
    control.start()

    control.poll_events(EpisodeState.END_EPISODE)
    first = control.variation
    control.on_episode_end(EpisodeOutcome.FAIL)
    control.poll_events(EpisodeState.END_EPISODE)

    assert first == {"episode": 7}
    assert control.variation == {"episode": 7}

