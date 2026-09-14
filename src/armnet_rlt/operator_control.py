"""Episode lifecycle driven by the Armnet cell API."""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, replace
from typing import Any

from armnet_rlt.state_machine import EpisodeEvent, EpisodeOutcome, EpisodeState

log = logging.getLogger(__name__)
Event = tuple[EpisodeEvent, None]


@dataclass(frozen=True)
class VariationCurriculumConfig:
    enabled: bool = False
    start_scale: float = 0.25
    min_scale: float = 0.1
    max_scale: float = 1.0
    step_up: float = 0.05
    step_down: float = 0.1
    window: int = 20
    promote_threshold: float = 0.8
    demote_threshold: float = 0.55
    frontier_fraction: float = 0.2

    def __post_init__(self) -> None:
        values = (
            self.start_scale,
            self.min_scale,
            self.max_scale,
            self.step_up,
            self.step_down,
            self.promote_threshold,
            self.demote_threshold,
            self.frontier_fraction,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("variation curriculum values must be finite")
        if not 0.0 <= self.min_scale <= self.start_scale <= self.max_scale <= 1.0:
            raise ValueError(
                "variation curriculum scales must satisfy "
                "0 <= min <= start <= max <= 1"
            )
        if self.step_up <= 0 or self.step_down <= 0:
            raise ValueError("variation curriculum steps must be positive")
        if self.window <= 0:
            raise ValueError("variation curriculum window must be positive")
        if not 0.0 <= self.demote_threshold < self.promote_threshold <= 1.0:
            raise ValueError(
                "variation curriculum thresholds must satisfy "
                "0 <= demote < promote <= 1"
            )
        if not 0.0 <= self.frontier_fraction <= 1.0:
            raise ValueError("frontier_fraction must be in [0, 1]")


class VariationCurriculum:
    def __init__(self, config: VariationCurriculumConfig) -> None:
        self.config = config
        self.scale = config.start_scale
        self._results: deque[bool] = deque(maxlen=config.window)

    def observe(self, success: bool) -> tuple[str, float, float] | None:
        if not self.config.enabled:
            return None
        self._results.append(success)
        if len(self._results) < self.config.window:
            return None
        rate = sum(self._results) / len(self._results)
        previous = self.scale
        action = ""
        if rate >= self.config.promote_threshold and self.scale < self.config.max_scale:
            self.scale = min(
                self.config.max_scale,
                self.scale + self.config.step_up,
            )
            action = "promoted"
        elif rate <= self.config.demote_threshold and self.scale > self.config.min_scale:
            self.scale = max(
                self.config.min_scale,
                self.scale - self.config.step_down,
            )
            action = "demoted"
        if not action:
            return None
        self._results.clear()
        return action, previous, rate

    def frontier_axis(
        self, axis_names: list[str], *, episode: int
    ) -> str | None:
        fraction = self.config.frontier_fraction
        if not self.config.enabled or not axis_names or fraction <= 0:
            return None
        interval = max(1, round(1.0 / fraction))
        if (episode + 1) % interval:
            return None
        slot = episode // interval
        return sorted(axis_names)[slot % len(axis_names)]

    def metadata(self, *, frontier_axis: str | None) -> dict[str, Any]:
        count = len(self._results)
        return {
            "enabled": self.config.enabled,
            "scale": self.scale,
            "window_count": count,
            "window_success_rate": (
                sum(self._results) / count if count else None
            ),
            "frontier_axis": frontier_axis,
        }


def _variation_axes(cell: Any) -> dict[str, Any]:
    device_config = cell.device_config
    axes: dict[str, Any] = {}
    for rail in device_config.rails:
        name = f"rail.{rail.arm}" if rail.arm else "rail"
        axes[name] = rail.position
    for camera, mount in device_config.camera_mounts.items():
        axes[f"camera.{camera}.pan"] = mount.pan
        axes[f"camera.{camera}.tilt"] = mount.tilt
    if device_config.lightbox_brightness is not None:
        axes["brightness"] = device_config.lightbox_brightness
    return axes


def _variation_setting_enabled(name: str, args: dict[str, Any]) -> bool:
    default = bool(args.get("variation", False))
    if name.startswith("rail"):
        return bool(args.get("variation_rail", default))
    if name.startswith("camera."):
        return bool(args.get("variation_cameras", default))
    if name == "brightness":
        return bool(args.get("variation_lighting", default))
    return default


def _apply_curriculum_variation(
    ctx: Any,
    episode: int,
    curriculum: VariationCurriculum,
) -> dict[str, Any] | None:
    from armnet_core import VARIATION_CLIP_SIGMA
    from armnet_runtime.variation import (
        sample_scenario,
        variation_request,
    )

    enabled, seed = variation_request(ctx.args)
    if not enabled:
        return None
    scenario = sample_scenario(
        ctx.cell,
        seed=seed,
        episode=episode,
    )
    scenario = replace(
        scenario,
        settings=tuple(
            setting
            for setting in scenario.settings
            if _variation_setting_enabled(setting.name, ctx.args)
        ),
    )
    if not scenario.settings:
        return None
    axes = _variation_axes(ctx.cell)
    frontier_axis = curriculum.frontier_axis(
        [setting.name for setting in scenario.settings],
        episode=episode,
    )
    scaled_settings = []
    for setting in scenario.settings:
        axis = axes[setting.name]
        raw = setting.default + curriculum.scale * (
            setting.value - setting.default
        )
        clamped = bool(setting.clamped and curriculum.scale >= 1.0)
        if setting.name == frontier_axis:
            direction = 1.0 if setting.value >= setting.default else -1.0
            raw = (
                setting.default
                + direction
                * curriculum.scale
                * VARIATION_CLIP_SIGMA
                * axis.std
            )
            value = max(axis.minimum, min(axis.maximum, raw))
            clamped = value != raw
        else:
            value = raw
        scaled_settings.append(
            replace(setting, value=value, clamped=clamped)
        )
    scaled = replace(scenario, settings=tuple(scaled_settings))
    ctx.report_progress(
        "varying the scene "
        f"(curriculum scale={curriculum.scale:.2f}; {scaled.describe()})"
    )
    unapplied = scaled.apply(ctx.cell, notify=ctx.report_progress)
    record = scaled.as_dict()
    if unapplied:
        record["not_applied"] = list(unapplied)
    record["curriculum"] = curriculum.metadata(
        frontier_axis=frontier_axis
    )
    return record


class CellOperatorControl:
    """Translate cell rollout status into actor state-machine events."""

    def __init__(
        self,
        cell: Any,
        *,
        total_rollouts: int | None = None,
        ctx: Any | None = None,
        variation_episode_offset: int = 0,
        variation_repeat: int = 1,
        variation_curriculum: VariationCurriculumConfig | None = None,
    ) -> None:
        if total_rollouts is not None and total_rollouts <= 0:
            raise ValueError("total_rollouts must be positive or None")
        if variation_episode_offset < 0:
            raise ValueError("variation_episode_offset must be nonnegative")
        if variation_repeat <= 0:
            raise ValueError("variation_repeat must be positive")
        self._cell = cell
        self._ctx = ctx
        self._total = total_rollouts
        self._variation_episode_offset = variation_episode_offset
        self._variation_repeat = variation_repeat
        self._curriculum = VariationCurriculum(
            variation_curriculum or VariationCurriculumConfig()
        )
        self._index = 0
        self._rollout_active = False
        self._started = False
        self._bad_reset = False
        self._reset_problems: tuple[str, ...] = ()
        self._variation: dict[str, Any] | None = None
        self._scored_by: str | None = None

    @property
    def completed_rollouts(self) -> int:
        return self._index - int(self._rollout_active)

    @property
    def rollout_index(self) -> int:
        return self._index

    @property
    def rollout_total(self) -> int | None:
        return self._total

    @property
    def variation(self) -> dict[str, Any] | None:
        return self._variation

    @property
    def reset_problems(self) -> tuple[str, ...]:
        return self._reset_problems

    @property
    def scored_by(self) -> str | None:
        return self._scored_by

    def start(self) -> None:
        self._started = True

    def poll_events(self, state: EpisodeState) -> list[Event]:
        if not self._started:
            raise RuntimeError("CellOperatorControl.start() must be called first")

        if state is EpisodeState.END_EPISODE:
            if self._rollout_active or (
                self._total is not None and self._index >= self._total
            ):
                return []
            next_index = self._index + 1
            self._cell.reset()
            self._variation = None
            if self._ctx is not None:
                episode = self._variation_episode_offset + (
                    (next_index - 1) // self._variation_repeat
                )
                if self._curriculum.config.enabled:
                    self._variation = _apply_curriculum_variation(
                        self._ctx,
                        episode,
                        self._curriculum,
                    )
                else:
                    from armnet_runtime.variation import apply_episode_variation

                    self._variation = apply_episode_variation(
                        self._ctx,
                        episode,
                    )
            problems = self._cell.rollout_begin(
                index=next_index,
                total=self._total,
                outcome_controls=True,
            )
            self._index = next_index
            self._reset_problems = tuple(str(problem) for problem in (problems or ()))
            self._bad_reset = bool(self._reset_problems)
            self._scored_by = None
            self._rollout_active = True
            return [(EpisodeEvent.START, None)]

        status = self._cell.is_complete()
        complete = bool(getattr(status, "complete", status))
        if not complete:
            return []
        self._scored_by = getattr(status, "scored_by", None)
        if self._bad_reset and getattr(status, "scored_by", None) not in (
            None,
            "operator",
        ):
            # An automated goal monitor must not reward a rollout that began
            # with the goal already satisfied. An operator can still override.
            return []
        success = bool(getattr(status, "success", False))
        return [
            (EpisodeEvent.SUCCESS if success else EpisodeEvent.FAIL, None)
        ]

    def on_episode_end(self, outcome: EpisodeOutcome | None = None) -> None:
        if not self._rollout_active:
            return
        success = outcome is EpisodeOutcome.SUCCESS
        try:
            self._cell.rollout_end(success=success)
        finally:
            self._rollout_active = False
        update = self._curriculum.observe(success)
        if update is not None and self._ctx is not None:
            action, previous, rate = update
            self._ctx.report_progress(
                f"variation curriculum {action}: "
                f"{previous:.2f} -> {self._curriculum.scale:.2f} "
                f"after {rate:.0%} success"
            )

    def should_stop(self) -> bool:
        if (
            self._total is not None
            and self._index >= self._total
            and not self._rollout_active
        ):
            return True
        for name in ("should_stop", "is_shutting_down"):
            try:
                if bool(getattr(self._cell, name)()):
                    return True
            except Exception:  # status polling is best effort
                log.warning("cell.%s failed", name, exc_info=True)
        return False

    def stop(self) -> None:
        if self._rollout_active:
            try:
                self._cell.rollout_end(aborted=True)
            except Exception:
                log.warning("cell.rollout_end(aborted=True) failed", exc_info=True)
            self._rollout_active = False
        self._started = False

