"""Episode lifecycle driven by the Armnet cell API."""

from __future__ import annotations

import logging
from typing import Any

from armnet_rlt.state_machine import EpisodeEvent, EpisodeOutcome, EpisodeState

log = logging.getLogger(__name__)
Event = tuple[EpisodeEvent, None]


class CellOperatorControl:
    """Translate cell rollout status into actor state-machine events."""

    def __init__(
        self,
        cell: Any,
        *,
        total_rollouts: int | None = None,
        ctx: Any | None = None,
        variation_episode_offset: int = 0,
    ) -> None:
        if total_rollouts is not None and total_rollouts <= 0:
            raise ValueError("total_rollouts must be positive or None")
        if variation_episode_offset < 0:
            raise ValueError("variation_episode_offset must be nonnegative")
        self._cell = cell
        self._ctx = ctx
        self._total = total_rollouts
        self._variation_episode_offset = variation_episode_offset
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
                from armnet_runtime.variation import apply_episode_variation

                self._variation = apply_episode_variation(
                    self._ctx,
                    self._variation_episode_offset + next_index - 1,
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

