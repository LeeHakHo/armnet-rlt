"""Small deterministic episode state machine for the production actor."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass
from typing import Callable


class EpisodeState(enum.Enum):
    IN_EPISODE = "in_episode"
    END_EPISODE = "end_episode"


class EpisodeEvent(enum.Enum):
    START = "start"
    SUCCESS = "success"
    FAIL = "fail"
    TIMEOUT = "timeout"


class EpisodeOutcome(enum.Enum):
    SUCCESS = "success"
    FAIL = "fail"
    TIMEOUT = "timeout"


_TRANSITIONS = {
    EpisodeState.END_EPISODE: {EpisodeEvent.START: EpisodeState.IN_EPISODE},
    EpisodeState.IN_EPISODE: {
        EpisodeEvent.SUCCESS: EpisodeState.END_EPISODE,
        EpisodeEvent.FAIL: EpisodeState.END_EPISODE,
        EpisodeEvent.TIMEOUT: EpisodeState.END_EPISODE,
    },
}


@dataclass
class EpisodeContext:
    start_time: float = 0.0
    outcome: EpisodeOutcome | None = None
    total_chunk_count: int = 0


class RLTStateMachine:
    """Track a binary-reward rollout and reject invalid transitions."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.state = EpisodeState.END_EPISODE
        self.episode_ctx = EpisodeContext()
        self.episode_count = 0
        self._clock = clock

    def handle_event(self, event: EpisodeEvent) -> EpisodeState:
        valid = _TRANSITIONS[self.state]
        if event not in valid:
            raise ValueError(
                f"Event {event.value!r} not allowed in state {self.state.value!r}; "
                f"valid events: {[item.value for item in valid]}"
            )
        previous = self.state
        self.state = valid[event]
        if previous is EpisodeState.END_EPISODE:
            self.episode_ctx = EpisodeContext(start_time=self._clock())
        else:
            self.episode_ctx.outcome = {
                EpisodeEvent.SUCCESS: EpisodeOutcome.SUCCESS,
                EpisodeEvent.FAIL: EpisodeOutcome.FAIL,
                EpisodeEvent.TIMEOUT: EpisodeOutcome.TIMEOUT,
            }[event]
            self.episode_count += 1
        return self.state

    @property
    def episode_reward(self) -> float:
        return float(self.episode_ctx.outcome is EpisodeOutcome.SUCCESS)

    @property
    def episode_duration(self) -> float:
        if not self.episode_ctx.start_time:
            return 0.0
        return max(0.0, self._clock() - self.episode_ctx.start_time)

