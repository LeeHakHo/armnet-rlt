from __future__ import annotations

import pytest

from armnet_rlt.state_machine import (
    EpisodeEvent,
    EpisodeOutcome,
    EpisodeState,
    RLTStateMachine,
)


def test_binary_episode_lifecycle_and_duration() -> None:
    now = [10.0]
    machine = RLTStateMachine(clock=lambda: now[0])
    assert machine.state is EpisodeState.END_EPISODE

    machine.handle_event(EpisodeEvent.START)
    now[0] = 12.5
    assert machine.episode_duration == 2.5
    machine.handle_event(EpisodeEvent.SUCCESS)

    assert machine.state is EpisodeState.END_EPISODE
    assert machine.episode_ctx.outcome is EpisodeOutcome.SUCCESS
    assert machine.episode_reward == 1.0
    assert machine.episode_count == 1


def test_invalid_transition_is_rejected() -> None:
    machine = RLTStateMachine()
    with pytest.raises(ValueError, match="not allowed"):
        machine.handle_event(EpisodeEvent.FAIL)


@pytest.mark.parametrize("event", [EpisodeEvent.FAIL, EpisodeEvent.TIMEOUT])
def test_non_success_has_zero_reward(event: EpisodeEvent) -> None:
    machine = RLTStateMachine()
    machine.handle_event(EpisodeEvent.START)
    machine.handle_event(event)
    assert machine.episode_reward == 0.0

