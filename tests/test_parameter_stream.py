from queue import Queue
from threading import Event

import pytest

pytest.importorskip("lerobot")

from armnet_rlt.learner import _active_parameter_stream


class Context:
    def __init__(self, active: list[bool]) -> None:
        self._active = active

    def is_active(self) -> bool:
        return self._active.pop(0) if len(self._active) > 1 else self._active[0]


def stream(queue: Queue, context: Context, shutdown: Event):
    return _active_parameter_stream(
        queue, context, shutdown, seconds_between_pushes=0.0, queue_get_timeout=0.01
    )


def test_ended_stream_leaves_the_payload_for_the_next_actor():
    queue: Queue = Queue(maxsize=1)
    queue.put(b"weights")
    assert list(stream(queue, Context([False]), Event())) == []
    assert queue.get_nowait() == b"weights"


def test_stream_ending_during_the_queue_wait_returns_the_payload():
    queue: Queue = Queue(maxsize=1)
    queue.put(b"weights")
    assert list(stream(queue, Context([True, False]), Event())) == []
    assert queue.get_nowait() == b"weights"


def test_active_stream_sends_the_payload():
    queue: Queue = Queue(maxsize=1)
    queue.put(b"weights")
    messages = list(stream(queue, Context([True, True, False]), Event()))
    assert b"".join(message.data for message in messages) == b"weights"
    assert queue.empty()
