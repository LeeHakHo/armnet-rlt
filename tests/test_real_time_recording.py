from types import SimpleNamespace

import armnet_rlt.so101_actor as actor
from armnet_rlt.so101_actor import record_rollout_frame


class Writer:
    def __init__(self):
        self.frames = 0

    def submit_observation(self, obs, action, task):
        self.frames += 1


def test_each_tick_is_one_frame_with_its_real_time(monkeypatch) -> None:
    clock = {"now": 0.0}
    monkeypatch.setattr(actor.time, "perf_counter", lambda: clock["now"])
    ctx = SimpleNamespace(cell=SimpleNamespace(record_frame=lambda: None))
    writer = Writer()
    # 50 ms ticks, then a 0.5 s wait for the next action chunk. The loop's own tick time
    # is stale on an episode's first tick (it began before the scene reset).
    for i, now in enumerate([10.0, 10.05, 10.10, 10.60, 10.65]):
        clock["now"] = now
        record_rollout_frame(ctx, {}, writer, None, raw_obs={}, requested_action={}, actual_action={},
                             task="t", frame_index=i, period_s=0.05, tick=now - (15.0 if i == 0 else 0.0),
                             action_write_s=0.0)
    assert writer.frames == 5
    assert writer.frame_times == [0.0, 0.05, 0.1, 0.6, 0.65]
    clock["now"] = 100.0
    record_rollout_frame(ctx, {}, writer, None, raw_obs={}, requested_action={}, actual_action={},
                         task="t", frame_index=0, period_s=0.05, tick=80.0, action_write_s=0.0)
    assert writer.frame_times == [0.0]


def test_call_while_recording_records_until_the_call_returns() -> None:
    import time as _time

    frames = []
    result = actor.call_while_recording(lambda: (_time.sleep(0.23), "chunk")[1], 0.05, lambda: frames.append(1))
    assert result == "chunk"
    assert 3 <= len(frames) <= 5


def test_call_while_recording_raises_the_call_error() -> None:
    import pytest

    def boom():
        raise RuntimeError("server down")

    with pytest.raises(RuntimeError, match="server down"):
        actor.call_while_recording(boom, 0.05, lambda: None)
