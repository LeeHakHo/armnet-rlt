from types import SimpleNamespace

import armnet_rlt.so101_actor as actor
from armnet_rlt.so101_actor import record_rollout_frame


class Writer:
    def __init__(self):
        self.frames = 0

    def submit_observation(self, obs, action, task):
        self.frames += 1


def test_a_stalled_tick_is_written_as_the_periods_it_covers(monkeypatch) -> None:
    clock = {"now": 0.0}
    monkeypatch.setattr(actor.time, "perf_counter", lambda: clock["now"])
    ctx = SimpleNamespace(cell=SimpleNamespace(record_frame=lambda: None))
    writer = Writer()
    period = 0.05
    # Ticks every 50 ms, then a 0.5 s wait for the next action chunk, then 50 ms again.
    ticks = [0.0, 0.05, 0.10, 0.60, 0.65]
    for i, now in enumerate(ticks):
        clock["now"] = now
        # The loop's own tick time is stale on an episode's first tick (it began before
        # the scene reset); recording must not depend on it.
        record_rollout_frame(ctx, {}, writer, None, raw_obs={}, requested_action={}, actual_action={},
                             task="t", frame_index=i, period_s=period, tick=now - (15.0 if i == 0 else 0.0), action_write_s=0.0)
    assert writer.frames == 14  # 0.65 s of episode at 20 fps
    # A new episode starts its own clock.
    clock["now"] = 100.0
    record_rollout_frame(ctx, {}, writer, None, raw_obs={}, requested_action={}, actual_action={},
                         task="t", frame_index=0, period_s=period, tick=80.0, action_write_s=0.0)
    assert writer.frames == 15
