import numpy as np
import torch

from armnet_rlt import robometer_reward
from armnet_rlt.episode_frames import pack_episode_frames, spread, unpack_episode_frames


def test_episode_frames_round_trip_in_order() -> None:
    frames = [{"top": np.array([i, 1], dtype=np.uint8), "wrist": np.array([i], dtype=np.uint8)} for i in range(12)]
    views = unpack_episode_frames(torch.from_numpy(pack_episode_frames(frames)))
    assert sorted(views) == ["top", "wrist"]
    assert [v[0] for v in views["top"]] == list(range(12))
    assert views["wrist"][3] == bytes([3])


def test_boundary_progress_spans_the_curve() -> None:
    points = robometer_reward.boundary_progress([0.0, 0.5, 1.0], 4)
    assert np.allclose(points, [0.0, 0.25, 0.5, 0.75, 1.0])


def test_shape_adds_the_progress_gain_of_each_transition(monkeypatch) -> None:
    monkeypatch.setenv("ROBOMETER_WEIGHT", "2.0")
    monkeypatch.setattr(robometer_reward, "progress_curve", lambda frames: ([0.0, 0.2, 0.8], {}))
    episode = [{"reward": torch.tensor(-1.0)} for _ in range(2)]
    frames = {view: [b"x", b"y", b"z"] for view in robometer_reward.VIEWS}
    robometer_reward.shape(episode, frames)
    assert np.allclose([float(f["reward"]) for f in episode], [-0.6, 0.2])


def test_shape_leaves_rewards_without_frames_or_on_failure(monkeypatch) -> None:
    episode = [{"reward": torch.tensor(0.5)}]
    robometer_reward.shape(episode, None)

    def reject(frames):
        raise ValueError("400 Bad Request")

    monkeypatch.setattr(robometer_reward, "progress_curve", reject)
    robometer_reward.shape(episode, {view: [b"a", b"b"] for view in robometer_reward.VIEWS})
    assert float(episode[0]["reward"]) == 0.5

    def down(frames):
        raise ConnectionError("down")

    monkeypatch.setenv("ROBOMETER_WAIT_S", "0")
    monkeypatch.setattr(robometer_reward, "progress_curve", down)
    robometer_reward.shape(episode, {view: [b"a", b"b"] for view in robometer_reward.VIEWS})
    assert float(episode[0]["reward"]) == 0.5


def test_spread_keeps_the_ends_and_caps_the_count() -> None:
    assert spread(list(range(5)), 16) == [0, 1, 2, 3, 4]
    kept = spread(list(range(60)), 16)
    assert len(kept) == 16 and kept[0] == 0 and kept[-1] == 59
    assert kept == sorted(kept)


def test_shape_waits_for_an_unreachable_server(monkeypatch) -> None:
    calls = []

    def flaky(frames):
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("down")
        return [0.0, 1.0], {}

    monkeypatch.setattr(robometer_reward, "progress_curve", flaky)
    monkeypatch.setattr(robometer_reward.time, "sleep", lambda s: None)
    episode = [{"reward": torch.tensor(0.0)}]
    robometer_reward.shape(episode, {view: [b"a", b"b"] for view in robometer_reward.VIEWS})
    assert len(calls) == 3 and float(episode[0]["reward"]) == 1.0
