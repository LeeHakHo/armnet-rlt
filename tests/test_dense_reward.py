from collections import deque
from queue import Queue
from threading import Event

import numpy as np
import pytest
import torch

from armnet_rlt import dense_reward

REST = [-4.3, -97.3, 91.0, 75.2, -1.4, 9.6]


def test_tip_position_batches_and_matches_single_poses() -> None:
    poses = np.array([REST, [0, 0, 0, 0, 0, 0], [10, -40, 50, 60, 5, 20]])
    batched = dense_reward.tip_position(poses)
    single = np.stack([dense_reward.tip_position(pose) for pose in poses])
    assert batched.shape == (3, 3)
    np.testing.assert_allclose(batched, single, atol=1e-12)
    # The folded rest pose keeps the tip close to the base.
    assert np.linalg.norm(batched[0]) < 0.2


def test_button_moves_with_the_rail_and_needs_a_known_cell() -> None:
    low = dense_reward.button_position(0.6, "cell-03")
    high = dense_reward.button_position(0.7, "cell-03")
    np.testing.assert_allclose(high - low, 0.1 * np.asarray(dense_reward.BUTTON[3][1]))
    assert dense_reward.cell_number("cell-01") == 1
    with pytest.raises(KeyError):
        dense_reward.button_position(0.7, "cell-02")


def test_reward_has_the_three_terms() -> None:
    gap = float(dense_reward.distance(REST, 0.7, 3))
    value = dense_reward.reward(REST, 0.7, 3, pressed=True, tripped=False, alpha=0.5)
    assert float(value) == pytest.approx(1.0 - 0.5 * gap)
    tripped = dense_reward.reward(REST, 0.7, 3, tripped=True, alpha=0.5, gamma=2.0)
    assert float(tripped) == pytest.approx(-2.0 - 0.5 * gap)


def _transition(pose: list[float]):
    from armnet_rlt.so101_actor import _ChunkTransition

    state = torch.tensor(pose, dtype=torch.float32)
    return _ChunkTransition(
        rl_token=torch.zeros(4),
        proprioception=state,
        reference_action=torch.zeros(6),
        curriculum_scale=torch.tensor(0.0),
        action=torch.zeros(6),
        next_rl_token=torch.zeros(4),
        next_proprioception=state,
        next_reference_action=torch.zeros(6),
    )


def test_scene_reaches_the_plugin_and_shapes_every_transition(monkeypatch) -> None:
    pytest.importorskip("lerobot")
    from armnet_rlt.learner import _drain_online_transitions
    from armnet_rlt.so101_actor import _episode_scene, _serialize_episode

    monkeypatch.setenv("DENSE_ALPHA", "0.5")
    poses = [REST, [10, -40, 50, 60, 5, 20], [0, 0, 0, 0, 0, 0]]
    scene = _episode_scene({"values": {"rail": 0.7}}, "cell-03")
    assert scene == (0.7, 3.0)
    payload = _serialize_episode([_transition(p) for p in poses], reward=1.0, scene=scene)
    queue: Queue = Queue()
    queue.put(payload)
    buffer: deque = deque()
    assert (
        _drain_online_transitions(
            queue, buffer, torch.device("cpu"), Event(), reward_plugin=dense_reward.shape
        )
        == 3
    )
    gaps = dense_reward.distance(np.array(poses), 0.7, 3)
    expected = -0.5 * gaps + np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose([float(t["reward"]) for t in buffer], expected, atol=1e-5)


def test_episode_without_a_scene_is_left_unchanged(capsys) -> None:
    from armnet_rlt.so101_actor import _episode_scene

    assert _episode_scene({"values": {}}, "cell-03") is None
    assert _episode_scene({"values": {"rail": 0.7}}, "") is None
    episode = [{"reward": torch.tensor(1.0), "next_proprioception": torch.tensor(REST)}]
    dense_reward.shape(episode, None, scene=None)
    assert float(episode[0]["reward"]) == 1.0
    assert "rewards unchanged" in capsys.readouterr().out
