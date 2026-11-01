from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import armnet_rlt.lerobot_learner as lerobot_learner
from armnet_rlt.lerobot_learner import frame_layout, online_batch


def _episode(length: int) -> dict:
    # Frames are stored as bytes; the test decodes them without cv2.
    return {
        "state": np.arange(length, dtype=np.float32)[:, None].repeat(6, axis=1),
        "action": (100 + np.arange(length, dtype=np.float32))[:, None].repeat(6, axis=1),
        "images": {"observation.images.front": [bytes([i]) for i in range(length)]},
    }


def _decode(data: bytes) -> np.ndarray:
    return np.full((8, 8, 3), data[0], dtype=np.uint8)


def test_frame_layout_reads_the_policy_config() -> None:
    act_like = type("C", (), {"chunk_size": 4, "observation_delta_indices": None, "action_delta_indices": [0, 1, 2, 3]})()
    assert frame_layout(act_like) == (None, [0, 1, 2, 3])
    dit_like = type("C", (), {"observation_delta_indices": [-1, 0], "action_delta_indices": [-1, 0, 1, 2]})()
    assert frame_layout(dit_like) == ([-1, 0], [-1, 0, 1, 2])
    bare = type("C", (), {"chunk_size": 3})()
    assert frame_layout(bare) == (None, [0, 1, 2])


def test_online_batch_stacks_observation_history_and_pads_the_action_window(monkeypatch) -> None:
    import random

    random.seed(0)
    episodes = [_episode(5)]
    monkeypatch.setattr(lerobot_learner, "decode_jpeg", _decode)
    monkeypatch.setattr(random, "choices", lambda population, weights, k: population * k)
    monkeypatch.setattr(random, "randrange", lambda n: 0)
    with ThreadPoolExecutor(max_workers=2) as pool:
        batch = online_batch(
            episodes,
            2,
            observation_offsets=[-1, 0],
            action_offsets=[-1, 0, 1, 2],
            image_keys=["observation.images.front"],
            pool=pool,
            task="push the green button",
        )
    assert batch["observation.state"].shape == (2, 2, 6)
    assert batch["observation.images.front"].shape == (2, 2, 3, 8, 8)
    assert batch["observation.images.front"][0, :, 0, 0, 0].tolist() == [0.0, 0.0]
    assert batch["action"].shape == (2, 4, 6)
    # Frame 0: the history row before the episode and the action before it are clamped and flagged.
    assert batch["observation.state"][0, 0, 0] == 0 and batch["observation.state"][0, 1, 0] == 0
    assert batch["action_is_pad"][0].tolist() == [True, False, False, False]
    assert batch["action"][0, :, 0].tolist() == [100, 100, 101, 102]
    assert batch["task"] == ["push the green button"] * 2

    monkeypatch.setattr(random, "randrange", lambda n: 4)
    with ThreadPoolExecutor(max_workers=2) as pool:
        batch = online_batch(
            episodes,
            1,
            observation_offsets=None,
            action_offsets=[0, 1, 2],
            image_keys=["observation.images.front"],
            pool=pool,
        )
    assert batch["observation.state"].shape == (1, 6)
    assert batch["observation.images.front"].shape == (1, 3, 8, 8)
    assert abs(float(batch["observation.images.front"][0, 0, 0, 0]) - 4 / 255) < 1e-6
    assert batch["action_is_pad"][0].tolist() == [False, True, True]
    assert batch["action"][0, :, 0].tolist() == [104, 104, 104]
    assert "task" not in batch


def test_policy_weights_push_only_what_changed_since_the_last_push() -> None:
    from armnet_rlt.lerobot_learner import _PolicyWeights

    policy = torch.nn.Linear(2, 2)
    weights = _PolicyWeights(policy)
    assert weights.actor_state_bytes() == {}
    with torch.no_grad():
        policy.bias.add_(1.0)
    pushed = weights.actor_state_bytes()
    assert list(pushed) == ["bias"] and pushed["bias"].dtype == torch.float16
    assert weights.actor_state_bytes() == {}
    weights.rebase()
    assert list(weights.actor_state_bytes()) == ["bias"]
    assert weights.last_push_tensors == 1
