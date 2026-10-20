from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from armnet_rlt import act_learner
from armnet_rlt.act_actor import resolve_act_policy_dir


def _episode(length: int) -> dict:
    return {
        "state": np.arange(length * 6, dtype=np.float32).reshape(length, 6),
        "action": np.arange(length * 6, dtype=np.float32).reshape(length, 6) + 0.5,
        "images": {"observation.images.front": [bytes([i]) for i in range(length)]},
    }


def test_online_batch_pads_actions_past_the_episode_end(monkeypatch) -> None:
    monkeypatch.setattr(
        act_learner,
        "decode_jpeg",
        lambda data: np.full((4, 5, 3), data[0], dtype=np.uint8),
    )
    episode = _episode(3)
    with ThreadPoolExecutor(max_workers=2) as pool:
        batch = act_learner._online_batch(
            [episode],
            16,
            chunk_size=5,
            image_keys=["observation.images.front"],
            pool=pool,
        )
    assert batch["action"].shape == (16, 5, 6)
    assert batch["observation.images.front"].shape == (16, 3, 4, 5)
    for row in range(16):
        index = int(batch["observation.state"][row, 0]) // 6
        expected_pad = torch.arange(5) + index >= 3
        assert torch.equal(batch["action_is_pad"][row], expected_pad)
        assert torch.equal(
            batch["action"][row, 0], torch.from_numpy(episode["action"][index])
        )
        # Padded rows repeat the last action.
        assert torch.equal(
            batch["action"][row, -1], torch.from_numpy(episode["action"][2])
        )
        assert batch["observation.images.front"][row].max() == pytest.approx(index / 255)


def test_resolve_act_policy_dir_accepts_a_step_directory(tmp_path) -> None:
    model = tmp_path / "000020" / "pretrained_model"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    assert resolve_act_policy_dir(tmp_path / "000020") == model
    assert resolve_act_policy_dir(model) == model
    with pytest.raises(FileNotFoundError):
        resolve_act_policy_dir(tmp_path)
