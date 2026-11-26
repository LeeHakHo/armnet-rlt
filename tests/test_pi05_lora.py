import json

import numpy as np
import pytest
import torch

from armnet_rlt import pi05_lora
from armnet_rlt.openpi_rlt import PolicyOutput


class _FakePolicy:
    def __init__(self) -> None:
        self.observations = []
        self.installed = []

    def infer(self, observation):
        self.observations.append(observation)
        chunk = np.arange(30 * 6, dtype=np.float32).reshape(30, 6)
        return PolicyOutput(action_chunk=chunk, rl_token=np.zeros(4, np.float32))

    def install(self, arrays, *, partial=False):
        self.installed.append((arrays, partial))


def _request(cameras=pi05_lora.CAMERAS, task="press it"):
    from armnet_rlt.remote_policy import encode_image

    arrays = {"state": np.ones(6, np.float32), "codec": "jpeg", "task": task}
    for camera in cameras:
        image = np.full((*pi05_lora.IMAGE_SHAPE, 3), 7, np.uint8)
        arrays[f"image/{pi05_lora.IMAGE_PREFIX}{camera}"] = encode_image(image, "jpeg", 90)
    return arrays


def test_backend_speaks_the_lerobot_protocol() -> None:
    pytest.importorskip("cv2")
    policy = _FakePolicy()
    backend = pi05_lora.Pi05LoRABackend(policy, prompt="default", served_from_learner=True)
    meta = json.loads(backend.info()["meta"])
    assert meta["policy_type"] == "lerobot"
    assert meta["chunked"] and meta["uses_task"]
    assert meta["n_action_steps"] == pi05_lora.ACTIONS_PER_CALL
    assert set(meta["shapes"]) == {f"observation.images.{c}" for c in ("front", "top", "wrist")}
    out = backend.infer(_request())
    assert out["actions"].shape == (pi05_lora.ACTIONS_PER_CALL, 6)
    assert out["policy_step"] == -1
    observation = policy.observations[0]
    assert observation.prompt == "press it"
    assert set(observation.images) == set(pi05_lora.CAMERAS)
    with pytest.raises(ValueError):
        backend.infer(_request(cameras=("top",)))


def test_backend_installs_pushed_adapters_and_reports_the_step() -> None:
    policy = _FakePolicy()
    backend = pi05_lora.Pi05LoRABackend(policy, prompt="p")
    backend.apply_weights({"a/lora_a": torch.ones(2, 3, dtype=torch.bfloat16)}, 300, True)
    arrays, partial = policy.installed[0]
    assert partial and arrays["a/lora_a"].dtype == np.float32
    assert backend.learner_step == 300
    assert json.loads(backend.info()["meta"])["learner_step"] == 300


def test_episode_sample_takes_the_action_window_and_repeats_the_last() -> None:
    pytest.importorskip("cv2")
    from armnet_rlt.lerobot_actor import encode_jpeg

    length = 5
    frame = encode_jpeg(np.zeros((*pi05_lora.IMAGE_SHAPE, 3), np.uint8))
    episode = {
        "state": np.arange(length * 6, dtype=np.float32).reshape(length, 6),
        "action": np.arange(length * 6, dtype=np.float32).reshape(length, 6) + 100,
        "images": {f"observation.images.{c}": [frame] * length for c in pi05_lora.CAMERAS},
    }
    sample = pi05_lora.episode_sample(episode, 3, 4, "task")
    np.testing.assert_array_equal(sample["observation.state"], episode["state"][3])
    np.testing.assert_array_equal(sample["actions"], episode["action"][[3, 4, 4, 4]])
    assert sample["prompt"] == "task"
    assert sample["observation.images.top"].shape == (*pi05_lora.IMAGE_SHAPE, 3)


def test_flat_and_nest_round_trip_and_lora_init() -> None:
    tree = {"a": {"b": {0: 1, "lora_a": 2}}, "c": 3}
    assert pi05_lora._nest(pi05_lora._flat(tree)) == tree
    assert pi05_lora.path_name(("a", "b", 0)) == "a/b/0"
    rng = np.random.default_rng(0)
    assert not pi05_lora._lora_init("linear_lora_b", (2, 3), rng).any()
    assert pi05_lora._lora_init("lora_a", (50, 50), rng).std() == pytest.approx(0.01, rel=0.1)


def test_lora_checkpoint_names_its_base(tmp_path) -> None:
    target = pi05_lora.save_lora(
        {"x/lora_a": np.ones((2, 2))}, tmp_path / "pretrained_model",
        base_checkpoint=tmp_path / "base", config_name="cfg",
    )
    base, lora_file, config = pi05_lora.resolve_checkpoint(target)
    assert base == tmp_path / "base" and config == "cfg"
    assert dict(np.load(lora_file))["x/lora_a"].shape == (2, 2)
    assert pi05_lora.resolve_checkpoint(tmp_path / "base") == (tmp_path / "base", None, None)
