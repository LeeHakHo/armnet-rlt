from types import SimpleNamespace

import numpy as np
import torch

from armnet_rlt.cache_builder import build_demo_cache
from armnet_rlt.openpi_rlt import PolicyOutput


class _Dataset:
    def __init__(self):
        self.meta = SimpleNamespace(episodes=[{"length": 12}, {"length": 8}])
        self._frames = []
        for index in range(20):
            self._frames.append(
                {
                    "observation.state": torch.arange(6).float() + index,
                    "action": torch.arange(6).float() + index,
                    "observation.images.front": torch.zeros(3, 8, 10),
                    "observation.images.top": torch.zeros(3, 8, 10),
                    "observation.images.wrist": torch.zeros(3, 8, 10),
                }
            )

    def __getitem__(self, index):
        return self._frames[index]


class _Policy:
    def __init__(self):
        self.prompts = []

    def infer_batch(self, observations):
        self.prompts.extend(observation.prompt for observation in observations)
        return [
            PolicyOutput(
                action_chunk=np.zeros((30, 6), np.float32),
                rl_token=np.full(2048, index, np.float32),
            )
            for index, _ in enumerate(observations)
        ]


def test_cache_has_one_terminal_positive_transition_per_episode(tmp_path) -> None:
    output = tmp_path / "cache.pt"

    summary = build_demo_cache(
        dataset=_Dataset(),
        policy=_Policy(),
        output_path=output,
        prompt="push the green button",
        inference_batch_size=2,
    )

    transitions = torch.load(output, weights_only=False)
    assert summary["transition_count"] == 3
    assert summary["terminal_count"] == 2
    assert summary["positive_count"] == 2
    assert [float(item["done"]) for item in transitions] == [0.0, 1.0, 1.0]
    assert all(item["reference_action"].shape == (180,) for item in transitions)
    assert all(item["action"].shape == (60,) for item in transitions)


def test_multitask_cache_uses_each_episode_prompt(tmp_path) -> None:
    dataset = _Dataset()
    dataset.meta.episodes = [
        {"length": 12, "tasks": ["push the green button"]},
        {"length": 8, "tasks": ["flip the left switch off"]},
    ]
    for index, frame in enumerate(dataset._frames):
        frame["task"] = (
            "push the green button"
            if index < 12
            else "flip the left switch off"
        )
    policy = _Policy()

    build_demo_cache(
        dataset=dataset,
        policy=policy,
        output_path=tmp_path / "multitask.pt",
        prompt=None,
        inference_batch_size=2,
    )

    assert policy.prompts == [
        "push the green button",
        "push the green button",
        "flip the left switch off",
        "flip the left switch off",
    ]
