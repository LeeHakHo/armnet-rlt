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
    def infer_batch(self, observations):
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
