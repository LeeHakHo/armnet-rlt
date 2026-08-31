from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from armnet_rlt.artifacts import load_norm_stats, load_transition_cache
from armnet_rlt.config import so101_network_config


def _transition(network):
    return {
        "rl_token": torch.zeros(2048),
        "proprioception": torch.zeros(network.action_dim),
        "reference_action": torch.zeros(network.reference_action_dim),
        "action": torch.zeros(network.predicted_action_dim),
        "reward": torch.tensor(1.0),
        "next_rl_token": torch.ones(2048),
        "next_proprioception": torch.ones(network.action_dim),
        "next_reference_action": torch.ones(network.reference_action_dim),
        "done": torch.tensor(1.0),
    }


def test_loads_global_and_per_timestep_norm_stats(tmp_path) -> None:
    network = so101_network_config(reference_action_len=3, predicted_action_len=2)
    global_stats = {
        "norm_stats": {
            "state": {
                "mean": [0.0] * 6,
                "std": [1.0] * 6,
                "q01": [-1.0] * 6,
                "q99": [1.0] * 6,
            },
            "actions": {"mean": [0.0] * 6, "std": [1.0] * 6},
        }
    }
    (tmp_path / "norm_stats.json").write_text(json.dumps(global_stats))
    per_timestep = {
        "actions": {
            "mean": np.zeros((3, 6)).tolist(),
            "std": np.ones((3, 6)).tolist(),
        }
    }
    (tmp_path / "norm_stats_actions_per_timestep.json").write_text(
        json.dumps(per_timestep)
    )

    loaded = load_norm_stats(
        tmp_path, use_delta_actions=True, network=network
    )
    assert loaded["state"].mean.shape == (6,)
    assert loaded["actions"].mean.shape == (3, 6)


def test_loads_and_summarizes_valid_transition_cache(tmp_path) -> None:
    network = so101_network_config(reference_action_len=3, predicted_action_len=2)
    path = tmp_path / "cache.pt"
    torch.save([_transition(network), _transition(network)], path)

    transitions, summary = load_transition_cache(path, network)
    assert len(transitions) == 2
    assert summary.transition_count == 2
    assert summary.positive_rate == 1.0
    assert summary.action_dim == 6


def test_rejects_missing_bad_shape_and_nonfinite_cache_values(tmp_path) -> None:
    network = so101_network_config(reference_action_len=3, predicted_action_len=2)
    transition = _transition(network)
    del transition["next_rl_token"]
    path = tmp_path / "missing.pt"
    torch.save([transition], path)
    with pytest.raises(ValueError, match="missing keys"):
        load_transition_cache(path, network)

    transition = _transition(network)
    transition["action"] = torch.zeros(network.predicted_action_dim + 1)
    torch.save([transition], path)
    with pytest.raises(ValueError, match="shape"):
        load_transition_cache(path, network)

    transition = _transition(network)
    transition["reward"] = torch.tensor(float("nan"))
    torch.save([transition], path)
    with pytest.raises(ValueError, match="non-finite"):
        load_transition_cache(path, network)
