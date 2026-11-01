from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from armnet_rlt.config import RLTNetworkConfig


@dataclass(frozen=True)
class NormStats:
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray | None = None
    q99: np.ndarray | None = None


@dataclass(frozen=True)
class TransitionSummary:
    transition_count: int
    positive_count: int
    positive_rate: float
    terminal_count: int
    rl_token_dim: int
    action_dim: int
    proprioception_dim: int
    reference_action_len: int
    reference_action_dim: int
    predicted_action_len: int
    predicted_action_dim: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "transition_count": self.transition_count,
            "positive_count": self.positive_count,
            "positive_rate": self.positive_rate,
            "terminal_count": self.terminal_count,
            "rl_token_dim": self.rl_token_dim,
            "action_dim": self.action_dim,
            "proprioception_dim": self.proprioception_dim,
            "reference_action_len": self.reference_action_len,
            "reference_action_dim": self.reference_action_dim,
            "predicted_action_len": self.predicted_action_len,
            "predicted_action_dim": self.predicted_action_dim,
        }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"required artifact not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    inner = value.get("norm_stats", value)
    if not isinstance(inner, dict):
        raise ValueError(f"{path}: norm_stats must be an object")
    return inner


def _parse_stats(blob: object, *, source: str) -> NormStats:
    if not isinstance(blob, dict):
        raise ValueError(f"{source} must be an object")
    missing = {"mean", "std"} - blob.keys()
    if missing:
        raise ValueError(f"{source} missing keys: {', '.join(sorted(missing))}")

    def array(name: str, *, optional: bool = False) -> np.ndarray | None:
        value = blob.get(name)
        if value is None and optional:
            return None
        result = np.asarray(value, dtype=np.float32)
        if result.size == 0:
            raise ValueError(f"{source}.{name} must not be empty")
        if not np.isfinite(result).all():
            raise ValueError(f"{source}.{name} contains non-finite values")
        return result

    mean = array("mean")
    std = array("std")
    assert mean is not None and std is not None
    if mean.shape != std.shape:
        raise ValueError(
            f"{source} mean/std shapes differ: {mean.shape} != {std.shape}"
        )
    if np.any(std <= 0):
        raise ValueError(f"{source}.std must be strictly positive")
    q01 = array("q01", optional=True)
    q99 = array("q99", optional=True)
    if (q01 is None) != (q99 is None):
        raise ValueError(f"{source} must provide both q01 and q99")
    if q01 is not None and q99 is not None:
        if q01.shape != mean.shape or q99.shape != mean.shape:
            raise ValueError(f"{source} quantile shapes must match mean")
        if np.any(q99 <= q01):
            raise ValueError(f"{source} requires q99 > q01 element-wise")
    return NormStats(mean=mean, std=std, q01=q01, q99=q99)


def _validate_norm_dimensions(
    stats: dict[str, NormStats], network: RLTNetworkConfig
) -> None:
    state = np.squeeze(stats["state"].mean)
    if state.ndim != 1 or state.shape[0] < network.proprioception_dim:
        raise ValueError(
            "state norm stats need at least "
            f"{network.proprioception_dim} values, got {state.shape}"
        )
    actions = np.squeeze(stats["actions"].mean)
    if actions.ndim == 1:
        if actions.shape[0] < network.action_dim:
            raise ValueError(
                "action norm stats need at least "
                f"{network.action_dim} values, got {actions.shape}"
            )
    elif actions.ndim == 2:
        if actions.shape[1] < network.action_dim:
            raise ValueError(
                "per-timestep action stats need at least "
                f"{network.action_dim} columns, got {actions.shape}"
            )
        if actions.shape[0] < network.reference_action_len:
            raise ValueError(
                "per-timestep action stats need at least "
                f"{network.reference_action_len} rows, got {actions.shape[0]}"
            )
    else:
        raise ValueError(
            f"action norm stats must be 1-D or 2-D, got shape {actions.shape}"
        )


def load_norm_stats(
    assets_dir: str | Path,
    *,
    use_delta_actions: bool = False,
    network: RLTNetworkConfig | None = None,
) -> dict[str, NormStats]:
    directory = Path(assets_dir)
    raw = _read_json(directory / "norm_stats.json")
    missing = {"state", "actions"} - raw.keys()
    if missing:
        raise ValueError(
            "norm_stats.json missing entries: " + ", ".join(sorted(missing))
        )
    result = {
        "state": _parse_stats(raw["state"], source="norm_stats.state"),
        "actions": _parse_stats(raw["actions"], source="norm_stats.actions"),
    }
    if use_delta_actions:
        per_timestep = directory / "norm_stats_actions_per_timestep.json"
        if not per_timestep.exists():
            raise FileNotFoundError(
                "delta actions require per-timestep action stats at "
                f"{per_timestep}"
            )
        per_raw = _read_json(per_timestep)
        if "actions" not in per_raw:
            raise ValueError(f"{per_timestep} is missing the actions entry")
        result["actions"] = _parse_stats(
            per_raw["actions"], source="per_timestep.actions"
        )
    if network is not None:
        _validate_norm_dimensions(result, network)
    return result


_VECTOR_KEYS = (
    "rl_token",
    "proprioception",
    "reference_action",
    "action",
    "next_rl_token",
    "next_proprioception",
    "next_reference_action",
)
_SCALAR_KEYS = ("reward", "done")
_REQUIRED_KEYS = frozenset(_VECTOR_KEYS + _SCALAR_KEYS)


def _tensor(value: object, *, label: str) -> Tensor:
    try:
        result = torch.as_tensor(value, dtype=torch.float32, device="cpu")
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{label} cannot be converted to a float tensor") from exc
    if not torch.isfinite(result).all().item():
        raise ValueError(f"{label} contains non-finite values")
    return result.contiguous()


def load_transition_cache(
    cache_path: str | Path,
    network: RLTNetworkConfig,
) -> tuple[list[dict[str, Tensor]], TransitionSummary]:
    path = Path(cache_path)
    if not path.is_file():
        raise FileNotFoundError(f"prebuilt transition cache not found: {path}")
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"failed to load transition cache {path}: {exc}") from exc
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a list of transitions")
    if not raw:
        raise ValueError(f"{path} contains no transitions")

    shapes = {
        "rl_token": (2048,),
        "proprioception": (network.action_dim,),
        "reference_action": (network.reference_action_dim,),
        "action": (network.predicted_action_dim,),
        "next_rl_token": (2048,),
        "next_proprioception": (network.action_dim,),
        "next_reference_action": (network.reference_action_dim,),
        "reward": (),
        "done": (),
    }
    transitions: list[dict[str, Tensor]] = []
    positive_count = 0
    terminal_count = 0
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"transition {index} must be a dictionary")
        missing = _REQUIRED_KEYS - item.keys()
        if missing:
            raise ValueError(
                f"transition {index} missing keys: {', '.join(sorted(missing))}"
            )
        transition: dict[str, Tensor] = {}
        for key in _VECTOR_KEYS + _SCALAR_KEYS:
            value = _tensor(item[key], label=f"transition {index}.{key}")
            if tuple(value.shape) != shapes[key]:
                raise ValueError(
                    f"transition {index}.{key} has shape {tuple(value.shape)}, "
                    f"expected {shapes[key]}"
                )
            transition[key] = value
        done = float(transition["done"])
        if done not in (0.0, 1.0):
            raise ValueError(f"transition {index}.done must be 0 or 1")
        positive_count += int(float(transition["reward"]) > 0.0)
        terminal_count += int(done == 1.0)
        transitions.append(transition)

    count = len(transitions)
    summary = TransitionSummary(
        transition_count=count,
        positive_count=positive_count,
        positive_rate=positive_count / count,
        terminal_count=terminal_count,
        rl_token_dim=2048,
        action_dim=network.action_dim,
        proprioception_dim=network.proprioception_dim,
        reference_action_len=network.reference_action_len,
        reference_action_dim=network.reference_action_dim,
        predicted_action_len=network.predicted_action_len,
        predicted_action_dim=network.predicted_action_dim,
    )
    return transitions, summary
