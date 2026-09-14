"""Portable, inference-only exports of RLT learner actor weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from armnet_rlt.config import (
    RLTConfig,
    RLTNetworkConfig,
    SO101_JERK_JOINT_WEIGHTS,
)


FROZEN_SCHEMA_VERSION = 1


def freeze_learner_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Strip optimizers and critics from one learner checkpoint."""
    policy_state = checkpoint.get("policy_state_dict")
    config = checkpoint.get("config")
    step = checkpoint.get("step")
    if not isinstance(policy_state, Mapping):
        raise ValueError("learner checkpoint lacks policy_state_dict")
    if not isinstance(config, Mapping):
        raise ValueError("learner checkpoint lacks config")
    if not isinstance(step, int) or step < 0:
        raise ValueError("learner checkpoint has an invalid step")
    actor_state = {
        str(name).removeprefix("actor."): value.detach().cpu()
        for name, value in policy_state.items()
        if str(name).startswith("actor.") and isinstance(value, Tensor)
    }
    if not actor_state:
        raise ValueError("learner checkpoint contains no actor parameters")
    return {
        "schema_version": FROZEN_SCHEMA_VERSION,
        "learner_step": step,
        "actor_state_dict": actor_state,
        "config": dict(config),
        "source": dict(source or {}),
    }


def load_frozen_checkpoint(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    try:
        artifact = torch.load(
            artifact_path,
            map_location="cpu",
            weights_only=False,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"frozen RLT checkpoint not found: {artifact_path}"
        ) from None
    except Exception as exc:
        raise ValueError(
            f"failed to load frozen RLT checkpoint {artifact_path}: {exc}"
        ) from exc
    if not isinstance(artifact, dict):
        raise ValueError("frozen RLT checkpoint must contain a dictionary")
    if artifact.get("schema_version") != FROZEN_SCHEMA_VERSION:
        raise ValueError(
            "unsupported frozen RLT checkpoint schema "
            f"{artifact.get('schema_version')!r}"
        )
    if not isinstance(artifact.get("actor_state_dict"), Mapping):
        raise ValueError("frozen RLT checkpoint lacks actor_state_dict")
    if not isinstance(artifact.get("config"), Mapping):
        raise ValueError("frozen RLT checkpoint lacks config")
    step = artifact.get("learner_step")
    if not isinstance(step, int) or step < 0:
        raise ValueError("frozen RLT checkpoint has an invalid learner_step")
    return artifact


def config_from_frozen_checkpoint(artifact: Mapping[str, Any]) -> RLTConfig:
    raw = artifact.get("config")
    if not isinstance(raw, Mapping):
        raise ValueError("frozen RLT checkpoint lacks config")
    embodiment = str(raw.get("embodiment", "so101"))
    if embodiment != "so101":
        raise ValueError("frozen Armnet evaluation supports only so101")
    raw_network = raw.get("network")
    if not isinstance(raw_network, Mapping):
        raise ValueError("frozen RLT config lacks network")
    network_values = dict(raw_network)
    for name in ("hidden_dims", "delta_action_mask"):
        if name in network_values:
            network_values[name] = tuple(network_values[name])
    network = RLTNetworkConfig(**network_values)
    jerk = tuple(
        raw.get("jerk_joint_weights", SO101_JERK_JOINT_WEIGHTS)
    )
    return RLTConfig(
        embodiment=embodiment,
        network=network,
        actions_to_execute=network.predicted_action_len,
        use_subsampling=bool(raw.get("use_subsampling", True)),
        sub_chunk_stride=int(raw.get("sub_chunk_stride", 2)),
        use_quantile_norm=bool(raw.get("use_quantile_norm", True)),
        policy_uses_delta_actions=bool(
            raw.get("policy_uses_delta_actions", False)
        ),
        jerk_joint_weights=jerk,
    )
