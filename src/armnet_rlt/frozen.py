"""Inference-only exports of learner actor weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from armnet_rlt.config import (
    RLTConfig,
    SO101_JERK_JOINT_WEIGHTS,
    network_config_from_dict,
    without_secrets,
)


# 1: rlt/dsrl actor only. 2: adds algorithm and actor_meta.
FROZEN_SCHEMA_VERSION = 2
_READABLE_SCHEMAS = {1, 2}


def freeze_learner_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep the actor of one learner checkpoint and what rebuilds it."""
    config = checkpoint.get("config")
    step = checkpoint.get("step")
    if not isinstance(config, Mapping):
        raise ValueError("learner checkpoint lacks config")
    if not isinstance(step, int) or step < 0:
        raise ValueError("learner checkpoint has an invalid step")
    actor_state = checkpoint.get("actor_state")
    if isinstance(actor_state, Mapping):
        actor_state = {str(name): value.detach().cpu() for name, value in actor_state.items()}
    else:
        # Pre-plugin checkpoint: actor.* keys of the rlt/dsrl policy.
        policy_state = checkpoint.get("policy_state_dict")
        if policy_state is None:
            policy_state = (checkpoint.get("algorithm_state") or {}).get("policy_state_dict")
        if not isinstance(policy_state, Mapping):
            raise ValueError("learner checkpoint lacks policy_state_dict")
        actor_state = {
            str(name).removeprefix("actor."): value.detach().cpu()
            for name, value in policy_state.items()
            if str(name).startswith("actor.") and isinstance(value, Tensor)
        }
    if not actor_state:
        raise ValueError("learner checkpoint contains no actor parameters")
    clean = without_secrets(config)
    algorithm = str(checkpoint.get("algorithm") or clean.get("algorithm", "rlt"))
    meta = checkpoint.get("actor_meta")
    artifact = {
        "schema_version": FROZEN_SCHEMA_VERSION,
        "learner_step": step,
        "algorithm": algorithm,
        "actor_state_dict": actor_state,
        "config": clean,
        "source": dict(source or {}),
    }
    artifact["actor_meta"] = str(meta) if meta else _legacy_actor_meta(artifact)
    return artifact


def _legacy_actor_meta(artifact: Mapping[str, Any]) -> str:
    """actor_meta of an rlt/dsrl actor saved without one."""
    from armnet_rlt.algorithm import actor_meta, load_algorithm, spec_for

    config = config_from_frozen_checkpoint(artifact)
    return actor_meta(config.algorithm, spec_for(config, load_algorithm(config.algorithm)), config)


def frozen_actor_payload(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """The ``algorithm.actor_payload`` of a frozen checkpoint."""
    from armnet_rlt.algorithm import actor_payload

    return actor_payload(
        artifact["actor_state_dict"],
        str(artifact["actor_meta"]),
        learner_step=int(artifact["learner_step"]),
    )


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
    if artifact.get("schema_version") not in _READABLE_SCHEMAS:
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
    if not artifact.get("actor_meta"):
        artifact["actor_meta"] = _legacy_actor_meta(artifact)
        artifact.setdefault("algorithm", str(artifact["config"].get("algorithm", "rlt")))
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
    network = network_config_from_dict(raw_network)
    jerk = tuple(
        raw.get("jerk_joint_weights", SO101_JERK_JOINT_WEIGHTS)
    )
    algorithm = str(raw.get("algorithm", "rlt"))
    return RLTConfig(
        embodiment=embodiment,
        network=network,
        algorithm=algorithm,
        algorithm_options=dict(raw.get("algorithm_options") or {}),
        dsrl_noise_bound=float(raw.get("dsrl_noise_bound", 2.5)),
        dsrl_noise_steps=int(raw.get("dsrl_noise_steps", 1)),
        actions_to_execute=(
            int(raw.get("actions_to_execute", 10))
            if algorithm == "dsrl"
            else network.predicted_action_len
        ),
        max_demo_pretrain_steps=0 if algorithm != "rlt" else 500,
        online_batch_fraction=1.0 if algorithm != "rlt" else 0.5,
        use_subsampling=bool(raw.get("use_subsampling", True)),
        sub_chunk_stride=int(raw.get("sub_chunk_stride", 2)),
        use_quantile_norm=bool(raw.get("use_quantile_norm", True)),
        policy_uses_delta_actions=bool(
            raw.get("policy_uses_delta_actions", False)
        ),
        jerk_joint_weights=jerk,
    )
