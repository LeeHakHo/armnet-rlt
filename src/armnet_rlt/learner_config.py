"""Learner configuration shared by the Modal and local entrypoints."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from armnet_rlt.config import (
    BISO101_JERK_JOINT_WEIGHTS,
    SO101_JERK_JOINT_WEIGHTS,
    RLTConfig,
    RLTNetworkConfig,
    biso101_network_config,
    so101_network_config,
)
from armnet_rlt.resources import task_paths


def network_for(
    embodiment: str, reference_action_len: int, predicted_action_len: int = 10
) -> tuple[RLTNetworkConfig, tuple[float, ...]]:
    if embodiment == "so101":
        return (
            so101_network_config(
                reference_action_len=reference_action_len,
                predicted_action_len=predicted_action_len,
            ),
            SO101_JERK_JOINT_WEIGHTS,
        )
    if embodiment == "biso101":
        return (
            biso101_network_config(
                reference_action_len=reference_action_len,
                predicted_action_len=predicted_action_len,
            ),
            BISO101_JERK_JOINT_WEIGHTS,
        )
    raise ValueError("embodiment must be 'so101' or historical learner-only 'biso101'")


def learner_config(
    *,
    data_root: str | os.PathLike[str],
    embodiment: str,
    config_name: str,
    run_id: str,
    output_dir: str | None = None,
    batch_size: int = 256,
    max_demo_pretrain_steps: int = 5_000,
    online_steps: int = 300_000,
    utd_ratio: int | None = None,
    online_batch_fraction: float = 0.5,
    q_loss_normalization: bool = False,
    discount: float | None = None,
    num_critics: int | None = None,
    bc_deviation_max: float = 0.0,
    bc_deviation_weight: float = 1.0,
    actor_lr: float = 0.0,
    actor_lr_min: float = 0.0,
    save_freq: int | None = None,
    device: str = "cuda",
    algorithm: str = "rlt",
    dsrl_noise_bound: float = 2.5,
    dsrl_noise_steps: int = 1,
    dsrl_target_entropy: float = 0.0,
    dsrl_alpha_init: float = 1.0,
    actions_to_execute: int = 10,
) -> RLTConfig:
    cache_path, assets_dir, production_output = task_paths(config_name, data_root)
    network, jerk = network_for(embodiment, reference_action_len=30)
    dsrl_kwargs: dict[str, object] = {}
    if algorithm == "dsrl":
        # The official pi0 DSRL settings (nakamotoo/dsrl_pi0, run_real.sh): the
        # noise spans the base policy's full 30-step chunk, 1024x3 networks,
        # two critics, 30 gradient steps per transition each on a fresh batch,
        # discount 0.99, Adam 1e-4 for the actor and 3e-4 for the critics with
        # no schedule, learning from the first episode on, no demonstrations,
        # and a replay buffer of 2 * steps / UTD transitions.
        network, jerk = network_for(
            embodiment, reference_action_len=30, predicted_action_len=30
        )
        network = replace(network, hidden_dims=(1024, 1024, 1024))
        max_demo_pretrain_steps = 0
        online_batch_fraction = 1.0
        discount = 0.99 if discount is None else discount
        num_critics = 2 if num_critics is None else num_critics
        utd_ratio = 30 if utd_ratio is None else utd_ratio
        actor_lr = actor_lr if actor_lr > 0 else 1e-4
        actor_lr_min = actor_lr
        dsrl_kwargs = {
            "critic_lr": 3e-4,
            "critic_lr_min": 3e-4,
            "policy_update_freq": 1,
            "max_q": None,
            "online_step_before_learning": 1,
            "online_buffer_capacity": max(1, 2 * online_steps // utd_ratio),
        }
    kwargs = {
        "algorithm": algorithm,
        "dsrl_noise_bound": dsrl_noise_bound,
        "dsrl_noise_steps": dsrl_noise_steps,
        "dsrl_target_entropy": dsrl_target_entropy,
        "dsrl_alpha_init": dsrl_alpha_init,
        **dsrl_kwargs,
        "actions_to_execute": actions_to_execute,
        "embodiment": embodiment,
        "network": network,
        "jerk_joint_weights": jerk,
        "assets_dir": assets_dir,
        "demo_cache_path": cache_path,
        "output_dir": output_dir or production_output,
        "run_id": run_id,
        "device": device,
        "batch_size": batch_size,
        "max_demo_pretrain_steps": max_demo_pretrain_steps,
        "online_steps": online_steps,
        "policy_uses_delta_actions": True,
        "online_batch_fraction": online_batch_fraction,
        "q_loss_normalization": q_loss_normalization,
        "bc_deviation_max": bc_deviation_max,
        "bc_deviation_weight": bc_deviation_weight,
    }
    if discount is not None:
        kwargs["discount"] = discount
    if num_critics is not None:
        kwargs["num_critics"] = num_critics
    if utd_ratio is not None:
        kwargs["utd_ratio"] = utd_ratio
    if save_freq is not None:
        kwargs["save_freq"] = save_freq
    # Override the actor learning rate and its floor.
    if actor_lr > 0:
        kwargs["actor_lr"] = actor_lr
        kwargs["actor_lr_min"] = actor_lr_min if actor_lr_min > 0 else actor_lr / 4
    return RLTConfig(**kwargs)


def resolve_init_checkpoint(
    config: RLTConfig, init_run_id: str, init_step: int = 0
) -> Path | None:
    """Locate ``init_run_id``'s checkpoint at ``init_step``, or its latest."""
    if not init_run_id:
        return None
    root = config.checkpoint_dir.parent / init_run_id
    if init_step:
        checkpoint = root / f"step_{init_step:06d}" / "checkpoint.pt"
    else:
        found = sorted(root.glob("step_*/checkpoint.pt"))
        if not found:
            raise FileNotFoundError(f"no checkpoint under {root}")
        checkpoint = found[-1]
    if not checkpoint.exists():
        raise FileNotFoundError(str(checkpoint))
    return checkpoint
