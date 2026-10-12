"""Build TD3 demonstration transitions from a LeRobot imitation dataset."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from collections.abc import Container
from typing import Any

import numpy as np
import torch

from armnet_rlt.artifacts import load_transition_cache
from armnet_rlt.config import so101_network_config
from armnet_rlt.openpi_rlt import OpenPIRLTPolicy, PolicyObservation
from armnet_rlt.resources import (
    GREEN_BUTTON_PROMPT,
    MULTITASK_CHECKPOINT,
    MULTITASK_CONFIG,
    task_paths,
)


CAMERAS = ("front", "top", "wrist")


def _image(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"camera frame must be 3-D, got {image.shape}")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if image.size and float(image.max()) <= 1.0 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8, copy=False)
    return image[..., :3]


def _frame_observation(frame: dict[str, Any], prompt: str) -> PolicyObservation:
    state = torch.as_tensor(frame["observation.state"]).detach().cpu().numpy()
    images = {
        camera: _image(frame[f"observation.images.{camera}"])
        for camera in CAMERAS
    }
    return PolicyObservation(
        state=np.asarray(state, dtype=np.float32).reshape(-1)[:6],
        images=images,
        prompt=prompt,
    )


def _episode_prompt(
    frame: dict[str, Any],
    metadata: dict[str, Any],
    fixed_prompt: str | None,
) -> str:
    if fixed_prompt:
        return fixed_prompt
    task = frame.get("task")
    if isinstance(task, str) and task.strip():
        return task.strip()
    tasks = metadata.get("tasks")
    if isinstance(tasks, (list, tuple)) and len(tasks) == 1:
        task = tasks[0]
        if isinstance(task, str) and task.strip():
            return task.strip()
    raise ValueError(
        "multitask cache episode has no unambiguous task prompt in its "
        "first frame or episode metadata"
    )


def _executed_chunk(
    dataset: Any,
    *,
    episode_start: int,
    episode_length: int,
    frame_offset: int,
    chunk_length: int,
) -> torch.Tensor:
    actions: list[torch.Tensor] = []
    last_index = episode_start + episode_length - 1
    for step in range(chunk_length):
        index = min(episode_start + frame_offset + step, last_index)
        action = torch.as_tensor(dataset[index]["action"], dtype=torch.float32)
        actions.append(action.reshape(-1)[:6].cpu())
    return torch.stack(actions)


def _infer_fixed_batches(
    policy: OpenPIRLTPolicy,
    observations: list[PolicyObservation],
    batch_size: int,
) -> list:
    outputs = []
    for start in range(0, len(observations), batch_size):
        batch = observations[start : start + batch_size]
        actual = len(batch)
        if actual < batch_size:
            batch = [*batch, *([batch[-1]] * (batch_size - actual))]
        outputs.extend(policy.infer_batch(batch)[:actual])
    return outputs


def build_demo_cache(
    *,
    dataset: Any,
    policy: OpenPIRLTPolicy,
    output_path: str | Path,
    prompt: str | None,
    inference_batch_size: int = 4,
    predicted_action_len: int = 10,
    reference_action_len: int = 30,
    episode_filter: Container[int] | None = None,
) -> dict[str, int | float]:
    """Extract one transition per non-overlapping action chunk.

    The source is an imitation-learning dataset, so every episode is a positive
    demonstration. Exactly one transition per episode is terminal and receives
    reward 1; earlier chunks receive reward 0.
    """
    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be positive")
    episodes = list(dataset.meta.episodes)
    transitions: list[dict[str, torch.Tensor]] = []
    episode_start = 0
    for episode_index, metadata in enumerate(episodes):
        episode_length = int(metadata["length"])
        if episode_filter is not None and episode_index not in episode_filter:
            episode_start += episode_length
            continue
        offsets = list(range(0, episode_length, predicted_action_len))
        episode_prompt = _episode_prompt(
            dataset[episode_start],
            metadata,
            prompt,
        )
        observations = [
            _frame_observation(
                dataset[episode_start + offset],
                episode_prompt,
            )
            for offset in offsets
        ]
        executed = [
            _executed_chunk(
                dataset,
                episode_start=episode_start,
                episode_length=episode_length,
                frame_offset=offset,
                chunk_length=predicted_action_len,
            )
            for offset in offsets
        ]
        outputs = _infer_fixed_batches(
            policy, observations, inference_batch_size
        )
        tick_data = []
        for output, observation, action in zip(
            outputs, observations, executed, strict=True
        ):
            reference = np.asarray(output.action_chunk, dtype=np.float32)
            if reference.shape[0] < reference_action_len or reference.shape[1] < 6:
                raise ValueError(
                    f"episode {episode_index} reference action has shape "
                    f"{reference.shape}, expected at least "
                    f"({reference_action_len}, 6)"
                )
            tick_data.append(
                {
                    "rl_token": torch.from_numpy(output.rl_token).float(),
                    "proprioception": torch.from_numpy(observation.state).float(),
                    "reference_action": torch.from_numpy(
                        reference[:reference_action_len, :6].copy()
                    ).flatten(),
                    "action": action.flatten(),
                }
            )
        for index, current in enumerate(tick_data):
            terminal = index == len(tick_data) - 1
            following = tick_data[min(index + 1, len(tick_data) - 1)]
            transitions.append(
                {
                    **current,
                    "reward": torch.tensor(float(terminal)),
                    "next_rl_token": following["rl_token"],
                    "next_proprioception": following["proprioception"],
                    "next_reference_action": following["reference_action"],
                    "done": torch.tensor(float(terminal)),
                }
            )
        episode_start += episode_length
        print(
            f"[RLT cache] episode {episode_index + 1}/{len(episodes)}: "
            f"{len(tick_data)} transitions",
            flush=True,
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".pt.tmp")
    torch.save(transitions, temporary)
    temporary.replace(output)
    _, summary = load_transition_cache(
        output,
        so101_network_config(
            predicted_action_len=predicted_action_len,
            reference_action_len=reference_action_len,
        ),
    )
    return summary.to_dict()


def copy_norm_stats(checkpoint_dir: str | Path, target_assets: str | Path) -> None:
    """Copy both learner normalization files beside the generated cache."""
    source = Path(checkpoint_dir) / "assets"
    target = Path(target_assets)
    target.mkdir(parents=True, exist_ok=True)
    for name in (
        "norm_stats.json",
        "norm_stats_actions_per_timestep.json",
    ):
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(path)
        shutil.copy2(path, target / name)


def build_task_cache(
    *,
    config_name: str,
    checkpoint_repo: str,
    dataset_repo: str,
    prompt: str | None,
    inference_batch_size: int,
    data_root: str | os.PathLike[str],
) -> dict:
    """Build the demonstration cache and normalization assets under ``data_root``."""
    from huggingface_hub import snapshot_download
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    checkpoint = snapshot_download(checkpoint_repo)
    print(f"[RLT cache] checkpoint={checkpoint}", flush=True)
    policy = OpenPIRLTPolicy.from_checkpoint(
        checkpoint,
        config_name,
        default_prompt=prompt or "",
    )
    # Load the dataset from the main revision.
    dataset = LeRobotDataset(
        dataset_repo, revision="main", video_backend="pyav"
    )
    cache_path, assets_dir, _output_dir = task_paths(config_name, data_root)
    summary = build_demo_cache(
        dataset=dataset,
        policy=policy,
        output_path=cache_path,
        prompt=prompt,
        inference_batch_size=inference_batch_size,
    )
    copy_norm_stats(checkpoint, assets_dir)
    result = {
        **summary,
        "config_name": config_name,
        "dataset": dataset_repo,
        "checkpoint": checkpoint_repo,
        "cache_path": cache_path,
        "assets_dir": assets_dir,
    }
    print(f"[RLT cache] complete: {result}", flush=True)
    return result


def build_rollout_cache(
    *,
    sources: list[dict],
    data_root: str | os.PathLike[str],
    inference_batch_size: int = 4,
    prompt: str = GREEN_BUTTON_PROMPT,
    cache_name: str = "rlt_bc_cache.pt",
) -> dict:
    """Build one cache from selected episodes of recorded rollout datasets."""
    from huggingface_hub import snapshot_download
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    checkpoint = snapshot_download(MULTITASK_CHECKPOINT)
    policy = OpenPIRLTPolicy.from_checkpoint(
        checkpoint, MULTITASK_CONFIG, default_prompt=prompt
    )
    demo_cache_path, assets_dir, _ = task_paths(MULTITASK_CONFIG, data_root)
    target = Path(demo_cache_path).with_name(cache_name)

    merged: list[dict] = []
    per_source = []
    for source in sources:
        repo_id = str(source["repo_id"])
        episodes = sorted({int(index) for index in source["episodes"]})
        if not episodes:
            continue
        dataset = LeRobotDataset(repo_id, revision="main", video_backend="pyav")
        part = target.with_name(f"{target.stem}.part.pt")
        build_demo_cache(
            dataset=dataset,
            policy=policy,
            output_path=part,
            prompt=prompt,
            inference_batch_size=inference_batch_size,
            episode_filter=set(episodes),
        )
        chunk = torch.load(part, map_location="cpu", weights_only=False)
        merged.extend(chunk)
        part.unlink(missing_ok=True)
        per_source.append(
            {"repo_id": repo_id, "episodes": len(episodes), "transitions": len(chunk)}
        )
        print(f"[RLT bc] {repo_id}: {len(episodes)} episodes -> {len(chunk)} transitions", flush=True)

    if not merged:
        raise ValueError("no episodes were selected, so no cache was written")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".pt.tmp")
    torch.save(merged, temporary)
    temporary.replace(target)
    _, summary = load_transition_cache(target, so101_network_config())
    copy_norm_stats(checkpoint, assets_dir)
    result = {
        **summary.to_dict(),
        "cache_path": str(target),
        "sources": per_source,
        "episodes_total": sum(item["episodes"] for item in per_source),
    }
    print(f"[RLT bc] complete: {result}", flush=True)
    return result


def manifest_sources(manifest: str | os.PathLike[str]) -> list[dict]:
    """Read ``datasets`` (each ``repo_id`` + ``episodes``) from a JSON manifest."""
    import json

    spec = json.loads(Path(manifest).read_text())
    return [
        {"repo_id": entry["repo_id"], "episodes": entry["episodes"]}
        for entry in spec["datasets"]
        if entry.get("episodes")
    ]
