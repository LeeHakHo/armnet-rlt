"""Build TD3 demonstration transitions from a LeRobot imitation dataset."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from armnet_rlt.artifacts import load_transition_cache
from armnet_rlt.config import so101_network_config
from armnet_rlt.openpi_rlt import OpenPIRLTPolicy, PolicyObservation


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
    prompt: str,
    inference_batch_size: int = 4,
    predicted_action_len: int = 10,
    reference_action_len: int = 30,
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
        offsets = list(range(0, episode_length, predicted_action_len))
        observations = [
            _frame_observation(dataset[episode_start + offset], prompt)
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
