"""Streaming BC learner for a LeRobot policy: fine-tune on the successes a cell actor streams in."""

from __future__ import annotations

import random
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np
import torch
from torch import Tensor

from armnet_rlt.config import episodes_for_round, parse_round_schedule
from armnet_rlt.jsonl_log import append_jsonl
from armnet_rlt.learner import (
    _drain_interactions,
    _logged_episode_keys,
    _push_weights,
    _replayed,
    _restore_rolling_metrics,
    _start_online_transport,
)
from armnet_rlt.lerobot_actor import EPISODE_SCHEMA, decode_jpeg, load_lerobot_policy
from armnet_rlt.metrics import EpisodeRecord

# Episode payload schemas accepted from actors, newest first.
EPISODE_SCHEMAS = (EPISODE_SCHEMA, "act_episode_v1")


class _PolicyWeights:
    """Gives ``_push_weights`` the policy's full state dict."""

    def __init__(self, policy: Any) -> None:
        self.policy = policy

    def actor_state_bytes(self) -> dict[str, Tensor]:
        return {
            key: value.detach().cpu() for key, value in self.policy.state_dict().items()
        }


def frame_layout(config: Any) -> tuple[list[int] | None, list[int]]:
    """Observation and action frame offsets a training sample takes around its
    frame, from the policy config (``None`` observation offsets: one frame)."""
    observation = getattr(config, "observation_delta_indices", None)
    action = getattr(config, "action_delta_indices", None)
    if action is None:
        action = list(range(int(config.chunk_size)))
    return (None if observation is None else [int(i) for i in observation]), [
        int(i) for i in action
    ]


def _drain_episodes(
    queue: Any,
    completed: deque[dict[str, Any]],
    shutdown: Event,
    seen: set[bytes] | None = None,
) -> int:
    """Queue episode payloads in arrival order, dropping replayed ones."""
    from lerobot.transport.utils import bytes_to_python_object

    count = 0
    while not queue.empty() and not shutdown.is_set():
        payload = queue.get()
        if _replayed(payload, seen):
            continue
        episode = bytes_to_python_object(payload)
        if not isinstance(episode, dict) or episode.get("schema") not in EPISODE_SCHEMAS:
            raise ValueError("episode payload has an unknown schema")
        completed.append(episode)
        count += 1
    return count


def online_batch(
    episodes: list[dict[str, Any]],
    size: int,
    *,
    observation_offsets: list[int] | None,
    action_offsets: list[int],
    image_keys: list[str],
    pool: ThreadPoolExecutor,
    task: str = "",
) -> dict[str, Any]:
    """Sample frames uniformly over all frames with the observation history and
    action window the policy trains on; frames outside the episode are clamped
    and flagged in ``action_is_pad``."""
    lengths = [len(episode["state"]) for episode in episodes]
    picks = [
        (episode, random.randrange(len(episode["state"])))
        for episode in random.choices(episodes, weights=lengths, k=size)
    ]
    action_rows = np.asarray(action_offsets)
    states, actions, pads, image_rows = [], [], [], []
    for episode, index in picks:
        length = len(episode["action"])
        rows = index + action_rows
        pads.append(torch.from_numpy((rows < 0) | (rows >= length)))
        actions.append(torch.from_numpy(episode["action"][np.clip(rows, 0, length - 1)]))
        if observation_offsets is None:
            states.append(torch.from_numpy(episode["state"][index]))
            image_rows.append([index])
        else:
            obs = np.clip(index + np.asarray(observation_offsets), 0, length - 1)
            states.append(torch.from_numpy(episode["state"][obs]))
            image_rows.append([int(i) for i in obs])
    batch: dict[str, Any] = {
        "observation.state": torch.stack(states),
        "action": torch.stack(actions),
        "action_is_pad": torch.stack(pads),
    }
    for key in image_keys:
        jobs = [
            (episode, row) for (episode, _), rows in zip(picks, image_rows) for row in rows
        ]
        decoded = list(pool.map(lambda job: decode_jpeg(job[0]["images"][key][job[1]]), jobs))
        images = torch.from_numpy(np.stack(decoded)).permute(0, 3, 1, 2).float() / 255.0
        if observation_offsets is None:
            batch[key] = images
        else:
            batch[key] = images.reshape(size, len(observation_offsets), *images.shape[1:])
    if task:
        batch["task"] = [task] * size
    return batch


def _base_loader(
    repo_id: str,
    *,
    observation_offsets: list[int] | None,
    action_offsets: list[int],
    image_keys: list[str],
    batch_size: int,
    num_workers: int,
) -> Any:
    """Endless iterator over an offline LeRobot dataset with the policy's frame layout."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    fps = LeRobotDatasetMetadata(repo_id).fps
    delta_timestamps = {"action": [i / fps for i in action_offsets]}
    if observation_offsets is not None:
        for key in ("observation.state", *image_keys):
            delta_timestamps[key] = [i / fps for i in observation_offsets]
    dataset = LeRobotDataset(
        repo_id, delta_timestamps=delta_timestamps, video_backend="torchcodec"
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        # Forked workers break the live gRPC server's connections to the actor.
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
    print(
        f"[lerobot sbc] base dataset {repo_id}: {dataset.num_episodes} episodes, "
        f"{dataset.num_frames} frames",
        flush=True,
    )
    while True:
        yield from loader


def _merge(first: dict[str, Any], second: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    merged = {
        key: torch.cat([first[key], second[key].to(first[key].dtype)]) for key in keys
    }
    if "task" in first:
        merged["task"] = list(first["task"]) + list(second.get("task", first["task"][:1] * len(second["action"])))
    return merged


def _restore_run(run_dir: Path) -> tuple[Path, int, list[dict[str, Any]]]:
    """The newest checkpoint of an earlier run, its step and the saved success episodes."""
    from armnet_rlt.checkpoint_eval import saved_checkpoints

    checkpoints = saved_checkpoints(run_dir)
    if not checkpoints:
        raise FileNotFoundError(f"no finished checkpoint under {run_dir / 'checkpoints'}")
    step, checkpoint = max(checkpoints.items())
    files = sorted(
        (run_dir / "success_episodes").glob("rollout_*_*.pt"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]),
    )
    return checkpoint, step, [torch.load(path, weights_only=False) for path in files]


def _save(policy: Any, preprocessor: Any, postprocessor: Any, run_dir: Path, step: int) -> Path:
    target = run_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
    policy.save_pretrained(target)
    preprocessor.save_pretrained(target)
    postprocessor.save_pretrained(target)
    last = run_dir / "checkpoints" / "last"
    last.unlink(missing_ok=True)
    last.symlink_to(target.parent.name)
    return target


def run_lerobot_streaming_bc(
    config: Any,
    *,
    policy_path: str = "",
    run_dir: Path,
    run_id: str,
    task: str = "",
    episodes_per_round: str | int,
    steps_per_round: int,
    batch_size: int = 32,
    online_steps: int = 30_000,
    lr: float = 0.0,
    base_dataset: str = "",
    base_batch_fraction: float = 0.0,
    num_workers: int = 8,
    min_shaped_reward: float = 0.0,
    idle_timeout_s: float = 600.0,
    idle_min_episodes: int = 5,
    log_freq: int = 50,
    grad_clip_norm: float = 10.0,
    seed: int = 42,
    device: str = "cuda",
    wandb_project: str | None = None,
    resume: bool = False,
) -> dict[str, object]:
    """Fine-tune a LeRobot policy on the successes a live actor streams in, round by round.

    ``config.actor_learner`` holds the gRPC server settings. ``task`` is the
    language instruction given to a language-conditioned policy. With
    ``resume`` the policy, step count and success episodes come from
    ``run_dir``; ``policy_path`` is then ignored.
    """
    round_schedule = parse_round_schedule(episodes_per_round)
    if not round_schedule or steps_per_round <= 0:
        raise ValueError("episodes_per_round and steps_per_round are required")
    if not 0.0 <= base_batch_fraction < 1.0:
        raise ValueError("base_batch_fraction must be in [0, 1)")
    if base_batch_fraction > 0 and not base_dataset:
        raise ValueError("base_batch_fraction needs base_dataset")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    episode_dir = run_dir / "success_episodes"
    episode_dir.mkdir(exist_ok=True)
    rollout_log = run_dir / "rollout_metrics.jsonl"
    training_log = run_dir / "training_metrics.jsonl"
    successes: list[dict[str, Any]] = []
    completed_steps = 0
    if resume:
        checkpoint, completed_steps, successes = _restore_run(run_dir)
        policy_path = str(checkpoint)
        print(
            f"[lerobot sbc] resuming {run_id} at step {completed_steps} from {checkpoint} "
            f"with {len(successes)} success episodes",
            flush=True,
        )
    elif not policy_path:
        raise ValueError("policy_path is required unless resume is set")

    policy, preprocessor, postprocessor = load_lerobot_policy(policy_path, device)
    policy.train()
    observation_offsets, action_offsets = frame_layout(policy.config)
    image_keys = list(policy.config.image_features)
    batch_keys = ["observation.state", *image_keys, "action", "action_is_pad"]
    learning_rate = lr or float(policy.config.optimizer_lr)
    param_groups = policy.get_optim_params()
    if lr:
        for group in param_groups:
            if "lr" in group:
                group["lr"] = lr * group["lr"] / float(policy.config.optimizer_lr)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=learning_rate,
        weight_decay=float(getattr(policy.config, "optimizer_weight_decay", 0.0)),
    )
    base_size = round(batch_size * base_batch_fraction) if base_batch_fraction else 0
    base_iter = (
        _base_loader(
            base_dataset,
            observation_offsets=observation_offsets,
            action_offsets=action_offsets,
            image_keys=image_keys,
            batch_size=base_size,
            num_workers=num_workers,
        )
        if base_size
        else None
    )

    wandb_run = None
    if wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=wandb_project,
            id=run_id,
            resume="allow",
            config={
                "policy_path": policy_path,
                "policy_type": str(policy.config.type),
                "task": task,
                "episodes_per_round": str(episodes_per_round),
                "steps_per_round": steps_per_round,
                "batch_size": batch_size,
                "lr": learning_rate,
                "base_dataset": base_dataset,
                "base_batch_fraction": base_batch_fraction,
                "min_shaped_reward": min_shaped_reward,
                "action_offsets": action_offsets,
                "observation_offsets": observation_offsets,
            },
            tags=["streaming-bc", str(policy.config.type)],
        )
        print(f"[lerobot sbc] WandB run: {wandb_run.url}", flush=True)

    rolling, episode_count = _restore_rolling_metrics(rollout_log, window=10)
    seen_episodes = _logged_episode_keys(rollout_log)
    seen_transitions: set[bytes] = set()
    shutdown_event = Event()
    actor_subscribed = Event()
    weights = _PolicyWeights(policy)
    server, transition_queue, interaction_queue, parameters_queue = (
        _start_online_transport(
            config,
            weights,
            shutdown_event,
            learner_step=completed_steps,
            on_parameter_stream=actor_subscribed.set,
        )
    )
    pool = ThreadPoolExecutor(max_workers=8)
    arrived_episodes: deque[dict[str, Any]] = deque()
    pending_labels: deque[EpisodeRecord] = deque()
    pending_updates = 0
    episodes_since_round = 0
    rounds = completed_steps // steps_per_round
    rejected_episodes = 0
    dropped_episodes = 0
    final_loss = 0.0
    last_arrival = time.monotonic()
    stop_reason = "step_budget"
    print(
        f"[lerobot sbc] serving {policy.config.type} from {policy_path}; "
        f"{steps_per_round} step(s) per round on the episode schedule "
        f"{list(round_schedule)}, batch={batch_size} (base {base_size}), "
        f"lr={learning_rate:g}, actions {action_offsets[0]}..{action_offsets[-1]}, "
        f"observations {observation_offsets}",
        flush=True,
    )
    try:
        while not shutdown_event.is_set() and completed_steps < online_steps:
            arrived = _drain_episodes(
                transition_queue, arrived_episodes, shutdown_event, seen=seen_transitions
            )
            labels: list[EpisodeRecord] = []
            arrived += _drain_interactions(
                interaction_queue,
                rolling,
                rollout_log=rollout_log,
                episode_count=episode_count,
                learner_step=completed_steps,
                wandb_run=wandb_run,
                sink=labels,
                seen=seen_episodes,
            )
            episode_count += len(labels)
            episodes_since_round += len(labels)
            pending_labels.extend(labels)

            # Pair success records with their frames in arrival order.
            while pending_labels:
                record = pending_labels[0]
                if record.num_transitions == 0:
                    pending_labels.popleft()
                    continue
                if not arrived_episodes:
                    break
                pending_labels.popleft()
                episode = arrived_episodes.popleft()
                if len(episode["state"]) != record.num_transitions:
                    dropped_episodes += 1
                    print(
                        f"[lerobot sbc] dropped an episode: {len(episode['state'])} "
                        f"frames arrived but the record reports {record.num_transitions}",
                        flush=True,
                    )
                    continue
                if record.shaped_reward < min_shaped_reward - 1e-9:
                    rejected_episodes += 1
                    print(
                        f"[lerobot sbc] skipped a success: reward "
                        f"{record.shaped_reward:.2f} < {min_shaped_reward:.2f}",
                        flush=True,
                    )
                    continue
                successes.append(episode)
                torch.save(
                    episode,
                    episode_dir / f"rollout_{record.session_rollout_index:04d}_{len(successes):04d}.pt",
                )
                print(
                    f"[lerobot sbc] success {len(successes)}: +{len(episode['state'])} "
                    f"frames, total {sum(len(e['state']) for e in successes)}",
                    flush=True,
                )

            while (
                not pending_labels
                and episodes_since_round >= episodes_for_round(round_schedule, rounds + 1)
            ):
                episodes_since_round -= episodes_for_round(round_schedule, rounds + 1)
                rounds += 1
                if successes:
                    pending_updates += steps_per_round
                    print(
                        f"[lerobot sbc] round {rounds}: {steps_per_round} steps on "
                        f"{len(successes)} success episodes",
                        flush=True,
                    )
                else:
                    print(f"[lerobot sbc] round {rounds}: no successes yet", flush=True)

            if arrived:
                last_arrival = time.monotonic()
            elif (
                episode_count >= idle_min_episodes
                and pending_updates <= 0
                and time.monotonic() - last_arrival >= idle_timeout_s
            ):
                stop_reason = "actor_idle"
                print(
                    f"[lerobot sbc] no actor traffic for {idle_timeout_s:.0f}s after "
                    f"{episode_count} episodes; stopping",
                    flush=True,
                )
                break

            if actor_subscribed.is_set():
                actor_subscribed.clear()
                _push_weights(parameters_queue, weights, learner_step=completed_steps)

            if pending_updates <= 0:
                time.sleep(0.05)
                continue

            batch = online_batch(
                successes,
                batch_size - base_size,
                observation_offsets=observation_offsets,
                action_offsets=action_offsets,
                image_keys=image_keys,
                pool=pool,
                task=task,
            )
            if base_iter is not None:
                batch = _merge(batch, next(base_iter), batch_keys)
            loss, loss_dict = policy.forward(preprocessor(batch))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
            optimizer.step()
            final_loss = float(loss.detach())
            completed_steps += 1
            pending_updates -= 1
            loss_dict = {
                key: float(value) for key, value in (loss_dict or {}).items()
                if isinstance(value, (int, float)) or hasattr(value, "item")
            }

            if completed_steps % log_freq == 0:
                print(
                    f"[lerobot sbc] step={completed_steps} loss={final_loss:.4f} "
                    f"successes={len(successes)}/{episode_count} pending={pending_updates}"
                    + "".join(f" {key}={value:.4f}" for key, value in loss_dict.items()),
                    flush=True,
                )
                append_jsonl(
                    training_log,
                    {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "phase": "lerobot_streaming_bc",
                        "learner_step": completed_steps,
                        "loss": final_loss,
                        **loss_dict,
                        "grad_norm": float(grad_norm),
                        "success_episodes": len(successes),
                        "episodes_total": episode_count,
                        "pending_updates": pending_updates,
                    },
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "bc/loss": final_loss,
                            "bc/grad_norm": float(grad_norm),
                            "bc/step": completed_steps,
                            "bc/success_episodes": len(successes),
                            "bc/episodes": episode_count,
                            **{f"bc/{key}": value for key, value in loss_dict.items()},
                        },
                        step=completed_steps,
                    )
            if pending_updates == 0:
                policy.eval()
                checkpoint = _save(policy, preprocessor, postprocessor, run_dir, completed_steps)
                _push_weights(parameters_queue, weights, learner_step=completed_steps)
                policy.train()
                print(
                    f"[lerobot sbc] round {rounds} done at step {completed_steps}; "
                    f"pushed weights, saved {checkpoint}",
                    flush=True,
                )
    finally:
        shutdown_event.set()
        server.stop(grace=1.0)
        pool.shutdown(wait=False)

    policy.eval()
    checkpoint = _save(policy, preprocessor, postprocessor, run_dir, completed_steps)
    summary = {
        "phase": "lerobot_streaming_bc",
        "policy_type": str(policy.config.type),
        "steps": completed_steps,
        "loss": final_loss,
        "checkpoint": str(checkpoint),
        "episodes": episode_count,
        "success_episodes": len(successes),
        "rejected_episodes": rejected_episodes,
        "dropped_episodes": dropped_episodes,
        "stop_reason": stop_reason,
        "rounds": rounds,
        **rolling.to_dict(),
    }
    if wandb_run is not None:
        wandb_run.summary.update(summary)
        wandb_run.finish()
    print(f"[lerobot sbc] complete: {summary}", flush=True)
    return summary
