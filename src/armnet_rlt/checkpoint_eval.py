"""Serves a LeRobot training run's checkpoints to a cell actor and logs their success rates."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import Any

from torch import Tensor

from armnet_rlt.jsonl_log import append_jsonl
from armnet_rlt.learner import (
    _drain_interactions,
    _push_weights,
    _restore_rolling_metrics,
    _start_online_transport,
)
from armnet_rlt.metrics import EpisodeRecord

MODEL_FILE = "model.safetensors"


def saved_checkpoints(train_dir: str | Path) -> dict[int, Path]:
    """Map step to ``pretrained_model`` for every finished checkpoint of a lerobot-train run.

    A checkpoint is finished once ``checkpoints/last`` points at it or at a later step.
    """
    root = Path(train_dir) / "checkpoints"
    last = root / "last"
    if not last.exists():
        return {}
    last_step = int(Path(os.readlink(last)).name) if last.is_symlink() else int(last.resolve().name)
    found = {}
    for path in root.iterdir():
        if path.name.isdigit() and int(path.name) <= last_step:
            model = path / "pretrained_model"
            if (model / MODEL_FILE).is_file():
                found[int(path.name)] = model
    return dict(sorted(found.items()))


class _CheckpointWeights:
    """Gives ``_push_weights`` the tensors of a checkpoint that differ from the actor's base."""

    def __init__(self, base: dict[str, Tensor]) -> None:
        self.base = base
        self.changed: dict[str, Tensor] = {}

    def load(self, model_dir: Path) -> None:
        from safetensors.torch import load_file

        state = load_file(str(model_dir / MODEL_FILE))
        if set(state) != set(self.base):
            raise ValueError(f"{model_dir} has different parameters from the base checkpoint")
        self.changed = {
            key: value for key, value in state.items() if not value.equal(self.base[key])
        }

    def actor_state_bytes(self) -> dict[str, Tensor]:
        return self.changed


def run_checkpoint_eval(
    config: Any,
    *,
    train_dir: str | Path,
    base_checkpoint: str | Path,
    run_dir: Path,
    run_id: str,
    rollouts_per_checkpoint: int = 10,
    final_step: int,
    first_step: int = 0,
    poll_s: float = 30.0,
    wandb_project: str | None = None,
) -> dict[str, Any]:
    """Push each checkpoint in step order and wait for its rollouts before the next.

    The actor starts from ``base_checkpoint``; each push carries only the
    tensors that differ from it. Stops after the checkpoint at ``final_step``.
    """
    from safetensors.torch import load_file

    if rollouts_per_checkpoint < 1:
        raise ValueError("rollouts_per_checkpoint must be at least 1")
    run_dir.mkdir(parents=True, exist_ok=True)
    rollout_log = run_dir / "rollout_metrics.jsonl"
    eval_log = run_dir / "checkpoint_success.jsonl"
    weights = _CheckpointWeights(load_file(str(Path(base_checkpoint) / MODEL_FILE)))

    wandb_run = None
    if wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=wandb_project,
            id=run_id,
            name=run_id,
            resume="allow",
            config={
                "train_dir": str(train_dir),
                "base_checkpoint": str(base_checkpoint),
                "rollouts_per_checkpoint": rollouts_per_checkpoint,
                "final_step": final_step,
            },
            tags=["checkpoint-eval"],
        )
        print(f"[ckpt eval] WandB run: {wandb_run.url}", flush=True)

    done_steps = set()
    if eval_log.is_file():
        import json

        for line in eval_log.read_text().splitlines():
            if line.strip():
                done_steps.add(int(json.loads(line)["checkpoint_step"]))

    results: list[dict[str, Any]] = []
    current_step: int | None = None
    current: list[EpisodeRecord] = []
    stale = 0
    parameters_queue: Any = None

    def advance() -> bool:
        """Load (and once serving, push) the next unevaluated checkpoint; false if none is saved yet."""
        nonlocal current_step, current
        pending = [
            (step, path)
            for step, path in saved_checkpoints(train_dir).items()
            if step >= first_step
            and step not in done_steps
            and (current_step is None or step > current_step)
        ]
        if not pending:
            return False
        step, path = pending[0]
        weights.load(path)
        current_step, current = step, []
        if parameters_queue is not None:
            _push_weights(parameters_queue, weights, learner_step=step, partial=True)
        print(
            f"[ckpt eval] step {step}: {len(weights.changed)} tensors differ from "
            f"the base; {path}",
            flush=True,
        )
        return True

    while not advance():
        time.sleep(poll_s)
    rolling, episode_count = _restore_rolling_metrics(rollout_log, window=10)
    shutdown_event = Event()
    actor_subscribed = Event()
    server, transition_queue, interaction_queue, parameters_queue = (
        _start_online_transport(
            config,
            weights,
            shutdown_event,
            learner_step=current_step,
            on_parameter_stream=actor_subscribed.set,
            partial=True,
        )
    )
    print(
        f"[ckpt eval] serving {train_dir} from base {base_checkpoint}; "
        f"{rollouts_per_checkpoint} rollouts per checkpoint up to step {final_step}",
        flush=True,
    )
    try:
        while not shutdown_event.is_set():
            while not transition_queue.empty():
                transition_queue.get()
            labels: list[EpisodeRecord] = []
            _drain_interactions(
                interaction_queue,
                rolling,
                rollout_log=rollout_log,
                episode_count=episode_count,
                learner_step=current_step,
                wandb_run=wandb_run,
                sink=labels,
            )
            episode_count += len(labels)
            for record in labels:
                if record.policy_step_start == current_step:
                    current.append(record)
                else:
                    stale += 1
                    print(
                        f"[ckpt eval] ignored a rollout on step {record.policy_step_start} "
                        f"while evaluating {current_step}",
                        flush=True,
                    )

            if actor_subscribed.is_set():
                actor_subscribed.clear()
                _push_weights(parameters_queue, weights, learner_step=current_step, partial=True)

            if len(current) >= rollouts_per_checkpoint:
                evaluated = current[:rollouts_per_checkpoint]
                successes = sum(record.success for record in evaluated)
                result = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "checkpoint_step": current_step,
                    "rollouts": len(evaluated),
                    "successes": successes,
                    "success_rate": successes / len(evaluated),
                    "mean_duration_s": sum(r.duration_s for r in evaluated) / len(evaluated),
                }
                append_jsonl(eval_log, result, sync=True)
                results.append(result)
                done_steps.add(current_step)
                print(
                    f"[ckpt eval] step {current_step}: {successes}/{len(evaluated)} "
                    f"success ({100 * result['success_rate']:.0f}%)",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "eval/success_rate": result["success_rate"],
                            "eval/successes": successes,
                            "eval/mean_duration_s": result["mean_duration_s"],
                            "eval/checkpoint_step": current_step,
                        },
                        step=current_step,
                    )
                if current_step >= final_step:
                    break
                while not advance() and not shutdown_event.is_set():
                    time.sleep(poll_s)
                continue
            time.sleep(0.2)
    finally:
        shutdown_event.set()
        server.stop(grace=1.0)

    summary = {
        "phase": "checkpoint_eval",
        "checkpoints": results,
        "episodes": episode_count,
        "stale_rollouts": stale,
    }
    if wandb_run is not None:
        wandb_run.summary.update({"checkpoints_evaluated": len(results)})
        wandb_run.finish()
    print(f"[ckpt eval] complete: {summary}", flush=True)
    return summary
