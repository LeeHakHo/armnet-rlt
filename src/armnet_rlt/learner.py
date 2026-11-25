from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty, Full, Queue
from collections.abc import Callable, Mapping
from threading import Event
from typing import Any

import numpy as np
import torch
from torch import Tensor

from armnet_rlt.artifacts import (
    TransitionSummary,
    load_norm_stats,
    load_transition_cache,
)
from armnet_rlt.config import (
    RLTConfig,
    RLTNetworkConfig,
    without_secrets,
)
from armnet_rlt.jsonl_log import append_jsonl, read_jsonl
from armnet_rlt.metrics import EpisodeRecord, RollingMetrics, episode_metrics
from armnet_rlt.policy import RLTPolicy


def _batch_from_transitions(
    transitions: list[dict[str, Tensor]],
    batch_size: int,
    device: torch.device,
    *,
    balance_key: str | None = None,
) -> dict[str, Any]:
    if balance_key is None:
        indices = torch.randint(len(transitions), (batch_size,)).tolist()
    else:
        groups: dict[float, list[int]] = {}
        for index, transition in enumerate(transitions):
            raw = transition.get(balance_key, torch.tensor(0.0))
            value = float(raw.item() if isinstance(raw, Tensor) else raw)
            groups.setdefault(round(value, 6), []).append(index)
        grouped_indices = list(groups.values())
        selected_groups = torch.randint(
            len(grouped_indices), (batch_size,)
        ).tolist()
        indices = [
            group[torch.randint(len(group), ()).item()]
            for group in (
                grouped_indices[group_index]
                for group_index in selected_groups
            )
        ]

    def stack(key: str) -> Tensor:
        return torch.stack([transitions[index][key] for index in indices]).to(
            device
        )

    batch = {
        "state": {
            "rl_token": stack("rl_token"),
            "proprioception": stack("proprioception"),
            "reference_action": stack("reference_action"),
        },
        "action": stack("action"),
        "reward": stack("reward"),
        "next_state": {
            "rl_token": stack("next_rl_token"),
            "proprioception": stack("next_proprioception"),
            "reference_action": stack("next_reference_action"),
        },
        "done": stack("done"),
    }
    if all("curriculum_scale" in item for item in transitions):
        batch["curriculum_scale"] = stack("curriculum_scale")
    return batch


def _merge_batches(first: Any, second: Any) -> Any:
    if isinstance(first, dict):
        return {
            key: _merge_batches(first[key], second[key])
            for key in first.keys() & second.keys()
        }
    return torch.cat((first, second), dim=0)


@torch.no_grad()
def _q_stats(policy: RLTPolicy, batch: dict[str, Any]) -> dict[str, float]:
    """Mean and spread of min-Q over the batch, split by reward when both
    rewarded and unrewarded transitions are present.
    """
    state = batch["state"]
    if hasattr(policy, "min_q"):
        q = policy.min_q(
            state["rl_token"], state["proprioception"], batch["action"]
        ).reshape(-1)
    else:
        normalized = policy.actor.normalize_action(
            batch["action"], state["proprioception"]
        ).clamp(-1.0, 1.0)
        reference_chunk = (
            policy.actor.normalize_action(
                state["reference_action"][:, : policy.predicted_action_dim],
                state["proprioception"],
            ).clamp(-1.0, 1.0)
            if policy.critic_ensemble.action_residual
            else None
        )
        q = policy.critic_ensemble(
            state["rl_token"], state["proprioception"], normalized, reference_chunk
        ).min(dim=0).values.reshape(-1)
    reward = batch["reward"].reshape(-1)
    stats = {
        "learner/q_mean": float(q.mean()),
        "learner/q_std": float(q.std()),
    }
    rewarded = reward > 0
    if bool(rewarded.any()) and bool((~rewarded).any()):
        high = float(q[rewarded].mean())
        low = float(q[~rewarded].mean())
        stats["learner/q_reward1_mean"] = high
        stats["learner/q_reward0_mean"] = low
        stats["learner/q_gap"] = high - low
    return stats


def _train_step(
    policy: RLTPolicy,
    batch: dict[str, Any],
    *,
    step: int,
    config: RLTConfig,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
) -> tuple[float, float | None]:
    state = batch["state"]
    next_state = batch["next_state"]
    critic_loss: Tensor | None = None
    for _ in range(config.utd_ratio):
        critic_loss = policy.compute_loss_critic(
            state["rl_token"],
            state["proprioception"],
            batch["action"],
            batch["reward"],
            next_state["rl_token"],
            next_state["proprioception"],
            next_state["reference_action"],
            batch["done"],
            reference_action=state["reference_action"],
        )
        critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            policy.critic_ensemble.parameters(), config.grad_clip_norm
        )
        critic_optimizer.step()
        policy.update_target_networks()

    actor_value: float | None = None
    if step % config.policy_update_freq == 0:
        actor_loss = policy.compute_loss_actor(
            state["rl_token"],
            state["proprioception"],
            state["reference_action"],
        )
        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            policy.actor.parameters(), config.grad_clip_norm
        )
        actor_optimizer.step()
        actor_value = float(actor_loss.detach())
    assert critic_loss is not None
    return float(critic_loss.detach()), actor_value


def _train_step_dsrl(
    policy: Any,
    batch: dict[str, Any],
    *,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    temperature_optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    """One SAC gradient step on one batch: critics, target critics, actor,
    temperature."""
    state = batch["state"]
    next_state = batch["next_state"]
    critic_loss = policy.compute_loss_critic(
        state["rl_token"],
        state["proprioception"],
        batch["action"],
        batch["reward"],
        next_state["rl_token"],
        next_state["proprioception"],
        next_state["reference_action"],
        batch["done"],
    )
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_optimizer.step()
    policy.update_target_networks()

    actor_loss = policy.compute_loss_actor(
        state["rl_token"], state["proprioception"], state["reference_action"]
    )
    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    actor_optimizer.step()

    temperature_loss = policy.compute_loss_temperature()
    temperature_optimizer.zero_grad(set_to_none=True)
    temperature_loss.backward()
    temperature_optimizer.step()
    return float(critic_loss.detach()), float(actor_loss.detach())


def _checkpoint_path(config: RLTConfig, step: int) -> Path:
    return config.checkpoint_dir / f"step_{step:06d}" / "checkpoint.pt"


def _save_checkpoint(
    policy: RLTPolicy,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    actor_scheduler: torch.optim.lr_scheduler.LRScheduler,
    critic_scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    step: int,
    config: RLTConfig,
    temperature_optimizer: torch.optim.Optimizer | None = None,
) -> Path:
    path = _checkpoint_path(config, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    payload = {
        "step": step,
        "policy_state_dict": policy.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "actor_scheduler": actor_scheduler.state_dict(),
        "critic_scheduler": critic_scheduler.state_dict(),
        "config": without_secrets(asdict(config)),
    }
    if temperature_optimizer is not None:
        payload["temperature_optimizer"] = temperature_optimizer.state_dict()
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def _latest_checkpoint(config: RLTConfig) -> Path | None:
    if not config.checkpoint_dir.exists():
        return None
    candidates = [
        path
        for path in config.checkpoint_dir.glob("step_*/checkpoint.pt")
        if path.parent.name.removeprefix("step_").isdigit()
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda path: int(path.parent.name.removeprefix("step_")),
    )


def _replay_buffer_path(config: RLTConfig) -> Path:
    return config.checkpoint_dir / "online_replay_buffer.pt"


def _save_replay_buffer(
    config: RLTConfig,
    replay_buffer: deque[dict[str, Tensor]],
    *,
    step: int,
) -> Path:
    path = _replay_buffer_path(config)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "schema_version": 1,
            "learner_step": step,
            "transitions": list(replay_buffer),
        },
        temporary,
    )
    temporary.replace(path)
    return path


def _restore_replay_buffer(
    config: RLTConfig,
    *,
    checkpoint_step: int,
) -> list[dict[str, Tensor]]:
    path = _replay_buffer_path(config)
    if not path.exists():
        return []
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"invalid online replay snapshot at {path}")
    replay_step = int(payload.get("learner_step", -1))
    if replay_step > checkpoint_step:
        raise ValueError(
            "online replay snapshot is newer than the learner checkpoint "
            f"({replay_step} > {checkpoint_step})"
        )
    transitions = payload.get("transitions")
    if not isinstance(transitions, list):
        raise ValueError(f"online replay snapshot at {path} lacks transitions")
    print(
        f"[RLT] restored {len(transitions)} online transitions "
        f"from learner step {replay_step}",
        flush=True,
    )
    return transitions


_RESIDUAL_NETWORK_FIELDS = (
    "residual_actor",
    "residual_scale",
    "critic_action_residual",
    "critic_action_gain",
)


def _check_resumed_network(saved: Any, config: RLTConfig, path: Path) -> None:
    """Raise when the residual settings differ from the checkpoint's."""
    network = (saved or {}).get("network") or {}
    defaults = {field.name: field.default for field in fields(RLTNetworkConfig)}
    mismatched = [
        f"{name}={network.get(name, defaults[name])!r} in the checkpoint, "
        f"{getattr(config.network, name)!r} now"
        for name in _RESIDUAL_NETWORK_FIELDS
        if network.get(name, defaults[name]) != getattr(config.network, name)
    ]
    if mismatched:
        raise ValueError(
            f"resume from {path} needs the same residual flags: "
            + "; ".join(mismatched)
        )


def _restore_checkpoint(
    config: RLTConfig,
    policy: RLTPolicy,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    actor_scheduler: torch.optim.lr_scheduler.LRScheduler,
    critic_scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    temperature_optimizer: torch.optim.Optimizer | None = None,
) -> int:
    path = _latest_checkpoint(config)
    if path is None:
        return 0
    payload = torch.load(path, map_location=device, weights_only=False)
    _check_resumed_network(payload.get("config"), config, path)
    policy.load_state_dict(payload["policy_state_dict"])
    actor_optimizer.load_state_dict(payload["actor_optimizer"])
    critic_optimizer.load_state_dict(payload["critic_optimizer"])
    actor_scheduler.load_state_dict(payload["actor_scheduler"])
    critic_scheduler.load_state_dict(payload["critic_scheduler"])
    if temperature_optimizer is not None and "temperature_optimizer" in payload:
        temperature_optimizer.load_state_dict(payload["temperature_optimizer"])
    return int(payload["step"])


def _make_auth_interceptor(grpc: Any, token: str) -> Any:
    class BearerAuthInterceptor(grpc.ServerInterceptor):
        def __init__(self) -> None:
            expected = f"Bearer {token}"

            def deny(_request: Any, context: Any) -> None:
                context.abort(
                    grpc.StatusCode.UNAUTHENTICATED,
                    "invalid or missing auth token",
                )

            self.expected = expected
            self.denied = grpc.unary_unary_rpc_method_handler(deny)

        def intercept_service(
            self, continuation: Any, call_details: Any
        ) -> Any:
            metadata = dict(call_details.invocation_metadata or ())
            if metadata.get("authorization") == self.expected:
                return continuation(call_details)
            return self.denied

    return BearerAuthInterceptor()


def _start_online_transport(
    config: RLTConfig,
    policy: RLTPolicy,
    shutdown_event: Event,
    *,
    learner_step: int,
    on_parameter_stream: Callable[[], None] | None = None,
    partial: bool = False,
) -> tuple[Any, Queue, Queue, Queue]:
    import grpc
    from lerobot.rl.learner_service import LearnerService
    from lerobot.transport import services_pb2_grpc
    from lerobot.transport.utils import MAX_MESSAGE_SIZE

    transition_queue: Queue = Queue()
    interaction_queue: Queue = Queue()
    parameters_queue: Queue = Queue(maxsize=1)
    _push_weights(
        parameters_queue,
        policy,
        learner_step=learner_step,
        partial=partial,
    )
    transport = config.actor_learner

    class Service(LearnerService):
        def StreamParameters(self, request, context):  # noqa: N802
            if on_parameter_stream is not None:
                on_parameter_stream()
            yield from _active_parameter_stream(
                parameters_queue,
                context,
                shutdown_event,
                seconds_between_pushes=transport.policy_parameters_push_frequency,
                queue_get_timeout=transport.queue_get_timeout,
            )

    service = Service(
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        seconds_between_pushes=transport.policy_parameters_push_frequency,
        transition_queue=transition_queue,
        interaction_message_queue=interaction_queue,
        queue_get_timeout=transport.queue_get_timeout,
    )
    interceptors = ()
    if transport.auth_token:
        interceptors = (
            _make_auth_interceptor(grpc, transport.auth_token),
        )
    server = grpc.server(
        ThreadPoolExecutor(max_workers=8),
        options=(
            ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
            ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
            # Accept the actors' keepalive pings on a stream with no data.
            ("grpc.keepalive_permit_without_calls", 1),
            ("grpc.http2.min_ping_interval_without_data_ms", 20_000),
            ("grpc.http2.max_ping_strikes", 0),
        ),
        interceptors=interceptors,
    )
    services_pb2_grpc.add_LearnerServiceServicer_to_server(service, server)
    address = f"{transport.server_bind_host}:{transport.learner_port}"
    if transport.tls_server_cert_path:
        key = Path(transport.tls_server_key_path).read_bytes()
        certificate = Path(transport.tls_server_cert_path).read_bytes()
        credentials = grpc.ssl_server_credentials(((key, certificate),))
        if server.add_secure_port(address, credentials) == 0:
            raise RuntimeError(f"failed to bind TLS learner server to {address}")
    elif server.add_insecure_port(address) == 0:
        raise RuntimeError(f"failed to bind learner server to {address}")
    server.start()
    return server, transition_queue, interaction_queue, parameters_queue


def _active_parameter_stream(
    parameters_queue: Queue,
    context: Any,
    shutdown_event: Event,
    *,
    seconds_between_pushes: float,
    queue_get_timeout: float,
) -> Any:
    """Yield queued parameters to one actor until its stream ends. A payload
    taken while the stream is ending returns to the queue for the next actor."""
    from lerobot.transport import services_pb2
    from lerobot.transport.utils import send_bytes_in_chunks

    last_push = 0.0
    while not shutdown_event.is_set() and context.is_active():
        remaining = seconds_between_pushes - (time.time() - last_push)
        if remaining > 0:
            shutdown_event.wait(min(remaining, 1.0))
            continue
        try:
            buffer = parameters_queue.get(timeout=min(queue_get_timeout, 1.0))
        except Empty:
            continue
        if not context.is_active():
            try:
                parameters_queue.put_nowait(buffer)
            except Full:
                pass
            return
        yield from send_bytes_in_chunks(
            buffer, services_pb2.Parameters, log_prefix="[LEARNER] Sending parameters", silent=True
        )
        last_push = time.time()


def _push_weights(
    parameters_queue: Queue,
    policy: RLTPolicy,
    *,
    learner_step: int,
    partial: bool = False,
) -> None:
    from lerobot.transport.utils import state_to_bytes

    state: dict[str, Any] = {
        "policy": policy.actor_state_bytes(),
        "learner_step": torch.tensor(learner_step, dtype=torch.int64),
    }
    if partial:
        state["partial"] = torch.tensor(True)
    payload = state_to_bytes(state)
    while True:
        try:
            parameters_queue.get_nowait()
        except Empty:
            break
    try:
        parameters_queue.put_nowait(payload)
    except Full:
        # Another producer refilled the slot; the queue is newest-wins.
        try:
            parameters_queue.get_nowait()
        except Empty:
            pass
        parameters_queue.put_nowait(payload)


def _transition_fields(transition: dict[str, Any]) -> dict[str, Tensor]:
    """Convert one wire transition into the tensors the learner batches."""

    def unbatch(value: Any) -> Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
        if tensor.ndim and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        return tensor.contiguous()

    return {
        "rl_token": unbatch(transition["state"]["rl_token"]),
        "proprioception": unbatch(transition["state"]["proprioception"]),
        "reference_action": unbatch(transition["state"]["reference_action"]),
        "action": unbatch(transition["action"]),
        "reward": torch.tensor(
            float(transition["reward"]), dtype=torch.float32
        ),
        "next_rl_token": unbatch(transition["next_state"]["rl_token"]),
        "next_proprioception": unbatch(
            transition["next_state"]["proprioception"]
        ),
        "next_reference_action": unbatch(
            transition["next_state"]["reference_action"]
        ),
        "done": torch.tensor(
            float(bool(transition["done"])), dtype=torch.float32
        ),
        "curriculum_scale": unbatch(
            transition["state"].get(
                "curriculum_scale",
                torch.tensor(0.0),
            )
        ),
        "safety_trip": unbatch(
            transition["state"].get("safety_trip", torch.tensor(0.0))
        ),
    }


def relabel_rlt_safety_trip(
    transitions: list[dict[str, Tensor]], penalty: float
) -> None:
    """Set the terminal reward of a safety-tripped RLT episode to ``-penalty``
    (success 1, failure 0)."""
    if transitions and bool(float(transitions[-1].get("safety_trip", 0.0))):
        transitions[-1]["reward"] = torch.tensor(-float(penalty), dtype=torch.float32)


def _replayed(payload: bytes, seen: set[bytes] | None) -> bool:
    """Whether ``payload`` was already received this session, by its digest;
    ``seen`` collects the digests and is extended here."""
    if seen is None:
        return False
    digest = hashlib.sha1(payload).digest()
    if digest in seen:
        print("[RLT] duplicate transitions ignored", flush=True)
        return True
    seen.add(digest)
    return False


def _drain_online_transitions(
    queue: Queue,
    replay_buffer: deque[dict[str, Tensor]],
    _device: torch.device,
    shutdown_event: Event,
    relabel: Callable[[list[dict[str, Tensor]]], None] | None = None,
    seen: set[bytes] | None = None,
    reward_plugin: Callable[[list[dict[str, Tensor]], dict[str, list[bytes]] | None], None]
    | None = None,
) -> int:
    """Append the queued transitions; each message holds one episode, which
    ``relabel`` may rewrite before it is stored, and then ``reward_plugin``
    with the episode's camera frames (None when the actor sent none).
    Replayed messages (digest in ``seen``) are dropped."""
    from lerobot.transport.utils import bytes_to_transitions

    count = 0
    while not queue.empty() and not shutdown_event.is_set():
        payload = queue.get()
        if _replayed(payload, seen):
            continue
        wire = bytes_to_transitions(payload)
        packed = wire[0]["state"].get("episode_frames") if wire else None
        episode = [_transition_fields(transition) for transition in wire]
        if relabel is not None:
            relabel(episode)
        if reward_plugin is not None:
            from armnet_rlt.episode_frames import unpack_episode_frames

            reward_plugin(episode, unpack_episode_frames(packed) if packed is not None else None)
        replay_buffer.extend(episode)
        count += len(episode)
    return count


def _drain_episode_segments(
    queue: Queue,
    pending: list[dict[str, Tensor]],
    completed: deque[list[dict[str, Tensor]]],
    shutdown_event: Event,
    seen: set[bytes] | None = None,
) -> int:
    """Split the transition stream into finished episodes on ``done``.

    Each list appended to ``completed`` holds one episode; a partial episode
    stays in ``pending``. Replayed messages (digest in ``seen``) are dropped.
    """
    from lerobot.transport.utils import bytes_to_transitions

    count = 0
    while not queue.empty() and not shutdown_event.is_set():
        payload = queue.get()
        if _replayed(payload, seen):
            continue
        for transition in bytes_to_transitions(payload):
            fields = _transition_fields(transition)
            pending.append(fields)
            if bool(fields["done"]):
                completed.append(list(pending))
                pending.clear()
                count += 1
    return count


def _restore_rolling_metrics(
    path: Path, *, window: int
) -> tuple[RollingMetrics, int]:
    rolling = RollingMetrics(window=window)
    records = read_jsonl(path)
    for value in records:
        rolling.add(EpisodeRecord.from_dict(value))
    return rolling, len(records)


def _episode_key(record: EpisodeRecord) -> tuple[str, int] | None:
    """The (actor session, rollout index) pair naming an episode; None for
    records from actors that do not fill in a session id."""
    if not record.session_id:
        return None
    return record.session_id, int(record.session_rollout_index)


def _logged_episode_keys(path: Path) -> set[tuple[str, int]]:
    """The episode keys already in the rollout log."""
    keys: set[tuple[str, int]] = set()
    for value in read_jsonl(path):
        key = _episode_key(EpisodeRecord.from_dict(value))
        if key is not None:
            keys.add(key)
    return keys


def _schedule_online_updates(
    pending_updates: int, new_transitions: int, *, steps_per_transition: int = 1
) -> int:
    """Schedule learner steps per transition: one for RLT, whose step applies
    UTD critic updates itself; ``utd_ratio`` for DSRL, whose step is one
    gradient step."""
    if pending_updates < 0 or new_transitions < 0 or steps_per_transition <= 0:
        raise ValueError("online update counts must be nonnegative")
    return pending_updates + new_transitions * steps_per_transition


def _drain_interactions(
    queue: Queue,
    rolling: RollingMetrics,
    *,
    rollout_log: Path,
    episode_count: int,
    learner_step: int,
    wandb_run: Any = None,
    sink: list[EpisodeRecord] | None = None,
    seen: set[tuple[str, int]] | None = None,
) -> int:
    """Log the episode records waiting in ``queue``; returns how many were new.

    ``seen`` holds the keys of the episodes logged so far and is extended
    here; a record whose key is already in it is dropped.
    """
    from lerobot.transport.utils import bytes_to_python_object

    count = 0
    while not queue.empty():
        message = bytes_to_python_object(queue.get())
        if not isinstance(message, dict):
            raise ValueError("RLT interaction payload must be a dictionary")
        record = EpisodeRecord.from_dict(message)
        key = _episode_key(record)
        if seen is not None and key is not None:
            if key in seen:
                print(
                    f"[RLT] duplicate episode {key[0][:8]}#{key[1]} ignored",
                    flush=True,
                )
                continue
            seen.add(key)
        rolling.add(record)
        if sink is not None:
            sink.append(record)
        count += 1
        global_index = episode_count + count
        derived = rolling.to_dict()
        persisted = {
            **record.to_dict(),
            "global_rollout_index": global_index,
            "learner_step_at_log": learner_step,
            **derived,
        }
        append_jsonl(rollout_log, persisted, sync=True)
        print(
            f"[RLT] rollout={global_index} "
            f"{'SUCCESS' if record.success else record.outcome.upper()} "
            f"duration={record.duration_s:.1f}s "
            f"policy={record.policy_step_start}..{record.policy_step_end} "
            f"success_10={derived.get('rolling/success_rate_10', 0.0):.3f} "
            f"deviation={record.action_deviation_mean:.3f} "
            f"reward={record.shaped_reward:.3f} "
            f"wrong_buttons={record.wrong_button_presses}",
            flush=True,
        )
        if wandb_run is not None:
            wandb_run.log(
                {**episode_metrics(record, global_index), **derived},
                step=learner_step,
            )
    return count


def _append_training_metrics(
    path: Path,
    *,
    phase: str,
    learner_step: int,
    critic_loss: float,
    actor_loss: float,
    online_buffer_size: int,
    pending_updates: int,
    episode_count: int,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    extra: Mapping[str, float] | None = None,
) -> None:
    append_jsonl(
        path,
        {
            "schema_version": 1,
            "timestamp": datetime.now(UTC).isoformat(),
            "phase": phase,
            "learner_step": learner_step,
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "online_buffer_size": online_buffer_size,
            "pending_updates": pending_updates,
            "episodes_total": episode_count,
            "actor_lr": actor_optimizer.param_groups[0]["lr"],
            "critic_lr": critic_optimizer.param_groups[0]["lr"],
            **dict(extra or {}),
        },
    )


def _episode_mark(episode_count: int, every: int, max_episodes: int) -> int:
    """The newest episode count due a checkpoint: a multiple of ``every`` or
    ``max_episodes``, whichever ``episode_count`` reached last; 0 if none."""
    marks = []
    if every > 0 and episode_count >= every:
        marks.append(episode_count // every * every)
    if max_episodes > 0 and episode_count >= max_episodes:
        marks.append(max_episodes)
    return max(marks, default=0)


def run_learner(
    config: RLTConfig,
    *,
    offline_only: bool = False,
    resume: bool = False,
    wandb_project: str | None = None,
    idle_timeout_s: float = 0.0,
    idle_min_episodes: int = 5,
    max_episodes: int = 0,
    checkpoint_every_episodes: int = 0,
    reward_plugin: Callable[[list[dict[str, Tensor]], dict[str, list[bytes]] | None], None]
    | None = None,
) -> dict[str, object]:
    """Pretrain on the demo cache, then learn from actor episodes.

    Training stops at ``config.online_steps`` or, when ``max_episodes`` is set,
    once that many episodes have arrived and been trained on. With
    ``checkpoint_every_episodes``, a checkpoint is saved each time the episode
    count reaches a multiple of it and its updates are done; each one is listed
    in ``episode_checkpoints.jsonl`` as ``{"episodes": E, "step": S}``.
    ``reward_plugin`` rewrites each episode's rewards after the algorithm's own
    relabelling, given the camera frames an ``--episode-frames`` actor sent.
    """
    dsrl = config.algorithm == "dsrl"
    if not config.assets_dir:
        raise ValueError("assets_dir is required")
    has_cache = bool(config.demo_cache_path) and Path(config.demo_cache_path).is_file()
    if not dsrl and not has_cache:
        # Without demonstrations the actor learns from its rollouts alone.
        if config.max_demo_pretrain_steps > 0 or config.online_batch_fraction < 1.0:
            raise ValueError(
                f"no demo cache at {config.demo_cache_path!r}: run with "
                "--pretrain-steps 0 --online-batch-fraction 1.0, or build one "
                "with `rlt-local cache`"
            )

    run_dir = config.checkpoint_dir
    if not resume and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"fresh RLT run directory is not empty: {run_dir}; "
            "use a new run_id or explicitly resume"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    rollout_log = run_dir / "rollout_metrics.jsonl"
    training_log = run_dir / "training_metrics.jsonl"
    episode_checkpoint_log = run_dir / "episode_checkpoints.jsonl"
    saved_episode_marks = (
        {json.loads(line)["episodes"] for line in episode_checkpoint_log.open()}
        if episode_checkpoint_log.is_file()
        else set()
    )

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True

    norm_stats = load_norm_stats(
        config.assets_dir,
        use_delta_actions=config.policy_uses_delta_actions,
        network=config.network,
    )
    if dsrl:
        # DSRL has no demonstration transitions.
        from armnet_rlt.dsrl import DSRLPolicy

        transitions = []
        artifact_summary = _empty_summary(config.network)
        policy: Any = DSRLPolicy(config)
    elif has_cache:
        transitions, artifact_summary = load_transition_cache(
            config.demo_cache_path, config.network
        )
        policy = RLTPolicy(config)
    else:
        transitions = []
        artifact_summary = _empty_summary(config.network)
        policy = RLTPolicy(config)
    policy.set_norm_stats(
        norm_stats, use_quantiles=config.use_quantile_norm
    )
    policy.to(device).train()

    actor_optimizer = torch.optim.Adam(
        policy.actor.parameters(), lr=config.actor_lr
    )
    critic_optimizer = torch.optim.Adam(
        policy.critic_ensemble.parameters(), lr=config.critic_lr
    )
    # The DSRL entropy temperature has its own optimizer.
    temperature_optimizer = (
        torch.optim.Adam(policy.temperature_parameters(), lr=config.dsrl_temperature_lr)
        if dsrl
        else None
    )
    actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        actor_optimizer,
        T_max=max(1, config.max_demo_pretrain_steps),
        eta_min=config.actor_lr_min,
    )
    critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        critic_optimizer,
        T_max=max(1, config.max_demo_pretrain_steps),
        eta_min=config.critic_lr_min,
    )
    completed_steps = (
        _restore_checkpoint(
            config,
            policy,
            actor_optimizer,
            critic_optimizer,
            actor_scheduler,
            critic_scheduler,
            device,
            temperature_optimizer=temperature_optimizer,
        )
        if resume
        else 0
    )
    restored_online_transitions = (
        _restore_replay_buffer(
            config,
            checkpoint_step=completed_steps,
        )
        if resume and completed_steps
        else []
    )
    rolling, episode_count = _restore_rolling_metrics(
        rollout_log, window=config.rolling_window
    )
    seen_episodes = _logged_episode_keys(rollout_log)
    seen_transitions: set[bytes] = set()
    final_actor_loss = 0.0
    final_critic_loss = 0.0
    wandb_run = None
    if wandb_project:
        try:
            import wandb

            # Appends to the W&B run named by ``config.run_id``.
            wandb_run = wandb.init(
                project=wandb_project,
                id=config.run_id,
                resume="allow",
                config=without_secrets(asdict(config)),
                tags=["offline-only" if offline_only else "online"],
            )
        except Exception as exc:
            raise RuntimeError(
                f"WandB logging was requested (project={wandb_project!r}) but "
                f"could not start: {exc}. Check that WANDB_API_KEY reaches the "
                "container, or drop --wandb-project to run without metrics."
            ) from exc
        print(f"[RLT] WandB run: {wandb_run.url}", flush=True)

    # Start the transport before demonstration pretraining so an actor can
    # connect during it.
    server = None
    transition_queue: Queue | None = None
    interaction_queue: Queue | None = None
    parameters_queue: Queue | None = None
    shutdown_event = Event()
    last_push = time.monotonic()
    if (
        not offline_only
        and max(completed_steps, config.max_demo_pretrain_steps)
        < config.online_steps
    ):
        (
            server,
            transition_queue,
            interaction_queue,
            parameters_queue,
        ) = _start_online_transport(
            config,
            policy,
            shutdown_event,
            learner_step=completed_steps,
        )

    while completed_steps < config.max_demo_pretrain_steps:
        batch = _batch_from_transitions(
            transitions, config.batch_size, device
        )
        critic_loss, actor_loss = _train_step(
            policy,
            batch,
            step=completed_steps,
            config=config,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
        )
        final_critic_loss = critic_loss
        if actor_loss is not None:
            final_actor_loss = actor_loss
        completed_steps += 1
        actor_scheduler.step()
        critic_scheduler.step()
        if parameters_queue is not None and (
            time.monotonic() - last_push
            >= config.actor_learner.policy_parameters_push_frequency
        ):
            _push_weights(
                parameters_queue, policy, learner_step=completed_steps
            )
            last_push = time.monotonic()
        if config.log_freq and completed_steps % config.log_freq == 0:
            print(
                f"[RLT] pretrain {completed_steps}/{config.max_demo_pretrain_steps} "
                f"critic={final_critic_loss:.6f} actor={final_actor_loss:.6f}",
                flush=True,
            )
            _append_training_metrics(
                training_log,
                phase="pretrain",
                learner_step=completed_steps,
                critic_loss=final_critic_loss,
                actor_loss=final_actor_loss,
                online_buffer_size=0,
                pending_updates=0,
                episode_count=episode_count,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        **_q_stats(policy, batch),
                        "learner/critic_loss": final_critic_loss,
                        "learner/actor_loss": final_actor_loss,
                        "learner/step": completed_steps,
                    },
                    step=completed_steps,
                )
        if (
            config.save_freq
            and completed_steps % config.save_freq == 0
        ):
            _save_checkpoint(
                policy,
                actor_optimizer,
                critic_optimizer,
                actor_scheduler,
                critic_scheduler,
                step=completed_steps,
                config=config,
                temperature_optimizer=temperature_optimizer,
            )

    online_buffer: deque[dict[str, Tensor]] | None = None
    if not offline_only and completed_steps < config.online_steps:
        assert transition_queue is not None
        assert interaction_queue is not None
        assert parameters_queue is not None
        online_buffer = deque(
            restored_online_transitions,
            maxlen=config.online_buffer_capacity,
        )
        pending_updates = 0
        session_episodes = 0
        last_arrival = time.monotonic()
        if dsrl:
            from armnet_rlt.dsrl import relabel_episode
        try:
            while (
                not shutdown_event.is_set()
                and completed_steps < config.online_steps
            ):
                # Episode checkpoints wait until the updates are done and nothing
                # has arrived for 10 s, so the last episode's transitions are in.
                if (
                    (checkpoint_every_episodes or max_episodes)
                    and pending_updates <= 0
                    and time.monotonic() - last_arrival >= 10.0
                ):
                    mark = _episode_mark(episode_count, checkpoint_every_episodes, max_episodes)
                    if mark and mark not in saved_episode_marks:
                        _save_checkpoint(
                            policy,
                            actor_optimizer,
                            critic_optimizer,
                            actor_scheduler,
                            critic_scheduler,
                            step=completed_steps,
                            config=config,
                            temperature_optimizer=temperature_optimizer,
                        )
                        _save_replay_buffer(config, online_buffer, step=completed_steps)
                        with episode_checkpoint_log.open("a") as handle:
                            handle.write(
                                json.dumps({"episodes": mark, "step": completed_steps}) + "\n"
                            )
                        saved_episode_marks.add(mark)
                        print(
                            f"[RLT] episode checkpoint: {mark} episodes at step "
                            f"{completed_steps}",
                            flush=True,
                        )
                    if max_episodes and episode_count >= max_episodes:
                        print(f"[RLT] reached {max_episodes} episodes; stopping", flush=True)
                        break
                was_empty = not online_buffer
                added = _drain_online_transitions(
                    transition_queue,
                    online_buffer,
                    device,
                    shutdown_event,
                    relabel=(
                        (
                            lambda episode: relabel_episode(
                                episode,
                                chunk_discount=policy.chunk_discount,
                                safety_trip_penalty=config.safety_trip_penalty,
                            )
                        )
                        if dsrl
                        else (
                            (
                                lambda episode: relabel_rlt_safety_trip(
                                    episode, config.safety_trip_penalty
                                )
                            )
                            if config.safety_trip_penalty is not None
                            else None
                        )
                    ),
                    seen=seen_transitions,
                    reward_plugin=reward_plugin,
                )
                if dsrl and added and was_empty and completed_steps == 0:
                    # DSRL trains dsrl_initial_steps on the first episode.
                    pending_updates = config.dsrl_initial_steps
                else:
                    # One RLT learner step already applies utd_ratio critic
                    # updates; a DSRL learner step is one gradient step.
                    pending_updates = _schedule_online_updates(
                        pending_updates,
                        added,
                        steps_per_transition=config.utd_ratio if dsrl else 1,
                    )
                finished = _drain_interactions(
                    interaction_queue,
                    rolling,
                    rollout_log=rollout_log,
                    episode_count=episode_count,
                    learner_step=completed_steps,
                    wandb_run=wandb_run,
                    seen=seen_episodes,
                )
                episode_count += finished
                session_episodes += finished
                now = time.monotonic()
                if added or finished:
                    last_arrival = now
                elif (
                    idle_timeout_s > 0
                    and session_episodes >= idle_min_episodes
                    and now - last_arrival >= idle_timeout_s
                ):
                    print(
                        f"[RLT] no actor traffic for {idle_timeout_s:.0f}s "
                        f"after {session_episodes} episodes; stopping",
                        flush=True,
                    )
                    break
                if (
                    now - last_push
                    >= config.actor_learner.policy_parameters_push_frequency
                ):
                    _push_weights(
                        parameters_queue, policy, learner_step=completed_steps
                    )
                    last_push = now
                if (
                    len(online_buffer)
                    < config.online_step_before_learning
                    or pending_updates <= 0
                ):
                    time.sleep(0.02)
                    continue
                online_batch_size = min(
                    config.batch_size,
                    max(
                        1,
                        round(config.batch_size * config.online_batch_fraction),
                    ),
                )
                online_batch = _batch_from_transitions(
                    list(online_buffer),
                    online_batch_size,
                    device,
                    balance_key=None if dsrl else "curriculum_scale",
                )
                demo_batch_size = (
                    config.batch_size - online_batch_size
                )
                if demo_batch_size:
                    demo_batch = _batch_from_transitions(
                        transitions, demo_batch_size, device
                    )
                    batch = _merge_batches(online_batch, demo_batch)
                else:
                    batch = online_batch
                if dsrl:
                    assert temperature_optimizer is not None
                    critic_loss, actor_loss = _train_step_dsrl(
                        policy,
                        batch,
                        actor_optimizer=actor_optimizer,
                        critic_optimizer=critic_optimizer,
                        temperature_optimizer=temperature_optimizer,
                    )
                else:
                    critic_loss, actor_loss = _train_step(
                        policy,
                        batch,
                        step=completed_steps,
                        config=config,
                        actor_optimizer=actor_optimizer,
                        critic_optimizer=critic_optimizer,
                    )
                final_critic_loss = critic_loss
                if actor_loss is not None:
                    final_actor_loss = actor_loss
                completed_steps += 1
                pending_updates -= 1
                if config.log_freq and completed_steps % config.log_freq == 0:
                    qs = _q_stats(policy, batch)
                    actor_stats = dict(getattr(policy, "last_actor_stats", {}))
                    print(
                        f"[RLT] online step={completed_steps} "
                        f"critic={final_critic_loss:.6f} "
                        f"actor={final_actor_loss:.6f} "
                        f"q={qs['learner/q_mean']:.4f}"
                        f"±{qs['learner/q_std']:.4f} "
                        f"gap={qs.get('learner/q_gap', float('nan')):.4f} "
                        + "".join(
                            f"{name}={value:.4f} " for name, value in actor_stats.items()
                        )
                        + f"online_buffer={len(online_buffer)}",
                        flush=True,
                    )
                    _append_training_metrics(
                        training_log,
                        phase="online",
                        learner_step=completed_steps,
                        critic_loss=final_critic_loss,
                        actor_loss=final_actor_loss,
                        online_buffer_size=len(online_buffer),
                        pending_updates=pending_updates,
                        episode_count=episode_count,
                        actor_optimizer=actor_optimizer,
                        critic_optimizer=critic_optimizer,
                        extra=actor_stats,
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                **_q_stats(policy, batch),
                                **{
                                    f"learner/{name}": value
                                    for name, value in actor_stats.items()
                                },
                                "learner/critic_loss": final_critic_loss,
                                "learner/actor_loss": final_actor_loss,
                                "learner/online_buffer": len(online_buffer),
                                "learner/step": completed_steps,
                                "learner/actor_lr": actor_optimizer.param_groups[0]["lr"],
                                "learner/critic_lr": critic_optimizer.param_groups[0]["lr"],
                            },
                            step=completed_steps,
                        )
                if (
                    config.save_freq
                    and completed_steps % config.save_freq == 0
                ):
                    _save_checkpoint(
                        policy,
                        actor_optimizer,
                        critic_optimizer,
                        actor_scheduler,
                        critic_scheduler,
                        step=completed_steps,
                        config=config,
                        temperature_optimizer=temperature_optimizer,
                    )
                    _save_replay_buffer(
                        config,
                        online_buffer,
                        step=completed_steps,
                    )
        finally:
            shutdown_event.set()
            if server is not None:
                server.stop(5)

    if not math.isfinite(final_actor_loss) or not math.isfinite(
        final_critic_loss
    ):
        raise FloatingPointError(
            "learner produced non-finite actor or critic loss"
        )
    checkpoint = _save_checkpoint(
        policy,
        actor_optimizer,
        critic_optimizer,
        actor_scheduler,
        critic_scheduler,
        step=completed_steps,
        config=config,
        temperature_optimizer=temperature_optimizer,
    )
    if online_buffer is not None:
        _save_replay_buffer(
            config,
            online_buffer,
            step=completed_steps,
        )
    summary = _learner_summary(
        artifact_summary,
        completed_steps=completed_steps,
        actor_loss=final_actor_loss,
        critic_loss=final_critic_loss,
        checkpoint=checkpoint,
    )
    summary.update(
        {
            "episodes_total": episode_count,
            "rollout_metrics_path": str(rollout_log),
            "training_metrics_path": str(training_log),
            **rolling.to_dict(),
        }
    )
    if wandb_run is not None:
        wandb_run.summary.update(summary)
        wandb_run.finish()
    return summary


def _empty_summary(network: Any) -> TransitionSummary:
    return TransitionSummary(
        transition_count=0,
        positive_count=0,
        positive_rate=0.0,
        terminal_count=0,
        rl_token_dim=network.rl_token_dim,
        action_dim=network.action_dim,
        proprioception_dim=network.proprioception_dim,
        reference_action_len=network.reference_action_len,
        reference_action_dim=network.reference_action_dim,
        predicted_action_len=network.predicted_action_len,
        predicted_action_dim=network.predicted_action_dim,
    )


def _learner_summary(
    artifacts: TransitionSummary,
    *,
    completed_steps: int,
    actor_loss: float,
    critic_loss: float,
    checkpoint: Path,
) -> dict[str, object]:
    return {
        **artifacts.to_dict(),
        "completed_steps": completed_steps,
        "final_actor_loss": actor_loss,
        "final_critic_loss": critic_loss,
        "checkpoint_path": str(checkpoint),
    }
