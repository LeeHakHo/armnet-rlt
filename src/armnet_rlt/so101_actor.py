"""Production single-arm SO-101 RL-token actor."""

from __future__ import annotations

import logging
import json
import math
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import numpy as np
import torch

from armnet_rlt.metrics import EpisodeRecord
from armnet_rlt.openpi_rlt import (
    MOTOR_NAMES,
    OpenPIRLTPolicy,
    build_policy_observation,
)
from armnet_rlt.operator_control import CellOperatorControl
from armnet_rlt.rerun_view import (
    KIND_FAIL,
    KIND_RUNNING,
    KIND_SUCCESS,
    KIND_TIMEOUT,
    as_bool,
    log_rollout,
)
from armnet_rlt.state_machine import (
    EpisodeEvent,
    EpisodeOutcome,
    EpisodeState,
    RLTStateMachine,
)

log = logging.getLogger(__name__)


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _delta_action_mode(cfg: Any, policy: Any) -> bool:
    """Resolve one delta mode and reject learner/OpenPI disagreements."""

    configured = _field(cfg, "policy_uses_delta_actions", None)
    detected = getattr(policy, "uses_delta_actions", None)
    if configured is None:
        return bool(detected)
    configured = bool(configured)
    if detected is not None and configured != bool(detected):
        raise ValueError(
            "delta-action mismatch: actor/learner config has "
            f"policy_uses_delta_actions={configured}, but OpenPI has "
            f"uses_delta_actions={bool(detected)}"
        )
    return configured


# Cell JSON includes mount blocks that are not LeRobot camera fields. The
# published runtime only turns type=opencv entries into OpenCVCameraConfig, so
# a Pi CSI camera (type=pi) arrives as a dict and SO101FollowerConfig crashes
# looking up width.
_OPENCV_CAMERA_KEYS = {
    "index_or_path",
    "fps",
    "width",
    "height",
    "color_mode",
    "rotation",
    "warmup_s",
    "fourcc",
    "backend",
}


def lerobot_camera_config(config: Any) -> Any:
    """Return a LeRobot camera config, dropping Armnet-only mount fields."""
    if not isinstance(config, dict):
        return config
    from lerobot.cameras.opencv import OpenCVCameraConfig

    kwargs = {
        key: value
        for key, value in config.items()
        if key in _OPENCV_CAMERA_KEYS
    }
    missing = {"width", "height", "fps", "index_or_path"}.difference(kwargs)
    if missing:
        raise ValueError(
            f"camera config is missing {sorted(missing)}; got keys {sorted(config)}"
        )
    return OpenCVCameraConfig(**kwargs)


def make_so101_robot(ctx: Any, cfg: Any) -> Any:
    """Build a LeRobot follower solely from Armnet connector resources."""

    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SO101FollowerConfig

    cell = ctx.cell
    so101 = _field(cfg, "so101")
    robot_port = cell.robot_port
    if not robot_port:
        raise ValueError("ctx.cell.robot_port is empty; an Armnet robot connector is required")
    cameras = {
        name: lerobot_camera_config(config)
        for name, config in dict(ctx.camera_configs).items()
    }
    if "front" not in cameras:
        raise ValueError(
            f"cell cameras {sorted(cameras)} are missing required camera 'front'"
        )
    robot_id = _field(so101, "robot_id", None) or cell.robot_id or "follower_so101"
    calibration_dir = _field(so101, "calibration_dir", None) or cell.calibration_dir
    robot_cfg = SO101FollowerConfig(
        port=robot_port,
        id=robot_id,
        calibration_dir=Path(calibration_dir) if calibration_dir else None,
        cameras=cameras,
        max_relative_target=cell.safety_limit,
    )
    return make_robot_from_config(robot_cfg)


def robot_action(action: np.ndarray) -> dict[str, float]:
    vector = np.asarray(action, dtype=np.float32).reshape(-1)
    if vector.size < len(MOTOR_NAMES):
        raise ValueError(
            f"SO-101 action has {vector.size} dimensions, expected at least 6"
        )
    return {f"{name}.pos": float(vector[i]) for i, name in enumerate(MOTOR_NAMES)}


@dataclass
class _ChunkTransition:
    rl_token: torch.Tensor
    proprioception: torch.Tensor
    reference_action: torch.Tensor
    curriculum_scale: torch.Tensor
    action: torch.Tensor
    next_rl_token: torch.Tensor
    next_proprioception: torch.Tensor
    next_reference_action: torch.Tensor


def _serialize_episode(
    transitions: list[_ChunkTransition],
    *,
    reward: float,
) -> bytes | None:
    if not transitions:
        return None
    from lerobot.transport.utils import transitions_to_bytes
    from lerobot.utils.transition import Transition

    labelled = []
    for index, item in enumerate(transitions):
        terminal = index == len(transitions) - 1
        labelled.append(
            Transition(
                state={
                    "rl_token": item.rl_token.unsqueeze(0),
                    "proprioception": item.proprioception.unsqueeze(0),
                    "reference_action": item.reference_action.unsqueeze(0),
                    "curriculum_scale": item.curriculum_scale.unsqueeze(0),
                },
                action=item.action.unsqueeze(0),
                reward=reward if terminal else 0.0,
                next_state={
                    "rl_token": item.next_rl_token.unsqueeze(0),
                    "proprioception": item.next_proprioception.unsqueeze(0),
                    "reference_action": item.next_reference_action.unsqueeze(0),
                    "curriculum_scale": item.curriculum_scale.unsqueeze(0),
                },
                done=terminal,
                truncated=False,
            )
        )
    return transitions_to_bytes(labelled)


def _serialize_interaction(record: EpisodeRecord) -> bytes:
    from lerobot.transport.utils import python_object_to_bytes

    return python_object_to_bytes(record.to_dict())


class LearnerTransport:
    """Bidirectional LeRobot gRPC streams used by the online learner."""

    def __init__(self, cfg: Any) -> None:
        self._cfg = cfg
        self._shutdown = threading.Event()
        self._parameters: Queue[bytes] = Queue()
        self._transitions: Queue[bytes] = Queue()
        self._interactions: Queue[bytes] = Queue()
        self._channel: Any = None
        self._stub: Any = None
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        self._channel, self._stub = _connect_learner(self._cfg)
        self._threads = [
            threading.Thread(target=self._receive_parameters, daemon=True),
            threading.Thread(
                target=self._send_stream,
                args=("SendTransitions", self._transitions, "Transition"),
                daemon=True,
            ),
            threading.Thread(
                target=self._send_stream,
                args=("SendInteractions", self._interactions, "InteractionMessage"),
                daemon=True,
            ),
        ]
        for thread in self._threads:
            thread.start()

    def wait_for_initial_parameters(self, timeout_s: float = 120.0) -> bytes:
        try:
            return self._parameters.get(timeout=timeout_s)
        except Empty as exc:
            raise ConnectionError(
                f"no RL actor parameters received within {timeout_s:.0f}s"
            ) from exc

    def latest_parameters(self) -> bytes | None:
        latest = None
        while True:
            try:
                latest = self._parameters.get_nowait()
            except Empty:
                return latest

    def send_episode(self, transitions: bytes | None, interaction: bytes) -> None:
        if transitions is not None:
            self._transitions.put(transitions)
        self._interactions.put(interaction)

    def close(self) -> None:
        deadline = time.monotonic() + 10.0
        outbound = (self._transitions, self._interactions)
        while (
            any(queue.unfinished_tasks for queue in outbound)
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        if any(queue.unfinished_tasks for queue in outbound):
            log.warning(
                "timed out flushing learner messages: transitions=%d interactions=%d",
                self._transitions.unfinished_tasks,
                self._interactions.unfinished_tasks,
            )
        else:
            # Give gRPC a short window to deliver its already-consumed chunks.
            time.sleep(0.25)
        self._shutdown.set()
        if self._channel is not None:
            self._channel.close()
        for thread in self._threads:
            thread.join(timeout=1.0)

    def _receive_parameters(self) -> None:
        from lerobot.transport import services_pb2
        from lerobot.transport.utils import receive_bytes_in_chunks

        try:
            receive_bytes_in_chunks(
                self._stub.StreamParameters(services_pb2.Empty()),
                self._parameters,
                self._shutdown,
                log_prefix="[RLT actor] parameters",
            )
        except Exception:
            if not self._shutdown.is_set():
                log.exception("parameter stream failed")
                self._shutdown.set()

    def _send_stream(self, rpc_name: str, queue: Queue, message_name: str) -> None:
        from lerobot.transport import services_pb2
        from lerobot.transport.utils import send_bytes_in_chunks

        message_type = getattr(services_pb2, message_name)

        def messages():
            while not self._shutdown.is_set():
                try:
                    payload = queue.get(timeout=float(_field(self._cfg, "queue_get_timeout", 2.0)))
                except Empty:
                    continue
                try:
                    yield from send_bytes_in_chunks(
                        payload,
                        message_type,
                        log_prefix=f"[RLT actor] {message_name}",
                    )
                finally:
                    queue.task_done()

        try:
            getattr(self._stub, rpc_name)(messages())
        except Exception:
            if not self._shutdown.is_set():
                log.exception("%s stream failed", rpc_name)
                self._shutdown.set()


class FrozenActorTransport:
    """Transport-shaped adapter that serves one immutable actor snapshot."""

    def __init__(self, artifact: dict[str, Any]) -> None:
        self._artifact = artifact

    def start(self) -> None:
        return None

    def wait_for_initial_parameters(
        self, timeout_s: float = 120.0
    ) -> dict[str, Any]:
        del timeout_s
        return {
            "policy": self._artifact["actor_state_dict"],
            "learner_step": self._artifact["learner_step"],
        }

    def latest_parameters(self) -> None:
        return None

    def send_episode(
        self, transitions: bytes | None, interaction: bytes
    ) -> None:
        del transitions, interaction

    def close(self) -> None:
        return None


def _auth_interceptor(metadata: list[tuple[str, str]]) -> Any:
    import grpc

    class Details(grpc.ClientCallDetails):
        def __init__(self, original: Any) -> None:
            self.method = original.method
            self.timeout = original.timeout
            self.metadata = [*(original.metadata or []), *metadata]
            self.credentials = original.credentials
            self.wait_for_ready = getattr(original, "wait_for_ready", None)
            self.compression = getattr(original, "compression", None)

    class Interceptor(
        grpc.UnaryUnaryClientInterceptor,
        grpc.UnaryStreamClientInterceptor,
        grpc.StreamUnaryClientInterceptor,
        grpc.StreamStreamClientInterceptor,
    ):
        def intercept_unary_unary(self, continuation, details, request):
            return continuation(Details(details), request)

        def intercept_unary_stream(self, continuation, details, request):
            return continuation(Details(details), request)

        def intercept_stream_unary(self, continuation, details, iterator):
            return continuation(Details(details), iterator)

        def intercept_stream_stream(self, continuation, details, iterator):
            return continuation(Details(details), iterator)

    return Interceptor()


LEARNER_MAX_MESSAGE_BYTES = 4 * 1024 * 1024
GRPC_RETRY_SERVICE_CONFIG = json.dumps(
    {
        "methodConfig": [
            {
                "name": [{}],
                "retryPolicy": {
                    "maxAttempts": 5,
                    "initialBackoff": "0.1s",
                    "maxBackoff": "2s",
                    "backoffMultiplier": 2,
                    "retryableStatusCodes": ["UNAVAILABLE", "DEADLINE_EXCEEDED"],
                },
            }
        ]
    }
)


def _grpc_channel_options(max_message_bytes: int) -> list[tuple[str, Any]]:
    """lerobot.transport.utils.grpc_channel_options with a configurable size cap."""
    return [
        ("grpc.max_receive_message_length", max_message_bytes),
        ("grpc.max_send_message_length", max_message_bytes),
        ("grpc.enable_retries", 1),
        ("grpc.service_config", GRPC_RETRY_SERVICE_CONFIG),
    ]


def _grpc_channel(
    host: str,
    port: int,
    *,
    use_tls: bool,
    token: str = "",
    server_name: str = "",
    root_cert_path: str = "",
    max_message_bytes: int = LEARNER_MAX_MESSAGE_BYTES,
) -> Any:
    import grpc

    options = _grpc_channel_options(max_message_bytes)
    if server_name:
        options.append(("grpc.ssl_target_name_override", server_name))
    target = f"{host}:{port}"
    if use_tls:
        root = Path(root_cert_path).read_bytes() if root_cert_path else None
        channel = grpc.secure_channel(
            target, grpc.ssl_channel_credentials(root_certificates=root), options
        )
    else:
        channel = grpc.insecure_channel(target, options)
    if token:
        channel = grpc.intercept_channel(
            channel, _auth_interceptor([("authorization", f"Bearer {token}")])
        )
    return channel


def _connect_learner(cfg: Any) -> tuple[Any, Any]:
    from lerobot.transport import services_pb2, services_pb2_grpc

    host = str(_field(cfg, "learner_host", ""))
    port = int(_field(cfg, "learner_port", 443))
    if not host:
        raise ValueError("learner_host is required")
    channel = _grpc_channel(
        host,
        port,
        use_tls=bool(_field(cfg, "use_tls", True)),
        token=str(_field(cfg, "auth_token", "") or ""),
        server_name=str(_field(cfg, "tls_server_name", "") or ""),
        root_cert_path=str(_field(cfg, "tls_root_cert_path", "") or ""),
    )
    stub = services_pb2_grpc.LearnerServiceStub(channel)
    stub.Ready(services_pb2.Empty(), timeout=30)
    _log_learner_rtt(stub, services_pb2.Empty())
    return channel, stub


def _make_policy(cfg: Any) -> Any:
    """Load the OpenPI policy locally, or connect to a remote policy server."""
    host = str(_field(cfg, "remote_policy_host", "") or "")
    if not host:
        return OpenPIRLTPolicy.from_checkpoint(
            str(cfg.checkpoint_dir),
            str(cfg.config_name),
            default_prompt=str(_field(cfg, "language_instruction", "") or ""),
            rl_token_dim=2048,
        )
    from armnet_rlt.remote_policy import MAX_MESSAGE_BYTES, RemoteOpenPIRLTPolicy

    port = int(_field(cfg, "remote_policy_port", 443))
    channel = _grpc_channel(
        host,
        port,
        use_tls=bool(_field(cfg, "remote_policy_use_tls", True)),
        token=str(_field(cfg, "remote_policy_auth_token", "") or ""),
        max_message_bytes=MAX_MESSAGE_BYTES,
    )
    policy = RemoteOpenPIRLTPolicy(
        channel,
        timeout_s=float(_field(cfg, "remote_policy_timeout_s", 5.0)),
        image_codec=str(_field(cfg, "remote_policy_image_codec", "png")),
    )
    print(f"[rlt] remote policy {host}:{port}", flush=True)
    return policy


def _log_learner_rtt(stub: Any, request: Any, samples: int = 20) -> None:
    """Print the round-trip time of empty Ready calls to the learner."""
    rtts = []
    for _ in range(samples):
        start = time.perf_counter()
        try:
            stub.Ready(request, timeout=10)
        except Exception:
            log.warning("learner RTT probe failed", exc_info=True)
            return
        rtts.append((time.perf_counter() - start) * 1000.0)
    rtts.sort()
    print(
        "[rlt rtt] "
        + json.dumps(
            {
                "samples": samples,
                "min_ms": round(rtts[0], 1),
                "p50_ms": round(rtts[len(rtts) // 2], 1),
                "p95_ms": round(rtts[int(len(rtts) * 0.95) - 1], 1),
                "max_ms": round(rtts[-1], 1),
            }
        ),
        flush=True,
    )


def _load_parameters(actor: Any, payload: Any) -> int | None:
    if isinstance(payload, dict):
        state = payload
    else:
        from lerobot.transport.utils import bytes_to_state_dict

        state = bytes_to_state_dict(payload)
    weights = state.get("policy", state)
    if bool(state.get("partial", False)):
        # A subset of the tensors; the rest keep their loaded values.
        result = actor.load_state_dict(weights, strict=False)
        if result.unexpected_keys:
            raise ValueError(
                f"learner sent unknown parameters: {result.unexpected_keys[:5]}"
            )
    else:
        actor.load_state_dict(weights)
    raw_step = state.get("learner_step") if isinstance(state, dict) else None
    if raw_step is None:
        return None
    if hasattr(raw_step, "item"):
        raw_step = raw_step.item()
    return int(raw_step)


def _sync_steering(
    policy: Any, actor: Any, *, network: Any, learner_step: int | None
) -> None:
    """Hand the DSRL actor's current weights to the policy that applies them."""
    from armnet_rlt.dsrl import steering_payload

    policy.set_steering(
        steering_payload(
            actor.state_dict(),
            network=network,
            noise_bound=float(actor.noise_bound),
            noise_steps=int(actor.noise_steps),
            learner_step=learner_step,
        )
    )


def _await_learner_round(
    transport: Any,
    actor: Any,
    current_step: int | None,
    *,
    steps_per_round: int,
    timeout_s: float,
) -> tuple[int | None, int]:
    """Block until the learner has applied ``steps_per_round`` more steps.

    The learner pushes parameters on a timer, so the first payload after a round
    starts already reports a higher step; waiting for any advance would resume
    collection mid-round. The target is taken from the learner's own step at the
    moment the wait begins. Returns the loaded step and how many payloads were
    applied. On timeout the latest step is returned, so a round in which the
    learner had nothing to clone does not stall collection.
    """
    deadline = time.monotonic() + timeout_s
    reloads = 0
    latest = current_step
    update = transport.latest_parameters()
    if update is not None:
        loaded = _load_parameters(actor, update)
        reloads += 1
        if loaded is not None:
            latest = loaded
    target = (latest or 0) + max(1, steps_per_round)
    while time.monotonic() < deadline:
        update = transport.latest_parameters()
        if update is not None:
            loaded = _load_parameters(actor, update)
            reloads += 1
            if loaded is not None:
                latest = loaded
                if loaded >= target:
                    return loaded, reloads
        time.sleep(0.05)
    log.warning(
        "learner reached step %s of the %d expected within %.0fs; continuing "
        "on the parameters already loaded",
        latest,
        target,
        timeout_s,
    )
    return latest, reloads


def _policy_variant(
    *,
    frozen_eval: bool,
    include_base: bool,
    rollout_index: int,
    base_only: bool = False,
    warmup_episodes: int = 0,
) -> str:
    if not frozen_eval:
        if warmup_episodes and rollout_index <= warmup_episodes:
            return "base"
        return "online_rlt"
    if base_only:
        return "base"
    if include_base:
        pair_index = (rollout_index - 1) // 2
        position_in_pair = (rollout_index - 1) % 2
        base_first = pair_index % 2 == 0
        if (position_in_pair == 0) == base_first:
            return "base"
    return "frozen_rlt"


def _paired_eval_summary(
    records: list[dict[str, Any]],
) -> dict[str, float | int] | None:
    if not records or len(records) % 2:
        return None
    counts = {
        "both_success": 0,
        "frozen_rlt_only": 0,
        "base_only": 0,
        "both_fail": 0,
    }
    base_successes = frozen_successes = 0
    for index in range(0, len(records), 2):
        pair = {
            str(record.get("policy_variant")): bool(record.get("success"))
            for record in records[index : index + 2]
        }
        if set(pair) != {"base", "frozen_rlt"}:
            return None
        base = pair["base"]
        frozen = pair["frozen_rlt"]
        base_successes += int(base)
        frozen_successes += int(frozen)
        key = (
            "both_success"
            if base and frozen
            else "frozen_rlt_only"
            if frozen
            else "base_only"
            if base
            else "both_fail"
        )
        counts[key] += 1
    pairs = len(records) // 2
    return {
        "n_pairs": pairs,
        "base_successes": base_successes,
        "frozen_rlt_successes": frozen_successes,
        "base_success_rate": base_successes / pairs,
        "frozen_rlt_success_rate": frozen_successes / pairs,
        "delta_percentage_points": (
            100.0 * (frozen_successes - base_successes) / pairs
        ),
        **counts,
    }


def _shape_button_reward(
    base_reward: float,
    *,
    press_counts: dict[str, int],
    target_button: str,
    curriculum_scale: float,
    penalty_min: float,
    penalty_max: float,
) -> tuple[float, int, float]:
    if not target_button:
        return base_reward, 0, 0.0
    wrong_presses = sum(
        max(0, int(count))
        for name, count in press_counts.items()
        if name != target_button
    )
    level = max(0.0, min(1.0, float(curriculum_scale)))
    penalty = penalty_min + level * (penalty_max - penalty_min)
    shaped = max(0.0, float(base_reward) - wrong_presses * penalty)
    return shaped, wrong_presses, penalty


def _button_press_counts(robot: Any) -> dict[str, int]:
    state = getattr(robot, "state", None)
    if not callable(state):
        return {}
    try:
        counts = getattr(state(), "press_count", {})
    except Exception:
        log.warning("failed to read BusyBox button press counts", exc_info=True)
        return {}
    if not isinstance(counts, dict):
        return {}
    return {
        str(name): max(0, int(count))
        for name, count in counts.items()
    }


def _correlated_noise(
    predicted_len: int,
    action_dim: int,
    correlation: float,
    *,
    device: Any,
    dtype: Any,
) -> torch.Tensor:
    """Gaussian noise whose adjacent timesteps are correlated.

    Independent noise on every point in a 20 Hz action chunk creates command
    acceleration that is absent from demonstrations and can trip the edge
    oscillation gate. This keeps the same marginal standard deviation while
    correlating neighbours.
    """
    innovations = torch.randn((1, predicted_len, action_dim), device=device, dtype=dtype)
    noise = torch.empty_like(innovations)
    noise[:, 0] = innovations[:, 0]
    innovation_scale = math.sqrt(1.0 - correlation**2)
    for index in range(1, predicted_len):
        noise[:, index] = (
            correlation * noise[:, index - 1] + innovation_scale * innovations[:, index]
        )
    return noise


def _jitter_chunk(
    actor: Any,
    chunk: torch.Tensor,
    proprioception: torch.Tensor,
    *,
    predicted_len: int,
    action_dim: int,
    exploration_correlation: float,
    exploration_scale: float,
) -> torch.Tensor:
    """Add exploration noise to an action chunk the actor did not produce.

    Used to explore around the base policy's own chunk, where there is no actor
    output to perturb. The noise is applied in normalized action space so its
    scale matches the actor's.
    """
    with torch.inference_mode():
        normalized = actor.normalize_action(
            chunk.flatten().unsqueeze(0), proprioception.unsqueeze(0)
        )
        noise = _correlated_noise(
            predicted_len,
            action_dim,
            exploration_correlation,
            device=normalized.device,
            dtype=normalized.dtype,
        )
        jittered = (
            normalized
            + noise.flatten(1) * actor.action_std.reshape(1, -1) * exploration_scale
        ).clamp(-1.0, 1.0)
        action = actor.unnormalize_action(jittered, proprioception.unsqueeze(0))
    return action.reshape(predicted_len, action_dim).detach()


def _refine_chunk(
    actor: Any,
    token: torch.Tensor,
    proprioception: torch.Tensor,
    reference: torch.Tensor,
    *,
    predicted_len: int,
    action_dim: int,
    exploration_correlation: float,
    exploration_scale: float = 1.0,
) -> torch.Tensor:
    with torch.inference_mode():
        _, mean = actor(
            token.unsqueeze(0),
            proprioception.unsqueeze(0),
            reference.flatten().unsqueeze(0),
            sample=False,
        )
        # Independent noise on every point in a 20 Hz action chunk creates
        # command acceleration that is absent from demonstrations and can trip
        # the edge oscillation gate. Preserve the same marginal exploration
        # standard deviation while correlating adjacent timesteps.
        noise = _correlated_noise(
            predicted_len,
            action_dim,
            exploration_correlation,
            device=mean.device,
            dtype=mean.dtype,
        )
        normalized = (
            mean
            + noise.flatten(1)
            * actor.action_std.reshape(1, -1)
            * exploration_scale
        ).clamp(-1.0, 1.0)
        action = actor.unnormalize_action(
            normalized, proprioception.unsqueeze(0)
        )
    return action.reshape(predicted_len, action_dim).detach()


def _close_robot_and_cell(robot: Any, cell: Any) -> None:
    try:
        try:
            robot.disconnect()
        except Exception:  # connector may already be severed by a safety trip
            log.warning("robot disconnect failed during cleanup", exc_info=True)
    finally:
        close = getattr(cell, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                log.warning("cell instrumentation cleanup failed", exc_info=True)


def open_rollout_recording(
    ctx: Any,
    cfg: Any,
    robot: Any,
    instruction: str,
    *,
    action_source: str,
) -> tuple[Any, Any, Any, str | None]:
    """Create the eval dataset, its frame writer and the telemetry session.

    Returns ``(None, None, None, None)`` when ``record_dataset`` is off.
    """
    args = getattr(ctx, "args", {})
    if not bool(_field(args, "record_dataset", True)):
        return None, None, None, None
    from armnet_rlt.dataset_recording import (
        DEFAULT_ENCODER_THREADS,
        DEFAULT_STREAMING_ENCODING,
        DEFAULT_VCODEC,
        DatasetFrameWriter,
        create_rlt_eval_dataset,
        dataset_features,
        generate_rlt_eval_repo_id,
    )

    eval_repo_id = str(
        _field(args, "record_dataset_repo_id", "")
        or generate_rlt_eval_repo_id(ctx)
    )
    features = dataset_features(robot)
    ctx.report_progress(f"creating RLT eval dataset {eval_repo_id}")
    eval_dataset = create_rlt_eval_dataset(
        eval_repo_id,
        features,
        int(_field(cfg, "command_fps", 20)),
        streaming_encoding=bool(
            _field(args, "streaming_encoding", DEFAULT_STREAMING_ENCODING)
        ),
        vcodec=str(_field(args, "vcodec", DEFAULT_VCODEC) or DEFAULT_VCODEC),
        encoder_threads=int(
            _field(args, "encoder_threads", DEFAULT_ENCODER_THREADS)
        ),
        num_cameras=len(getattr(robot, "cameras", {})),
    )
    frame_writer = DatasetFrameWriter(
        eval_dataset,
        features=features,
        task=instruction,
    )
    ctx.cell.attach_dataset(eval_dataset.root)
    from armnet_client.robot_telemetry.session import RobotTelemetrySession

    telemetry = RobotTelemetrySession(
        eval_dataset.root,
        context=ctx,
        mode=str(_field(args, "robot_telemetry", "full")),
        motor_layout="single_arm",
        fps=int(_field(cfg, "command_fps", 20)),
        action_source=action_source,
        strict=bool(_field(args, "robot_telemetry_strict", False)),
        max_buffered_rows=100_000,
    )
    return eval_dataset, frame_writer, telemetry, eval_repo_id


def record_rollout_frame(
    ctx: Any,
    cfg: Any,
    frame_writer: Any,
    telemetry: Any,
    *,
    raw_obs: dict[str, Any],
    requested_action: dict[str, float],
    actual_action: dict[str, float],
    task: str,
    frame_index: int,
    period_s: float,
    tick: float,
    action_write_s: float,
) -> None:
    """Write one control tick to the eval dataset and the telemetry sidecar."""
    if frame_writer is None:
        return
    frame_writer.submit_observation(raw_obs, actual_action, task=task)
    ctx.cell.record_frame()
    if telemetry is not None:
        telemetry.submit_frame(
            frame_index=frame_index,
            dataset_timestamp_s=frame_index
            / max(1, int(_field(cfg, "command_fps", 20))),
            action=requested_action,
            client_clipped_action=actual_action,
            observation=raw_obs,
            safety_outcome="accepted",
            loop_period_s=period_s,
            loop_overrun_s=max(0.0, time.perf_counter() - tick - period_s),
            action_to_write_latency_s=action_write_s,
        )


def finish_rollout_recording(
    ctx: Any,
    eval_dataset: Any,
    frame_writer: Any,
    telemetry: Any,
    eval_repo_id: str | None,
) -> str | None:
    """Finalize the eval dataset and push it unless ``no_push_to_hub`` is set."""
    if eval_dataset is None or frame_writer is None or eval_repo_id is None:
        return None
    from armnet_rlt.dataset_recording import finish_dataset

    return finish_dataset(
        ctx,
        eval_dataset,
        frame_writer,
        repo_id=eval_repo_id,
        push_to_hub=not bool(
            _field(getattr(ctx, "args", {}), "no_push_to_hub", False)
        ),
        telemetry=telemetry,
    )


def run_actor(
    ctx: Any,
    cfg: Any,
    *,
    operator: CellOperatorControl | None = None,
    robot: Any = None,
    policy: OpenPIRLTPolicy | None = None,
    transport: LearnerTransport | None = None,
) -> dict[str, Any]:
    """Run bounded, chunk-level online RLT on one Armnet SO-101 cell."""

    network = cfg.network
    action_dim = int(network.action_dim)
    predicted_len = int(network.predicted_action_len)
    reference_len = int(network.reference_action_len)
    dsrl = str(_field(cfg, "algorithm", "rlt")) == "dsrl"
    # DSRL's actor covers the whole base chunk and executes a prefix of it.
    execute_len = int(_field(cfg, "actions_to_execute", predicted_len)) if dsrl else predicted_len
    if not 1 <= execute_len <= reference_len:
        raise ValueError("actions_to_execute must be within the reference chunk")
    if action_dim != 6 or int(network.proprioception_dim) != 6:
        raise ValueError("production SO-101 actor requires 6-D action and proprioception")
    if int(network.rl_token_dim) != 2048:
        raise ValueError("production Pi0RL actor requires a 2048-D RL token")

    device_name = str(_field(cfg, "device", "cuda"))
    device = torch.device(
        device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu"
    )
    seed = int(_field(cfg, "seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    robot = robot or make_so101_robot(ctx, cfg)
    robot = ctx.cell.instrument(robot)
    if not getattr(robot, "is_connected", False):
        try:
            robot.connect(calibrate=False)
        except TypeError:
            robot.connect()

    try:
        if policy is None:
            policy = _make_policy(cfg)

        delta_actions = _delta_action_mode(cfg, policy)
        if dsrl:
            from armnet_rlt.dsrl import DSRLActor

            # The noise actor is evaluated inside the policy server; this copy
            # only receives learner updates and forwards them there.
            actor: Any = DSRLActor(
                network,
                noise_bound=float(_field(cfg, "dsrl_noise_bound", 2.5)),
                noise_steps=int(_field(cfg, "dsrl_noise_steps", 1)),
            )
            actor.set_norm_stats(
                policy.norm_stats,
                use_quantiles=policy.use_quantile_norm,
            )
            actor.eval()
        else:
            from armnet_rlt.networks import RLTActor

            actor = RLTActor(
                network,
                delta_actions=delta_actions,
            )
            actor.set_norm_stats(
                policy.norm_stats,
                use_quantiles=policy.use_quantile_norm,
            )
            # RLTActor samples fixed exploration only in training mode. It has no
            # dropout/batch-norm layers, so this affects exploration only.
            actor.to(device).train()
        log.info("RLT actor delta-action mode: %s", delta_actions)
    except Exception:
        _close_robot_and_cell(robot, ctx.cell)
        raise

    transport_cfg = cfg.actor_learner
    transport = transport or LearnerTransport(transport_cfg)
    operator = operator or CellOperatorControl(ctx.cell)
    state_machine = RLTStateMachine()
    period_s = 1.0 / max(1, int(_field(cfg, "command_fps", 20)))
    timeout_s = float(_field(cfg, "episode_timeout_s", 30.0))
    exploration_correlation = float(
        _field(cfg, "exploration_correlation", 0.85)
    )
    if not 0.0 <= exploration_correlation < 1.0:
        raise ValueError("exploration_correlation must be in [0, 1)")
    exploration_scale = float(_field(cfg, "exploration_scale", 1.0))
    if not math.isfinite(exploration_scale) or exploration_scale < 0:
        raise ValueError("exploration_scale must be finite and nonnegative")
    frozen_eval = bool(_field(cfg, "frozen_eval", False))
    include_base = bool(_field(cfg, "eval_include_base", False))
    collect = bool(_field(cfg, "collect", False))
    if frozen_eval and not collect and exploration_scale != 0:
        raise ValueError("frozen RLT evaluation requires exploration_scale=0")
    from armnet_rlt.config import episodes_for_round, parse_round_schedule

    round_schedule = parse_round_schedule(
        _field(cfg, "episodes_per_round", 0)
    )
    steps_per_round = int(_field(cfg, "steps_per_round", 0))
    if steps_per_round < 0:
        raise ValueError("steps_per_round must be nonnegative")
    if round_schedule and not steps_per_round:
        raise ValueError(
            "steps_per_round is required when episodes_per_round is set; "
            "it must match the learner's value"
        )
    round_wait_timeout_s = float(_field(cfg, "round_wait_timeout_s", 120.0))
    if round_wait_timeout_s <= 0:
        raise ValueError("round_wait_timeout_s must be positive")
    bc_warmup_episodes = int(_field(cfg, "bc_warmup_episodes", 0))
    if bc_warmup_episodes < 0:
        raise ValueError("bc_warmup_episodes must be nonnegative")
    # Noise on the base chunk during warmup; the head path uses exploration_scale.
    warmup_exploration_scale = float(
        _field(cfg, "warmup_exploration_scale", exploration_scale)
    )
    if not math.isfinite(warmup_exploration_scale) or warmup_exploration_scale < 0:
        raise ValueError(
            "warmup_exploration_scale must be finite and nonnegative"
        )
    base_only = bool(_field(cfg, "base_only", False))
    base_exploration_scale = float(_field(cfg, "base_exploration_scale", 0.0))
    if not math.isfinite(base_exploration_scale) or base_exploration_scale < 0:
        raise ValueError("base_exploration_scale must be finite and nonnegative")
    if base_only and not frozen_eval:
        raise ValueError("base_only requires frozen_eval")
    reward_target_button = str(
        _field(cfg, "reward_target_button", "") or ""
    )
    wrong_button_penalty_min = float(
        _field(cfg, "wrong_button_penalty_min", 0.0)
    )
    wrong_button_penalty_max = float(
        _field(cfg, "wrong_button_penalty_max", 0.0)
    )
    instruction = str(
        _field(cfg, "language_instruction", "")
        or getattr(ctx.cell, "language_instruction", "")
        or getattr(ctx, "task", "")
    )
    use_rerun = as_bool(_field(getattr(ctx, "args", {}), "use_rerun", False))
    eval_dataset_url: str | None = None
    eval_dataset, frame_writer, telemetry, eval_repo_id = open_rollout_recording(
        ctx, cfg, robot, instruction, action_source="rlt_policy"
    )

    completed = successes = 0
    session_id = uuid.uuid4().hex
    cell_id = str(getattr(getattr(ctx, "cell", None), "cell_id", "") or "")
    per_rollout: list[dict[str, Any]] = []
    transitions: list[_ChunkTransition] = []
    previous: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    chunk: torch.Tensor | None = None
    chunk_index = 0
    current_policy_step: int | None = None
    episode_policy_step_start: int | None = None
    policy_reload_count = 0
    episode_started_at = ""
    policy_variant = "online_rlt"
    episode_curriculum_scale = 0.0
    command_ticks = 0
    hardware_clamp_steps = 0
    deviation_sum = torch.zeros(action_dim, dtype=torch.float64)
    deviation_count = 0
    deviation_max = 0.0
    rerun_rollout_index = 0
    rerun_variant = "online_rlt"

    try:
        transport.start()
        current_policy_step = _load_parameters(
            actor, transport.wait_for_initial_parameters()
        )
        if dsrl:
            _sync_steering(policy, actor, network=network, learner_step=current_policy_step)
        operator.start()

        # Compile the combined Pi0RL path before opening the first rollout.
        warm_obs = robot.get_observation()
        if dsrl:
            policy.infer(
                build_policy_observation(warm_obs, instruction), exploration_std=0.0
            )
        else:
            policy.infer(build_policy_observation(warm_obs, instruction))

        rounds_done = 0
        next_round_at = (
            episodes_for_round(round_schedule, 1) if round_schedule else 0
        )
        while not operator.should_stop():
            tick = time.perf_counter()
            if (
                round_schedule
                and state_machine.state is EpisodeState.END_EPISODE
                and completed >= next_round_at
            ):
                rounds_done += 1
                next_round_at += episodes_for_round(
                    round_schedule, rounds_done + 1
                )
                print(
                    f"[rlt round] round {rounds_done} boundary at {completed} "
                    f"rollouts; waiting for {steps_per_round} learner steps "
                    f"past {current_policy_step}",
                    flush=True,
                )
                current_policy_step, applied = _await_learner_round(
                    transport,
                    actor,
                    current_policy_step,
                    steps_per_round=steps_per_round,
                    timeout_s=round_wait_timeout_s,
                )
                policy_reload_count += applied
                if dsrl and applied:
                    _sync_steering(
                        policy, actor, network=network, learner_step=current_policy_step
                    )
                release = getattr(operator, "release_round_retries", None)
                released = release() if callable(release) else 0
                print(
                    f"[rlt round] resuming at learner step "
                    f"{current_policy_step}"
                    + (f"; {released} scene(s) queued for retry" if released else ""),
                    flush=True,
                )
            previous_state = state_machine.state
            for event, _grade in operator.poll_events(state_machine.state):
                state_machine.handle_event(event)
                break
            if (
                previous_state is EpisodeState.END_EPISODE
                and state_machine.state is EpisodeState.IN_EPISODE
            ):
                # Keep one immutable actor snapshot for the whole episode.
                # Reloading every action chunk makes the behavior policy
                # non-stationary mid-rollout and can turn rapid learner updates
                # into visible command discontinuities.
                update = transport.latest_parameters()
                if update is not None:
                    loaded_step = _load_parameters(actor, update)
                    if loaded_step is not None:
                        current_policy_step = loaded_step
                    policy_reload_count += 1
                    if dsrl:
                        _sync_steering(
                            policy,
                            actor,
                            network=network,
                            learner_step=current_policy_step,
                        )
                episode_started_at = datetime.now(UTC).isoformat()
                episode_policy_step_start = current_policy_step
                policy_variant = _policy_variant(
                    frozen_eval=frozen_eval,
                    include_base=include_base or base_only,
                    base_only=base_only,
                    warmup_episodes=bc_warmup_episodes,
                    rollout_index=int(
                        getattr(operator, "rollout_index", completed + 1)
                    ),
                )
                variation = getattr(operator, "variation", None)
                curriculum = (
                    variation.get("curriculum", {})
                    if isinstance(variation, dict)
                    else {}
                )
                episode_curriculum_scale = float(
                    curriculum.get("scale", 0.0)
                )
                rerun_rollout_index = int(
                    getattr(operator, "rollout_index", completed + 1)
                )
                rerun_variant = policy_variant
                if telemetry is not None and eval_dataset is not None:
                    telemetry.begin_episode(eval_dataset.num_episodes)
            if (
                state_machine.state is EpisodeState.IN_EPISODE
                and state_machine.episode_duration >= timeout_s
            ):
                state_machine.handle_event(EpisodeEvent.TIMEOUT)

            if (
                previous_state is EpisodeState.IN_EPISODE
                and state_machine.state is EpisodeState.END_EPISODE
            ):
                base_reward = state_machine.episode_reward
                variation = getattr(operator, "variation", None)
                curriculum = (
                    variation.get("curriculum", {})
                    if isinstance(variation, dict)
                    else {}
                )
                press_counts = _button_press_counts(robot)
                reward, wrong_button_presses, wrong_button_penalty = (
                    _shape_button_reward(
                        base_reward,
                        press_counts=press_counts,
                        target_button=reward_target_button,
                        curriculum_scale=float(
                            curriculum.get("scale", 0.0)
                        ),
                        penalty_min=wrong_button_penalty_min,
                        penalty_max=wrong_button_penalty_max,
                    )
                )
                payload = _serialize_episode(transitions, reward=reward)
                outcome = state_machine.episode_ctx.outcome
                mean_by_joint = (
                    deviation_sum / deviation_count
                    if deviation_count
                    else deviation_sum
                )
                record = EpisodeRecord(
                    session_id=session_id,
                    cell_id=cell_id,
                    policy_variant=policy_variant,
                    session_rollout_index=int(
                        getattr(operator, "rollout_index", completed + 1)
                    ),
                    rollout_total=getattr(operator, "rollout_total", None),
                    started_at=episode_started_at,
                    ended_at=datetime.now(UTC).isoformat(),
                    success=outcome is EpisodeOutcome.SUCCESS,
                    duration_s=state_machine.episode_duration,
                    timeout=outcome is EpisodeOutcome.TIMEOUT,
                    outcome=(
                        outcome.name.lower() if outcome is not None else "unknown"
                    ),
                    scored_by=getattr(operator, "scored_by", None),
                    num_chunks=state_machine.episode_ctx.total_chunk_count,
                    num_transitions=len(transitions),
                    exploration_scale=exploration_scale,
                    exploration_correlation=exploration_correlation,
                    policy_step_start=episode_policy_step_start,
                    policy_step_end=current_policy_step,
                    policy_reload_count=policy_reload_count,
                    variation=variation,
                    reset_problems=tuple(
                        getattr(operator, "reset_problems", ())
                    ),
                    shaped_reward=reward,
                    wrong_button_presses=wrong_button_presses,
                    wrong_button_penalty=wrong_button_penalty,
                    command_ticks=command_ticks,
                    hardware_clamp_steps=hardware_clamp_steps,
                    action_deviation_mean=float(mean_by_joint.mean()),
                    action_deviation_max=deviation_max,
                    action_deviation_by_joint=tuple(
                        float(value) for value in mean_by_joint
                    ),
                )
                interaction = _serialize_interaction(record)
                transport.send_episode(payload, interaction)
                rollout_record = record.to_dict()
                per_rollout.append(rollout_record)
                # Emit each rollout label as it finishes, not at job end.
                print(
                    "[rlt rollout] " + json.dumps(rollout_record, default=str),
                    flush=True,
                )
                record_outcome = getattr(operator, "record_outcome", None)
                if callable(record_outcome):
                    record_outcome(outcome is EpisodeOutcome.SUCCESS)
                completed += 1
                successes += int(outcome is EpisodeOutcome.SUCCESS)
                if use_rerun:
                    if outcome is EpisodeOutcome.SUCCESS:
                        rerun_kind = KIND_SUCCESS
                    elif outcome is EpisodeOutcome.TIMEOUT:
                        rerun_kind = KIND_TIMEOUT
                    else:
                        rerun_kind = KIND_FAIL
                    log_rollout(
                        ctx,
                        None,
                        None,
                        rollout_index=rerun_rollout_index,
                        variant=rerun_variant,
                        kind=rerun_kind,
                    )
                operator.on_episode_end(state_machine.episode_ctx.outcome)
                if eval_dataset is not None and frame_writer is not None:
                    from armnet_rlt.dataset_recording import save_episode

                    save_episode(
                        ctx,
                        eval_dataset,
                        frame_writer,
                        telemetry=telemetry,
                        ticks=command_ticks,
                    )
                policy.reset()
                transitions = []
                previous = None
                chunk = None
                chunk_index = 0
                episode_policy_step_start = None
                policy_reload_count = 0
                episode_started_at = ""
                command_ticks = 0
                hardware_clamp_steps = 0
                deviation_sum.zero_()
                deviation_count = 0
                deviation_max = 0.0
                continue

            if state_machine.state is not EpisodeState.IN_EPISODE:
                time.sleep(min(0.05, period_s))
                continue

            raw_obs = robot.get_observation()
            proprioception = torch.as_tensor(
                [raw_obs[f"{name}.pos"] for name in MOTOR_NAMES],
                dtype=torch.float32,
                device=device,
            )
            if chunk is None or chunk_index >= execute_len:
                if dsrl:
                    # The base variant is the unsteered base policy: zero-mean
                    # unit noise. The steered variant explores around the
                    # actor's noise with exploration_scale as its deviation.
                    steer = policy_variant != "base"
                    result = policy.infer(
                        build_policy_observation(raw_obs, instruction),
                        exploration_std=exploration_scale if steer else 1.0,
                        steer=steer,
                    )
                    if result.noise is None:
                        raise RuntimeError("policy server returned no DSRL noise")
                else:
                    result = policy.infer(
                        build_policy_observation(raw_obs, instruction)
                    )
                token = torch.from_numpy(result.rl_token).float().to(device)
                reference_np = result.action_chunk[:reference_len, :action_dim]
                if reference_np.shape != (reference_len, action_dim):
                    raise RuntimeError(
                        f"Pi0RL reference chunk has shape {reference_np.shape}, "
                        f"expected {(reference_len, action_dim)}"
                    )
                reference = torch.from_numpy(reference_np).float().to(device)
                if previous is not None:
                    prev_token, prev_prop, prev_reference, executed = previous
                    if not torch.isfinite(executed).all():
                        raise RuntimeError(
                            "previous action chunk was not fully executed"
                        )
                    transitions.append(
                        _ChunkTransition(
                            rl_token=prev_token.cpu(),
                            proprioception=prev_prop.cpu(),
                            reference_action=prev_reference.flatten().cpu(),
                            curriculum_scale=torch.tensor(
                                episode_curriculum_scale,
                                dtype=torch.float32,
                            ),
                            action=executed.flatten().cpu(),
                            next_rl_token=token.cpu(),
                            next_proprioception=proprioception.cpu(),
                            next_reference_action=reference.flatten().cpu(),
                        )
                    )
                if dsrl:
                    chunk = reference[:execute_len].detach().clone()
                elif policy_variant == "base":
                    chunk = reference[:predicted_len].detach().clone()
                    base_scale = (
                        base_exploration_scale
                        if frozen_eval
                        else warmup_exploration_scale
                    )
                    if base_scale > 0:
                        chunk = _jitter_chunk(
                            actor,
                            chunk,
                            proprioception,
                            predicted_len=predicted_len,
                            action_dim=action_dim,
                            exploration_correlation=exploration_correlation,
                            exploration_scale=base_scale,
                        )
                else:
                    chunk = _refine_chunk(
                        actor,
                        token,
                        proprioception,
                        reference,
                        predicted_len=predicted_len,
                        action_dim=action_dim,
                        exploration_correlation=exploration_correlation,
                        exploration_scale=exploration_scale,
                    )
                deviation = (
                    chunk - reference[: chunk.shape[0]]
                ).detach().abs().double().cpu()
                deviation_sum += deviation.sum(dim=0)
                deviation_count += int(deviation.shape[0])
                deviation_max = max(
                    deviation_max, float(deviation.max().item())
                )
                previous = (
                    token.detach().clone(),
                    proprioception.detach().clone(),
                    reference.detach().clone(),
                    # DSRL's transition action is the noise itself; RLT's is the
                    # chunk as executed, filled in step by step below.
                    torch.from_numpy(result.noise).float().to(device)
                    if dsrl
                    else torch.full_like(chunk, torch.nan),
                )
                chunk_index = 0
                state_machine.episode_ctx.total_chunk_count += 1

            requested_action = robot_action(chunk[chunk_index].cpu().numpy())
            action_started = time.perf_counter()
            sent_action = robot.send_action(requested_action)
            action_write_s = time.perf_counter() - action_started
            actual_action = (
                sent_action if isinstance(sent_action, dict) else requested_action
            )
            if use_rerun:
                log_rollout(
                    ctx,
                    raw_obs,
                    actual_action,
                    rollout_index=rerun_rollout_index,
                    variant=rerun_variant,
                    kind=KIND_RUNNING,
                )
            record_rollout_frame(
                ctx,
                cfg,
                frame_writer,
                telemetry,
                raw_obs=raw_obs,
                requested_action=requested_action,
                actual_action=actual_action,
                task=(
                    instruction
                    if not frozen_eval
                    else f"{instruction} [{policy_variant}]"
                ),
                frame_index=command_ticks,
                period_s=period_s,
                tick=tick,
                action_write_s=action_write_s,
            )
            if any(
                not math.isclose(
                    float(actual_action.get(key, requested_action[key])),
                    float(requested_action[key]),
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
                for key in requested_action
            ):
                hardware_clamp_steps += 1
            assert previous is not None
            if not dsrl:
                previous[3][chunk_index] = torch.as_tensor(
                    [
                        actual_action.get(f"{name}.pos", requested_action[f"{name}.pos"])
                        for name in MOTOR_NAMES
                    ],
                    dtype=chunk.dtype,
                    device=chunk.device,
                )
            chunk_index += 1
            command_ticks += 1
            remaining = period_s - (time.perf_counter() - tick)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        try:
            operator.stop()
            eval_dataset_url = finish_rollout_recording(
                ctx, eval_dataset, frame_writer, telemetry, eval_repo_id
            )
        finally:
            transport.close()
            _close_robot_and_cell(robot, ctx.cell)

    by_variant: dict[str, dict[str, float | int]] = {}
    for variant in sorted(
        {str(record.get("policy_variant", "")) for record in per_rollout}
    ):
        records = [
            record
            for record in per_rollout
            if record.get("policy_variant") == variant
        ]
        variant_successes = sum(bool(record["success"]) for record in records)
        by_variant[variant] = {
            "n_rollouts": len(records),
            "n_success": variant_successes,
            "pc_success": (
                100.0 * variant_successes / len(records) if records else 0.0
            ),
        }

    return {
        "status": "actor_finished",
        "rollouts": completed,
        "successes": successes,
        "eval_dataset_repo_id": eval_repo_id,
        "eval_dataset_url": eval_dataset_url,
        "robot_telemetry": (
            telemetry.result_summary() if telemetry is not None else None
        ),
        "per_rollout": per_rollout,
        "aggregated": {
            "n_rollouts": completed,
            "n_success": successes,
            "pc_success": 100.0 * successes / completed if completed else 0.0,
        },
        "aggregated_by_variant": by_variant,
        "paired_comparison": (
            _paired_eval_summary(per_rollout)
            if frozen_eval and include_base
            else None
        ),
    }

