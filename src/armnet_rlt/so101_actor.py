"""Production single-arm SO-101 actor for RL algorithms on π0.5.

The cell executes the chunk the policy server returns and records one
transition per chunk; the algorithm's actor runs in the policy server."""

from __future__ import annotations

import logging
import json
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty, Queue
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from armnet_rlt.grpc_util import (
    auth_interceptor as _auth_interceptor,  # noqa: F401 - imported from here by tests
    grpc_channel as _grpc_channel,
)
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
    """The configured delta-action mode, or OpenPI's when unset; raises when they differ."""
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


# Fields a LeRobot OpenCV camera config accepts; other cell camera fields are dropped.
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
    """Build a LeRobot SO-101 follower from the cell's Armnet connector resources."""
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


def parse_action_noise_std(value: Any) -> np.ndarray | None:
    """Per-joint action noise std from one value or one per joint; None when zero."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, str):
        parts = [float(part) for part in value.split(",") if part.strip()]
    elif isinstance(value, (int, float)):
        parts = [float(value)]
    else:
        parts = [float(part) for part in value]
    if len(parts) == 1:
        parts = parts * len(MOTOR_NAMES)
    if len(parts) != len(MOTOR_NAMES):
        raise ValueError(
            f"action_noise_std needs 1 or {len(MOTOR_NAMES)} values, got {len(parts)}"
        )
    std = np.asarray(parts, dtype=np.float64)
    if not np.all(np.isfinite(std)) or np.any(std < 0):
        raise ValueError("action_noise_std must be finite and nonnegative")
    return std if np.any(std > 0) else None


def add_action_noise(
    action: dict[str, float], std: np.ndarray, rng: np.random.Generator
) -> dict[str, float]:
    """``action`` with Gaussian noise of ``std`` added to each joint target."""
    noise = rng.normal(0.0, std)
    noisy = dict(action)
    for index, name in enumerate(MOTOR_NAMES):
        key = f"{name}.pos"
        noisy[key] = float(noisy[key] + noise[index])
    return noisy


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
    # The extras the policy server's actor returned with this chunk.
    extras: dict[str, torch.Tensor] = field(default_factory=dict)
    # JPEG views of the state and next state (algorithms that need images).
    frames: dict[str, np.ndarray] | None = None
    next_frames: dict[str, np.ndarray] | None = None
    # A chunk the safety gate stopped: no next observation, and the rows not
    # executed hold the commanded targets.
    partial: bool = False


@dataclass
class _InFlight:
    """The chunk being executed and the state it was chosen in."""

    rl_token: torch.Tensor
    proprioception: torch.Tensor
    reference: torch.Tensor
    requested: torch.Tensor
    executed: torch.Tensor
    extras: dict[str, torch.Tensor]
    frames: dict[str, np.ndarray] | None


def _serialize_episode(
    transitions: list[_ChunkTransition],
    *,
    reward: float,
    safety_trip: bool = False,
    frames: list[dict[str, np.ndarray]] | None = None,
) -> bytes | None:
    """Serialize one episode's transitions as one learner message.

    The reward goes on the last transition. ``safety_trip`` marks an episode the
    cell's safety gate stopped (a terminal failure for the learner). ``frames``
    (JPEG views per policy call) ride on the first transition as ``episode_frames``.
    Each transition's extras go in its state as ``extra/<name>``; its JPEG views
    as ``image/<view>``, the last transition's next views as ``next_image/<view>``
    (the learner takes the others from the following transition); a partial
    last chunk is flagged ``partial``.
    """
    if not transitions:
        return None
    from lerobot.transport.utils import transitions_to_bytes
    from lerobot.utils.transition import Transition

    tripped = torch.tensor([1.0 if safety_trip else 0.0], dtype=torch.float32)
    labelled = []
    for index, item in enumerate(transitions):
        terminal = index == len(transitions) - 1
        state = {
            "rl_token": item.rl_token.unsqueeze(0),
            "proprioception": item.proprioception.unsqueeze(0),
            "reference_action": item.reference_action.unsqueeze(0),
            "curriculum_scale": item.curriculum_scale.unsqueeze(0),
            "safety_trip": tripped,
        }
        for name, value in item.extras.items():
            state[f"extra/{name}"] = value.unsqueeze(0)
        for view, data in (item.frames or {}).items():
            state[f"image/{view}"] = torch.from_numpy(np.ascontiguousarray(data, dtype=np.uint8))
        if terminal:
            for view, data in (item.next_frames or {}).items():
                state[f"next_image/{view}"] = torch.from_numpy(
                    np.ascontiguousarray(data, dtype=np.uint8)
                )
            if item.partial:
                state["partial"] = torch.tensor([1.0], dtype=torch.float32)
        labelled.append(
            Transition(
                state=state,
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
    if frames:
        from armnet_rlt.episode_frames import pack_episode_frames, spread

        labelled[0]["state"]["episode_frames"] = torch.from_numpy(pack_episode_frames(spread(frames)))
    return transitions_to_bytes(labelled)


def _jpeg_views(images: dict[str, Any], size: tuple[int, int] | None = None) -> dict[str, np.ndarray]:
    """JPEG of each image, resized to ``size`` (height, width) when given."""
    from armnet_rlt.remote_policy import encode_image, resize_for_model

    views = {}
    for name, image in images.items():
        image = np.asarray(image, dtype=np.uint8)
        if size is not None:
            image = resize_for_model(image, *size)
        views[name] = encode_image(image, "jpeg", 90)
    return views


def _encode_frame_views(raw_obs: dict[str, Any], width: int = 426, height: int = 240) -> dict[str, np.ndarray]:
    """The observation's top, front and wrist images, resized to ``width`` x
    ``height`` and JPEG-encoded."""
    import cv2

    from armnet_rlt.remote_policy import encode_image

    views = {}
    for name in ("top", "front", "wrist"):
        image = raw_obs.get(name)
        if image is None:
            continue
        image = np.asarray(image, dtype=np.uint8)
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        views[name] = encode_image(image, "jpeg", jpeg_quality=85)
    return views


def _is_safety_trip(exc: BaseException) -> bool:
    """Whether the cell's robot server stopped the arm on a current trip; the
    connection is severed afterwards."""
    return type(exc).__name__ == "RemoteCallError" and "tripped" in str(exc).lower()


def _is_unsafe_command(exc: BaseException) -> bool:
    """Whether the edge rejected the commanded motion as unsafe (speed,
    oscillation); it refuses motion until ``Cell.reset(unlock=True)``."""
    try:
        from armnet_runtime.context import is_command_safety_violation
    except ImportError:
        return False
    return bool(is_command_safety_violation(exc))


def _serialize_interaction(record: EpisodeRecord) -> bytes:
    from lerobot.transport.utils import python_object_to_bytes

    return python_object_to_bytes(record.to_dict())


class LearnerTransport:
    """Bidirectional LeRobot gRPC streams used by the online learner."""

    def __init__(self, cfg: Any, *, subscribe_parameters: bool = True) -> None:
        self._cfg = cfg
        self._subscribe_parameters = subscribe_parameters
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
        if self._subscribe_parameters:
            self._threads.insert(
                0, threading.Thread(target=self._receive_parameters, daemon=True)
            )
        for thread in self._threads:
            thread.start()

    @property
    def failed(self) -> bool:
        """Whether a stream to the learner ended with an error."""
        return self._shutdown.is_set()

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
            "actor_meta": self._artifact.get("actor_meta"),
        }

    def latest_parameters(self) -> None:
        return None

    def send_episode(
        self, transitions: bytes | None, interaction: bytes
    ) -> None:
        del transitions, interaction

    def close(self) -> None:
        return None


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
        retries=False,
        keepalive=True,
    )
    stub = services_pb2_grpc.LearnerServiceStub(channel)
    stub.Ready(services_pb2.Empty(), timeout=30)
    _log_learner_rtt(stub, services_pb2.Empty())
    return channel, stub


def _make_policy(cfg: Any) -> Any:
    """Load the OpenPI policy locally, or connect to a remote policy server."""
    host = str(_field(cfg, "remote_policy_host", "") or "")
    if not host:
        from armnet_rlt.policy_actor import AlgorithmPolicy

        # π0.5 on the cell; the actor the learner sends runs next to it.
        device = str(_field(cfg, "device", "cuda"))
        return AlgorithmPolicy(
            OpenPIRLTPolicy.from_checkpoint(
                str(cfg.checkpoint_dir),
                str(cfg.config_name),
                default_prompt=str(_field(cfg, "language_instruction", "") or ""),
                rl_token_dim=2048,
            ),
            device=device if device != "cuda" or torch.cuda.is_available() else "cpu",
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
    """Print min/p50/p95/max round-trip ms of ``samples`` Ready calls to the learner."""
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


class _LearnerActorState:
    """The newest actor weights and ``actor_meta`` from the learner (or a
    frozen checkpoint), kept for a policy server that does not take them from
    the learner itself."""

    def __init__(self) -> None:
        self.weights: dict[str, Any] = {}
        self.meta: str | None = None

    def load_state_dict(self, weights: dict[str, Any], strict: bool = True) -> Any:
        if strict:
            self.weights = dict(weights)
        else:
            self.weights.update(weights)
        return SimpleNamespace(unexpected_keys=[], missing_keys=[])


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
    meta = state.get("actor_meta") if isinstance(state, dict) else None
    if meta and hasattr(actor, "meta"):
        actor.meta = str(meta)
    raw_step = state.get("learner_step") if isinstance(state, dict) else None
    if raw_step is None:
        return None
    if hasattr(raw_step, "item"):
        raw_step = raw_step.item()
    return int(raw_step)


def _actor_source(policy: Any) -> str:
    source = getattr(policy, "actor_source", None)
    if source:
        return str(source)
    return "learner" if getattr(policy, "steering_from_learner", False) else "client"


def _sync_actor(policy: Any, actor: _LearnerActorState, *, learner_step: int | None) -> None:
    """Send the learner's actor to the policy, unless the policy server takes
    it from the learner or a checkpoint itself."""
    from armnet_rlt.algorithm import actor_payload

    source = _actor_source(policy)
    if source != "client":
        if not getattr(policy, "_actor_feed_logged", False):
            print(
                f"[rlt] policy server takes its actor from the {source}; "
                "not uploading it",
                flush=True,
            )
            policy._actor_feed_logged = True
        return
    if not actor.weights or not actor.meta:
        raise RuntimeError(
            "the learner sent no actor_meta, so its actor cannot be forwarded "
            "to the policy server; start the policy server with --steering-from-learner"
        )
    policy.set_steering(actor_payload(actor.weights, actor.meta, learner_step=learner_step))


def _await_learner_round(
    transport: Any,
    actor: Any,
    current_step: int | None,
    *,
    steps_per_round: int,
    timeout_s: float,
) -> tuple[int | None, int]:
    """Load learner updates until the learner is ``steps_per_round`` steps past
    ``current_step``.

    Returns the loaded step (the latest one on timeout) and the number of
    payloads applied.
    """
    deadline = time.monotonic() + timeout_s
    reloads = 0
    latest = current_step
    target = (current_step or 0) + max(1, steps_per_round)
    update = transport.latest_parameters()
    if update is not None:
        loaded = _load_parameters(actor, update)
        reloads += 1
        if loaded is not None:
            latest = loaded
            if loaded >= target:
                return loaded, reloads
    while time.monotonic() < deadline:
        if getattr(transport, "failed", False):
            raise ConnectionError("lost the learner connection during a round wait")
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


def call_while_recording(call: Any, period_s: float, record_wait: Any) -> Any:
    """Run ``call`` on a worker thread, calling ``record_wait`` every ``period_s``
    until it returns. Returns its result or raises its exception."""
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["result"] = call()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the loop's thread
            box["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(period_s)
    while worker.is_alive():
        record_wait()
        worker.join(period_s)
    if "error" in box:
        raise box["error"]
    return box["result"]


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
    """Write one frame to the eval dataset and the telemetry sidecar.

    Frames are not evenly spaced, so each frame's time since the episode's first
    frame is appended to ``frame_writer.frame_times`` (saved as
    meta/armnet_frame_times.jsonl).
    """
    if frame_writer is None:
        return
    now = time.perf_counter()
    if frame_index == 0 or getattr(frame_writer, "episode_start", None) is None:
        frame_writer.episode_start = now
        frame_writer.frame_times = []
    frame_writer.frame_times.append(round(now - frame_writer.episode_start, 4))
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
    episode_labels: list[dict[str, Any]] | None = None,
    episode_frame_times: list[list[float]] | None = None,
) -> str | None:
    """Finalize the eval dataset and push it unless ``no_push_to_hub`` is set
    (public with ``record_public``). Returns the dataset URL when pushed."""
    if eval_dataset is None or frame_writer is None or eval_repo_id is None:
        return None
    from armnet_rlt.dataset_recording import finish_dataset

    args = getattr(ctx, "args", {})
    return finish_dataset(
        ctx,
        eval_dataset,
        frame_writer,
        repo_id=eval_repo_id,
        push_to_hub=not bool(_field(args, "no_push_to_hub", False)),
        telemetry=telemetry,
        public=as_bool(_field(args, "record_public", False)),
        episode_labels=episode_labels,
        episode_frame_times=episode_frame_times,
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
    """Run chunk-level online rollouts of an RL algorithm on one Armnet SO-101
    cell until the operator stops; returns the job result dict.

    The policy (a remote policy server, or π0.5 with the actor on the cell)
    returns each chunk to execute; the cell records one transition per chunk
    with the policy's extras and sends each episode to the learner."""
    network = cfg.network
    action_dim = int(network.action_dim)
    reference_len = int(network.reference_action_len)
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
        # The learner's actor weights: the policy server runs them; the cell
        # forwards them when the server does not take them from the learner.
        actor = _LearnerActorState()
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
    action_noise_std = parse_action_noise_std(_field(cfg, "action_noise_std", None))
    action_noise_rng = np.random.default_rng(int(_field(cfg, "action_noise_seed", 0)))
    if action_noise_std is not None:
        print(f"[RLT] action noise std {action_noise_std.tolist()}", flush=True)
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
    episode_labels: list[dict[str, Any]] = []
    episode_frame_times: list[list[float]] = []
    transitions: list[_ChunkTransition] = []
    # JPEG views at each policy call, sent with the episode for a reward plugin.
    send_frames = bool(_field(cfg, "episode_frames", False))
    episode_frames: list[dict[str, np.ndarray]] = []
    previous: _InFlight | None = None
    chunk: torch.Tensor | None = None
    chunk_index = 0
    current_policy_step: int | None = None
    episode_policy_step_start: int | None = None
    policy_reload_count = 0
    episode_started_at = ""
    policy_variant = "online_rlt"
    episode_curriculum_scale = 0.0
    command_ticks = 0
    # Frames written this episode, and the last tick's (requested, actual) action: frames
    # recorded while the loop waits for an action chunk repeat it.
    recording = {"frames": 0, "last_action": None}
    hardware_clamp_steps = 0
    deviation_sum = torch.zeros(action_dim, dtype=torch.float64)
    deviation_count = 0
    deviation_max = 0.0
    rerun_rollout_index = 0
    rerun_variant = "online_rlt"
    safety_trip: BaseException | None = None
    # An unsafe-command rejection, which the next reset unlocks.
    safety_trip_unlockable = False

    def abort_episode_on_trip(exc: BaseException) -> None:
        # The episode ends as a failure. The in-flight chunk is sent flagged
        # partial, its state as its next state and the commanded targets in
        # the rows not executed; the learner keeps it only for an algorithm
        # with keeps_partial_chunks.
        if previous is not None:
            executed = torch.where(
                torch.isfinite(previous.executed), previous.executed, previous.requested
            )
            transitions.append(
                _ChunkTransition(
                    rl_token=previous.rl_token.cpu(),
                    proprioception=previous.proprioception.cpu(),
                    reference_action=previous.reference.flatten().cpu(),
                    curriculum_scale=torch.tensor(
                        episode_curriculum_scale, dtype=torch.float32
                    ),
                    action=executed.flatten().cpu(),
                    next_rl_token=previous.rl_token.cpu(),
                    next_proprioception=previous.proprioception.cpu(),
                    next_reference_action=previous.reference.flatten().cpu(),
                    extras=previous.extras,
                    frames=previous.frames,
                    next_frames=previous.frames,
                    partial=True,
                )
            )
        print(
            f"[rlt] safety trip ends the episode as a failure after "
            f"{len(transitions)} transition(s): {exc}",
            flush=True,
        )

    try:
        transport.start()
        current_policy_step = _load_parameters(
            actor, transport.wait_for_initial_parameters()
        )
        _sync_actor(policy, actor, learner_step=current_policy_step)
        operator.start()

        # A server fed by the learner may not have the first actor yet.
        wait_for_actor = getattr(policy, "wait_for_actor", None)
        if _actor_source(policy) != "client" and callable(wait_for_actor):
            if not wait_for_actor(180.0):
                raise RuntimeError("the policy server has no actor after 180 s")
        # Compile the policy server's path before opening the first rollout.
        warm_obs = robot.get_observation()
        policy.act(build_policy_observation(warm_obs, instruction), explore=False)
        policy.reset()

        rounds_done = 0
        next_round_at = (
            episodes_for_round(round_schedule, 1) if round_schedule else 0
        )
        while not operator.should_stop():
            tick = time.perf_counter()
            if (
                safety_trip is not None
                and state_machine.state is EpisodeState.END_EPISODE
            ):
                # The tripped episode has been sent. Any trip other than an
                # unsafe command leaves the robot latched, so the job ends.
                if not safety_trip_unlockable:
                    raise safety_trip
                safety_trip = None
                safety_trip_unlockable = False
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
                if applied:
                    _sync_actor(policy, actor, learner_step=current_policy_step)
                release = getattr(operator, "release_round_retries", None)
                released = release() if callable(release) else 0
                print(
                    f"[rlt round] resuming at learner step "
                    f"{current_policy_step}"
                    + (f"; {released} scene(s) queued for retry" if released else ""),
                    flush=True,
                )
            previous_state = state_machine.state
            if safety_trip is not None:
                state_machine.handle_event(EpisodeEvent.FAIL)
            else:
                for event, _grade in operator.poll_events(state_machine.state):
                    state_machine.handle_event(event)
                    break
            if (
                previous_state is EpisodeState.END_EPISODE
                and state_machine.state is EpisodeState.IN_EPISODE
            ):
                # Keep one actor snapshot for the whole episode.
                update = transport.latest_parameters()
                if update is not None:
                    loaded_step = _load_parameters(actor, update)
                    if loaded_step is not None:
                        current_policy_step = loaded_step
                    policy_reload_count += 1
                    _sync_actor(policy, actor, learner_step=current_policy_step)
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
                if safety_trip is not None:
                    reward = -1.0
                payload = _serialize_episode(
                    transitions,
                    reward=reward,
                    safety_trip=safety_trip is not None,
                    frames=episode_frames if send_frames else None,
                )
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
                        "safety_trip"
                        if safety_trip is not None
                        else outcome.name.lower()
                        if outcome is not None
                        else "unknown"
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

                    if save_episode(
                        ctx,
                        eval_dataset,
                        frame_writer,
                        telemetry=telemetry,
                        ticks=recording["frames"],
                    ):
                        episode_labels.append(rollout_record)
                        episode_frame_times.append(list(getattr(frame_writer, "frame_times", [])))
                if getattr(transport, "failed", False):
                    raise ConnectionError(
                        "the learner connection is gone; this episode and "
                        "later ones would not reach it, so the actor stops"
                    )
                policy.reset()
                transitions = []
                episode_frames = []
                previous = None
                chunk = None
                chunk_index = 0
                episode_policy_step_start = None
                policy_reload_count = 0
                episode_started_at = ""
                command_ticks = 0
                recording = {"frames": 0, "last_action": None}
                hardware_clamp_steps = 0
                deviation_sum.zero_()
                deviation_count = 0
                deviation_max = 0.0
                continue

            if state_machine.state is not EpisodeState.IN_EPISODE:
                time.sleep(min(0.05, period_s))
                continue

            try:
                raw_obs = robot.get_observation()
            except Exception as exc:
                if not _is_safety_trip(exc) and not _is_unsafe_command(exc):
                    raise
                safety_trip = exc
                safety_trip_unlockable = _is_unsafe_command(exc)
                abort_episode_on_trip(exc)
                if safety_trip_unlockable:
                    operator.note_unsafe_command()
                continue
            proprioception = torch.as_tensor(
                [raw_obs[f"{name}.pos"] for name in MOTOR_NAMES],
                dtype=torch.float32,
                device=device,
            )
            def record_wait() -> None:
                # Frame during the policy request: current cameras and joints with
                # the last tick's action.
                if frame_writer is None or recording["last_action"] is None:
                    return
                try:
                    wait_obs = robot.get_observation()
                except Exception:  # noqa: BLE001 - the loop's next read reports it
                    return
                last_requested, last_actual = recording["last_action"]
                record_rollout_frame(
                    ctx, cfg, frame_writer, telemetry, raw_obs=wait_obs,
                    requested_action=last_requested, actual_action=last_actual,
                    task=instruction if not frozen_eval else f"{instruction} [{policy_variant}]",
                    frame_index=recording["frames"], period_s=period_s,
                    tick=time.perf_counter(), action_write_s=0.0,
                )
                recording["frames"] += 1
                recording["waited"] = True

            if chunk is None or chunk_index >= chunk.shape[0]:
                # A base episode (warmup, or the base half of a paired
                # evaluation) uses the base exploration scale.
                base_variant = policy_variant == "base"
                scale = (
                    (base_exploration_scale if frozen_eval else warmup_exploration_scale)
                    if base_variant
                    else exploration_scale
                )
                policy_obs = build_policy_observation(raw_obs, instruction)
                result = call_while_recording(
                    lambda: policy.act(
                        policy_obs,
                        explore=scale > 0,
                        exploration_scale=scale,
                        exploration_correlation=exploration_correlation,
                        base=base_variant,
                    ),
                    period_s,
                    record_wait,
                )
                if send_frames:
                    episode_frames.append(_encode_frame_views(raw_obs))
                frames = None
                if result.needs_images:
                    frames = (
                        result.frames
                        if result.frames is not None
                        else _jpeg_views(policy_obs.images, getattr(policy, "image_size", None))
                    )
                token = (
                    torch.from_numpy(np.asarray(result.rl_token)).float().to(device)
                    if result.rl_token is not None
                    else torch.zeros(0, device=device)
                )
                if result.base_chunk is not None:
                    reference_np = np.asarray(result.base_chunk)[:reference_len, :action_dim]
                    if reference_np.shape != (reference_len, action_dim):
                        raise RuntimeError(
                            f"Pi0RL reference chunk has shape {reference_np.shape}, "
                            f"expected {(reference_len, action_dim)}"
                        )
                    reference = torch.from_numpy(reference_np).float().to(device)
                else:
                    reference = torch.zeros((0, action_dim), device=device)
                if previous is not None:
                    if not torch.isfinite(previous.executed).all():
                        raise RuntimeError(
                            "previous action chunk was not fully executed"
                        )
                    transitions.append(
                        _ChunkTransition(
                            rl_token=previous.rl_token.cpu(),
                            proprioception=previous.proprioception.cpu(),
                            reference_action=previous.reference.flatten().cpu(),
                            curriculum_scale=torch.tensor(
                                episode_curriculum_scale,
                                dtype=torch.float32,
                            ),
                            action=previous.executed.flatten().cpu(),
                            next_rl_token=token.cpu(),
                            next_proprioception=proprioception.cpu(),
                            next_reference_action=reference.flatten().cpu(),
                            extras=previous.extras,
                            frames=previous.frames,
                            next_frames=frames,
                        )
                    )
                chunk = torch.from_numpy(np.asarray(result.chunk)).float().to(device)
                if chunk.ndim != 2 or chunk.shape[1] != action_dim or not chunk.shape[0]:
                    raise RuntimeError(
                        f"policy returned a chunk of shape {tuple(chunk.shape)}"
                    )
                if reference.shape[0]:
                    rows = min(chunk.shape[0], reference.shape[0])
                    deviation = (
                        chunk[:rows] - reference[:rows]
                    ).detach().abs().double().cpu()
                    deviation_sum += deviation.sum(dim=0)
                    deviation_count += int(deviation.shape[0])
                    deviation_max = max(
                        deviation_max, float(deviation.max().item())
                    )
                previous = _InFlight(
                    rl_token=token.detach().clone(),
                    proprioception=proprioception.detach().clone(),
                    reference=reference.detach().clone(),
                    requested=chunk.detach().clone(),
                    # The chunk as executed, filled in step by step below.
                    executed=torch.full_like(chunk, torch.nan),
                    extras={
                        name: torch.as_tensor(np.asarray(value)).cpu()
                        for name, value in result.extras.items()
                    },
                    frames=frames,
                )
                chunk_index = 0
                state_machine.episode_ctx.total_chunk_count += 1

            requested_action = robot_action(chunk[chunk_index].cpu().numpy())
            if action_noise_std is not None:
                requested_action = add_action_noise(
                    requested_action, action_noise_std, action_noise_rng
                )
            # raw_obs predates the frames recorded during the policy request;
            # re-read it so frames stay in time order.
            frame_obs = raw_obs
            if recording.pop("waited", False):
                try:
                    frame_obs = robot.get_observation()
                except Exception:  # noqa: BLE001 - keep the observation the policy saw
                    frame_obs = raw_obs
            action_started = time.perf_counter()
            try:
                sent_action = robot.send_action(requested_action)
            except Exception as exc:
                if not _is_safety_trip(exc) and not _is_unsafe_command(exc):
                    raise
                safety_trip = exc
                safety_trip_unlockable = _is_unsafe_command(exc)
                abort_episode_on_trip(exc)
                if safety_trip_unlockable:
                    # The edge returned the arm to rest and refuses motion until
                    # a reset unlocks it.
                    ctx.report_progress(
                        f"rollout {getattr(operator, 'rollout_index', completed + 1)} "
                        "failed on an unsafe command; the next reset will unlock "
                        f"the arm ({type(exc).__name__}: {exc})"
                    )
                    operator.note_unsafe_command()
                continue
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
                raw_obs=frame_obs,
                requested_action=requested_action,
                actual_action=actual_action,
                task=(
                    instruction
                    if not frozen_eval
                    else f"{instruction} [{policy_variant}]"
                ),
                frame_index=recording["frames"],
                period_s=period_s,
                tick=tick,
                action_write_s=action_write_s,
            )
            recording["frames"] += 1
            recording["last_action"] = (requested_action, actual_action)
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
            previous.executed[chunk_index] = torch.as_tensor(
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
                ctx, eval_dataset, frame_writer, telemetry, eval_repo_id, episode_labels,
                episode_frame_times,
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

