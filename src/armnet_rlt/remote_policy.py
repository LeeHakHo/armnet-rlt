"""Policy inference over gRPC: a policy server and cell-side clients.

Messages are ``np.savez`` archives (``allow_pickle=False``). An ``Infer`` request
with ``act`` runs the actor; without it, π0.5's own chunk and RL token.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import logging
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

from armnet_rlt.openpi_rlt import PolicyObservation, PolicyOutput
from armnet_rlt.policy_actor import AlgorithmPolicy, ChunkResult

log = logging.getLogger(__name__)

SERVICE = "armnet_rlt.RemotePolicy"
IMAGE_CODECS = ("png", "jpeg", "raw")
MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def pack(arrays: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **{key: np.asarray(value) for key, value in arrays.items()})
    return buffer.getvalue()


def unpack(data: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def encode_image(image: np.ndarray, codec: str, jpeg_quality: int = 90) -> np.ndarray:
    """One HWC uint8 RGB image as a flat uint8 array (OpenCV, else Pillow)."""
    if codec == "raw":
        return np.ascontiguousarray(image, dtype=np.uint8)
    if codec not in IMAGE_CODECS:
        raise ValueError(f"image codec must be one of {IMAGE_CODECS}, got {codec!r}")
    image = np.asarray(image, dtype=np.uint8)
    try:
        import cv2
    except ModuleNotFoundError:
        from PIL import Image

        buffer = io.BytesIO()
        options = {"quality": int(jpeg_quality)} if codec == "jpeg" else {}
        Image.fromarray(image).save(buffer, format="JPEG" if codec == "jpeg" else "PNG", **options)
        return np.frombuffer(buffer.getvalue(), dtype=np.uint8).copy()
    params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)] if codec == "jpeg" else []
    ok, encoded = cv2.imencode(
        ".jpg" if codec == "jpeg" else ".png",
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        params,
    )
    if not ok:
        raise RuntimeError(f"{codec} encoding failed")
    return encoded.reshape(-1)


def decode_image(data: np.ndarray, codec: str) -> np.ndarray:
    if codec == "raw":
        return np.asarray(data, dtype=np.uint8)
    if codec not in IMAGE_CODECS:
        raise ValueError(f"image codec must be one of {IMAGE_CODECS}, got {codec!r}")
    try:
        import cv2
    except ModuleNotFoundError:
        from PIL import Image

        with Image.open(io.BytesIO(np.asarray(data, dtype=np.uint8).tobytes())) as image:
            return np.asarray(image.convert("RGB"))
    image = cv2.imdecode(np.asarray(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"{codec} decoding failed")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_for_model(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize one HWC uint8 image to ``height`` x ``width`` with padding, as OpenPI does."""
    image = np.asarray(image)
    if image.shape[:2] == (height, width) and image.dtype == np.uint8:
        return image
    from openpi_client.image_tools import resize_with_pad

    return np.asarray(resize_with_pad(image, height, width))


def pack_norm_stats(norm_stats: dict[str, Any]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, stats in norm_stats.items():
        for field in ("mean", "std", "q01", "q99"):
            value = getattr(stats, field, None)
            if value is not None:
                arrays[f"norm/{name}/{field}"] = np.asarray(value, dtype=np.float32)
    return arrays


@dataclass(frozen=True)
class RemoteNormStats:
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray | None = None
    q99: np.ndarray | None = None


def unpack_norm_stats(arrays: dict[str, np.ndarray]) -> dict[str, RemoteNormStats]:
    fields: dict[str, dict[str, np.ndarray]] = {}
    for key, value in arrays.items():
        if not key.startswith("norm/"):
            continue
        _, name, field = key.split("/", 2)
        fields.setdefault(name, {})[field] = value
    return {name: RemoteNormStats(**values) for name, values in fields.items()}


class PolicyBackend:
    """What a policy server serves: Info arrays, Infer, an actor install and episode reset."""

    def info(self) -> dict[str, Any]:
        raise NotImplementedError

    def infer(self, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
        raise NotImplementedError

    def set_steering(self, payload: dict[str, Any]) -> int | None:
        raise ValueError("this policy server takes no steering")

    def reset(self) -> None:
        pass


def decode_images(arrays: dict[str, np.ndarray], codec: str) -> dict[str, np.ndarray]:
    """The ``image/<name>`` entries of a request, decoded."""
    return {
        key.removeprefix("image/"): decode_image(value, codec)
        for key, value in arrays.items()
        if key.startswith("image/")
    }


class OpenPIBackend(PolicyBackend):
    """Serve an ``AlgorithmPolicy``, or an ``OpenPIRLTPolicy`` wrapped in one."""

    def __init__(
        self,
        policy: Any,
        *,
        image_size: tuple[int, int] | None = None,
        steering_from_learner: bool = False,
    ) -> None:
        self._runner = (
            policy
            if isinstance(policy, AlgorithmPolicy)
            else AlgorithmPolicy(
                policy, actor_source="learner" if steering_from_learner else "client"
            )
        )
        self._image_size = tuple(image_size or self._runner.image_size)

    def info(self) -> dict[str, Any]:
        runner = self._runner
        base = runner.base
        meta: dict[str, Any] = {
            "policy_type": "openpi" if base is not None else "rl",
            **runner.info(),
            "image_size": list(self._image_size),
            "steering_from_learner": runner.actor_source != "client",
        }
        arrays: dict[str, Any] = {}
        if base is not None:
            meta.update(
                rl_token_dim=int(base.rl_token_dim),
                use_quantile_norm=bool(base.use_quantile_norm),
                uses_delta_actions=bool(base.uses_delta_actions),
            )
            arrays = pack_norm_stats(base.norm_stats)
        return {"meta": json.dumps(meta), **arrays}

    def infer(self, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
        observation = PolicyObservation(
            state=np.asarray(arrays["state"], dtype=np.float32),
            images=decode_images(arrays, str(arrays["codec"])),
            prompt=str(arrays["prompt"]),
        )
        if "act" not in arrays:
            if "exploration_std" in arrays:
                raise ValueError(
                    "this policy server runs the algorithm's actor itself; "
                    "rebuild the actor image"
                )
            if self._runner.base is None:
                raise ValueError("this policy server has no π0.5")
            plain = self._runner.base.infer(observation)
            return {"action_chunk": plain.action_chunk, "rl_token": plain.rl_token}
        result = self._runner.act(
            observation,
            explore=bool(arrays["explore"]),
            exploration_scale=float(arrays["exploration_scale"]),
            exploration_correlation=float(arrays["exploration_correlation"]),
            base=bool(arrays["base"]),
        )
        response: dict[str, Any] = {
            "action_chunk": result.chunk,
            "actor_step": -1 if result.actor_step is None else int(result.actor_step),
            "needs_images": bool(result.needs_images),
        }
        if result.base_chunk is not None:
            response["base_chunk"] = result.base_chunk
        if result.rl_token is not None:
            response["rl_token"] = result.rl_token
        for name, value in result.extras.items():
            response[f"extra/{name}"] = value
        return response

    def set_steering(self, payload: dict[str, Any]) -> int | None:
        return self._runner.install(payload)

    def reset(self) -> None:
        self._runner.reset()


class PolicyServer:
    """gRPC handlers around one backend; calls hold a lock."""

    def __init__(self, backend: PolicyBackend) -> None:
        self._backend = backend
        self._lock = threading.Lock()

    def apply_steering(self, payload: dict[str, Any]) -> int | None:
        """Install a steering payload from this process (the learner feed)."""
        with self._lock:
            return self._backend.set_steering(payload)

    def run_locked(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Run ``fn`` while no request is being served."""
        with self._lock:
            return fn(*args, **kwargs)

    def info(self, request: bytes, context: Any) -> bytes:
        with self._lock:
            return pack(self._backend.info())

    def infer(self, request: bytes, context: Any) -> bytes:
        started = time.perf_counter()
        arrays = unpack(request)
        with self._lock:
            response = self._backend.infer(arrays)
        response["server_ms"] = (time.perf_counter() - started) * 1000.0
        return pack(response)

    def set_steering(self, request: bytes, context: Any) -> bytes:
        payload = unpack(request)
        with self._lock:
            step = self._backend.set_steering(payload)
        return pack({"steering_step": -1 if step is None else int(step)})

    def reset(self, request: bytes, context: Any) -> bytes:
        with self._lock:
            self._backend.reset()
        return b""


@contextmanager
def serve(
    policy: Any,
    *,
    bind: str,
    image_size: tuple[int, int] | None = None,
    auth_token: str = "",
    tls_cert_path: str = "",
    tls_key_path: str = "",
    steering_from_learner: bool = False,
) -> Iterator[Any]:
    """Serve ``policy`` (backend, ``AlgorithmPolicy`` or ``OpenPIRLTPolicy``) on ``bind``.

    The yielded server has ``apply_steering`` and ``run_locked`` for a learner feed."""
    import grpc

    if bool(tls_cert_path) != bool(tls_key_path):
        raise ValueError("tls_cert_path and tls_key_path must be set together")

    from armnet_rlt.learner import _make_auth_interceptor

    if isinstance(policy, PolicyBackend):
        backend = policy
    else:
        if image_size is None and not isinstance(policy, AlgorithmPolicy):
            raise ValueError("image_size is required to serve an OpenPI policy")
        backend = OpenPIBackend(
            policy, image_size=image_size, steering_from_learner=steering_from_learner
        )
    server_impl = PolicyServer(backend)
    handlers = {
        name: grpc.unary_unary_rpc_method_handler(method)
        for name, method in (
            ("Info", server_impl.info),
            ("Infer", server_impl.infer),
            ("SetSteering", server_impl.set_steering),
            ("Reset", server_impl.reset),
        )
    }
    server = grpc.server(
        ThreadPoolExecutor(max_workers=4),
        options=(
            ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
            ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
        ),
        interceptors=(
            (_make_auth_interceptor(grpc, auth_token),) if auth_token else ()
        ),
    )
    server.add_generic_rpc_handlers(
        (grpc.method_handlers_generic_handler(SERVICE, handlers),)
    )
    if tls_cert_path:
        credentials = grpc.ssl_server_credentials(
            ((Path(tls_key_path).read_bytes(), Path(tls_cert_path).read_bytes()),)
        )
        bound = server.add_secure_port(bind, credentials)
    else:
        bound = server.add_insecure_port(bind)
    if bound == 0:
        raise RuntimeError(f"could not bind the policy server to {bind}")
    server.start()
    server.apply_steering = server_impl.apply_steering
    server.run_locked = server_impl.run_locked
    try:
        yield server
    finally:
        server.stop(grace=2)


class RemoteOpenPIRLTPolicy:
    """Client of a policy server that runs π0.5 and an RL algorithm's actor."""

    def __init__(
        self,
        channel: Any,
        *,
        timeout_s: float = 5.0,
        image_codec: str = "png",
        jpeg_quality: int = 90,
        retries: int = 1,
    ) -> None:
        if image_codec not in IMAGE_CODECS:
            raise ValueError(
                f"image codec must be one of {IMAGE_CODECS}, got {image_codec!r}"
            )
        self._channel = channel
        self._timeout_s = timeout_s
        self._image_codec = image_codec
        self._jpeg_quality = jpeg_quality
        self._retries = retries
        self._calls = {
            name: channel.unary_unary(f"/{SERVICE}/{name}")
            for name in ("Info", "Infer", "SetSteering", "Reset")
        }
        self.steering_step: int | None = None
        info = unpack(self._call("Info", b"", timeout_s=30.0))
        meta = json.loads(str(info["meta"]))
        if meta.get("policy_type", "openpi") not in {"openpi", "rl"}:
            raise RuntimeError(
                f"the policy server serves a {meta.get('policy_type')} policy, "
                "not OpenPI"
            )
        self.steering_from_learner = bool(meta.get("steering_from_learner", False))
        self.actor_source = str(
            meta.get("actor_source", "learner" if self.steering_from_learner else "client")
        )
        self.algorithm = str(meta.get("algorithm", ""))
        self.uses_base = bool(meta.get("uses_base", True))
        self.needs_images = bool(meta.get("needs_images", False))
        self.rl_token_dim = int(meta.get("rl_token_dim", 0))
        self.use_quantile_norm = bool(meta.get("use_quantile_norm", True))
        self.uses_delta_actions = meta.get("uses_delta_actions")
        self.norm_stats = unpack_norm_stats(info)
        self._image_size = tuple(int(v) for v in meta["image_size"])
        self._latencies_ms: list[float] = []
        self._server_ms: list[float] = []
        self._request_bytes: list[int] = []

    def _call(self, name: str, request: bytes, *, timeout_s: float | None = None) -> bytes:
        import grpc

        retryable = {grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED}
        for attempt in range(self._retries + 1):
            try:
                return self._calls[name](request, timeout=timeout_s or self._timeout_s)
            except grpc.RpcError as exc:
                if exc.code() not in retryable or attempt == self._retries:
                    raise
                log.warning("remote policy %s failed (%s); retrying", name, exc.code())
        raise AssertionError("unreachable")

    def encode_images(self, observation: PolicyObservation) -> dict[str, np.ndarray]:
        """The observation's images at the server's size, encoded with the codec."""
        height, width = self._image_size
        return {
            name: encode_image(
                resize_for_model(image, height, width),
                self._image_codec,
                self._jpeg_quality,
            )
            for name, image in observation.images.items()
        }

    def encode_observation(
        self,
        observation: PolicyObservation,
        *,
        extra: dict[str, Any] | None = None,
        images: dict[str, np.ndarray] | None = None,
    ) -> bytes:
        arrays: dict[str, Any] = {
            "state": np.asarray(observation.state, dtype=np.float32),
            "prompt": observation.prompt,
            "codec": self._image_codec,
            **(extra or {}),
        }
        encoded = self.encode_images(observation) if images is None else images
        for name, data in encoded.items():
            arrays[f"image/{name}"] = data
        return pack(arrays)

    def set_steering(self, payload: dict[str, Any]) -> int | None:
        """Send an actor (``algorithm.actor_payload``) to the server."""
        response = unpack(self._call("SetSteering", pack(payload), timeout_s=180.0))
        step = int(response["steering_step"])
        self.steering_step = step if step >= 0 else None
        return self.steering_step

    def wait_for_actor(self, timeout_s: float = 120.0) -> bool:
        """Poll the server until it has an actor; False on timeout."""
        deadline = time.monotonic() + timeout_s
        while True:
            meta = json.loads(str(unpack(self._call("Info", b"", timeout_s=30.0))["meta"]))
            if meta.get("actor_ready", True):
                self.needs_images = bool(meta.get("needs_images", self.needs_images))
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(2.0)

    def _send(self, request: bytes) -> dict[str, np.ndarray]:
        started = time.perf_counter()
        response = unpack(self._call("Infer", request))
        self._latencies_ms.append((time.perf_counter() - started) * 1000.0)
        self._server_ms.append(float(response["server_ms"]))
        self._request_bytes.append(len(request))
        return response

    def infer(self, observation: PolicyObservation) -> PolicyOutput:
        """π0.5's own chunk and RL token, without the actor."""
        if not isinstance(observation, PolicyObservation):
            raise TypeError("RemoteOpenPIRLTPolicy.infer expects a PolicyObservation")
        response = self._send(self.encode_observation(observation))
        token = np.asarray(response["rl_token"], dtype=np.float32).reshape(-1)
        if token.shape != (self.rl_token_dim,):
            raise RuntimeError(
                f"remote RL token has shape {token.shape}, "
                f"expected ({self.rl_token_dim},)"
            )
        return PolicyOutput(
            action_chunk=np.asarray(response["action_chunk"], dtype=np.float32),
            rl_token=token,
        )

    def act(
        self,
        observation: PolicyObservation,
        *,
        explore: bool,
        exploration_scale: float = 0.0,
        exploration_correlation: float = 0.85,
        base: bool = False,
    ) -> ChunkResult:
        """The actor's chunk; ``frames`` holds JPEG views when the algorithm needs images."""
        if not isinstance(observation, PolicyObservation):
            raise TypeError("RemoteOpenPIRLTPolicy.act expects a PolicyObservation")
        images = self.encode_images(observation)
        response = self._send(
            self.encode_observation(
                observation,
                images=images,
                extra={
                    "act": True,
                    "explore": bool(explore),
                    "exploration_scale": float(exploration_scale),
                    "exploration_correlation": float(exploration_correlation),
                    "base": bool(base),
                },
            )
        )
        needs_images = bool(response.get("needs_images", False))
        frames = None
        if needs_images:
            if self._image_codec == "jpeg":
                frames = images
            else:
                height, width = self._image_size
                frames = {
                    name: encode_image(resize_for_model(image, height, width), "jpeg", 90)
                    for name, image in observation.images.items()
                }
        token = None
        if "rl_token" in response:
            token = np.asarray(response["rl_token"], dtype=np.float32).reshape(-1)
        step = int(response.get("actor_step", -1))
        return ChunkResult(
            chunk=np.asarray(response["action_chunk"], dtype=np.float32),
            base_chunk=(
                np.asarray(response["base_chunk"], dtype=np.float32)
                if "base_chunk" in response
                else None
            ),
            rl_token=token,
            extras={
                key.removeprefix("extra/"): np.asarray(value)
                for key, value in response.items()
                if key.startswith("extra/")
            },
            actor_step=step if step >= 0 else None,
            frames=frames,
            needs_images=needs_images,
        )

    def latency_summary(self) -> dict[str, float | int]:
        if not self._latencies_ms:
            return {"calls": 0}
        total = sorted(self._latencies_ms)
        return {
            "calls": len(total),
            "p50_ms": round(total[len(total) // 2], 1),
            "p95_ms": round(total[max(0, int(len(total) * 0.95) - 1)], 1),
            "max_ms": round(total[-1], 1),
            "server_p50_ms": round(sorted(self._server_ms)[len(total) // 2], 1),
            "request_kb_mean": round(
                sum(self._request_bytes) / len(self._request_bytes) / 1024, 1
            ),
        }

    def reset(self) -> None:
        summary = self.latency_summary()
        if summary["calls"]:
            print("[rlt remote policy] " + json.dumps(summary), flush=True)
        self._latencies_ms.clear()
        self._server_ms.clear()
        self._request_bytes.clear()
        self._call("Reset", b"")


class RemoteLeRobotPolicy:
    """Client of a policy server that serves a LeRobot policy."""

    def __init__(
        self,
        channel: Any,
        *,
        timeout_s: float = 5.0,
        image_codec: str = "jpeg",
        jpeg_quality: int = 90,
        retries: int = 1,
    ) -> None:
        if image_codec not in IMAGE_CODECS:
            raise ValueError(
                f"image codec must be one of {IMAGE_CODECS}, got {image_codec!r}"
            )
        self._timeout_s = timeout_s
        self._image_codec = image_codec
        self._jpeg_quality = jpeg_quality
        self._retries = retries
        self._calls = {
            name: channel.unary_unary(f"/{SERVICE}/{name}")
            for name in ("Info", "Infer", "Reset")
        }
        self._latencies_ms: list[float] = []
        self._server_ms: list[float] = []
        self._request_bytes: list[int] = []
        self.policy_step: int | None = None
        meta = self.refresh()
        self.lerobot_type = str(meta["lerobot_type"])
        self.chunked = bool(meta["chunked"])
        self.n_action_steps = int(meta["n_action_steps"])
        self.uses_task = bool(meta["uses_task"])
        self.shapes = {
            key: (int(shape[0]), int(shape[1])) for key, shape in meta["shapes"].items()
        }
        self.served_from_learner = bool(meta.get("steering_from_learner", False))

    def _call(self, name: str, request: bytes, *, timeout_s: float | None = None) -> bytes:
        import grpc

        retryable = {grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED}
        for attempt in range(self._retries + 1):
            try:
                return self._calls[name](request, timeout=timeout_s or self._timeout_s)
            except grpc.RpcError as exc:
                if exc.code() not in retryable or attempt == self._retries:
                    raise
                log.warning("remote policy %s failed (%s); retrying", name, exc.code())
        raise AssertionError("unreachable")

    def refresh(self) -> dict[str, Any]:
        """Re-read the server's Info; updates ``policy_step``."""
        meta = json.loads(str(unpack(self._call("Info", b"", timeout_s=30.0))["meta"]))
        if meta.get("policy_type") != "lerobot":
            raise RuntimeError(
                f"the policy server serves a {meta.get('policy_type', 'openpi')} "
                "policy, not a LeRobot policy"
            )
        step = int(meta.get("learner_step", -1))
        self.policy_step = step if step >= 0 else None
        return meta

    def predict(
        self,
        state: np.ndarray,
        images: dict[str, np.ndarray],
        *,
        task: str | None = None,
        exploration_scale: float = 0.0,
        exploration_correlation: float = 0.85,
    ) -> np.ndarray:
        """Actions ``(n, action_dim)`` for an observation with images at the server's ``shapes``."""
        started = time.perf_counter()
        arrays: dict[str, Any] = {
            "state": np.asarray(state, dtype=np.float32),
            "codec": self._image_codec,
            "exploration_scale": float(exploration_scale),
            "exploration_correlation": float(exploration_correlation),
        }
        if task is not None:
            arrays["task"] = task
        for key, image in images.items():
            arrays[f"image/{key}"] = encode_image(image, self._image_codec, self._jpeg_quality)
        request = pack(arrays)
        response = unpack(self._call("Infer", request))
        self._latencies_ms.append((time.perf_counter() - started) * 1000.0)
        self._server_ms.append(float(response["server_ms"]))
        self._request_bytes.append(len(request))
        step = int(response["policy_step"])
        self.policy_step = step if step >= 0 else None
        return np.asarray(response["actions"], dtype=np.float32)

    def latency_summary(self) -> dict[str, float | int]:
        if not self._latencies_ms:
            return {"calls": 0}
        total = sorted(self._latencies_ms)
        return {
            "calls": len(total),
            "p50_ms": round(total[len(total) // 2], 1),
            "p95_ms": round(total[max(0, int(len(total) * 0.95) - 1)], 1),
            "max_ms": round(total[-1], 1),
            "server_p50_ms": round(sorted(self._server_ms)[len(total) // 2], 1),
            "request_kb_mean": round(
                sum(self._request_bytes) / len(self._request_bytes) / 1024, 1
            ),
        }

    def reset(self) -> None:
        summary = self.latency_summary()
        if summary["calls"]:
            print("[lerobot remote policy] " + json.dumps(summary), flush=True)
        self._latencies_ms.clear()
        self._server_ms.clear()
        self._request_bytes.clear()
        self._call("Reset", b"")
