"""OpenPI RLT inference over gRPC: a policy server and a cell-side client.

Messages are ``np.savez`` archives loaded with ``allow_pickle=False``.
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
    """Encode one HWC uint8 RGB image as a flat uint8 array."""
    if codec == "raw":
        return np.ascontiguousarray(image, dtype=np.uint8)
    if codec not in IMAGE_CODECS:
        raise ValueError(f"image codec must be one of {IMAGE_CODECS}, got {codec!r}")
    import cv2

    params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)] if codec == "jpeg" else []
    ok, encoded = cv2.imencode(
        ".jpg" if codec == "jpeg" else ".png",
        cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2BGR),
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
    import cv2

    image = cv2.imdecode(np.asarray(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"{codec} decoding failed")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_for_model(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize one HWC uint8 image with OpenPI's ResizeImages function."""
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


class PolicyServer:
    """Serve one loaded policy; calls hold a lock."""

    def __init__(self, policy: Any, *, image_size: tuple[int, int]) -> None:
        self._policy = policy
        self._image_size = image_size
        self._lock = threading.Lock()

    def info(self, request: bytes, context: Any) -> bytes:
        meta = {
            "rl_token_dim": int(self._policy.rl_token_dim),
            "use_quantile_norm": bool(self._policy.use_quantile_norm),
            "uses_delta_actions": bool(self._policy.uses_delta_actions),
            "image_size": list(self._image_size),
        }
        return pack(
            {"meta": json.dumps(meta), **pack_norm_stats(self._policy.norm_stats)}
        )

    def infer(self, request: bytes, context: Any) -> bytes:
        started = time.perf_counter()
        arrays = unpack(request)
        codec = str(arrays["codec"])
        images = {
            key.removeprefix("image/"): decode_image(value, codec)
            for key, value in arrays.items()
            if key.startswith("image/")
        }
        observation = PolicyObservation(
            state=np.asarray(arrays["state"], dtype=np.float32),
            images=images,
            prompt=str(arrays["prompt"]),
        )
        options: dict[str, Any] = {}
        if "exploration_std" in arrays:
            options["exploration_std"] = float(arrays["exploration_std"])
            options["steer"] = bool(arrays["steer"])
        with self._lock:
            result = self._policy.infer(observation, **options)
        response: dict[str, Any] = {
            "action_chunk": result.action_chunk,
            "rl_token": result.rl_token,
            "server_ms": (time.perf_counter() - started) * 1000.0,
        }
        if result.noise is not None:
            response["noise"] = result.noise
        if result.steering_step is not None:
            response["steering_step"] = result.steering_step
        return pack(response)

    def set_steering(self, request: bytes, context: Any) -> bytes:
        payload = unpack(request)
        with self._lock:
            step = self._policy.set_steering(payload)
        return pack({"steering_step": -1 if step is None else int(step)})

    def reset(self, request: bytes, context: Any) -> bytes:
        with self._lock:
            self._policy.reset()
        return b""


@contextmanager
def serve(
    policy: Any,
    *,
    bind: str,
    image_size: tuple[int, int],
    auth_token: str = "",
    tls_cert_path: str = "",
    tls_key_path: str = "",
) -> Iterator[Any]:
    """Run a gRPC server for ``policy`` on ``bind`` until the context exits.

    With ``tls_cert_path`` and ``tls_key_path`` the server terminates TLS
    itself, for a raw TCP forwarder in front of it."""
    import grpc

    if bool(tls_cert_path) != bool(tls_key_path):
        raise ValueError("tls_cert_path and tls_key_path must be set together")

    from armnet_rlt.learner import _make_auth_interceptor

    server_impl = PolicyServer(policy, image_size=image_size)
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
    try:
        yield server
    finally:
        server.stop(grace=2)


class RemoteOpenPIRLTPolicy:
    """``OpenPIRLTPolicy`` interface backed by a policy server."""

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
        self.rl_token_dim = int(meta["rl_token_dim"])
        self.use_quantile_norm = bool(meta["use_quantile_norm"])
        self.uses_delta_actions = bool(meta["uses_delta_actions"])
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

    def encode_observation(
        self, observation: PolicyObservation, *, extra: dict[str, Any] | None = None
    ) -> bytes:
        height, width = self._image_size
        arrays: dict[str, Any] = {
            "state": np.asarray(observation.state, dtype=np.float32),
            "prompt": observation.prompt,
            "codec": self._image_codec,
            **(extra or {}),
        }
        for name, image in observation.images.items():
            arrays[f"image/{name}"] = encode_image(
                resize_for_model(image, height, width),
                self._image_codec,
                self._jpeg_quality,
            )
        return pack(arrays)

    def set_steering(self, payload: dict[str, Any]) -> int | None:
        """Send a DSRL actor (see ``dsrl.steering_payload``) to the server."""
        response = unpack(self._call("SetSteering", pack(payload), timeout_s=30.0))
        step = int(response["steering_step"])
        self.steering_step = step if step >= 0 else None
        return self.steering_step

    def infer(
        self,
        observation: PolicyObservation | dict[str, Any],
        *,
        exploration_std: float | None = None,
        steer: bool = True,
    ) -> PolicyOutput:
        if not isinstance(observation, PolicyObservation):
            raise TypeError("RemoteOpenPIRLTPolicy.infer expects a PolicyObservation")
        started = time.perf_counter()
        request = self.encode_observation(
            observation,
            extra=(
                {"exploration_std": float(exploration_std), "steer": bool(steer)}
                if exploration_std is not None
                else None
            ),
        )
        response = unpack(self._call("Infer", request))
        self._latencies_ms.append((time.perf_counter() - started) * 1000.0)
        self._server_ms.append(float(response["server_ms"]))
        self._request_bytes.append(len(request))
        token = np.asarray(response["rl_token"], dtype=np.float32).reshape(-1)
        if token.shape != (self.rl_token_dim,):
            raise RuntimeError(
                f"remote RL token has shape {token.shape}, "
                f"expected ({self.rl_token_dim},)"
            )
        return PolicyOutput(
            action_chunk=np.asarray(response["action_chunk"], dtype=np.float32),
            rl_token=token,
            noise=(
                np.asarray(response["noise"], dtype=np.float32)
                if "noise" in response
                else None
            ),
            steering_step=(
                int(response["steering_step"]) if "steering_step" in response else None
            ),
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
