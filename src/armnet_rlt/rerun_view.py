"""Live Rerun view of RLT rollouts.

The cell container only calls ``ctx.log_rerun_data``. Armnet already carries
those packets to the machine that submitted the job. This module turns that
one long stream into one Rerun recording per rollout, then renames the
recording when the rollout ends so success, failure, and timeout are visible
in the recording list.
"""

from __future__ import annotations

import argparse
from typing import Any

INDEX_PATH = "observation.rollout_index"
VARIANT_PATH = "observation.rollout_variant"
KIND_PATH = "observation.rollout_kind"

KIND_RUNNING = 0
KIND_SUCCESS = 1
KIND_FAIL = 2
KIND_TIMEOUT = 3

_SENTINELS = {INDEX_PATH, VARIANT_PATH, KIND_PATH}
_VARIANT_CODES = {"online_rlt": 0, "base": 1, "frozen_rlt": 2}
_VARIANT_NAMES = {code: name for name, code in _VARIANT_CODES.items()}
_KIND_NAMES = {
    KIND_RUNNING: "running",
    KIND_SUCCESS: "success",
    KIND_FAIL: "fail",
    KIND_TIMEOUT: "timeout",
}
_GRPC_PORT = 9876
_GRPC_URL = f"rerun+http://127.0.0.1:{_GRPC_PORT}/proxy"
_APPLICATION_ID = "armnet-rlt"

_installed = False


def as_bool(value: Any, default: bool = False) -> bool:
    """Read a job arg that may already be a bool or a stringified bool."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def add_rerun_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--use-rerun",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Open a live Rerun viewer and record each rollout separately. "
            "Ignored with --detach, because the viewer has to stay connected."
        ),
    )


def rerun_job_enabled(args: argparse.Namespace) -> bool:
    """Whether this submit should both emit and display Rerun data."""
    return as_bool(getattr(args, "use_rerun", True), default=True) and not bool(
        getattr(args, "detach", False)
    )


def variant_code(variant: str) -> int:
    try:
        return _VARIANT_CODES[str(variant)]
    except KeyError as exc:
        raise ValueError(f"unknown RLT policy variant {variant!r}") from exc


def recording_id(rollout_index: int, variant: int) -> str:
    name = _VARIANT_NAMES.get(int(variant), "rlt")
    return f"rollout-{int(rollout_index):04d}-{name}"


def recording_label(rollout_index: int, variant: int, kind: int) -> str:
    name = _VARIANT_NAMES.get(int(variant), "rlt")
    status = _KIND_NAMES.get(int(kind), "running")
    return f"rollout {int(rollout_index):04d} {name} — {status}"


def rerun_observation(
    observation: dict[str, Any] | None,
    *,
    rollout_index: int,
    variant: str,
    kind: int | None = None,
) -> dict[str, Any]:
    """Namespace camera frames and attach the rollout identity sentinels."""
    logged: dict[str, Any] = {}
    for key, value in (observation or {}).items():
        if getattr(value, "ndim", 0) >= 2:
            name = str(key) if str(key).startswith("images.") else f"images.{key}"
            logged[name] = value
        else:
            logged[key] = value
    logged[INDEX_PATH] = float(rollout_index)
    logged[VARIANT_PATH] = float(variant_code(variant))
    if kind is not None:
        logged[KIND_PATH] = float(kind)
    return logged


def log_rollout(
    ctx: Any,
    observation: dict[str, Any] | None,
    action: dict[str, Any] | None,
    *,
    rollout_index: int,
    variant: str,
    kind: int,
) -> None:
    """Hand one control step, or the final outcome, to the Armnet Rerun stream."""
    ctx.log_rerun_data(
        observation=rerun_observation(
            observation,
            rollout_index=rollout_index,
            variant=variant,
            kind=kind,
        ),
        action=action,
    )


class RolloutRecordings:
    """Route Armnet Rerun packets into one recording per rollout."""

    def __init__(self, rr: Any, *, grpc_url: str = _GRPC_URL) -> None:
        self._rr = rr
        self._grpc_url = grpc_url
        self._recordings: dict[str, Any] = {}
        self._steps: dict[str, int] = {}
        self._kinds: dict[str, int] = {}

    def replay(self, packet: Any) -> bool:
        """Log ``packet`` into its rollout recording.

        Returns False when the packet is not an RLT rollout packet, so the
        caller can fall back to Armnet's single-stream logger.
        """
        entries = list(packet.entries)
        index = _sentinel(entries, INDEX_PATH)
        variant = _sentinel(entries, VARIANT_PATH)
        if index is None or variant is None:
            return False
        kind = _sentinel(entries, KIND_PATH)
        key = recording_id(int(index), int(variant))
        recording = self._open(key, int(index), int(variant), kind)
        payload = [entry for entry in entries if entry.entity_path not in _SENTINELS]
        step = self._steps.get(key, 0)
        if payload:
            self._rr.set_time("step", sequence=step, recording=recording)
            for entry in payload:
                _log_entry(self._rr, entry, recording)
            self._steps[key] = step + 1
        if kind is not None and int(kind) != self._kinds.get(key):
            resolved = int(kind)
            self._kinds[key] = resolved
            self._label(recording, int(index), int(variant), resolved)
            if resolved != KIND_RUNNING:
                self._rr.set_time(
                    "step",
                    sequence=max(self._steps.get(key, 1) - 1, 0),
                    recording=recording,
                )
                self._rr.log(
                    "outcome/success",
                    self._rr.Scalars(1.0 if resolved == KIND_SUCCESS else 0.0),
                    recording=recording,
                )
        return True

    def _open(self, key: str, index: int, variant: int, kind: float | None) -> Any:
        recording = self._recordings.get(key)
        if recording is not None:
            return recording
        recording = self._rr.RecordingStream(
            _APPLICATION_ID,
            recording_id=key,
        )
        self._rr.connect_grpc(self._grpc_url, recording=recording)
        self._recordings[key] = recording
        self._steps[key] = 0
        self._label(
            recording,
            index,
            variant,
            KIND_RUNNING if kind is None else int(kind),
        )
        _send_blueprint(self._rr, recording)
        return recording

    def _label(self, recording: Any, index: int, variant: int, kind: int) -> None:
        self._rr.send_property(
            "recording_info",
            self._rr.archetypes.RecordingInfo(
                name=recording_label(index, variant, kind)
            ),
            recording=recording,
        )


def start_live_view() -> None:
    """Open the viewer and make Armnet's packet logger split rollouts."""
    from armnet_client.rerun_stream import init_rerun

    init_rerun(
        _APPLICATION_ID,
        spawn=True,
        grpc_port=_GRPC_PORT,
        web_port=9090,
    )
    import rerun as rr

    install_rollout_recordings(RolloutRecordings(rr))
    print(
        "[rlt] Rerun is recording each rollout separately. "
        "The recording name changes to success, fail, or timeout when it ends.",
        flush=True,
    )


def install_rollout_recordings(recordings: RolloutRecordings) -> None:
    """Replace Armnet's single-stream packet logger for this process."""
    global _installed
    if _installed:
        return
    import armnet_runtime.rerun as rerun_mod

    original = rerun_mod.log_packet

    def log_packet(packet: Any) -> None:
        if not recordings.replay(packet):
            original(packet)

    rerun_mod.log_packet = log_packet
    _installed = True


def _sentinel(entries: list[Any], path: str) -> float | None:
    for entry in entries:
        if entry.entity_path == path and entry.WhichOneof("value") == "scalar":
            return float(entry.scalar)
    return None


def _log_entry(rr: Any, entry: Any, recording: Any) -> None:
    which = entry.WhichOneof("value")
    if which == "scalar":
        rr.log(entry.entity_path, rr.Scalars(entry.scalar), recording=recording)
        return
    if which != "image":
        return
    image = entry.image
    encoding = getattr(image, "encoding", None)
    if encoding == 1 or getattr(encoding, "name", "") == "IMAGE_ENCODING_JPEG":
        rr.log(
            entry.entity_path,
            rr.EncodedImage(contents=bytes(image.data), media_type="image/jpeg"),
            recording=recording,
        )
        return
    if encoding == 2 or getattr(encoding, "name", "") == "IMAGE_ENCODING_RAW_RGB":
        import numpy as np

        channels = int(getattr(image, "channels", 0) or 3)
        array = np.frombuffer(bytes(image.data), dtype=np.uint8)
        shape = (
            (int(image.height), int(image.width), channels)
            if channels > 1
            else (int(image.height), int(image.width))
        )
        rr.log(entry.entity_path, rr.Image(array.reshape(shape)), recording=recording)


def _send_blueprint(rr: Any, recording: Any) -> None:
    try:
        import rerun.blueprint as rrb
    except ImportError:
        return
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Vertical(
                rrb.Spatial2DView(origin="/", name="Cameras"),
                rrb.TimeSeriesView(origin="/", name="Joints and actions"),
            )
        ),
        recording=recording,
    )
