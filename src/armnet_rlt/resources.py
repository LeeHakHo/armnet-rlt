"""Names shared by the Modal learner, local learner and Armnet submitter."""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
from typing import Any


APP_NAME = "armnet-rlt-learner"
LEARNER_PORT = 50051
VOLUME_NAME = os.environ.get("RLT_MODAL_VOLUME", "pi0-rlt-data")
VOLUME_MOUNT = PurePosixPath("/data")
RENDEZVOUS_DICT = os.environ.get(
    "RLT_MODAL_RENDEZVOUS", "armnet-rlt-rendezvous"
)
AUTH_SECRET_NAME = os.environ.get("RLT_MODAL_AUTH_SECRET", "armnet-rlt-auth")

GREEN_BUTTON_CONFIG = "pi05_rlt_busybox_push_green_button"
GREEN_BUTTON_CHECKPOINT = "pravsels/pi05_rlt_busybox_push_green_button"
GREEN_BUTTON_DATASET = "villekuosmanen/busybox_push_green_button"
GREEN_BUTTON_PROMPT = "push the green button"
MULTITASK_CONFIG = "pi05_rlt_busybox_multitask_singlearm_minmax"
MULTITASK_CHECKPOINT = "AutoRLBench/pi05_rlt_busybox_multitask_singlearm_minmax"
MULTITASK_DATASET = "villekuosmanen/busybox_multitask"


def local_data_root() -> Path:
    """Directory a learner on this machine uses in place of the Modal volume."""
    default = Path(__file__).resolve().parents[2] / "experiment-data"
    return Path(os.environ.get("RLT_DATA_ROOT") or default).resolve()


def task_paths(
    config_name: str, root: str | os.PathLike[str] = VOLUME_MOUNT
) -> tuple[str, str, str]:
    """Return demo cache, stats directory and production output directory."""
    root = Path(root) / "tasks" / config_name
    return (
        str(root / "demo" / "rlt_demo_cache.pt"),
        str(root / "ckpt" / "assets"),
        str(root / "rlt_checkpoints"),
    )


def rendezvous_key(config_name: str, run_id: str) -> str:
    """Namespace endpoints so two learners cannot overwrite each other."""
    clean_config = config_name.strip().replace("/", "-")
    clean_run = run_id.strip().replace("/", "-")
    if not clean_config or not clean_run:
        raise ValueError("config_name and run_id are required for rendezvous")
    return f"{clean_config}:{clean_run}"


def local_rendezvous_path(key: str) -> Path:
    """File a learner on this machine registers its endpoint in."""
    return local_data_root() / "rendezvous" / f"{key.replace(':', '__')}.json"


def read_local_rendezvous(key: str) -> dict[str, Any] | None:
    path = local_rendezvous_path(key)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def policy_rendezvous_key(config_name: str) -> str:
    """Key a local policy server registers its endpoint under."""
    clean_config = config_name.strip().replace("/", "-")
    if not clean_config:
        raise ValueError("config_name is required for rendezvous")
    return f"policy:{clean_config}"


def add_remote_policy_arguments(parser: Any) -> None:
    parser.add_argument(
        "--remote-policy",
        action="store_true",
        help="Run OpenPI inference on the local `rlt-local policy-server` "
        "instead of the cell GPU",
    )
    parser.add_argument("--remote-policy-timeout-s", type=float, default=5.0)
    parser.add_argument(
        "--remote-policy-image-codec", choices=("png", "jpeg", "raw"), default="png"
    )


def remote_policy_job_args(
    config_name: str, *, timeout_s: float, image_codec: str
) -> dict[str, Any]:
    """Job args pointing the cell at the local policy server for ``config_name``."""
    key = policy_rendezvous_key(config_name)
    entry = read_local_rendezvous(key)
    if entry is None:
        raise RuntimeError(
            f"No policy server registered at {local_rendezvous_path(key)}. "
            "Start `rlt-local policy-server` first."
        )
    if entry.get("config_name") != config_name:
        raise ValueError(
            f"policy server serves {entry.get('config_name')!r}, "
            f"not {config_name!r}"
        )
    return {
        "remote_policy_host": entry["host"],
        "remote_policy_port": int(entry["port"]),
        "remote_policy_use_tls": bool(entry["use_tls"]),
        "remote_policy_timeout_s": float(timeout_s),
        "remote_policy_image_codec": image_codec,
    }


def scheduled_cell(require_cell: str) -> str | None:
    """The cell to dispatch to when ``require_cell`` names exactly one."""
    cells = [name.strip() for name in require_cell.split(",") if name.strip()]
    return cells[0] if len(cells) == 1 else None


# Per-cell camera aim used when --armnet.camera-aim is not given.
# cell-03 top tilt 0.1661 of 983-3192 steps = 1350.
CELL_CAMERA_AIM = {"cell-03": "top.tilt=0.1661"}


def camera_aim_args(
    spec: str | None, require_cell: str, job_args: dict[str, Any]
) -> dict[str, Any]:
    """Job args for the camera aim; turns camera variation off when aiming."""
    from armnet_client.camera_aim import camera_aim_job_args

    if spec is None:
        spec = CELL_CAMERA_AIM.get(scheduled_cell(require_cell) or "", "")
    if not spec.strip():
        return {}
    updates: dict[str, Any] = {}
    if job_args.get("variation") and "variation_cameras" not in job_args:
        updates = {
            "variation_rail": True,
            "variation_cameras": False,
            "variation_lighting": True,
        }
    return {**updates, **camera_aim_job_args(spec, {**job_args, **updates})}


def device_override_args(
    rail: float | None, lightbox_brightness: int | None
) -> dict[str, Any]:
    """Job args that position the rail and set the lightbox for the job."""
    updates: dict[str, Any] = {}
    if rail is not None:
        if not 0.0 <= rail <= 1.0:
            raise ValueError(f"rail must be within 0-1, got {rail}")
        updates["rail_position"] = float(rail)
    if lightbox_brightness is not None:
        if not 0 <= lightbox_brightness <= 100:
            raise ValueError(
                f"lightbox brightness must be within 0-100, got {lightbox_brightness}"
            )
        updates["lightbox_brightness"] = int(lightbox_brightness)
    return updates


def preflight_device_overrides(args: Any, *, embodiment: str) -> None:
    """Check --armnet.rail / --armnet.lightbox-brightness against the cell envelope."""
    from armnet_client.variation import preflight_manual_overrides

    preflight_manual_overrides(
        embodiment=embodiment,
        task=getattr(args, "task", None),
        rail=getattr(args, "rail", None),
        lightbox_brightness=getattr(args, "lightbox_brightness", None),
        cell=scheduled_cell(str(getattr(args, "require_cell", "") or "")),
    )


def report_job_outcome(result: Any, *, prefix: str) -> None:
    """Print the job outcome and raise on FAILED, TIMEOUT or CANCELLED.

    Other statuses print a warning and return.
    """
    from armnet_client import JobStatus

    print(f"[{prefix}] job status: {result.status}")
    if result.return_value is not None:
        print(f"[{prefix}] result: {result.return_value}")
    if result.status in (
        JobStatus.FAILED,
        JobStatus.TIMEOUT,
        JobStatus.CANCELLED,
    ):
        result.raise_for_status()
    if result.status != JobStatus.SUCCEEDED:
        print(
            f"[{prefix}] job is not finished (status {result.status}); "
            "nothing ran yet, so do not read this as a completed run"
        )
