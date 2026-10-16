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


def scheduled_cell(require_cell: str) -> str | None:
    """The cell to dispatch to when ``require_cell`` names exactly one."""
    cells = [name.strip() for name in require_cell.split(",") if name.strip()]
    return cells[0] if len(cells) == 1 else None


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
