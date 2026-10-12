"""Names shared by the Modal learner and Armnet submitter."""

from __future__ import annotations

import os
from pathlib import PurePosixPath
from typing import Any


APP_NAME = "armnet-rlt-learner"
LEARNER_PORT = 50051
VOLUME_NAME = os.environ.get("RLT_MODAL_VOLUME", "pi0-rlt-data")
VOLUME_MOUNT = PurePosixPath("/data")
RENDEZVOUS_DICT = os.environ.get(
    "RLT_MODAL_RENDEZVOUS", "armnet-rlt-rendezvous"
)
AUTH_SECRET_NAME = os.environ.get("RLT_MODAL_AUTH_SECRET", "armnet-rlt-auth")


def task_paths(config_name: str) -> tuple[str, str, str]:
    """Return demo cache, stats directory and production output directory."""
    root = VOLUME_MOUNT / "tasks" / config_name
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
