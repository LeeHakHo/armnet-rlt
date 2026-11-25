"""Modal app for BACKEND=modal: a run's learner and policy server in one GPU container.

    uv run --extra modal modal deploy -m armnet_rlt.modal_app

Images: the envs/ environments from scripts/setup_local.sh and plugins/; src/ is added at
container start. The volume RLT_MODAL_VOLUME is mounted as RLT_DATA_ROOT (/data).
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import modal

from armnet_rlt.modal_backend import CONTAINER_DATA, CONTAINER_ROOT, run_command as _run_command, supervise
from armnet_rlt.resources import (
    MODAL_APP,
    MODAL_AUTH_SECRET,
    MODAL_HF_SECRET,
    MODAL_RENDEZVOUS,
    MODAL_VOLUME,
)


REPO = Path(__file__).resolve().parents[2]
GPU = os.environ.get("RLT_MODAL_GPU", "L40S")
TIMEOUT_S = 24 * 60 * 60
IGNORE = ["**/__pycache__", "**/*.pyc", "**/*.egg-info", "**/.git"]

volume = modal.Volume.from_name(MODAL_VOLUME, create_if_missing=True)
rendezvous = modal.Dict.from_name(MODAL_RENDEZVOUS, create_if_missing=True)
# RLT_LEARNER_AUTH_TOKEN (and WANDB_API_KEY, if set); HF_TOKEN.
secrets = [modal.Secret.from_name(MODAL_AUTH_SECRET), modal.Secret.from_name(MODAL_HF_SECRET)]


def _image(*envs: str) -> modal.Image:
    """scripts/setup_local.sh <env> for each env, then the plugins, then src/."""
    stub = f"mkdir -p {CONTAINER_ROOT}/src/armnet_rlt && touch {CONTAINER_ROOT}/src/armnet_rlt/__init__.py {CONTAINER_ROOT}/README.md"
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git", "ca-certificates", "openssl", "ffmpeg", "build-essential")
        .pip_install("uv")
        .env({"UV_LINK_MODE": "copy"})
        .add_local_file(REPO / "pyproject.toml", f"{CONTAINER_ROOT}/pyproject.toml", copy=True)
        .add_local_file(REPO / "scripts/setup_local.sh", f"{CONTAINER_ROOT}/scripts/setup_local.sh", copy=True)
        .add_local_dir(REPO / "third_party/lerobot", f"{CONTAINER_ROOT}/third_party/lerobot", copy=True, ignore=IGNORE)
        .run_commands(
            stub,
            *(f"cd {CONTAINER_ROOT} && SKIP_CHECK=1 scripts/setup_local.sh {env}" for env in envs),
            f"rm -rf {CONTAINER_ROOT}/src",
        )
        .add_local_file(REPO / "scripts/install_plugins.sh", f"{CONTAINER_ROOT}/scripts/install_plugins.sh", copy=True)
        .add_local_dir(REPO / "plugins", f"{CONTAINER_ROOT}/plugins", copy=True, ignore=IGNORE)
        .run_commands(f"cd {CONTAINER_ROOT} && scripts/install_plugins.sh")
        .env({
            "PYTHONPATH": f"{CONTAINER_ROOT}/src",
            "RLT_DATA_ROOT": CONTAINER_DATA,
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        })
        .add_local_dir(REPO / "src", f"{CONTAINER_ROOT}/src", ignore=IGNORE)
    )


rl_image = _image("learner", "openpi")
lerobot_image = _image("lerobot")

app = modal.App(MODAL_APP, include_source=False)
options = dict(gpu=GPU, volumes={CONTAINER_DATA: volume}, secrets=secrets, timeout=TIMEOUT_S)


@contextlib.contextmanager
def _forward(port: int):
    # TLS with HTTP/2 at Modal's edge; plaintext gRPC on the port.
    with modal.forward(port, h2_enabled=True) as tunnel:
        yield tunnel.tls_socket


def _serve(processes: list[dict]) -> dict[str, int]:
    return supervise(
        processes,
        forward=_forward,
        publish=rendezvous.put,
        unpublish=lambda key: rendezvous.pop(key, None),
        commit=volume.commit,
        call_key=f"call:{modal.current_function_call_id()}",
        rendezvous_dir=Path("/tmp/rlt-rendezvous"),
    )


@app.function(image=rl_image, **options)
def serve(processes: list[dict]) -> dict[str, int]:
    """Learner and policy server (rlt, dsrl, rl), or a policy server alone."""
    return _serve(processes)


@app.function(image=lerobot_image, **options)
def serve_lerobot(processes: list[dict]) -> dict[str, int]:
    """LeRobot learner and policy server (dit), or a policy server alone."""
    return _serve(processes)


@app.function(image=rl_image, **options)
def run_command(argv: list[str], log: str, env: dict[str, str]) -> dict:
    """One command in the rl image: assets, cache, hf download."""
    return _run_command(argv, log, env, commit=volume.commit)


@app.function(image=lerobot_image, **options)
def run_command_lerobot(argv: list[str], log: str, env: dict[str, str]) -> dict:
    """One command in the lerobot image: lerobot-train."""
    return _run_command(argv, log, env, commit=volume.commit)
