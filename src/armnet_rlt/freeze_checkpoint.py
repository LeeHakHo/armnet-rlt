"""Export one Modal RLT learner checkpoint as a frozen Armnet actor."""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import torch
from dotenv import load_dotenv

from armnet_rlt.frozen import freeze_learner_checkpoint
from armnet_rlt.resources import VOLUME_NAME


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--step",
        default="latest",
        help="Learner step number or 'latest'",
    )
    parser.add_argument("--modal-volume", default=VOLUME_NAME)
    parser.add_argument(
        "--output",
        type=Path,
        help="Local frozen_actor.pt destination",
    )
    parser.add_argument(
        "--armnet-volume-path",
        help="Optionally upload the frozen file to this Armnet volume path",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _run_root(config_name: str, run_id: str) -> str:
    clean_config = config_name.strip().strip("/")
    clean_run = run_id.strip().strip("/")
    if not clean_config or "/" in clean_config:
        raise ValueError("config_name must be a nonempty path component")
    if not clean_run or "/" in clean_run:
        raise ValueError("run_id must be a nonempty path component")
    return f"tasks/{clean_config}/rlt_checkpoints/{clean_run}"


def _resolve_step(volume, root: str, requested: str) -> int:
    if requested != "latest":
        try:
            step = int(requested)
        except ValueError as exc:
            raise ValueError("step must be an integer or 'latest'") from exc
        if step < 0:
            raise ValueError("step must be nonnegative")
        return step
    steps = []
    for entry in volume.iterdir(root, recursive=False):
        name = Path(entry.path).name
        suffix = name.removeprefix("step_")
        if name.startswith("step_") and suffix.isdigit():
            steps.append(int(suffix))
    if not steps:
        raise FileNotFoundError(f"no learner checkpoints found under {root}")
    return max(steps)


def _read_modal_checkpoint(
    *,
    volume_name: str,
    config_name: str,
    run_id: str,
    step: str,
) -> tuple[dict, int, str]:
    import modal

    volume = modal.Volume.from_name(volume_name)
    root = _run_root(config_name, run_id)
    resolved_step = _resolve_step(volume, root, step)
    remote_path = (
        f"{root}/step_{resolved_step:06d}/checkpoint.pt"
    )
    payload = b"".join(volume.read_file(remote_path))
    checkpoint = torch.load(
        io.BytesIO(payload),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{remote_path} must contain a dictionary")
    return checkpoint, resolved_step, remote_path


def _default_output(config_name: str, run_id: str, step: int) -> Path:
    return (
        PROJECT_ROOT
        / "frozen_checkpoints"
        / config_name
        / run_id
        / f"step_{step:06d}"
        / "frozen_actor.pt"
    )


def _load_environment() -> None:
    for candidate in (
        PROJECT_ROOT / ".env",
        PROJECT_ROOT.parents[1] / "armnet" / ".env",
    ):
        if candidate.exists():
            load_dotenv(candidate)
            return


def _upload(path: Path, volume_path: str, *, overwrite: bool) -> str:
    from armnet_client._config import orchestrator_url
    from armnet_client.client import OrchestratorClient
    from armnet_client.volume import upload_to_cloud_volume

    if not os.environ.get("ARMNET_API_KEY"):
        raise RuntimeError(
            "ARMNET_API_KEY is required to upload to the Armnet volume"
        )
    with OrchestratorClient(orchestrator_url()) as client:
        username = client.whoami().username
        credentials = client.get_volume_credentials()
    clean = volume_path.removeprefix("volume://").strip("/")
    if not clean:
        raise ValueError("armnet_volume_path must not be empty")
    upload_to_cloud_volume(
        credentials=credentials,
        username=username,
        local_path=path,
        volume_path=clean,
        overwrite=overwrite,
    )
    return f"volume://{clean}"


def main(argv: list[str] | None = None) -> None:
    _load_environment()
    args = _parser().parse_args(argv)
    checkpoint, step, remote_path = _read_modal_checkpoint(
        volume_name=args.modal_volume,
        config_name=args.config_name,
        run_id=args.run_id,
        step=args.step,
    )
    artifact = freeze_learner_checkpoint(
        checkpoint,
        source={
            "modal_volume": args.modal_volume,
            "remote_path": remote_path,
            "config_name": args.config_name,
            "run_id": args.run_id,
        },
    )
    output = args.output or _default_output(
        args.config_name, args.run_id, step
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    print(f"[rlt freeze] step={step} output={output}")
    if args.armnet_volume_path:
        uploaded = _upload(
            output,
            args.armnet_volume_path,
            overwrite=args.overwrite,
        )
        print(f"[rlt freeze] uploaded={uploaded}")


if __name__ == "__main__":
    main()
