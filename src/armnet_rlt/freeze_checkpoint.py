"""Export one local RLT learner checkpoint as a frozen Armnet actor."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from dotenv import load_dotenv

from armnet_rlt.frozen import freeze_learner_checkpoint
from armnet_rlt.resources import local_data_root


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


def _resolve_step(requested: str) -> int:
    try:
        step = int(requested)
    except ValueError as exc:
        raise ValueError("step must be an integer or 'latest'") from exc
    if step < 0:
        raise ValueError("step must be nonnegative")
    return step


def _read_local_checkpoint(
    *,
    config_name: str,
    run_id: str,
    step: str,
) -> tuple[dict, int, Path]:
    root = local_data_root() / _run_root(config_name, run_id)
    if step == "latest":
        steps = [
            int(entry.name.removeprefix("step_"))
            for entry in root.glob("step_*")
            if entry.name.removeprefix("step_").isdigit()
        ]
        if not steps:
            raise FileNotFoundError(f"no learner checkpoints found under {root}")
        resolved_step = max(steps)
    else:
        resolved_step = _resolve_step(step)
    path = root / f"step_{resolved_step:06d}" / "checkpoint.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path} must contain a dictionary")
    return checkpoint, resolved_step, path


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
    source = {"config_name": args.config_name, "run_id": args.run_id}
    checkpoint, step, local_path = _read_local_checkpoint(
        config_name=args.config_name,
        run_id=args.run_id,
        step=args.step,
    )
    source["local_path"] = str(local_path)
    artifact = freeze_learner_checkpoint(checkpoint, source=source)
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
