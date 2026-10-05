"""Submit a paired base-vs-frozen-RLT evaluation to Armnet."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from armnet_client.variation import (
    DEFAULT_VARIATION_SEED,
    variation_job_args,
)

from armnet_rlt.rerun_view import (
    add_rerun_argument,
    rerun_job_enabled,
    start_live_view,
)


SO101_EMBODIMENT = "lerobot/so-101"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="OpenPI checkpoint on the Armnet volume",
    )
    parser.add_argument(
        "--frozen-rlt-checkpoint",
        required=True,
        help="Frozen actor file on the Armnet volume",
    )
    parser.add_argument("--language-instruction", required=True)
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=20,
        help="Number of paired scenes; base and frozen RLT each run once",
    )
    parser.add_argument(
        "--include-base",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate the base OpenPI policy on each paired scene",
    )
    parser.add_argument(
        "--armnet.variation",
        dest="variation",
        action="store_true",
    )
    parser.add_argument(
        "--armnet.variation-seed",
        dest="variation_seed",
        type=int,
        default=DEFAULT_VARIATION_SEED,
    )
    parser.add_argument(
        "--armnet.variation-episode-offset",
        dest="variation_episode_offset",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--record-dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--record-dataset-repo-id")
    parser.add_argument("--hf-user")
    parser.add_argument("--no-push-to-hub", action="store_true")
    parser.add_argument(
        "--robot-telemetry",
        choices=("off", "context", "safety", "full"),
        default="full",
    )
    parser.add_argument("--robot-telemetry-strict", action="store_true")
    parser.add_argument("--hf-secret", default="huggingface-token")
    add_rerun_argument(parser)
    parser.add_argument("--timeout-seconds", type=int, default=10800)
    parser.add_argument("--image-name", default="armnet-rlt-frozen-eval")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _volume_path(value: str, *, name: str) -> str:
    if not value.startswith("volume://"):
        raise ValueError(f"{name} must be a volume:// path")
    if not value.removeprefix("volume://").strip("/"):
        raise ValueError(f"{name} must not be empty")
    return value


def build_job_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_rollouts <= 0:
        raise ValueError("num_rollouts must be positive")
    if args.variation_episode_offset < 0:
        raise ValueError("variation_episode_offset must be nonnegative")
    job_args: dict[str, Any] = {
        "config_name": args.config_name,
        "checkpoint_dir": _volume_path(
            args.checkpoint_dir,
            name="checkpoint_dir",
        ),
        "frozen_rlt_checkpoint": _volume_path(
            args.frozen_rlt_checkpoint,
            name="frozen_rlt_checkpoint",
        ),
        "language_instruction": args.language_instruction,
        "num_rollouts": int(args.num_rollouts),
        "eval_include_base": bool(args.include_base),
        "exploration_scale": 0.0,
        "record_dataset": bool(args.record_dataset),
        "robot_telemetry": str(args.robot_telemetry),
        "robot_telemetry_strict": bool(args.robot_telemetry_strict),
        "use_rerun": rerun_job_enabled(args),
    }
    job_args.update(
        variation_job_args(
            bool(args.variation),
            int(args.variation_seed),
        )
    )
    if args.variation_episode_offset:
        job_args["variation_episode_offset"] = int(
            args.variation_episode_offset
        )
    if args.record_dataset_repo_id:
        job_args["record_dataset_repo_id"] = args.record_dataset_repo_id
    if args.hf_user:
        job_args["hf_user"] = args.hf_user
    if args.no_push_to_hub:
        job_args["no_push_to_hub"] = True
    return job_args


def main(argv: list[str] | None = None) -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = _parser().parse_args(argv)
    if not os.environ.get("ARMNET_API_KEY") and not args.dry_run:
        raise SystemExit(
            "ARMNET_API_KEY is missing. Export it or use --dry-run."
        )
    job_args = build_job_args(args)
    secrets = {"HF_TOKEN": args.hf_secret} if args.hf_secret else {}
    print(f"[rlt eval] task={args.task} args={job_args}")
    print(f"[rlt eval] secrets={secrets}")
    if args.use_rerun and args.detach:
        print(
            "[rlt eval] Rerun stays on this machine, so --detach disables it. "
            "Submit without --detach to watch each rollout live."
        )
    if args.dry_run:
        return
    if rerun_job_enabled(args):
        start_live_view()

    from armnet_client import Image, execute

    image = Image.build(
        dockerfile=PROJECT_ROOT / "Dockerfile.actor",
        context_dir=PROJECT_ROOT,
        name=args.image_name,
    ).push(if_possible=True)
    result = execute(
        image=image,
        embodiment=SO101_EMBODIMENT,
        task=args.task,
        args=job_args,
        secrets=secrets,
        detach=args.detach,
        stream_logs=not args.detach,
        use_rerun=rerun_job_enabled(args),
        timeout_seconds=args.timeout_seconds,
    )
    print(f"[rlt eval] job status: {result.status}")
    if result.return_value is not None:
        print(f"[rlt eval] result: {result.return_value}")
    result.raise_for_status()


if __name__ == "__main__":
    main()
