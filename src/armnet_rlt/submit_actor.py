"""Build and submit the single-arm RLT actor to Armnet."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from armnet_client.variation import DEFAULT_VARIATION_SEED, variation_job_args

from armnet_rlt.rerun_view import (
    add_rerun_argument,
    rerun_job_enabled,
    start_live_view,
)
from armnet_rlt.resources import RENDEZVOUS_DICT


SO101_EMBODIMENT = "lerobot/so-101"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Armnet task slug")
    parser.add_argument("--config-name", required=True, help="OpenPI RLT config")
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument(
        "--checkpoint-dir",
        help="Armnet volume path, e.g. volume://openpi/checkpoints/<name>",
    )
    checkpoint.add_argument("--hf-checkpoint-repo")
    parser.add_argument("--hf-checkpoint-revision", default="")
    parser.add_argument("--language-instruction", required=True)
    parser.add_argument(
        "--learner-key",
        required=True,
        help="<config-name>:<run-id> printed by the Modal learner",
    )
    parser.add_argument("--num-rollouts", type=int, default=40)
    parser.add_argument("--reference-action-len", type=int, default=30)
    parser.add_argument(
        "--exploration-scale",
        type=float,
        default=0.5,
        help="Multiplier for RLT exploration noise; use 0 for deterministic validation",
    )
    parser.add_argument(
        "--armnet.variation",
        dest="variation",
        action="store_true",
        help="Sample configured cell variation independently for each rollout",
    )
    parser.add_argument(
        "--armnet.variation-rail",
        dest="variation_rail",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override rail variation independently",
    )
    parser.add_argument(
        "--armnet.variation-cameras",
        dest="variation_cameras",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override camera variation independently",
    )
    parser.add_argument(
        "--armnet.variation-lighting",
        dest="variation_lighting",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override lighting variation independently",
    )
    parser.add_argument(
        "--armnet.variation-seed",
        dest="variation_seed",
        type=int,
        default=DEFAULT_VARIATION_SEED,
        help="Seed for deterministic per-rollout scene variation",
    )
    parser.add_argument(
        "--armnet.variation-episode-offset",
        dest="variation_episode_offset",
        type=int,
        default=0,
        help="Continue a deterministic variation sequence across actor jobs",
    )
    parser.add_argument(
        "--variation-curriculum",
        action="store_true",
        help="Adapt scene-variation scale from rolling rollout success",
    )
    parser.add_argument("--variation-scale-start", type=float, default=0.25)
    parser.add_argument("--variation-scale-min", type=float, default=0.1)
    parser.add_argument("--variation-scale-max", type=float, default=1.0)
    parser.add_argument("--variation-scale-step-up", type=float, default=0.05)
    parser.add_argument("--variation-scale-step-down", type=float, default=0.1)
    parser.add_argument("--variation-window", type=int, default=20)
    parser.add_argument(
        "--variation-promote-threshold", type=float, default=0.8
    )
    parser.add_argument(
        "--variation-demote-threshold", type=float, default=0.55
    )
    parser.add_argument(
        "--variation-frontier-fraction", type=float, default=0.2
    )
    parser.add_argument(
        "--record-dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record rollouts into a private eval_rlt_* LeRobot dataset",
    )
    parser.add_argument(
        "--record-dataset-repo-id",
        help="Explicit HuggingFace dataset repo id (defaults to eval_rlt_<task>_<timestamp>)",
    )
    parser.add_argument("--hf-user", help="HuggingFace username for generated dataset ids")
    parser.add_argument("--no-push-to-hub", action="store_true")
    parser.add_argument(
        "--robot-telemetry",
        choices=("off", "context", "safety", "full"),
        default="full",
        help="Frame-aligned motor telemetry sidecar detail",
    )
    parser.add_argument(
        "--robot-telemetry-strict",
        action="store_true",
        help="Fail the actor if telemetry recording is incomplete",
    )
    add_rerun_argument(parser)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--auth-secret", default="armnet-rlt-auth")
    parser.add_argument("--hf-secret", default="huggingface-token")
    parser.add_argument("--image-name", default="armnet-rlt-actor")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def resolve_rendezvous(key: str) -> dict[str, Any]:
    """Read one namespaced learner endpoint from Modal."""
    import modal

    entry = modal.Dict.from_name(RENDEZVOUS_DICT).get(key)
    if not entry:
        raise RuntimeError(
            f"No learner registered in Modal Dict {RENDEZVOUS_DICT!r} at "
            f"{key!r}. Start the learner and use the key it prints."
        )
    required = {"host", "port", "use_tls", "config_name", "run_id"}
    missing = required.difference(entry)
    if missing:
        raise RuntimeError(f"Learner rendezvous {key!r} lacks {sorted(missing)}")
    return dict(entry)


def build_job_args(args: argparse.Namespace, entry: dict[str, Any]) -> dict[str, Any]:
    """Construct runtime args, rejecting accidental cross-task connections."""
    if entry["config_name"] != args.config_name:
        raise ValueError(
            f"learner {args.learner_key!r} serves {entry['config_name']!r}, "
            f"not requested config {args.config_name!r}"
        )
    job_args: dict[str, Any] = {
        "config_name": args.config_name,
        "language_instruction": args.language_instruction,
        "reference_action_len": args.reference_action_len,
        "learner_host": entry["host"],
        "learner_port": int(entry["port"]),
        "use_tls": bool(entry["use_tls"]),
        "delta_actions": True,
        "exploration_scale": float(getattr(args, "exploration_scale", 0.5)),
        "record_dataset": bool(getattr(args, "record_dataset", True)),
        "robot_telemetry": str(getattr(args, "robot_telemetry", "full")),
        "robot_telemetry_strict": bool(
            getattr(args, "robot_telemetry_strict", False)
        ),
        "use_rerun": rerun_job_enabled(args),
    }
    if args.checkpoint_dir:
        job_args["checkpoint_dir"] = args.checkpoint_dir
    else:
        job_args["hf_checkpoint_repo"] = args.hf_checkpoint_repo
    if args.hf_checkpoint_revision and args.hf_checkpoint_repo:
        job_args["hf_checkpoint_revision"] = args.hf_checkpoint_revision
    if args.num_rollouts > 0:
        job_args["num_rollouts"] = args.num_rollouts
    curriculum_enabled = bool(
        getattr(args, "variation_curriculum", False)
    )
    selectors = {
        "variation_rail": getattr(args, "variation_rail", None),
        "variation_cameras": getattr(args, "variation_cameras", None),
        "variation_lighting": getattr(args, "variation_lighting", None),
    }
    variation_enabled = (
        bool(getattr(args, "variation", False))
        or curriculum_enabled
        or any(value is True for value in selectors.values())
    )
    job_args.update(
        variation_job_args(
            variation_enabled,
            int(getattr(args, "variation_seed", DEFAULT_VARIATION_SEED)),
        )
    )
    job_args.update(
        {
            name: bool(value)
            for name, value in selectors.items()
            if value is not None
        }
    )
    variation_offset = int(
        getattr(args, "variation_episode_offset", 0)
    )
    if variation_offset < 0:
        raise ValueError("variation_episode_offset must be nonnegative")
    if variation_offset:
        job_args["variation_episode_offset"] = variation_offset
    if curriculum_enabled:
        job_args.update(
            {
                "variation_curriculum": True,
                "variation_scale_start": float(args.variation_scale_start),
                "variation_scale_min": float(args.variation_scale_min),
                "variation_scale_max": float(args.variation_scale_max),
                "variation_scale_step_up": float(
                    args.variation_scale_step_up
                ),
                "variation_scale_step_down": float(
                    args.variation_scale_step_down
                ),
                "variation_window": int(args.variation_window),
                "variation_promote_threshold": float(
                    args.variation_promote_threshold
                ),
                "variation_demote_threshold": float(
                    args.variation_demote_threshold
                ),
                "variation_frontier_fraction": float(
                    args.variation_frontier_fraction
                ),
            }
        )
    if repo_id := getattr(args, "record_dataset_repo_id", None):
        job_args["record_dataset_repo_id"] = repo_id
    if hf_user := getattr(args, "hf_user", None):
        job_args["hf_user"] = hf_user
    if bool(getattr(args, "no_push_to_hub", False)):
        job_args["no_push_to_hub"] = True
    if cert := entry.get("tls_root_cert_pem"):
        # Supported for a future self-signed endpoint. Modal's h2-enabled TLS
        # tunnel uses a public CA, so current entries normally omit this.
        job_args["tls_root_cert_pem"] = cert
    return job_args


def main(argv: list[str] | None = None) -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = _parser().parse_args(argv)
    if not os.environ.get("ARMNET_API_KEY") and not args.dry_run:
        raise SystemExit(
            "ARMNET_API_KEY is missing. Export it or put it in "
            f"{PROJECT_ROOT / '.env'}."
        )

    entry = resolve_rendezvous(args.learner_key)
    job_args = build_job_args(args, entry)
    secrets = {"RLT_LEARNER_AUTH_TOKEN": args.auth_secret}
    if args.hf_secret:
        secrets["HF_TOKEN"] = args.hf_secret

    print(f"[rlt] learner={entry['host']}:{entry['port']} tls={entry['use_tls']}")
    print(f"[rlt] embodiment={SO101_EMBODIMENT} task={args.task}")
    print(f"[rlt] args={job_args}")
    print(f"[rlt] secrets={secrets}")
    if args.use_rerun and args.detach:
        print(
            "[rlt] Rerun stays on this machine, so --detach disables it. "
            "Submit without --detach to watch each rollout live."
        )
    if args.dry_run:
        print("[rlt] dry run: no image build or job submission")
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
    print(f"[rlt] job status: {result.status}")
    if result.return_value is not None:
        print(f"[rlt] result: {result.return_value}")


if __name__ == "__main__":
    main()
