"""Run the offline (demo-only) RLT pretraining locally, without Modal.

Uses the same defaults as the Modal learner wrapper (``modal_app._config``):
5,000 pretrain steps, batch 256, delta actions, seed 42. It reads the cache and
norm stats written by ``armnet_rlt.local_cache`` and writes checkpoints every
``--save-freq`` steps, so the critic can be inspected as it trains::

    uv run python -m armnet_rlt.local_offline --run-id offline-baseline
    uv run rlt-analyze-q --checkpoint <run>/step_005000/checkpoint.pt ...

It needs only torch, so the project environment is enough.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

from armnet_rlt.config import RLTConfig, so101_network_config
from armnet_rlt.local_cache import DEFAULT_OUT_DIR, MULTITASK_CONFIG, output_paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="the --out-dir used by armnet_rlt.local_cache",
    )
    parser.add_argument("--config-name", default=MULTITASK_CONFIG)
    parser.add_argument(
        "--run-id",
        default=None,
        help="run directory name (default: offline-<UTC timestamp>)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="default: <data-dir>/tasks/<config>/rlt_checkpoints",
    )
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--save-freq", type=int, default=500)
    parser.add_argument("--log-freq", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    # Optional overrides for ablations; unset keeps the RLTConfig default.
    parser.add_argument("--bc-beta", type=float)
    parser.add_argument("--utd-ratio", type=int)
    parser.add_argument("--discount", type=float)
    parser.add_argument("--jerk-lambda", type=float)
    parser.add_argument("--max-q", type=float)
    parser.add_argument("--critic-lr", type=float)
    parser.add_argument("--actor-lr", type=float)
    # Structural variants (see RLTNetworkConfig).
    parser.add_argument(
        "--residual-actor",
        action="store_true",
        help="actor outputs reference + residual_scale * tanh(head), head starts at 0",
    )
    parser.add_argument("--residual-scale", type=float)
    parser.add_argument(
        "--critic-action-residual",
        action="store_true",
        help="critic sees gain * (action - reference chunk)",
    )
    parser.add_argument("--critic-action-gain", type=float)
    return parser


_OVERRIDES = (
    "bc_beta",
    "utd_ratio",
    "discount",
    "jerk_lambda",
    "max_q",
    "critic_lr",
    "actor_lr",
)


def build_config(args: argparse.Namespace) -> RLTConfig:
    data_dir = args.data_dir.expanduser()
    cache_path, assets_dir = output_paths(data_dir, args.config_name)
    output_dir = (
        args.output_dir.expanduser()
        if args.output_dir is not None
        else cache_path.parents[1] / "rlt_checkpoints"
    )
    run_id = args.run_id or dt.datetime.now(dt.UTC).strftime(
        "offline-%Y%m%d-%H%M%S"
    )
    values: dict[str, Any] = {
        "assets_dir": assets_dir,
        "demo_cache_path": cache_path,
        "output_dir": output_dir,
        "run_id": run_id,
        "device": args.device,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_demo_pretrain_steps": args.steps,
        # The Modal wrapper's value; unused when offline_only=True.
        "online_steps": 300_000,
        "policy_uses_delta_actions": True,
        "save_freq": args.save_freq,
        "log_freq": args.log_freq,
    }
    for name in _OVERRIDES:
        value = getattr(args, name)
        if value is not None:
            values[name] = value
    network_overrides: dict[str, Any] = {}
    if args.residual_actor:
        network_overrides["residual_actor"] = True
    if args.critic_action_residual:
        network_overrides["critic_action_residual"] = True
    for name in ("residual_scale", "critic_action_gain"):
        if getattr(args, name) is not None:
            network_overrides[name] = getattr(args, name)
    if network_overrides:
        values["network"] = so101_network_config(**network_overrides)
    return RLTConfig.for_embodiment("so101", **values)


def run(argv: list[str] | None = None) -> dict[str, object]:
    """Train and return the learner summary."""
    from armnet_rlt.learner import run_learner

    config = build_config(_parser().parse_args(argv))
    for path in (config.demo_cache_path, config.assets_dir):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{path} not found; build it with armnet_rlt.local_cache first"
            )
    print(f"[RLT offline] run dir: {config.checkpoint_dir}", flush=True)
    summary = run_learner(config, offline_only=True)
    print(f"[RLT offline] done: {summary}", flush=True)
    print(
        "[RLT offline] analyse with:\n"
        f"  uv run rlt-analyze-q --checkpoint {summary['checkpoint_path']} "
        f"--assets-dir {config.assets_dir} "
        f"--demo-cache {config.demo_cache_path}",
        flush=True,
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    # Console scripts hand main()'s return value to sys.exit, which would print
    # the summary again and exit with status 1.
    run(argv)


if __name__ == "__main__":
    main()
