"""Build the single-arm RLT demonstration cache on a local GPU.

This runs the same ``build_demo_cache`` and ``copy_norm_stats`` as
``modal_cache.py`` without Modal. It needs the VLA stack (OpenPI fork, JAX with
CUDA, LeRobot), which the project environment does not install, so run it from
an environment that has it::

    PYTHONPATH=src <cache-env>/bin/python -m armnet_rlt.local_cache \\
        --out-dir ~/.local/share/armnet-rlt/data

The output mirrors ``resources.task_paths`` under ``--out-dir``, so it can be
copied to a Modal volume later with ``modal volume put``::

    <out-dir>/tasks/<config>/demo/rlt_demo_cache.pt
    <out-dir>/tasks/<config>/ckpt/assets/norm_stats*.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any


MULTITASK_CONFIG = "pi05_rlt_busybox_multitask_singlearm_minmax"
MULTITASK_CHECKPOINT = "AutoRLBench/pi05_rlt_busybox_multitask_singlearm_minmax"
MULTITASK_DATASET = "AutoRLBench/busybox_multitask"
DEFAULT_OUT_DIR = Path("~/.local/share/armnet-rlt/data")


def output_paths(out_dir: Path, config_name: str) -> tuple[Path, Path]:
    """Return ``(demo cache file, assets dir)`` using the learner's layout."""
    root = Path(out_dir) / "tasks" / config_name
    return root / "demo" / "rlt_demo_cache.pt", root / "ckpt" / "assets"


class _EpisodeLimited:
    """View of a LeRobot dataset restricted to its first ``count`` episodes.

    ``build_demo_cache`` reads ``meta.episodes`` and indexes frames from a
    running offset starting at zero, so a prefix of the episodes is exact.
    """

    def __init__(self, dataset: Any, count: int) -> None:
        self._dataset = dataset
        self.meta = SimpleNamespace(
            episodes=list(dataset.meta.episodes)[:count]
        )

    def __getitem__(self, index: int) -> Any:
        return self._dataset[index]


def limit_episodes(dataset: Any, count: int | None) -> Any:
    if count is None:
        return dataset
    if count <= 0:
        raise ValueError("--max-episodes must be positive")
    return _EpisodeLimited(dataset, count)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"root of the output tree (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument("--config-name", default=MULTITASK_CONFIG)
    parser.add_argument("--checkpoint-repo", default=MULTITASK_CHECKPOINT)
    parser.add_argument("--dataset-repo", default=MULTITASK_DATASET)
    parser.add_argument(
        "--prompt",
        default=None,
        help="one fixed prompt for every episode; omit to use each episode's "
        "own task instruction (multitask)",
    )
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="only process the first N episodes (smoke test)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing cache",
    )
    return parser


def build_cache(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir.expanduser()
    cache_path, assets_dir = output_paths(out_dir, args.config_name)
    if args.max_episodes is not None:
        # A partial cache must never be mistaken for, or overwrite, the real one.
        cache_path = cache_path.with_name("rlt_demo_cache.partial.pt")
    if cache_path.exists() and not args.force:
        raise FileExistsError(
            f"{cache_path} already exists; pass --force to overwrite it"
        )

    # The image sets these; without them JAX takes 75% of the GPU up front.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")

    from huggingface_hub import snapshot_download
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from armnet_rlt.cache_builder import build_demo_cache, copy_norm_stats
    from armnet_rlt.openpi_rlt import OpenPIRLTPolicy

    checkpoint = snapshot_download(args.checkpoint_repo)
    print(f"[RLT cache] checkpoint={checkpoint}", flush=True)
    policy = OpenPIRLTPolicy.from_checkpoint(
        checkpoint,
        args.config_name,
        default_prompt=args.prompt or "",
    )
    # revision="main" skips LeRobot's codebase_version tag lookup.
    dataset = LeRobotDataset(
        args.dataset_repo, revision="main", video_backend="pyav"
    )
    dataset = limit_episodes(dataset, args.max_episodes)
    summary = build_demo_cache(
        dataset=dataset,
        policy=policy,
        output_path=cache_path,
        prompt=args.prompt,
        inference_batch_size=args.inference_batch_size,
    )
    copy_norm_stats(checkpoint, assets_dir)
    result = {
        **summary,
        "config_name": args.config_name,
        "dataset": args.dataset_repo,
        "checkpoint": args.checkpoint_repo,
        "max_episodes": args.max_episodes,
        "cache_path": str(cache_path),
        "assets_dir": str(assets_dir),
    }
    (cache_path.parent / "cache_summary.json").write_text(
        json.dumps(result, indent=2)
    )
    print(f"[RLT cache] complete: {result}", flush=True)
    return result


def main(argv: list[str] | None = None) -> None:
    build_cache(_parser().parse_args(argv))


if __name__ == "__main__":
    main()
