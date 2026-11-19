"""Leaderboard submissions and scoring.

  python scripts/leaderboard.py submit --name NAME --run RUN --cell CELL [--code URL] [--notes TEXT]
      Opens a pull request on the leaderboard dataset with the run's evaluations: for every
      checkpoint, the recorded public evaluation datasets. scripts/run.sh --leaderboard
      calls it after the last evaluation.

  python scripts/leaderboard.py score [--dry-run]
      Maintainers: checks every open submission against the protocol from the episode
      labels in its datasets (meta/armnet_rollouts.jsonl, written on the cell), merges the
      valid ones, comments on the rest, and rebuilds data/leaderboard.json and
      data/rollouts.json in the leaderboard dataset, which the project page reads.

A method is listed once it has every checkpoint on every leaderboard cell. ID and OOD each
use the method's best checkpoint, summed over the cells.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

LEADERBOARD = os.environ.get("LEADERBOARD_REPO", "AutoRLBench/armnet-leaderboard")
CELLS = ("cell-01", "cell-03")
CHECKPOINTS = (100, 200, 300, 400)
SEEDS = (1042, 2042, 3042)
PER_SEED = {"id": 10, "ood": 10}
OOD = ("ood-easy", "ood-hard")
DATA = Path(os.environ.get("RLT_DATA_ROOT") or Path(__file__).resolve().parents[1] / "experiment-data")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


# ----------------------------------------------------------------------------- submit

def eval_datasets(run: str) -> dict[int, list[str]]:
    """Recorded evaluation datasets of each checkpoint of RUN, from its cell job logs."""
    found: dict[int, list[str]] = {}
    for out in sorted((DATA / "eval").glob(f"{run}-ep*")):
        episodes = int(out.name.rsplit("-ep", 1)[1])
        repos = []
        for log in sorted(out.glob("*/job*.log")):
            text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(errors="replace"))
            repos += re.findall(r'"eval_dataset_url": "https://huggingface.co/datasets/([^"]+)"', text)
        if repos:
            found[episodes] = sorted(set(repos))
    return found


def submit(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    datasets = eval_datasets(args.run)
    missing = [c for c in CHECKPOINTS if c not in datasets]
    if missing:
        sys.exit(f"{args.run} has no recorded evaluation after {missing} episodes; "
                 "evaluations must be recorded (RECORD_REPO set, RECORD not 0)")
    submission = {
        "name": args.name, "code": args.code, "notes": args.notes, "cell": args.cell, "run": args.run,
        "evaluations": {str(c): datasets[c] for c in CHECKPOINTS},
    }
    path = f"submissions/{slug(args.name)}/{args.cell}__{args.run}.json"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(submission, handle, indent=1)
    commit = HfApi().upload_file(
        path_or_fileobj=handle.name, path_in_repo=path, repo_id=LEADERBOARD, repo_type="dataset",
        commit_message=f"Submit {args.name} on {args.cell} ({args.run})", create_pr=True,
    )
    print(f"submitted {args.name} on {args.cell}: {commit.pr_url}")


# ----------------------------------------------------------------------------- score

def labels(repo: str) -> tuple[list[dict], list[dict]]:
    """Episode labels and episode video spans (meta/episodes) of a recorded dataset."""
    import pandas as pd
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(repo, repo_type="dataset", allow_patterns=["meta/*", "meta/**"]))
    rows = [json.loads(l) for l in (root / "meta/armnet_rollouts.jsonl").read_text().splitlines() if l.strip()]
    files = sorted((root / "meta/episodes").glob("*/*.parquet"))
    spans = pd.concat(pd.read_parquet(p) for p in files).sort_values("episode_index").to_dict("records") if files else []
    return rows, spans


def check(submission: dict) -> tuple[dict, list[dict]]:
    """Counts per checkpoint and one row per rollout; raises ValueError when the
    submission does not follow the protocol."""
    if submission.get("cell") not in CELLS:
        raise ValueError(f"cell must be one of {CELLS}")
    evals, rows = {}, []
    for episodes in CHECKPOINTS:
        repos = submission["evaluations"].get(str(episodes)) or []
        if not repos:
            raise ValueError(f"no evaluation after {episodes} episodes")
        counts = {"id": [0, 0], "ood-easy": [0, 0], "ood-hard": [0, 0], "trips": 0}
        per_seed = Counter()
        for repo in repos:
            labelled, spans = labels(repo)
            for i, r in enumerate(labelled):
                v = r.get("variation") or {}
                cond = v.get("condition")
                if cond not in counts or v.get("seed") not in SEEDS:
                    raise ValueError(f"{repo} episode {i}: condition {cond!r}, seed {v.get('seed')!r}")
                if r.get("cell_id") not in (None, "", submission["cell"]):
                    raise ValueError(f"{repo} episode {i} ran on {r.get('cell_id')}, not {submission['cell']}")
                counts[cond][0] += bool(r.get("success"))
                counts[cond][1] += 1
                counts["trips"] += r.get("outcome") == "safety_trip"
                per_seed[(v["seed"], "id" if cond == "id" else "ood")] += 1
                video = None
                if i < len(spans):
                    s = spans[i]
                    key = "videos/observation.images.top"
                    video = (f"https://huggingface.co/datasets/{repo}/resolve/main/{key}/"
                             f"chunk-{int(s[key + '/chunk_index']):03d}/file-{int(s[key + '/file_index']):03d}.mp4"
                             f"#t={float(s[key + '/from_timestamp']):.2f},{float(s[key + '/to_timestamp']):.2f}")
                rows.append({
                    "cell": submission["cell"], "episodes": episodes, "condition": cond, "seed": v.get("seed"),
                    "rail": round(float((v.get("values") or {}).get("rail", 0)), 3), "light": v.get("brightness"),
                    "tilt": v.get("top_tilt"), "success": bool(r.get("success")), "outcome": r.get("outcome"),
                    "duration_s": round(float(r.get("duration_s") or 0), 1), "video": video,
                    "dataset": repo, "dataset_episode": i,
                })
        for seed in SEEDS:
            for part, need in PER_SEED.items():
                if per_seed[(seed, part)] != need:
                    raise ValueError(f"after {episodes} episodes, seed {seed} has {per_seed[(seed, part)]} "
                                     f"{part.upper()} rollouts, not {need}")
        evals[episodes] = counts
    return evals, rows


def build(entries: list[dict]) -> tuple[list[dict], list[dict]]:
    """Leaderboard rows and rollout rows from checked submissions grouped by method."""
    methods: dict[str, dict] = {}
    for e in entries:
        m = methods.setdefault(slug(e["name"]), {"name": e["name"], "code": e.get("code"), "notes": e.get("notes"), "cells": {}, "rows": []})
        m["cells"][e["cell"]] = e["evals"]
        m["rows"] += [{**r, "method": slug(e["name"])} for r in e["rows"]]
    board, rollouts = [], []
    for key, m in methods.items():
        rollouts += m["rows"]
        if set(m["cells"]) != set(CELLS):
            continue
        entry = {"key": key, "name": m["name"], "detail": m["notes"] or "", "open_source": bool(m["code"]),
                 "code": m["code"], "best": {}, "curve": {c: {str(k): v for k, v in m["cells"][c].items()} for c in sorted(m["cells"])}}
        for part, conds in (("id", ("id",)), ("ood", OOD)):
            total = lambda ep, i: sum(m["cells"][c][ep][x][i] for c in CELLS for x in conds)
            best = max(CHECKPOINTS, key=lambda ep: total(ep, 0))
            entry["best"][part] = [total(best, 0), total(best, 1), best]
        board.append(entry)
    return board, rollouts


def score(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files

    api = HfApi()
    for pr in api.get_repo_discussions(LEADERBOARD, repo_type="dataset", discussion_type="pull_request", discussion_status="open"):
        rev = f"refs/pr/{pr.num}"
        files = [f for f in list_repo_files(LEADERBOARD, repo_type="dataset", revision=rev) if f.startswith("submissions/")]
        try:
            for f in files:
                check(json.loads(Path(hf_hub_download(LEADERBOARD, f, repo_type="dataset", revision=rev)).read_text()))
        except Exception as exc:  # noqa: BLE001 - reported on the pull request
            print(f"PR #{pr.num} rejected: {exc}")
            if not args.dry_run:
                api.comment_discussion(LEADERBOARD, pr.num, f"Not listed: {exc}", repo_type="dataset")
                api.change_discussion_status(LEADERBOARD, pr.num, "closed", repo_type="dataset")
            continue
        print(f"PR #{pr.num} accepted")
        if not args.dry_run:
            api.merge_pull_request(LEADERBOARD, pr.num, repo_type="dataset")
    entries = []
    for f in list_repo_files(LEADERBOARD, repo_type="dataset"):
        if f.startswith("submissions/") and f.endswith(".json"):
            sub = json.loads(Path(hf_hub_download(LEADERBOARD, f, repo_type="dataset")).read_text())
            evals, rows = check(sub)
            entries.append({**sub, "evals": evals, "rows": rows})
        elif f.startswith("legacy/") and f.endswith(".json"):
            # Results from before episode labels were recorded, added by the maintainers.
            for e in json.loads(Path(hf_hub_download(LEADERBOARD, f, repo_type="dataset")).read_text()):
                entries.append({**e, "evals": {int(k): v for k, v in e["evals"].items()}})
    board, rollouts = build(entries)
    print(f"{len(board)} methods listed, {len(rollouts)} rollouts")
    if args.dry_run:
        return
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "data").mkdir()
        (Path(tmp) / "data/leaderboard.json").write_text(json.dumps(board, indent=1, ensure_ascii=False))
        (Path(tmp) / "data/rollouts.json").write_text(json.dumps(rollouts, separators=(",", ":")))
        api.upload_folder(folder_path=tmp, repo_id=LEADERBOARD, repo_type="dataset", commit_message="Rebuild the leaderboard")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("submit")
    p.add_argument("--name", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--cell", required=True)
    p.add_argument("--code", default="")
    p.add_argument("--notes", default="")
    p = sub.add_parser("score")
    p.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    submit(args) if args.command == "submit" else score(args)


if __name__ == "__main__":
    main()
