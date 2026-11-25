"""Leaderboard submissions and scoring.

  python scripts/leaderboard.py submit --name NAME --run RUN --cell CELL [--tag TAG] [--code URL] [--notes TEXT]
      Opens a pull request on the leaderboard dataset with the run's recorded evaluation
      datasets per checkpoint. scripts/run.sh --leaderboard calls it after the last evaluation.

  python scripts/leaderboard.py score [--dry-run]
      Maintainers: checks open submissions against the protocol from their episode labels
      (meta/armnet_rollouts.jsonl), merges the valid ones, comments on and closes the rest,
      and rebuilds data/leaderboard.json and data/rollouts.json.

An entry is a method name plus a tag. A submission with the same name, tag and cell replaces
the earlier one. An entry is listed once it has every checkpoint on both cells; ID and OOD
each use its best checkpoint, summed over the cells. Training successes per block of 100
episodes and the exploration scale come from the run's rollout_metrics.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def load_dotenv(path: Path) -> None:
    """Sets the KEY=VALUE lines of path that the environment does not set."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().removeprefix("export ").partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key or key.startswith("#") or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        os.environ[key] = value


load_dotenv(Path(__file__).resolve().parents[1] / ".env")
LEADERBOARD = os.environ.get("LEADERBOARD_REPO", "AutoRLBench/armnet-leaderboard")
CELLS = ("cell-01", "cell-03")
CHECKPOINTS = (100, 200, 300, 400)
SEEDS = (1042, 2042, 3042)
PER_SEED = {"id": 10, "ood": 10}
# Conditions counted as OOD; ood-easy and ood-hard are the OOD levels of earlier submissions.
OOD = ("ood", "ood-easy", "ood-hard")
# Scene per cell: ID rail range (OOD: 0.05 below or above it), light, top-camera aim.
ID_RAIL = {"cell-01": (0.55, 0.65), "cell-03": (0.65, 0.75)}
LIGHT = {"id": (30,), "ood": (20, 40)}
TOP_AIM = {"cell-01": {"top_pan": 0.405, "top_tilt": 0.175}, "cell-03": {"top_tilt": 0.1661}}
OOD_TILT = 0.1358


def scene_error(cell: str, cond: str, v: dict) -> str | None:
    """Why an ID or OOD episode's scene is not the protocol's, or None."""
    rail = (v.get("values") or {}).get("rail")
    lo, hi = ID_RAIL[cell]
    ranges = [(lo, hi)] if cond == "id" else [(lo - 0.05, lo), (hi, hi + 0.05)]
    if rail is None or not any(a - 1e-3 <= float(rail) <= b + 1e-3 for a, b in ranges):
        return f"rail {rail} outside {ranges}"
    if v.get("brightness") is None or round(float(v["brightness"])) not in LIGHT[cond]:
        return f"light {v.get('brightness')} not in {LIGHT[cond]}"
    for key, want in TOP_AIM[cell].items():
        if key == "top_tilt" and cond == "ood":
            want += OOD_TILT
        if v.get(key) is not None and abs(float(v[key]) - want) > 1e-3:
            return f"{key} {v[key]} not {want:.4f}"
    return None


DATA = Path(os.environ.get("RLT_DATA_ROOT") or Path(__file__).resolve().parents[1] / "experiment-data")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def entry_key(name: str, tag: str | None) -> str:
    return slug(name) + (f"--{slug(tag)}" if tag else "")


def eval_datasets(run: str) -> dict[int, list[str]]:
    """Recorded evaluation datasets of each checkpoint of RUN, from its cell job logs."""
    found: dict[int, list[str]] = {}
    for out in sorted((DATA / "eval").glob(f"{run}-ep*")):
        episodes = int(out.name.rsplit("-ep", 1)[1])
        repos = []
        for log in sorted(out.glob("*/job*.log")):
            text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(errors="replace"))
            # Job args, or the result line of a job that finished.
            repos += re.findall(r"'record_dataset_repo_id': '([^']+)'", text)
            repos += re.findall(r'"eval_dataset_url": "https://huggingface.co/datasets/([^"]+)"', text)
        if repos:
            found[episodes] = sorted(set(repos))
    return found


def training_curve(run: str) -> tuple[dict[str, list[int]], float] | None:
    """Successes of each block of 100 training episodes ending at a checkpoint, and the
    largest exploration noise scale of the run, from its rollout_metrics.jsonl."""
    paths = sorted(DATA.glob(f"tasks/*/rlt_checkpoints/{run}/rollout_metrics.jsonl")) \
        + sorted(DATA.glob(f"lerobot-sbc/{run}/rollout_metrics.jsonl"))
    if not paths:
        return None
    rows = [json.loads(line) for line in paths[0].read_text().splitlines() if line.strip()]
    blocks = {}
    for end in CHECKPOINTS:
        block = rows[end - 100:end]
        if block:
            blocks[str(end)] = [sum(bool(r.get("success")) for r in block), len(block)]
    return blocks, max(float(r.get("exploration_scale") or 0.0) for r in rows)


def submit(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi, repo_exists

    datasets = {ep: [r for r in repos if repo_exists(r, repo_type="dataset")]
                for ep, repos in eval_datasets(args.run).items()}
    datasets = {ep: repos for ep, repos in datasets.items() if repos}
    missing = [c for c in CHECKPOINTS if c not in datasets]
    if missing:
        sys.exit(f"{args.run} has no recorded evaluation after {missing} episodes; "
                 "evaluations must be recorded (RECORD_REPO set, RECORD not 0)")
    submission = {
        "name": args.name, "tag": args.tag, "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "code": args.code, "notes": args.notes, "cell": args.cell, "run": args.run,
        "evaluations": {str(c): datasets[c] for c in CHECKPOINTS},
    }
    training = training_curve(args.run)
    if training:
        submission["training"], submission["training_exploration"] = training
    path = f"submissions/{entry_key(args.name, args.tag)}/{args.cell}.json"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(submission, handle, indent=1)
    commit = HfApi().upload_file(
        path_or_fileobj=handle.name, path_in_repo=path, repo_id=LEADERBOARD, repo_type="dataset",
        commit_message=f"Submit {args.name} on {args.cell} ({args.run})", create_pr=True,
    )
    print(f"submitted {args.name} on {args.cell}: {commit.pr_url}")


def labels(repo: str) -> tuple[list[dict], list[dict]]:
    """Episode labels and episode video spans (meta/episodes) of a recorded dataset."""
    import pandas as pd
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(repo, repo_type="dataset", allow_patterns=["meta/*", "meta/**"]))
    rows = [json.loads(l) for l in (root / "meta/armnet_rollouts.jsonl").read_text().splitlines() if l.strip()]
    files = sorted((root / "meta/episodes").glob("*/*.parquet"))
    spans = pd.concat(pd.read_parquet(p) for p in files).sort_values("episode_index").to_dict("records") if files else []
    return rows, spans


def evaluation_episodes(rows: list[dict], episodes: int) -> list[tuple[int, dict]]:
    """(episode index, label) of the evaluation after ``episodes`` training episodes: every
    episode of a one-job dataset, or the matching phase "eval" labels of a run's dataset."""
    indexed = [(int(r.get("episode_index", i)), r) for i, r in enumerate(rows)]
    if not any("phase" in r for r in rows):
        return indexed
    return [(i, r) for i, r in indexed if r.get("phase") == "eval" and r.get("checkpoint_episodes") == episodes]


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
        counts = {"id": [0, 0], "ood": [0, 0], "ood-easy": [0, 0], "ood-hard": [0, 0], "trips": 0}
        per_seed = Counter()
        for repo in repos:
            labelled, spans = labels(repo)
            spans = {int(s.get("episode_index", j)): s for j, s in enumerate(spans)}
            for i, r in evaluation_episodes(labelled, episodes):
                v = r.get("variation") or {}
                cond = v.get("condition")
                if cond not in counts or v.get("seed") not in SEEDS:
                    raise ValueError(f"{repo} episode {i}: condition {cond!r}, seed {v.get('seed')!r}")
                if r.get("cell_id") not in (None, "", submission["cell"]):
                    raise ValueError(f"{repo} episode {i} ran on {r.get('cell_id')}, not {submission['cell']}")
                if cond in LIGHT and (err := scene_error(submission["cell"], cond, v)):
                    raise ValueError(f"{repo} episode {i} ({cond}): {err}")
                counts[cond][0] += bool(r.get("success"))
                counts[cond][1] += 1
                counts["trips"] += r.get("outcome") == "safety_trip"
                per_seed[(v["seed"], "id" if cond == "id" else "ood")] += 1
                video = None
                if i in spans:
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
    for e in sorted(entries, key=lambda e: e.get("date") or ""):
        key = entry_key(e["name"], e.get("tag"))
        m = methods.setdefault(key, {"name": e["name"], "tag": e.get("tag") or "", "code": e.get("code"),
                                     "notes": e.get("notes"), "cells": {}, "rows": {}, "date": "", "training": {},
                                     "exploration": None})
        m["cells"][e["cell"]] = e["evals"]
        if e.get("training"):
            m["training"][e["cell"]] = e["training"]
            m["exploration"] = e.get("training_exploration")
        m["rows"][e["cell"]] = [{**r, "method": key, "name": e["name"], "tag": e.get("tag") or ""} for r in e["rows"]]
        m["date"] = max(m["date"], e.get("date") or "")
    board, rollouts = [], []
    for key, m in methods.items():
        rollouts += [r for rows in m["rows"].values() for r in rows]
        if set(m["cells"]) != set(CELLS):
            continue
        entry = {"key": key, "name": m["name"], "tag": m["tag"], "date": m["date"], "detail": m["notes"] or "", "open_source": bool(m["code"]),
                 "code": m["code"], "best": {}, "curve": {c: {str(k): v for k, v in m["cells"][c].items()} for c in sorted(m["cells"])}}
        if m["training"]:
            entry["training"] = {c: m["training"][c] for c in sorted(m["training"])}
            entry["training_exploration"] = m["exploration"]
        for part, conds in (("id", ("id",)), ("ood", OOD)):
            total = lambda ep, i: sum(m["cells"][c][ep].get(x, [0, 0])[i] for c in CELLS for x in conds)
            best = max(CHECKPOINTS, key=lambda ep: total(ep, 0))
            entry["best"][part] = [total(best, 0), total(best, 1), best]
        board.append(entry)
    return board, rollouts


CLIPS = os.environ.get("CLIPS_REPO", "AutoRLBench/armnet-benchmark-rollouts")


def clips(repo: str, existing: set[str], episodes: set[int] | None = None) -> dict[int, str]:
    """Real-time clips (top above front and wrist, 640x540) of a recorded dataset's episodes,
    timed by meta/armnet_frame_times.jsonl when present, else stretched to duration_s.
    Uploads missing ones to CLIPS; returns {episode index: URL}."""
    import subprocess

    from huggingface_hub import HfApi, snapshot_download

    prefix = f"clips/v2/{repo.replace('/', '__')}/"
    labelled, span_rows = labels(repo)
    rows = {int(r.get("episode_index", i)): r for i, r in enumerate(labelled)}
    spans = {int(s.get("episode_index", i)): s for i, s in enumerate(span_rows)}
    urls = {i: f"https://huggingface.co/datasets/{CLIPS}/resolve/main/{prefix}{i:03d}.mp4"
            for i in sorted(rows.keys() & spans.keys()) if episodes is None or i in episodes}
    todo = [i for i in urls if f"{prefix}{i:03d}.mp4" not in existing]
    if not todo:
        return urls
    files = {f"videos/observation.images.{view}/chunk-{int(spans[i][f'videos/observation.images.{view}/chunk_index']):03d}/"
             f"file-{int(spans[i][f'videos/observation.images.{view}/file_index']):03d}.mp4"
             for i in todo for view in ("top", "front", "wrist")}
    root = Path(snapshot_download(repo, repo_type="dataset", allow_patterns=["meta/**", *sorted(files)], max_workers=4))
    times_file = root / "meta/armnet_frame_times.jsonl"
    frame_times = {}
    if times_file.exists():
        for line in times_file.read_text().splitlines():
            row = json.loads(line)
            frame_times[int(row["episode_index"])] = row["frame_times_s"]
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / prefix
        out_dir.mkdir(parents=True)
        for i in todo:
            s, parts = spans[i], []
            for view in ("top", "front", "wrist"):
                key = f"videos/observation.images.{view}"
                src = root / key / f"chunk-{int(s[key + '/chunk_index']):03d}" / f"file-{int(s[key + '/file_index']):03d}.mp4"
                parts += ["-ss", f"{float(s[key + '/from_timestamp']):.3f}", "-to", f"{float(s[key + '/to_timestamp']):.3f}", "-i", str(src)]
            key = "videos/observation.images.top"
            recorded = float(s[key + "/to_timestamp"]) - float(s[key + "/from_timestamp"])
            stretch = (rows[i].get("duration_s") or recorded) / recorded if recorded > 0 else 1.0
            tile = ("[0:v]scale=640:360,setsar=1[t];[1:v]scale=320:180,setsar=1[f];[2:v]scale=320:180,setsar=1[w];"
                    "[f][w]hstack=2[b];[t][b]vstack=2:shortest=1")
            out = out_dir / f"{i:03d}.mp4"
            encode = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
            times = frame_times.get(i)
            if times and len(times) > 1:
                frames = Path(tmp) / f"frames-{i}"
                frames.mkdir()
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *parts, "-filter_complex", tile + "[v]",
                                "-map", "[v]", "-q:v", "3", str(frames / "%05d.jpg")], check=True)
                images = sorted(frames.glob("*.jpg"))[: len(times)]
                durations = [b - a for a, b in zip(times, times[1:])] + [0.05]
                listing = frames / "list.txt"
                listing.write_text("".join(f"file '{img}'\nduration {max(d, 0.001):.4f}\n" for img, d in zip(images, durations))
                                   + f"file '{images[-1]}'\n")
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                                "-vf", "fps=20", *encode, str(out)], check=True)
                shutil.rmtree(frames)
            else:
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *parts, "-filter_complex",
                                tile + f",setpts={stretch:.4f}*PTS,fps=20[v]", "-map", "[v]", *encode, str(out)], check=True)
        HfApi().upload_folder(folder_path=tmp, allow_patterns=[f"{prefix}*.mp4"], repo_id=CLIPS, repo_type="dataset",
                              commit_message=f"Episode videos of {repo}")
    return urls


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
    # A submission without a date takes its file's commit date.
    files = list_repo_files(LEADERBOARD, repo_type="dataset")
    dates = {}
    if any(f.startswith("submissions/") for f in files):
        dates = {item.path: item.last_commit.date.strftime("%Y-%m-%d")
                 for item in api.list_repo_tree(LEADERBOARD, path_in_repo="submissions", recursive=True, expand=True, repo_type="dataset")
                 if getattr(item, "last_commit", None)}
    for f in files:
        if f.startswith("submissions/") and f.endswith(".json"):
            sub = json.loads(Path(hf_hub_download(LEADERBOARD, f, repo_type="dataset")).read_text())
            sub.setdefault("date", dates.get(f, ""))
            evals, rows = check(sub)
            entries.append({**sub, "evals": evals, "rows": rows})
        elif f.startswith("legacy/") and f.endswith(".json"):
            # Results without recorded episode labels, added by the maintainers.
            for e in json.loads(Path(hf_hub_download(LEADERBOARD, f, repo_type="dataset")).read_text()):
                entries.append({**e, "evals": {int(k): v for k, v in e["evals"].items()}})
    board, rollouts = build(entries)
    # NO_CLIPS=1 publishes the counts now and leaves episode videos to a later run.
    if not args.dry_run and os.environ.get("NO_CLIPS") != "1":
        existing = set(api.list_repo_files(CLIPS, repo_type="dataset"))
        urls: dict[str, dict[int, str]] = {}
        wanted: dict[str, set[int]] = {}
        for r in rollouts:
            if r.get("dataset"):
                wanted.setdefault(r["dataset"], set()).add(r["dataset_episode"])
        for r in rollouts:
            if r.get("dataset"):
                if r["dataset"] not in urls:
                    urls[r["dataset"]] = clips(r["dataset"], existing, wanted[r["dataset"]])
                r["video"] = urls[r["dataset"]].get(r["dataset_episode"], r.get("video"))
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
    p.add_argument("--tag", default="", help="separates entries of the same method (e.g. v2, lr3e-4)")
    p.add_argument("--code", default="")
    p.add_argument("--notes", default="")
    p = sub.add_parser("score")
    p.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    submit(args) if args.command == "submit" else score(args)


if __name__ == "__main__":
    main()
