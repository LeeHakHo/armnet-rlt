"""Robometer progress as a shaping reward, for ``rlt-local online --reward-plugin
armnet_rlt.robometer_reward:shape`` with an ``--episode-frames`` actor.

Each episode's frames (top, front and wrist at up to 16 policy calls) go to a Robometer
tiled eval server (``POST /evaluate_videos``), which returns a progress value per
frame. Transition ``i`` runs from policy call ``i`` to ``i + 1``; its reward gains
``ROBOMETER_WEIGHT * (progress[i + 1] - progress[i])`` on top of the algorithm's own
reward, with the progress curve interpolated onto the transition boundaries.

Environment:
  ROBOMETER_URL     server base URL (http://127.0.0.1:8010)
  ROBOMETER_WEIGHT  scale of the shaping term (1.0)
  ROBOMETER_TASK    language instruction sent to the server (push the green button)
  ROBOMETER_LOG     JSON lines file for one record per episode (unset: no log)
  ROBOMETER_FRAMES  frames the server samples per episode (8, as the busybox checkpoint
                    was trained; on 60 labelled cell-03 episodes at the actor's ~2 fps it
                    separates successes from failures with AUC 0.976)

A failed request leaves the episode's rewards unchanged and is reported once per
episode on stdout.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

VIEWS = ("top", "front", "wrist")


def _video(jpegs: list[bytes], path: Path) -> None:
    """Writes JPEG frames to an H.264 file with ffmpeg."""
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "image2pipe", "-c:v", "mjpeg", "-framerate", "5",
         "-i", "-", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)],
        input=b"".join(jpegs),
        check=True,
    )


def progress_curve(frames: dict[str, list[bytes]]) -> tuple[list[float], dict]:
    """Per-frame progress from the server, and its full response."""
    import requests

    url = os.environ.get("ROBOMETER_URL", "http://127.0.0.1:8010").rstrip("/")
    count = min(len(frames.get(view, [])) for view in VIEWS)
    with tempfile.TemporaryDirectory() as tmp:
        files = {}
        for view in VIEWS:
            path = Path(tmp) / f"{view}.mp4"
            _video(frames[view][:count], path)
            files[f"view_{view}"] = (path.name, path.read_bytes(), "video/mp4")
        response = requests.post(
            f"{url}/evaluate_videos",
            data={
                "task": os.environ.get("ROBOMETER_TASK", "push the green button"),
                "max_frames": str(min(count, int(os.environ.get("ROBOMETER_FRAMES", "8")))),
                "primary_view": "top",
                "secondary_views": "front,wrist",
            },
            files=files,
            timeout=120,
            proxies={"http": None, "https": None},
        )
    response.raise_for_status()
    result = response.json()
    return [float(value) for value in result["outputs_progress"]["progress_pred"][0]], result


def boundary_progress(curve: list[float], transitions: int) -> np.ndarray:
    """Progress at the ``transitions + 1`` boundaries between policy calls,
    linearly interpolated from a curve over the same span."""
    if len(curve) == 1:
        return np.full(transitions + 1, curve[0])
    positions = np.linspace(0.0, len(curve) - 1, transitions + 1)
    return np.interp(positions, np.arange(len(curve)), np.asarray(curve, dtype=np.float64))


def shape(episode: list[dict[str, torch.Tensor]], frames: dict[str, list[bytes]] | None) -> None:
    """Adds the Robometer progress gain of each transition to its reward."""
    if not episode or not frames or any(len(frames.get(view, [])) < 2 for view in VIEWS):
        return
    weight = float(os.environ.get("ROBOMETER_WEIGHT", "1.0"))
    start = time.time()
    try:
        curve, result = progress_curve(frames)
    except Exception as exc:  # the episode keeps the algorithm's reward
        print(f"[robometer] reward request failed, episode unshaped: {exc}", flush=True)
        return
    points = boundary_progress(curve, len(episode))
    for index, fields in enumerate(episode):
        gain = weight * float(points[index + 1] - points[index])
        fields["reward"] = fields["reward"] + torch.tensor(gain, dtype=torch.float32)
    success = (result.get("outputs_success") or {}).get("success_probs") or [[None]]
    record = {
        "transitions": len(episode),
        "frames": min(len(frames[view]) for view in VIEWS),
        "progress_first": curve[0],
        "progress_last": curve[-1],
        "success_prob_last": (success[0] or [None])[-1],
        "shaping_total": weight * float(points[-1] - points[0]),
        "latency_s": round(time.time() - start, 2),
    }
    log = os.environ.get("ROBOMETER_LOG")
    if log:
        with open(log, "a") as handle:
            handle.write(json.dumps(record) + "\n")
    print(f"[robometer] {json.dumps(record)}", flush=True)
