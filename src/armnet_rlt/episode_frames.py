"""Camera frames an actor sends with an episode, packed into one uint8 array."""

from __future__ import annotations

import io
from typing import Any

import numpy as np


# Frames kept per episode: 16 x 3 views of 426x240 JPEG is about 1.2 MB, well under
# the learner channel's 4 MB message limit (a 30 s episode has about 60 policy calls).
MAX_EPISODE_FRAMES = 16


def spread(frames: list, limit: int = MAX_EPISODE_FRAMES) -> list:
    """At most ``limit`` items, evenly spaced and including the first and last."""
    if len(frames) <= limit:
        return list(frames)
    return [frames[round(i * (len(frames) - 1) / (limit - 1))] for i in range(limit)]


def pack_episode_frames(frames: list[dict[str, np.ndarray]]) -> np.ndarray:
    """Encoded views at each policy call as one flat uint8 array: an
    ``np.savez`` archive with ``<view>_<index>`` entries."""
    buffer = io.BytesIO()
    np.savez(buffer, **{f"{view}_{i:04d}": data for i, step in enumerate(frames) for view, data in step.items()})
    return np.frombuffer(buffer.getvalue(), dtype=np.uint8).copy()


def unpack_episode_frames(packed: Any) -> dict[str, list[bytes]]:
    """The inverse of :func:`pack_episode_frames`: the encoded bytes per view, in order."""
    raw = np.asarray(packed, dtype=np.uint8).reshape(-1).tobytes()
    archive = np.load(io.BytesIO(raw))
    views: dict[str, list[tuple[int, bytes]]] = {}
    for key in archive.files:
        view, _, index = key.rpartition("_")
        views.setdefault(view, []).append((int(index), archive[key].tobytes()))
    return {view: [data for _, data in sorted(items)] for view, items in views.items()}
