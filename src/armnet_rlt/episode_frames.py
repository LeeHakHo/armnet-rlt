"""Camera frames an actor sends with an episode, packed into one uint8 array."""

from __future__ import annotations

import io
from typing import Any

import numpy as np


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
