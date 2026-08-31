"""Small append-only JSONL helpers for durable RLT metrics."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


def append_jsonl(
    path: str | Path, record: dict[str, Any], *, sync: bool = False
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        json.dump(record, stream, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        if sync:
            os.fsync(stream.fileno())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        return []
    records: list[dict[str, Any]] = []
    with target.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{target}:{line_number} must contain a JSON object"
                )
            records.append(value)
    return records


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    yield from read_jsonl(path)
