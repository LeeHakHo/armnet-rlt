from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from armnet_rlt.local_cache import (
    _parser,
    build_cache,
    limit_episodes,
    output_paths,
)
from armnet_rlt.resources import task_paths


def test_output_layout_matches_the_learner_task_paths(tmp_path: Path) -> None:
    config = "pi05_rlt_busybox_multitask_singlearm_minmax"

    cache, assets = output_paths(tmp_path, config)
    learner_cache, learner_assets, _ = task_paths(config)

    assert str(cache.relative_to(tmp_path)) == learner_cache.removeprefix("/data/")
    assert str(assets.relative_to(tmp_path)) == learner_assets.removeprefix(
        "/data/"
    )


def test_limit_episodes_keeps_a_prefix_and_indexing() -> None:
    class Dataset:
        meta = SimpleNamespace(episodes=[{"length": 3}, {"length": 2}, {"length": 4}])

        def __getitem__(self, index: int) -> int:
            return index * 10

    limited = limit_episodes(Dataset(), 2)

    assert limited.meta.episodes == [{"length": 3}, {"length": 2}]
    assert limited[4] == 40
    assert limit_episodes(Dataset(), None).meta.episodes[-1] == {"length": 4}
    with pytest.raises(ValueError, match="positive"):
        limit_episodes(Dataset(), 0)


def test_existing_cache_is_not_overwritten_without_force(tmp_path: Path) -> None:
    args = _parser().parse_args(["--out-dir", str(tmp_path)])
    cache, _ = output_paths(tmp_path, args.config_name)
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"real cache")

    with pytest.raises(FileExistsError, match="--force"):
        build_cache(args)
    assert cache.read_bytes() == b"real cache"


def test_smoke_runs_use_a_separate_file_even_with_force(tmp_path: Path) -> None:
    args = _parser().parse_args(
        ["--out-dir", str(tmp_path), "--max-episodes", "2", "--force"]
    )
    cache, _ = output_paths(tmp_path, args.config_name)
    cache.parent.mkdir(parents=True)
    partial = cache.with_name("rlt_demo_cache.partial.pt")
    partial.write_bytes(b"old partial")
    cache.write_bytes(b"real cache")
    args2 = _parser().parse_args(
        ["--out-dir", str(tmp_path), "--max-episodes", "2"]
    )

    with pytest.raises(FileExistsError, match="partial"):
        build_cache(args2)
    assert cache.read_bytes() == b"real cache"
