from __future__ import annotations

import json
from pathlib import Path

import torch

from armnet_rlt.local_cache import output_paths
from armnet_rlt.local_offline import _parser, build_config, main, run


def test_defaults_match_the_modal_learner_wrapper(tmp_path: Path) -> None:
    config = build_config(_parser().parse_args(["--data-dir", str(tmp_path)]))

    assert config.max_demo_pretrain_steps == 5_000
    assert config.batch_size == 256
    assert config.policy_uses_delta_actions is True
    assert config.online_batch_fraction == 0.5
    assert config.network.reference_action_len == 30
    assert config.seed == 42
    cache, assets = output_paths(tmp_path, "pi05_rlt_busybox_multitask_singlearm_minmax")
    assert Path(config.demo_cache_path) == cache
    assert Path(config.assets_dir) == assets
    assert config.checkpoint_dir.parent == cache.parents[1] / "rlt_checkpoints"
    assert config.run_id.startswith("offline-")


def test_ablation_flags_override_only_what_is_given(tmp_path: Path) -> None:
    args = _parser().parse_args(
        [
            "--data-dir", str(tmp_path),
            "--run-id", "bc-low",
            "--bc-beta", "0.1",
            "--utd-ratio", "1",
        ]
    )

    config = build_config(args)

    assert config.run_id == "bc-low"
    assert config.bc_beta == 0.1
    assert config.utd_ratio == 1
    assert config.discount == 0.985


def test_main_fails_clearly_when_the_cache_is_missing(tmp_path: Path) -> None:
    try:
        run(["--data-dir", str(tmp_path), "--device", "cpu"])
    except FileNotFoundError as error:
        assert "local_cache" in str(error)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_tiny_run_writes_checkpoints_every_save_freq(tmp_path: Path) -> None:
    cache, assets = output_paths(
        tmp_path, "pi05_rlt_busybox_multitask_singlearm_minmax"
    )
    cache.parent.mkdir(parents=True)
    assets.mkdir(parents=True)
    stats = {
        key: {"mean": [0.0] * 6, "std": [1.0] * 6, "q01": [-1.0] * 6, "q99": [1.0] * 6}
        for key in ("state", "actions")
    }
    (assets / "norm_stats.json").write_text(json.dumps(stats))
    (assets / "norm_stats_actions_per_timestep.json").write_text(
        json.dumps({"actions": stats["actions"]})
    )
    transitions = []
    for index in range(4):
        terminal = index == 3
        transitions.append(
            {
                "rl_token": torch.randn(2048) * 0.01,
                "proprioception": torch.zeros(6),
                "reference_action": torch.zeros(180),
                "action": torch.zeros(60),
                "reward": torch.tensor(float(terminal)),
                "next_rl_token": torch.randn(2048) * 0.01,
                "next_proprioception": torch.zeros(6),
                "next_reference_action": torch.zeros(180),
                "done": torch.tensor(float(terminal)),
            }
        )
    torch.save(transitions, cache)

    summary = run(
        [
            "--data-dir", str(tmp_path),
            "--run-id", "tiny",
            "--device", "cpu",
            "--steps", "4",
            "--batch-size", "2",
            "--save-freq", "2",
            "--utd-ratio", "1",
        ]
    )

    run_dir = Path(summary["checkpoint_path"]).parents[1]
    assert sorted(p.name for p in run_dir.glob("step_*")) == [
        "step_000002",
        "step_000004",
    ]
    assert summary["completed_steps"] == 4


def test_main_returns_none_so_the_console_script_exits_cleanly(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        main(["--data-dir", str(tmp_path), "--device", "cpu"])
    assert main.__annotations__["return"] == "None"
