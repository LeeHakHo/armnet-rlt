from argparse import Namespace

import pytest

from armnet_rlt.submit_eval import build_job_args


def _args(**updates) -> Namespace:
    values = {
        "config_name": "pi05_rlt_busybox_multitask_singlearm_minmax",
        "checkpoint_dir": "volume://openpi/checkpoints/multitask",
        "frozen_rlt_checkpoint": "volume://rlt/frozen/actor.pt",
        "language_instruction": "push the green button",
        "num_rollouts": 20,
        "include_base": True,
        "variation": True,
        "variation_seed": 42,
        "variation_episode_offset": 0,
        "record_dataset": True,
        "record_dataset_repo_id": None,
        "hf_user": None,
        "no_push_to_hub": False,
        "robot_telemetry": "full",
        "robot_telemetry_strict": True,
        "use_rerun": True,
        "detach": False,
    }
    values.update(updates)
    return Namespace(**values)


def test_frozen_eval_is_paired_deterministic_and_volume_backed() -> None:
    result = build_job_args(_args())

    assert result["checkpoint_dir"].startswith("volume://")
    assert result["frozen_rlt_checkpoint"].startswith("volume://")
    assert result["eval_include_base"] is True
    assert result["num_rollouts"] == 20
    assert result["exploration_scale"] == 0.0
    assert result["variation"] is True
    assert result["variation_seed"] == 42
    assert result["use_rerun"] is True


def test_frozen_eval_can_skip_base() -> None:
    result = build_job_args(_args(include_base=False))
    assert result["eval_include_base"] is False


@pytest.mark.parametrize(
    "field",
    ["checkpoint_dir", "frozen_rlt_checkpoint"],
)
def test_frozen_eval_requires_volume_paths(field: str) -> None:
    with pytest.raises(ValueError, match="volume://"):
        build_job_args(_args(**{field: "/tmp/local"}))


def test_frozen_eval_rejects_invalid_rollout_count() -> None:
    with pytest.raises(ValueError, match="positive"):
        build_job_args(_args(num_rollouts=0))


def test_scheduled_cell_only_pins_a_single_cell() -> None:
    from armnet_rlt.resources import scheduled_cell

    assert scheduled_cell("cell-01") == "cell-01"
    assert scheduled_cell(" cell-01 ") == "cell-01"
    assert scheduled_cell("") is None
    assert scheduled_cell("cell-01,cell-04") is None


def test_variation_axes_are_off_by_default_and_overridable() -> None:
    plain = build_job_args(_args())
    assert "variation_cameras" not in plain

    fixed_cameras = build_job_args(_args(variation_cameras=False))
    assert fixed_cameras["variation_cameras"] is False
    assert fixed_cameras["variation"] is True

    only_rail = build_job_args(_args(variation=False, variation_rail=True))
    assert only_rail["variation"] is True
    assert only_rail["variation_rail"] is True


def test_eval_cli_accepts_the_axis_flags() -> None:
    from armnet_rlt.submit_eval import _parser

    args = _parser().parse_args(
        [
            "--task", "push_green_button",
            "--config-name", "cfg",
            "--checkpoint-dir", "volume://openpi/checkpoints/x",
            "--frozen-rlt-checkpoint", "volume://rlt/frozen/a.pt",
            "--language-instruction", "push the green button",
            "--armnet.variation", "--no-armnet.variation-cameras",
        ]
    )
    assert args.variation_cameras is False
    assert args.variation_rail is None
