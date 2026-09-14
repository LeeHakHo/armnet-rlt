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
