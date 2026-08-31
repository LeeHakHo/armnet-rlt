from argparse import Namespace

import pytest

from armnet_rlt.submit_actor import SO101_EMBODIMENT, build_job_args


def _args(**updates):
    values = {
        "task": "push_green_button",
        "config_name": "pi05_rlt_so101_busybox",
        "checkpoint_dir": None,
        "hf_checkpoint_repo": "org/checkpoint",
        "hf_checkpoint_revision": "",
        "language_instruction": "Press the green button",
        "learner_key": "pi05_rlt_so101_busybox:run-1",
        "num_rollouts": 3,
        "reference_action_len": 30,
        "exploration_scale": 0.0,
        "variation": False,
        "variation_seed": 42,
        "variation_episode_offset": 0,
        "record_dataset": True,
        "record_dataset_repo_id": None,
        "hf_user": None,
        "no_push_to_hub": False,
        "robot_telemetry": "full",
        "robot_telemetry_strict": False,
    }
    values.update(updates)
    return Namespace(**values)


def _entry(**updates):
    values = {
        "host": "example.modal.run",
        "port": 443,
        "use_tls": True,
        "config_name": "pi05_rlt_so101_busybox",
        "run_id": "run-1",
    }
    values.update(updates)
    return values


def test_submitter_is_fixed_to_single_arm() -> None:
    assert SO101_EMBODIMENT == "lerobot/so-101"


def test_job_args_use_tls_modal_endpoint() -> None:
    result = build_job_args(_args(), _entry())
    assert result["learner_host"] == "example.modal.run"
    assert result["learner_port"] == 443
    assert result["use_tls"] is True
    assert result["num_rollouts"] == 3
    assert result["exploration_scale"] == 0.0
    assert result["record_dataset"] is True
    assert result["robot_telemetry"] == "full"


def test_submitter_rejects_wrong_learner_config() -> None:
    with pytest.raises(ValueError, match="not requested config"):
        build_job_args(_args(), _entry(config_name="other"))


def test_job_args_can_use_an_armnet_volume_checkpoint() -> None:
    result = build_job_args(
        _args(
            checkpoint_dir="volume://openpi/checkpoints/green",
            hf_checkpoint_repo=None,
        ),
        _entry(),
    )
    assert result["checkpoint_dir"] == "volume://openpi/checkpoints/green"
    assert "hf_checkpoint_repo" not in result


def test_job_args_enable_deterministic_variation() -> None:
    result = build_job_args(
        _args(
            variation=True,
            variation_seed=123,
            variation_episode_offset=7,
        ),
        _entry(),
    )
    assert result["variation"] is True
    assert result["variation_seed"] == 123
    assert result["variation_episode_offset"] == 7


def test_job_args_can_name_the_rlt_eval_dataset() -> None:
    result = build_job_args(
        _args(
            record_dataset_repo_id="villekuosmanen/eval_rlt_green",
            hf_user="villekuosmanen",
        ),
        _entry(),
    )
    assert result["record_dataset_repo_id"] == (
        "villekuosmanen/eval_rlt_green"
    )
    assert result["hf_user"] == "villekuosmanen"
