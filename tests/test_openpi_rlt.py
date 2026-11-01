from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from armnet_rlt.openpi_rlt import (
    MOTOR_NAMES,
    _build_policy_obs,
    build_policy_observation,
    uses_delta_actions,
)


def _raw_obs(*, wrist: bool = True) -> dict:
    result = {f"{name}.pos": index + 0.5 for index, name in enumerate(MOTOR_NAMES)}
    result["front"] = np.zeros((8, 10, 3), dtype=np.uint8)
    if wrist:
        result["wrist"] = np.ones((4, 6, 3), dtype=np.uint8)
    return result


def test_single_arm_observation_mapping_with_optional_wrist() -> None:
    observation = build_policy_observation(_raw_obs(wrist=False), "pick")
    assert observation.state.dtype == np.float32
    assert observation.state.tolist() == [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    assert set(observation.images) == {"front"}

    plain = _build_policy_obs(_raw_obs(wrist=True), "pick")
    assert plain["prompt"] == "pick"
    assert plain["observation.images.front"].shape == (8, 10, 3)
    assert plain["observation.images.wrist"].shape == (4, 6, 3)


def test_three_camera_observation_forwards_top_view() -> None:
    raw = _raw_obs()
    raw["top"] = np.full((5, 7, 3), 2, dtype=np.uint8)

    plain = _build_policy_obs(raw, "push the green button")

    assert plain["observation.images.top"].shape == (5, 7, 3)


def test_front_camera_is_required() -> None:
    raw = _raw_obs()
    raw.pop("front")
    with pytest.raises(KeyError, match="front"):
        build_policy_observation(raw, "pick")


def test_all_six_joint_positions_are_required() -> None:
    raw = _raw_obs()
    raw.pop("gripper.pos")
    with pytest.raises(KeyError, match="gripper.pos"):
        build_policy_observation(raw, None)


def test_delta_mode_comes_from_openpi_data_factory() -> None:
    config = SimpleNamespace(data=SimpleNamespace(use_delta_actions=True))
    materialized_data_config = SimpleNamespace()

    assert uses_delta_actions(config, materialized_data_config) is True

