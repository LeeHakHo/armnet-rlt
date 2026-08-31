from __future__ import annotations

import pytest

from armnet_rlt.config import (
    BISO101_JERK_JOINT_WEIGHTS,
    RLTConfig,
    RLTNetworkConfig,
    biso101_network_config,
)


def test_supported_presets_have_expected_dimensions() -> None:
    single = RLTConfig()
    assert single.embodiment == "so101"
    assert single.network.action_dim == 6
    assert single.network.delta_action_mask[-1] is False

    bimanual = RLTConfig.for_embodiment("biso101")
    assert bimanual.network.action_dim == 12
    assert bimanual.network.delta_action_mask[5] is False
    assert bimanual.network.delta_action_mask[11] is False
    assert bimanual.jerk_joint_weights == BISO101_JERK_JOINT_WEIGHTS


def test_rejects_unsupported_or_mismatched_embodiment() -> None:
    with pytest.raises(ValueError, match="embodiment"):
        RLTConfig(embodiment="arx5")
    with pytest.raises(ValueError, match="action_dim=6"):
        RLTConfig(network=biso101_network_config())


def test_validates_chunk_jerk_and_subsampling_settings() -> None:
    with pytest.raises(ValueError, match="actions_to_execute"):
        RLTConfig(actions_to_execute=9)
    with pytest.raises(ValueError, match="jerk_joint_weights"):
        RLTConfig(jerk_joint_weights=(1.0,))
    with pytest.raises(ValueError, match="divisible"):
        RLTConfig(sub_chunk_stride=3)


def test_network_validates_delta_mask_and_dimensions() -> None:
    with pytest.raises(ValueError, match="delta_action_mask"):
        RLTNetworkConfig(delta_action_mask=(True,))
    with pytest.raises(ValueError, match="2048"):
        RLTNetworkConfig(rl_token_dim=32)
