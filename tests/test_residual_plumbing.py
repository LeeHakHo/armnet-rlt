from __future__ import annotations

from pathlib import Path

import pytest

from armnet_rlt.learner_config import learner_config


def _cfg(tmp_path, **overrides):
    return learner_config(
        data_root=tmp_path,
        embodiment="so101",
        config_name="pi05_rlt_busybox_multitask_singlearm_minmax",
        run_id="r",
        device="cpu",
        **overrides,
    )


def test_learner_config_passes_the_residual_flags_for_rlt(tmp_path) -> None:
    network = _cfg(
        tmp_path,
        residual_actor=True,
        residual_scale=0.5,
        critic_action_residual=True,
        critic_action_gain=5.0,
    ).network

    assert network.residual_actor is True
    assert network.residual_scale == 0.5
    assert network.critic_action_residual is True
    assert network.critic_action_gain == 5.0
    assert network.reference_action_len == 30


def test_learner_config_defaults_keep_the_original_architecture(tmp_path) -> None:
    network = _cfg(tmp_path).network

    assert network.residual_actor is False
    assert network.critic_action_residual is False


def test_dsrl_ignores_the_rlt_residual_flags(tmp_path) -> None:
    network = _cfg(
        tmp_path, algorithm="dsrl", residual_actor=True, critic_action_residual=True
    ).network

    assert network.residual_actor is False
    assert network.critic_action_residual is False


def test_rlt_local_online_accepts_the_flags() -> None:
    from armnet_rlt.local_app import _parser

    args = _parser().parse_args(
        ["online", "--residual-actor", "--critic-action-residual", "--residual-scale", "0.3"]
    )

    assert args.residual_actor is True
    assert args.critic_action_residual is True
    assert args.residual_scale == 0.3
    assert args.critic_action_gain == 10.0


def test_modal_learner_forwards_the_flags() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "src" / "armnet_rlt" / "modal_app.py").read_text()

    for name in (
        "residual_actor",
        "residual_scale",
        "critic_action_residual",
        "critic_action_gain",
    ):
        # serve_learner -> _config and main -> serve_learner.remote.
        assert source.count(f"{name}={name},") == 2


def test_resume_rejects_residual_flags_that_differ_from_the_checkpoint(tmp_path) -> None:
    from dataclasses import asdict, replace

    from armnet_rlt.config import RLTConfig, so101_network_config
    from armnet_rlt.learner import _check_resumed_network

    plain = RLTConfig(network=so101_network_config(hidden_dims=(8,)))
    residual = replace(
        plain,
        network=replace(plain.network, residual_actor=True, critic_action_residual=True),
    )
    path = tmp_path / "step_000010.pt"

    _check_resumed_network(asdict(residual), residual, path)
    _check_resumed_network(asdict(plain), plain, path)
    older = asdict(plain)
    for name in ("residual_actor", "residual_scale", "critic_action_residual", "critic_action_gain"):
        older["network"].pop(name)
    _check_resumed_network(older, plain, path)
    with pytest.raises(ValueError, match="critic_action_residual=True in the checkpoint"):
        _check_resumed_network(asdict(residual), plain, path)
    with pytest.raises(ValueError, match="residual_actor=False in the checkpoint"):
        _check_resumed_network(older, residual, path)
