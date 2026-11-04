from __future__ import annotations

from pathlib import Path

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
