import json
from argparse import Namespace
from pathlib import Path

import pytest

from armnet_rlt import local_app
from armnet_rlt.learner_config import learner_config, resolve_init_checkpoint
from armnet_rlt.resources import (
    local_rendezvous_path,
    read_local_rendezvous,
    task_paths,
)
from armnet_rlt.submit_actor import build_job_args, resolve_rendezvous


ROOT = Path(__file__).resolve().parents[1]
CONFIG = "pi05_rlt_busybox_multitask_singlearm_minmax"


def test_local_modules_do_not_import_modal() -> None:
    for name in ("local_app", "learner_config", "cache_builder", "resources"):
        source = (ROOT / "src" / "armnet_rlt" / f"{name}.py").read_text()
        assert "import modal" not in source, name


def test_task_paths_follow_the_given_root(tmp_path) -> None:
    cache, assets, output = task_paths(CONFIG, tmp_path)
    assert cache == str(tmp_path / "tasks" / CONFIG / "demo" / "rlt_demo_cache.pt")
    assert assets == str(tmp_path / "tasks" / CONFIG / "ckpt" / "assets")
    assert output == str(tmp_path / "tasks" / CONFIG / "rlt_checkpoints")


def test_local_config_writes_under_the_data_root(tmp_path) -> None:
    cfg = learner_config(
        data_root=tmp_path, embodiment="so101", config_name=CONFIG, run_id="run-1"
    )
    assert cfg.checkpoint_dir == tmp_path / "tasks" / CONFIG / "rlt_checkpoints" / "run-1"
    assert cfg.policy_uses_delta_actions is True


def test_init_checkpoint_defaults_to_the_latest_step(tmp_path) -> None:
    cfg = learner_config(
        data_root=tmp_path, embodiment="so101", config_name=CONFIG, run_id="run-2"
    )
    assert resolve_init_checkpoint(cfg, "") is None
    for step in (500, 5000):
        path = cfg.checkpoint_dir.parent / "run-1" / f"step_{step:06d}"
        path.mkdir(parents=True)
        (path / "checkpoint.pt").write_bytes(b"")
    latest = resolve_init_checkpoint(cfg, "run-1")
    assert latest is not None and latest.parent.name == "step_005000"
    assert resolve_init_checkpoint(cfg, "run-1", 500).parent.name == "step_000500"
    with pytest.raises(FileNotFoundError):
        resolve_init_checkpoint(cfg, "run-1", 123)


def test_submitter_prefers_a_local_learner_and_pins_its_certificate(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("RLT_DATA_ROOT", str(tmp_path))
    key = f"{CONFIG}:run-1"
    assert read_local_rendezvous(key) is None
    path = local_rendezvous_path(key)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "host": "0.tcp.ngrok.io",
                "port": 12345,
                "use_tls": True,
                "config_name": CONFIG,
                "run_id": "run-1",
                "tls_server_name": local_app.TLS_SERVER_NAME,
                "tls_root_cert_pem": "-----BEGIN CERTIFICATE-----\n",
            }
        )
    )
    entry = resolve_rendezvous(key)
    args = Namespace(
        config_name=CONFIG,
        learner_key=key,
        language_instruction="push the green button",
        reference_action_len=30,
        checkpoint_dir=None,
        hf_checkpoint_repo="org/checkpoint",
        hf_checkpoint_revision="",
        num_rollouts=3,
        use_rerun=False,
        detach=True,
    )
    job_args = build_job_args(args, entry)
    assert job_args["learner_host"] == "0.tcp.ngrok.io"
    assert job_args["learner_port"] == 12345
    assert job_args["use_tls"] is True
    assert job_args["tls_server_name"] == local_app.TLS_SERVER_NAME
    assert job_args["tls_root_cert_pem"].startswith("-----BEGIN CERTIFICATE")


def test_self_signed_certificate_names_the_learner(tmp_path) -> None:
    certificate, key = local_app.ensure_tls_certificate(tmp_path / "tls")
    assert "BEGIN CERTIFICATE" in certificate.read_text()
    assert key.stat().st_mode & 0o077 == 0
    assert local_app.ensure_tls_certificate(tmp_path / "tls") == (certificate, key)


def test_streaming_bc_requires_a_route_for_the_cell() -> None:
    args = local_app._parser().parse_args(["streaming-bc", "--tunnel", "none"])
    with pytest.raises(SystemExit):
        with local_app._public_endpoint(args):
            pass
    args = local_app._parser().parse_args(
        ["streaming-bc", "--public-endpoint", "learner.example.org:50051"]
    )
    with local_app._public_endpoint(args) as endpoint:
        assert endpoint == ("learner.example.org", 50051)
