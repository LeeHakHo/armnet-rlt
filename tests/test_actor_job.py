from __future__ import annotations

import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from armnet_rlt import actor_job
from armnet_rlt.so101_actor import (
    LearnerTransport,
    _delta_action_mode,
    _load_parameters,
    _refine_chunk,
)


def test_checkpoint_root_prefers_highest_step(tmp_path: Path) -> None:
    for step in (5, 20):
        (tmp_path / f"step_{step}" / "params").mkdir(parents=True)
    assert actor_job._checkpoint_root(tmp_path) == str(tmp_path / "step_20")


def test_bimanual_is_rejected_before_actor_setup(monkeypatch) -> None:
    monkeypatch.setattr(actor_job, "require_so101_embodiment", lambda *_: True)
    with pytest.raises(NotImplementedError, match="single-arm"):
        actor_job.run(SimpleNamespace())


def test_current_cell_fields_are_mapped_into_actor_config(
    monkeypatch, tmp_path: Path
) -> None:
    (tmp_path / "params").mkdir()
    captured = {}

    class Network:
        def __init__(self, **kwargs):
            self.action_dim = 6
            self.proprioception_dim = 6
            self.rl_token_dim = 2048
            self.reference_action_len = kwargs["reference_action_len"]
            self.predicted_action_len = kwargs["predicted_action_len"]

    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.actor_learner = SimpleNamespace()

    config_module = types.ModuleType("armnet_rlt.config")
    config_module.SO101_JERK_JOINT_WEIGHTS = [1.0] * 6
    config_module.RLTConfig = Config
    config_module.so101_network_config = lambda **kwargs: Network(**kwargs)

    actor_module = types.ModuleType("armnet_rlt.so101_actor")

    def fake_run_actor(ctx, cfg, *, operator):
        captured.update(ctx=ctx, cfg=cfg, operator=operator)
        return {"status": "ok"}

    actor_module.run_actor = fake_run_actor
    monkeypatch.setitem(sys.modules, "armnet_rlt.config", config_module)
    monkeypatch.setitem(sys.modules, "armnet_rlt.so101_actor", actor_module)
    monkeypatch.setattr(actor_job, "require_so101_embodiment", lambda *_: False)

    cell = SimpleNamespace(
        robot_port="tcp://edge:9876",
        robot_id="cell-so101",
        calibration_file_path=None,
        calibration_dir=tmp_path / "calibration",
        safety_limit=20.0,
        language_instruction="cell task",
    )
    ctx = SimpleNamespace(
        args={
            "checkpoint_dir": str(tmp_path),
            "learner_host": "learner.example",
            "num_rollouts": 2,
        },
        cell=cell,
        camera_configs={"front": object(), "wrist": object()},
        task="fallback task",
        secrets={"RLT_LEARNER_AUTH_TOKEN": "secret"},
        volume=SimpleNamespace(root=None),
        cache_home=None,
        report_progress=lambda _message: None,
    )

    assert actor_job.run(ctx) == {"status": "ok"}
    cfg = captured["cfg"]
    assert cfg.network.action_dim == 6
    assert cfg.network.reference_action_len == 30
    assert cfg.so101.robot_port == "tcp://edge:9876"
    assert cfg.so101.robot_id == "cell-so101"
    assert cfg.so101.cameras == ctx.camera_configs
    assert cfg.so101.max_relative_target == 20.0
    assert cfg.actor_learner.learner_host == "learner.example"
    assert cfg.actor_learner.auth_token == "secret"


def test_tls_pem_is_written_to_a_temporary_file() -> None:
    pem = "-----BEGIN CERTIFICATE-----\nabc\n-----END CERTIFICATE-----\n"
    path, temporary = actor_job._tls_root_cert_file({"tls_root_cert_pem": pem})
    try:
        assert Path(path).read_text() == pem
        assert temporary == path
    finally:
        Path(path).unlink()


def test_actor_exploration_is_smoothed_across_chunk(monkeypatch) -> None:
    class Actor:
        action_std = torch.ones(60)

        def __call__(self, *_args, **_kwargs):
            return None, torch.zeros(1, 60)

        def unnormalize_action(self, value, _proprioception):
            return value

    alternating = torch.tensor(
        [1.0, -1.0] * 30, dtype=torch.float32
    ).reshape(1, 10, 6)
    monkeypatch.setattr(
        torch,
        "randn",
        lambda *_args, **_kwargs: alternating.clone(),
    )

    chunk = _refine_chunk(
        Actor(),
        torch.zeros(2048),
        torch.zeros(6),
        torch.zeros(30, 6),
        predicted_len=10,
        action_dim=6,
        exploration_correlation=0.85,
    )

    second_difference = chunk[2:] - 2 * chunk[1:-1] + chunk[:-2]
    assert second_difference.abs().max() < 2.0


def test_actor_exploration_can_be_disabled(monkeypatch) -> None:
    class Actor:
        action_std = torch.ones(60)

        def __call__(self, *_args, **_kwargs):
            return None, torch.zeros(1, 60)

        def unnormalize_action(self, value, _proprioception):
            return value

    monkeypatch.setattr(
        torch, "randn", lambda *_args, **_kwargs: torch.ones(1, 10, 6)
    )
    chunk = _refine_chunk(
        Actor(),
        torch.zeros(2048),
        torch.zeros(6),
        torch.zeros(30, 6),
        predicted_len=10,
        action_dim=6,
        exploration_correlation=0.85,
        exploration_scale=0.0,
    )

    assert torch.count_nonzero(chunk) == 0


def test_delta_action_mode_must_match_openpi_and_learner() -> None:
    cfg = SimpleNamespace(policy_uses_delta_actions=True)
    policy = SimpleNamespace(uses_delta_actions=True)
    assert _delta_action_mode(cfg, policy) is True

    policy.uses_delta_actions = False
    with pytest.raises(ValueError, match="delta-action mismatch"):
        _delta_action_mode(cfg, policy)


def test_parameter_envelope_exposes_learner_step(monkeypatch) -> None:
    transport = types.ModuleType("lerobot.transport")
    utils = types.ModuleType("lerobot.transport.utils")
    utils.bytes_to_state_dict = lambda _payload: {
        "policy": {"weight": torch.tensor([1.0])},
        "learner_step": torch.tensor(321),
    }
    monkeypatch.setitem(sys.modules, "lerobot.transport", transport)
    monkeypatch.setitem(sys.modules, "lerobot.transport.utils", utils)

    class Actor:
        def load_state_dict(self, state):
            assert state["weight"].item() == 1.0

    assert _load_parameters(Actor(), b"payload") == 321


def test_transport_close_waits_for_outbound_messages() -> None:
    transport = LearnerTransport(SimpleNamespace())
    transport._transitions.put(b"transition")
    transport._interactions.put(b"interaction")

    def consume(queue):
        time.sleep(0.02)
        queue.get()
        queue.task_done()

    workers = [
        threading.Thread(target=consume, args=(queue,))
        for queue in (transport._transitions, transport._interactions)
    ]
    for worker in workers:
        worker.start()
    transport.close()
    for worker in workers:
        worker.join()

    assert transport._transitions.unfinished_tasks == 0
    assert transport._interactions.unfinished_tasks == 0

