from __future__ import annotations

from dataclasses import asdict

import torch

from armnet_rlt.config import RLTConfig, so101_network_config
from armnet_rlt.frozen import (
    config_from_frozen_checkpoint,
    freeze_learner_checkpoint,
    load_frozen_checkpoint,
)
from armnet_rlt.policy import RLTPolicy
from armnet_rlt.so101_actor import (
    FrozenActorTransport,
    _paired_eval_summary,
    _policy_variant,
)


def _learner_checkpoint() -> dict:
    network = so101_network_config(
        hidden_dims=(8,),
        reference_action_len=3,
        predicted_action_len=3,
        fixed_action_std=0.0,
    )
    config = RLTConfig(
        network=network,
        actions_to_execute=3,
        use_subsampling=False,
        policy_uses_delta_actions=True,
    )
    policy = RLTPolicy(config)
    return {
        "step": 123,
        "policy_state_dict": policy.state_dict(),
        "config": asdict(config),
        "actor_optimizer": {"discarded": True},
    }


def test_freeze_keeps_only_actor_and_inference_config(tmp_path) -> None:
    artifact = freeze_learner_checkpoint(
        _learner_checkpoint(),
        source={"run_id": "run-1"},
    )

    assert artifact["learner_step"] == 123
    assert artifact["source"]["run_id"] == "run-1"
    assert artifact["actor_state_dict"]
    assert all(
        not name.startswith("actor.")
        for name in artifact["actor_state_dict"]
    )
    assert "actor_optimizer" not in artifact

    path = tmp_path / "frozen_actor.pt"
    torch.save(artifact, path)
    loaded = load_frozen_checkpoint(path)
    config = config_from_frozen_checkpoint(loaded)
    assert config.network.hidden_dims == (8,)
    assert config.network.predicted_action_len == 3
    assert config.policy_uses_delta_actions is True


def test_frozen_dsrl_checkpoint_restores_its_noise_actor(tmp_path) -> None:
    from armnet_rlt.dsrl import DSRLActor, DSRLPolicy

    network = so101_network_config(
        hidden_dims=(8,), reference_action_len=3, predicted_action_len=3
    )
    config = RLTConfig(
        network=network,
        algorithm="dsrl",
        dsrl_noise_bound=1.5,
        dsrl_noise_steps=1,
        actions_to_execute=2,
        use_subsampling=False,
        max_demo_pretrain_steps=0,
        online_batch_fraction=1.0,
        policy_uses_delta_actions=True,
    )
    policy = DSRLPolicy(config)
    artifact = freeze_learner_checkpoint(
        {"step": 9, "policy_state_dict": policy.state_dict(), "config": asdict(config)}
    )
    path = tmp_path / "frozen_actor.pt"
    torch.save(artifact, path)
    restored = config_from_frozen_checkpoint(load_frozen_checkpoint(path))

    assert restored.algorithm == "dsrl"
    assert restored.dsrl_noise_bound == 1.5
    assert restored.actions_to_execute == 2
    assert restored.network.predicted_action_len == 3
    actor = DSRLActor(restored.network, noise_bound=restored.dsrl_noise_bound)
    actor.load_state_dict(artifact["actor_state_dict"])
    assert float(actor.noise_bound) == 1.5


def test_freeze_blanks_the_auth_token_of_older_checkpoints() -> None:
    checkpoint = _learner_checkpoint()
    checkpoint["config"]["actor_learner"]["auth_token"] = "learner-secret"

    artifact = freeze_learner_checkpoint(checkpoint)

    assert artifact["config"]["actor_learner"]["auth_token"] == ""
    assert checkpoint["config"]["actor_learner"]["auth_token"] == "learner-secret"


def test_residual_mode_survives_freezing() -> None:
    checkpoint = _learner_checkpoint()
    network = so101_network_config(
        hidden_dims=(8,),
        reference_action_len=3,
        predicted_action_len=3,
        residual_actor=True,
        residual_scale=0.5,
    )
    config = RLTConfig(
        network=network, actions_to_execute=3, use_subsampling=False
    )
    checkpoint["config"] = asdict(config)
    checkpoint["policy_state_dict"] = RLTPolicy(config).state_dict()

    artifact = freeze_learner_checkpoint(checkpoint)
    rebuilt = config_from_frozen_checkpoint(artifact)

    assert rebuilt.network.residual_actor is True
    assert rebuilt.network.residual_scale == 0.5
    assert bool(artifact["actor_state_dict"]["residual_mode"])


def test_freeze_rejects_missing_actor_state() -> None:
    checkpoint = _learner_checkpoint()
    checkpoint["policy_state_dict"] = {"critic.weight": torch.ones(1)}

    try:
        freeze_learner_checkpoint(checkpoint)
    except ValueError as exc:
        assert "no actor parameters" in str(exc)
    else:
        raise AssertionError("missing actor state was accepted")


def test_frozen_transport_never_updates_or_sends() -> None:
    artifact = freeze_learner_checkpoint(_learner_checkpoint())
    transport = FrozenActorTransport(artifact)

    transport.start()
    payload = transport.wait_for_initial_parameters()
    assert payload["learner_step"] == 123
    assert payload["policy"] is artifact["actor_state_dict"]
    assert transport.latest_parameters() is None
    transport.send_episode(b"ignored", b"ignored")
    transport.close()


def test_paired_frozen_eval_counterbalances_pair_order() -> None:
    assert _policy_variant(
        frozen_eval=True, include_base=True, rollout_index=1
    ) == "base"
    assert _policy_variant(
        frozen_eval=True, include_base=True, rollout_index=2
    ) == "frozen_rlt"
    assert _policy_variant(
        frozen_eval=True, include_base=True, rollout_index=3
    ) == "frozen_rlt"
    assert _policy_variant(
        frozen_eval=True, include_base=True, rollout_index=4
    ) == "base"
    assert _policy_variant(
        frozen_eval=True, include_base=False, rollout_index=1
    ) == "frozen_rlt"


def test_paired_summary_reports_matched_delta() -> None:
    summary = _paired_eval_summary(
        [
            {"policy_variant": "base", "success": False},
            {"policy_variant": "frozen_rlt", "success": True},
            {"policy_variant": "frozen_rlt", "success": True},
            {"policy_variant": "base", "success": True},
        ]
    )

    assert summary is not None
    assert summary["n_pairs"] == 2
    assert summary["base_successes"] == 1
    assert summary["frozen_rlt_successes"] == 2
    assert summary["frozen_rlt_only"] == 1
    assert summary["both_success"] == 1
    assert summary["delta_percentage_points"] == 50.0


def test_a_frozen_learner_checkpoint_rebuilds_the_actor_in_the_policy_server(tmp_path) -> None:
    from fakes import FakeBase, observation

    from armnet_rlt.algorithm import AlgorithmSpec, actor_meta
    from armnet_rlt.algorithms.rlt import RLTAlgorithm
    from armnet_rlt.frozen import FROZEN_SCHEMA_VERSION, frozen_actor_payload
    from armnet_rlt.policy_actor import AlgorithmPolicy

    network = so101_network_config(
        hidden_dims=(8,), reference_action_len=30, predicted_action_len=10
    )
    config = RLTConfig(
        network=network, actions_to_execute=10, use_subsampling=False,
        policy_uses_delta_actions=True, device="cpu",
    )
    algorithm = RLTAlgorithm(AlgorithmSpec(), config)
    checkpoint = {
        "step": 40,
        "algorithm": "rlt",
        "algorithm_state": algorithm.state_dict(),
        "actor_state": algorithm.actor_state(),
        "actor_meta": actor_meta("rlt", algorithm.spec, config),
        "config": asdict(config),
    }
    path = tmp_path / "frozen_actor.pt"
    torch.save(freeze_learner_checkpoint(checkpoint), path)
    artifact = load_frozen_checkpoint(path)
    assert artifact["schema_version"] == FROZEN_SCHEMA_VERSION
    assert artifact["algorithm"] == "rlt"

    runner = AlgorithmPolicy(FakeBase(), actor_source="checkpoint")
    assert runner.install(frozen_actor_payload(artifact)) == 40
    result = runner.act(observation(), explore=False)
    assert result.chunk.shape == (10, 6) and result.actor_step == 40
    for name, value in algorithm.actor_state().items():
        assert torch.equal(runner.actor.state_dict()[name], value)


def test_a_schema_1_frozen_actor_gets_its_actor_meta(tmp_path) -> None:
    import json

    artifact = freeze_learner_checkpoint(_learner_checkpoint())
    artifact["schema_version"] = 1
    del artifact["actor_meta"], artifact["algorithm"]
    path = tmp_path / "old.pt"
    torch.save(artifact, path)

    loaded = load_frozen_checkpoint(path)

    meta = json.loads(loaded["actor_meta"])
    assert loaded["algorithm"] == "rlt" and meta["algorithm"] == "rlt"
    assert meta["config"]["network"]["predicted_action_len"] == 3
    assert meta["spec"]["delta_actions"] is True
