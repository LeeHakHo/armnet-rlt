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
