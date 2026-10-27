"""Needs LeRobot (tools/venv-act); skipped where it is not installed."""

import pytest
import torch

lerobot = pytest.importorskip("lerobot")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402

from armnet_rlt.act_gmm import ACTGMMConfig, ACTGMMPolicy, config_from_act, from_act_checkpoint  # noqa: E402


def _config(**overrides):
    values = dict(
        chunk_size=4,
        n_action_steps=4,
        dim_model=16,
        n_heads=2,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        latent_dim=4,
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        device="cpu",
    )
    values.update(overrides)
    return values


def _batch(batch_size=2):
    torch.manual_seed(0)
    return {
        "observation.state": torch.randn(batch_size, 6),
        "observation.images.front": torch.rand(batch_size, 3, 32, 32),
        "action": torch.randn(batch_size, 4, 6),
        "action_is_pad": torch.zeros(batch_size, 4, dtype=torch.bool),
    }


def test_act_gmm_trains_on_the_chunk_likelihood_and_executes_one_mode() -> None:
    torch.manual_seed(0)
    policy = ACTGMMPolicy(ACTGMMConfig(**_config(num_modes=3, init_std=0.2)))
    assert policy.config.type == "act_gmm"
    policy.train()
    loss, info = policy.forward(_batch())
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert {"nll_loss", "l1_loss", "kld_loss", "mode_entropy", "std_mean"} <= info.keys()
    loss.backward()
    assert policy.model.action_head.linear.weight.grad is not None

    chunk = policy.predict_action_chunk(_batch())
    assert chunk.shape == (2, 4, 6)
    weights = policy.mode_weights(_batch())
    assert weights.shape == (2, 3)
    # Sampling a mode by weight stays on one of the modes.
    policy.config.sample_mode = True
    sampled = policy.predict_action_chunk(_batch())
    assert sampled.shape == (2, 4, 6)


def test_act_gmm_warm_starts_from_an_act_checkpoint(tmp_path) -> None:
    torch.manual_seed(0)
    act = ACTPolicy(ACTConfig(**_config()))
    act.save_pretrained(tmp_path / "act")

    policy = from_act_checkpoint(tmp_path / "act", num_modes=2, init_std=0.3)
    assert isinstance(policy.config, ACTGMMConfig) and policy.config.num_modes == 2
    # Backbone and transformer weights carried over.
    for key, value in act.state_dict().items():
        if not key.startswith("model.action_head."):
            assert torch.equal(policy.state_dict()[key], value)
    # Every mode starts near the old head's prediction.
    act.eval()
    policy.eval()
    batch = _batch()
    with torch.no_grad():
        reference = act.predict_action_chunk(batch)
        chunk = policy.predict_action_chunk(batch)
    assert torch.allclose(chunk, reference, atol=0.5)

    # The act_gmm checkpoint round-trips through LeRobot's registry.
    policy.save_pretrained(tmp_path / "gmm")
    from lerobot.configs.policies import PreTrainedConfig

    loaded_config = PreTrainedConfig.from_pretrained(str(tmp_path / "gmm"))
    assert loaded_config.type == "act_gmm" and loaded_config.num_modes == 2
    loaded = ACTGMMPolicy.from_pretrained(str(tmp_path / "gmm"))
    with torch.no_grad():
        assert torch.allclose(loaded.predict_action_chunk(batch), chunk)


def test_config_from_act_copies_every_act_field() -> None:
    act = ACTConfig(**_config(kl_weight=3.0))
    config = config_from_act(act, num_modes=4)
    assert config.kl_weight == 3.0 and config.chunk_size == 4 and config.num_modes == 4
