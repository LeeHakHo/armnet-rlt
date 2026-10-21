from types import SimpleNamespace

import pytest
import torch

save_file = pytest.importorskip("safetensors.torch").save_file

from armnet_rlt.act_actor import image_shapes, parse_image_size
from armnet_rlt.checkpoint_eval import _CheckpointWeights, saved_checkpoints
from armnet_rlt.so101_actor import _load_parameters


def _checkpoint(root, step, weight):
    model = root / "checkpoints" / f"{step:06d}" / "pretrained_model"
    model.mkdir(parents=True)
    save_file({"w": torch.full((2,), weight), "frozen": torch.zeros(3)}, model / "model.safetensors")
    return model


def test_saved_checkpoints_stop_at_last(tmp_path) -> None:
    first = _checkpoint(tmp_path, 5000, 1.0)
    _checkpoint(tmp_path, 10000, 2.0)
    assert saved_checkpoints(tmp_path) == {}
    (tmp_path / "checkpoints" / "last").symlink_to("005000")
    assert saved_checkpoints(tmp_path) == {5000: first}


def test_checkpoint_weights_send_only_changed_tensors(tmp_path) -> None:
    base = _checkpoint(tmp_path, 5000, 1.0)
    later = _checkpoint(tmp_path, 10000, 2.0)
    from safetensors.torch import load_file

    weights = _CheckpointWeights(load_file(str(base / "model.safetensors")))
    weights.load(later)
    assert set(weights.actor_state_bytes()) == {"w"}

    module = torch.nn.Module()
    module.w = torch.nn.Parameter(torch.ones(2))
    module.frozen = torch.nn.Parameter(torch.ones(3))
    step = _load_parameters(
        module,
        {
            "policy": weights.actor_state_bytes(),
            "learner_step": torch.tensor(10000),
            "partial": torch.tensor(True),
        },
    )
    assert step == 10000
    assert torch.equal(module.w.data, torch.full((2,), 2.0))
    assert torch.equal(module.frozen.data, torch.ones(3))
    with pytest.raises(ValueError, match="unknown parameters"):
        _load_parameters(
            module, {"policy": {"other": torch.zeros(1)}, "partial": torch.tensor(True)}
        )


def test_image_shapes_map_renamed_features_back_to_dataset_keys() -> None:
    feature = SimpleNamespace(shape=(3, 256, 256))
    policy = SimpleNamespace(
        config=SimpleNamespace(
            image_features={
                "observation.images.camera1": feature,
                "observation.images.front": feature,
            }
        )
    )
    rename = {"observation.images.top": "observation.images.camera1"}
    assert image_shapes(policy, rename_map=rename) == {
        "observation.images.top": (256, 256),
        "observation.images.front": (256, 256),
    }
    assert image_shapes(policy, rename_map=rename, size=parse_image_size("240x426")) == {
        "observation.images.top": (240, 426),
        "observation.images.front": (240, 426),
    }
    with pytest.raises(ValueError):
        parse_image_size("240")
