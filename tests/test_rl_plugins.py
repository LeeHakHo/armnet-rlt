"""The example algorithm plugins register through their entry points, train on
synthetic transitions, and serve their actor through the policy server."""

from __future__ import annotations

import math
import socket
import sys
import tomllib
from pathlib import Path

import grpc
import numpy as np
import pytest
import torch

from armnet_rlt.algorithm import (
    ActorInput,
    AlgorithmSpec,
    actor_meta,
    actor_payload,
    load_algorithm,
    norm_stats_lists,
)
from armnet_rlt.artifacts import NormStats
from armnet_rlt.config import RLTConfig
from armnet_rlt.learner import _batch_from_transitions, _decode_episode_images, _transition_fields
from armnet_rlt.learner_config import network_for
from armnet_rlt.policy_actor import AlgorithmPolicy
from armnet_rlt.remote_policy import RemoteOpenPIRLTPolicy, encode_image, serve
from armnet_rlt.so101_actor import _auth_interceptor
from fakes import FakeBase, observation

PLUGINS = Path(__file__).resolve().parents[1] / "plugins"


@pytest.fixture
def installed(tmp_path, monkeypatch):
    """Both plugins on sys.path with their entry points, as an install does."""
    for package in ("rl_algorithm_example", "rl_algorithm_from_scratch"):
        project = tomllib.loads((PLUGINS / package / "pyproject.toml").read_text())["project"]
        points = project["entry-points"]["armnet_rlt.algorithms"]
        info = tmp_path / f"{package}-{project['version']}.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {package}\nVersion: {project['version']}\n"
        )
        (info / "entry_points.txt").write_text(
            "[armnet_rlt.algorithms]\n"
            + "".join(f"{name} = {target}\n" for name, target in points.items())
        )
        monkeypatch.syspath_prepend(str(PLUGINS / package))
    monkeypatch.syspath_prepend(str(tmp_path))
    yield
    for name in ("rl_algorithm_example", "rl_algorithm_from_scratch"):
        sys.modules.pop(name, None)


def _config(algorithm: str, **options) -> RLTConfig:
    network, jerk = network_for("so101", reference_action_len=30, predicted_action_len=10)
    return RLTConfig(
        network=network,
        jerk_joint_weights=jerk,
        algorithm=algorithm,
        algorithm_options=options,
        actions_to_execute=10,
        use_subsampling=False,
        max_demo_pretrain_steps=0,
        online_batch_fraction=1.0,
        policy_uses_delta_actions=True,
        device="cpu",
    )


def _spec(cls, config: RLTConfig) -> AlgorithmSpec:
    from armnet_rlt.algorithm import spec_for

    stats = {
        "state": NormStats(np.zeros(6, np.float32), np.full(6, 20.0, np.float32)),
        "actions": NormStats(np.zeros((30, 6), np.float32), np.ones((30, 6), np.float32)),
    }
    return spec_for(config, cls, stats)


def _wire_episode(*, base: bool, images: bool, size: int = 96, length: int = 6) -> list[dict]:
    """One episode as the cell sends it (lerobot Transition dicts)."""
    generator = torch.Generator().manual_seed(0)
    wire = []
    for index in range(length):
        state = {
            "rl_token": torch.randn(1, 2048 if base else 0, generator=generator),
            "proprioception": torch.randn(1, 6, generator=generator) * 20,
            "reference_action": torch.randn(1, 180 if base else 0, generator=generator) * 20,
            "curriculum_scale": torch.tensor([0.2]),
            "safety_trip": torch.tensor([0.0]),
        }
        if images:
            for view in ("top", "front", "wrist"):
                pixels = np.full((size, size, 3), 20 * index + 5, dtype=np.uint8)
                state[f"image/{view}"] = torch.from_numpy(encode_image(pixels, "jpeg"))
                if index == length - 1:
                    state[f"next_image/{view}"] = torch.from_numpy(encode_image(pixels, "jpeg"))
        wire.append(
            {
                "state": state,
                "action": torch.randn(1, 60, generator=generator) * 20,
                "reward": 1.0 if index == length - 1 else 0.0,
                "next_state": {
                    "rl_token": torch.randn(1, 2048 if base else 0, generator=generator),
                    "proprioception": torch.randn(1, 6, generator=generator) * 20,
                    "reference_action": torch.randn(1, 180 if base else 0, generator=generator) * 20,
                },
                "done": index == length - 1,
            }
        )
    return wire


def _train(algorithm, wire: list[dict], steps: int) -> list[dict[str, float]]:
    episode = [_transition_fields(item) for item in wire]
    _decode_episode_images(episode)
    torch.manual_seed(0)
    return [
        algorithm.update(_batch_from_transitions(episode, 8, torch.device("cpu")))
        for _ in range(steps)
    ]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _remote(port: int) -> RemoteOpenPIRLTPolicy:
    channel = grpc.intercept_channel(
        grpc.insecure_channel(f"127.0.0.1:{port}"),
        _auth_interceptor([("authorization", "Bearer secret")]),
    )
    return RemoteOpenPIRLTPolicy(channel, image_codec="jpeg")


def test_the_examples_register_under_their_entry_point_names(installed) -> None:
    td3 = load_algorithm("residual_td3")
    sac = load_algorithm("pixel_sac")
    assert td3.__name__ == "ResidualTD3" and td3.uses_base and not td3.needs_images
    assert sac.__name__ == "PixelSAC" and not sac.uses_base and sac.needs_images
    assert load_algorithm("rl_algorithm_example:ResidualTD3") is td3
    with pytest.raises(ValueError, match="residual_td3"):
        load_algorithm("no_such_algorithm")


def test_residual_td3_trains_and_serves_its_actor(installed) -> None:
    from armnet_rlt.learner_config import learner_config

    cls = load_algorithm("residual_td3")
    config = _config("residual_td3", hidden=32)
    torch.manual_seed(0)
    algorithm = cls(_spec(cls, config), config)
    before = {name: value.clone() for name, value in algorithm.actor_state().items()}
    metrics = _train(algorithm, _wire_episode(base=True, images=False), steps=4)
    assert all(math.isfinite(m["critic_loss"]) for m in metrics)
    assert "actor_loss" in metrics[1] and math.isfinite(metrics[1]["actor_loss"])
    after = algorithm.actor_state()
    assert any(not torch.equal(before[name], after[name]) for name in before)
    assert algorithm.state_dict()["steps"] == 4

    base = FakeBase()
    port = _free_port()
    runner = AlgorithmPolicy(base, algorithm="residual_td3")
    with serve(runner, bind=f"127.0.0.1:{port}", auth_token="secret"):
        remote = _remote(port)
        assert remote.actor_source == "client" and remote.algorithm == "residual_td3"
        step = remote.set_steering(
            actor_payload(after, actor_meta("residual_td3", algorithm.spec, config), learner_step=4)
        )
        result = remote.act(observation(state0=0.4), explore=False)
    assert step == 4 and result.actor_step == 4
    for name, value in after.items():
        assert torch.equal(runner.actor.state_dict()[name], value)
    # The served chunk is π0.5's first ten rows plus the learner actor's residual.
    actor = algorithm.make_actor().eval()
    seen = observation(state0=0.4)
    with torch.no_grad():
        expected = actor.act(
            ActorInput(
                proprioception=torch.from_numpy(seen.state),
                rl_token=torch.from_numpy(base.infer(seen).rl_token),
            ),
            torch.from_numpy(base.infer(seen).action_chunk[:30]),
            explore=False,
        ).chunk
    np.testing.assert_allclose(result.chunk, expected.numpy(), rtol=1e-6, atol=1e-6)
    assert result.base_chunk.shape == (30, 6) and result.extras["residual"].shape == (60,)

    # rlt-local online fills the plugin's learner config.
    plugin_config = learner_config(
        data_root=PLUGINS.parent / "experiment-data",
        embodiment="so101",
        config_name="pi05_rlt_busybox_multitask_singlearm_minmax",
        run_id="plugin",
        algorithm="residual_td3",
        algorithm_options={"hidden": 32},
        device="cpu",
    )
    assert plugin_config.max_demo_pretrain_steps == 0
    assert plugin_config.online_batch_fraction == 1.0
    assert plugin_config.algorithm_options == {"hidden": 32}


def test_pixel_sac_trains_on_images_and_serves_without_the_base(installed) -> None:
    cls = load_algorithm("pixel_sac")
    config = _config("pixel_sac", hidden=32, features=16)
    spec = _spec(cls, config)
    assert spec.image_size == (96, 96)
    torch.manual_seed(0)
    algorithm = cls(spec, config)
    before = {name: value.clone() for name, value in algorithm.actor_state().items()}
    metrics = _train(algorithm, _wire_episode(base=False, images=True), steps=3)
    assert all(math.isfinite(value) for m in metrics for value in m.values())
    after = algorithm.actor_state()
    assert any(not torch.equal(before[name], after[name]) for name in before)

    with pytest.raises(ValueError, match="--algorithm"):
        AlgorithmPolicy(None)
    runner = AlgorithmPolicy(None, algorithm="pixel_sac")
    port = _free_port()
    with serve(runner, bind=f"127.0.0.1:{port}", auth_token="secret"):
        remote = _remote(port)
        assert remote.uses_base is False and remote.needs_images is True
        assert remote._image_size == (96, 96)
        remote.set_steering(
            actor_payload(after, actor_meta("pixel_sac", spec, config), learner_step=3)
        )
        seen = observation(size=96)
        result = remote.act(seen, explore=True, exploration_scale=0.5)
        again = remote.act(seen, explore=False)
        assert remote.act(seen, explore=False).chunk.tolist() == again.chunk.tolist()
    assert result.base_chunk is None and result.rl_token is None
    assert result.chunk.shape == (10, 6)
    assert set(result.frames) == {"front", "top", "wrist"}
    # Within action_scale state stds of the current joints.
    bound = 0.3 * 20.0 + 1e-4
    assert np.abs(result.chunk - seen.state[None]).max() <= bound
