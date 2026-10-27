from __future__ import annotations

import json
import socket
from dataclasses import dataclass

import grpc
import numpy as np
import pytest

from armnet_rlt.openpi_rlt import PolicyObservation, PolicyOutput
from armnet_rlt.remote_policy import (
    MAX_MESSAGE_BYTES,
    RemoteOpenPIRLTPolicy,
    pack,
    serve,
    unpack,
)
from armnet_rlt.so101_actor import _auth_interceptor, _grpc_channel


@dataclass
class _Stats:
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray | None = None
    q99: np.ndarray | None = None


class _FakePolicy:
    rl_token_dim = 2048
    use_quantile_norm = True
    uses_delta_actions = True

    def __init__(self) -> None:
        self.norm_stats = {
            "state": _Stats(np.arange(6.0), np.ones(6), np.zeros(6), np.full(6, 2.0)),
            "actions": _Stats(np.zeros((30, 6)), np.ones((30, 6))),
        }
        self.observations: list[PolicyObservation] = []
        self.steered: list[tuple[float, bool]] = []
        self.steering_payloads: list[dict] = []
        self.steering_step: int | None = None
        self.resets = 0

    def infer(
        self,
        observation: PolicyObservation,
        *,
        exploration_std: float | None = None,
        steer: bool = True,
    ) -> PolicyOutput:
        self.observations.append(observation)
        if exploration_std is None:
            return PolicyOutput(
                action_chunk=np.full((30, 6), observation.state[0], dtype=np.float32),
                rl_token=np.arange(2048, dtype=np.float32),
            )
        self.steered.append((exploration_std, steer))
        return PolicyOutput(
            action_chunk=np.full((30, 6), -observation.state[0], dtype=np.float32),
            rl_token=np.arange(2048, dtype=np.float32),
            noise=np.full((30, 6), exploration_std, dtype=np.float32),
            steering_step=self.steering_step if steer else None,
        )

    def set_steering(self, payload: dict) -> int | None:
        self.steering_payloads.append(payload)
        meta = json.loads(str(payload["meta"]))
        self.steering_step = meta["learner_step"]
        return self.steering_step

    def reset(self) -> None:
        self.resets += 1


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _channel(port: int, token: str) -> grpc.Channel:
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    return grpc.intercept_channel(
        channel, _auth_interceptor([("authorization", f"Bearer {token}")])
    )


def _observation() -> PolicyObservation:
    rng = np.random.default_rng(0)
    return PolicyObservation(
        state=np.array([3.0, 0, 0, 0, 0, 0], dtype=np.float32),
        images={
            name: rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
            for name in ("front", "top", "wrist")
        },
        prompt="push the green button",
    )


def test_pack_round_trip_needs_no_pickle() -> None:
    arrays = unpack(pack({"x": np.arange(3), "prompt": "hi", "ms": 1.5}))
    assert arrays["x"].tolist() == [0, 1, 2]
    assert str(arrays["prompt"]) == "hi"
    assert float(arrays["ms"]) == 1.5


def test_remote_policy_matches_the_served_policy() -> None:
    policy = _FakePolicy()
    port = _free_port()
    with serve(
        policy, bind=f"127.0.0.1:{port}", image_size=(224, 224), auth_token="secret"
    ):
        remote = RemoteOpenPIRLTPolicy(_channel(port, "secret"), image_codec="raw")
        observation = _observation()
        result = remote.infer(observation)
        remote.reset()

    assert remote.rl_token_dim == 2048
    assert remote.use_quantile_norm and remote.uses_delta_actions
    np.testing.assert_array_equal(remote.norm_stats["state"].q99, np.full(6, 2.0))
    assert remote.norm_stats["actions"].q01 is None
    assert remote.norm_stats["actions"].mean.shape == (30, 6)
    np.testing.assert_array_equal(result.action_chunk, np.full((30, 6), 3.0))
    np.testing.assert_array_equal(result.rl_token, np.arange(2048))
    served = policy.observations[0]
    assert served.prompt == observation.prompt
    np.testing.assert_array_equal(served.state, observation.state)
    for name, image in observation.images.items():
        np.testing.assert_array_equal(served.images[name], image)
    assert policy.resets == 1


def test_remote_policy_rejects_a_wrong_token() -> None:
    port = _free_port()
    with serve(
        _FakePolicy(),
        bind=f"127.0.0.1:{port}",
        image_size=(224, 224),
        auth_token="secret",
    ):
        with pytest.raises(grpc.RpcError) as error:
            RemoteOpenPIRLTPolicy(_channel(port, "wrong"))
    assert error.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_remote_policy_forwards_steering_and_noise() -> None:
    policy = _FakePolicy()
    port = _free_port()
    with serve(
        policy, bind=f"127.0.0.1:{port}", image_size=(224, 224), auth_token="secret"
    ):
        remote = RemoteOpenPIRLTPolicy(_channel(port, "secret"), image_codec="raw")
        plain = remote.infer(_observation())
        step = remote.set_steering(
            {"meta": json.dumps({"learner_step": 7}), "param/w": np.ones((2, 3))}
        )
        steered = remote.infer(_observation(), exploration_std=0.5)
        unsteered = remote.infer(_observation(), exploration_std=1.0, steer=False)

    assert plain.noise is None and plain.steering_step is None
    assert step == 7 and remote.steering_step == 7
    np.testing.assert_array_equal(policy.steering_payloads[0]["param/w"], np.ones((2, 3)))
    assert policy.steered == [(0.5, True), (1.0, False)]
    np.testing.assert_array_equal(steered.action_chunk, np.full((30, 6), -3.0))
    np.testing.assert_array_equal(steered.noise, np.full((30, 6), 0.5))
    assert steered.steering_step == 7
    assert unsteered.steering_step is None
    np.testing.assert_array_equal(unsteered.noise, np.full((30, 6), 1.0))


def test_actor_channel_carries_a_steering_payload_over_4mb() -> None:
    policy = _FakePolicy()
    port = _free_port()
    weights = np.ones((2048, 1024), dtype=np.float32)  # 8 MiB
    with serve(
        policy, bind=f"127.0.0.1:{port}", image_size=(224, 224), auth_token="secret"
    ):
        lerobot_default = _grpc_channel("127.0.0.1", port, use_tls=False, token="secret")
        with pytest.raises(grpc.RpcError) as error:
            RemoteOpenPIRLTPolicy(lerobot_default).set_steering(
                {"meta": json.dumps({"learner_step": 1}), "param/w": weights}
            )
        channel = _grpc_channel(
            "127.0.0.1",
            port,
            use_tls=False,
            token="secret",
            max_message_bytes=MAX_MESSAGE_BYTES,
        )
        step = RemoteOpenPIRLTPolicy(channel).set_steering(
            {"meta": json.dumps({"learner_step": 2}), "param/w": weights}
        )

    assert error.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
    assert step == 2
    assert policy.steering_payloads[0]["param/w"].shape == (2048, 1024)


def test_jpeg_requests_are_much_smaller_than_png() -> None:
    pytest.importorskip("cv2")
    from armnet_rlt.remote_policy import decode_image, encode_image

    rng = np.random.default_rng(0)
    gradient = np.linspace(0, 255, 224, dtype=np.float32)
    image = np.stack(
        [np.add.outer(gradient, gradient[::-1]) / 2 + rng.normal(0, 4, (224, 224))] * 3,
        axis=-1,
    ).clip(0, 255).astype(np.uint8)
    png = encode_image(image, "png")
    jpeg = encode_image(image, "jpeg")
    assert jpeg.size < png.size / 3
    decoded = decode_image(jpeg, "jpeg")
    assert decoded.shape == image.shape
    assert np.abs(decoded.astype(int) - image.astype(int)).mean() < 3.0


def test_remote_policy_defaults_to_jpeg_images() -> None:
    import argparse

    from armnet_rlt.resources import add_remote_policy_arguments

    parser = argparse.ArgumentParser()
    add_remote_policy_arguments(parser)
    assert parser.parse_args([]).remote_policy_image_codec == "jpeg"


def test_policy_server_can_terminate_tls_itself(tmp_path) -> None:
    from armnet_rlt.local_app import TLS_SERVER_NAME, ensure_tls_certificate

    certificate, key = ensure_tls_certificate(tmp_path)
    port = _free_port()
    with serve(
        _FakePolicy(),
        bind=f"127.0.0.1:{port}",
        image_size=(224, 224),
        auth_token="secret",
        tls_cert_path=str(certificate),
        tls_key_path=str(key),
    ):
        channel = _grpc_channel(
            "127.0.0.1",
            port,
            use_tls=True,
            token="secret",
            server_name=TLS_SERVER_NAME,
            root_cert_path=str(certificate),
        )
        remote = RemoteOpenPIRLTPolicy(channel, image_codec="raw")
        result = remote.infer(_observation())
        assert result.action_chunk.shape == (30, 6)
        with pytest.raises(grpc.RpcError):
            RemoteOpenPIRLTPolicy(_channel(port, "secret"), image_codec="raw")


def test_server_fed_by_the_learner_tells_actors_to_skip_the_upload() -> None:
    policy = _FakePolicy()
    port = _free_port()
    with serve(
        policy,
        bind=f"127.0.0.1:{port}",
        image_size=(224, 224),
        auth_token="secret",
        steering_from_learner=True,
    ) as server:
        remote = RemoteOpenPIRLTPolicy(_channel(port, "secret"), image_codec="raw")
        assert remote.steering_from_learner
        step = server.apply_steering(
            {"meta": json.dumps({"learner_step": 11}), "param/w": np.ones((2, 3))}
        )
        steered = remote.infer(_observation(), exploration_std=0.5)
    assert step == 11 and steered.steering_step == 11

    with serve(
        policy, bind=f"127.0.0.1:{port}", image_size=(224, 224), auth_token="secret"
    ):
        assert not RemoteOpenPIRLTPolicy(_channel(port, "secret")).steering_from_learner


def test_actor_skips_the_steering_upload_for_a_fed_server(capsys) -> None:
    import torch

    from armnet_rlt.so101_actor import _sync_steering

    class Fed:
        steering_from_learner = True
        calls = 0

        def set_steering(self, payload):
            self.calls += 1

    class Actor:
        noise_bound = torch.tensor(1.5)
        noise_steps = 1

        def state_dict(self):
            return {"w": torch.ones(2)}

    policy = Fed()
    _sync_steering(policy, Actor(), network=None, learner_step=3)
    _sync_steering(policy, Actor(), network=None, learner_step=4)
    assert policy.calls == 0
    assert capsys.readouterr().out.count("steering from the learner") == 1
