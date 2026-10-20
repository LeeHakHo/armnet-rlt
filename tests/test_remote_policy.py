from __future__ import annotations

import socket
from dataclasses import dataclass

import grpc
import numpy as np
import pytest

from armnet_rlt.openpi_rlt import PolicyObservation, PolicyOutput
from armnet_rlt.remote_policy import RemoteOpenPIRLTPolicy, pack, serve, unpack
from armnet_rlt.so101_actor import _auth_interceptor


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
        self.resets = 0

    def infer(self, observation: PolicyObservation) -> PolicyOutput:
        self.observations.append(observation)
        return PolicyOutput(
            action_chunk=np.full((30, 6), observation.state[0], dtype=np.float32),
            rl_token=np.arange(2048, dtype=np.float32),
        )

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
