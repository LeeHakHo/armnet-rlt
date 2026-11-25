"""The front: one port for the cell's learner and policy-server calls."""

from __future__ import annotations

import contextlib
import json
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
import numpy as np
import pytest

from armnet_rlt import front as front_module
from armnet_rlt import remote_policy
from armnet_rlt.front import serve_front
from armnet_rlt.grpc_util import grpc_channel
from armnet_rlt.learner import _make_auth_interceptor
from armnet_rlt.openpi_rlt import PolicyObservation
from armnet_rlt.remote_policy import PolicyBackend, RemoteOpenPIRLTPolicy, serve

TOKEN = "secret"
LEARNER = "/transport.LearnerService"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def fake_learner(port: int):
    """LearnerService methods of each kind, on bytes: Ready (unary), StreamParameters
    (server stream), SendTransitions (client stream), and a bidirectional Echo."""
    received: list[bytes] = []

    def ready(request, context):
        context.set_trailing_metadata((("x-learner", "up"),))
        return b"ready:" + request

    def stream_parameters(request, context):
        for i in range(int(request or b"3")):
            yield b"params-%d" % i
            time.sleep(0.01)

    def send_transitions(requests, context):
        for request in requests:
            received.append(request)
        return b"got-%d" % len(received)

    def echo(requests, context):
        for request in requests:
            yield request[::-1]

    def fail(request, context):
        context.abort(grpc.StatusCode.FAILED_PRECONDITION, "not now")

    handlers = {
        "Ready": grpc.unary_unary_rpc_method_handler(ready),
        "StreamParameters": grpc.unary_stream_rpc_method_handler(stream_parameters),
        "SendTransitions": grpc.stream_unary_rpc_method_handler(send_transitions),
        "Echo": grpc.stream_stream_rpc_method_handler(echo),
        "Fail": grpc.unary_unary_rpc_method_handler(fail),
    }
    server = grpc.server(
        ThreadPoolExecutor(max_workers=8), interceptors=(_make_auth_interceptor(grpc, TOKEN),)
    )
    server.add_generic_rpc_handlers(
        (grpc.method_handlers_generic_handler("transport.LearnerService", handlers),)
    )
    assert server.add_insecure_port(f"127.0.0.1:{port}")
    server.start()
    server.received = received
    try:
        yield server
    finally:
        server.stop(grace=0)


class TaggedBackend(PolicyBackend):
    """A policy whose chunks are filled with ``tag``."""

    def __init__(self, tag: float) -> None:
        self.tag = tag
        self.resets = 0
        self.steering: list[dict] = []

    def info(self):
        meta = {"policy_type": "rl", "algorithm": f"tag{self.tag}", "uses_base": False,
                "image_size": [224, 224], "actor_ready": True}
        return {"meta": json.dumps(meta)}

    def infer(self, arrays):
        assert sorted(k for k in arrays if k.startswith("image/")) == ["image/front", "image/top", "image/wrist"]
        return {"action_chunk": np.full((10, 6), self.tag, dtype=np.float32), "actor_step": 5,
                "needs_images": False}

    def set_steering(self, payload):
        self.steering.append(payload)
        return int(payload["step"])

    def reset(self):
        self.resets += 1


def _observation(size: int = 224) -> PolicyObservation:
    rng = np.random.default_rng(0)
    return PolicyObservation(
        state=np.zeros(6, dtype=np.float32),
        images={n: rng.integers(0, 256, (size, size, 3), dtype=np.uint8) for n in ("front", "top", "wrist")},
        prompt="push the green button",
    )


@pytest.fixture
def ports():
    return {name: _free_port() for name in ("front", "learner", "policy")}


@contextlib.contextmanager
def front(ports, **kwargs):
    with serve_front(
        bind=f"127.0.0.1:{ports['front']}",
        learner=f"127.0.0.1:{ports['learner']}",
        policy=f"127.0.0.1:{ports['policy']}",
        auth_token=TOKEN,
        **kwargs,
    ) as server:
        yield server


def _act(remote: RemoteOpenPIRLTPolicy) -> float:
    result = remote.act(_observation(), explore=False)
    return float(result.chunk[0, 0])


def test_one_port_serves_the_learner_and_the_policy_methods(ports) -> None:
    with fake_learner(ports["learner"]) as learner, serve(
        TaggedBackend(1.0), bind=f"127.0.0.1:{ports['policy']}", auth_token=TOKEN
    ), front(ports):
        channel = grpc_channel("127.0.0.1", ports["front"], use_tls=False, token=TOKEN, retries=False)
        # Policy methods, as the cell's client calls them.
        remote = RemoteOpenPIRLTPolicy(channel, image_codec="jpeg")
        assert remote.algorithm == "tag1.0"
        assert _act(remote) == 1.0
        assert remote.set_steering({"step": np.int64(12)}) == 12
        remote.reset()
        # Learner methods of every kind on the same channel.
        ready = channel.unary_unary(f"{LEARNER}/Ready")
        reply, call = ready.with_call(b"x", timeout=5)
        assert reply == b"ready:x"
        assert ("x-learner", "up") in call.trailing_metadata()
        stream = channel.unary_stream(f"{LEARNER}/StreamParameters")(b"4", timeout=5)
        assert list(stream) == [b"params-0", b"params-1", b"params-2", b"params-3"]
        send = channel.stream_unary(f"{LEARNER}/SendTransitions")
        assert send(iter([b"a", b"b", b"c"]), timeout=5) == b"got-3"
        assert learner.received == [b"a", b"b", b"c"]
        echo = channel.stream_stream(f"{LEARNER}/Echo")
        assert list(echo(iter([b"abc", b"xy"]), timeout=5)) == [b"cba", b"yx"]
        # Upstream errors come back with their status.
        with pytest.raises(grpc.RpcError) as error:
            channel.unary_unary(f"{LEARNER}/Fail")(b"", timeout=5)
        assert error.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        assert error.value.details() == "not now"


def test_the_front_enforces_the_auth_token(ports) -> None:
    with fake_learner(ports["learner"]), serve(
        TaggedBackend(1.0), bind=f"127.0.0.1:{ports['policy']}", auth_token=TOKEN
    ), front(ports):
        for token in ("wrong", ""):
            channel = grpc_channel("127.0.0.1", ports["front"], use_tls=False, token=token, retries=False)
            for method in (f"{LEARNER}/Ready", "/armnet_rlt.RemotePolicy/Info"):
                with pytest.raises(grpc.RpcError) as error:
                    channel.unary_unary(method)(b"", timeout=5)
                assert error.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_the_front_terminates_tls_and_a_second_front_cannot_bind(ports, tmp_path) -> None:
    from armnet_rlt.local_app import TLS_SERVER_NAME, ensure_tls_certificate

    cert, key = ensure_tls_certificate(tmp_path)
    with fake_learner(ports["learner"]), serve(
        TaggedBackend(2.0), bind=f"127.0.0.1:{ports['policy']}", auth_token=TOKEN
    ), front(ports, tls_cert_path=str(cert), tls_key_path=str(key)):
        channel = grpc_channel("127.0.0.1", ports["front"], use_tls=True, token=TOKEN,
                               server_name=TLS_SERVER_NAME, root_cert_path=str(cert), retries=False)
        assert _act(RemoteOpenPIRLTPolicy(channel, image_codec="jpeg")) == 2.0
        assert channel.unary_unary(f"{LEARNER}/Ready")(b"t", timeout=5) == b"ready:t"
        plain = grpc_channel("127.0.0.1", ports["front"], use_tls=False, token=TOKEN, retries=False)
        with pytest.raises(grpc.RpcError):
            plain.unary_unary(f"{LEARNER}/Ready")(b"", timeout=2)
        with pytest.raises(RuntimeError, match="could not bind"):
            with front(ports):
                pass


def test_evaluation_server_replaces_the_training_server_behind_the_running_front(ports) -> None:
    """run.sh: the learner and the front stay up while the training policy server stops
    and an evaluation server takes its port."""
    with fake_learner(ports["learner"]), front(ports):
        channel = grpc_channel("127.0.0.1", ports["front"], use_tls=False, token=TOKEN, retries=False)
        stream = channel.unary_stream(f"{LEARNER}/StreamParameters")(b"200", timeout=30)
        assert next(stream) == b"params-0"
        with serve(TaggedBackend(1.0), bind=f"127.0.0.1:{ports['policy']}", auth_token=TOKEN):
            assert _act(RemoteOpenPIRLTPolicy(channel, image_codec="jpeg")) == 1.0
        info = channel.unary_unary("/armnet_rlt.RemotePolicy/Info")
        with pytest.raises(grpc.RpcError) as error:
            info(b"", timeout=5)
        assert error.value.code() == grpc.StatusCode.UNAVAILABLE
        with serve(TaggedBackend(7.0), bind=f"127.0.0.1:{ports['policy']}", auth_token=TOKEN):
            deadline = time.monotonic() + 5
            while True:
                try:
                    remote = RemoteOpenPIRLTPolicy(channel, image_codec="jpeg")
                    break
                except grpc.RpcError:
                    assert time.monotonic() < deadline, "the front did not reconnect"
                    time.sleep(0.1)
            assert remote.algorithm == "tag7.0" and _act(remote) == 7.0
        # The learner stream was not interrupted.
        rest = list(stream)
        assert rest[-1] == b"params-199" and len(rest) == 199


def test_the_front_caps_messages_like_the_policy_server() -> None:
    assert front_module.MAX_MESSAGE_BYTES == remote_policy.MAX_MESSAGE_BYTES


def test_rlt_local_front_needs_tls(monkeypatch) -> None:
    from armnet_rlt import local_app

    monkeypatch.setenv("RLT_LEARNER_AUTH_TOKEN", TOKEN)
    for argv in (["front", "--tunnel", "none", "--public-endpoint", "h:443"],
                 ["front", "--tls-cert", "c", "--tls-key", "k"]):
        args = local_app._parser().parse_args(argv)
        with pytest.raises(SystemExit, match="over TLS"):
            args.handler(args)


def test_rlt_local_front_runs_behind_edge_tls(ports, tmp_path) -> None:
    """The command scripts/lib.sh starts: its log line names the public endpoint."""
    import os
    import sys

    root = Path(__file__).resolve().parents[1]
    log = tmp_path / "front.log"
    env = {**os.environ, "RLT_LEARNER_AUTH_TOKEN": TOKEN, "PYTHONPATH": str(root / "src")}
    with log.open("w") as out:
        process = subprocess.Popen(
            [sys.executable, "-m", "armnet_rlt.local_app", "front", "--port", str(ports["front"]),
             "--learner-port", str(ports["learner"]), "--policy-port", str(ports["policy"]),
             "--tunnel", "none", "--public-endpoint", "h.example:443", "--public-endpoint-tls"],
            stdout=out, stderr=subprocess.STDOUT, env=env,
        )
    try:
        with fake_learner(ports["learner"]):
            deadline = time.monotonic() + 30
            while "[RLT] front" not in log.read_text():
                assert process.poll() is None and time.monotonic() < deadline, log.read_text()
                time.sleep(0.1)
            assert log.read_text().startswith(
                f"[RLT] front h.example:443 (TLS at the edge) on 127.0.0.1:{ports['front']} -> "
                f"learner 127.0.0.1:{ports['learner']}, policy 127.0.0.1:{ports['policy']}")
            channel = grpc_channel("127.0.0.1", ports["front"], use_tls=False, token=TOKEN, retries=False)
            assert channel.unary_unary(f"{LEARNER}/Ready")(b"p", timeout=5) == b"ready:p"
    finally:
        process.terminate()
        assert process.wait(10) == 0
