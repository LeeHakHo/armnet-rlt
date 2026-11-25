"""One gRPC endpoint for a run: the cell's learner and policy-server calls arrive on one
port and go to the learner or the policy server on this machine.

Calls to ``armnet_rlt.RemotePolicy`` go to the policy server, all others to the learner,
both plaintext. Requests and replies pass through as bytes, every call as a stream (a
unary call is a stream of one message), with the caller's metadata and deadline; the
upstream status and trailing metadata come back to the caller.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any


POLICY_SERVICE = "armnet_rlt.RemotePolicy"
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
# Upstreams are local; reconnect quickly after a server restarts.
UPSTREAM_OPTIONS = (
    ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
    ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
    ("grpc.enable_retries", 0),
    ("grpc.initial_reconnect_backoff_ms", 100),
    ("grpc.min_reconnect_backoff_ms", 100),
    ("grpc.max_reconnect_backoff_ms", 1000),
)
SERVER_OPTIONS = (
    ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
    ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
    # A second front on the port fails to bind.
    ("grpc.so_reuseport", 0),
    # The actors' keepalive pings, as the learner accepts them.
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.min_ping_interval_without_data_ms", 20_000),
    ("grpc.http2.max_ping_strikes", 0),
)


def _forwarded(metadata: Any) -> list[tuple[str, Any]]:
    return [
        (key, value)
        for key, value in (metadata or ())
        if not key.startswith((":", "grpc-")) and key not in {"user-agent", "te", "content-type"}
    ]


class _Router:
    """Generic handler that relays each call to its upstream channel."""

    def __init__(self, grpc: Any, learner: Any, policy: Any, token: str) -> None:
        self._grpc = grpc
        self._learner = learner
        self._policy = policy
        self._expected = f"Bearer {token}" if token else ""

    def upstream(self, method: str) -> Any:
        return self._policy if method.startswith(f"/{POLICY_SERVICE}/") else self._learner

    def service(self, details: Any) -> Any:
        grpc = self._grpc
        method = details.method
        channel = self.upstream(method)
        expected = self._expected

        def relay(requests: Iterator[bytes], context: Any) -> Iterator[bytes]:
            metadata = context.invocation_metadata()
            if expected and dict(metadata or ()).get("authorization") != expected:
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid or missing auth token")
            call = channel.stream_stream(method)(
                requests, metadata=_forwarded(metadata), timeout=context.time_remaining()
            )
            context.add_callback(call.cancel)
            try:
                yield from call
                context.set_trailing_metadata(_forwarded(call.trailing_metadata()))
            except grpc.RpcError as exc:
                context.set_trailing_metadata(_forwarded(exc.trailing_metadata()))
                context.abort(exc.code(), exc.details() or "")

        return grpc.stream_stream_rpc_method_handler(relay)


@contextmanager
def serve_front(
    *,
    bind: str,
    learner: str,
    policy: str,
    auth_token: str = "",
    tls_cert_path: str = "",
    tls_key_path: str = "",
    max_workers: int = 32,
) -> Iterator[Any]:
    """Run the front on ``bind`` until the context exits; ``learner`` and ``policy``
    are the upstream ``host:port``. With ``tls_cert_path`` and ``tls_key_path`` the
    front terminates TLS."""
    import grpc

    if bool(tls_cert_path) != bool(tls_key_path):
        raise ValueError("tls_cert_path and tls_key_path must be set together")
    learner_channel = grpc.insecure_channel(learner, UPSTREAM_OPTIONS)
    policy_channel = grpc.insecure_channel(policy, UPSTREAM_OPTIONS)
    router = _Router(grpc, learner_channel, policy_channel, auth_token)

    class Handler(grpc.GenericRpcHandler):
        def service(self, details: Any) -> Any:
            return router.service(details)

    server = grpc.server(ThreadPoolExecutor(max_workers=max_workers), options=SERVER_OPTIONS)
    server.add_generic_rpc_handlers((Handler(),))
    try:
        if tls_cert_path:
            credentials = grpc.ssl_server_credentials(
                ((Path(tls_key_path).read_bytes(), Path(tls_cert_path).read_bytes()),)
            )
            bound = server.add_secure_port(bind, credentials)
        else:
            bound = server.add_insecure_port(bind)
    except RuntimeError:
        bound = 0
    if bound == 0:
        learner_channel.close()
        policy_channel.close()
        raise RuntimeError(f"could not bind the front to {bind}")
    server.start()
    server.bound_port = bound
    try:
        yield server
    finally:
        server.stop(grace=2)
        learner_channel.close()
        policy_channel.close()
