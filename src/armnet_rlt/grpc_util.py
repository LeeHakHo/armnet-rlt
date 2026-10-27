"""gRPC channel helpers shared by the cell actor and the policy server."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


LEARNER_MAX_MESSAGE_BYTES = 4 * 1024 * 1024
GRPC_RETRY_SERVICE_CONFIG = json.dumps(
    {
        "methodConfig": [
            {
                "name": [{}],
                "retryPolicy": {
                    "maxAttempts": 5,
                    "initialBackoff": "0.1s",
                    "maxBackoff": "2s",
                    "backoffMultiplier": 2,
                    "retryableStatusCodes": ["UNAVAILABLE", "DEADLINE_EXCEEDED"],
                },
            }
        ]
    }
)


def auth_interceptor(metadata: list[tuple[str, str]]) -> Any:
    """Client interceptor that adds ``metadata`` to every call."""
    import grpc

    class Details(grpc.ClientCallDetails):
        def __init__(self, original: Any) -> None:
            self.method = original.method
            self.timeout = original.timeout
            self.metadata = [*(original.metadata or []), *metadata]
            self.credentials = original.credentials
            self.wait_for_ready = getattr(original, "wait_for_ready", None)
            self.compression = getattr(original, "compression", None)

    class Interceptor(
        grpc.UnaryUnaryClientInterceptor,
        grpc.UnaryStreamClientInterceptor,
        grpc.StreamUnaryClientInterceptor,
        grpc.StreamStreamClientInterceptor,
    ):
        def intercept_unary_unary(self, continuation, details, request):
            return continuation(Details(details), request)

        def intercept_unary_stream(self, continuation, details, request):
            return continuation(Details(details), request)

        def intercept_stream_unary(self, continuation, details, iterator):
            return continuation(Details(details), iterator)

        def intercept_stream_stream(self, continuation, details, iterator):
            return continuation(Details(details), iterator)

    return Interceptor()


def grpc_channel_options(max_message_bytes: int) -> list[tuple[str, Any]]:
    """lerobot.transport.utils.grpc_channel_options with a configurable size cap."""
    return [
        ("grpc.max_receive_message_length", max_message_bytes),
        ("grpc.max_send_message_length", max_message_bytes),
        ("grpc.enable_retries", 1),
        ("grpc.service_config", GRPC_RETRY_SERVICE_CONFIG),
    ]


def grpc_channel(
    host: str,
    port: int,
    *,
    use_tls: bool,
    token: str = "",
    server_name: str = "",
    root_cert_path: str = "",
    max_message_bytes: int = LEARNER_MAX_MESSAGE_BYTES,
) -> Any:
    """A channel to ``host:port``; ``server_name`` overrides the TLS name
    (a learner on localhost serving its public certificate)."""
    import grpc

    options = grpc_channel_options(max_message_bytes)
    if server_name:
        options.append(("grpc.ssl_target_name_override", server_name))
    target = f"{host}:{port}"
    if use_tls:
        root = Path(root_cert_path).read_bytes() if root_cert_path else None
        channel = grpc.secure_channel(
            target, grpc.ssl_channel_credentials(root_certificates=root), options
        )
    else:
        channel = grpc.insecure_channel(target, options)
    if token:
        channel = grpc.intercept_channel(
            channel, auth_interceptor([("authorization", f"Bearer {token}")])
        )
    return channel
