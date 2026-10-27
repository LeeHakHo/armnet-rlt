"""Keep a policy server's DSRL steering at the learner's newest actor.

The cell actor otherwise uploads the actor (17 MB for the 1024x3 network) to
the policy server through the Funnel at every parameter update, which takes
10 s from this machine and exceeds the 30 s SetSteering deadline from the
cell. The learner and the policy server share a machine, so the server
subscribes to the learner's parameter stream itself and the actor sends
nothing.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Mapping

import torch

from armnet_rlt.config import RLTNetworkConfig
from armnet_rlt.dsrl import steering_payload
from armnet_rlt.grpc_util import grpc_channel


log = logging.getLogger(__name__)


class LearnerSteeringFeed:
    """Subscribe to ``StreamParameters`` on a thread and hand each actor state
    to ``apply`` as a steering payload. A partial update merges into the last
    full state."""

    def __init__(
        self,
        apply: Callable[[dict[str, Any]], Any],
        *,
        host: str,
        port: int,
        use_tls: bool,
        token: str,
        server_name: str = "",
        network: RLTNetworkConfig,
        noise_steps: int = 1,
        max_message_bytes: int,
        retry_s: float = 5.0,
    ) -> None:
        self._apply = apply
        self._host = host
        self._port = port
        self._use_tls = use_tls
        self._token = token
        self._server_name = server_name
        self._network = network
        self._noise_steps = int(noise_steps)
        self._max_message_bytes = max_message_bytes
        self._retry_s = retry_s
        self._state: dict[str, torch.Tensor] = {}
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self.learner_step: int | None = None
        self.updates = 0

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="steering-feed", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        self._shutdown.set()

    def apply_state(self, state: Mapping[str, Any]) -> int | None:
        """Install one decoded parameter message: ``policy`` (the actor's
        state dict), ``learner_step`` and an optional ``partial`` flag."""
        weights = dict(state.get("policy", state))
        if bool(_scalar(state.get("partial", False))):
            if not self._state:
                log.warning("partial actor update before any full one; skipped")
                return self.learner_step
            self._state.update(weights)
        else:
            self._state = weights
        if "noise_bound" not in self._state:
            raise ValueError("actor state has no noise_bound; not a DSRL actor")
        raw_step = state.get("learner_step")
        step = None if raw_step is None else int(_scalar(raw_step))
        payload = steering_payload(
            self._state,
            network=self._network,
            noise_bound=float(_scalar(self._state["noise_bound"])),
            noise_steps=self._noise_steps,
            learner_step=step,
        )
        self._apply(payload)
        self.learner_step = step
        self.updates += 1
        return step

    def _run(self) -> None:
        from lerobot.transport import services_pb2, services_pb2_grpc
        from lerobot.transport.utils import bytes_to_state_dict, receive_bytes_in_chunks
        from queue import Queue

        while not self._shutdown.is_set():
            messages: Queue = Queue()
            channel = grpc_channel(
                self._host,
                self._port,
                use_tls=self._use_tls,
                token=self._token,
                server_name=self._server_name,
                max_message_bytes=self._max_message_bytes,
            )
            stub = services_pb2_grpc.LearnerServiceStub(channel)
            receiver = threading.Thread(
                target=self._receive,
                args=(stub, services_pb2, receive_bytes_in_chunks, messages),
                daemon=True,
            )
            receiver.start()
            while receiver.is_alive() and not self._shutdown.is_set():
                try:
                    payload = messages.get(timeout=1.0)
                except Exception:
                    continue
                try:
                    step = self.apply_state(bytes_to_state_dict(payload))
                except Exception:
                    log.exception("steering feed could not apply an update")
                    continue
                print(f"[RLT policy] steering from learner step {step}", flush=True)
            channel.close()
            if not self._shutdown.is_set():
                time.sleep(self._retry_s)

    def _receive(self, stub: Any, services_pb2: Any, receive: Any, messages: Any) -> None:
        try:
            receive(
                stub.StreamParameters(services_pb2.Empty()),
                messages,
                self._shutdown,
                log_prefix="[RLT policy] steering feed",
            )
        except Exception as exc:
            if not self._shutdown.is_set():
                print(
                    f"[RLT policy] steering feed lost the learner at "
                    f"{self._host}:{self._port} ({type(exc).__name__}); retrying in "
                    f"{self._retry_s:g}s",
                    flush=True,
                )


def _scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value
