"""Stream one Armnet job's logs without terminal spinner rendering."""

from __future__ import annotations

import argparse
import json
import sys
import time
from urllib.parse import urlparse, urlunparse

import websocket
from armnet_core import API_KEY_HEADER
from armnet_client._config import api_key, orchestrator_url


def _websocket_url(base_url: str, job_id: str) -> str:
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunparse(
        (
            scheme,
            parsed.netloc,
            f"/jobs/{job_id}/logs",
            "",
            "",
            "",
        )
    )


def stream(
    job_id: str,
    *,
    heartbeat_seconds: float = 30.0,
    color: bool = True,
    stdout_only: bool = False,
) -> None:
    key = api_key()
    if not key:
        raise RuntimeError("ARMNET_API_KEY is unavailable")
    url = _websocket_url(orchestrator_url(), job_id)
    while True:
        socket = None
        try:
            socket = websocket.create_connection(
                url,
                header=[f"{API_KEY_HEADER}: {key}"],
                timeout=8,
            )
            socket.settimeout(5)
            print(f"[listener] connected job={job_id}", flush=True)
            last_heartbeat = 0.0
            while True:
                try:
                    raw = socket.recv()
                except websocket.WebSocketTimeoutException:
                    now = time.monotonic()
                    if now - last_heartbeat >= heartbeat_seconds:
                        print("[listener] alive; waiting for logs", flush=True)
                        last_heartbeat = now
                    continue
                payload = json.loads(raw)
                kind = payload.get("type")
                if kind == "terminal":
                    print(
                        "[listener] terminal "
                        f"status={payload.get('status')}",
                        flush=True,
                    )
                    return
                if kind == "log":
                    stream_name = payload.get("stream", "stdout")
                    if stdout_only and stream_name == "stderr":
                        continue
                    line = (
                        f"[{payload.get('timestamp', '')}] "
                        f"[{stream_name}] {payload.get('line', '')}"
                    )
                    ansi = (
                        "\033[31m"
                        if stream_name == "stderr"
                        else "\033[34m"
                    )
                    reset = "\033[0m" if color else ""
                    sys.stdout.write(
                        f"{ansi if color else ''}{line}{reset}"
                    )
                    if line and not line.endswith("\n"):
                        sys.stdout.write("\n")
                    sys.stdout.flush()
        except Exception as exc:  # noqa: BLE001 - reconnect until terminal
            print(
                "[listener] reconnecting after "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(1)
        finally:
            if socket is not None:
                try:
                    socket.close()
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument(
        "--color",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--stdout-only", action="store_true")
    args = parser.parse_args()
    stream(
        args.job_id,
        heartbeat_seconds=args.heartbeat_seconds,
        color=args.color,
        stdout_only=args.stdout_only,
    )


if __name__ == "__main__":
    main()
