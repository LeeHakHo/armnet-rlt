"""Modal backend: the process supervisor that runs in a Modal container and the
commands scripts/lib.sh runs on this machine with BACKEND=modal.

    python -m armnet_rlt.modal_backend start --image rl --specs F --call-file C
    python -m armnet_rlt.modal_backend alive|stop C
    python -m armnet_rlt.modal_backend run --image rl --log L -- <argv>
    python -m armnet_rlt.modal_backend exists|ls|get|put <local path>
    python -m armnet_rlt.modal_backend put-dir <local dir> <local target>

Local paths under RLT_DATA_ROOT map to the root of the Modal volume and paths under
<repo>/checkpoints to its checkpoints/ directory; the container mounts the volume at
/data, its RLT_DATA_ROOT.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from armnet_rlt.resources import MODAL_APP, MODAL_RENDEZVOUS, MODAL_VOLUME, local_data_root


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTAINER_ROOT = "/root/armnet-rlt"
CONTAINER_DATA = "/data"
# --public-endpoint value the container replaces with its Modal tunnel address.
ENDPOINT_PLACEHOLDER = "modal"
FORWARDED_ENV_PREFIXES = ("RLT_POLICY_INSTANCE", "ROBOMETER_")
FUNCTIONS = {"rl": ("serve", "run_command"), "lerobot": ("serve_lerobot", "run_command_lerobot")}


def _prefixes() -> list[tuple[str, str]]:
    """(local prefix, volume prefix) pairs, longest local prefix first."""
    pairs = []
    for local, volume in ((PROJECT_ROOT / "checkpoints", "checkpoints"), (local_data_root(), "")):
        for form in {os.path.abspath(local), os.path.realpath(local)}:
            pairs.append((form, volume))
    return sorted(pairs, key=lambda pair: -len(pair[0]))


def volume_path(local: str | os.PathLike[str]) -> str | None:
    """Path in the volume of a local data or checkpoint path, or of a container path
    under /data; None outside them."""
    path = os.path.abspath(local)
    for prefix, volume in [*_prefixes(), (CONTAINER_DATA, "")]:
        if path == prefix or path.startswith(prefix + os.sep):
            rest = os.path.relpath(path, prefix)
            rest = "" if rest == "." else rest
            return str(PurePosixPath(volume, rest)) if volume or rest else "."
    return None


def container_path(local: str | os.PathLike[str]) -> str | None:
    """Path in the container of a local data or checkpoint path."""
    path = volume_path(local)
    if path is None:
        return None
    return CONTAINER_DATA if path == "." else f"{CONTAINER_DATA}/{path}"


def remote_argv(argv: list[str]) -> list[str]:
    """argv with local data and checkpoint paths, also in --flag=PATH, as container paths."""
    out = []
    for token in argv:
        flag, sep, value = token.partition("=")
        if token.startswith("/"):
            token = container_path(token) or token
        elif sep and flag.startswith("-") and value.startswith("/"):
            token = f"{flag}={container_path(value) or value}"
        out.append(token)
    return out


def argv_option(argv: list[str], name: str) -> str | None:
    for i, token in enumerate(argv):
        if token == name and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return None


def _is_placeholder(value: str | None) -> bool:
    if value == ENDPOINT_PLACEHOLDER:
        return True
    prefix, sep, port = (value or "").partition(":")
    return prefix == ENDPOINT_PLACEHOLDER and bool(sep) and port.isdigit()


def needs_tunnel(argv: list[str]) -> bool:
    return _is_placeholder(argv_option(argv, "--public-endpoint"))


def tunnel_port(argv: list[str]) -> int:
    """Container port of the tunnel a placeholder names: ``modal:<port>``, or the
    process's --port for ``modal``."""
    value = argv_option(argv, "--public-endpoint") or ""
    _, _, port = value.partition(":")
    return int(port or argv_option(argv, "--port") or 0)


def container_argv(argv: list[str], endpoint: str) -> list[str]:
    """argv with the --public-endpoint placeholder replaced by the tunnel address."""
    out = list(argv)
    for i, token in enumerate(out):
        if token == "--public-endpoint" and i + 1 < len(out) and _is_placeholder(out[i + 1]):
            out[i + 1] = endpoint
        elif token.startswith("--public-endpoint=") and _is_placeholder(token.split("=", 1)[1]):
            out[i] = f"--public-endpoint={endpoint}"
    return out


def forwarded_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    environ = os.environ if environ is None else environ
    return {k: v for k, v in environ.items() if k.startswith(FORWARDED_ENV_PREFIXES)}


def process_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """A local process spec ({name, argv, log}) as the container runs it."""
    log = container_path(spec["log"])
    if log is None:
        raise SystemExit(f"log {spec['log']} is outside {local_data_root()}")
    return {
        "name": spec["name"],
        "argv": remote_argv(list(spec["argv"])),
        "log": log,
        "env": {**forwarded_env(), **spec.get("env", {})},
    }


# Container side.


def _terminate(processes: list[subprocess.Popen], grace_s: float = 30.0) -> None:
    for process in processes:
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    for process in processes:
        try:
            process.wait(max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def supervise(
    processes: list[dict[str, Any]],
    *,
    forward: Callable[[int], contextlib.AbstractContextManager[tuple[str, int]]],
    publish: Callable[[str, Any], None],
    unpublish: Callable[[str], None],
    commit: Callable[[], None],
    call_key: str,
    rendezvous_dir: Path,
    cwd: str = CONTAINER_ROOT,
    poll_s: float = 2.0,
    commit_every_s: float = 15.0,
) -> dict[str, int]:
    """Runs the processes until one exits, then stops the rest.

    A process whose --public-endpoint is the placeholder ``modal`` gets a Modal TLS tunnel
    to its --port; ``modal:<port>`` gives it the tunnel to <port>, one per port. The rendezvous entries the processes write to ``rendezvous_dir`` are copied to
    the Modal Dict under their key; ``call_key`` lists them and turns ready once every
    tunnelled process has registered. Returns each process's exit code.
    """
    rendezvous_dir.mkdir(parents=True, exist_ok=True)
    for stale in rendezvous_dir.glob("*.json"):
        stale.unlink()
    started: list[tuple[str, subprocess.Popen]] = []
    published: dict[str, str] = {}
    expected = sum(needs_tunnel(spec["argv"]) for spec in processes)
    tunnels: dict[int, tuple[str, int]] = {}
    status: dict[str, Any] = {}
    with contextlib.ExitStack() as stack:
        try:
            for spec in processes:
                argv = list(spec["argv"])
                if needs_tunnel(argv):
                    port = tunnel_port(argv)
                    if not port:
                        raise ValueError(f"{spec['name']}: --public-endpoint modal needs --port")
                    if port not in tunnels:
                        tunnels[port] = stack.enter_context(forward(port))
                    host, public_port = tunnels[port]
                    argv = container_argv(argv, f"{host}:{public_port}")
                    print(f"[modal] {spec['name']}: {host}:{public_port} -> :{port}", flush=True)
                log_path = Path(spec["log"])
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = stack.enter_context(log_path.open("ab"))
                env = {**os.environ, **spec.get("env", {}), "RLT_RENDEZVOUS_DIR": str(rendezvous_dir)}
                print(f"[modal] {spec['name']}: {' '.join(argv)}", flush=True)
                process = subprocess.Popen(
                    argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                started.append((spec["name"], process))
            last_commit = time.monotonic()
            while all(process.poll() is None for _, process in started):
                current = {}
                for path in rendezvous_dir.glob("*.json"):
                    try:
                        entry = json.loads(path.read_text())
                    except (OSError, json.JSONDecodeError):
                        continue
                    key = entry.get("key")
                    if key:
                        current[key] = json.dumps(entry, sort_keys=True)
                        if published.get(key) != current[key]:
                            publish(key, entry)
                for key in set(published) - set(current):
                    unpublish(key)
                published = current
                new_status = {"ready": len(published) >= expected, "keys": sorted(published)}
                if new_status != status:
                    status = new_status
                    publish(call_key, status)
                if time.monotonic() - last_commit >= commit_every_s:
                    commit()
                    last_commit = time.monotonic()
                time.sleep(poll_s)
            for name, process in started:
                if process.poll() is not None:
                    print(f"[modal] {name} exited with {process.returncode}", flush=True)
        finally:
            _terminate([process for _, process in started])
            for key in published:
                unpublish(key)
            unpublish(call_key)
            commit()
    return {name: process.returncode for name, process in started}


def run_command(
    argv: list[str], log: str, env: dict[str, str], *, commit: Callable[[], None], cwd: str = CONTAINER_ROOT
) -> dict[str, Any]:
    """Runs one command to completion, appending its output to ``log``."""
    log_path = Path(log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("ab") as out:
            code = subprocess.run(
                argv, cwd=cwd, env={**os.environ, **env}, stdout=out, stderr=subprocess.STDOUT
            ).returncode
    finally:
        commit()
    tail = log_path.read_bytes()[-4000:].decode(errors="replace")
    return {"returncode": code, "tail": tail}


# Local side.


def _modal():
    try:
        import modal
    except ModuleNotFoundError as exc:
        raise SystemExit("BACKEND=modal needs the modal extra: uv sync --extra modal") from exc
    return modal


def _volume():
    return _modal().Volume.from_name(MODAL_VOLUME)


def _rendezvous():
    return _modal().Dict.from_name(MODAL_RENDEZVOUS)


def _function(image: str, role: int):
    return _modal().Function.from_name(MODAL_APP, FUNCTIONS[image][role])


def _require_volume_path(local: str) -> str:
    path = volume_path(local)
    if path is None:
        raise SystemExit(f"{local} is outside {local_data_root()} and {PROJECT_ROOT / 'checkpoints'}")
    return path


def call_finished(call: Any) -> tuple[bool, str]:
    """Whether a spawned call has ended, and how; a lookup error counts as running."""
    modal = _modal()
    try:
        result = call.get(timeout=0)
    except (TimeoutError, modal.exception.TimeoutError):
        return False, ""
    except (OSError, modal.exception.ConnectionError) as exc:
        print(f"[modal] could not reach the call: {exc}", file=sys.stderr)
        return False, ""
    except Exception as exc:  # noqa: BLE001 - the call raised or was cancelled
        return True, f"{type(exc).__name__}: {exc}"
    return True, f"exit codes {result}"


def wait_ready(call: Any, rendezvous: Any, *, poll_s: float = 5.0, timeout_s: float = 3600.0) -> dict:
    """Waits until the call's processes have registered their endpoints."""
    key = f"call:{call.object_id}"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status = rendezvous.get(key) or {}
        if status.get("ready"):
            return status
        finished, how = call_finished(call)
        if finished:
            raise SystemExit(f"Modal call {call.object_id} ended before it was ready ({how})")
        time.sleep(poll_s)
    raise SystemExit(f"Modal call {call.object_id} not ready after {timeout_s:.0f} s")


def _read_specs(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def cmd_start(args: argparse.Namespace) -> None:
    processes = [process_spec(spec) for spec in _read_specs(args.specs)]
    call = _function(args.image, 0).spawn(processes)
    Path(args.call_file).write_text(call.object_id + "\n")
    print(f"[modal] call {call.object_id}: {', '.join(p['name'] for p in processes)}", flush=True)
    rendezvous = _rendezvous()
    status = wait_ready(call, rendezvous, timeout_s=args.timeout_s)
    for key in status["keys"]:
        entry = rendezvous.get(key) or {}
        print(f"[modal] {key}: {entry.get('host')}:{entry.get('port')}", flush=True)


def _call(call_file: str):
    return _modal().FunctionCall.from_id(Path(call_file).read_text().strip())


def cmd_alive(args: argparse.Namespace) -> None:
    finished, how = call_finished(_call(args.call_file))
    if finished:
        print(f"[modal] call ended: {how}", file=sys.stderr)
    raise SystemExit(1 if finished else 0)


def cmd_stop(args: argparse.Namespace) -> None:
    if not Path(args.call_file).is_file():
        return
    call = _call(args.call_file)
    rendezvous = _rendezvous()
    key = f"call:{call.object_id}"
    status = rendezvous.get(key) or {}
    with contextlib.suppress(Exception):
        call.cancel()
    for name in [*status.get("keys", []), key]:
        with contextlib.suppress(Exception):
            rendezvous.pop(name)
    Path(args.call_file).unlink(missing_ok=True)


def cmd_run(args: argparse.Namespace) -> None:
    spec = process_spec({"name": "run", "argv": args.argv, "log": args.log})
    result = _function(args.image, 1).remote(spec["argv"], spec["log"], spec["env"])
    sys.stdout.write(result["tail"])
    raise SystemExit(result["returncode"])


def _entries(path: str) -> list[Any]:
    modal = _modal()
    try:
        return _volume().listdir(path)
    except (modal.exception.NotFoundError, FileNotFoundError):
        return []


def cmd_exists(args: argparse.Namespace) -> None:
    if volume_path(args.path) is None:
        raise SystemExit(1)
    path = PurePosixPath(_require_volume_path(args.path))
    names = {PurePosixPath(e.path).name for e in _entries(str(path.parent))}
    raise SystemExit(0 if path.name in names else 1)


def cmd_ls(args: argparse.Namespace) -> None:
    for entry in sorted(_entries(_require_volume_path(args.path)), key=lambda e: e.path):
        print(PurePosixPath(entry.path).name)


def cmd_get(args: argparse.Namespace) -> None:
    path = _require_volume_path(args.path)
    target = Path(args.path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    modal = _modal()
    try:
        with partial.open("wb") as out:
            _volume().read_file_into_fileobj(path, out)
    except (modal.exception.NotFoundError, FileNotFoundError):
        partial.unlink(missing_ok=True)
        raise SystemExit(1)
    partial.replace(target)


def cmd_put(args: argparse.Namespace) -> None:
    with _volume().batch_upload(force=True) as batch:
        batch.put_file(args.path, _require_volume_path(args.path))


def cmd_put_dir(args: argparse.Namespace) -> None:
    with _volume().batch_upload(force=True) as batch:
        batch.put_directory(args.path, _require_volume_path(args.target))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start", help="spawn processes in one GPU container; wait for their endpoints")
    start.add_argument("--image", choices=sorted(FUNCTIONS), default="rl")
    start.add_argument("--specs", required=True, help="JSON lines {name, argv, log}")
    start.add_argument("--call-file", required=True, help="file the call id is written to")
    start.add_argument("--timeout-s", type=float, default=3600.0)
    start.set_defaults(handler=cmd_start)
    for name, handler in (("alive", cmd_alive), ("stop", cmd_stop)):
        command = commands.add_parser(name)
        command.add_argument("call_file")
        command.set_defaults(handler=handler)
    run = commands.add_parser("run", help="run one command in a GPU container to completion")
    run.add_argument("--image", choices=sorted(FUNCTIONS), default="rl")
    run.add_argument("--log", required=True)
    run.add_argument("argv", nargs=argparse.REMAINDER)
    run.set_defaults(handler=cmd_run)
    for name, handler in (("exists", cmd_exists), ("ls", cmd_ls), ("get", cmd_get), ("put", cmd_put)):
        command = commands.add_parser(name)
        command.add_argument("path")
        command.set_defaults(handler=handler)
    put_dir = commands.add_parser("put-dir")
    put_dir.add_argument("path")
    put_dir.add_argument("target")
    put_dir.set_defaults(handler=cmd_put_dir)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "run" and args.argv[:1] == ["--"]:
        args.argv = args.argv[1:]
    args.handler(args)



if __name__ == "__main__":
    main()
