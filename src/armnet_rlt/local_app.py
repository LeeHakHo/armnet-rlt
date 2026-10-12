"""Run the RLT learner and cache builders on this machine's GPU."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import shutil
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from armnet_rlt.resources import (
    GREEN_BUTTON_CHECKPOINT,
    GREEN_BUTTON_CONFIG,
    GREEN_BUTTON_DATASET,
    GREEN_BUTTON_PROMPT,
    LEARNER_PORT,
    MULTITASK_CHECKPOINT,
    MULTITASK_CONFIG,
    MULTITASK_DATASET,
    local_data_root,
    local_rendezvous_path,
    rendezvous_key,
    task_paths,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TLS_SERVER_NAME = "armnet-rlt-learner"
NORM_STATS_FILES = ("norm_stats.json", "norm_stats_actions_per_timestep.json")
CHECKPOINT_REPOS = {
    GREEN_BUTTON_CONFIG: GREEN_BUTTON_CHECKPOINT,
    MULTITASK_CONFIG: MULTITASK_CHECKPOINT,
}


def _load_env() -> None:
    """Read ``.env`` when python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return
    load_dotenv(PROJECT_ROOT / ".env")


def _select_gpu(gpu: str) -> None:
    # Set the GPU and JAX memory environment variables.
    if gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _config(args: argparse.Namespace, **overrides: Any):
    from armnet_rlt.learner_config import learner_config

    return learner_config(
        data_root=local_data_root(),
        embodiment=args.embodiment,
        config_name=args.config_name,
        run_id=args.run_id,
        **overrides,
    )


def ensure_norm_stats(config_name: str, checkpoint_repo: str = "") -> Path:
    """Fetch the checkpoint's normalization files when they are missing."""
    _cache, assets_dir, _output = task_paths(config_name, local_data_root())
    target = Path(assets_dir)
    if all((target / name).is_file() for name in NORM_STATS_FILES):
        return target
    repo = checkpoint_repo or CHECKPOINT_REPOS.get(config_name)
    if not repo:
        raise FileNotFoundError(
            f"{target} has no normalization stats and {config_name!r} has no "
            "known checkpoint; pass --checkpoint-repo"
        )
    from huggingface_hub import hf_hub_download

    target.mkdir(parents=True, exist_ok=True)
    for name in NORM_STATS_FILES:
        shutil.copy2(hf_hub_download(repo, f"assets/{name}"), target / name)
    print(f"[RLT local] normalization stats from {repo} -> {target}", flush=True)
    return target


def ensure_tls_certificate(directory: Path) -> tuple[Path, Path]:
    """Create a self-signed certificate and key, or return the existing pair."""
    certificate, key = directory / "learner.crt", directory / "learner.key"
    if certificate.is_file() and key.is_file():
        return certificate, key
    directory.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-days", "365",
            "-subj", f"/CN={TLS_SERVER_NAME}",
            "-addext", f"subjectAltName=DNS:{TLS_SERVER_NAME}",
            "-keyout", str(key),
            "-out", str(certificate),
        ],
        check=True,
        capture_output=True,
    )
    key.chmod(0o600)
    return certificate, key


@contextlib.contextmanager
def ngrok_tunnel(
    port: int, *, tcp: bool = False, timeout_s: float = 30.0
) -> Iterator[tuple[str, int]]:
    """Expose ``port`` through ngrok; yields the public host and port."""
    scheme = "tcp://" if tcp else "https://"
    command = (
        ["ngrok", "tcp", str(port)]
        if tcp
        else ["ngrok", "http", str(port), "--upstream-protocol", "http2"]
    )
    process = subprocess.Popen(
        [*command, "--log", "stdout", "--log-format", "json"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    found: list[str] = []
    errors: list[str] = []
    ready = threading.Event()

    def watch() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                errors.append(line.strip())
                continue
            url = str(event.get("url", ""))
            if url.startswith(scheme) and not found:
                found.append(url)
                ready.set()
            if event.get("lvl") in {"eror", "crit"}:
                errors.append(str(event.get("err") or event.get("msg")))
        ready.set()

    threading.Thread(target=watch, daemon=True).start()
    try:
        ready.wait(timeout_s)
        if not found:
            detail = "; ".join(filter(None, errors[-3:])) or "no URL was reported"
            raise RuntimeError(
                f"ngrok did not open a tunnel: {detail}. Run "
                "`ngrok config add-authtoken <token>` first."
            )
        host, _, public_port = found[0].removeprefix(scheme).partition(":")
        yield host, int(public_port or 443)
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def _edge_terminates_tls(args: argparse.Namespace) -> bool:
    return args.tunnel == "ngrok" and not args.public_endpoint


@contextlib.contextmanager
def _public_endpoint(args: argparse.Namespace) -> Iterator[tuple[str, int]]:
    if args.public_endpoint:
        host, _, port = args.public_endpoint.rpartition(":")
        if not host or not port.isdigit():
            raise SystemExit("--public-endpoint must be <host>:<port>")
        yield host, int(port)
    elif args.tunnel in {"ngrok", "ngrok-tcp"}:
        with ngrok_tunnel(args.port, tcp=args.tunnel == "ngrok-tcp") as endpoint:
            yield endpoint
    else:
        raise SystemExit(
            "--tunnel none needs --public-endpoint <host>:<port>, the address "
            "the Armnet cell reaches this learner at"
        )


def cmd_assets(args: argparse.Namespace) -> None:
    print(ensure_norm_stats(args.config_name, args.checkpoint_repo))


def cmd_cache(args: argparse.Namespace) -> None:
    _select_gpu(args.gpu)
    from armnet_rlt.cache_builder import build_task_cache

    presets = {
        "multitask": (MULTITASK_CONFIG, MULTITASK_CHECKPOINT, MULTITASK_DATASET, None),
        "green-button": (
            GREEN_BUTTON_CONFIG,
            GREEN_BUTTON_CHECKPOINT,
            GREEN_BUTTON_DATASET,
            GREEN_BUTTON_PROMPT,
        ),
    }
    config_name, checkpoint, dataset, prompt = presets[args.preset]
    build_task_cache(
        config_name=config_name,
        checkpoint_repo=checkpoint,
        dataset_repo=dataset,
        prompt=prompt,
        inference_batch_size=args.inference_batch_size,
        data_root=local_data_root(),
    )


def cmd_bc_cache(args: argparse.Namespace) -> None:
    _select_gpu(args.gpu)
    from armnet_rlt.cache_builder import build_rollout_cache, manifest_sources

    sources = manifest_sources(args.manifest)
    total = sum(len(entry["episodes"]) for entry in sources)
    print(f"[RLT bc] {len(sources)} datasets, {total} episodes -> {args.cache_name}")
    build_rollout_cache(
        sources=sources,
        data_root=local_data_root(),
        inference_batch_size=args.inference_batch_size,
        prompt=args.prompt,
        cache_name=args.cache_name,
    )


def cmd_bc(args: argparse.Namespace) -> None:
    _select_gpu(args.gpu)
    from armnet_rlt.learner import run_bc_training

    ensure_norm_stats(args.config_name, args.checkpoint_repo)
    cfg = _config(args)
    cfg.demo_cache_path = str(Path(cfg.demo_cache_path).with_name(args.cache_name))
    run_bc_training(cfg, steps=args.steps, wandb_project=args.wandb_project or None)


def cmd_streaming_bc(args: argparse.Namespace) -> None:
    _select_gpu(args.gpu)
    token = os.environ.get("RLT_LEARNER_AUTH_TOKEN", "")
    if not token:
        raise SystemExit(
            "RLT_LEARNER_AUTH_TOKEN is missing. Export it or put it in .env; it "
            "must equal the Armnet secret the actor is submitted with."
        )
    from armnet_rlt.learner import run_streaming_bc
    from armnet_rlt.learner_config import resolve_init_checkpoint

    ensure_norm_stats(args.config_name, args.checkpoint_repo)
    cfg = _config(
        args,
        bc_deviation_max=args.bc_deviation_max,
        bc_deviation_weight=args.bc_deviation_weight,
        actor_lr=args.actor_lr,
    )
    if args.online_steps:
        cfg.online_steps = args.online_steps
    transport = cfg.actor_learner
    transport.learner_port = args.port
    transport.policy_parameters_push_frequency = args.push_interval_s
    transport.auth_token = token
    entry: dict[str, Any] = {
        "use_tls": True,
        "config_name": args.config_name,
        "run_id": args.run_id,
    }
    if _edge_terminates_tls(args):
        # Bind the cleartext port to localhost.
        transport.server_bind_host = "127.0.0.1"
    else:
        certificate, tls_key = ensure_tls_certificate(local_data_root() / "tls")
        transport.server_bind_host = args.bind_host
        transport.tls_server_cert_path = str(certificate)
        transport.tls_server_key_path = str(tls_key)
        entry["tls_server_name"] = TLS_SERVER_NAME
        entry["tls_root_cert_pem"] = certificate.read_text()
    init_checkpoint = resolve_init_checkpoint(cfg, args.init_run_id, args.init_step)

    key = rendezvous_key(args.config_name, args.run_id)
    entry_path = local_rendezvous_path(key)
    with _public_endpoint(args) as (host, port):
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        entry_path.write_text(
            json.dumps({"host": host, "port": port, **entry}, indent=2)
        )
        print(
            f"[RLT] rendezvous {key}: {host}:{port} "
            f"({'TLS at ngrok' if _edge_terminates_tls(args) else 'TLS, self-signed'}) "
            f"-> {entry_path}",
            flush=True,
        )
        try:
            run_streaming_bc(
                cfg,
                wandb_project=args.wandb_project or None,
                min_success_transitions=args.min_success_transitions,
                steps_per_transition=args.steps_per_transition,
                init_checkpoint=init_checkpoint,
                min_shaped_reward=args.min_shaped_reward,
                base_batch_fraction=args.base_batch_fraction,
                episodes_per_round=args.episodes_per_round,
                steps_per_round=args.steps_per_round,
                idle_timeout_s=args.idle_timeout_s,
                idle_min_episodes=args.idle_min_episodes,
                # Skip pushes of unchanged weights.
                push_unchanged_weights=False,
            )
        finally:
            entry_path.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name: str, handler: Any, help_text: str) -> argparse.ArgumentParser:
        command = commands.add_parser(name, help=help_text)
        command.set_defaults(handler=handler)
        command.add_argument(
            "--gpu", default="", help="CUDA_VISIBLE_DEVICES for this run, e.g. 1"
        )
        return command

    def add_run(command: argparse.ArgumentParser) -> None:
        command.add_argument("--config-name", default=MULTITASK_CONFIG)
        command.add_argument("--run-id", default="")
        command.add_argument("--embodiment", default="so101")
        command.add_argument("--wandb-project", default="")
        command.add_argument(
            "--checkpoint-repo",
            default="",
            help="HuggingFace checkpoint to take normalization stats from",
        )

    assets = add("assets", cmd_assets, "download the normalization stats")
    assets.add_argument("--config-name", default=MULTITASK_CONFIG)
    assets.add_argument("--checkpoint-repo", default="")

    cache = add("cache", cmd_cache, "build a demonstration cache (needs OpenPI)")
    cache.add_argument(
        "--preset", choices=("multitask", "green-button"), default="multitask"
    )
    cache.add_argument("--inference-batch-size", type=int, default=4)

    bc_cache = add(
        "bc-cache", cmd_bc_cache, "build a cache from recorded rollouts (needs OpenPI)"
    )
    bc_cache.add_argument(
        "--manifest",
        required=True,
        help="JSON file with `datasets`, each holding `repo_id` and `episodes`",
    )
    bc_cache.add_argument("--inference-batch-size", type=int, default=4)
    bc_cache.add_argument("--prompt", default=GREEN_BUTTON_PROMPT)
    bc_cache.add_argument("--cache-name", default="rlt_bc_cache.pt")

    bc = add("bc", cmd_bc, "train an actor by cloning a cache's recorded actions")
    add_run(bc)
    bc.add_argument("--steps", type=int, default=5_000)
    bc.add_argument("--cache-name", default="rlt_bc_cache.pt")

    streaming = add(
        "streaming-bc",
        cmd_streaming_bc,
        "serve a learner that clones the successes a live actor sends",
    )
    add_run(streaming)
    streaming.add_argument("--min-success-transitions", type=int, default=32)
    streaming.add_argument("--steps-per-transition", type=int, default=1)
    streaming.add_argument("--online-steps", type=int, default=0)
    streaming.add_argument("--init-run-id", default="")
    streaming.add_argument("--init-step", type=int, default=0)
    streaming.add_argument("--min-shaped-reward", type=float, default=0.0)
    streaming.add_argument("--base-batch-fraction", type=float, default=0.0)
    streaming.add_argument("--bc-deviation-max", type=float, default=0.0)
    streaming.add_argument("--bc-deviation-weight", type=float, default=1.0)
    streaming.add_argument("--episodes-per-round", default="0")
    streaming.add_argument("--steps-per-round", type=int, default=0)
    streaming.add_argument("--actor-lr", type=float, default=0.0)
    streaming.add_argument("--idle-timeout-s", type=float, default=600.0)
    streaming.add_argument("--idle-min-episodes", type=int, default=5)
    streaming.add_argument(
        "--push-interval-s",
        type=float,
        default=5.0,
        help="Seconds between weight pushes while a round is training; a "
        "finished round always pushes",
    )
    streaming.add_argument("--bind-host", default="0.0.0.0")
    streaming.add_argument("--port", type=int, default=LEARNER_PORT)
    streaming.add_argument(
        "--tunnel",
        choices=("ngrok", "ngrok-tcp", "none"),
        default="ngrok",
        help="How the Armnet cell reaches this machine: an ngrok HTTPS "
        "endpoint, a raw ngrok TCP tunnel (paid accounts), or --public-endpoint",
    )
    streaming.add_argument(
        "--public-endpoint",
        default="",
        help="<host>:<port> the cell connects to; skips starting a tunnel",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    _load_env()
    args = _parser().parse_args(argv)
    if hasattr(args, "run_id") and not args.run_id:
        args.run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
    if hasattr(args, "run_id"):
        print(
            f"[RLT] learner key: {rendezvous_key(args.config_name, args.run_id)}",
            flush=True,
        )
    args.handler(args)


if __name__ == "__main__":
    main()
