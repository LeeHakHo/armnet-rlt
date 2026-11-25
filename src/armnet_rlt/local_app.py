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
    policy_rendezvous_key,
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
    """Whether TLS is terminated before this process, which then serves plaintext."""
    if args.public_endpoint:
        return bool(getattr(args, "public_endpoint_tls", False))
    return args.tunnel == "ngrok"


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
    config_name = args.config_name or config_name
    checkpoint = args.checkpoint or checkpoint
    dataset = args.dataset or dataset
    prompt = args.prompt or prompt
    build_task_cache(
        config_name=config_name,
        checkpoint_repo=checkpoint,
        dataset_repo=dataset,
        prompt=prompt,
        inference_batch_size=args.inference_batch_size,
        data_root=local_data_root(),
    )


def _learner_token() -> str:
    token = os.environ.get("RLT_LEARNER_AUTH_TOKEN", "")
    if not token:
        raise SystemExit(
            "RLT_LEARNER_AUTH_TOKEN is missing. Export it or put it in .env; it "
            "must equal the Armnet secret the actor is submitted with."
        )
    return token


@contextlib.contextmanager
def _serve(args: argparse.Namespace, cfg: Any, token: str) -> Iterator[None]:
    """Configure the gRPC transport and publish the rendezvous entry."""
    transport = cfg.actor_learner
    transport.learner_port = args.port
    transport.policy_parameters_push_frequency = args.push_interval_s
    transport.auth_token = token
    key = rendezvous_key(args.config_name, args.run_id)
    entry: dict[str, Any] = {
        "key": key,
        "use_tls": True,
        "config_name": args.config_name,
        "run_id": args.run_id,
    }
    tls_cert = str(getattr(args, "tls_cert", "") or "")
    tls_key = str(getattr(args, "tls_key", "") or "")
    if bool(tls_cert) != bool(tls_key):
        raise SystemExit("--tls-cert and --tls-key must be given together")
    if _edge_terminates_tls(args):
        if tls_cert:
            raise SystemExit("--tls-cert serves TLS itself; drop --public-endpoint-tls")
        transport.server_bind_host = args.bind_host or "127.0.0.1"
        tls_label = "TLS at the edge"
    elif tls_cert:
        # Serve TLS with a certificate the cell already trusts.
        transport.server_bind_host = args.bind_host or "0.0.0.0"
        transport.tls_server_cert_path = tls_cert
        transport.tls_server_key_path = tls_key
        tls_label = f"TLS, {tls_cert}"
    else:
        certificate, key_path = ensure_tls_certificate(local_data_root() / "tls")
        transport.server_bind_host = args.bind_host or "0.0.0.0"
        transport.tls_server_cert_path = str(certificate)
        transport.tls_server_key_path = str(key_path)
        entry["tls_server_name"] = TLS_SERVER_NAME
        entry["tls_root_cert_pem"] = certificate.read_text()
        tls_label = "TLS, self-signed"

    entry_path = local_rendezvous_path(key)
    with _public_endpoint(args) as (host, port):
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        entry_path.write_text(
            json.dumps({"host": host, "port": port, **entry}, indent=2)
        )
        print(
            f"[RLT] rendezvous {key}: {host}:{port} ({tls_label}) -> {entry_path}",
            flush=True,
        )
        try:
            yield
        finally:
            entry_path.unlink(missing_ok=True)


def _openpi_backend(args: argparse.Namespace) -> tuple[Any, tuple[int, int]]:
    """The OpenPI policy to serve and its image size, warmed up."""
    import numpy as np
    from openpi.models.model import IMAGE_RESOLUTION

    from armnet_rlt.openpi_rlt import OpenPIRLTPolicy, PolicyObservation

    policy = OpenPIRLTPolicy.from_checkpoint(
        args.checkpoint, args.config_name, default_prompt=args.prompt
    )
    height, width = IMAGE_RESOLUTION
    # Compile the sampler and the split prefix/denoise path before serving.
    warm_observation = PolicyObservation(
        state=np.zeros(6, dtype=np.float32),
        images={
            name: np.zeros((height, width, 3), dtype=np.uint8)
            for name in ("front", "top", "wrist")
        },
        prompt=args.prompt,
    )
    policy.infer(warm_observation)
    handle, _token = policy.prefix(warm_observation)
    policy.denoise(handle, None)
    return policy, (height, width)


def _rl_backend(args: argparse.Namespace) -> tuple[Any, tuple[int, int]]:
    """π0.5 (unless the algorithm does without it) and the algorithm's actor,
    from --actor-checkpoint, the learner feed or the cell."""
    import torch

    from armnet_rlt.algorithm import load_algorithm
    from armnet_rlt.frozen import frozen_actor_payload, load_frozen_checkpoint
    from armnet_rlt.policy_actor import AlgorithmPolicy

    frozen = load_frozen_checkpoint(args.actor_checkpoint) if args.actor_checkpoint else None
    name = args.algorithm or (str(frozen["algorithm"]) if frozen else "")
    if frozen and args.algorithm and str(frozen["algorithm"]) != args.algorithm:
        raise SystemExit(
            f"--actor-checkpoint holds a {frozen['algorithm']} actor, not {args.algorithm}"
        )
    uses_base = load_algorithm(name).uses_base if name else True
    base = image_size = None
    if uses_base:
        if not args.checkpoint:
            raise SystemExit("--checkpoint, the π0.5 checkpoint, is required")
        base, image_size = _openpi_backend(args)
    device = args.actor_device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    source = "checkpoint" if frozen else "learner" if args.steering_from_learner else "client"
    runner = AlgorithmPolicy(base, algorithm=name, device=device, actor_source=source)
    if frozen is not None:
        step = runner.install(frozen_actor_payload(frozen))
        print(
            f"[RLT] policy server runs the {frozen['algorithm']} actor of "
            f"{args.actor_checkpoint} (learner step {step})",
            flush=True,
        )
    return runner, tuple(image_size or runner.image_size)


def _lerobot_backend(args: argparse.Namespace) -> Any:
    """A LeRobot policy backend for the checkpoint in --checkpoint."""
    import torch

    from armnet_rlt.lerobot_actor import (
        image_shapes,
        load_lerobot_policy,
        parse_image_size,
        policy_rename_map,
    )
    from armnet_rlt.lerobot_server import LeRobotPolicyBackend

    if not args.checkpoint:
        raise SystemExit("--policy-type lerobot needs --checkpoint")
    device = "cuda" if args.gpu != "cpu" and torch.cuda.is_available() else "cpu"
    policy, preprocessor, postprocessor = load_lerobot_policy(args.checkpoint, device)
    return LeRobotPolicyBackend(
        policy,
        preprocessor,
        postprocessor,
        shapes=image_shapes(
            policy,
            rename_map=policy_rename_map(args.checkpoint),
            size=parse_image_size(args.policy_image_size),
        ),
        n_action_steps=args.n_action_steps,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
        served_from_learner=bool(args.steering_from_learner),
    )


def cmd_policy_server(args: argparse.Namespace) -> None:
    _select_gpu(args.gpu)
    from armnet_rlt.remote_policy import serve

    token = _learner_token()
    if args.policy_type == "lerobot":
        policy = _lerobot_backend(args)
        image_size = None
    else:
        policy, image_size = _rl_backend(args)
    edge_tls = _edge_terminates_tls(args)
    own_tls = bool(args.tls_cert or args.tls_key)
    if own_tls and edge_tls:
        raise SystemExit("--tls-cert serves TLS itself; drop --public-endpoint-tls")
    bind_host = args.bind_host or ("127.0.0.1" if edge_tls else "0.0.0.0")
    key = policy_rendezvous_key(args.config_name)
    entry_path = local_rendezvous_path(key)
    with serve(
        policy,
        bind=f"{bind_host}:{args.port}",
        image_size=image_size,
        auth_token=token,
        tls_cert_path=args.tls_cert,
        tls_key_path=args.tls_key,
        steering_from_learner=bool(args.steering_from_learner),
    ) as server, _public_endpoint(args) as (host, port):
        if args.steering_from_learner:
            from armnet_rlt.remote_policy import MAX_MESSAGE_BYTES
            from armnet_rlt.steering_feed import LearnerSteeringFeed

            learner_host, _, learner_port = args.steering_from_learner.rpartition(":")
            feed_options: dict[str, Any] = {}
            if args.policy_type == "lerobot":
                feed_options["installer"] = lambda weights, step, partial: server.run_locked(
                    policy.apply_weights, weights, step, partial
                )
            feed = LearnerSteeringFeed(
                server.apply_steering,
                host=learner_host or "127.0.0.1",
                port=int(learner_port),
                use_tls=bool(args.steering_learner_tls_name),
                token=token,
                server_name=args.steering_learner_tls_name,
                max_message_bytes=MAX_MESSAGE_BYTES,
                **feed_options,
            )
            feed.start()
            print(
                f"[RLT] policy server taking updates from the learner at "
                f"{args.steering_from_learner}",
                flush=True,
            )
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        entry_path.write_text(
            json.dumps(
                {
                    "key": key,
                    "host": host,
                    "port": port,
                    "use_tls": edge_tls or own_tls,
                    "config_name": args.config_name,
                    "checkpoint": str(args.checkpoint),
                    "policy_type": args.policy_type,
                    "algorithm": str(getattr(args, "algorithm", "") or ""),
                },
                indent=2,
            )
        )
        tls_label = (
            "TLS at the edge"
            if edge_tls
            else f"TLS, {args.tls_cert}" if own_tls else "no TLS"
        )
        print(
            f"[RLT] policy server {key}: {host}:{port} ({tls_label}) -> {entry_path}",
            flush=True,
        )
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            entry_path.unlink(missing_ok=True)


def cmd_front(args: argparse.Namespace) -> None:
    import signal
    import sys

    from armnet_rlt.front import serve_front

    token = _learner_token()
    own_tls = bool(args.tls_cert or args.tls_key)
    edge_tls = args.public_endpoint_tls or (args.tunnel == "ngrok" and not args.public_endpoint)
    if own_tls == edge_tls:
        raise SystemExit(
            "the front serves the cell over TLS: --tls-cert and --tls-key, or an ngrok "
            "tunnel or --public-endpoint-tls, not both"
        )
    bind = f"{args.bind_host or ('127.0.0.1' if edge_tls else '0.0.0.0')}:{args.port}"
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    upstreams = f"learner 127.0.0.1:{args.learner_port}, policy 127.0.0.1:{args.policy_port}"
    with serve_front(
        bind=bind,
        learner=f"127.0.0.1:{args.learner_port}",
        policy=f"127.0.0.1:{args.policy_port}",
        auth_token=token,
        tls_cert_path=args.tls_cert,
        tls_key_path=args.tls_key,
    ), contextlib.ExitStack() as stack:
        label = "TLS at the edge" if edge_tls else f"TLS, {args.tls_cert}"
        if args.tunnel == "none" and not args.public_endpoint:
            print(f"[RLT] front on {bind} ({label}) -> {upstreams}", flush=True)
        else:
            host, port = stack.enter_context(_public_endpoint(args))
            print(f"[RLT] front {host}:{port} ({label}) on {bind} -> {upstreams}", flush=True)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass


def cmd_lerobot_streaming_bc(args: argparse.Namespace) -> None:
    _select_gpu(args.gpu)
    token = _learner_token()
    from types import SimpleNamespace

    from armnet_rlt.config import RLTActorLearnerConfig
    from armnet_rlt.lerobot_learner import run_lerobot_streaming_bc

    cfg = SimpleNamespace(actor_learner=RLTActorLearnerConfig())
    with _serve(args, cfg, token):
        run_lerobot_streaming_bc(
            cfg,
            policy_path=args.policy_path,
            run_dir=local_data_root() / "lerobot-sbc" / args.run_id,
            run_id=args.run_id,
            task=args.task,
            episodes_per_round=args.episodes_per_round,
            steps_per_round=args.steps_per_round,
            save_freq=args.save_freq,
            batch_size=args.batch_size,
            online_steps=args.online_steps,
            lr=args.lr,
            ema_decay=args.ema_decay,
            base_dataset=args.base_dataset,
            base_batch_fraction=args.base_batch_fraction,
            num_workers=args.num_workers,
            min_shaped_reward=args.min_shaped_reward,
            idle_timeout_s=args.idle_timeout_s,
            idle_min_episodes=args.idle_min_episodes,
            wandb_project=args.wandb_project or None,
            resume=args.resume,
        )


def cmd_checkpoint_eval(args: argparse.Namespace) -> None:
    token = _learner_token()
    from types import SimpleNamespace

    from armnet_rlt.checkpoint_eval import run_checkpoint_eval
    from armnet_rlt.config import RLTActorLearnerConfig

    cfg = SimpleNamespace(actor_learner=RLTActorLearnerConfig())
    with _serve(args, cfg, token):
        run_checkpoint_eval(
            cfg,
            train_dir=args.train_dir or None,
            checkpoints=args.checkpoint or None,
            base_checkpoint=args.base_checkpoint or None,
            run_dir=local_data_root() / "checkpoint-eval" / args.run_id,
            run_id=args.run_id,
            rollouts_per_checkpoint=args.rollouts_per_checkpoint,
            final_step=args.final_step,
            first_step=args.first_step,
            wandb_project=args.wandb_project or None,
        )


def cmd_online(args: argparse.Namespace) -> None:
    _select_gpu(args.gpu)
    token = _learner_token()
    from armnet_rlt.learner import run_learner

    ensure_norm_stats(args.config_name, args.checkpoint_repo)
    cfg = _config(
        args,
        max_demo_pretrain_steps=args.pretrain_steps,
        online_steps=args.online_steps,
        online_batch_fraction=args.online_batch_fraction,
        save_freq=args.save_freq,
        algorithm=args.algorithm,
        dsrl_noise_bound=args.dsrl_noise_bound,
        dsrl_noise_steps=args.dsrl_noise_steps,
        dsrl_target_entropy=args.dsrl_target_entropy,
        dsrl_alpha_init=args.dsrl_alpha_init,
        residual_actor=args.residual_actor,
        residual_scale=args.residual_scale,
        critic_action_residual=args.critic_action_residual,
        critic_action_gain=args.critic_action_gain,
        safety_trip_penalty=args.safety_trip_penalty,
        algorithm_options=_algorithm_options(args.algorithm_option),
    )
    with _serve(args, cfg, token):
        run_learner(
            cfg,
            offline_only=False,
            resume=args.resume,
            wandb_project=args.wandb_project or None,
            idle_timeout_s=args.idle_timeout_s,
            idle_min_episodes=args.idle_min_episodes,
            max_episodes=args.max_episodes,
            checkpoint_every_episodes=args.checkpoint_every_episodes,
            reward_plugin=_load_callable(args.reward_plugin) if args.reward_plugin else None,
        )


def _algorithm_options(values: list[str] | None) -> dict[str, Any]:
    """``KEY=VALUE`` pairs; a value is read as JSON when it parses, else kept
    as a string."""
    options: dict[str, Any] = {}
    for item in values or ():
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--algorithm-option must be KEY=VALUE, got {item!r}")
        try:
            options[key] = json.loads(value)
        except json.JSONDecodeError:
            options[key] = value
    return options


def _load_callable(spec: str) -> Any:
    """The function named by ``module:function``."""
    import importlib

    module, _, name = spec.partition(":")
    if not module or not name:
        raise SystemExit(f"--reward-plugin must be module:function, got {spec!r}")
    return getattr(importlib.import_module(module), name)


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

    def add_transport(command: argparse.ArgumentParser, push_help: str) -> None:
        command.add_argument(
            "--push-interval-s", type=float, default=5.0, help=push_help
        )
        command.add_argument(
            "--bind-host",
            default="",
            help="Listen address (default 0.0.0.0, or 127.0.0.1 with --public-endpoint-tls)",
        )
        command.add_argument("--port", type=int, default=LEARNER_PORT)
        command.add_argument(
            "--tunnel",
            choices=("ngrok", "ngrok-tcp", "none"),
            default="ngrok",
            help="How the cell reaches this machine: an ngrok HTTPS endpoint, "
            "a raw ngrok TCP tunnel, or --public-endpoint",
        )
        command.add_argument(
            "--public-endpoint",
            default="",
            help="<host>:<port> the cell connects to; skips starting a tunnel",
        )
        command.add_argument(
            "--public-endpoint-tls",
            action="store_true",
            help="The public endpoint terminates TLS and forwards plaintext; "
            "serve without TLS",
        )
        command.add_argument(
            "--tls-cert",
            default="",
            help="Serve TLS with this PEM certificate chain instead of the "
            "self-signed one",
        )
        command.add_argument("--tls-key", default="", help="Key for --tls-cert")

    assets = add("assets", cmd_assets, "download the normalization stats")
    assets.add_argument("--config-name", default=MULTITASK_CONFIG)
    assets.add_argument("--checkpoint-repo", default="")

    cache = add("cache", cmd_cache, "build a demonstration cache (needs OpenPI)")
    cache.add_argument(
        "--preset", choices=("multitask", "green-button"), default="multitask"
    )
    cache.add_argument("--inference-batch-size", type=int, default=4)
    cache.add_argument("--config-name", default="", help="Overrides the preset")
    cache.add_argument(
        "--checkpoint",
        default="",
        help="HuggingFace repo or local OpenPI checkpoint; overrides the preset",
    )
    cache.add_argument("--dataset", default="", help="Overrides the preset")
    cache.add_argument("--prompt", default="", help="Overrides the preset")

    lerobot_sbc = add(
        "lerobot-streaming-bc",
        cmd_lerobot_streaming_bc,
        "serve a learner that fine-tunes a LeRobot policy on the successes a live actor sends",
    )
    lerobot_sbc.add_argument("--config-name", default="eval")
    lerobot_sbc.add_argument("--run-id", default="")
    lerobot_sbc.add_argument("--wandb-project", default="")
    lerobot_sbc.add_argument(
        "--policy-path", default="", help="pretrained_model directory to start from"
    )
    lerobot_sbc.add_argument(
        "--task", default=GREEN_BUTTON_PROMPT, help="Instruction for a language-conditioned policy"
    )
    lerobot_sbc.add_argument(
        "--resume",
        action="store_true",
        help="Continue the run of --run-id from its newest checkpoint and saved successes",
    )
    lerobot_sbc.add_argument("--episodes-per-round", default="30,30,20")
    lerobot_sbc.add_argument("--steps-per-round", type=int, default=300)
    lerobot_sbc.add_argument(
        "--save-freq", type=int, default=300, help="Learner steps between checkpoints and weight pushes"
    )
    lerobot_sbc.add_argument("--batch-size", type=int, default=32)
    lerobot_sbc.add_argument("--online-steps", type=int, default=1800)
    lerobot_sbc.add_argument(
        "--lr", type=float, default=0.0, help="0 keeps the policy's optimizer_lr"
    )
    lerobot_sbc.add_argument(
        "--ema-decay",
        type=float,
        default=0.998,
        help="Checkpoint and push a moving average of the weights; 0 turns it off",
    )
    lerobot_sbc.add_argument(
        "--base-dataset", default="", help="Offline LeRobot dataset mixed into every batch"
    )
    lerobot_sbc.add_argument("--base-batch-fraction", type=float, default=0.0)
    lerobot_sbc.add_argument("--num-workers", type=int, default=8)
    lerobot_sbc.add_argument("--min-shaped-reward", type=float, default=0.0)
    lerobot_sbc.add_argument("--idle-timeout-s", type=float, default=600.0)
    lerobot_sbc.add_argument("--idle-min-episodes", type=int, default=5)
    add_transport(
        lerobot_sbc,
        "Minimum seconds between weight pushes; weights are pushed when a round finishes",
    )

    checkpoint_eval = add(
        "checkpoint-eval",
        cmd_checkpoint_eval,
        "serve a lerobot-train run's checkpoints to a cell actor and log each one's success rate",
    )
    checkpoint_eval.add_argument("--config-name", default="eval")
    checkpoint_eval.add_argument("--run-id", default="")
    checkpoint_eval.add_argument("--wandb-project", default="")
    checkpoint_eval.add_argument(
        "--train-dir", default="", help="lerobot-train --output_dir whose checkpoints are evaluated in step order"
    )
    checkpoint_eval.add_argument(
        "--checkpoint",
        action="append",
        help="A pretrained_model directory to evaluate (repeatable); labelled by "
        "the step in its path. Replaces --train-dir",
    )
    checkpoint_eval.add_argument(
        "--base-checkpoint",
        default="",
        help="pretrained_model directory the actor loads at start (default: the first --checkpoint)",
    )
    checkpoint_eval.add_argument("--rollouts-per-checkpoint", type=int, default=10)
    checkpoint_eval.add_argument(
        "--final-step", type=int, default=None, help="Stop after this checkpoint (default: the last --checkpoint)"
    )
    checkpoint_eval.add_argument(
        "--first-step", type=int, default=0, help="Skip checkpoints before this step"
    )
    add_transport(checkpoint_eval, "Minimum seconds between weight pushes")

    online = add(
        "online",
        cmd_online,
        "serve the TD3 learner: demo pretraining, then online updates",
    )
    add_run(online)
    online.add_argument("--pretrain-steps", type=int, default=5_000)
    online.add_argument("--online-steps", type=int, default=300_000)
    online.add_argument("--online-batch-fraction", type=float, default=0.5)
    online.add_argument(
        "--save-freq", type=int, default=None, help="Learner steps between checkpoints (config default 500)"
    )
    online.add_argument("--resume", action="store_true")
    online.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Stop after this many actor episodes (0: stop at --online-steps)",
    )
    online.add_argument(
        "--reward-plugin",
        default="",
        help="module:function called with each episode's transitions and camera frames "
        "(from an --episode-frames actor) to rewrite its rewards, e.g. "
        "armnet_rlt.robometer_reward:shape",
    )
    online.add_argument(
        "--checkpoint-every-episodes",
        type=int,
        default=0,
        help="Also save a checkpoint every this many episodes (episode_checkpoints.jsonl)",
    )
    online.add_argument("--idle-timeout-s", type=float, default=600.0)
    online.add_argument("--idle-min-episodes", type=int, default=5)
    online.add_argument(
        "--algorithm",
        default="rlt",
        help="A registered RL algorithm: rlt refines the base chunk; dsrl steers "
        "the base policy's noise (no demo pretraining, online transitions only); "
        "an installed plugin's entry-point name; or module:Class",
    )
    online.add_argument(
        "--algorithm-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Option of a plugin algorithm (config.algorithm_options); repeatable, "
        "values parsed as JSON where they parse",
    )
    online.add_argument(
        "--dsrl-noise-bound",
        type=float,
        default=1.5,
        help="DSRL noise actions are clipped to [-bound, bound]",
    )
    online.add_argument(
        "--dsrl-noise-steps",
        type=int,
        default=5,
        help="Rows of noise the DSRL actor outputs; the last row is repeated "
        "over the rest of the chunk",
    )
    online.add_argument(
        "--dsrl-target-entropy",
        type=float,
        default=0.0,
        help="DSRL SAC target entropy of the noise policy, summed over its "
        "noise dimensions",
    )
    online.add_argument(
        "--dsrl-alpha-init",
        type=float,
        default=1.0,
        help="DSRL SAC initial entropy temperature",
    )
    online.add_argument(
        "--residual-actor",
        action="store_true",
        help="rlt only: actor = reference + residual_scale * tanh(head), starting at the reference",
    )
    online.add_argument("--residual-scale", type=float, default=0.25)
    online.add_argument(
        "--critic-action-residual",
        action="store_true",
        help="rlt only: critic sees critic_action_gain * (action - reference chunk)",
    )
    online.add_argument(
        "--safety-trip-penalty",
        type=float,
        default=None,
        help="terminal reward of a safety-tripped episode is this many "
        "success-failure reward gaps below a failure (dsrl and rlt)",
    )
    online.add_argument("--critic-action-gain", type=float, default=10.0)
    add_transport(online, "Seconds between weight pushes")

    policy_server = add(
        "policy-server",
        cmd_policy_server,
        "serve π0.5 and an RL algorithm's actor to cell actors started with --remote-policy",
    )
    policy_server.add_argument("--config-name", default=MULTITASK_CONFIG)
    policy_server.add_argument(
        "--policy-type",
        choices=("openpi", "lerobot"),
        default="openpi",
        help="openpi: the RLT/DSRL base policy; lerobot: a LeRobot policy "
        "(Diffusion, SmolVLA, ...) for --policy-type lerobot actors",
    )
    policy_server.add_argument(
        "--checkpoint",
        default="",
        help="Local OpenPI checkpoint directory (not needed by an algorithm that "
        "does not use π0.5), or a LeRobot pretrained_model directory with "
        "--policy-type lerobot",
    )
    policy_server.add_argument(
        "--algorithm",
        default="",
        help="The RL algorithm whose actor the server runs (rlt, dsrl, a plugin); "
        "empty takes it from the actor the learner or the cell sends. Required "
        "for an algorithm that does not use π0.5",
    )
    policy_server.add_argument(
        "--actor-checkpoint",
        default="",
        help="Frozen actor (rlt-freeze-checkpoint output) to serve, for evaluation",
    )
    policy_server.add_argument(
        "--actor-device",
        default="cuda",
        help="Torch device of the algorithm's actor (cpu when CUDA is missing)",
    )
    policy_server.add_argument("--prompt", default=GREEN_BUTTON_PROMPT)
    policy_server.add_argument(
        "--n-action-steps",
        type=int,
        default=0,
        help="Actions a chunking LeRobot policy returns per inference; "
        "0 keeps the policy's n_action_steps",
    )
    policy_server.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=None,
        help="Temporal ensembling coefficient for a chunking LeRobot policy",
    )
    policy_server.add_argument(
        "--policy-image-size",
        default="",
        help="<height>x<width> the server resizes camera images to for a "
        "LeRobot policy; empty uses the policy config",
    )
    policy_server.add_argument(
        "--bind-host",
        default="",
        help="Listen address (default 0.0.0.0, or 127.0.0.1 with --public-endpoint-tls "
        "or an ngrok tunnel)",
    )
    policy_server.add_argument("--port", type=int, default=LEARNER_PORT + 1)
    policy_server.add_argument(
        "--tunnel",
        choices=("ngrok", "none"),
        default="ngrok",
        help="An ngrok HTTPS endpoint, or --public-endpoint (plaintext unless "
        "--public-endpoint-tls)",
    )
    policy_server.add_argument(
        "--public-endpoint",
        default="",
        help="<host>:<port> the cell connects to; skips starting a tunnel",
    )
    policy_server.add_argument(
        "--public-endpoint-tls",
        action="store_true",
        help="The public endpoint terminates TLS and forwards plaintext; "
        "the cell connects with TLS",
    )
    policy_server.add_argument(
        "--tls-cert",
        default="",
        help="Serve TLS with this PEM certificate chain behind a raw TCP forwarder",
    )
    policy_server.add_argument("--tls-key", default="", help="Key for --tls-cert")
    policy_server.add_argument(
        "--steering-from-learner",
        default="",
        help="<host>:<port> of the learner; the server installs the algorithm's "
        "actor or the checkpoint-eval weights from its parameter stream",
    )
    policy_server.add_argument(
        "--steering-learner-tls-name",
        default="",
        help="TLS name of the learner's certificate; plaintext when empty",
    )
    policy_server.add_argument("--run-id", default="policy-server")
    policy_server.add_argument("--embodiment", default="so101")

    front = commands.add_parser(
        "front",
        help="One public endpoint for the learner and the policy server on this machine",
    )
    front.set_defaults(handler=cmd_front)
    front.add_argument("--port", type=int, default=LEARNER_PORT - 1)
    front.add_argument("--learner-port", type=int, default=LEARNER_PORT)
    front.add_argument("--policy-port", type=int, default=LEARNER_PORT + 1)
    front.add_argument(
        "--bind-host",
        default="",
        help="Listen address (default 0.0.0.0, or 127.0.0.1 behind TLS at the edge)",
    )
    front.add_argument(
        "--tunnel",
        choices=("ngrok", "none"),
        default="ngrok",
        help="An ngrok HTTPS endpoint, or --public-endpoint",
    )
    front.add_argument(
        "--public-endpoint", default="", help="<host>:<port> the cell connects to"
    )
    front.add_argument(
        "--public-endpoint-tls",
        action="store_true",
        help="TLS is terminated before the front, which serves plaintext",
    )
    front.add_argument("--tls-cert", default="", help="Serve TLS with this PEM certificate chain")
    front.add_argument("--tls-key", default="", help="Key for --tls-cert")
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
