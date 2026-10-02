"""Modal deployment for the standalone RLT learner."""

from __future__ import annotations

import datetime as dt
import os

import modal

from armnet_rlt.resources import (
    APP_NAME,
    AUTH_SECRET_NAME,
    LEARNER_PORT,
    RENDEZVOUS_DICT,
    VOLUME_MOUNT,
    VOLUME_NAME,
    rendezvous_key,
    task_paths,
)


LEROBOT_REVISION = "4eecbad32b0380c64bd5574bbf74e56e5255bdea"
GPU = "A10G"
FUNCTION_TIMEOUT_S = 24 * 60 * 60

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
rendezvous = modal.Dict.from_name(RENDEZVOUS_DICT, create_if_missing=True)
auth_secret = modal.Secret.from_name(AUTH_SECRET_NAME)
# Carries WANDB_API_KEY from .env, resolved from this file upward.
dotenv_secret = modal.Secret.from_dotenv(__file__)

# This image deliberately has no OpenPI/JAX. The learner consumes cached tokens
# and normalization JSON, so installing the VLA would only add build time and
# GPU-memory contention.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04", add_python="3.12"
    )
    .env({"PYTHONPATH": "/root"})
    .apt_install("git", "ca-certificates")
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel uv",
        "printf '%s\\n' "
        "'pynput>=0.0.0; sys_platform == \"never\"' "
        "'evdev>=0.0.0; sys_platform == \"never\"' "
        "'torchcodec>=0.0.0; sys_platform == \"never\"' "
        "'torch==2.8.0' "
        "'torchvision==0.23.0' "
        "'numpy>=2.0,<2.3' "
        "'grpcio==1.73.1' "
        "'protobuf>=6.31.1,<6.32.0' "
        "'setuptools>=71,<81' "
        "> /tmp/overrides.txt",
        "uv pip install --system --override /tmp/overrides.txt "
        "grpcio==1.73.1 'protobuf>=6.31.1,<6.32.0' "
        "'numpy>=2.0,<2.3' tqdm 'wandb>=0.24,<0.25' "
        "'huggingface-hub>=1,<2'",
        "uv pip install --system --no-deps --override /tmp/overrides.txt "
        f"'lerobot @ git+https://github.com/huggingface/lerobot.git@{LEROBOT_REVISION}'",
        # Torch last prevents LeRobot or another dependency selecting a CPU wheel.
        "uv pip install --upgrade --system --override /tmp/overrides.txt "
        "--extra-index-url https://download.pytorch.org/whl/cu129 "
        "torch==2.8.0 torchvision==0.23.0",
    )
    .add_local_dir(
        "src/armnet_rlt",
        remote_path="/root/armnet_rlt",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc"],
    )
    .run_commands(
        "python -c \"import torch; from lerobot.rl.learner_service import LearnerService; "
        "from armnet_rlt.learner import run_learner; "
        "print('RLT learner image imports OK', torch.__version__)\""
    )
)

app = modal.App(APP_NAME)


def _network(embodiment: str, reference_action_len: int):
    from armnet_rlt.config import (
        BISO101_JERK_JOINT_WEIGHTS,
        SO101_JERK_JOINT_WEIGHTS,
        biso101_network_config,
        so101_network_config,
    )

    if embodiment == "so101":
        return (
            so101_network_config(reference_action_len=reference_action_len),
            SO101_JERK_JOINT_WEIGHTS,
        )
    if embodiment == "biso101":
        return (
            biso101_network_config(reference_action_len=reference_action_len),
            BISO101_JERK_JOINT_WEIGHTS,
        )
    raise ValueError("embodiment must be 'so101' or historical learner-only 'biso101'")


def _config(
    *,
    embodiment: str,
    config_name: str,
    run_id: str,
    output_dir: str | None = None,
    batch_size: int = 256,
    max_demo_pretrain_steps: int = 5_000,
    online_steps: int = 300_000,
    utd_ratio: int | None = None,
    online_batch_fraction: float = 0.5,
):
    from armnet_rlt.config import RLTConfig

    cache_path, assets_dir, production_output = task_paths(config_name)
    network, jerk = _network(embodiment, reference_action_len=30)
    kwargs = {
        "embodiment": embodiment,
        "network": network,
        "jerk_joint_weights": jerk,
        "assets_dir": assets_dir,
        "demo_cache_path": cache_path,
        "output_dir": output_dir or production_output,
        "run_id": run_id,
        "device": "cuda",
        "batch_size": batch_size,
        "max_demo_pretrain_steps": max_demo_pretrain_steps,
        "online_steps": online_steps,
        "policy_uses_delta_actions": True,
        "online_batch_fraction": online_batch_fraction,
    }
    if utd_ratio is not None:
        kwargs["utd_ratio"] = utd_ratio
    return RLTConfig(**kwargs)


@app.function(
    image=image,
    gpu=GPU,
    volumes={str(VOLUME_MOUNT): volume},
    secrets=[auth_secret, dotenv_secret],
    timeout=FUNCTION_TIMEOUT_S,
)
def serve_learner(
    config_name: str,
    run_id: str,
    embodiment: str = "so101",
    resume: bool = False,
    wandb_project: str = "",
    online_batch_fraction: float = 0.5,
) -> dict:
    """Train and serve an online learner through Modal's HTTP/2 TLS tunnel."""
    from armnet_rlt.learner import run_learner

    cfg = _config(
        embodiment=embodiment,
        config_name=config_name,
        run_id=run_id,
        online_batch_fraction=online_batch_fraction,
    )
    cfg.actor_learner.server_bind_host = "0.0.0.0"
    cfg.actor_learner.learner_port = LEARNER_PORT
    cfg.actor_learner.auth_token = os.environ["RLT_LEARNER_AUTH_TOKEN"]
    key = rendezvous_key(config_name, run_id)

    # Modal terminates TLS but explicitly advertises h2 via ALPN. gRPC is
    # plaintext only on the private container side; public traffic is encrypted.
    with modal.forward(LEARNER_PORT, h2_enabled=True) as tunnel:
        host, port = tunnel.tls_socket
        rendezvous[key] = {
            "host": host,
            "port": port,
            "use_tls": True,
            "config_name": config_name,
            "run_id": run_id,
        }
        print(f"[RLT] rendezvous {key}: {host}:{port} (TLS/h2)", flush=True)
        try:
            return run_learner(
                cfg,
                offline_only=False,
                wandb_project=wandb_project or None,
                resume=resume,
            )
        finally:
            rendezvous.pop(key, None)


@app.function(
    image=image,
    volumes={str(VOLUME_MOUNT): volume},
    timeout=20 * 60,
)
def inspect_artifacts(
    config_name: str,
    embodiment: str = "so101",
    run_id: str = "",
    sample_size: int = 128,
) -> dict:
    """Validate cache/norm shapes and optionally inspect one learned actor."""
    import torch

    from armnet_rlt.artifacts import load_norm_stats, load_transition_cache
    from armnet_rlt.learner import _latest_checkpoint
    from armnet_rlt.policy import RLTPolicy

    cfg = _config(
        embodiment=embodiment,
        config_name=config_name,
        run_id=run_id or "artifact-inspection",
    )
    cfg.device = "cpu"
    norm_stats = load_norm_stats(
        cfg.assets_dir,
        use_delta_actions=cfg.policy_uses_delta_actions,
        network=cfg.network,
    )
    transitions, summary = load_transition_cache(
        cfg.demo_cache_path, cfg.network
    )
    result: dict = {
        **summary.to_dict(),
        "config_name": config_name,
        "policy_uses_delta_actions": cfg.policy_uses_delta_actions,
        "norm_stats": {
            name: {
                field: list(getattr(stats, field).shape)
                if getattr(stats, field) is not None
                else None
                for field in ("mean", "std", "q01", "q99")
            }
            for name, stats in norm_stats.items()
        },
    }
    if not run_id:
        print(f"[RLT inspect] {result}", flush=True)
        return result

    checkpoint = _latest_checkpoint(cfg)
    if checkpoint is None:
        raise FileNotFoundError(f"no checkpoint found for run {run_id!r}")
    policy = RLTPolicy(cfg)
    policy.set_norm_stats(norm_stats, use_quantiles=cfg.use_quantile_norm)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    policy.load_state_dict(payload["policy_state_dict"])
    policy.eval()

    count = min(max(1, int(sample_size)), len(transitions))
    sampled = transitions[:count]
    token = torch.stack([item["rl_token"] for item in sampled])
    proprioception = torch.stack(
        [item["proprioception"] for item in sampled]
    )
    reference = torch.stack(
        [item["reference_action"] for item in sampled]
    )
    with torch.inference_mode():
        _, normalized_mean = policy.actor(
            token, proprioception, reference, sample=False
        )
        commands = policy.actor.unnormalize_action(
            normalized_mean, proprioception
        )
    commands = commands.reshape(count, cfg.network.predicted_action_len, -1)
    base = reference[:, : cfg.network.predicted_action_dim].reshape_as(commands)
    deviation = (commands - base).abs()
    result["actor_sanity"] = {
        "checkpoint": str(checkpoint),
        "learner_step": int(payload["step"]),
        "sample_size": count,
        "finite": bool(torch.isfinite(commands).all()),
        "normalized_abs_max": float(normalized_mean.abs().max()),
        "normalized_saturation_rate": float(
            (normalized_mean.abs() >= 0.99).float().mean()
        ),
        "command_min_by_joint": commands.amin(dim=(0, 1)).tolist(),
        "command_max_by_joint": commands.amax(dim=(0, 1)).tolist(),
        "reference_min_by_joint": base.amin(dim=(0, 1)).tolist(),
        "reference_max_by_joint": base.amax(dim=(0, 1)).tolist(),
        "deviation_mean_by_joint": deviation.mean(dim=(0, 1)).tolist(),
        "deviation_max_by_joint": deviation.amax(dim=(0, 1)).tolist(),
    }
    print(f"[RLT inspect] {result}", flush=True)
    return result


@app.function(
    image=image,
    gpu=GPU,
    volumes={str(VOLUME_MOUNT): volume},
    timeout=20 * 60,
)
def smoke_bimanual() -> dict:
    """Run a few real TD3 updates against the historical 12-DoF cache."""
    from armnet_rlt.learner import run_learner

    cfg = _config(
        embodiment="biso101",
        config_name="pi05_rlt_busybox_multitask",
        run_id="migration-smoke",
        output_dir="/tmp/armnet-rlt-smoke",
        batch_size=8,
        max_demo_pretrain_steps=2,
        online_steps=2,
        utd_ratio=1,
    )
    result = run_learner(
        cfg,
        offline_only=True,
        wandb_project=None,
        resume=False,
    )
    print(f"[RLT] bimanual migration smoke: {result}", flush=True)
    return result


@app.local_entrypoint()
def main(
    config_name: str,
    run_id: str = "",
    embodiment: str = "so101",
    resume: bool = False,
    wandb_project: str = "",
    online_batch_fraction: float = 0.5,
) -> None:
    """Start the online learner. Use `modal run --detach` for long runs."""
    run_id = run_id or dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
    key = rendezvous_key(config_name, run_id)
    print(f"[RLT] learner key: {key}", flush=True)
    serve_learner.remote(
        config_name=config_name,
        run_id=run_id,
        embodiment=embodiment,
        resume=resume,
        wandb_project=wandb_project,
        online_batch_fraction=online_batch_fraction,
    )


@app.local_entrypoint()
def address(key: str) -> None:
    """Print one active learner endpoint."""
    entry = rendezvous.get(key)
    if not entry:
        raise SystemExit(f"No active learner at {key!r} in {RENDEZVOUS_DICT!r}")
    print(f"{key}: {entry['host']}:{entry['port']} tls={entry['use_tls']}")
