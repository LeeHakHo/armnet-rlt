"""Armnet ``@main`` entrypoint for the single-arm RL-token actor."""

from __future__ import annotations

import inspect
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from armnet_runtime import Context, main, require_so101_embodiment
except ModuleNotFoundError:  # unit-test/learner environments omit the actor runtime
    Context = Any

    def main(function):
        return function

    def require_so101_embodiment(*_args, **_kwargs):
        raise RuntimeError("armnet-runtime is required to run the actor job")


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _checkpoint_root(path: str | Path) -> str:
    """Resolve flat exports and repositories containing ``step_<N>`` exports."""

    root = Path(path)

    def looks_like_checkpoint(candidate: Path) -> bool:
        return (candidate / "params").exists() or (candidate / "assets").exists()

    if looks_like_checkpoint(root):
        return str(root)
    subdirs = [item for item in root.iterdir() if item.is_dir()] if root.exists() else []
    candidates = [item for item in subdirs if looks_like_checkpoint(item)]
    step_dirs = sorted(
        (item for item in candidates if item.name.startswith("step_")),
        key=lambda item: (
            int(item.name.removeprefix("step_"))
            if item.name.removeprefix("step_").isdigit()
            else -1
        ),
    )
    if step_dirs:
        return str(step_dirs[-1])
    if len(candidates) == 1:
        return str(candidates[0])
    raise ValueError(
        f"Could not locate an OpenPI checkpoint under {str(path)!r}; "
        f"subdirectories: {sorted(item.name for item in subdirs)}"
    )


def _resolve_checkpoint(ctx: Context) -> str:
    args = ctx.args
    explicit = args.get("checkpoint_dir") or os.environ.get("RLT_CHECKPOINT_DIR")
    if explicit:
        value = str(explicit)
        if value.startswith("volume://"):
            value = str(ctx.volume.path(value.removeprefix("volume://")))
        elif not Path(value).is_absolute() and getattr(ctx.volume, "root", None):
            value = str(ctx.volume.path(value))
        return _checkpoint_root(value)

    repo = args.get("hf_checkpoint_repo") or os.environ.get(
        "RLT_HF_CHECKPOINT_REPO"
    )
    if not repo:
        raise ValueError(
            "provide args['checkpoint_dir'] or args['hf_checkpoint_repo']"
        )
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id=str(repo),
        revision=args.get("hf_checkpoint_revision") or None,
        token=ctx.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN"),
        cache_dir=str(ctx.cache_home) if ctx.cache_home else None,
    )
    return _checkpoint_root(local)


def _resolve_volume_file(ctx: Context, value: str, *, label: str) -> Path:
    path = value
    if path.startswith("volume://"):
        path = str(ctx.volume.path(path.removeprefix("volume://")))
    elif not Path(path).is_absolute() and getattr(ctx.volume, "root", None):
        path = str(ctx.volume.path(path))
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _calibration_lookup(ctx: Context) -> tuple[str, str]:
    """Stage a cell calibration file in LeRobot's ``<id>.json`` layout."""

    calibration_file = ctx.cell.calibration_file_path
    if calibration_file:
        source = Path(calibration_file)
        robot_id = ctx.cell.robot_id or source.stem.removesuffix("_calib")
        target = Path(tempfile.mkdtemp(prefix="armnet-rlt-calibration-"))
        shutil.copy2(source, target / f"{robot_id}.json")
        return robot_id, str(target)
    return (
        ctx.cell.robot_id or "follower_so101",
        str(ctx.cell.calibration_dir) if ctx.cell.calibration_dir else "",
    )


def _tls_root_cert_file(args: dict[str, Any]) -> tuple[str, str | None]:
    """Return a cert path and an optional temporary path to remove."""

    value = args.get("tls_root_cert_pem") or args.get("tls_root_cert") or ""
    value = str(value)
    if not value:
        return "", None
    if "-----BEGIN CERTIFICATE-----" not in value:
        return value, None
    handle = tempfile.NamedTemporaryFile(
        mode="w", prefix="armnet-rlt-ca-", suffix=".pem", delete=False
    )
    try:
        handle.write(value)
        handle.flush()
    finally:
        handle.close()
    return handle.name, handle.name


def _construct(cls: type, **values: Any) -> Any:
    """Construct evolving migration dataclasses from their supported fields."""

    parameters = inspect.signature(cls).parameters
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return cls(**values)
    return cls(**{name: value for name, value in values.items() if name in parameters})


class _ActorConfigView:
    """Add deployment-only fields without expanding the learner's RLTConfig."""

    def __init__(self, rlt_config: Any, **runtime_values: Any) -> None:
        self.rlt_config = rlt_config
        self.__dict__.update(runtime_values)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.rlt_config, name)


def _check_required_cell(actual: str, required: str) -> str:
    """Raise unless ``actual`` is one of the comma-separated ``required`` cells.

    An empty ``required`` accepts anything. Returns ``actual``.
    """
    required = required.strip()
    if not required:
        return actual
    allowed = {name.strip() for name in required.split(",") if name.strip()}
    if not actual:
        raise ValueError(
            f"require_cell={sorted(allowed)} was requested but the runtime "
            "reports no cell_id, so the assignment cannot be verified"
        )
    if actual not in allowed:
        raise ValueError(
            f"this job was scheduled onto {actual!r} but require_cell asks for "
            f"{sorted(allowed)}. Resubmit until it lands on one of those, or drop "
            "--armnet.require-cell to accept whichever cell the scheduler picks."
        )
    return actual


@main
def run(ctx: Context) -> dict[str, Any]:
    # Validate before downloads, OpenPI imports, or robot construction.
    if require_so101_embodiment(ctx, "Armnet RLT actor"):
        raise NotImplementedError(
            "the production RLT actor supports only single-arm 'lerobot/so-101'; "
            "historical bimanual support exists on the learner only"
        )

    from armnet_rlt.config import (
        SO101_JERK_JOINT_WEIGHTS,
        RLTConfig,
        so101_network_config,
    )
    from armnet_rlt.frozen import (
        config_from_frozen_checkpoint,
        load_frozen_checkpoint,
    )
    from armnet_rlt.operator_control import (
        CellOperatorControl,
        VariationCurriculumConfig,
    )
    from armnet_rlt.so101_actor import run_actor

    args = ctx.args
    cell_id = _check_required_cell(
        str(getattr(ctx.cell, "cell_id", "") or ""),
        str(args.get("require_cell", "") or ""),
    )
    ctx.report_progress(f"running on cell {cell_id or '<unknown>'}")
    if not ctx.cell.robot_port:
        raise ValueError("ctx.cell.robot_port is empty; no robot connector is available")
    cameras = dict(ctx.camera_configs)
    config_name = str(args.get("config_name", "pi05_rlt_so101"))
    required_cameras = {"front"}
    if config_name in {
        "pi05_rlt_busybox_push_green_button",
        "pi05_rlt_busybox_multitask_singlearm_minmax",
    }:
        required_cameras.update(("top", "wrist"))
    missing_cameras = required_cameras.difference(cameras)
    if missing_cameras:
        raise ValueError(
            f"cell cameras {sorted(cameras)} are missing cameras "
            f"{sorted(missing_cameras)} required by {config_name!r}"
        )

    frozen_path = args.get("frozen_rlt_checkpoint")
    frozen_artifact = None
    if frozen_path:
        frozen_artifact = load_frozen_checkpoint(
            _resolve_volume_file(
                ctx,
                str(frozen_path),
                label="frozen RLT checkpoint",
            )
        )
        base_cfg = config_from_frozen_checkpoint(frozen_artifact)
        network = base_cfg.network
    else:
        reference_len = int(args.get("reference_action_len", 30))
        predicted_len = int(
            args.get(
                "predicted_action_len",
                args.get("actions_to_execute", 10),
            )
        )
        network = so101_network_config(
            reference_action_len=reference_len,
            predicted_action_len=predicted_len,
        )
        base_cfg = _construct(
            RLTConfig,
            embodiment="so101",
            network=network,
            jerk_joint_weights=list(SO101_JERK_JOINT_WEIGHTS),
            device=str(args.get("device", "cuda")),
            actions_to_execute=predicted_len,
            use_quantile_norm=_as_bool(args.get("quantile_norm"), True),
            policy_uses_delta_actions=_as_bool(
                args.get("delta_actions"), True
            ),
        )
    if (
        network.action_dim != 6
        or network.proprioception_dim != 6
        or network.rl_token_dim != 2048
    ):
        raise ValueError(
            "single-arm actor config must use action_dim=6, "
            "proprioception_dim=6, rl_token_dim=2048"
        )

    checkpoint = _resolve_checkpoint(ctx)
    robot_id, calibration_dir = _calibration_lookup(ctx)
    temporary_calibration = (
        calibration_dir if ctx.cell.calibration_file_path else None
    )
    instruction = str(
        args.get("language_instruction")
        or getattr(ctx.cell, "language_instruction", "")
        or ctx.task
    )
    base_cfg.device = str(args.get("device", "cuda"))
    # Robot connection values are deployment resources, not learner config.
    so101 = SimpleNamespace(
        robot_port=ctx.cell.robot_port,
        robot_id=robot_id,
        calibration_dir=calibration_dir,
        cameras=cameras,
        max_relative_target=ctx.cell.safety_limit,
    )
    cfg = _ActorConfigView(
        base_cfg,
        checkpoint_dir=checkpoint,
        config_name=config_name,
        language_instruction=instruction,
        episode_timeout_s=float(args.get("episode_timeout_s", 30.0)),
        policy_fps=int(args.get("policy_fps", 20)),
        command_fps=int(args.get("command_fps", 20)),
        exploration_correlation=float(
            args.get("exploration_correlation", 0.85)
        ),
        exploration_scale=float(args.get("exploration_scale", 1.0)),
        frozen_eval=frozen_artifact is not None,
        eval_include_base=_as_bool(args.get("eval_include_base"), False),
        reward_target_button=str(
            args.get("reward_target_button")
            or ("green_button" if ctx.task == "push_green_button" else "")
        ),
        so101=so101,
    )

    learner = cfg.actor_learner
    if frozen_artifact is None:
        learner.learner_host = str(
            args.get("learner_host")
            or os.environ.get("RLT_LEARNER_HOST", "")
        )
        learner.learner_port = int(
            args.get("learner_port")
            or os.environ.get("RLT_LEARNER_PORT", 443)
        )
        learner.auth_token = str(
            ctx.secrets.get("RLT_LEARNER_AUTH_TOKEN")
            or os.environ.get("RLT_LEARNER_AUTH_TOKEN", "")
        )
        learner.use_tls = _as_bool(
            args.get(
                "use_tls",
                os.environ.get("RLT_LEARNER_USE_TLS", "1"),
            ),
            True,
        )
        learner.tls_server_name = str(
            args.get("tls_server_name", "") or ""
        )
        if not learner.learner_host:
            raise ValueError("learner_host is required")
        if not learner.auth_token:
            raise ValueError("RLT_LEARNER_AUTH_TOKEN is required")

    cert_path, temporary_cert = _tls_root_cert_file(args)
    learner.tls_root_cert_path = cert_path
    requested_rollouts = int(args.get("num_rollouts", 0))
    include_base = bool(
        frozen_artifact is not None
        and _as_bool(args.get("eval_include_base"), False)
    )
    total_rollouts = (
        requested_rollouts * (2 if include_base else 1)
    ) or None
    operator = CellOperatorControl(
        ctx.cell,
        total_rollouts=total_rollouts,
        ctx=ctx,
        variation_episode_offset=int(
            args.get("variation_episode_offset", 0)
        ),
        variation_repeat=2 if include_base else 1,
        variation_curriculum=VariationCurriculumConfig(
            enabled=_as_bool(args.get("variation_curriculum"), False),
            start_scale=float(args.get("variation_scale_start", 0.25)),
            min_scale=float(args.get("variation_scale_min", 0.1)),
            max_scale=float(args.get("variation_scale_max", 1.0)),
            step_up=float(args.get("variation_scale_step_up", 0.05)),
            step_down=float(
                args.get("variation_scale_step_down", 0.1)
            ),
            window=int(args.get("variation_window", 20)),
            promote_threshold=float(
                args.get("variation_promote_threshold", 0.8)
            ),
            demote_threshold=float(
                args.get("variation_demote_threshold", 0.55)
            ),
            frontier_fraction=float(
                args.get("variation_frontier_fraction", 0.2)
            ),
        ),
    )
    transport = None
    if frozen_artifact is not None:
        from armnet_rlt.so101_actor import FrozenActorTransport

        transport = FrozenActorTransport(frozen_artifact)
    if frozen_artifact is not None:
        ctx.report_progress(
            "starting frozen single-arm RLT evaluation: "
            f"cameras={sorted(cameras)}, "
            f"learner_step={frozen_artifact['learner_step']}, "
            f"paired_base={include_base}"
        )
    else:
        ctx.report_progress(
            f"starting single-arm RLT actor: cameras={sorted(cameras)}, "
            f"learner={learner.learner_host}:{learner.learner_port}"
        )
    try:
        runtime_kwargs = {"operator": operator}
        if transport is not None:
            runtime_kwargs["transport"] = transport
        return run_actor(ctx, cfg, **runtime_kwargs)
    finally:
        if temporary_cert:
            Path(temporary_cert).unlink(missing_ok=True)
        if temporary_calibration:
            shutil.rmtree(temporary_calibration, ignore_errors=True)

