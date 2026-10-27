"""Single-arm SO-101 actor that rolls out a LeRobot policy (ACT, DiT, SmolVLA, ...)."""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from armnet_rlt.metrics import EpisodeRecord
from armnet_rlt.openpi_rlt import MOTOR_NAMES
from armnet_rlt.operator_control import CellOperatorControl
from armnet_rlt.so101_actor import (
    LearnerTransport,
    _await_learner_round,
    _button_press_counts,
    _close_robot_and_cell,
    _correlated_noise,
    _field,
    _load_parameters,
    _serialize_interaction,
    _shape_button_reward,
    finish_rollout_recording,
    make_so101_robot,
    open_rollout_recording,
    record_rollout_frame,
    robot_action,
)
from armnet_rlt.state_machine import (
    EpisodeEvent,
    EpisodeOutcome,
    EpisodeState,
    RLTStateMachine,
)

log = logging.getLogger(__name__)

ACT_CONFIG_NAME = "act"
IMAGE_PREFIX = "observation.images."
EPISODE_SCHEMA = "act_episode_v1"
JPEG_QUALITY = 90


def resolve_act_policy_dir(path: str | Path) -> Path:
    """Return the directory holding the policy's ``config.json``."""
    root = Path(path)
    for candidate in (root, root / "pretrained_model"):
        if (candidate / "config.json").is_file():
            return candidate
    raise FileNotFoundError(f"no policy config.json under {root}")


def load_lerobot_policy(path: str | Path, device: str) -> tuple[Any, Any, Any]:
    """Load a LeRobot policy of the type in its config with its pre- and post-processors."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    from armnet_rlt import act_gmm  # registers the act_gmm type

    policy_dir = resolve_act_policy_dir(path)
    policy_type = PreTrainedConfig.from_pretrained(str(policy_dir)).type
    policy_class = (
        act_gmm.ACTGMMPolicy if policy_type == act_gmm.POLICY_TYPE else get_policy_class(policy_type)
    )
    policy = policy_class.from_pretrained(str(policy_dir))
    policy.config.device = device
    policy.to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(policy_dir),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor


def policy_rename_map(path: str | Path) -> dict[str, str]:
    """The dataset-to-policy key renames from the checkpoint's ``train_config.json``."""
    train_config = resolve_act_policy_dir(path) / "train_config.json"
    if not train_config.is_file():
        return {}
    return dict(json.loads(train_config.read_text()).get("rename_map") or {})


def image_shapes(
    policy: Any,
    *,
    rename_map: dict[str, str] | None = None,
    size: tuple[int, int] | None = None,
) -> dict[str, tuple[int, int]]:
    """Map each image key the preprocessor takes to the (height, width) fed to it.

    Keys are the dataset keys before ``rename_map``; ``size`` replaces the
    shapes in the policy config.
    """
    dataset_keys = {new: old for old, new in (rename_map or {}).items()}
    return {
        dataset_keys.get(key, key): size
        or (int(feature.shape[1]), int(feature.shape[2]))
        for key, feature in policy.config.image_features.items()
    }


def parse_image_size(value: str | None) -> tuple[int, int] | None:
    """Parse ``<height>x<width>``."""
    if not value:
        return None
    height, _, width = str(value).partition("x")
    if not height.isdigit() or not width.isdigit():
        raise ValueError(f"image size must be <height>x<width>, got {value!r}")
    return int(height), int(width)


def resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import cv2

    height, width = size
    if image.shape[:2] == (height, width):
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def encode_jpeg(image: np.ndarray) -> bytes:
    import cv2

    ok, buffer = cv2.imencode(
        ".jpg",
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
    )
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buffer.tobytes()


def decode_jpeg(data: bytes) -> np.ndarray:
    import cv2

    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("JPEG decoding failed")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def act_observation(
    raw_obs: dict[str, Any], shapes: dict[str, tuple[int, int]]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Joint state and the policy's camera images resized to training size."""
    state = np.asarray(
        [raw_obs[f"{name}.pos"] for name in MOTOR_NAMES], dtype=np.float32
    )
    images = {}
    for key, size in shapes.items():
        camera = key.removeprefix(IMAGE_PREFIX)
        if camera not in raw_obs:
            raise KeyError(f"robot observation has no camera {camera!r} for {key}")
        images[key] = resize_image(np.asarray(raw_obs[camera]), size)
    return state, images


def act_frame(
    state: np.ndarray, images: dict[str, np.ndarray], task: str | None = None
) -> dict[str, Any]:
    """One unbatched policy input frame; images become CHW floats in [0, 1]."""
    frame: dict[str, Any] = {"observation.state": torch.from_numpy(state)}
    for key, image in images.items():
        frame[key] = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    if task is not None:
        frame["task"] = task
    return frame


def serialize_act_episode(
    states: list[np.ndarray],
    actions: list[np.ndarray],
    images: dict[str, list[bytes]],
) -> bytes:
    from lerobot.transport.utils import python_object_to_bytes

    return python_object_to_bytes(
        {
            "schema": EPISODE_SCHEMA,
            "state": np.stack(states).astype(np.float32),
            "action": np.stack(actions).astype(np.float32),
            "images": images,
        }
    )


def _predict_chunk(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    frame: dict[str, torch.Tensor],
    *,
    n_action_steps: int,
    exploration_scale: float,
    exploration_correlation: float,
) -> np.ndarray:
    """Predict ``n_action_steps`` actions, with correlated noise in normalized space."""
    with torch.inference_mode():
        normalized = policy.predict_action_chunk(preprocessor(frame))
        normalized = normalized[:, :n_action_steps]
        if exploration_scale > 0:
            noise = _correlated_noise(
                normalized.shape[1],
                normalized.shape[2],
                exploration_correlation,
                device=normalized.device,
                dtype=normalized.dtype,
            )
            normalized = normalized + noise * exploration_scale
        actions = postprocessor(normalized)
    return actions.reshape(-1, actions.shape[-1]).cpu().numpy()


def _select_action(
    policy: Any, preprocessor: Any, postprocessor: Any, frame: dict[str, Any]
) -> np.ndarray:
    """One action from the policy's own action queue."""
    with torch.inference_mode():
        action = postprocessor(policy.select_action(preprocessor(frame)))
    return action.reshape(1, -1).cpu().numpy()


def run_act_actor(
    ctx: Any,
    cfg: Any,
    *,
    operator: CellOperatorControl | None = None,
    robot: Any = None,
    transport: LearnerTransport | None = None,
) -> dict[str, Any]:
    """Roll out a LeRobot policy on one Armnet SO-101 cell and report to the learner.

    ACT predicts a chunk of ``n_action_steps`` with optional exploration noise;
    other policy types use ``select_action``. Successful episodes are streamed
    to the learner unless ``stream_episodes`` is false.
    """
    device_name = str(_field(cfg, "device", "cuda"))
    device = (
        device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu"
    )
    seed = int(_field(cfg, "seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    robot = robot or make_so101_robot(ctx, cfg)
    robot = ctx.cell.instrument(robot)
    if not getattr(robot, "is_connected", False):
        try:
            robot.connect(calibrate=False)
        except TypeError:
            robot.connect()

    try:
        policy, preprocessor, postprocessor = load_lerobot_policy(
            cfg.policy_dir, device
        )
        policy.eval()
        shapes = image_shapes(
            policy,
            rename_map=policy_rename_map(cfg.policy_dir),
            size=parse_image_size(_field(cfg, "image_size", None)),
        )
    except Exception:
        _close_robot_and_cell(robot, ctx.cell)
        raise
    policy_type = str(policy.config.type)
    is_act = policy_type == ACT_CONFIG_NAME
    ensemble_coeff = _field(cfg, "temporal_ensemble_coeff", None)
    if ensemble_coeff is not None:
        if not is_act:
            raise ValueError(f"temporal_ensemble_coeff needs an ACT policy, not {policy_type}")
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

        policy.config.temporal_ensemble_coeff = float(ensemble_coeff)
        policy.config.n_action_steps = 1
        policy.temporal_ensembler = ACTTemporalEnsembler(
            float(ensemble_coeff), policy.config.chunk_size
        )
    # ACT without temporal ensembling executes whole chunks; everything else uses select_action.
    chunked = is_act and policy.config.temporal_ensemble_coeff is None
    n_action_steps = int(_field(cfg, "n_action_steps", 0)) or int(
        policy.config.n_action_steps
    )
    if is_act and not 1 <= n_action_steps <= int(policy.config.chunk_size):
        raise ValueError(
            f"n_action_steps={n_action_steps} must be in "
            f"[1, chunk_size={policy.config.chunk_size}]"
        )
    stream_episodes = bool(_field(cfg, "stream_episodes", True))

    transport = transport or LearnerTransport(cfg.actor_learner)
    operator = operator or CellOperatorControl(ctx.cell)
    state_machine = RLTStateMachine()
    fps = max(1, int(_field(cfg, "command_fps", 20)))
    period_s = 1.0 / fps
    timeout_s = float(_field(cfg, "episode_timeout_s", 30.0))
    exploration_correlation = float(_field(cfg, "exploration_correlation", 0.85))
    if not 0.0 <= exploration_correlation < 1.0:
        raise ValueError("exploration_correlation must be in [0, 1)")
    exploration_scale = float(_field(cfg, "exploration_scale", 0.0))
    if not math.isfinite(exploration_scale) or exploration_scale < 0:
        raise ValueError("exploration_scale must be finite and nonnegative")
    if exploration_scale > 0 and not chunked:
        raise ValueError(
            f"exploration_scale needs an ACT policy without temporal ensembling, not {policy_type}"
        )
    from armnet_rlt.config import episodes_for_round, parse_round_schedule

    round_schedule = parse_round_schedule(_field(cfg, "episodes_per_round", 0))
    steps_per_round = int(_field(cfg, "steps_per_round", 0))
    if round_schedule and steps_per_round <= 0:
        raise ValueError(
            "steps_per_round is required when episodes_per_round is set; "
            "it must match the learner's value"
        )
    round_wait_timeout_s = float(_field(cfg, "round_wait_timeout_s", 600.0))
    reward_target_button = str(_field(cfg, "reward_target_button", "") or "")
    instruction = str(
        _field(cfg, "language_instruction", "")
        or getattr(ctx.cell, "language_instruction", "")
        or getattr(ctx, "task", "")
    )
    task = None if is_act else instruction
    eval_dataset, frame_writer, telemetry, eval_repo_id = open_rollout_recording(
        ctx, cfg, robot, instruction, action_source=f"{policy_type}_policy"
    )
    eval_dataset_url: str | None = None

    completed = successes = 0
    session_id = uuid.uuid4().hex
    cell_id = str(getattr(getattr(ctx, "cell", None), "cell_id", "") or "")
    per_rollout: list[dict[str, Any]] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    frames: dict[str, list[bytes]] = {key: [] for key in shapes}
    chunk: np.ndarray | None = None
    chunk_index = 0
    current_policy_step: int | None = None
    episode_policy_step_start: int | None = None
    policy_reload_count = 0
    episode_started_at = ""
    command_ticks = 0
    hardware_clamp_steps = 0

    try:
        transport.start()
        current_policy_step = _load_parameters(
            policy,
            transport.wait_for_initial_parameters(
                timeout_s=float(_field(cfg, "initial_parameters_timeout_s", 600.0))
            ),
        )
        policy.eval()
        print(
            f"[act] {policy_type} policy {cfg.policy_dir}: images={shapes} "
            f"n_action_steps={n_action_steps} learner_step={current_policy_step}",
            flush=True,
        )
        operator.start()

        # Run one inference before the first rollout.
        warm_state, warm_images = act_observation(robot.get_observation(), shapes)
        if chunked:
            _predict_chunk(
                policy,
                preprocessor,
                postprocessor,
                act_frame(warm_state, warm_images),
                n_action_steps=n_action_steps,
                exploration_scale=0.0,
                exploration_correlation=exploration_correlation,
            )
        else:
            _select_action(
                policy, preprocessor, postprocessor, act_frame(warm_state, warm_images, task)
            )
            policy.reset()

        rounds_done = 0
        next_round_at = (
            episodes_for_round(round_schedule, 1) if round_schedule else 0
        )
        while not operator.should_stop():
            tick = time.perf_counter()
            if (
                round_schedule
                and state_machine.state is EpisodeState.END_EPISODE
                and completed >= next_round_at
            ):
                rounds_done += 1
                next_round_at += episodes_for_round(round_schedule, rounds_done + 1)
                print(
                    f"[act round] round {rounds_done} boundary at {completed} "
                    f"rollouts; waiting for {steps_per_round} learner steps "
                    f"past {current_policy_step}",
                    flush=True,
                )
                current_policy_step, applied = _await_learner_round(
                    transport,
                    policy,
                    current_policy_step,
                    steps_per_round=steps_per_round,
                    timeout_s=round_wait_timeout_s,
                )
                policy.eval()
                policy_reload_count += applied
                release = getattr(operator, "release_round_retries", None)
                released = release() if callable(release) else 0
                print(
                    f"[act round] resuming at learner step {current_policy_step}"
                    + (f"; {released} scene(s) queued for retry" if released else ""),
                    flush=True,
                )
            previous_state = state_machine.state
            for event, _grade in operator.poll_events(state_machine.state):
                state_machine.handle_event(event)
                break
            if (
                previous_state is EpisodeState.END_EPISODE
                and state_machine.state is EpisodeState.IN_EPISODE
            ):
                update = transport.latest_parameters()
                if update is not None:
                    loaded_step = _load_parameters(policy, update)
                    policy.eval()
                    if loaded_step is not None:
                        current_policy_step = loaded_step
                    policy_reload_count += 1
                episode_started_at = datetime.now(UTC).isoformat()
                episode_policy_step_start = current_policy_step
                if telemetry is not None and eval_dataset is not None:
                    telemetry.begin_episode(eval_dataset.num_episodes)
            if (
                state_machine.state is EpisodeState.IN_EPISODE
                and state_machine.episode_duration >= timeout_s
            ):
                state_machine.handle_event(EpisodeEvent.TIMEOUT)

            if (
                previous_state is EpisodeState.IN_EPISODE
                and state_machine.state is EpisodeState.END_EPISODE
            ):
                variation = getattr(operator, "variation", None)
                curriculum = (
                    variation.get("curriculum", {})
                    if isinstance(variation, dict)
                    else {}
                )
                reward, wrong_button_presses, wrong_button_penalty = (
                    _shape_button_reward(
                        state_machine.episode_reward,
                        press_counts=_button_press_counts(robot),
                        target_button=reward_target_button,
                        curriculum_scale=float(curriculum.get("scale", 0.0)),
                        penalty_min=float(_field(cfg, "wrong_button_penalty_min", 0.0)),
                        penalty_max=float(_field(cfg, "wrong_button_penalty_max", 0.0)),
                    )
                )
                outcome = state_machine.episode_ctx.outcome
                success = outcome is EpisodeOutcome.SUCCESS
                # Frames for successful rollouts only.
                payload = (
                    serialize_act_episode(states, actions, frames)
                    if stream_episodes and success and states
                    else None
                )
                record = EpisodeRecord(
                    session_id=session_id,
                    cell_id=cell_id,
                    policy_variant=policy_type,
                    session_rollout_index=int(
                        getattr(operator, "rollout_index", completed + 1)
                    ),
                    rollout_total=getattr(operator, "rollout_total", None),
                    started_at=episode_started_at,
                    ended_at=datetime.now(UTC).isoformat(),
                    success=success,
                    duration_s=state_machine.episode_duration,
                    timeout=outcome is EpisodeOutcome.TIMEOUT,
                    outcome=(outcome.name.lower() if outcome is not None else "unknown"),
                    scored_by=getattr(operator, "scored_by", None),
                    num_chunks=state_machine.episode_ctx.total_chunk_count,
                    num_transitions=len(states) if payload is not None else 0,
                    exploration_scale=exploration_scale,
                    exploration_correlation=exploration_correlation,
                    policy_step_start=episode_policy_step_start,
                    policy_step_end=current_policy_step,
                    policy_reload_count=policy_reload_count,
                    variation=variation,
                    reset_problems=tuple(getattr(operator, "reset_problems", ())),
                    shaped_reward=reward,
                    wrong_button_presses=wrong_button_presses,
                    wrong_button_penalty=wrong_button_penalty,
                    command_ticks=command_ticks,
                    hardware_clamp_steps=hardware_clamp_steps,
                )
                transport.send_episode(payload, _serialize_interaction(record))
                rollout_record = record.to_dict()
                per_rollout.append(rollout_record)
                print(
                    "[act rollout] " + json.dumps(rollout_record, default=str),
                    flush=True,
                )
                record_outcome = getattr(operator, "record_outcome", None)
                if callable(record_outcome):
                    record_outcome(success)
                completed += 1
                successes += int(success)
                operator.on_episode_end(outcome)
                if eval_dataset is not None and frame_writer is not None:
                    from armnet_rlt.dataset_recording import save_episode

                    save_episode(
                        ctx,
                        eval_dataset,
                        frame_writer,
                        telemetry=telemetry,
                        ticks=command_ticks,
                    )
                policy.reset()
                states = []
                actions = []
                frames = {key: [] for key in shapes}
                chunk = None
                chunk_index = 0
                episode_policy_step_start = None
                policy_reload_count = 0
                episode_started_at = ""
                command_ticks = 0
                hardware_clamp_steps = 0
                continue

            if state_machine.state is not EpisodeState.IN_EPISODE:
                time.sleep(min(0.05, period_s))
                continue

            raw_obs = robot.get_observation()
            state, images = act_observation(raw_obs, shapes)
            if not chunked:
                chunk = _select_action(
                    policy, preprocessor, postprocessor, act_frame(state, images, task)
                )
                chunk_index = 0
            elif chunk is None or chunk_index >= len(chunk):
                chunk = _predict_chunk(
                    policy,
                    preprocessor,
                    postprocessor,
                    act_frame(state, images),
                    n_action_steps=n_action_steps,
                    exploration_scale=exploration_scale,
                    exploration_correlation=exploration_correlation,
                )
                chunk_index = 0
                state_machine.episode_ctx.total_chunk_count += 1

            requested_action = robot_action(chunk[chunk_index])
            action_started = time.perf_counter()
            sent_action = robot.send_action(requested_action)
            action_write_s = time.perf_counter() - action_started
            actual_action = (
                sent_action if isinstance(sent_action, dict) else requested_action
            )
            record_rollout_frame(
                ctx,
                cfg,
                frame_writer,
                telemetry,
                raw_obs=raw_obs,
                requested_action=requested_action,
                actual_action=actual_action,
                task=instruction,
                frame_index=command_ticks,
                period_s=period_s,
                tick=tick,
                action_write_s=action_write_s,
            )
            if any(
                not math.isclose(
                    float(actual_action.get(key, requested_action[key])),
                    float(requested_action[key]),
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
                for key in requested_action
            ):
                hardware_clamp_steps += 1
            states.append(state)
            actions.append(
                np.asarray(
                    [
                        actual_action.get(f"{name}.pos", requested_action[f"{name}.pos"])
                        for name in MOTOR_NAMES
                    ],
                    dtype=np.float32,
                )
            )
            if stream_episodes:
                for key, image in images.items():
                    frames[key].append(encode_jpeg(image))
            chunk_index += 1
            command_ticks += 1
            remaining = period_s - (time.perf_counter() - tick)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        try:
            operator.stop()
            eval_dataset_url = finish_rollout_recording(
                ctx, eval_dataset, frame_writer, telemetry, eval_repo_id
            )
        finally:
            transport.close()
            _close_robot_and_cell(robot, ctx.cell)

    return {
        "status": "actor_finished",
        "rollouts": completed,
        "successes": successes,
        "eval_dataset_repo_id": eval_repo_id,
        "eval_dataset_url": eval_dataset_url,
        "robot_telemetry": (
            telemetry.result_summary() if telemetry is not None else None
        ),
        "per_rollout": per_rollout,
        "aggregated": {
            "n_rollouts": completed,
            "n_success": successes,
            "pc_success": 100.0 * successes / completed if completed else 0.0,
        },
    }
