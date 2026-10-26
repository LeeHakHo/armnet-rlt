from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path


SO101_JERK_JOINT_WEIGHTS: tuple[float, ...] = (1.0, 1.0, 1.0, 0.2, 0.1, 0.1)
BISO101_JERK_JOINT_WEIGHTS: tuple[float, ...] = (
    SO101_JERK_JOINT_WEIGHTS + SO101_JERK_JOINT_WEIGHTS
)


def parse_round_schedule(spec: str | int) -> tuple[int, ...]:
    """Parse ``"30,30,10"`` into the episode count for each training round.

    The last entry repeats for later rounds. An empty spec or ``0`` returns an
    empty tuple.
    """
    if isinstance(spec, int):
        values = [spec]
    else:
        text = str(spec).strip()
        if not text:
            return ()
        values = [int(part) for part in text.split(",") if part.strip()]
    if not values or values == [0]:
        return ()
    if any(value <= 0 for value in values):
        raise ValueError(f"round schedule must be positive: {spec!r}")
    return tuple(values)


def episodes_for_round(schedule: tuple[int, ...], round_index: int) -> int:
    """Episodes required by round ``round_index`` (1-based); last entry repeats."""
    if not schedule:
        raise ValueError("round schedule is empty")
    if round_index < 1:
        raise ValueError("round_index must be at least 1")
    return schedule[min(round_index - 1, len(schedule) - 1)]


# The official pi0 DSRL noise actor and critics (nakamotoo/dsrl_pi0): 1024x3 MLPs.
DSRL_HIDDEN_DIMS: tuple[int, ...] = (1024, 1024, 1024)


@dataclass
class RLTNetworkConfig:
    hidden_dims: tuple[int, ...] = (512, 512, 512, 512)
    init_final: float | None = 0.05
    action_dim: int = 6
    reference_action_len: int = 30
    predicted_action_len: int = 10
    rl_token_dim: int = 2048
    proprioception_dim: int = 6
    fixed_action_std: float = 0.05
    ref_action_dropout: float = 0.25
    delta_action_mask: tuple[bool, ...] = (True, True, True, True, True, False)

    def __post_init__(self) -> None:
        if not self.hidden_dims or any(size <= 0 for size in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive layer sizes")
        for name in (
            "action_dim",
            "reference_action_len",
            "predicted_action_len",
            "rl_token_dim",
            "proprioception_dim",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.rl_token_dim != 2048:
            raise ValueError("rl_token_dim must be 2048 for RLT transition caches")
        if self.proprioception_dim != self.action_dim:
            raise ValueError("proprioception_dim must equal action_dim")
        if self.reference_action_len < self.predicted_action_len:
            raise ValueError(
                "reference_action_len must be at least predicted_action_len"
            )
        if len(self.delta_action_mask) != self.action_dim:
            raise ValueError(
                "delta_action_mask length must equal action_dim "
                f"({len(self.delta_action_mask)} != {self.action_dim})"
            )
        if not math.isfinite(self.fixed_action_std) or self.fixed_action_std < 0:
            raise ValueError("fixed_action_std must be finite and nonnegative")
        if not 0.0 <= self.ref_action_dropout <= 1.0:
            raise ValueError("ref_action_dropout must be in [0, 1]")

    @property
    def predicted_action_dim(self) -> int:
        return self.predicted_action_len * self.action_dim

    @property
    def reference_action_dim(self) -> int:
        return self.reference_action_len * self.action_dim


def so101_network_config(**overrides: object) -> RLTNetworkConfig:
    values: dict[str, object] = {
        "action_dim": 6,
        "proprioception_dim": 6,
        "delta_action_mask": (True, True, True, True, True, False),
    }
    values.update(overrides)
    return RLTNetworkConfig(**values)


def biso101_network_config(**overrides: object) -> RLTNetworkConfig:
    arm_mask = (True, True, True, True, True, False)
    values: dict[str, object] = {
        "action_dim": 12,
        "proprioception_dim": 12,
        "delta_action_mask": arm_mask + arm_mask,
    }
    values.update(overrides)
    return RLTNetworkConfig(**values)


@dataclass
class RLTActorLearnerConfig:
    learner_host: str = "127.0.0.1"
    learner_port: int = 50051
    server_bind_host: str = "0.0.0.0"
    auth_token: str = ""
    use_tls: bool = False
    tls_root_cert_path: str = ""
    tls_server_name: str = ""
    tls_server_cert_path: str = ""
    tls_server_key_path: str = ""
    policy_parameters_push_frequency: float = 0.2
    queue_get_timeout: float = 2.0

    def __post_init__(self) -> None:
        if not 1 <= self.learner_port <= 65535:
            raise ValueError("learner_port must be in [1, 65535]")
        if not self.learner_host:
            raise ValueError("learner_host must not be empty")
        if not self.server_bind_host:
            raise ValueError("server_bind_host must not be empty")
        if (self.tls_server_cert_path == "") != (self.tls_server_key_path == ""):
            raise ValueError(
                "tls_server_cert_path and tls_server_key_path must be set together"
            )
        if self.policy_parameters_push_frequency <= 0:
            raise ValueError("policy_parameters_push_frequency must be positive")
        if self.queue_get_timeout <= 0:
            raise ValueError("queue_get_timeout must be positive")

    @property
    def auth_metadata(self) -> list[tuple[str, str]]:
        if not self.auth_token:
            return []
        return [("authorization", f"Bearer {self.auth_token}")]


@dataclass
class RLTConfig:
    seed: int = 42
    embodiment: str = "so101"
    network: RLTNetworkConfig = field(default_factory=so101_network_config)

    assets_dir: str | Path = ""
    demo_cache_path: str | Path = ""
    output_dir: str | Path = "outputs"
    run_id: str = "default"

    discount: float = 0.985
    num_critics: int = 4
    critic_target_update_weight: float = 0.005
    utd_ratio: int = 5
    policy_update_freq: int = 4
    critic_lr: float = 1e-4
    actor_lr: float = 1e-4
    critic_lr_min: float = 5e-5
    actor_lr_min: float = 2.5e-5
    grad_clip_norm: float = 5.0
    max_q: float | None = 2.0
    bc_beta: float = 0.5
    bc_gripper_weight: float = 0.33
    jerk_lambda: float = 0.2
    jerk_joint_weights: tuple[float, ...] = SO101_JERK_JOINT_WEIGHTS
    wrong_button_penalty_min: float = 0.05
    wrong_button_penalty_max: float = 0.5
    target_policy_noise: float = 0.05
    target_noise_clip: float = 0.1

    online_buffer_capacity: int = 15_000
    online_step_before_learning: int = 500
    max_demo_pretrain_steps: int = 500
    online_steps: int = 100_000
    batch_size: int = 256
    online_batch_fraction: float = 0.5
    q_loss_normalization: bool = False
    # Hinge cap in normalized action space; 0 disables it.
    bc_deviation_max: float = 0.0
    bc_deviation_weight: float = 1.0
    q_loss_alpha: float = 2.5
    actions_to_execute: int = 10
    use_subsampling: bool = True
    sub_chunk_stride: int = 2

    use_quantile_norm: bool = True
    policy_uses_delta_actions: bool = False
    # "rlt": the actor outputs an action chunk refined from the base policy's.
    # "dsrl": the actor outputs the base flow policy's initial noise (Wagenmaker
    # et al., 2025); the base policy turns it into the executed chunk.
    algorithm: str = "rlt"
    # DSRL noise actions live in [-dsrl_noise_bound, dsrl_noise_bound]; the
    # actor outputs dsrl_noise_steps rows of noise, repeated over the chunk.
    dsrl_noise_bound: float = 2.5
    dsrl_noise_steps: int = 1
    # DSRL soft actor-critic: the entropy the temperature steers the actor
    # toward (summed over noise dimensions), the initial temperature, the
    # temperature's Adam learning rate, and the gradient steps taken after the
    # first episode.
    dsrl_target_entropy: float = 0.0
    dsrl_alpha_init: float = 1.0
    dsrl_temperature_lr: float = 3e-4
    dsrl_initial_steps: int = 5000
    device: str = "cuda"
    storage_device: str = "cpu"
    log_freq: int = 10
    save_freq: int = 500
    rolling_window: int = 10
    actor_learner: RLTActorLearnerConfig = field(default_factory=RLTActorLearnerConfig)

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.embodiment not in {"so101", "biso101"}:
            raise ValueError("embodiment must be 'so101' or historical 'biso101'")
        expected_dim = 6 if self.embodiment == "so101" else 12
        if self.network.action_dim != expected_dim:
            raise ValueError(
                f"{self.embodiment} requires action_dim={expected_dim}, "
                f"got {self.network.action_dim}"
            )
        if self.algorithm not in {"rlt", "dsrl"}:
            raise ValueError("algorithm must be 'rlt' or 'dsrl'")
        if self.algorithm == "dsrl":
            # The noise covers the base policy's whole chunk; the actor executes
            # a prefix of the chunk the base policy produces from it.
            if self.network.predicted_action_len != self.network.reference_action_len:
                raise ValueError(
                    "dsrl requires network.predicted_action_len to equal "
                    "network.reference_action_len"
                )
            if not 1 <= self.actions_to_execute <= self.network.predicted_action_len:
                raise ValueError(
                    "dsrl requires 1 <= actions_to_execute <= "
                    "network.predicted_action_len"
                )
            if not math.isfinite(self.dsrl_noise_bound) or self.dsrl_noise_bound <= 0:
                raise ValueError("dsrl_noise_bound must be finite and positive")
            if not math.isfinite(self.dsrl_target_entropy):
                raise ValueError("dsrl_target_entropy must be finite")
            if not math.isfinite(self.dsrl_alpha_init) or self.dsrl_alpha_init <= 0:
                raise ValueError("dsrl_alpha_init must be finite and positive")
            if not 1 <= self.dsrl_noise_steps <= self.network.predicted_action_len:
                raise ValueError(
                    "dsrl_noise_steps must be within 1..network.predicted_action_len"
                )
            if (
                not math.isfinite(self.dsrl_temperature_lr)
                or self.dsrl_temperature_lr <= 0
            ):
                raise ValueError("dsrl_temperature_lr must be finite and positive")
            if self.dsrl_initial_steps < 0:
                raise ValueError("dsrl_initial_steps must be nonnegative")
            if self.max_demo_pretrain_steps:
                raise ValueError("dsrl has no demonstration pretraining")
            if self.online_batch_fraction != 1.0:
                raise ValueError("dsrl trains on online transitions only")
        elif self.actions_to_execute != self.network.predicted_action_len:
            raise ValueError(
                "actions_to_execute must equal network.predicted_action_len"
            )
        if len(self.jerk_joint_weights) != self.network.action_dim:
            raise ValueError(
                "jerk_joint_weights length must equal network.action_dim"
            )
        if not math.isfinite(self.bc_deviation_max) or self.bc_deviation_max < 0:
            raise ValueError("bc_deviation_max must be finite and nonnegative")
        if (
            not math.isfinite(self.bc_deviation_weight)
            or self.bc_deviation_weight < 0
        ):
            raise ValueError("bc_deviation_weight must be finite and nonnegative")
        if any(
            not math.isfinite(weight) or weight < 0
            for weight in self.jerk_joint_weights
        ):
            raise ValueError("jerk_joint_weights must be finite and nonnegative")
        if self.use_subsampling:
            stride = self.sub_chunk_stride
            chunk = self.network.predicted_action_len
            if not 1 <= stride <= chunk:
                raise ValueError("sub_chunk_stride must be within the action chunk")
            if chunk % stride:
                raise ValueError(
                    "predicted_action_len must be divisible by sub_chunk_stride"
                )
        for name in (
            "num_critics",
            "utd_ratio",
            "policy_update_freq",
            "online_buffer_capacity",
            "online_step_before_learning",
            "batch_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("max_demo_pretrain_steps", "online_steps", "log_freq", "save_freq"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0.0 < self.online_batch_fraction <= 1.0:
            raise ValueError("online_batch_fraction must be in (0, 1]")
        if not math.isfinite(self.q_loss_alpha) or self.q_loss_alpha <= 0:
            raise ValueError("q_loss_alpha must be finite and positive")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("discount must be in (0, 1]")
        if not 0.0 < self.critic_target_update_weight <= 1.0:
            raise ValueError("critic_target_update_weight must be in (0, 1]")
        for name in (
            "critic_lr",
            "actor_lr",
            "critic_lr_min",
            "actor_lr_min",
            "grad_clip_norm",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.critic_lr_min > self.critic_lr:
            raise ValueError("critic_lr_min must not exceed critic_lr")
        if self.actor_lr_min > self.actor_lr:
            raise ValueError("actor_lr_min must not exceed actor_lr")
        for name in ("target_policy_noise", "target_noise_clip"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.max_q is not None and (
            not math.isfinite(self.max_q) or self.max_q <= 0
        ):
            raise ValueError("max_q must be finite and positive when set")
        if not (
            0.0
            <= self.wrong_button_penalty_min
            <= self.wrong_button_penalty_max
            <= 1.0
        ):
            raise ValueError(
                "wrong-button penalties must satisfy 0 <= min <= max <= 1"
            )
        if not self.run_id or self.run_id in {".", ".."}:
            raise ValueError("run_id must be a nonempty directory name")
        if Path(self.run_id).name != self.run_id:
            raise ValueError("run_id must not contain path separators")

    @classmethod
    def for_embodiment(cls, embodiment: str, **overrides: object) -> RLTConfig:
        if embodiment == "so101":
            defaults: dict[str, object] = {
                "network": so101_network_config(),
                "jerk_joint_weights": SO101_JERK_JOINT_WEIGHTS,
            }
        elif embodiment == "biso101":
            defaults = {
                "network": biso101_network_config(),
                "jerk_joint_weights": BISO101_JERK_JOINT_WEIGHTS,
            }
        else:
            raise ValueError("embodiment must be 'so101' or historical 'biso101'")
        defaults.update(overrides)
        return cls(embodiment=embodiment, **defaults)

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.output_dir) / self.run_id
