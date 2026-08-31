from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class EpisodeRecord:
    schema_version: int = 1
    session_id: str = ""
    session_rollout_index: int = 0
    rollout_total: int | None = None
    started_at: str = ""
    ended_at: str = ""
    success: bool = False
    duration_s: float = 0.0
    timeout: bool = False
    outcome: str = "fail"
    scored_by: str | None = None
    teleop_override_rate: float = 0.0
    num_chunks: int = 0
    num_transitions: int = 0
    exploration_scale: float = 0.0
    exploration_correlation: float = 0.0
    policy_step_start: int | None = None
    policy_step_end: int | None = None
    policy_reload_count: int = 0
    variation: dict[str, Any] | None = None
    reset_problems: tuple[str, ...] = ()
    command_ticks: int = 0
    hardware_clamp_steps: int = 0
    action_deviation_mean: float = 0.0
    action_deviation_max: float = 0.0
    action_deviation_by_joint: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EpisodeRecord:
        fields = cls.__dataclass_fields__
        kwargs = {key: item for key, item in value.items() if key in fields}
        for key in ("reset_problems", "action_deviation_by_joint"):
            if key in kwargs:
                kwargs[key] = tuple(kwargs[key] or ())
        return cls(**kwargs)


class RollingMetrics:
    def __init__(self, window: int = 10):
        if window <= 0:
            raise ValueError("window must be positive")
        self.window = window
        self._records: deque[EpisodeRecord] = deque(maxlen=window)
        self._success_durations: deque[float] = deque(maxlen=window)

    def add(self, record: EpisodeRecord) -> None:
        self._records.append(record)
        if record.success:
            self._success_durations.append(record.duration_s)

    def to_dict(self) -> dict[str, float]:
        count = len(self._records)
        suffix = str(self.window)
        return {
            f"rolling/success_rate_{suffix}": (
                sum(record.success for record in self._records) / count
                if count
                else 0.0
            ),
            f"rolling/mean_success_duration_s_{suffix}": (
                sum(self._success_durations) / len(self._success_durations)
                if self._success_durations
                else 0.0
            ),
            f"rolling/timeout_rate_{suffix}": (
                sum(record.timeout for record in self._records) / count
                if count
                else 0.0
            ),
            f"rolling/action_deviation_mean_{suffix}": (
                sum(record.action_deviation_mean for record in self._records)
                / count
                if count
                else 0.0
            ),
            f"rolling/teleop_override_rate_{suffix}": (
                sum(record.teleop_override_rate for record in self._records)
                / count
                if count
                else 0.0
            ),
        }


def episode_metrics(
    record: EpisodeRecord, episode_index: int
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "episode/success": int(record.success),
        "episode/duration_seconds": record.duration_s,
        "episode/timeout": int(record.timeout),
        "episode/teleop_override_rate": record.teleop_override_rate,
        "episode/num_chunks": record.num_chunks,
        "episode/num_transitions": record.num_transitions,
        "episode/action_deviation_mean": record.action_deviation_mean,
        "episode/action_deviation_max": record.action_deviation_max,
        "episode/policy_reload_count": record.policy_reload_count,
        "episode/index": episode_index,
    }
    for index, value in enumerate(record.action_deviation_by_joint):
        metrics[f"episode/action_deviation_joint_{index}"] = value
    if record.policy_step_start is not None:
        metrics["episode/policy_step_start"] = record.policy_step_start
    if record.policy_step_end is not None:
        metrics["episode/policy_step_end"] = record.policy_step_end
    return metrics


def learner_metrics(
    *,
    critic_loss: float,
    actor_loss: float,
    mean_q: float,
    updates_per_second: float,
    total_updates: int,
) -> dict[str, float | int]:
    return {
        "learner/critic_loss": critic_loss,
        "learner/actor_loss": actor_loss,
        "learner/mean_q_value": mean_q,
        "learner/updates_per_second": updates_per_second,
        "learner/total_updates": total_updates,
    }
