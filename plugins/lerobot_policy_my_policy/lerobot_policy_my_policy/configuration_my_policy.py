"""Settings of the example policy. Every field is a `--policy.<field>` flag of lerobot-train."""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


# The name passed as --policy.type and scripts/run.sh dit --policy-type.
@PreTrainedConfig.register_subclass("my_policy")
@dataclass
class MyPolicyConfig(PreTrainedConfig):
    # Actions predicted per forward pass, and how many of them run before the next prediction.
    chunk_size: int = 30
    n_action_steps: int = 10

    # Network size.
    image_channels: tuple[int, ...] = (32, 64, 128)
    hidden_dim: int = 512

    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 1 <= self.n_action_steps <= self.chunk_size:
            raise ValueError("n_action_steps must be between 1 and chunk_size")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.robot_state_feature:
            raise ValueError("my_policy needs at least one camera or the robot state as input")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
