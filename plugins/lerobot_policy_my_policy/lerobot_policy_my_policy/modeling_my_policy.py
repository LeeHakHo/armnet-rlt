"""Example policy: a small CNN per camera and an MLP that predicts a chunk of actions.

Replace the network in `__init__` and `_predict`, and the loss in `forward`. The rest is
the interface the learner, the policy server and the evaluation call.
"""

from collections import deque

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_my_policy import MyPolicyConfig


class MyPolicyPolicy(PreTrainedPolicy):
    # LeRobot finds this class as <config name without "Config"> + "Policy".
    config_class = MyPolicyConfig
    name = "my_policy"

    def __init__(self, config: MyPolicyConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.image_keys = list(config.image_features)
        state = config.robot_state_feature
        self.state_dim = state.shape[0] if state else 0
        self.action_dim = config.action_feature.shape[0]

        # One encoder shared by all cameras: strided convolutions, then a global average.
        layers, channels = [], 3
        for out in config.image_channels:
            layers += [nn.Conv2d(channels, out, 3, stride=2, padding=1), nn.ReLU()]
            channels = out
        self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())

        in_dim = channels * len(self.image_keys) + self.state_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.chunk_size * self.action_dim),
        )
        self.reset()

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self) -> None:
        """Called at the start of every episode."""
        self._queue: deque[Tensor] = deque(maxlen=self.config.n_action_steps)

    def _predict(self, batch: dict[str, Tensor]) -> Tensor:
        """Normalized action chunk, shape (batch, chunk_size, action_dim)."""
        features = [self.encoder(batch[key]) for key in self.image_keys]
        if self.state_dim:
            features.append(batch[OBS_STATE])
        out = self.head(torch.cat(features, dim=-1))
        return out.view(-1, self.config.chunk_size, self.action_dim)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        return self._predict(batch)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """One action per call; a new chunk is predicted every n_action_steps calls."""
        if not self._queue:
            chunk = self._predict(batch)[:, : self.config.n_action_steps]
            self._queue.extend(chunk.transpose(0, 1))
        return self._queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training loss on a batch of normalized observations and action chunks."""
        error = F.l1_loss(self._predict(batch), batch[ACTION], reduction="none")
        if "action_is_pad" in batch:
            error = error * ~batch["action_is_pad"].unsqueeze(-1)
        loss = error.mean()
        return loss, {"l1_loss": loss.item()}
