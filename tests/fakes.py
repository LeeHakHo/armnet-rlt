"""A stand-in for OpenPIRLTPolicy and helpers shared by the policy server tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from armnet_rlt.openpi_rlt import PolicyObservation, PolicyOutput


@dataclass
class Stats:
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray | None = None
    q99: np.ndarray | None = None


class FakeBase:
    """π0.5 stand-in: the chunk is the first state entry everywhere, or the
    noise it was given; the RL token is 0..2047."""

    rl_token_dim = 2048
    use_quantile_norm = True
    uses_delta_actions = True

    def __init__(self, horizon: int = 30) -> None:
        self.horizon = horizon
        self.norm_stats = {
            "state": Stats(np.arange(6.0), np.ones(6), np.zeros(6), np.full(6, 2.0)),
            "actions": Stats(np.zeros((horizon, 6)), np.ones((horizon, 6))),
        }
        self.observations: list[PolicyObservation] = []
        self.noises: list[np.ndarray | None] = []
        self.resets = 0

    def _token(self) -> np.ndarray:
        return np.arange(2048, dtype=np.float32) / 2048.0

    def infer(self, observation: PolicyObservation) -> PolicyOutput:
        self.observations.append(observation)
        return PolicyOutput(
            action_chunk=np.full((self.horizon, 6), observation.state[0], dtype=np.float32),
            rl_token=self._token(),
        )

    def noise_shape(self) -> tuple[int, int]:
        return self.horizon, 6

    def prefix(self, observation: PolicyObservation):
        self.observations.append(observation)
        return observation, self._token()

    def denoise(self, handle: PolicyObservation, noise: np.ndarray | None) -> PolicyOutput:
        self.noises.append(noise)
        chunk = (
            np.full((self.horizon, 6), -handle.state[0], dtype=np.float32)
            if noise is None
            else np.asarray(noise, dtype=np.float32) + handle.state[0]
        )
        return PolicyOutput(action_chunk=chunk, rl_token=self._token())

    def reset(self) -> None:
        self.resets += 1


def observation(seed: int = 0, state0: float = 3.0, size: int = 224) -> PolicyObservation:
    rng = np.random.default_rng(seed)
    return PolicyObservation(
        state=np.array([state0, 0.1, -0.2, 0.3, 0, 0.5], dtype=np.float32),
        images={
            name: rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
            for name in ("front", "top", "wrist")
        },
        prompt="push the green button",
    )
