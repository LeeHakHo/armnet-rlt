"""Minimal OpenPI Pi0RL inference and RL-token extraction.

OpenPI and JAX are deliberately imported only while loading or running a
policy, so this module remains importable in the actor's lightweight tests.
"""

from __future__ import annotations

import logging
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

MOTOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
RL_TOKEN_DIM = 2048


def _install_training_stubs() -> None:
    """Keep OpenPI inference independent of its training-only dataset plugins.

    ``openpi.policies.policy_config`` imports the training data-loader module,
    whose type annotations reference RoboCandyWrapper and RewAct plugins. The
    actor never constructs a dataset, so installing those legacy projects would
    add exactly the cross-repository dependency this migration removes.
    """
    if "robocandywrapper" in sys.modules:
        return

    def stub(name: str, attrs: tuple[str, ...]) -> types.ModuleType:
        module = types.ModuleType(name)
        for attr in attrs:
            setattr(module, attr, type(attr, (), {}))
        sys.modules[name] = module
        return module

    wrapper = stub("robocandywrapper", ())
    wrapper.factory = stub(
        "robocandywrapper.factory", ("make_dataset_without_config",)
    )
    wrapper.plugins = stub(
        "robocandywrapper.plugins",
        ("EpisodeOutcomePlugin", "ControlModePlugin"),
    )
    stub("robocandywrapper.plugins.subtask", ("SubtaskPlugin",))
    stub("rewact_tools", ("PiStar0_6CumulativeRewardPlugin",))
    # policy_config imports checkpoints, which imports data_loader solely for
    # deferred type annotations. Avoid importing LeRobot's full datasets stack.
    stub("openpi.training.data_loader", ())


class RLTokenExtractionUnsupportedError(RuntimeError):
    """The loaded checkpoint is not a Pi0RL checkpoint."""


@dataclass(frozen=True)
class PolicyObservation:
    state: np.ndarray
    images: dict[str, np.ndarray]
    prompt: str

    def as_openpi_dict(self) -> dict[str, Any]:
        return {
            "observation.state": self.state,
            **{
                f"observation.images.{name}": image
                for name, image in self.images.items()
            },
            "prompt": self.prompt,
        }


@dataclass(frozen=True)
class PolicyOutput:
    action_chunk: np.ndarray
    rl_token: np.ndarray


def build_policy_observation(
    raw_obs: dict[str, Any], language_instruction: str | None
) -> PolicyObservation:
    """Map one Armnet SO-101 observation to OpenPI's input schema.

    ``front`` is mandatory. ``wrist`` is forwarded when the cell provides it.
    """

    joint_keys = [f"{name}.pos" for name in MOTOR_NAMES]
    missing = [key for key in joint_keys if key not in raw_obs]
    if missing:
        raise KeyError(f"observation missing joint positions: {missing}")
    if "front" not in raw_obs:
        raise KeyError(
            "observation missing required 'front' camera; "
            f"available keys: {sorted(raw_obs)}"
        )
    images = {
        name: np.asarray(raw_obs[name])
        for name in ("front", "top", "wrist")
        if name in raw_obs
    }
    return PolicyObservation(
        state=np.asarray([raw_obs[key] for key in joint_keys], dtype=np.float32),
        images=images,
        prompt=language_instruction or "",
    )


def _build_policy_obs(
    raw_obs: dict[str, Any], task: str | None
) -> dict[str, Any]:
    """Compatibility helper returning the plain OpenPI dictionary."""

    return build_policy_observation(raw_obs, task).as_openpi_dict()


def find_assets_dir(checkpoint_dir: str | Path) -> Path:
    checkpoint = Path(checkpoint_dir)
    candidates = [
        checkpoint / "assets",
        *(parent / "assets" for parent in checkpoint.parents[:3]),
    ]
    for candidate in candidates:
        if (candidate / "norm_stats.json").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find assets/norm_stats.json near "
        f"{checkpoint}; looked in {', '.join(map(str, candidates))}"
    )


def load_norm_stats(checkpoint_dir: str | Path, data_config: Any = None) -> dict:
    """Load OpenPI normalization statistics without importing OpenPI at module load."""

    from openpi.shared import normalize

    assets = find_assets_dir(checkpoint_dir)
    stats = normalize.load(assets)
    if bool(getattr(data_config, "use_per_timestep_action_norm", False)):
        try:
            stats = dict(stats)
            stats["actions"] = normalize.load_actions_per_timestep(assets)
        except FileNotFoundError:
            log.warning("per-timestep action stats missing in %s", assets)
    return stats


def uses_delta_actions(config: Any, data_config: Any) -> bool:
    """Read delta semantics from the data-config factory that defines them.

    OpenPI's concrete ``DataConfig`` contains the resulting transforms but does
    not retain factory-only fields such as ``use_delta_actions``.
    """

    factory = getattr(config, "data", None)
    if hasattr(factory, "use_delta_actions"):
        return bool(factory.use_delta_actions)
    return bool(getattr(data_config, "use_delta_actions", False))


class OpenPIRLTPolicy:
    """OpenPI policy wrapper returning a reference chunk and 2048-D token."""

    def __init__(
        self,
        policy: Any,
        sample_actions_with_rl_token: Any,
        *,
        rl_token_dim: int = RL_TOKEN_DIM,
        norm_stats: dict | None = None,
        use_quantile_norm: bool = True,
        uses_delta_actions: bool = False,
    ) -> None:
        self._policy = policy
        self._sample = sample_actions_with_rl_token
        self.rl_token_dim = rl_token_dim
        self.norm_stats = norm_stats or {}
        self.use_quantile_norm = use_quantile_norm
        self.uses_delta_actions = uses_delta_actions

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        config_name: str,
        *,
        default_prompt: str | None = None,
        rl_token_dim: int = RL_TOKEN_DIM,
    ) -> "OpenPIRLTPolicy":
        """Load a trained Pi0RL policy and JIT the combined inference method."""

        _install_training_stubs()
        from openpi.policies.policy_config import create_trained_policy
        from openpi.shared import nnx_utils
        from openpi.training.config import get_config

        config = get_config(config_name)
        data_config = config.data.create(config.assets_dirs, config.model)
        if type(data_config).__name__ == "LeRobotSO101BimanualDataConfig":
            raise ValueError(
                f"OpenPI config {config_name!r} is bimanual; the production "
                "Armnet RLT actor accepts only a single-arm SO-101 config"
            )
        norm_stats = load_norm_stats(checkpoint_dir, data_config)
        policy = create_trained_policy(
            config,
            checkpoint_dir,
            default_prompt=default_prompt,
            norm_stats=norm_stats,
        )
        model = getattr(policy, "_model", None)
        method = getattr(model, "sample_actions_with_rl_token", None)
        if method is None:
            raise RLTokenExtractionUnsupportedError(
                "The checkpoint model does not expose "
                "sample_actions_with_rl_token; a Pi0RL checkpoint is required"
            )
        try:
            sample = nnx_utils.module_jit(method)
        except Exception as exc:
            raise RLTokenExtractionUnsupportedError(
                "Could not JIT sample_actions_with_rl_token"
            ) from exc
        return cls(
            policy,
            sample,
            rl_token_dim=rl_token_dim,
            norm_stats=norm_stats,
            use_quantile_norm=bool(
                getattr(data_config, "use_quantile_norm", True)
            ),
            uses_delta_actions=uses_delta_actions(config, data_config),
        )

    def infer(
        self, observation: PolicyObservation | dict[str, Any]
    ) -> PolicyOutput:
        """Run one combined action/token forward pass."""
        return self.infer_batch([observation])[0]

    def infer_batch(
        self, observations: list[PolicyObservation | dict[str, Any]]
    ) -> list[PolicyOutput]:
        """Run one fixed-size JAX batch of combined action/token inference."""

        import jax
        import jax.numpy as jnp
        from openpi.models import model as model_module

        if not observations:
            return []
        transformed = [
            self._policy._input_transform(
                jax.tree.map(
                    lambda value: value,
                    observation.as_openpi_dict()
                    if isinstance(observation, PolicyObservation)
                    else observation,
                )
            )
            for observation in observations
        ]
        batched = jax.tree.map(
            lambda *values: jnp.stack(
                [jnp.asarray(value) for value in values], axis=0
            ),
            *transformed,
        )
        model_obs = model_module.Observation.from_dict(batched)
        self._policy._rng, sample_rng = jax.random.split(self._policy._rng)
        actions, tokens = self._sample(sample_rng, model_obs)
        results: list[PolicyOutput] = []
        for index in range(len(observations)):
            token = np.asarray(tokens[index], dtype=np.float32).reshape(-1)
            if token.shape != (self.rl_token_dim,):
                raise RLTokenExtractionUnsupportedError(
                    f"RL token has shape {token.shape}, "
                    f"expected ({self.rl_token_dim},)"
                )
            outputs = {
                "state": batched["state"][index],
                "actions": actions[index],
            }
            outputs = jax.tree.map(np.asarray, outputs)
            outputs = self._policy._output_transform(outputs)
            chunk = np.asarray(outputs["actions"], dtype=np.float32)
            if chunk.ndim != 2:
                raise RuntimeError(
                    f"OpenPI action chunk has shape {chunk.shape}, "
                    "expected [time, action]"
                )
            results.append(PolicyOutput(action_chunk=chunk, rl_token=token))
        return results

    def predict_with_rl_token(
        self, observation: PolicyObservation | dict[str, Any]
    ) -> tuple[PolicyOutput, np.ndarray]:
        """Compatibility form used by older actor call sites."""

        result = self.infer(observation)
        return result, result.rl_token

    def reset(self) -> None:
        reset = getattr(self._policy, "reset", None)
        if callable(reset):
            reset()


RLTokenExtractor = OpenPIRLTPolicy

