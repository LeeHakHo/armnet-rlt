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
    # DSRL only: the base policy's initial noise, ``(action_horizon, action_dim)``
    # in the base policy's normalized units, and the steering actor's learner step.
    noise: np.ndarray | None = None
    steering_step: int | None = None




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
        model: Any = None,
        noise_action_dim: int = len(MOTOR_NAMES),
        noise_seed: int | None = None,
    ) -> None:
        self._policy = policy
        self._sample = sample_actions_with_rl_token
        self.rl_token_dim = rl_token_dim
        self.norm_stats = norm_stats or {}
        self.use_quantile_norm = use_quantile_norm
        self.uses_delta_actions = uses_delta_actions
        self._model = model if model is not None else getattr(policy, "_model", None)
        self.noise_action_dim = noise_action_dim
        self._noise_rng = np.random.default_rng(noise_seed)
        self._steering: Any = None
        self.steering_step: int | None = None
        self._prefix_fn: Any = None
        self._denoise_fn: Any = None

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
            model=model,
        )

    def infer(
        self,
        observation: PolicyObservation | dict[str, Any],
        *,
        exploration_std: float | None = None,
        steer: bool = True,
    ) -> PolicyOutput:
        """Run one combined action/token forward pass.

        With ``exploration_std`` the initial noise of the flow is chosen here
        instead of sampled: the steering actor's output (or zero when ``steer``
        is false) plus Gaussian exploration of that standard deviation, and the
        result reports it as ``noise``.
        """
        if exploration_std is None:
            return self.infer_batch([observation])[0]
        return self._infer_with_noise(
            observation, exploration_std=exploration_std, steer=steer
        )

    def set_steering(self, payload: dict[str, Any]) -> int | None:
        """Install the DSRL actor that picks the flow's initial noise."""
        from armnet_rlt.dsrl import actor_from_steering_payload

        actor, step = actor_from_steering_payload(payload)
        if actor.per_step_action_dim != self.noise_action_dim:
            raise ValueError(
                f"steering actor emits {actor.per_step_action_dim}-D noise rows, "
                f"expected {self.noise_action_dim}"
            )
        self._steering = actor
        self.steering_step = step
        return step

    def _transform_batch(
        self, observations: list[PolicyObservation | dict[str, Any]]
    ) -> dict[str, Any]:
        import jax
        import jax.numpy as jnp

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
        return jax.tree.map(
            lambda *values: jnp.stack(
                [jnp.asarray(value) for value in values], axis=0
            ),
            *transformed,
        )

    def to_model_observation(
        self, observations: list[PolicyObservation | dict[str, Any]]
    ) -> Any:
        """Batch observations into the openpi ``Observation`` the model expects.

        Shares the transform path with ``infer_batch`` so a cache built from these
        embeddings matches what inference would have produced.
        """
        from openpi.models import model as model_module

        return model_module.Observation.from_dict(self._transform_batch(observations))

    def _output(self, state: Any, actions: Any, token: Any) -> PolicyOutput:
        import jax

        token = np.asarray(token, dtype=np.float32).reshape(-1)
        if token.shape != (self.rl_token_dim,):
            raise RLTokenExtractionUnsupportedError(
                f"RL token has shape {token.shape}, "
                f"expected ({self.rl_token_dim},)"
            )
        outputs = jax.tree.map(np.asarray, {"state": state, "actions": actions})
        outputs = self._policy._output_transform(outputs)
        chunk = np.asarray(outputs["actions"], dtype=np.float32)
        if chunk.ndim != 2:
            raise RuntimeError(
                f"OpenPI action chunk has shape {chunk.shape}, "
                "expected [time, action]"
            )
        return PolicyOutput(action_chunk=chunk, rl_token=token)

    def infer_batch(
        self, observations: list[PolicyObservation | dict[str, Any]]
    ) -> list[PolicyOutput]:
        """Run one fixed-size JAX batch of combined action/token inference."""

        import jax
        from openpi.models import model as model_module

        if not observations:
            return []
        batched = self._transform_batch(observations)
        model_obs = model_module.Observation.from_dict(batched)
        self._policy._rng, sample_rng = jax.random.split(self._policy._rng)
        actions, tokens = self._sample(sample_rng, model_obs)
        return [
            self._output(batched["state"][index], actions[index], tokens[index])
            for index in range(len(observations))
        ]

    def _split_sampler(self) -> tuple[Any, Any]:
        """JIT the Pi0RL sampler as a prefix pass (RL token, KV cache) and a
        denoising pass that starts from a given noise."""
        if self._prefix_fn is None:
            if self._model is None:
                raise RLTokenExtractionUnsupportedError(
                    "noise steering needs the loaded Pi0RL model"
                )
            import jax
            import jax.numpy as jnp
            from openpi.models import pi0 as _pi0
            from openpi.shared import nnx_utils

            def prefix_with_token(model: Any, observation: Any) -> Any:
                observation = model._preprocess_observation(
                    None, observation, train=False
                )
                prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(
                    observation
                )
                attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
                positions = jnp.cumsum(prefix_mask, axis=1) - 1
                outputs, kv_cache = model.PaliGemma.llm(
                    [prefix_tokens, None], mask=attn_mask, positions=positions
                )
                rl_token = model.rl_encoder(
                    jax.lax.stop_gradient(outputs[0]), mask=prefix_mask
                )
                return observation, prefix_mask, kv_cache, rl_token

            def denoise(
                model: Any, observation: Any, prefix_mask: Any, kv_cache: Any, noise: Any
            ) -> Any:
                return model._denoise_actions(
                    observation, prefix_mask, kv_cache, num_steps=10, noise=noise
                )

            self._prefix_fn = nnx_utils.module_jit(
                types.MethodType(prefix_with_token, self._model)
            )
            self._denoise_fn = nnx_utils.module_jit(
                types.MethodType(denoise, self._model)
            )
        return self._prefix_fn, self._denoise_fn

    def noise_shape(self) -> tuple[int, int]:
        """Rows and columns of the noise the steering actor controls."""
        if self._model is None:
            raise RLTokenExtractionUnsupportedError(
                "noise steering needs the loaded Pi0RL model"
            )
        return int(self._model.action_horizon), self.noise_action_dim

    def _infer_with_noise(
        self,
        observation: PolicyObservation | dict[str, Any],
        *,
        exploration_std: float,
        steer: bool,
    ) -> PolicyOutput:
        import jax.numpy as jnp
        from openpi.models import model as model_module

        from armnet_rlt.dsrl import choose_noise, random_noise

        if steer and self._steering is None:
            raise RuntimeError("no steering actor has been set on this policy")
        if not np.isfinite(exploration_std) or exploration_std < 0:
            raise ValueError("exploration_std must be finite and nonnegative")
        prefix, denoise = self._split_sampler()
        batched = self._transform_batch([observation])
        model_obs = model_module.Observation.from_dict(batched)
        preprocessed, prefix_mask, kv_cache, tokens = prefix(model_obs)
        token = np.asarray(tokens[0], dtype=np.float32).reshape(-1)
        rows, columns = self.noise_shape()
        proprioception = np.asarray(
            observation.state
            if isinstance(observation, PolicyObservation)
            else observation["observation.state"],
            dtype=np.float32,
        )
        if steer:
            actor = self._steering
            if (actor.noise_len, actor.per_step_action_dim) != (rows, columns):
                raise ValueError(
                    f"steering actor emits {(actor.noise_len, actor.per_step_action_dim)} "
                    f"noise, the base policy takes {(rows, columns)}"
                )
            noise = choose_noise(
                actor,
                token,
                proprioception,
                exploration_std=exploration_std,
                noise_bound=float(actor.noise_bound),
                rng=self._noise_rng,
            )
        else:
            noise = random_noise(
                (rows, columns),
                exploration_std=exploration_std,
                noise_bound=(
                    float(self._steering.noise_bound)
                    if self._steering is not None
                    else float("inf")
                ),
                rng=self._noise_rng,
            )
        full = np.zeros((1, rows, int(self._model.action_dim)), dtype=np.float32)
        full[0, :, :columns] = noise
        actions = denoise(preprocessed, prefix_mask, kv_cache, jnp.asarray(full))
        result = self._output(batched["state"][0], actions[0], token)
        return PolicyOutput(
            action_chunk=result.action_chunk,
            rl_token=result.rl_token,
            noise=noise,
            steering_step=self.steering_step if steer else None,
        )

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

