"""The interface an RL algorithm implements.

``Algorithm`` runs in the learner (``update``, ``actor_state``, ``state_dict``,
``load_state_dict``); its ``Actor`` runs in the policy server once per chunk
(``noise`` may set π0.5's initial flow noise, ``act`` returns the chunk).
``uses_base = False`` skips π0.5; ``needs_images = True`` adds the cameras.

Algorithms are found by name: built-in ``rlt`` and ``dsrl``, an entry point in
the ``armnet_rlt.algorithms`` group, or ``module:Class``.

Batches for ``update`` and ``pretrain`` (leading dimension ``B``):

- ``state``, ``next_state``: ``rl_token`` ``[B, 2048]``, ``proprioception``
  ``[B, 6]``, ``reference_action`` ``[B, base_chunk_len * 6]`` (π0.5's chunk);
  ``rl_token`` and ``reference_action`` are ``[B, 0]`` without the base;
- ``action`` ``[B, chunk_len * 6]``: the chunk as executed;
- ``reward``, ``done`` ``[B]``; ``curriculum_scale`` ``[B]`` when known;
- ``extras``: the ``extras`` ``act`` returned, ``[B, ...]``;
- ``images``, ``next_images`` (``needs_images``): view to uint8 ``[B, H, W, 3]``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

ENTRY_POINT_GROUP = "armnet_rlt.algorithms"
BUILTIN_ALGORITHMS = {
    "rlt": "armnet_rlt.algorithms.rlt:RLTAlgorithm",
    "dsrl": "armnet_rlt.algorithms.dsrl:DSRLAlgorithm",
}
BASE_IMAGE_SIZE = (224, 224)
CAMERA_VIEWS = ("top", "front", "wrist")
PAYLOAD_SCHEMA_VERSION = 4


@dataclass
class ActorInput:
    """What the actor sees once per chunk.

    ``images`` (``needs_images`` only): view to uint8 ``[H, W, 3]`` RGB at
    224x224 with the base, else at the algorithm's ``image_size``.
    ``rl_token`` is None without the base.
    """

    proprioception: Tensor
    rl_token: Tensor | None = None
    images: dict[str, Tensor] | None = None
    # --exploration-scale, and the lag-one noise correlation along the chunk.
    exploration_scale: float = 0.0
    exploration_correlation: float = 0.85
    # Base-policy episode (warmup or the base half of a paired evaluation).
    base: bool = False
    # Chunks already executed in this episode.
    chunk_index: int = 0


@dataclass
class ActorOutput:
    """``chunk`` ``[L, action_dim]`` absolute joint targets the cell executes;
    ``extras`` are stored with the transition and come back in batches."""

    chunk: Tensor
    extras: dict[str, Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class AlgorithmSpec:
    """The robot and base policy an algorithm is built for."""

    action_dim: int = 6
    proprioception_dim: int = 6
    rl_token_dim: int = 2048
    # Rows of π0.5's chunk kept as the reference chunk.
    base_chunk_len: int = 30
    # Rows the cell executes per chunk.
    chunk_len: int = 10
    image_views: tuple[str, ...] = CAMERA_VIEWS
    image_size: tuple[int, int] = BASE_IMAGE_SIZE
    # π0.5 predicts actions relative to the current joint positions.
    delta_actions: bool = True
    use_quantile_norm: bool = True
    # name ("state", "actions") to {"mean", "std", "q01", "q99"} lists.
    norm_stats: Mapping[str, Mapping[str, list[float]]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> AlgorithmSpec:
        values = {key: raw[key] for key in cls.__dataclass_fields__ if key in raw}
        for name in ("image_views", "image_size"):
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values)

    def norm_stats_objects(self) -> dict[str, Any] | None:
        """``norm_stats`` as ``artifacts.NormStats`` objects."""
        if self.norm_stats is None:
            return None
        from armnet_rlt.artifacts import NormStats

        return {
            name: NormStats(
                **{
                    key: np.asarray(value, dtype=np.float32) if value is not None else None
                    for key, value in stats.items()
                }
            )
            for name, stats in self.norm_stats.items()
        }


def norm_stats_lists(norm_stats: Mapping[str, Any] | None) -> dict[str, dict[str, list]] | None:
    """NormStats objects to the JSON form ``AlgorithmSpec.norm_stats`` holds."""
    if norm_stats is None:
        return None
    return {
        name: {
            key: (np.asarray(value).tolist() if value is not None else None)
            for key in ("mean", "std", "q01", "q99")
            for value in (getattr(stats, key, None),)
        }
        for name, stats in norm_stats.items()
    }


class Actor(nn.Module):
    """The part of an algorithm that runs in the policy server, once per chunk."""

    def noise(self, obs: ActorInput, explore: bool) -> Tensor | None:
        """Initial noise ``[horizon, action_dim]`` for π0.5's flow sampler, or
        None for π0.5's own. Called before π0.5."""
        return None

    def act(self, obs: ActorInput, base_chunk: Tensor | None, explore: bool) -> ActorOutput:
        """The chunk to execute. ``base_chunk`` ``[base_chunk_len, action_dim]``
        is π0.5's chunk (None without the base); ``explore`` False is evaluation."""
        raise NotImplementedError


class Algorithm:
    """Base class of an RL algorithm. ``config`` is the run's ``RLTConfig``;
    plugin options are in ``config.algorithm_options``."""

    name = ""
    needs_images = False
    uses_base = True
    # Image size the server resizes the cameras to when the base is not used.
    image_size: tuple[int, int] = BASE_IMAGE_SIZE
    # Mixes the demonstration cache into batches and may pretrain on it.
    uses_demos = False
    # Learner steps scheduled per new transition, and after the first episode.
    updates_per_transition = 1
    initial_updates = 0
    # Online batches draw each curriculum level equally often.
    balance_curriculum = True
    # Keep the in-flight chunk of a safety-stopped episode.
    keeps_partial_chunks = False

    def __init__(self, spec: AlgorithmSpec, config: Any) -> None:
        self.spec = spec
        self.config = config

    @classmethod
    def build_actor(cls, spec: AlgorithmSpec, config: Any) -> Actor:
        """A fresh actor for the policy server; weights come from ``actor_state``."""
        return cls(spec, config).make_actor()

    def make_actor(self) -> Actor:
        raise NotImplementedError

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        """One learner step; returns metrics to log."""
        raise NotImplementedError

    def actor_state(self) -> dict[str, Tensor]:
        """The actor's state dict as sent to the policy server."""
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def pretrain(self, batch: dict[str, Any]) -> dict[str, float]:
        """One step on a demonstration batch; only called when overridden."""
        raise NotImplementedError

    def relabel(self, episode: list[dict[str, Tensor]]) -> None:
        """Rewrite one episode's rewards in place before it is stored; by default
        a safety trip ends at ``-config.safety_trip_penalty``."""
        penalty = getattr(self.config, "safety_trip_penalty", None)
        if penalty is not None and episode and bool(float(episode[-1].get("safety_trip", 0.0))):
            episode[-1]["reward"] = torch.tensor(-float(penalty), dtype=torch.float32)

    def diagnostics(self, batch: dict[str, Any]) -> dict[str, float]:
        """Extra metrics logged with the losses every ``log_freq`` steps."""
        return {}


def has_pretraining(algorithm: type[Algorithm] | Algorithm) -> bool:
    cls = algorithm if isinstance(algorithm, type) else type(algorithm)
    return cls.pretrain is not Algorithm.pretrain


def steers_noise(actor: Actor) -> bool:
    return type(actor).noise is not Actor.noise


def load_algorithm(name: str) -> type[Algorithm]:
    """The algorithm class for a built-in, entry-point or ``module:Class`` name."""
    import importlib

    target = BUILTIN_ALGORITHMS.get(name)
    if target is None and ":" not in name:
        from importlib.metadata import entry_points

        found = [point for point in entry_points(group=ENTRY_POINT_GROUP) if point.name == name]
        if not found:
            known = sorted({*BUILTIN_ALGORITHMS, *(p.name for p in entry_points(group=ENTRY_POINT_GROUP))})
            raise ValueError(
                f"unknown algorithm {name!r}; known: {', '.join(known)} "
                "(or module:Class)"
            )
        cls = found[0].load()
    else:
        module, _, attribute = (target or name).partition(":")
        cls = getattr(importlib.import_module(module), attribute)
    if not (isinstance(cls, type) and issubclass(cls, Algorithm)):
        raise TypeError(f"{name!r} is not an armnet_rlt.algorithm.Algorithm subclass")
    return cls


def spec_for(
    config: Any,
    algorithm: type[Algorithm],
    norm_stats: Mapping[str, Any] | None = None,
) -> AlgorithmSpec:
    """The spec of a run with ``config`` (an ``RLTConfig``)."""
    network = config.network
    return AlgorithmSpec(
        action_dim=int(network.action_dim),
        proprioception_dim=int(network.proprioception_dim),
        rl_token_dim=int(network.rl_token_dim),
        base_chunk_len=int(network.reference_action_len),
        chunk_len=int(config.actions_to_execute),
        image_size=tuple(BASE_IMAGE_SIZE if algorithm.uses_base else algorithm.image_size),
        delta_actions=bool(config.policy_uses_delta_actions),
        use_quantile_norm=bool(config.use_quantile_norm),
        norm_stats=norm_stats_lists(norm_stats),
    )


def actor_meta(name: str, spec: AlgorithmSpec, config: Any) -> str:
    """JSON the policy server rebuilds an actor from: algorithm, spec, config."""
    from armnet_rlt.config import without_secrets

    raw = asdict(config) if hasattr(config, "__dataclass_fields__") else dict(config)
    return json.dumps(
        {"algorithm": name, "spec": spec.to_dict(), "config": without_secrets(raw)},
        default=str,
    )


def actor_payload(
    state: Mapping[str, Any], meta: str, *, learner_step: int | None
) -> dict[str, Any]:
    """Flat ``np.savez``-able actor description for a policy server."""
    info = json.loads(meta)
    info.update(schema_version=PAYLOAD_SCHEMA_VERSION, learner_step=learner_step)
    payload: dict[str, Any] = {"meta": json.dumps(info)}
    for name, value in state.items():
        payload[f"param/{name}"] = (
            value.detach().cpu().numpy() if isinstance(value, Tensor) else np.asarray(value)
        )
    return payload


def payload_meta(payload: Mapping[str, Any]) -> dict[str, Any]:
    meta = json.loads(str(payload["meta"]))
    if meta.get("schema_version") != PAYLOAD_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported actor payload schema {meta.get('schema_version')!r}; "
            "rebuild the actor image and restart the learner"
        )
    return meta


def actor_from_payload(
    payload: Mapping[str, Any], *, device: str | torch.device = "cpu"
) -> tuple[Actor, type[Algorithm], AlgorithmSpec, int | None]:
    """Rebuild the actor an ``actor_payload`` describes, in eval mode."""
    from armnet_rlt.config import config_from_dict

    meta = payload_meta(payload)
    cls = load_algorithm(str(meta["algorithm"]))
    spec = AlgorithmSpec.from_dict(meta["spec"])
    config = config_from_dict(meta["config"])
    actor = cls.build_actor(spec, config)
    state = {
        key.removeprefix("param/"): torch.as_tensor(np.asarray(value))
        for key, value in payload.items()
        if key.startswith("param/")
    }
    actor.load_state_dict(state)
    actor.to(device).eval()
    step = meta.get("learner_step")
    return actor, cls, spec, (int(step) if step is not None else None)
