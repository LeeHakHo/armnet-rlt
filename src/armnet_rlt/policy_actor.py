"""π0.5 plus an algorithm's actor: what the policy server runs once per chunk."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch

from armnet_rlt.algorithm import (
    BASE_IMAGE_SIZE,
    ActorInput,
    Algorithm,
    actor_from_payload,
    load_algorithm,
    steers_noise,
)
from armnet_rlt.openpi_rlt import PolicyObservation


@dataclass
class ChunkResult:
    """One chunk for the cell; ``base_chunk`` and ``rl_token`` are None without the base."""

    chunk: np.ndarray
    base_chunk: np.ndarray | None
    rl_token: np.ndarray | None
    extras: dict[str, np.ndarray] = field(default_factory=dict)
    actor_step: int | None = None
    # JPEG views of the observation, filled by the cell for needs_images.
    frames: dict[str, np.ndarray] | None = None
    needs_images: bool = False


class AlgorithmPolicy:
    """π0.5 (``base``, or None) and the installed actor.

    ``install`` takes an ``algorithm.actor_payload``; one that arrives during an
    episode is installed at ``reset``. ``algorithm`` pins the accepted
    algorithm and is required without the base."""

    def __init__(
        self,
        base: Any = None,
        *,
        algorithm: str = "",
        device: str = "cpu",
        actor_source: str = "client",
    ) -> None:
        self.base = base
        self.algorithm_class: type[Algorithm] | None = load_algorithm(algorithm) if algorithm else None
        if base is None and self.algorithm_class is None:
            raise ValueError("a policy server without π0.5 needs --algorithm")
        if base is None and self.algorithm_class.uses_base:
            raise ValueError(f"{algorithm} acts on π0.5's chunk; give the π0.5 checkpoint")
        self.device = torch.device(device)
        # "learner", "checkpoint" or "client".
        self.actor_source = actor_source
        self.actor: Any = None
        self.spec: Any = None
        self.actor_step: int | None = None
        self._pending: tuple[Any, type[Algorithm], Any, int | None] | None = None
        self._in_episode = False
        self._chunks = 0

    @property
    def uses_base(self) -> bool:
        cls = self.algorithm_class
        return cls.uses_base if cls is not None else True

    @property
    def needs_images(self) -> bool:
        cls = self.algorithm_class
        return bool(cls.needs_images) if cls is not None else False

    @property
    def image_size(self) -> tuple[int, int]:
        cls = self.algorithm_class
        if cls is None or cls.uses_base:
            return BASE_IMAGE_SIZE
        return tuple(cls.image_size)

    def install(self, payload: Mapping[str, Any]) -> int | None:
        actor, cls, spec, step = actor_from_payload(payload, device=self.device)
        if self.algorithm_class is not None and cls is not self.algorithm_class:
            raise ValueError(
                f"this server runs {self.algorithm_class.name or self.algorithm_class.__name__}, "
                f"not {cls.name or cls.__name__}"
            )
        if cls.uses_base:
            if self.base is None:
                raise ValueError(f"{cls.name} acts on π0.5's chunk; this server has no π0.5")
            detected = getattr(self.base, "uses_delta_actions", None)
            if detected is not None and bool(detected) != bool(spec.delta_actions):
                raise ValueError(
                    f"delta-action mismatch: the actor has delta_actions={spec.delta_actions}, "
                    f"π0.5 has uses_delta_actions={bool(detected)}"
                )
        self._pending = (actor, cls, spec, step)
        if not self._in_episode or self.actor is None:
            self._activate()
        return step

    set_steering = install

    def _activate(self) -> None:
        if self._pending is None:
            return
        self.actor, self.algorithm_class, self.spec, self.actor_step = self._pending
        self._pending = None

    def reset(self) -> None:
        """End of an episode: a waiting actor is installed."""
        self._in_episode = False
        self._chunks = 0
        self._activate()
        if self.base is not None:
            self.base.reset()

    def act(
        self,
        observation: PolicyObservation,
        *,
        explore: bool,
        exploration_scale: float = 0.0,
        exploration_correlation: float = 0.85,
        base: bool = False,
    ) -> ChunkResult:
        if not self._in_episode or self.actor is None:
            self._activate()
        actor = self.actor
        if actor is None:
            raise RuntimeError("no actor installed yet; the learner or the cell sends it")
        self._in_episode = True
        cls = self.algorithm_class
        spec = self.spec
        device = self.device
        obs = ActorInput(
            proprioception=torch.as_tensor(
                np.asarray(observation.state, dtype=np.float32), device=device
            ),
            exploration_scale=float(exploration_scale),
            exploration_correlation=float(exploration_correlation),
            base=bool(base),
            chunk_index=self._chunks,
        )
        if cls.needs_images:
            obs.images = {
                name: torch.as_tensor(_model_image(observation.images[name], spec.image_size), device=device)
                for name in spec.image_views
                if name in observation.images
            }
        token = base_chunk = None
        with torch.no_grad():
            if cls.uses_base:
                if steers_noise(actor):
                    handle, token = self.base.prefix(observation)
                    obs.rl_token = torch.as_tensor(token, device=device)
                    noise = actor.noise(obs, explore)
                    output = self.base.denoise(
                        handle, None if noise is None else noise.detach().cpu().numpy()
                    )
                else:
                    output = self.base.infer(observation)
                    token = np.asarray(output.rl_token, dtype=np.float32).reshape(-1)
                    obs.rl_token = torch.as_tensor(token, device=device)
                base_chunk = np.asarray(
                    output.action_chunk[: spec.base_chunk_len, : spec.action_dim],
                    dtype=np.float32,
                )
                if base_chunk.shape != (spec.base_chunk_len, spec.action_dim):
                    raise RuntimeError(
                        f"π0.5 chunk has shape {base_chunk.shape}, expected "
                        f"{(spec.base_chunk_len, spec.action_dim)}"
                    )
            result = actor.act(
                obs,
                None if base_chunk is None else torch.as_tensor(base_chunk, device=device),
                bool(explore),
            )
        chunk = result.chunk.detach().cpu().numpy().astype(np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != spec.action_dim or not len(chunk):
            raise RuntimeError(
                f"{cls.name} returned a chunk of shape {chunk.shape}; expected [L, {spec.action_dim}]"
            )
        if not np.isfinite(chunk).all():
            raise RuntimeError(f"{cls.name} returned a non-finite chunk")
        self._chunks += 1
        return ChunkResult(
            chunk=chunk,
            base_chunk=base_chunk,
            rl_token=token,
            extras={
                name: value.detach().cpu().numpy() for name, value in result.extras.items()
            },
            actor_step=self.actor_step,
            needs_images=bool(cls.needs_images),
        )

    def info(self) -> dict[str, Any]:
        """Fields of the server's Info reply."""
        cls = self.algorithm_class
        return {
            "algorithm": (cls.name or f"{cls.__module__}:{cls.__qualname__}") if cls else "",
            "uses_base": self.uses_base,
            "needs_images": self.needs_images,
            "image_size": list(self.image_size),
            "actor_source": self.actor_source,
            "actor_step": -1 if self.actor_step is None else int(self.actor_step),
            "actor_ready": self.actor is not None or self._pending is not None,
        }


def _model_image(image: Any, size: tuple[int, int]) -> np.ndarray:
    """One HWC uint8 image at ``size`` (height, width)."""
    image = np.asarray(image, dtype=np.uint8)
    if image.shape[:2] == tuple(size):
        return image
    from armnet_rlt.remote_policy import resize_for_model

    return resize_for_model(image, *size)
