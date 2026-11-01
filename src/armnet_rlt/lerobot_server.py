"""Serve a LeRobot policy from a policy server, with checkpoint updates from a learner."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from armnet_rlt.lerobot_actor import (
    ACT_CONFIG_NAME,
    _predict_chunk,
    _select_action,
    policy_frame,
    resize_image,
)
from armnet_rlt.remote_policy import PolicyBackend, decode_images
from armnet_rlt.so101_actor import _load_parameters


class LeRobotPolicyBackend(PolicyBackend):
    """A loaded LeRobot policy behind the policy server.

    A chunking policy (ACT without temporal ensembling) answers with
    ``n_action_steps`` actions and takes exploration noise; other policies
    answer with one action from ``select_action``.
    """

    def __init__(
        self,
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        *,
        shapes: Mapping[str, tuple[int, int]],
        n_action_steps: int = 0,
        temporal_ensemble_coeff: float | None = None,
        learner_step: int | None = None,
        served_from_learner: bool = False,
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self.shapes = {key: (int(h), int(w)) for key, (h, w) in shapes.items()}
        self.lerobot_type = str(policy.config.type)
        is_act = self.lerobot_type == ACT_CONFIG_NAME
        if temporal_ensemble_coeff is not None:
            if not is_act:
                raise ValueError(
                    f"temporal_ensemble_coeff needs an ACT policy, not {self.lerobot_type}"
                )
            from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

            policy.config.temporal_ensemble_coeff = float(temporal_ensemble_coeff)
            policy.config.n_action_steps = 1
            policy.temporal_ensembler = ACTTemporalEnsembler(
                float(temporal_ensemble_coeff), policy.config.chunk_size
            )
        self.chunked = is_act and policy.config.temporal_ensemble_coeff is None
        self.n_action_steps = int(n_action_steps) or int(policy.config.n_action_steps)
        if is_act and not 1 <= self.n_action_steps <= int(policy.config.chunk_size):
            raise ValueError(
                f"n_action_steps={self.n_action_steps} must be in "
                f"[1, chunk_size={policy.config.chunk_size}]"
            )
        self.uses_task = not is_act
        self.learner_step = learner_step
        self.served_from_learner = served_from_learner
        policy.eval()

    def info(self) -> dict[str, Any]:
        meta = {
            "policy_type": "lerobot",
            "lerobot_type": self.lerobot_type,
            "chunked": self.chunked,
            "n_action_steps": self.n_action_steps,
            "uses_task": self.uses_task,
            "shapes": {key: list(shape) for key, shape in self.shapes.items()},
            "learner_step": -1 if self.learner_step is None else int(self.learner_step),
            "steering_from_learner": bool(self.served_from_learner),
        }
        return {"meta": json.dumps(meta)}

    def infer(self, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
        images = decode_images(arrays, str(arrays["codec"]))
        missing = [key for key in self.shapes if key not in images]
        if missing:
            raise ValueError(f"request has no image for {missing}")
        images = {key: resize_image(images[key], size) for key, size in self.shapes.items()}
        state = np.asarray(arrays["state"], dtype=np.float32)
        task = str(arrays["task"]) if "task" in arrays and self.uses_task else None
        frame = policy_frame(state, images, task)
        if self.chunked:
            actions = _predict_chunk(
                self._policy,
                self._preprocessor,
                self._postprocessor,
                frame,
                n_action_steps=self.n_action_steps,
                exploration_scale=float(arrays.get("exploration_scale", 0.0)),
                exploration_correlation=float(arrays.get("exploration_correlation", 0.85)),
            )
        else:
            actions = _select_action(
                self._policy, self._preprocessor, self._postprocessor, frame
            )
        return {
            "actions": np.asarray(actions, dtype=np.float32),
            "policy_step": -1 if self.learner_step is None else int(self.learner_step),
        }

    def apply_weights(
        self, weights: Mapping[str, torch.Tensor], learner_step: int | None, partial: bool
    ) -> None:
        """Install a learner update: a full state dict or the tensors that changed."""
        step = _load_parameters(
            self._policy,
            {"policy": dict(weights), "partial": partial, "learner_step": learner_step},
        )
        self._policy.eval()
        if step is not None:
            self.learner_step = step

    def reset(self) -> None:
        self._policy.reset()
