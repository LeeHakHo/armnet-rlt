"""Check the DSRL inference path of an OpenPI checkpoint on this GPU.

Compares the split prefix/denoise sampler against the combined sampler,
installs a fresh DSRL actor through SetSteering and checks that steered
inference uses its noise. Run inside the openpi environment:

    RLT_CONFIG=<config> RLT_CHECKPOINT=<dir> python scripts/dsrl_gpu_check.py
"""

from __future__ import annotations

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
from openpi.models import model as model_module
from openpi.models.model import IMAGE_RESOLUTION

from armnet_rlt.config import DSRL_HIDDEN_DIMS, so101_network_config
from armnet_rlt.dsrl import DSRLActor, steering_payload
from armnet_rlt.openpi_rlt import OpenPIRLTPolicy, PolicyObservation
from armnet_rlt.resources import GREEN_BUTTON_PROMPT, MULTITASK_CONFIG

CONFIG = os.environ.get("RLT_CONFIG", MULTITASK_CONFIG)
CHECKPOINT = os.environ["RLT_CHECKPOINT"]
PROMPT = os.environ.get("RLT_PROMPT", GREEN_BUTTON_PROMPT)
NOISE_BOUND = float(os.environ.get("RLT_DSRL_NOISE_BOUND", "2.5"))


def main() -> None:
    started = time.time()
    policy = OpenPIRLTPolicy.from_checkpoint(CHECKPOINT, CONFIG, default_prompt=PROMPT)
    print(f"loaded in {time.time() - started:.0f}s", flush=True)
    height, width = IMAGE_RESOLUTION
    rng = np.random.default_rng(0)
    obs = PolicyObservation(
        state=np.zeros(6, np.float32),
        images={
            name: rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
            for name in ("front", "top", "wrist")
        },
        prompt=PROMPT,
    )
    plain = policy.infer(obs)

    rows, cols = policy.noise_shape()
    noise = np.clip(rng.normal(size=(rows, cols)), -2, 2).astype(np.float32)
    full = np.zeros((1, rows, int(policy._model.action_dim)), np.float32)
    full[0, :, :cols] = noise
    model_obs = model_module.Observation.from_dict(policy._transform_batch([obs]))
    ref_actions, ref_token = policy._sample(
        jax.random.key(0), model_obs, noise=jnp.asarray(full)
    )
    prefix, denoise = policy._split_sampler()
    pre, mask, kv, token = prefix(model_obs)
    actions = denoise(pre, mask, kv, jnp.asarray(full))
    print(
        "token max|diff|",
        float(jnp.abs(token - ref_token).max()),
        " actions max|diff|",
        float(jnp.abs(actions - ref_actions).max()),
        flush=True,
    )

    network = so101_network_config(
        reference_action_len=rows, predicted_action_len=rows, hidden_dims=DSRL_HIDDEN_DIMS
    )
    actor = DSRLActor(network, noise_bound=NOISE_BOUND, noise_steps=1)
    actor.set_norm_stats(policy.norm_stats, use_quantiles=policy.use_quantile_norm)
    actor.eval()
    step = policy.set_steering(
        steering_payload(
            actor.state_dict(),
            network=network,
            noise_bound=NOISE_BOUND,
            noise_steps=1,
            learner_step=1,
        )
    )
    result = policy.infer(obs, exploration_std=0.0)
    with torch.inference_mode():
        _, mean = actor(
            torch.from_numpy(result.rl_token)[None], torch.from_numpy(obs.state)[None]
        )
    expected = np.repeat(mean.numpy().reshape(1, cols) * NOISE_BOUND, rows, axis=0)
    print(
        "steering step",
        step,
        result.steering_step,
        " noise==actor mean:",
        bool(np.allclose(result.noise, expected, atol=1e-5)),
        flush=True,
    )
    print("steered chunk[0]", np.round(result.action_chunk[0], 2), flush=True)
    print("plain chunk[0]", np.round(plain.action_chunk[0], 2), flush=True)


if __name__ == "__main__":
    main()
