# Armnet RL Token

Online RL-Token (TD3) for Pi0 / Pi0.5 on a single-arm SO-101.

The included backends are **Modal** (learner GPU) and **Armnet** (robot cell).
Both are explicit dependencies you can swap: run `run_learner` on a local GPU
or another cloud, and replace the Armnet actor job with a local robot or
simulator (`operator_control.py` is the cell/reset/scoring seam).

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked
cp .env.example .env
```

Set `ARMNET_API_KEY` in `.env`. Create matching secrets once:

```bash
TOKEN="$(openssl rand -hex 24)"
uv run modal secret create armnet-rlt-auth RLT_LEARNER_AUTH_TOKEN="$TOKEN"
uv run armnet secret create armnet-rlt-auth "$TOKEN"
uv run modal secret create huggingface-secret HF_TOKEN="$HF_TOKEN"
uv run armnet secret create huggingface-secret "$HF_TOKEN"
```

## Run

Build the demonstration cache (Modal A100), start the learner, then submit
the actor. Stop the detached Modal app when collection ends.

```bash
uv run modal run src/armnet_rlt/modal_cache.py::build_green_button_cache \
  --inference-batch-size 4

uv run modal run --detach src/armnet_rlt/modal_app.py::main \
  --config-name pi05_rlt_busybox_push_green_button \
  --run-id <run_id>

uv run rlt-submit-actor \
  --task <armnet_task_slug> \
  --config-name pi05_rlt_busybox_push_green_button \
  --checkpoint-dir volume://openpi/checkpoints/<checkpoint> \
  --language-instruction "push the green button" \
  --learner-key 'pi05_rlt_busybox_push_green_button:<run_id>' \
  --num-rollouts 20
```

For adaptive scene variation, add `--armnet.variation` and
`--variation-curriculum`. The curriculum starts at 25% of each cell's
configured variation spread, promotes by 0.05 after at least 80% success over
20 rollouts, and demotes by 0.10 at or below 55%. Every fifth episode places
one rotating variation axis at the current curriculum frontier. The scale,
rolling success rate, and frontier axis are saved in the rollout metadata.
Use `--no-armnet.variation-cameras` (or the corresponding rail/lighting
selector) to exclude one device family while retaining the others.
The actor's default exploration multiplier is 0.5; use 0 for a deterministic
validation run. Actor submissions default to 40 rollouts.

Online checkpoints also update `online_replay_buffer.pt` beside the step
directories. Resuming restores that buffer when present; older runs created
before this format resume their weights, optimizers, schedulers, and rollout
history with an empty online buffer.

For `push_green_button`, online rollouts apply a curriculum-scaled penalty for
pressing any non-green button. One wrong press changes the terminal reward from
`1.0` to `0.95` at variation scale 0 and to `0.5` at scale 1; a failed rollout
remains at zero. Multiple wrong presses accumulate down to a floor of `0.0`;
rewards are always in `[0, 1]`. Binary success reporting is unchanged. Actor weights are
reloaded only between episodes, so one rollout never changes policy midway
through its action chunks.

The example cache uses `pravsels/pi05_rlt_busybox_push_green_button` and
`villekuosmanen/busybox_push_green_button`. Replace those with your own
checkpoint and dataset for a different task.

For the single-arm 27-task min/max model, build a separate cache whose
inference prompt comes from each demonstration episode:

```bash
uv run modal run src/armnet_rlt/modal_cache.py::build_multitask_cache \
  --inference-batch-size 4
```

This uses `pravsels/pi05_rlt_busybox_multitask_singlearm_minmax`,
`villekuosmanen/busybox_multitask`, and config
`pi05_rlt_busybox_multitask_singlearm_minmax`. The actor checkpoint belongs at
`volume://openpi/checkpoints/pi05_rlt_busybox_multitask_singlearm_minmax`.

## Frozen evaluation

Freeze a selected learner checkpoint into an actor-only artifact. The command
reads the checkpoint from the Modal learner volume, removes critics and
optimizer state, and can upload the result directly to the Armnet volume:

```bash
uv run rlt-freeze-checkpoint \
  --config-name pi05_rlt_busybox_multitask_singlearm_minmax \
  --run-id <learner_run_id> \
  --step latest \
  --armnet-volume-path rlt/frozen/<learner_run_id>/frozen_actor.pt
```

Evaluate the immutable actor against its base OpenPI policy on paired scenes:

```bash
uv run rlt-submit-eval \
  --task push_green_button \
  --config-name pi05_rlt_busybox_multitask_singlearm_minmax \
  --checkpoint-dir \
    volume://openpi/checkpoints/pi05_rlt_busybox_multitask_singlearm_minmax \
  --frozen-rlt-checkpoint \
    volume://rlt/frozen/<learner_run_id>/frozen_actor.pt \
  --language-instruction "push the green button" \
  --num-rollouts 20 \
  --armnet.variation \
  --robot-telemetry-strict
```

Each requested rollout is one matched scene: the base policy and frozen RLT
actor run after separate resets to the same seeded variation. Their order
alternates between pairs to avoid systematic first-run bias. Frozen evaluation
forces exploration to zero, never opens a learner connection, and never sends
transitions or updates weights. The recorded dataset tags frames with `[base]`
or `[frozen_rlt]`, and the result reports per-variant aggregates. Use
`--no-include-base` to evaluate only the frozen actor.

```bash
uv run pytest
```
