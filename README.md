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

The example cache uses `pravsels/pi05_rlt_busybox_push_green_button` and
`villekuosmanen/busybox_push_green_button`. Replace those with your own
checkpoint and dataset for a different task.

```bash
uv run pytest
```
