# Armnet RL Token

Online RL-Token (TD3) for Pi0/Pi0.5 on an Armnet SO-101 cell. The learner runs
on Modal; the actor runs beside the physical robot on Armnet.

## Prerequisites

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- An Armnet API key
- A Modal account
- A Hugging Face token that can read the model and create private eval datasets

Credentials are supplied at setup time and are never built into an image or
committed to this repository.

## 1. Install and sign in

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
ARMNET_API_KEY=<your-armnet-api-key>
HF_TOKEN=<your-huggingface-token>
```

Install the locked environment and authenticate the Modal CLI:

```bash
uv sync --locked
uv run modal setup
```

`modal setup` stores Modal credentials in the user's normal Modal
configuration. The repository does not read a Modal API token from `.env`.

## 2. Create secrets once

The learner and actor share a randomly generated bearer token. Store the same
value under the same secret name in Modal and Armnet:

```bash
set -a
source .env
set +a

LEARNER_TOKEN="$(openssl rand -hex 32)"

uv run modal secret create armnet-rlt-auth \
  RLT_LEARNER_AUTH_TOKEN="$LEARNER_TOKEN"
uv run armnet secret create armnet-rlt-auth "$LEARNER_TOKEN"

uv run modal secret create huggingface-secret HF_TOKEN="$HF_TOKEN"
uv run armnet secret create huggingface-token "$HF_TOKEN"

unset LEARNER_TOKEN
```

If a team administrator has already provisioned these named secrets, do not
overwrite them.

## 3. Build the demonstration cache

The recommended single-arm multitask setup uses:

- model: `AutoRLBench/pi05_rlt_busybox_multitask_singlearm_minmax`
- demonstrations: `AutoRLBench/busybox_multitask`
- config: `pi05_rlt_busybox_multitask_singlearm_minmax`

Build the cache once in the configured Modal volume:

```bash
uv run --env-file .env modal run \
  src/armnet_rlt/modal_cache.py::build_multitask_cache \
  --inference-batch-size 4
```

The cache and normalization assets are reusable across learner runs whose
network/action dimensions match.

## 4. Start a learner

Choose a unique run ID. In a shared Modal workspace, include your username:

```bash
export RLT_CONFIG=pi05_rlt_busybox_multitask_singlearm_minmax
export RLT_RUN_ID="${USER}-$(date -u +%Y%m%d-%H%M%S)"

uv run --env-file .env modal run --detach \
  src/armnet_rlt/modal_app.py::main \
  --config-name "$RLT_CONFIG" \
  --run-id "$RLT_RUN_ID" \
  --embodiment so101
```

Wait for the learner to print its rendezvous entry. Keep this process open, or
disconnect it after the remote Modal function is running; `--detach` keeps the
learner alive.

## 5. Start an actor

In another terminal, set the same config/run ID and submit the robot job:

```bash
export RLT_CONFIG=pi05_rlt_busybox_multitask_singlearm_minmax
export RLT_RUN_ID=<the-run-id-from-step-4>

uv run --env-file .env rlt-submit-actor \
  --task push_green_button \
  --config-name "$RLT_CONFIG" \
  --hf-checkpoint-repo \
    AutoRLBench/pi05_rlt_busybox_multitask_singlearm_minmax \
  --language-instruction "push the green button" \
  --learner-key "${RLT_CONFIG}:${RLT_RUN_ID}" \
  --num-rollouts 20 \
  --armnet.variation \
  --variation-curriculum \
  --robot-telemetry-strict \
  --detach
```

Alternatively, put the OpenPI checkpoint on the Armnet volume and replace
`--hf-checkpoint-repo` with:

```text
--checkpoint-dir volume://openpi/checkpoints/<checkpoint-name>
```

The actor records each rollout as a private LeRobot dataset and uploads it to
Hugging Face unless `--no-push-to-hub` is supplied.

## How actor-to-learner discovery works

There is no hard-coded IP address or DNS name in this repository:

1. Modal starts the learner and creates an ephemeral, public-CA TLS endpoint
   with HTTP/2 enabled for gRPC.
2. The learner registers `{host, port, TLS, config, run ID}` in the Modal Dict
   named `armnet-rlt-rendezvous`.
3. `rlt-submit-actor` resolves `<config>:<run-id>` from that Dict immediately
   before submitting the Armnet job.
4. The resolved endpoint is injected into that job's arguments.
5. The actor connects over TLS and sends the bearer token from its Armnet
   secret as gRPC authorization metadata.

The bearer token is never stored in the rendezvous Dict, job arguments, image,
or logs. The Modal TLS endpoint changes whenever a learner is restarted, so an
actor must be submitted against the currently active rendezvous entry.

## Multiple collaborators

Different Modal accounts have independent volumes, Dicts, and secrets.

When collaborators share one Modal workspace:

- always use unique run IDs;
- do not reuse another learner's `<config>:<run-id>` key;
- the shared cache volume is safe to reuse;
- each run writes checkpoints under its own run-ID directory;
- use one actor stream per learner unless combining actors is intentional,
  because one learner mixes every connected actor's transitions.

The default Modal resource names can be overridden in `.env`:

```dotenv
RLT_MODAL_VOLUME=pi0-rlt-data
RLT_MODAL_RENDEZVOUS=armnet-rlt-rendezvous
RLT_MODAL_AUTH_SECRET=armnet-rlt-auth
```

For a per-user learner token in a shared workspace, choose a unique secret
name in `RLT_MODAL_AUTH_SECRET`, create the same named Armnet secret, and pass
that name to `rlt-submit-actor --auth-secret <name>`.

## Stop and resume

Actor jobs stop after `--num-rollouts`. The learner continues waiting for data
and continues consuming a Modal GPU, so stop it when collection finishes:

```bash
uv run modal app list
uv run modal app stop -y <learner-app-id>
```

Resume persisted weights, optimizer state, scheduler state, rollout history,
and `online_replay_buffer.pt` with:

```bash
uv run --env-file .env modal run --detach \
  src/armnet_rlt/modal_app.py::main \
  --config-name "$RLT_CONFIG" \
  --run-id "$RLT_RUN_ID" \
  --embodiment so101 \
  --resume
```

## Frozen evaluation

Freeze an actor-only checkpoint and optionally upload it to the Armnet volume:

```bash
uv run --env-file .env rlt-freeze-checkpoint \
  --config-name "$RLT_CONFIG" \
  --run-id "$RLT_RUN_ID" \
  --step latest \
  --armnet-volume-path \
    "rlt/frozen/${RLT_CONFIG}/${RLT_RUN_ID}/frozen_actor.pt"
```

Then compare frozen RLT against base OpenPI on matched scenes:

```bash
uv run --env-file .env rlt-submit-eval \
  --task push_green_button \
  --config-name "$RLT_CONFIG" \
  --checkpoint-dir volume://openpi/checkpoints/<checkpoint-name> \
  --frozen-rlt-checkpoint \
    "volume://rlt/frozen/${RLT_CONFIG}/${RLT_RUN_ID}/frozen_actor.pt" \
  --language-instruction "push the green button" \
  --num-rollouts 20 \
  --armnet.variation \
  --robot-telemetry-strict
```

Frozen evaluation disables exploration and weight updates. Base and RLT run on
matched seeded scenes with counterbalanced order.

## Development

```bash
uv run pytest
```

The actor image intentionally installs released Armnet packages from PyPI. It
does not copy or import the surrounding `alpha-robotics` checkout.
