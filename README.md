# armnet-rlt

Online reinforcement learning on top of a pretrained π0.5 policy, running on
Armnet SO-101 robot cells.

- **RLT** (`--algorithm rlt`): a TD3 head reads the RL token and joint state
  and outputs an action chunk that refines π0.5's chunk.
- **DSRL** (`--algorithm dsrl`): π0.5 stays frozen; a SAC actor chooses the
  initial noise of its flow-matching sampler, so every executed chunk is one
  π0.5 produced. Follows the official π0 DSRL agent.
- **Streaming BC** / **offline BC**: clone successful rollouts or a recorded
  cache into the RLT head.
- **Checkpoint evaluation** of LeRobot policies (Diffusion Policy, SmolVLA,
  multi-task DiT) on the same cells, as baselines.

The learner runs on a local GPU (`rlt-local`) or on Modal (`modal run`). The
actor runs on the cell through Armnet and talks to the learner over gRPC.
Policy inference runs on your GPU by default: `rlt-local policy-server`
serves π0.5 or a LeRobot policy and the cell only streams observations to it
(`--no-remote-policy` loads the policy on the cell instead).

## Install

Requirements: Python 3.12, [`uv`](https://docs.astral.sh/uv/), Docker (image
build), an Armnet API key, a Hugging Face token.

```bash
cp .env.example .env            # fill in ARMNET_API_KEY, HF_TOKEN, RLT_HF_SECRET
uv sync --locked                # rlt-submit-actor, rlt-submit-eval, rlt-freeze-checkpoint
scripts/setup_local.sh learner  # envs/venv-learner: rlt-local bc, streaming-bc, online, checkpoint-eval
scripts/setup_local.sh openpi   # envs/venv-openpi: rlt-local cache, bc-cache, policy-server
```

LeRobot is installed from `third_party/lerobot` (upstream 0.5.2). OpenPI with
the RL-token model comes from `OPENPI_REPO` at `OPENPI_REVISION`
(`scripts/setup_local.sh`, `Dockerfile.actor`).

## Secrets

The actor and learner share a bearer token. Store it as an Armnet secret and
export it for the local learner:

```bash
set -a; source .env; set +a
TOKEN="$(openssl rand -hex 32)"
uv run armnet secret create armnet-rlt-auth "$TOKEN"
uv run armnet secret create "$RLT_HF_SECRET" "$HF_TOKEN"
echo "RLT_LEARNER_AUTH_TOKEN=$TOKEN" >> .env
```

`rlt-submit-actor --auth-secret <name>` and `--hf-secret <name>` select the
secrets a job reads. Secrets are never written into images, job arguments or
logs.

## Actor image

```bash
uv run python scripts/build_actor_image.py            # builds, pushes, prints the ref
uv run python scripts/build_actor_image.py --dry-run  # ref only
```

Pass the printed ref as `--image-ref` to every submit command below.

## Base checkpoint and caches

```bash
export RLT_CONFIG=pi05_rlt_busybox_multitask_singlearm_minmax
envs/venv-learner/bin/rlt-local assets --config-name "$RLT_CONFIG"      # normalization stats
envs/venv-openpi/bin/rlt-local cache --preset multitask                 # demo cache (RLT pretraining)
envs/venv-openpi/bin/rlt-local bc-cache --manifest manifest.json        # cache from recorded rollouts
```

Learner data lives under `RLT_DATA_ROOT` (default `./experiment-data`):
`tasks/<config>/demo`, `tasks/<config>/rlt_checkpoints/<run-id>`,
`rendezvous/`.

## Learner

Every learner publishes its endpoint in `experiment-data/rendezvous/` and the
submitter resolves `--learner-key <config>:<run-id>` from it. How the cell
reaches the machine is chosen with `--tunnel ngrok|ngrok-tcp|none`; with
`none`, pass `--public-endpoint <host>:<port>` and optionally
`--tls-cert/--tls-key` for your own ingress.

RLT (TD3), demo pretraining then online updates:

```bash
export RLT_RUN_ID="rlt-$(date -u +%Y%m%d-%H%M%S)"
envs/venv-learner/bin/rlt-local online --gpu 0 --run-id "$RLT_RUN_ID" \
  --algorithm rlt --pretrain-steps 5000 --online-steps 30000 --tunnel ngrok-tcp
```

Without a demo cache, pass `--pretrain-steps 0 --online-batch-fraction 1.0`
and start the actor with `--bc-warmup-episodes N` so the first episodes come
from π0.5 itself.

DSRL (SAC over π0.5 noise), online only. The policy server installs the
actor straight from the learner's parameter stream:

```bash
export RLT_RUN_ID="dsrl-$(date -u +%Y%m%d-%H%M%S)"
envs/venv-learner/bin/rlt-local online --gpu 0 --run-id "$RLT_RUN_ID" \
  --algorithm dsrl --pretrain-steps 0 --online-steps 500000 \
  --dsrl-noise-bound 2.5 --dsrl-noise-steps 1 --tunnel ngrok-tcp
envs/venv-openpi/bin/rlt-local policy-server --gpu 0 --config-name "$RLT_CONFIG" \
  --checkpoint <openpi-checkpoint-dir> --steering-from-learner 127.0.0.1:50051
```

`--dsrl-noise-steps 1` repeats one row of noise over the 30-step chunk;
`30` draws every step independently. Steps are gradient steps (30 per
transition).

Streaming BC and offline BC:

```bash
envs/venv-learner/bin/rlt-local streaming-bc --gpu 0 --run-id "$RLT_RUN_ID" \
  --episodes-per-round 30,30,10 --steps-per-round 500 --online-steps 30000
envs/venv-learner/bin/rlt-local bc --gpu 0 --run-id "$RLT_RUN_ID" --steps 5000
```

`--resume` continues a run from its newest checkpoint and replay snapshot.

Streaming BC for a LeRobot policy (Diffusion, DiT, ACT, ...) works the same
way: the learner fine-tunes the checkpoint on the successful episodes the
actor streams, pushes the weights to the policy server after every round and
saves a `pretrained_model` checkpoint that `checkpoint-eval` can evaluate:

```bash
envs/venv-learner/bin/rlt-local lerobot-streaming-bc --gpu 0 --run-id sbc-dit \
  --policy-path <pretrained_model dir> --task "push the green button" \
  --episodes-per-round 30,30,10 --steps-per-round 500 --online-steps 20000 \
  --base-dataset <repo_id> --base-batch-fraction 0.5 --tunnel ngrok-tcp
envs/venv-learner/bin/rlt-local policy-server --gpu 0 --config-name eval \
  --policy-type lerobot --checkpoint <pretrained_model dir> \
  --steering-from-learner 127.0.0.1:50051 --tunnel ngrok
```

The actor is submitted with `--policy-type lerobot --config-name eval`
(see Evaluation).

## Policy server

The policy server registers itself under `--config-name`; every actor and
evaluation submitted for that config sends observations to it.

```bash
envs/venv-openpi/bin/rlt-local policy-server --gpu 0 --config-name "$RLT_CONFIG" \
  --checkpoint <openpi-checkpoint-dir> --tunnel ngrok            # π0.5 for RLT and DSRL
envs/venv-learner/bin/rlt-local policy-server --gpu 0 --config-name eval \
  --policy-type lerobot --checkpoint <pretrained_model dir> \
  --steering-from-learner 127.0.0.1:50051 --tunnel ngrok           # a LeRobot policy
```

With `--steering-from-learner` the server takes the DSRL actor, or the
checkpoint-eval weight diffs, from the learner on this machine, so nothing
large crosses to the cell. `--tunnel none --public-endpoint <host>:<port>`
with `--tls-cert/--tls-key` uses your own ingress.

## Actor

```bash
uv run --env-file .env rlt-submit-actor \
  --task push_green_button --config-name "$RLT_CONFIG" \
  --checkpoint-dir volume://openpi/checkpoints/<checkpoint-name> \
  --language-instruction "push the green button" \
  --learner-key "${RLT_CONFIG}:${RLT_RUN_ID}" \
  --num-rollouts 100 --episode-timeout-s 45 \
  --armnet.require-cell cell-03 --armnet.rail 0.7 --armnet.lightbox-brightness 30 \
  --image-ref <actor-image-ref>
```

For DSRL add `--algorithm dsrl --dsrl-noise-bound 2.5 --dsrl-noise-steps 1
--exploration-scale 1.0` (`--exploration-scale` scales the actor's sampling
std, 0 is deterministic). The checkpoint is still named so the cell can fall
back to it with `--no-remote-policy`.
For RLT, `--exploration-scale` scales the fixed exploration noise and
`--bc-warmup-episodes N` rolls out π0.5 with noise before the head takes over.

`--armnet.variation` samples rail, camera and lighting variation per rollout
(`--armnet.variation-seed` makes it reproducible); `--armnet.rail`,
`--armnet.lightbox-brightness` and `--armnet.camera-aim` set them by hand.
`--record-dataset` uploads every rollout as a LeRobot dataset to the Hub.

## Evaluation

Freeze the actor of a run and evaluate it on matched scenes, optionally
against unsteered π0.5 (`--include-base`):

```bash
uv run --env-file .env rlt-freeze-checkpoint --local --config-name "$RLT_CONFIG" \
  --run-id "$RLT_RUN_ID" --step latest \
  --armnet-volume-path "rlt/frozen/${RLT_CONFIG}/${RLT_RUN_ID}/frozen_actor.pt"
uv run --env-file .env rlt-submit-eval \
  --task push_green_button --config-name "$RLT_CONFIG" \
  --checkpoint-dir volume://openpi/checkpoints/<checkpoint-name> \
  --frozen-rlt-checkpoint "volume://rlt/frozen/${RLT_CONFIG}/${RLT_RUN_ID}/frozen_actor.pt" \
  --language-instruction "push the green button" --num-rollouts 10 \
  --exploration-scale 0 --image-ref <actor-image-ref>
```

The frozen file records the algorithm, so a DSRL run evaluates the same way.

LeRobot baselines trained with `lerobot-train` are evaluated checkpoint by
checkpoint. The learner pushes each checkpoint's weight diff to the policy
server; the actor rolls it out and reports success:

```bash
envs/venv-learner/bin/rlt-local checkpoint-eval --run-id eval-dit \
  --train-dir <lerobot-train output_dir> --base-checkpoint <pretrained_model dir> \
  --final-step 30000 --rollouts-per-checkpoint 10 --tunnel ngrok-tcp
envs/venv-learner/bin/rlt-local policy-server --gpu 0 --config-name eval \
  --policy-type lerobot --checkpoint <pretrained_model dir> \
  --steering-from-learner 127.0.0.1:50051 --tunnel ngrok
uv run --env-file .env rlt-submit-actor --policy-type lerobot \
  --task push_green_button --config-name eval \
  --language-instruction "push the green button" --learner-key eval:eval-dit \
  --num-rollouts 70 --image-ref <actor-image-ref>
```

To run the policy on the cell instead, upload it and pass
`--no-remote-policy --checkpoint-dir volume://<path> --upload-checkpoint <pretrained_model dir>`.

Results go to `experiment-data/checkpoint-eval/<run-id>/checkpoint_success.jsonl`.

## Modal learner

```bash
uv run modal setup
uv run modal secret create armnet-rlt-auth RLT_LEARNER_AUTH_TOKEN="$TOKEN"
uv run modal secret create huggingface-secret HF_TOKEN="$HF_TOKEN"
uv run --env-file .env modal run src/armnet_rlt/modal_cache.py::build_multitask_cache
uv run --env-file .env modal run --detach src/armnet_rlt/modal_app.py::main \
  --config-name "$RLT_CONFIG" --run-id "$RLT_RUN_ID" --embodiment so101   # add --resume to continue
uv run --env-file .env modal run --detach src/armnet_rlt/modal_app.py::streaming_bc \
  --config-name "$RLT_CONFIG" --run-id "$RLT_RUN_ID"
uv run modal app stop -y <app-id>
```

The Modal learner registers its endpoint in the Modal Dict `RLT_MODAL_RENDEZVOUS`
instead of the local rendezvous directory; `rlt-submit-actor` resolves both.
`rlt-freeze-checkpoint` without `--local` reads the Modal volume.

## Configuration

| Variable | Meaning |
|---|---|
| `ARMNET_API_KEY` | Armnet API key for submitting jobs and building images |
| `HF_TOKEN` | Hugging Face token (secret creation only) |
| `RLT_HF_SECRET` | Name of the Armnet secret holding the Hugging Face token |
| `RLT_LEARNER_AUTH_TOKEN` | Bearer token of the local learner; equals the `armnet-rlt-auth` secret |
| `RLT_DATA_ROOT` | Local learner data root (default `./experiment-data`) |
| `RLT_CELL_CAMERA_AIM` | Camera aim per cell, `cell-03=top.tilt=0.1661;cell-01=...` |
| `RLT_MODAL_VOLUME`, `RLT_MODAL_RENDEZVOUS`, `RLT_MODAL_AUTH_SECRET` | Modal resource names |
| `OPENPI_REPO`, `OPENPI_REVISION` | OpenPI fork with the `pi0_rl` model |

Camera aim values are 0-1 of each pan/tilt axis's travel. A per-cell default
keeps every run of a task on the same view; `--armnet.camera-aim ''` disables
it. Rail position (0-1 of the usable rail) and lightbox brightness (0-100 %)
are checked against the cell's published envelope before a job is queued.

## Development

```bash
uv run pytest
RLT_CHECKPOINT=<openpi-checkpoint-dir> envs/venv-openpi/bin/python scripts/dsrl_gpu_check.py
```
