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

## Quickstart

```bash
scripts/setup.sh                                # once: environments, .env, secrets, base model, actor image
CELL=<cell-id> scripts/run.sh dsrl              # train: rlt, dsrl or dit
CELL=<cell-id> scripts/eval.sh <run-id>         # evaluate: 7 conditions x 3 seeds x 5 rollouts
```

`setup.sh` asks for an Armnet API key and a Hugging Face token with read access
to the AutoRLBench models, and writes everything else to `.env` itself. It needs
a GPU for the RLT demo cache (`--skip-cache` skips it) and Docker for the actor
image; without Docker it prints the command to build the image on another machine. The cell reaches your machine through ngrok by default: install
it and run `ngrok config add-authtoken <token>` once, or set `PUBLIC_HOST`,
`TLS_CERT` and `TLS_KEY` in `.env` for your own TLS ingress.

`run.sh` starts the policy server and the learner on this machine's GPU (`GPU=<index>`
picks one), submits the actor to `CELL`, resubmits it after a cell or safety stop, and
stops everything on Ctrl-C. Logs go to `experiment-data/logs/<run-id>/`.

| Option | Meaning |
|---|---|
| `--model <dir or HF repo>` | Starting policy: an OpenPI π0.5 checkpoint for `rlt`/`dsrl`, a LeRobot `pretrained_model` for `dit` |
| `--policy-type <type>` | `dit` without `--model`: LeRobot policy trained on the demos first (`multi_task_dit`, `diffusion`, `smolvla`, ...) |
| `--steps N` | Learner steps (`rlt` 30000, `dsrl` 400000, `dit` 1800) |
| `--std X`, `--trip-penalty X` | Exploration std (0.5) and safety-trip reward (`rlt` 5, `dsrl` 2) |
| `--resume <run-id>` | Continue a run from its newest checkpoint |
| `--learner-args "..."`, `--actor-args "..."` | Extra flags for the learner or `rlt-submit-actor` |

`eval.sh` takes an `rlt`/`dsrl` run id (newest step, or `--step N`) or a LeRobot
`pretrained_model` directory, and prints success and safety trips per condition.
`--conditions`, `--seeds` and `--rollouts` narrow it. Results go to
`experiment-data/eval/<label>-<condition>/results.jsonl`; a rerun skips seeds already
there. `WANDB_PROJECT=<project>` logs learners to Weights & Biases (`wandb login` first).

### Changing the models

| What | Where |
|---|---|
| RLT actor and critic (TD3 head, residual, layer sizes) | `src/armnet_rlt/networks.py`; sizes and defaults in `RLTNetworkConfig`, `src/armnet_rlt/config.py` |
| DSRL noise actor and critics (SAC) | `src/armnet_rlt/dsrl.py`; layer sizes in `DSRL_HIDDEN_DIMS`, `src/armnet_rlt/config.py` |
| Learner hyperparameters (learning rates, UTD, batch size) | RLT defaults in `RLTConfig`, `src/armnet_rlt/config.py`; DSRL overrides in `src/armnet_rlt/learner_config.py` |
| DiT or another LeRobot policy | its config in `third_party/lerobot/src/lerobot/policies/<type>/configuration_*.py`, or `--policy-type` |
| The base VLA | `--model` with another OpenPI checkpoint that has the RL-token head |

The actor image contains `src/armnet_rlt`, so run `scripts/setup.sh` again after
changing it; it rebuilds the image only when the code changed.

The sections below describe the individual commands the scripts run.

## Install

Requirements: Python 3.12, [`uv`](https://docs.astral.sh/uv/), Docker (image
build), an Armnet API key, a Hugging Face token.

```bash
cp .env.example .env            # fill in ARMNET_API_KEY, HF_TOKEN, RLT_HF_SECRET
uv sync --locked                # rlt-submit-actor, rlt-submit-eval, rlt-freeze-checkpoint
scripts/setup_local.sh learner  # envs/venv-learner: rlt-local bc, streaming-bc, online
scripts/setup_local.sh openpi   # envs/venv-openpi: rlt-local cache, bc-cache, policy-server (π0.5)
scripts/setup_local.sh lerobot  # envs/venv-lerobot: lerobot-train, lerobot-streaming-bc,
                                #   checkpoint-eval, policy-server --policy-type lerobot
```

With conda instead of `uv sync`:

```bash
conda create -n armnet-rlt python=3.12 -y && conda activate armnet-rlt
pip install uv && UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync --locked
set -a; source .env; set +a     # then drop `uv run` / `uv run --env-file .env` below
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
uv run hf download AutoRLBench/pi05_busybox_minmax_step5k_rltoken \
  --local-dir checkpoints/pi05_rltoken                                  # π0.5 with the RL token
envs/venv-learner/bin/rlt-local assets --config-name "$RLT_CONFIG"      # normalization stats
envs/venv-openpi/bin/rlt-local cache --preset multitask                 # demo cache (RLT pretraining)
envs/venv-openpi/bin/rlt-local bc-cache --manifest manifest.json        # cache from recorded rollouts
```

`checkpoints/pi05_rltoken` is the base policy of RLT and DSRL: π0.5 trained
on the busybox tasks (step 5000) with an RL-token encoder and decoder.
The LeRobot demo dataset is `AutoRLBench/busybox_push_green_button_426x240`
(39 teleoperated demos at 426x240).

Learner data lives under `RLT_DATA_ROOT` (default `./experiment-data`):
`tasks/<config>/demo`, `tasks/<config>/rlt_checkpoints/<run-id>`,
`rendezvous/`.

## Learner

Every learner publishes its endpoint in `experiment-data/rendezvous/` and the
submitter resolves `--learner-key <config>:<run-id>` from it. How the cell
reaches the machine is chosen with `--tunnel ngrok|ngrok-tcp|none` (default
`ngrok`); with `none`, pass `--public-endpoint <host>:<port>` and optionally
`--tls-cert/--tls-key` for your own ingress.

RLT (TD3), demo pretraining then online updates:

```bash
export RLT_RUN_ID="rlt-$(date -u +%Y%m%d-%H%M%S)"
envs/venv-learner/bin/rlt-local online --gpu 0 --run-id "$RLT_RUN_ID" \
  --online-steps 30000 --tunnel ngrok-tcp
```

Without a demo cache, pass `--pretrain-steps 0 --online-batch-fraction 1.0`
and start the actor with `--bc-warmup-episodes N` so the first episodes come
from π0.5 itself.

Residual RLT outputs `reference + 0.25 * tanh(head)` with the head initialised
to zero, so training starts exactly at the π0.5 chunk, and its critic sees
`10 * (action - reference)`:

```bash
envs/venv-learner/bin/rlt-local online --gpu 0 --run-id "$RLT_RUN_ID" \
  --online-steps 30000 --tunnel ngrok-tcp \
  --residual-actor --critic-action-residual
```

`--residual-scale` and `--critic-action-gain` change the two factors. Pass the
same flags with `--resume`; a resume with different residual flags stops with an
error. They apply to `--algorithm rlt` only. The actor needs
no flag: residual mode travels with the policy weights and frozen checkpoints.

DSRL (SAC over π0.5 noise), online only. The policy server installs the
actor straight from the learner's parameter stream:

```bash
export RLT_RUN_ID="dsrl-$(date -u +%Y%m%d-%H%M%S)"
envs/venv-learner/bin/rlt-local online --gpu 0 --run-id "$RLT_RUN_ID" \
  --algorithm dsrl --pretrain-steps 0 --online-steps 500000 --tunnel ngrok-tcp
envs/venv-openpi/bin/rlt-local policy-server --gpu 0 \
  --checkpoint checkpoints/pi05_rltoken --steering-from-learner 127.0.0.1:50051
```

Defaults: `--dsrl-noise-steps 5` (five noise rows, the last repeated over
the 30-step chunk), `--dsrl-noise-bound 1.5`. Steps are gradient steps (30
per transition).

Streaming BC and offline BC:

```bash
envs/venv-learner/bin/rlt-local streaming-bc --gpu 0 --run-id "$RLT_RUN_ID" \
  --episodes-per-round 30,30,10 --steps-per-round 500 --online-steps 30000
envs/venv-learner/bin/rlt-local bc --gpu 0 --run-id "$RLT_RUN_ID"
```

`--resume` continues a run from its newest checkpoint and replay snapshot.

Iterative BC of a multi-task DiT runs end to end from one script:

```bash
CELL=<cell-id> scripts/run.sh dit
```

It trains the starting policy on the demo dataset (300 `lerobot-train`
steps; reused once it exists), starts the LeRobot policy server and the
learner, and submits the actor until the learner finishes. The learner
fine-tunes on the successful episodes the actor streams: each round is 30, 30,
then 20 rollouts (the last entry repeats) followed by 300 learner steps, half
of every batch from the demo dataset; every round saves a `pretrained_model`
checkpoint of the EMA weights (decay 0.998) and pushes it to the policy
server. A run ends at 1800 steps. Logs go to `experiment-data/logs/<run-id>`.
`--actor-args` takes extra actor flags.

## Policy server

The policy server registers itself under `--config-name`; every actor and
evaluation submitted for that config sends observations to it.

```bash
envs/venv-openpi/bin/rlt-local policy-server --gpu 0 \
  --checkpoint checkpoints/pi05_rltoken                          # π0.5 for RLT and DSRL
envs/venv-lerobot/bin/rlt-local policy-server --gpu 0 --config-name eval \
  --policy-type lerobot --checkpoint <pretrained_model dir> \
  --steering-from-learner 127.0.0.1:50051                          # a LeRobot policy
```

With `--steering-from-learner` the server takes the DSRL actor, or the
checkpoint-eval weight diffs, from the learner on this machine, so nothing
large crosses to the cell. `--tunnel none --public-endpoint <host>:<port>`
with `--tls-cert/--tls-key` uses your own ingress. Run one policy server per
port: gRPC binds with `SO_REUSEPORT`, so a second server on the same port
starts without error and silently takes part of the traffic.

## Actor

```bash
uv run --env-file .env rlt-submit-actor \
  --task push_green_button --config-name "$RLT_CONFIG" \
  --checkpoint-dir volume://openpi/checkpoints/pi05_rltoken \
  --upload-checkpoint checkpoints/pi05_rltoken \
  --language-instruction "push the green button" \
  --learner-key "${RLT_CONFIG}:${RLT_RUN_ID}" \
  --num-rollouts 100 --armnet.require-cell <cell-id> \
  --image-ref <actor-image-ref>
```

`--armnet.require-cell <cell-id>` runs the job only on that cell: the job
waits in the Armnet queue until the cell is free, and the actor stops at
startup if it lands anywhere else. A comma-separated list accepts any of the
cells listed. Without the flag the queue picks any free cell.

`--upload-checkpoint` copies the base checkpoint to the Armnet volume once;
later jobs read the volume copy (the cell loads it with `--no-remote-policy`).
For DSRL add `--algorithm dsrl` (`--exploration-scale`, default 0.5, scales
the actor's sampling std; 0 is deterministic). For RLT, `--exploration-scale`
scales the fixed exploration noise and
`--bc-warmup-episodes N` rolls out π0.5 with noise before the head takes over.

`--armnet.variation` samples rail, camera and lighting variation per rollout
(`--armnet.variation-seed` makes it reproducible); `--armnet.rail`,
`--armnet.lightbox-brightness` and `--armnet.camera-aim` set them by hand.
Defaults: rail 0.7, brightness 30, 30 s episodes, every rollout uploaded as
a LeRobot dataset to the Hub (`--no-record-dataset` turns it off).
`--use-rerun` opens a live Rerun viewer.

## Evaluation

Freeze the actor of a run and evaluate it on matched scenes, optionally
against unsteered π0.5 (`--include-base`):

```bash
uv run --env-file .env rlt-freeze-checkpoint --local --config-name "$RLT_CONFIG" \
  --run-id "$RLT_RUN_ID" \
  --armnet-volume-path "rlt/frozen/${RLT_CONFIG}/${RLT_RUN_ID}/frozen_actor.pt"
uv run --env-file .env rlt-submit-eval \
  --task push_green_button --config-name "$RLT_CONFIG" \
  --checkpoint-dir volume://openpi/checkpoints/pi05_rltoken \
  --frozen-rlt-checkpoint "volume://rlt/frozen/${RLT_CONFIG}/${RLT_RUN_ID}/frozen_actor.pt" \
  --language-instruction "push the green button" --num-rollouts 10 \
  --image-ref <actor-image-ref>
```

The frozen file records the algorithm, so a DSRL run evaluates the same way.

LeRobot baselines trained with `lerobot-train` are evaluated checkpoint by
checkpoint. The learner pushes each checkpoint's weight diff to the policy
server; the actor rolls it out and reports success:

```bash
envs/venv-lerobot/bin/rlt-local checkpoint-eval --run-id eval-dit \
  --train-dir <lerobot-train output_dir> --base-checkpoint <pretrained_model dir> \
  --final-step 30000 --tunnel ngrok-tcp
envs/venv-lerobot/bin/rlt-local policy-server --gpu 0 --config-name eval \
  --policy-type lerobot --checkpoint <pretrained_model dir> \
  --steering-from-learner 127.0.0.1:50051
uv run --env-file .env rlt-submit-actor --policy-type lerobot \
  --task push_green_button --config-name eval \
  --language-instruction "push the green button" --learner-key eval:eval-dit \
  --num-rollouts 70 --image-ref <actor-image-ref>
```

To run the policy on the cell instead, upload it and pass
`--no-remote-policy --checkpoint-dir volume://<path> --upload-checkpoint <pretrained_model dir>`.

Results go to `experiment-data/checkpoint-eval/<run-id>/checkpoint_success.jsonl`.
`--checkpoint <pretrained_model dir>` (repeatable) evaluates those directories
instead of a training run; with one checkpoint whose copy the cell loads as
`--checkpoint-dir`, nothing has to be pushed.

### ID / OOD settings

In-distribution training and evaluation vary only the rail, over 0.65-0.75:

```bash
--armnet.variation-rail --no-armnet.variation-lighting --no-armnet.variation-cameras \
--variation-curriculum --variation-scale-start 0.2 --variation-scale-min 0.2 \
--variation-scale-max 0.2 --variation-frontier-fraction 0
```

Each out-of-distribution condition adds one change to the ID flags, so the
rail is still drawn per episode from the variation seed. The flags go to
`rlt-submit-actor` (LeRobot policies) and `rlt-submit-eval`
(frozen RLT and DSRL actors):

| Axis | Flag | Levels |
|---|---|---|
| Rail | `--variation-rail-range` | `0.60,0.65`, `0.75,0.80` |
| Brightness | `--armnet.lightbox-brightness` | 15, 5 |
| Top camera tilt | `--armnet.camera-aim top.tilt=` | 0.2567, 0.3472 |
| Action noise | `--action-noise-std` | k = 0.05, 0.15 |

`--variation-rail-range` draws each episode's rail uniformly from the range.
Action noise std is k times the per-joint std of the demo actions, one value
per joint (k = 0.05: `0.57,2.40,2.05,0.26,1.27,0.52`; k = 0.15:
`1.71,7.19,6.14,0.79,3.82,1.55`).

Evaluation runs every condition with seeds 1042, 2042 and 3042
(`--armnet.variation-seed S --action-noise-seed S`), 10 deterministic
rollouts per seed, for every policy.

## Modal learner

```bash
uv run modal setup
uv run modal secret create armnet-rlt-auth RLT_LEARNER_AUTH_TOKEN="$TOKEN"
uv run modal secret create huggingface-secret HF_TOKEN="$HF_TOKEN"
uv run --env-file .env modal run src/armnet_rlt/modal_cache.py::build_multitask_cache
uv run --env-file .env modal run --detach src/armnet_rlt/modal_app.py::main \
  --config-name "$RLT_CONFIG" --run-id "$RLT_RUN_ID" --embodiment so101   # add --resume to continue; --residual-actor --critic-action-residual for residual RLT
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
| `HF_TOKEN` | Hugging Face token: model downloads and the Armnet secret |
| `RLT_HF_SECRET` | Name of the Armnet secret holding the Hugging Face token |
| `RLT_AUTH_SECRET` | Name of the Armnet secret holding `RLT_LEARNER_AUTH_TOKEN` (default `armnet-rlt-auth`) |
| `IMAGE_REF` | Actor image the scripts submit (written by `scripts/setup.sh`) |
| `CELL`, `GPU` | Cell and local GPU index for `scripts/run.sh` and `scripts/eval.sh` |
| `PUBLIC_HOST`, `TLS_CERT`, `TLS_KEY` | Your own TLS ingress instead of ngrok; ports in `scripts/lib.sh` |
| `WANDB_PROJECT` | Weights & Biases project for the learners (optional) |
| `RLT_LEARNER_AUTH_TOKEN` | Bearer token of the local learner; equals the `RLT_AUTH_SECRET` secret |
| `RLT_DATA_ROOT` | Local learner data root (default `./experiment-data`) |
| `RLT_CELL_CAMERA_AIM` | Camera aim per cell, `cell-01=top.tilt=0.5;cell-02=...` |
| `RLT_MODAL_VOLUME`, `RLT_MODAL_RENDEZVOUS`, `RLT_MODAL_AUTH_SECRET` | Modal resource names |
| `OPENPI_REPO`, `OPENPI_REVISION` | OpenPI fork with the `pi0_rl` model |

Camera aim values are 0-1 of each pan/tilt axis's travel. A per-cell default
keeps every run of a task on the same view; `--armnet.camera-aim ''` disables
it. Rail position (0-1 of the usable rail) and lightbox brightness (0-100 %)
are checked against the cell's published envelope before a job is queued.

## Development

```bash
uv run pytest
RLT_CHECKPOINT=checkpoints/pi05_rltoken envs/venv-openpi/bin/python scripts/dsrl_gpu_check.py
```
