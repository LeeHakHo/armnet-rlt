# armnet-rlt

A real-robot reinforcement learning benchmark on Armnet SO-101 cells. Every method
trains online for the same budget of robot episodes on one busybox task (push the
green button), and its checkpoints are evaluated on a fixed set of in-distribution (ID)
and out-of-distribution (OOD) scenes. Results go on a public
[leaderboard](https://leehakho.github.io/armnet-rlt/#/leaderboard).

Baselines:

- **DSRL**: π0.5 stays frozen; a SAC actor picks the initial noise of its flow-matching
  sampler, so every executed chunk is one π0.5 produced.
- **DSRL + Robometer reward**: DSRL with the progress estimate of a
  [Robometer](https://github.com/jesbu1/robometer/tree/armnet-tiled) reward model added
  to the reward.
- **Iterative DiT**: a multi-task DiT trained on the demos, then fine-tuned by behaviour
  cloning on its own successful rollouts. Any LeRobot policy can take its place.
- **RLT**: a TD3 head reads π0.5's RL token and joint state and outputs a residual on
  π0.5's action chunk.

The learner and the policy server run on your GPU (`rlt-local`). The actor runs on the
cell as an Armnet job and talks to both over gRPC.

## Install

Requirements: Linux with an NVIDIA GPU, Python 3.12, [`uv`](https://docs.astral.sh/uv/),
Docker (actor image), an Armnet API key and a Hugging Face token with read access to
the AutoRLBench models.

```bash
scripts/setup.sh
```

`setup.sh` is safe to rerun; finished steps are skipped. It

1. creates `.env` from `.env.example` and asks for `ARMNET_API_KEY` and `HF_TOKEN`;
2. runs `uv sync`, builds `envs/venv-learner`, `envs/venv-openpi` and `envs/venv-lerobot`
   (`scripts/setup_local.sh`) and installs the policy plugins under `plugins/`;
3. creates the learner auth token and the two Armnet secrets;
4. downloads the base π0.5 checkpoint (`--model`, default
   `AutoRLBench/pi05_rlt_busybox_multitask_singlearm_minmax_step5k`) and builds its normalization stats
   and the RLT demo cache (needs a GPU; `--skip-cache` skips it);
5. builds and pushes the actor image and writes `IMAGE_REF` to `.env` (`--skip-image`
   skips it; without Docker it prints the command to run elsewhere).

The cell reaches your machine through ngrok: install it and run
`ngrok config add-authtoken <token>` once, or set `PUBLIC_HOST`, `TLS_CERT` and `TLS_KEY`
for your own TLS ingress. `setup.sh` also sets `RECORD_REPO=<your HF user>/armnet`, where
the cell records every training and evaluation episode.

The actor image contains `src/armnet_rlt`; rerun `scripts/setup.sh` after changing it
(it rebuilds the image only when the code changed).

## Quickstart

```bash
CELL=cell-03 scripts/run.sh dsrl                # train 400 episodes, evaluate after every 100
CELL=cell-03 scripts/eval.sh <checkpoint>       # evaluate one checkpoint: 30 ID + 30 OOD rollouts
```

`run.sh` starts the policy server and the learner on this machine (`GPU=<index>` picks
the GPU) and submits the actor to `CELL` in segments of `--eval-every` episodes,
resubmitting after a cell or safety stop. After each segment the learner saves a
checkpoint and the evaluation protocol runs on it; evaluation rollouts are not training
episodes. Ctrl-C stops everything, the cell job included.

| Output | Path |
|---|---|
| Logs | `experiment-data/logs/<run-id>/` |
| Checkpoints | `experiment-data/tasks/<config>/rlt_checkpoints/<run-id>` (rlt, dsrl), `experiment-data/lerobot-sbc/<run-id>` (dit) |
| Evaluations | `experiment-data/eval/<run-id>-ep<episodes>/results.jsonl` |

## Baselines

### DSRL

```bash
CELL=cell-03 scripts/run.sh dsrl
```

SAC over π0.5's noise, online only: 5 noise rows (the last repeated over the chunk),
noise bound 1.5, exploration std 0.5, trip penalty 2, 400000 learner steps. The policy
server takes the DSRL actor from the learner's parameter stream.

### DSRL + Robometer reward

Serve the Robometer busybox checkpoint on a GPU of this machine:

```bash
git clone -b armnet-tiled https://github.com/jesbu1/robometer && cd robometer && uv sync
uv pip install --python .venv/bin/python transformers==5.3.0     # the version the checkpoint was saved with
hf download jesbu1/robometer-4b-fft-busybox-tiled --local-dir ckpt
sed -i 's/^  use_unsloth: true$/  use_unsloth: false/' ckpt/config.yaml   # Unsloth fails with a dtype error
.venv/bin/python robometer/evals/eval_server.py model_path="'$PWD/ckpt'" \
  num_gpus=1 batch_size=4 server_url=127.0.0.1 server_port=8010
```

Then train with the reward plugin:

```bash
CELL=cell-03 scripts/run.sh dsrl --reward-plugin armnet_rlt.robometer_reward:shape
```

The actor sends each episode's top, front and wrist frames (`--episode-frames`); the
learner adds `ROBOMETER_WEIGHT` (1.0) times each transition's progress gain to its reward.
`ROBOMETER_URL`, `ROBOMETER_FRAMES`, `ROBOMETER_LOG` and the other settings are listed in
`src/armnet_rlt/robometer_reward.py`. `--reward-plugin <module>:<function>` takes any
function with the same signature, for `rlt` as well.

### Iterative DiT

```bash
CELL=cell-03 scripts/run.sh dit
```

Without `--model` it first trains a `multi_task_dit` on the demo dataset
`AutoRLBench/busybox_push_green_button_426x240` for 300 `lerobot-train` steps (reused once
it exists). The learner then runs in rounds of 20 rollouts and 300 learner steps on the
successful episodes, half of each batch from the demos, and pushes each round's EMA
weights (decay 0.998) to the policy server. `--policy-type diffusion`, `smolvla` or your
own plugin replaces the DiT; `--model <pretrained_model dir>` starts from a policy trained
elsewhere.

To add a policy, copy `plugins/lerobot_policy_my_policy` (a small CNN and MLP that
predict an action chunk) to `plugins/lerobot_policy_<name>` and rename `my_policy` /
`MyPolicy` inside: the type in `@PreTrainedConfig.register_subclass`, `<Name>Config`,
`<Name>Policy` and `make_<name>_pre_post_processors`. Write the network in
`modeling_<name>.py` (`__init__`, `_predict`, the loss in `forward`) and its settings in
`configuration_<name>.py`, then run `scripts/setup.sh` and
`scripts/run.sh dit --policy-type <name>`.

### RLT

```bash
CELL=cell-03 scripts/run.sh rlt
```

The base (the default for every method) is the step-5k π0.5 checkpoint with an RL-token
encoder and decoder trained on that VLA; DSRL and iterative DiT do not use the RL token. The TD3
head outputs `reference + 0.1 * tanh(head)`, so training starts at π0.5's chunk, and its
critic sees the residual. RLT pretrains on the demo cache from `scripts/setup.sh`
(`experiment-data/tasks/<config>/demo/rlt_demo_cache.pt`); without it, training starts
online right away. Exploration std 0.5, trip penalty 5, 30000 learner steps.

### run.sh options

`CELL=<cell-id> scripts/run.sh <rlt|dsrl|dit> [options]`; `--help` prints them.

| Option | Meaning |
|---|---|
| `--episodes N` | Training episodes (400) |
| `--eval-every N` | Evaluate after every N training episodes (100); 0 disables |
| `--skip-eval` | Save the checkpoints but do not evaluate; they are listed in `experiment-data/logs/<run-id>/checkpoints.txt` |
| `--model <dir or HF repo>` | Starting policy: an OpenPI π0.5 checkpoint for rlt/dsrl (default `AutoRLBench/pi05_rlt_busybox_multitask_singlearm_minmax_step5k`), a LeRobot `pretrained_model` for dit |
| `--policy-type <type>` | dit: LeRobot policy trained from the demos (`multi_task_dit`) |
| `--steps N` | rlt/dsrl learner step cap, which also sets the learning-rate schedule (rlt 30000, dsrl 400000) |
| `--std X` | rlt/dsrl exploration std (0.5) |
| `--trip-penalty X` | rlt/dsrl: a safety-tripped episode ends X success-failure reward gaps below a failure (rlt 5, dsrl 2) |
| `--run-id ID` | Run name (`<method>-<UTC time>`) |
| `--resume ID` | Continue rlt/dsrl run ID from its newest checkpoint |
| `--reward-plugin M:F` | rlt/dsrl: rewrite each episode's rewards with function F of module M |
| `--learner-args "..."`, `--actor-args "..."` | Extra flags for the learner (`rlt-local online` or `lerobot-streaming-bc`) and for `rlt-submit-actor` |
| `--leaderboard NAME`, `--leaderboard-tag TAG`, `--leaderboard-code URL`, `--leaderboard-notes TEXT` | Submit to the leaderboard after the last evaluation |

## Evaluation protocol

`scripts/eval.sh <checkpoint>` and every evaluation inside `run.sh` run the same
protocol. For each scene seed 1042, 2042 and 3042:

| Condition | Rollouts | Rail | Light | Top camera tilt |
|---|---|---|---|---|
| ID | 10 | 0.65-0.75 | 30 % | the cell's aim |
| OOD | 10 | 0.60-0.65 or 0.75-0.80 | 40 % for 5, 20 % for 5 | the cell's aim + 0.1358 |

That is 30 ID and 30 OOD rollouts per checkpoint, on the same scenes for every policy.
Rollouts are deterministic, episodes end after 30 s, and a safety trip counts as a
failure. Training episodes use the ID scenes.

The top camera aim is tilt 0.1661 on cell-03 (OOD 0.3019). cell-01 uses rails 0.10
lower (ID 0.55-0.65, OOD 0.50-0.55 or 0.65-0.70) and top aim pan 0.405, tilt 0.175, which
gives the same view of the busybox. Rail positions are fractions of the usable rail, camera
aim fractions of each pan/tilt axis's travel.

The protocol runs as one cell job that sets rail, light and tilt per episode
(`--eval-schedule`); on cell-01, whose aim includes a pan, it runs one job per condition.
Conditions already in `results.jsonl` are skipped, so an interrupted evaluation continues
where it stopped.

`eval.sh` takes an rlt/dsrl run id (newest step, or `--step N`; `--model` must be the
base the run trained on) or a LeRobot `pretrained_model` directory, and prints successes
per condition and seed. `--label NAME` writes to `experiment-data/eval/<NAME>/`.

## Leaderboard

A run with `--leaderboard "<method name>"` opens a pull request on the
`AutoRLBench/armnet-leaderboard` dataset after its last evaluation. The submission lists
the recorded evaluation datasets of each checkpoint and the run's training successes per
100 episodes. To submit a finished run by hand:

```bash
envs/venv-learner/bin/python scripts/leaderboard.py submit --name "<method>" --run <run-id> \
  --cell cell-03 [--tag TAG] [--code URL] [--notes TEXT]
```

A submission needs

- runs on both cell-01 and cell-03 with `--episodes 400 --eval-every 100` (checkpoints at
  100, 200, 300 and 400 episodes), one submission per cell;
- recorded evaluations (`RECORD_REPO` set, `RECORD` not 0).

The maintainers' scorer (`scripts/leaderboard.py score`) recounts every checkpoint from the
episode labels in the recorded datasets and rejects submissions that do not follow the
protocol. A method is listed once both cells are in; ID and OOD each take the best
checkpoint, summed over the two cells. An entry is a name plus `--tag`; a new submission
with the same name, tag and cell replaces the earlier one.

## Configuration

Set in the environment or in `.env`; the environment takes precedence.

| Variable | Meaning |
|---|---|
| `ARMNET_API_KEY`, `HF_TOKEN` | Armnet API key; Hugging Face token (models, episode datasets, an Armnet secret) |
| `RLT_HF_SECRET`, `RLT_AUTH_SECRET` | Armnet secrets holding `HF_TOKEN` and `RLT_LEARNER_AUTH_TOKEN` (`huggingface-token`, `armnet-rlt-auth`) |
| `RLT_LEARNER_AUTH_TOKEN` | Bearer token shared by the actor and the learner (written by `setup.sh`) |
| `IMAGE_REF` | Actor image (written by `setup.sh`) |
| `CELL`, `GPU` | Cell to run on (empty: any free cell) and local GPU index |
| `PUBLIC_HOST`, `TLS_CERT`, `TLS_KEY` | Your own TLS ingress instead of ngrok; ports `PUBLIC_LEARNER_PORT` (443), `PUBLIC_POLICY_PORT` (8443) |
| `LEARNER_PORT`, `POLICY_PORT` | Local ports (50051, 50052) |
| `RECORD_REPO`, `RECORD_PUBLIC`, `RECORD` | Episodes are recorded (three cameras, one label per episode in `meta/armnet_rollouts.jsonl`) as public datasets `<RECORD_REPO>_<name>`; `RECORD_PUBLIC=0` keeps them private, `RECORD=0` turns recording off |
| `WANDB_PROJECT` | Log learners to this Weights & Biases project |
| `RLT_DATA_ROOT` | Data root (`./experiment-data`) |
| `RLT_CELL_CAMERA_AIM` | Top-camera aim per cell (`cell-03=top.tilt=0.1661;cell-01=top.pan=0.405,top.tilt=0.175`); empty keeps each cell's own aim |
| `RLT_CELL_RAIL` | ID rail range per cell (`cell-01=0.55,0.65`; other cells 0.65-0.75); the OOD ranges shift with it |
| `EVAL_SEEDS`, `EVAL_ID_ROLLOUTS`, `EVAL_OOD_ROLLOUTS` | Evaluation seeds and rollouts per seed (`1042 2042 3042`, 10, 10) |
| `EVAL_LEVELS`, `EVAL_ONE_JOB`, `EVAL_LIGHT` | `id` or `ood` evaluates only that condition; `0` runs one job per condition; one light level for every condition |
| `CONFIG`, `TASK`, `PROMPT`, `DATASET` | OpenPI config, task, language instruction and demo dataset |
| `OPENPI_REPO`, `OPENPI_REVISION` | OpenPI fork with the `pi0_rl` model (`scripts/setup_local.sh`, `Dockerfile.actor`) |

Rail and lightbox brightness are checked against the cell's published envelope before a
job is queued.

## Code

| What | Where |
|---|---|
| RLT actor and critic | `src/armnet_rlt/networks.py`; sizes and defaults in `RLTNetworkConfig`, `src/armnet_rlt/config.py` |
| DSRL noise actor and critics | `src/armnet_rlt/dsrl.py`; layer sizes in `DSRL_HIDDEN_DIMS`, `src/armnet_rlt/config.py` |
| Learner hyperparameters | RLT defaults in `RLTConfig`, `src/armnet_rlt/config.py`; DSRL overrides in `src/armnet_rlt/learner_config.py` |
| LeRobot policies | `third_party/lerobot` (upstream 0.5.2), `plugins/` |
| Scripts | `scripts/run.sh`, `scripts/eval.sh`; helpers and the evaluation protocol in `scripts/lib.sh` |

The scripts call these commands, which also run on their own (`--help` on each):

- `rlt-local online`, `rlt-local lerobot-streaming-bc`, `rlt-local checkpoint-eval`: learners;
- `rlt-local policy-server`: serves π0.5 or a LeRobot policy (`--policy-type lerobot`);
  `--steering-from-learner` takes the DSRL actor or LeRobot weights from a local learner.
  Run one server per port: gRPC binds with `SO_REUSEPORT`, so a second server on the same
  port starts without error and takes part of the traffic;
- `rlt-submit-actor`, `rlt-submit-eval`: training and evaluation cell jobs;
- `rlt-freeze-checkpoint`: exports a learner checkpoint as a frozen actor, optionally
  uploaded to an Armnet volume.

## Development

```bash
uv run pytest
RLT_CHECKPOINT=checkpoints/pi05_rltoken5k envs/venv-openpi/bin/python scripts/dsrl_gpu_check.py
```
