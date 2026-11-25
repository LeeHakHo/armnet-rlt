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

The learner and the policy server run on your GPU (`rlt-local`) or on Modal; the policy
server runs π0.5 and the RL algorithm's actor. The actor job runs on the cell, executes the chunks
the policy server returns and sends the episodes to the learner, all over gRPC. Your own
RL algorithm plugs in without changing this code (see [Your own RL algorithm](#your-own-rl-algorithm)).

## Install

Requirements:

- Linux, Python 3.12, [`uv`](https://docs.astral.sh/uv/), Docker (actor image);
- a GPU for the learner and the policy server, one of
  - your own NVIDIA GPU and a public TLS address the cell can reach: ngrok, or your own
    ingress (`PUBLIC_HOST`, `TLS_CERT`, `TLS_KEY`);
  - a [Modal](https://modal.com) account (see [Running on Modal](#running-on-modal));
- an [Armnet](https://armnet.dev) account and API key, in `.env` as `ARMNET_API_KEY`
  ([client docs](https://docs.armnet.dev/packages/client/));
- a Hugging Face token (`HF_TOKEN`) with read access to the AutoRLBench models and
  datasets (public), write access to your namespace (the cell uploads the episode
  recordings to `RECORD_REPO`), and the right to open pull requests on the dataset
  `AutoRLBench/armnet-leaderboard` (submissions).

```bash
scripts/setup.sh                 # own GPU
BACKEND=modal scripts/setup.sh   # Modal
```

`setup.sh` is safe to rerun; finished steps are skipped. It

1. creates `.env` from `.env.example` and asks for `ARMNET_API_KEY` and `HF_TOKEN`;
2. runs `uv sync`, builds `envs/venv-learner`, `envs/venv-openpi` and `envs/venv-lerobot`
   (`scripts/setup_local.sh`) and installs the plugins under `plugins/` (LeRobot
   policies into the lerobot environment, RL algorithms into the learner and openpi ones);
3. creates the learner auth token and the two Armnet secrets: `armnet-rlt-auth`
   (`RLT_AUTH_SECRET`) and `huggingface-token` (`RLT_HF_SECRET`, your `HF_TOKEN`, which the
   cell uses from there);
4. downloads the base π0.5 checkpoint (`--model`, default
   `AutoRLBench/pi05_rlt_busybox_multitask_singlearm_minmax_step5k`) and builds its normalization stats
   and the RLT demo cache (needs a GPU; `--skip-cache` skips it);
5. builds and pushes the actor image and writes `IMAGE_REF` to `.env` (`--skip-image`
   skips it; without Docker it prints the command to run elsewhere).

The cell reaches your machine through ngrok: install it and run
`ngrok config add-authtoken <token>` once, or set `PUBLIC_HOST`, `TLS_CERT` and `TLS_KEY`
for your own TLS ingress. `setup.sh` also sets `RECORD_REPO=<your HF user>/armnet`; a run's episodes are
recorded to one LeRobot dataset `<RECORD_REPO>_<run>`.

The actor image contains `src/armnet_rlt`; rerun `scripts/setup.sh` after changing it
(it rebuilds the image only when the code changed).

To rotate the Hugging Face token, replace the Armnet secret (and the Modal one) and
`HF_TOKEN` in `.env`:

```bash
uv run --env-file .env armnet secret create huggingface-token <token>
uv run --extra modal modal secret create --force huggingface-secret HF_TOKEN=<token>
```

## Robot interface

Fixed by the cell; every policy and algorithm sees the same.

| | |
|---|---|
| Robot | SO-101, single arm |
| Cameras | `top`, `front`, `wrist`; recorded at 426x240 RGB |
| Control | 20 Hz |
| State, action | 6 absolute joint targets: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll` in degrees, `gripper` 0-100 |
| π0.5 path | the policy server feeds 224x224 images; π0.5 predicts a 30-step chunk relative to the current joints; the cell executes the first 10 steps of each chunk |
| LeRobot policies | `observation.state` and `observation.images.{top,front,wrist}`, resized to the policy's configured shapes; return absolute joint targets |
| Episode end | success, 30 s timeout or a safety trip |

The exact fields an RL algorithm gets are in `AlgorithmSpec`, `src/armnet_rlt/algorithm.py`.

## Quickstart

```bash
CELL=cell-03 scripts/run.sh dsrl                # train 400 episodes, evaluate after every 100
CELL=cell-03 scripts/eval.sh <checkpoint>       # evaluate one checkpoint: 30 ID + 30 OOD rollouts
```

`run.sh` starts the policy server and the learner on this machine (`GPU=<index>` picks
the GPU; on Modal with `BACKEND=modal`) and submits the actor to `CELL` in segments of `--eval-every` episodes,
resubmitting after a cell or safety stop. After each segment the learner saves a
checkpoint and the evaluation protocol runs on it; evaluation rollouts are not training
episodes. Ctrl-C stops everything, the cell job included.

| Output | Path |
|---|---|
| Logs | `experiment-data/logs/<run-id>/` |
| Checkpoints | `experiment-data/tasks/<config>/rlt_checkpoints/<run-id>` (rlt, dsrl, rl), `experiment-data/lerobot-sbc/<run-id>` (dit) |
| Evaluations | `experiment-data/eval/<run-id>-ep<episodes>/results.jsonl` |

## Baselines

### DSRL

```bash
CELL=cell-03 scripts/run.sh dsrl
```

SAC over π0.5's noise, online only: 5 noise rows (the last repeated over the chunk),
noise bound 1.5, exploration std 0.5, trip penalty 2, 400000 learner steps. The policy
server runs the DSRL actor and takes its weights from the learner's parameter stream.

### DSRL + Robometer reward

The reward model is [Robometer-4B](https://huggingface.co/robometer/Robometer-4B) tuned with
LoRA on the same 39 demonstrations
([`AutoRLBench/robometer-4b-lora-busybox-demo39`](https://huggingface.co/AutoRLBench/robometer-4b-lora-busybox-demo39)).
Serve it on a GPU of this machine:

```bash
git clone -b armnet-tiled https://github.com/jesbu1/robometer && cd robometer && uv sync
uv pip install --python .venv/bin/python transformers==5.3.0     # the version the model was saved with
hf download AutoRLBench/robometer-4b-lora-busybox-demo39 --local-dir ckpt
.venv/bin/python robometer/evals/eval_server.py model_path="'$PWD/ckpt'" \
  num_gpus=1 batch_size=4 server_url=127.0.0.1 server_port=8010
```

`reward_models/robometer/` rebuilds this model from the demos.

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
critic sees the residual. RLT pretrains on the demo cache that `scripts/setup.sh` builds
from the 39 green-button demos (`AutoRLBench/busybox_push_green_button_426x240`) through
the base (`experiment-data/tasks/<config>/demo/rlt_demo_cache.pt`); without it, training starts
online right away. Exploration std 0.5, trip penalty 5, 30000 learner steps. The head
runs in the policy server, which takes its weights from the learner.

### run.sh options

`CELL=<cell-id> scripts/run.sh <rlt|dsrl|dit> [options]`, or
`scripts/run.sh rl --algorithm <name> [options]` for any RL algorithm; `--help` prints them.

| Option | Meaning |
|---|---|
| `--algorithm <name>` | rl: the algorithm's entry-point name (`rlt`, `dsrl`, a plugin) or `module:Class` |
| `--episodes N` | Training episodes (400) |
| `--eval-every N` | Evaluate after every N training episodes (100); 0 disables |
| `--skip-eval` | Save the checkpoints but do not evaluate; they are listed in `experiment-data/logs/<run-id>/checkpoints.txt` |
| `--model <dir or HF repo>` | Starting policy: an OpenPI π0.5 checkpoint for rlt/dsrl/rl (default `AutoRLBench/pi05_rlt_busybox_multitask_singlearm_minmax_step5k`), a LeRobot `pretrained_model` for dit |
| `--policy-type <type>` | dit: LeRobot policy trained from the demos (`multi_task_dit`) |
| `--steps N` | rlt/dsrl/rl learner step cap, which also sets the learning-rate schedule (rlt 30000, dsrl and rl 400000) |
| `--std X` | rlt/dsrl/rl exploration std (0.5), the actor's `exploration_scale` |
| `--trip-penalty X` | rlt/dsrl/rl: a safety-tripped episode ends X success-failure reward gaps below a failure (rlt 5, dsrl and rl 2) |
| `--run-id ID` | Run name (`<method>-<UTC time>`; rl: `<algorithm>-<UTC time>`) |
| `--resume ID` | Continue an rlt/dsrl/rl run ID from its newest checkpoint |
| `--reward-plugin M:F` | rlt/dsrl/rl: rewrite each episode's rewards with function F of module M |
| `--learner-args "..."`, `--actor-args "..."` | Extra flags for the learner (`rlt-local online` or `lerobot-streaming-bc`) and for `rlt-submit-actor` |
| `--leaderboard NAME`, `--leaderboard-tag TAG`, `--leaderboard-code URL`, `--leaderboard-notes TEXT` | Submit to the leaderboard after the last evaluation |

## Your own RL algorithm

Three routes, from least to most code of yours:

- **A LeRobot policy** trained by iterative BC: see [Iterative DiT](#iterative-dit).
- **An RL algorithm on top of π0.5**: your actor gets π0.5's RL token and chunk and
  returns the chunk to execute; it may also choose π0.5's initial flow noise, as DSRL
  does. Example: `plugins/rl_algorithm_example` (`residual_td3`, a residual MLP on π0.5's
  chunk, TD3).
- **An RL algorithm from scratch**: no π0.5 at all; your actor gets the three camera
  images and the joint positions. Example: `plugins/rl_algorithm_from_scratch`
  (`pixel_sac`, a CNN on the cameras, SAC).

rlt and dsrl are built the same way (`src/armnet_rlt/algorithms/`). Copy an example to
`plugins/rl_algorithm_<name>`, rename the package and the entry point in its
`pyproject.toml`, and implement the interface in `src/armnet_rlt/algorithm.py`:

| Part | Where it runs | What it does |
|---|---|---|
| `Algorithm.make_actor()` / `build_actor(spec, config)` | learner, policy server | The actor, an `nn.Module` |
| `Actor.noise(obs, explore)` | policy server, before π0.5 | Optional initial noise for π0.5's flow sampler; leave it out to let π0.5 draw its own |
| `Actor.act(obs, base_chunk, explore)` | policy server, after π0.5 | `ActorOutput(chunk [L, 6], extras)`: absolute joint targets the cell executes; `extras` come back in batches |
| `Algorithm.update(batch)` | learner | One learner step; returns metrics (`critic_loss`, `actor_loss`, ...) |
| `Algorithm.actor_state()` | learner | Actor weights, pushed to the policy server every `--push-interval-s` |
| `state_dict()` / `load_state_dict()` | learner | Checkpoint contents; `--resume` restores them |
| `pretrain(demo_batch)`, `relabel(episode)` | learner | Optional: demo-cache pretraining (with `uses_demos`); reward relabelling |

Class attributes: `needs_images` (False) puts the top, front and wrist images in
`ActorInput.images` and in batches; `uses_base` (True) runs π0.5, False skips it (then
`rl_token` and `base_chunk` are None and `image_size` sets the image size);
`updates_per_transition`, `uses_demos`, `keeps_partial_chunks`. `explore=False` is the
deterministic evaluation mode; `obs.exploration_scale` is the run's `--std`.

A batch has `state` / `next_state` (`rl_token` [B, 2048], `proprioception` [B, 6],
`reference_action` [B, 180]: π0.5's chunk), `action` [B, L * 6] (the chunk as executed),
`reward`, `done`, `extras`, and with `needs_images` `images` / `next_images` (view to
uint8 [B, H, W, 3]). Without π0.5, `rl_token` and `reference_action` are [B, 0]. Images
are the ones the policy server saw (π0.5's 224x224, or the algorithm's `image_size`),
sent as JPEG: about 10 KB more per transition at 96x96, 30 KB at 224x224. Options
reach the algorithm as `config.algorithm_options` through
`--learner-args "--algorithm-option KEY=VALUE"`.

The episode loop, rewards and `--reward-plugin`, the replay buffer, checkpoints, metrics,
transport and the cell's safety logic stay the same for every algorithm.

```bash
scripts/setup.sh                                           # installs plugins/rl_algorithm_*
CELL=cell-03 scripts/run.sh rl --algorithm residual_td3    # 400 episodes, evaluated every 100
CELL=cell-03 scripts/run.sh rl --algorithm pixel_sac --leaderboard pixel-sac
CELL=cell-03 scripts/eval.sh <run-id>                      # evaluate a run's newest checkpoint
```

`--algorithm module:Class` runs a class that is on the Python path without installing
it. `rlt-local policy-server --algorithm <name>` serves an algorithm on its own
(`--checkpoint` is the π0.5 checkpoint, not needed without the base;
`--actor-checkpoint <frozen .pt>` serves a frozen actor for evaluation). The actor image
does not contain your plugin: the cell only executes the chunks, so no rebuild is needed.

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

`eval.sh` evaluates one checkpoint on these scenes on its own: the base model, a LeRobot
`pretrained_model`, a checkpoint trained elsewhere, debugging. `run.sh` runs the same
evaluation every 100 episodes. An `eval.sh` result alone cannot be submitted to the
leaderboard.

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

- full `run.sh` runs on both cell-01 and cell-03: 400 training episodes and the
  evaluations at 100, 200, 300 and 400 (`--episodes 400 --eval-every 100`, the defaults),
  one submission per cell; `eval.sh` results are not accepted;
- the default `EVAL_*` settings;
- public recorded evaluations (`RECORD_REPO` set, `RECORD` not 0, `RECORD_PUBLIC` not 0).

`run.sh --leaderboard` checks these before it uses any cell time.

The maintainers' scorer (`scripts/leaderboard.py score`) recounts every checkpoint from the
episode labels in the recorded datasets and rejects submissions that do not follow the
protocol. A method is listed once both cells are in; ID and OOD each take the best
checkpoint, summed over the two cells. An entry is a name plus `--tag`; a new submission
with the same name, tag and cell replaces the earlier one.

## Running on Modal

With `BACKEND=modal` the learner and the policy server run on [Modal](https://modal.com)
instead of your GPU; the cell jobs are still submitted from your machine. Every command
above works the same.

Prerequisites: a Modal account (GPU time is billed; see Modal's pricing for the GPU
type) and the other [Install](#install) requirements. `RLT_MODAL_GPU` sets the GPU type
(default `L40S`; the learner and π0.5 share it).

```bash
BACKEND=modal scripts/setup.sh                    # once; writes BACKEND=modal to .env
CELL=cell-03 scripts/run.sh dsrl                  # same options as on your GPU
CELL=cell-03 scripts/run.sh rl --algorithm residual_td3 --leaderboard residual-td3
CELL=cell-03 scripts/eval.sh <run-id>
```

`setup.sh` with `BACKEND=modal`

1. installs the `modal` extra (`uv sync --extra modal`) and runs `modal setup` if Modal
   is not logged in; it builds no `envs/`;
2. creates the Modal secrets `armnet-rlt-auth` (`RLT_LEARNER_AUTH_TOKEN`, and
   `WANDB_API_KEY` if set) and `huggingface-secret` (`HF_TOKEN`);
3. deploys the app (`src/armnet_rlt/modal_app.py`; the first image build takes a while)
   and downloads the base checkpoint, the normalization stats and the demo cache to the
   Modal volume `armnet-rlt-data`;
4. builds the actor image locally with Docker, as without Modal.

Each `run.sh` or `eval.sh` redeploys the app with your current code and plugins
(`plugins/` is installed into the images). One container runs the run's learner and
policy server, the same `rlt-local` commands as on your GPU, with one Modal TLS endpoint
each; the endpoints are in the Modal Dict `armnet-rlt-rendezvous`. Each evaluation gets a
second container with a policy server. `run.sh dit` runs the same way in a LeRobot image.

| What | Where |
|---|---|
| Checkpoints, replay buffer, learner and policy-server logs, demo cache, base checkpoint | the volume, laid out as `experiment-data/` (`checkpoints/` for base checkpoints) |
| Cell job logs, `results.jsonl`, `rollout_metrics.jsonl`, frozen actors | `experiment-data/` on your machine, as without Modal (the scripts copy what they read from the volume) |

Ctrl-C stops the Modal containers with the cell job. A run whose script died without
cleaning up keeps its container (up to Modal's 24 h limit); stop it, then continue the
run:

```bash
uv run --extra modal python -m armnet_rlt.modal_backend stop experiment-data/logs/<run-id>/train.call
CELL=cell-03 scripts/run.sh dsrl --resume <run-id>
```

A Modal call ends after 24 h; `--resume` continues the run from its newest checkpoint.
`uv run --extra modal modal volume ls armnet-rlt-data tasks/` lists the volume. A
`--reward-plugin` reward server must be reachable from Modal (`ROBOMETER_URL`;
`ROBOMETER_*` variables are passed to the container).

## Configuration

Set in the environment or in `.env`; the environment takes precedence.

| Variable | Meaning |
|---|---|
| `ARMNET_API_KEY`, `HF_TOKEN` | Armnet API key; Hugging Face token (models, episode datasets, an Armnet secret) |
| `RLT_HF_SECRET`, `RLT_AUTH_SECRET` | Armnet secrets holding `HF_TOKEN` and `RLT_LEARNER_AUTH_TOKEN` (`huggingface-token`, `armnet-rlt-auth`) |
| `RLT_LEARNER_AUTH_TOKEN` | Bearer token shared by the actor and the learner (written by `setup.sh`) |
| `IMAGE_REF` | Actor image (written by `setup.sh`) |
| `CELL`, `GPU` | Cell to run on (`cell-01` or `cell-03`; `run.sh` and `eval.sh` need it) and local GPU index |
| `BACKEND` | `local` (default) or `modal` ([Running on Modal](#running-on-modal)) |
| `RLT_MODAL_GPU`, `RLT_MODAL_VOLUME`, `RLT_MODAL_RENDEZVOUS`, `RLT_MODAL_AUTH_SECRET`, `RLT_MODAL_HF_SECRET`, `RLT_MODAL_APP` | Modal GPU type (`L40S`) and resource names (`armnet-rlt-data`, `armnet-rlt-rendezvous`, `armnet-rlt-auth`, `huggingface-secret`, `armnet-rlt`) |
| `PUBLIC_HOST`, `TLS_CERT`, `TLS_KEY` | Your own TLS ingress instead of ngrok; ports `PUBLIC_LEARNER_PORT` (443), `PUBLIC_POLICY_PORT` (8443) |
| `LEARNER_PORT`, `POLICY_PORT` | Local ports (50051, 50052) |
| `RECORD_REPO`, `RECORD_PUBLIC`, `RECORD` | Episodes are recorded (three cameras, one label per episode in `meta/armnet_rollouts.jsonl`) in one public LeRobot dataset `<RECORD_REPO>_<run>` per run (`eval.sh`: `<RECORD_REPO>_<label>`); `RECORD_PUBLIC=0` keeps it private, `RECORD=0` turns recording off |
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

## Watching a run live

The cell job can stream its camera frames, joint state and actions to a
[Rerun](https://rerun.io) viewer on the machine that submitted it, one recording per
episode, named by its outcome when it ends. The scripts turn it off; pass it to the
training actor with

```bash
CELL=cell-03 scripts/run.sh dsrl --actor-args "--use-rerun"
```

or add `--use-rerun` to `rlt-submit-actor` / `rlt-submit-eval` directly. It needs a
display for the viewer window; without one it serves the web viewer on port 9090 (data on
9876). It does not work with `--detach`, and episode recording (`RECORD_REPO`) does not
depend on it.

## Code

| What | Where |
|---|---|
| RL algorithm interface | `src/armnet_rlt/algorithm.py`; rlt and dsrl in `src/armnet_rlt/algorithms/`; the actor in the policy server in `src/armnet_rlt/policy_actor.py` |
| RLT actor and critic | `src/armnet_rlt/networks.py`; sizes and defaults in `RLTNetworkConfig`, `src/armnet_rlt/config.py` |
| DSRL noise actor and critics | `src/armnet_rlt/dsrl.py`; layer sizes in `DSRL_HIDDEN_DIMS`, `src/armnet_rlt/config.py` |
| Learner hyperparameters | RLT defaults in `RLTConfig`, `src/armnet_rlt/config.py`; DSRL overrides in `src/armnet_rlt/learner_config.py` |
| LeRobot policies, RL algorithm plugins | `third_party/lerobot` (upstream 0.5.2), `plugins/` |
| Scripts | `scripts/run.sh`, `scripts/eval.sh`; helpers and the evaluation protocol in `scripts/lib.sh` |
| Modal backend | `src/armnet_rlt/modal_app.py` (images, functions), `src/armnet_rlt/modal_backend.py` (container supervisor, local commands) |

The scripts call these commands, which also run on their own (`--help` on each):

- `rlt-local online`, `rlt-local lerobot-streaming-bc`, `rlt-local checkpoint-eval`: learners;
- `rlt-local policy-server`: serves π0.5 with an RL algorithm's actor, or a LeRobot policy
  (`--policy-type lerobot`); `--steering-from-learner` takes the actor or LeRobot weights
  from a local learner, `--actor-checkpoint` a frozen actor.
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
