# armnet-rlt

## Website: **[leehakho.github.io/armnet-rlt](https://leehakho.github.io/armnet-rlt/)**

Leaderboard, rules, how to submit and every evaluation episode on video.

A real-robot reinforcement learning benchmark on [Armnet](https://armnet.dev) SO-101 cells.
Every method trains online for 400 robot episodes on one busybox task (push the green
button) and is evaluated on fixed in-distribution (ID) and out-of-distribution (OOD) scenes.

```bash
git clone https://github.com/LeeHakHo/armnet-rlt && cd armnet-rlt
```

The learner and the policy server (π0.5 plus the RL actor, or a LeRobot policy) run on your
GPU or on Modal. The cell runs the actor job: it executes the chunks the policy server
returns and sends the episodes to the learner, over gRPC at one TLS address.

## Requirements

- Linux, Python 3.12, [`uv`](https://docs.astral.sh/uv/).
- An [Armnet](https://armnet.dev) account and API key
  ([client docs](https://docs.armnet.dev/packages/client/)).
- A Hugging Face token with write access to your namespace (episode recordings) and pull
  requests on the dataset `AutoRLBench/armnet-leaderboard` (submissions).
- One of
  - an NVIDIA GPU machine the cell reaches at one TLS address: ngrok (default; run
    `ngrok config add-authtoken <token>` once), Tailscale Funnel (`TUNNEL=tailscale`; Funnel
    enabled for the tailnet) or your own ingress (`PUBLIC_HOST`, `TLS_CERT`, `TLS_KEY`, raw
    TCP from `PUBLIC_HOST:PUBLIC_PORT` to `FRONT_PORT`);
  - a [Modal](https://modal.com) account ([Running on Modal](#running-on-modal)).
- Docker, only to build the actor image; without it, set `IMAGE_REF` in `.env` to an image
  built elsewhere (`uv run python scripts/build_actor_image.py` prints the ref).

## Setup

```bash
scripts/setup.sh                 # own GPU
BACKEND=modal scripts/setup.sh   # Modal
```

Safe to rerun; finished steps are skipped. It

1. creates `.env` from `.env.example` and asks for `ARMNET_API_KEY` and `HF_TOKEN`;
2. runs `uv sync`, builds `envs/venv-learner`, `envs/venv-openpi`, `envs/venv-lerobot` and
   installs the plugins under `plugins/`;
3. creates `RLT_LEARNER_AUTH_TOKEN` and the Armnet secrets `armnet-rlt-auth-$USER` and
   `huggingface-token-$USER` (existing secrets are never overwritten), and sets
   `RECORD_REPO=<your HF user>/armnet`;
4. downloads the base π0.5 checkpoint (`--model`, default
   `AutoRLBench/pi05_rlt_busybox_multitask_singlearm_minmax_step5k`), its normalization
   stats and the RLT demo cache (`--skip-cache` skips the cache);
5. builds and pushes the actor image and writes `IMAGE_REF` (`--skip-image` skips it).

The actor image is built from `src/`, `third_party/lerobot`, `pyproject.toml`, `README.md` and
`Dockerfile.actor`; rerun `scripts/setup.sh` after changing them. Plugins are not in the image.

## Train

```bash
CELL=cell-03 scripts/run.sh dsrl                           # DSRL
CELL=cell-03 scripts/run.sh rlt                            # RLT
CELL=cell-03 scripts/run.sh dit                            # iterative DiT
CELL=cell-03 scripts/run.sh rl --algorithm residual_td3    # any RL algorithm plugin
```

Each run trains 400 episodes and evaluates the checkpoint after 100, 200, 300 and 400.
Evaluation rollouts are not training episodes. Ctrl-C stops everything, the cell job
included.

| Method | What it is |
|---|---|
| `dsrl` | π0.5 frozen; a SAC actor picks the initial noise of its flow sampler. 5 noise rows, bound 1.5, trip penalty 2, 400000 learner steps |
| `rlt` | TD3 head on π0.5's RL token, joint state and reference chunk; outputs `reference + 0.1 * tanh(head)`. Pretrains on the demo cache, trip penalty 5, 30000 learner steps |
| `dit` | `multi_task_dit` trained on the 39 demos (300 steps), then BC on its own successes in rounds of 20 episodes and 300 steps, half of each batch from the demos |
| `rl` | `--algorithm <name>`: an installed plugin, `rlt`, `dsrl`, or `module:Class` |

DSRL with a [Robometer](https://github.com/jesbu1/robometer/tree/armnet-tiled) progress
reward: serve
[`AutoRLBench/robometer-4b-lora-busybox-demo39`](https://huggingface.co/AutoRLBench/robometer-4b-lora-busybox-demo39)
on this machine, then add the reward plugin.

```bash
git clone -b armnet-tiled https://github.com/jesbu1/robometer && cd robometer && uv sync
uv pip install --python .venv/bin/python transformers==5.3.0
hf download AutoRLBench/robometer-4b-lora-busybox-demo39 --local-dir ckpt
.venv/bin/python robometer/evals/eval_server.py model_path="'$PWD/ckpt'" \
  num_gpus=1 batch_size=4 server_url=127.0.0.1 server_port=8010
CELL=cell-03 scripts/run.sh dsrl --reward-plugin armnet_rlt.robometer_reward:shape
```

Its settings (`ROBOMETER_URL`, `ROBOMETER_WEIGHT`, ...) are in
`src/armnet_rlt/robometer_reward.py`; `reward_models/robometer/` rebuilds the model. With
`BACKEND=modal`, set `ROBOMETER_URL` to an address the Modal container can reach.

Options:

| Option | Meaning |
|---|---|
| `--algorithm <name>` | rl: entry-point name or `module:Class` |
| `--model <dir or HF repo>` | π0.5 base for rlt/dsrl/rl, or a LeRobot `pretrained_model` for dit |
| `--policy-type <type>` | dit: LeRobot policy trained from the demos (`multi_task_dit`) |
| `--reward-plugin M:F` | rlt/dsrl/rl: rewrite each episode's rewards with function F of module M |
| `--resume <run-id>` | Continue a run from its newest checkpoint |
| `--leaderboard NAME`, `--leaderboard-tag TAG`, `--leaderboard-code URL`, `--leaderboard-notes TEXT` | Submit after the last evaluation ([Leaderboard](#leaderboard)) |

| Output | Path |
|---|---|
| Logs | `experiment-data/logs/<run-id>/` |
| Checkpoints | `experiment-data/tasks/<config>/rlt_checkpoints/<run-id>` (rlt, dsrl, rl), `experiment-data/lerobot-sbc/<run-id>` (dit) |
| Evaluations | `experiment-data/eval/<run-id>-ep<episodes>/results.jsonl` |

## Evaluate

```bash
CELL=cell-03 scripts/eval.sh <run-id>                  # newest step of an rlt/dsrl/rl run
CELL=cell-03 scripts/eval.sh <run-id> --step 20000
CELL=cell-03 scripts/eval.sh <pretrained_model dir>    # a LeRobot policy
```

`eval.sh` runs the `run.sh` protocol on one checkpoint: the base model, a policy trained
elsewhere, debugging. `--model` is the base the run trained on; `--label NAME` writes to
`experiment-data/eval/<NAME>/`. Conditions already in `results.jsonl` are skipped, so an
interrupted evaluation continues. Its results cannot be submitted to the leaderboard.

Protocol, per scene seed 1042, 2042 and 3042:

| Condition | Rollouts | Rail | Light | Top camera tilt |
|---|---|---|---|---|
| ID | 10 | 0.65-0.75 | 30 % | the cell's aim |
| OOD | 10 | 0.60-0.65 or 0.75-0.80 | 40 % for 5, 20 % for 5 | the cell's aim + 0.1358 |

30 ID + 30 OOD deterministic rollouts per checkpoint, 30 s episodes, a safety trip counts as a
failure. Training episodes use the ID scenes. cell-03 aims the top camera at tilt 0.1661;
cell-01 at pan 0.405, tilt 0.175, with rails 0.10 lower, which gives the same view.

## Leaderboard

```bash
CELL=cell-01 scripts/run.sh <method> --leaderboard "My Method" --leaderboard-tag v1 --leaderboard-code <repo url>
CELL=cell-03 scripts/run.sh <method> --leaderboard "My Method" --leaderboard-tag v1 --leaderboard-code <repo url>
```

- One full `run.sh` run on each of cell-01 and cell-03: 400 training episodes, evaluated
  after 100, 200, 300 and 400 (the defaults), each evaluation 30 ID + 30 OOD over 3 seeds,
  30 s episodes, safety trip = failure.
- Rules: same task, cells and protocol; at most 400 training episodes per cell; no training
  on evaluation episodes. Extra offline data (demonstrations, reward models) is allowed; say
  so in `--leaderboard-notes`.
- After the last evaluation the run opens a pull request on `AutoRLBench/armnet-leaderboard`
  with the recorded evaluation datasets of each checkpoint and the training successes per
  100 episodes.
- Recordings must be public: `RECORD_REPO` set, `RECORD` and `RECORD_PUBLIC` not 0. A run is
  recorded to one LeRobot dataset `<RECORD_REPO>_<run-id>`, all three cameras and one label
  per episode.
- An entry is a name plus `--leaderboard-tag`. Use the same tag on both cells; a new
  submission with the same name, tag and cell replaces the earlier one.
  `--leaderboard-code` marks the entry open source.
- `eval.sh` results cannot be submitted. `run.sh --leaderboard` checks the settings before
  it uses any cell time.

The scorer (`scripts/leaderboard.py score`, maintainers) recounts every checkpoint from the
episode labels and merges valid submissions. A method is listed once both cells are in; ID
and OOD each take the best checkpoint, summed over the cells. To submit a finished run:

```bash
envs/venv-learner/bin/python scripts/leaderboard.py submit --name "<method>" --run <run-id> \
  --cell cell-03 [--tag TAG] [--code URL] [--notes TEXT]
uv run --extra modal python scripts/leaderboard.py submit ...    # BACKEND=modal
```

## Robot interface

| | |
|---|---|
| Robot | SO-101, single arm, 20 Hz |
| Cameras | `top`, `front`, `wrist`; recorded at 426x240 RGB |
| State, action | 6 absolute joint targets: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll` in degrees, `gripper` 0-100 |
| π0.5 | 224x224 images; a 30-step chunk relative to the current joints, gripper absolute; the cell executes the first 10 steps |
| LeRobot policies | `observation.state`, `observation.images.{top,front,wrist}` resized to the policy's shapes; return absolute joint targets |
| Episode end | success, 30 s timeout or a safety trip |

## Your own policy

A LeRobot policy trained by iterative BC (the `dit` flow):

1. Copy `plugins/lerobot_policy_my_policy` to `plugins/lerobot_policy_<name>`.
2. Rename `my_policy` / `MyPolicy` everywhere: the package directory, `name` and `packages`
   in `pyproject.toml`, the module file names, the type in
   `@PreTrainedConfig.register_subclass`, `<Name>Config`, `<Name>Policy` and
   `make_<name>_pre_post_processors`.
3. Write the network in `modeling_<name>.py` (`__init__`, `_predict`, the loss in
   `forward`) and its settings in `configuration_<name>.py`.
4. `scripts/setup.sh`, then `CELL=cell-03 scripts/run.sh dit --policy-type <name>`.

Plugins are installed with `--no-deps`: list your dependencies in `pyproject.toml` and install
them into the environment yourself, e.g. `uv pip install --python envs/venv-lerobot <pkg>`
(`envs/venv-learner` and `envs/venv-openpi` for RL algorithms).

## Your own RL algorithm

| Example | What it is |
|---|---|
| `plugins/rl_algorithm_example` (`residual_td3`) | on top of π0.5: residual MLP on π0.5's chunk, TD3 |
| `plugins/rl_algorithm_from_scratch` (`pixel_sac`) | no π0.5: CNN on the three cameras and the joints, SAC |

Copy one to `plugins/rl_algorithm_<name>`, rename the package directory, `name`, `packages`
and the `armnet_rlt.algorithms` entry point in `pyproject.toml`, and implement the interface
in `src/armnet_rlt/algorithm.py` (rlt and dsrl in `src/armnet_rlt/algorithms/` use it too):

| Part | Runs on | Does |
|---|---|---|
| `Algorithm.make_actor()` / `build_actor(spec, config)` | learner, policy server | the actor, an `nn.Module` |
| `Actor.noise(obs, explore)` | policy server, before π0.5 | optional initial noise for π0.5's sampler |
| `Actor.act(obs, base_chunk, explore)` | policy server | `ActorOutput(chunk [L, 6], extras)`: absolute joint targets |
| `Algorithm.update(batch)` | learner | one learner step; returns metrics |
| `Algorithm.actor_state()` | learner | actor weights pushed to the policy server |
| `state_dict()` / `load_state_dict()` | learner | checkpoint contents (`--resume`) |
| `pretrain(demo_batch)`, `relabel(episode)` | learner | optional: demo pretraining (`uses_demos`), reward relabelling |

Class attributes: `needs_images` puts the camera images in `ActorInput.images` and in
batches; `uses_base = False` skips π0.5 (`rl_token` and `base_chunk` are then None and
`image_size` sets the image size). A batch has `state` / `next_state` (`rl_token` [B, 2048],
`proprioception` [B, 6], `reference_action` [B, 180]), `action` [B, L * 6], `reward`,
`done`, `extras`, and with `needs_images` `images` / `next_images`. `explore=False` is the
evaluation mode. Hyperparameters live in the plugin. The episode loop, rewards, replay buffer,
checkpoints, transport and the cell's safety logic are shared.

Offline check, no robot or GPU: trains both examples on synthetic transitions and serves
their actors (add your package to the `installed` fixture to cover it).

```bash
uv run pytest tests/test_rl_plugins.py
```

Then on the robot:

```bash
scripts/setup.sh                                          # installs plugins/rl_algorithm_*
CELL=cell-03 scripts/run.sh rl --algorithm <name>
CELL=cell-03 scripts/eval.sh <run-id>
```

`--algorithm module:Class` runs a class on the Python path without installing it.

## Running on Modal

`BACKEND=modal` runs the learner and the policy server on Modal; cell jobs are still
submitted from your machine and every command above works the same. GPU time is billed;
`RLT_MODAL_GPU` sets the type (`L40S`).

```bash
BACKEND=modal scripts/setup.sh        # once; writes BACKEND=modal to .env
CELL=cell-03 scripts/run.sh dsrl
```

`setup.sh` installs the `modal` extra, runs `modal setup` if needed, creates the Modal
secrets, deploys `src/armnet_rlt/modal_app.py` and prepares the base checkpoint and demo
cache on the volume `armnet-rlt-data`. Each `run.sh` and `eval.sh` redeploys the app with
your code and `plugins/`. Checkpoints and server logs stay on the volume
(`uv run --extra modal modal volume ls armnet-rlt-data tasks/`); cell logs and results are in
`experiment-data/` locally. A Modal call ends after 24 h; continue with `--resume <run-id>`.
To stop a container left by a script that died:

```bash
uv run --extra modal python -m armnet_rlt.modal_backend stop experiment-data/logs/<run-id>/train.call
```

## Watching live with Rerun

```bash
CELL=cell-03 scripts/run.sh dsrl --actor-args "--use-rerun"
```

Streams camera frames, joint state and actions to a [Rerun](https://rerun.io) viewer on this
machine, one recording per episode. Without a display it serves the web viewer on port 9090
(data on 9876). Recording to `RECORD_REPO` does not depend on it.

## Settings

Environment or `.env`; the environment takes precedence.

| Variable | Default | Meaning |
|---|---|---|
| `ARMNET_API_KEY`, `HF_TOKEN` | | Armnet API key; Hugging Face token |
| `RLT_AUTH_SECRET`, `RLT_HF_SECRET` | `armnet-rlt-auth-$USER`, `huggingface-token-$USER` | Armnet secrets with `RLT_LEARNER_AUTH_TOKEN` and `HF_TOKEN` |
| `RLT_LEARNER_AUTH_TOKEN` | written by `setup.sh` | Token shared by the actor and the learner |
| `IMAGE_REF` | written by `setup.sh` | Actor image |
| `CELL`, `GPU` | | `cell-01` or `cell-03`; local GPU index |
| `BACKEND` | `local` | `local` or `modal` |
| `TUNNEL`, `TAILSCALE` | `ngrok`, `tailscale` | `ngrok` or `tailscale` (Funnel); the tailscale CLI |
| `PUBLIC_HOST`, `TLS_CERT`, `TLS_KEY` | | Your own TLS ingress |
| `PUBLIC_PORT` | `443` | Public port with `PUBLIC_HOST` or Funnel |
| `PUBLIC_POLICY_PORT` | | `PUBLIC_HOST` only: a separate public port for the policy server |
| `FRONT_PORT`, `LEARNER_PORT`, `POLICY_PORT` | `50050`, `50051`, `50052` | Local ports; two runs on one machine need different ones, and with Funnel a different `PUBLIC_PORT` (`443`, `8443` or `10000`) |
| `RECORD_REPO`, `RECORD`, `RECORD_PUBLIC` | written by `setup.sh` (`<HF user>/armnet`), `1`, `1` | Episode datasets `<RECORD_REPO>_<run-id>` (`eval.sh`: `_<label>`); unset `RECORD_REPO` or `RECORD=0`: no recording; `RECORD_PUBLIC=0` private |
| `WANDB_PROJECT` | | Weights & Biases project for the learners |
| `RLT_DATA_ROOT` | `./experiment-data` | Data root |
| `CONFIG`, `TASK`, `PROMPT`, `DATASET` | `pi05_rlt_busybox_multitask_singlearm_minmax`, `push_green_button`, `push the green button`, `AutoRLBench/busybox_push_green_button_426x240` | OpenPI config, task, instruction, demo dataset for `dit` (the RLT cache always uses `AutoRLBench/busybox_push_green_button_426x240`) |
| `OPENPI_REPO`, `OPENPI_REVISION` | `https://github.com/pravsels/openpi` | OpenPI fork with the `pi0_rl` model |
| `RLT_MODAL_GPU`, `RLT_MODAL_VOLUME`, `RLT_MODAL_RENDEZVOUS`, `RLT_MODAL_AUTH_SECRET`, `RLT_MODAL_HF_SECRET`, `RLT_MODAL_APP` | `L40S`, `armnet-rlt-data`, `armnet-rlt-rendezvous`, `armnet-rlt-auth`, `huggingface-secret`, `armnet-rlt` | Modal GPU type and resource names |

## Troubleshooting

- **Cell busy**: the script waits and logs every 10 minutes.
- **A run stops** (failed cell jobs, lost connection): continue it with
  `scripts/run.sh <method> --resume <run-id>`.
- **Script killed** (SSH drop; run inside `tmux`): its cell job keeps the cell. Find it with
  `uv run --env-file .env armnet jobs list` and stop it with
  `uv run --env-file .env python -c "from armnet_client import OrchestratorClient; OrchestratorClient().stop_job('<job-id>')"`.

## Time per run

3.5-7 h of training (400 episodes) and about 45-60 min per evaluation (60 rollouts), so
7-11 h per cell, plus time waiting for the cell.

## Code

| What | Where |
|---|---|
| RL algorithm interface, rlt, dsrl | `src/armnet_rlt/algorithm.py`, `src/armnet_rlt/algorithms/`, `src/armnet_rlt/policy_actor.py` |
| Networks, hyperparameters | `src/armnet_rlt/networks.py`, `src/armnet_rlt/dsrl.py`, `src/armnet_rlt/config.py`, `src/armnet_rlt/learner_config.py` |
| Scripts, evaluation protocol | `scripts/run.sh`, `scripts/eval.sh`, `scripts/lib.sh` |
| Modal | `src/armnet_rlt/modal_app.py`, `src/armnet_rlt/modal_backend.py` |
| LeRobot | `third_party/lerobot` (0.5.2) |

The scripts call `rlt-local online`, `rlt-local lerobot-streaming-bc`, `rlt-local
policy-server`, `rlt-local front`, `rlt-submit-actor`, `rlt-submit-eval` and
`rlt-freeze-checkpoint`; `--help` on each. Run one policy server per port: gRPC binds with
`SO_REUSEPORT`, so a second server on the same port starts and takes part of the traffic.

```bash
uv run pytest
```
