# Shared helpers for setup.sh, run.sh and eval.sh. Source it; do not run it.
#
# Settings (environment or .env):
#   CELL              Armnet cell to run on: cell-01 or cell-03 (run.sh and eval.sh need it)
#   IMAGE_REF         actor image (setup.sh writes it to .env)
#   GPU               GPU index for the learner and policy server (unset: visible GPU)
#   RLT_AUTH_SECRET   Armnet secret with RLT_LEARNER_AUTH_TOKEN (default armnet-rlt-auth-$USER)
#   RLT_HF_SECRET     Armnet secret with HF_TOKEN (default huggingface-token-$USER)
#   BACKEND           local (default: this machine's GPU) or modal (learner and policy
#                     server on Modal, src/armnet_rlt/modal_app.py)
#   TUNNEL            local: ngrok (default, one HTTPS endpoint) or tailscale (Funnel,
#                     raw TCP on PUBLIC_PORT; TAILSCALE is the CLI, default tailscale)
#   PUBLIC_HOST       your own TLS ingress instead of a tunnel; then also TLS_CERT, TLS_KEY
#   PUBLIC_PORT       public port with PUBLIC_HOST or tailscale (443); PUBLIC_POLICY_PORT
#                     gives the policy server its own port (two addresses)
#   FRONT_PORT        local port of the run's one endpoint (50050); it passes the calls on
#                     to LEARNER_PORT and POLICY_PORT on 127.0.0.1 (50051, 50052)
#   WANDB_PROJECT     log learners to this Weights & Biases project
#   RECORD_REPO       <hf user or org>/<prefix> (setup.sh: <your user>/armnet): the cell
#                     records training and evaluation episodes, all three cameras and a
#                     label per episode (meta/armnet_rollouts.jsonl); a run's episodes are
#                     recorded to one LeRobot dataset <prefix>_<run> (eval.sh: <prefix>_<label>),
#                     public unless RECORD_PUBLIC=0; RECORD=0 turns recording off

CALLER_PWD=$PWD
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# .env fills only variables the environment does not already set.
if [ -f .env ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)= ]] || continue
    key=${BASH_REMATCH[1]}
    [ -n "${!key:-}" ] && continue
    eval "export $line"
  done < .env
fi
# '' keeps each cell's own aim.
# Protocol per cell (fixed): top-camera aim, and the ID rail range where it is not 0.65-0.75.
export RLT_CELL_CAMERA_AIM="cell-03=top.tilt=0.1661;cell-01=top.pan=0.405,top.tilt=0.175"
export RLT_CELL_RAIL="cell-01=0.55,0.65"
# Policy server registration key, per port.
export RLT_POLICY_INSTANCE=${RLT_POLICY_INSTANCE:-${POLICY_PORT:-50052}}
UVRUN=(uv run --quiet)
[ -f .env ] && UVRUN+=(--env-file .env)

CONFIG=${CONFIG:-pi05_rlt_busybox_multitask_singlearm_minmax}
TASK=${TASK:-push_green_button}
PROMPT=${PROMPT:-push the green button}
DATA=${RLT_DATA_ROOT:-$ROOT/experiment-data}
export RLT_DATA_ROOT=$DATA
CELL=${CELL:-}
GPU=${GPU:-}
GPU_ARG=${GPU:+--gpu $GPU}
RLT_AUTH_SECRET=${RLT_AUTH_SECRET:-armnet-rlt-auth-$USER}
RLT_HF_SECRET=${RLT_HF_SECRET:-huggingface-token-$USER}
LEARNER_PORT=${LEARNER_PORT:-50051}
POLICY_PORT=${POLICY_PORT:-50052}
WANDB_ARG=${WANDB_PROJECT:+--wandb-project $WANDB_PROJECT}
BASE_REPO=AutoRLBench/pi05_rlt_busybox_multitask_singlearm_minmax_step5k
BASE_DIR=$ROOT/checkpoints/pi05_rltoken5k
# Checkpoint path passed to the cell (unused with --remote-policy).
CELL_CHECKPOINT=${CELL_CHECKPOINT:-volume://openpi/checkpoints/pi05_rltoken}
LEARNER_BIN=envs/venv-learner/bin
OPENPI_BIN=envs/venv-openpi/bin
LEROBOT_BIN=envs/venv-lerobot/bin

log() { echo "[$(basename "$0" .sh) $(date -u +%H:%M:%S)] $*"; }
die() { log "$*"; exit 1; }
py() { "${UVRUN[@]}" python - "$@"; }

BACKEND=${BACKEND:-local}
case "$BACKEND" in local|modal) ;; *) die "BACKEND must be local or modal, not $BACKEND" ;; esac
# Python and hf on this machine: envs/venv-learner, or the uv env with the modal extra.
HOSTPY=($LEARNER_BIN/python) HF=($LEARNER_BIN/hf)
if [ "$BACKEND" = modal ]; then
  export RLT_BACKEND=modal
  UVRUN+=(--extra modal)
  HOSTPY=("${UVRUN[@]}" python) HF=("${UVRUN[@]}" hf)
  GPU_ARG=""
fi

# One public address for the learner and the policy server: rlt-local front on FRONT_PORT.
# LEARNER_TUNNEL and POLICY_TUNNEL register it; front_start sets them for ngrok and tailscale.
FRONT_PORT=${FRONT_PORT:-50050}
PUBLIC_PORT=${PUBLIC_PORT:-443}
TUNNEL=${TUNNEL:-ngrok}
TAILSCALE=(${TAILSCALE:-tailscale})
FRONT=($LEARNER_BIN/rlt-local front --port $FRONT_PORT --learner-port $LEARNER_PORT --policy-port $POLICY_PORT)
FRONT_PID="" FUNNEL_ON=0
STEERING="--steering-from-learner 127.0.0.1:$LEARNER_PORT"
front_tunnels() {  # <public endpoint>
  LEARNER_TUNNEL="--tunnel none --public-endpoint $1 --public-endpoint-tls --port $LEARNER_PORT"
  POLICY_TUNNEL="--tunnel none --public-endpoint $1 --public-endpoint-tls --port $POLICY_PORT"
}
if [ "$BACKEND" = modal ]; then
  # modal_start adds the front to each container; "modal:<port>" is its Modal TLS tunnel.
  FRONT+=(--tunnel none --public-endpoint-tls --bind-host 0.0.0.0)
  front_tunnels "modal:$FRONT_PORT"
elif [ -n "${PUBLIC_HOST:-}" ]; then
  : "${TLS_CERT:?PUBLIC_HOST needs TLS_CERT}" "${TLS_KEY:?PUBLIC_HOST needs TLS_KEY}"
  if [ -n "${PUBLIC_POLICY_PORT:-}" ]; then
    FRONT=()
    LEARNER_TUNNEL="--tunnel none --public-endpoint $PUBLIC_HOST:$PUBLIC_PORT --port $LEARNER_PORT --tls-cert $TLS_CERT --tls-key $TLS_KEY"
    POLICY_TUNNEL="--tunnel none --public-endpoint $PUBLIC_HOST:$PUBLIC_POLICY_PORT --port $POLICY_PORT --tls-cert $TLS_CERT --tls-key $TLS_KEY"
    STEERING="--steering-from-learner 127.0.0.1:$LEARNER_PORT --steering-learner-tls-name $PUBLIC_HOST"
  else
    FRONT+=(--tunnel none --public-endpoint "$PUBLIC_HOST:$PUBLIC_PORT" --tls-cert "$TLS_CERT" --tls-key "$TLS_KEY")
    front_tunnels "$PUBLIC_HOST:$PUBLIC_PORT"
  fi
else
  case "$TUNNEL" in ngrok|tailscale) ;; *) die "TUNNEL must be ngrok or tailscale, not $TUNNEL" ;; esac
  [ "$TUNNEL" = tailscale ] || FRONT+=(--tunnel ngrok)
  LEARNER_TUNNEL="" POLICY_TUNNEL=""
fi

# Tailscale Funnel: PUBLIC_HOST is this machine's tailnet name, TLS_CERT and TLS_KEY its
# certificate from tailscale cert, and PUBLIC_PORT a raw TCP funnel to FRONT_PORT.
tailscale_denied() {  # <output>
  log "tailscale: $1"
  log "run these as a tailscale operator (or once: sudo tailscale set --operator=\$USER):"
  log "  ${TAILSCALE[*]} cert --cert-file $TLS_CERT --key-file $TLS_KEY $PUBLIC_HOST"
  log "  ${TAILSCALE[*]} funnel --bg --tcp $PUBLIC_PORT tcp://localhost:$FRONT_PORT"
  die "tailscale needs operator rights for cert and funnel"
}
tailscale_up() {
  local out forward
  out=$("${TAILSCALE[@]}" status --json 2>&1) || die "tailscale status failed: $out"
  PUBLIC_HOST=$(echo "$out" | python3 -c 'import json, sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))') \
    || die "tailscale status has no DNS name for this machine"
  TLS_CERT=$DATA/tls/$PUBLIC_HOST.crt TLS_KEY=$DATA/tls/$PUBLIC_HOST.key
  mkdir -p "$DATA/tls"
  if ! openssl x509 -checkend 604800 -noout -in "$TLS_CERT" >/dev/null 2>&1 || [ ! -f "$TLS_KEY" ]; then
    out=$("${TAILSCALE[@]}" cert --cert-file "$TLS_CERT" --key-file "$TLS_KEY" "$PUBLIC_HOST" 2>&1) || tailscale_denied "$out"
    chmod 600 "$TLS_KEY"
  fi
  out=$("${TAILSCALE[@]}" funnel status --json 2>&1) || die "tailscale funnel status failed: $out"
  forward=$(echo "$out" | python3 -c 'import json, sys; print(((json.load(sys.stdin).get("TCP") or {}).get(sys.argv[1]) or {}).get("TCPForward", ""))' "$PUBLIC_PORT")
  case "$forward" in
    "") out=$("${TAILSCALE[@]}" funnel --bg --tcp "$PUBLIC_PORT" "tcp://localhost:$FRONT_PORT" 2>&1) || tailscale_denied "$out"
        FUNNEL_ON=1; log "funnel $PUBLIC_HOST:$PUBLIC_PORT -> localhost:$FRONT_PORT" ;;
    localhost:$FRONT_PORT|127.0.0.1:$FRONT_PORT) ;;
    *) die "funnel port $PUBLIC_PORT goes to $forward, not localhost:$FRONT_PORT; set PUBLIC_PORT or FRONT_PORT" ;;
  esac
  FRONT+=(--tunnel none --public-endpoint "$PUBLIC_HOST:$PUBLIC_PORT" --tls-cert "$TLS_CERT" --tls-key "$TLS_KEY")
}

# Starts the front on this machine and sets LEARNER_TUNNEL and POLICY_TUNNEL to its
# public address; front_stop stops it (and a funnel front_start turned on).
front_start() {  # <log>
  [ "$BACKEND" != modal ] && [ ${#FRONT[@]} -gt 0 ] || return 0
  [ -z "${PUBLIC_HOST:-}" ] && [ "$TUNNEL" = tailscale ] && tailscale_up
  "${FRONT[@]}" > "$1" 2>&1 &
  FRONT_PID=$!
  wait_for $FRONT_PID "$1" "\[RLT\] front " front
  front_tunnels "$(sed -nE 's/^\[RLT\] front ([^ ]+) .*/\1/p' "$1" | head -1)"
  log "cell endpoint $(echo "$LEARNER_TUNNEL" | cut -d' ' -f4) (front on $FRONT_PORT)"
}
front_stop() {
  [ -z "$FRONT_PID" ] || { kill "$FRONT_PID" 2>/dev/null || true; wait "$FRONT_PID" 2>/dev/null || true; }
  FRONT_PID=""
  if [ "$FUNNEL_ON" = 1 ]; then
    "${TAILSCALE[@]}" funnel --tcp "$PUBLIC_PORT" "tcp://localhost:$FRONT_PORT" off >/dev/null 2>&1 \
      || log "could not turn off the funnel; run: ${TAILSCALE[*]} funnel --tcp $PUBLIC_PORT tcp://localhost:$FRONT_PORT off"
    FUNNEL_ON=0
  fi
}

# Modal: python -m armnet_rlt.modal_backend (see there).
modal_cli() { "${UVRUN[@]}" python -m armnet_rlt.modal_backend "$@"; }
modal_deploy() {
  log "deploying the Modal app (the first build takes a while)"
  "${UVRUN[@]}" modal deploy -m armnet_rlt.modal_app >&2 || die "modal deploy failed"
}
# Appends a process to a spec file for modal_start.
modal_proc() {  # <spec file> <name> <log> <argv...>
  local f=$1; shift
  python3 -c 'import json, sys; print(json.dumps({"name": sys.argv[1], "log": sys.argv[2], "argv": sys.argv[3:]}))' "$@" >> "$f"
}
# Spawns the front and the processes of a spec file in one Modal GPU container, waits for
# their endpoints, and writes the call id to <call file>.
modal_start() {  # <rl|lerobot> <spec file> <call file>
  local specs=$2.front bin=$LEARNER_BIN
  [ "$1" = lerobot ] && bin=$LEROBOT_BIN
  rm -f "$specs"
  modal_proc "$specs" front "$(dirname "$2")/front.log" $bin/rlt-local "${FRONT[@]:1}"
  cat "$2" >> "$specs"
  modal_cli start --image "$1" --specs "$specs" --call-file "$3" || die "Modal call failed; logs: $(dirname "$2")"
}
modal_stop() { [ -f "$1" ] && modal_cli stop "$1" || true; }

# Data files. Local: plain file tests. Modal: the paths name their copy on the volume
# (RLT_DATA_ROOT is its root, checkpoints/ its checkpoints/); data_pull copies a file
# down to its local path, data_push up.
data_exists() { if [ "$BACKEND" = modal ]; then modal_cli exists "$1"; else [ -e "$1" ]; fi; }
data_ls() { if [ "$BACKEND" = modal ]; then modal_cli ls "$1"; else ls "$1" 2>/dev/null || true; fi; }
data_pull() { [ "$BACKEND" != modal ] || modal_cli get "$1" 2>/dev/null || true; }
data_push() { [ "$BACKEND" != modal ] || modal_cli put "$1"; }
# Modal: a local directory as a volume path, uploaded to <target> unless the volume has it.
data_dir() {  # <local dir> <target>
  if [ "$BACKEND" != modal ]; then echo "$1"; return; fi
  local dir=$1
  case "$dir" in "$DATA"/*|"$ROOT"/checkpoints/*|/data/*) ;; *) dir=$2 ;; esac
  if ! data_exists "$dir"; then
    log "uploading $1 to the Modal volume" >&2
    modal_cli put-dir "$1" "$dir" >&2 || die "could not upload $1"
  fi
  echo "$dir"
}
# Runs one command to completion: on this machine, or in a Modal container.
run_cmd() {  # <rl|lerobot> <log> <argv...>
  local image=$1 log=$2; shift 2
  if [ "$BACKEND" = modal ]; then
    modal_cli run --image "$image" --log "$log" -- "$@"
  else
    env ${GPU:+CUDA_VISIBLE_DEVICES=$GPU} "$@" > "$log" 2>&1
  fi
}

# Common actor flags in ACTOR: rail 0.7, light 30, 30 s episodes, no recording, policy on
# your GPU.
actor_common() {
  : "${IMAGE_REF:?IMAGE_REF is unset; run scripts/setup.sh}"
  ACTOR=(--task "$TASK" --language-instruction "$PROMPT"
    --armnet.rail 0.7 --armnet.lightbox-brightness 30
    --episode-timeout-s 30 --robot-telemetry full --no-record-dataset --no-use-rerun
    --remote-policy --auth-secret "$RLT_AUTH_SECRET" --hf-secret "$RLT_HF_SECRET"
    --image-ref "$IMAGE_REF")
  [ -n "$CELL" ] && ACTOR+=(--armnet.require-cell "$CELL")
  return 0
}

# Recording flags for one cell job in RECORD_ARGS: its episodes are appended to the
# LeRobot dataset <RECORD_REPO>_<run>, each labelled with the KEY=VALUE pairs; public unless
# RECORD_PUBLIC=0, none when RECORD=0 or RECORD_REPO is unset.
record_args() {  # <run> [KEY=VALUE...]
  RECORD_ARGS=()
  [ "${RECORD:-1}" = 1 ] && [ -n "${RECORD_REPO:-}" ] || return 0
  local run=$1 label; shift
  RECORD_ARGS=(--record-dataset --record-dataset-repo-id "${RECORD_REPO}_$run" --record-append)
  for label in "$@"; do RECORD_ARGS+=(--record-label "$label"); done
  [ "${RECORD_PUBLIC:-1}" = 0 ] || RECORD_ARGS+=(--record-public)
}

# ID scene variation: rail drawn per episode from 0.65-0.75.
ID_ARGS="--armnet.variation-rail --no-armnet.variation-lighting --no-armnet.variation-cameras --variation-curriculum --variation-scale-start 0.2 --variation-scale-min 0.2 --variation-scale-max 0.2 --variation-frontier-fraction 0"
# A cell in RLT_CELL_RAIL draws from its own range (training and ID evaluation); RAIL_SHIFT,
# that range's centre minus 0.7, shifts the OOD rail ranges.
CELL_RAIL=$(echo "${RLT_CELL_RAIL:-}" | tr ';' '\n' | grep "^${CELL:-none}=" | cut -d= -f2 || true)
RAIL_SHIFT=0
if [ -n "$CELL_RAIL" ]; then
  ID_ARGS="$ID_ARGS --variation-rail-range $CELL_RAIL"
  RAIL_SHIFT=$(python3 -c "import sys; lo, hi = map(float, sys.argv[1].split(',')); print(round((lo + hi) / 2 - 0.7, 4))" "$CELL_RAIL")
fi

# Benchmark cells; the rail and camera-aim corrections are keyed by cell id.
CELLS="cell-01 cell-03"
require_cell() {
  case " $CELLS " in
    *" ${CELL:-none} "*) ;;
    *) die "${CELL:+unknown cell $CELL; }set CELL to cell-01 or cell-03" ;;
  esac
}

# Succeeds when CELL is available or unset.
cell_free() {
  [ -n "$CELL" ] || return 0
  CELL=$CELL py <<'PY'
import os, sys
from armnet_client import OrchestratorClient
try:
    cells = OrchestratorClient().get_cell_status().cells
except Exception as exc:
    print(f"cell status: {exc}", file=sys.stderr); sys.exit(1)
sys.exit(0 if any(c.cell_id == os.environ["CELL"] and c.status.value == "available" for c in cells) else 1)
PY
}
wait_cell() {
  local waited=0
  until cell_free; do
    [ $((waited % 600)) -eq 0 ] && log "$CELL is busy; waiting"
    sleep 60; waited=$((waited + 60))
  done
}

# Stops the cell job a submit log reports if it is still active; waits up to 15 minutes
# for it to end.
stop_cell_job() {
  local id
  id=$(grep -oE "job [0-9a-f-]{36} dispatched|job queued: [0-9a-f-]{36}" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9a-f-]{36}" || true)
  [ -n "$id" ] || return 0
  py "$id" <<'PY'
import sys, time
from armnet_client import OrchestratorClient
ACTIVE = ("running", "queued", "dispatched", "submitted")
client = OrchestratorClient()
try:
    if client.get(sys.argv[1]).status.value in ACTIVE:
        client.stop_job(sys.argv[1], reason="finished")
        deadline = time.time() + 900
        while client.get(sys.argv[1]).status.value in ACTIVE and time.time() < deadline:
            time.sleep(10)
except Exception as exc:
    print(f"stop {sys.argv[1]}: {exc}", file=sys.stderr)
PY
}

# Waits until a log contains a pattern; fails if the process exits first.
wait_for() {  # <pid> <log> <pattern> <name>
  until grep -q "$3" "$2" 2>/dev/null; do
    kill -0 "$1" 2>/dev/null || die "$4 exited; see $2"
    sleep 5
  done
}

# Local OpenPI checkpoint dir for --model: a directory as is, a Hugging Face repo
# downloaded under checkpoints/.
resolve_model() {
  local m=${1:-$BASE_REPO}
  case "$m" in /*) ;; *) [ -d "$CALLER_PWD/$m" ] && m=$CALLER_PWD/$m ;; esac
  if [ -d "$m" ]; then m=$(cd "$m" && pwd); data_dir "$m" "$ROOT/checkpoints/$(basename "$m")"; return; fi
  local dir=$ROOT/checkpoints/$(basename "$m")
  [ "$m" = "$BASE_REPO" ] && dir=$BASE_DIR
  if [ "$BACKEND" = modal ]; then
    data_exists "$dir" || run_cmd rl "$DATA/logs/download-$(basename "$m").log" \
      $LEARNER_BIN/hf download "$m" --local-dir "$dir" >&2 \
      || die "could not download $m on Modal; remove a partial copy with: uv run --extra modal modal volume rm -r ${RLT_MODAL_VOLUME:-armnet-rlt-data} checkpoints/$(basename "$dir")"
  elif [ ! -d "$dir" ]; then
    $LEARNER_BIN/hf download "$m" --local-dir "$dir" >&2 || { rm -rf "$dir"; die "could not download $m"; }
  fi
  echo "$dir"
}

# Evaluation protocol (fixed), per seed 1042, 2042, 3042: 10 ID and 10 OOD rollouts. OOD: rail 0.60-0.65 or
# 0.75-0.80, top camera tilted 0.1358 above the cell's ID tilt (0.3019 on cell-03), light
# 40 % for the first half of each seed's episodes and 20 % for the rest (the lightbox
# flickers below 15 % on cell-03 and 18 % on cell-01). Rollouts are deterministic; a safety
# trip is a failure.
EVAL_SEEDS="1042 2042 3042"
EVAL_ID_ROLLOUTS=10
EVAL_OOD_ROLLOUTS=10

# Dies unless the run follows the leaderboard protocol; lists each deviation.
leaderboard_preflight() {  # <episodes> <eval every>
  local bad=() b
  [ "$1" = 400 ] || bad+=("--episodes $1: checkpoints must be at 100, 200, 300 and 400 episodes; use 400")
  [ "$2" = 100 ] || bad+=("--eval-every $2: checkpoints must be at 100, 200, 300 and 400 episodes; use 100")
  case " $CELLS " in *" ${CELL:-none} "*) ;; *) bad+=("CELL '${CELL:-}': leaderboard cells are cell-01 and cell-03") ;; esac
  [ -z "${EVAL_LEVELS:-}" ] || bad+=("EVAL_LEVELS '$EVAL_LEVELS': the protocol evaluates ID and OOD; unset it")
  [ "${RECORD:-1}" = 1 ] || bad+=("RECORD ${RECORD}: the scorer recounts the recorded episodes; set RECORD=1")
  [ -n "${RECORD_REPO:-}" ] || bad+=("RECORD_REPO unset: the scorer recounts the recorded episodes; scripts/setup.sh sets it")
  [ "${RECORD_PUBLIC:-1}" != 0 ] || bad+=("RECORD_PUBLIC=0: the scorer reads public datasets")
  [ ${#bad[@]} -eq 0 ] && return 0
  for b in "${bad[@]}"; do log "--leaderboard: $b"; done
  die "--leaderboard: the run does not follow the protocol (${#bad[@]} problems above)"
}
EVAL_PIDS=() EVAL_CALLS=()
EVAL_JOB_LOG=""

# Freezes step STEP of an rlt/dsrl/rl run, uploads it, and prints its volume path; the
# local copy is at experiment-data/tasks/<config>/frozen/<run>/step_<STEP>.pt.
# With modal the learner checkpoint is copied down first and the frozen actor back up.
freeze_rlt() {  # <run> <step>
  local dir; dir=$(printf step_%06d "$2")
  local frozen=$DATA/tasks/$CONFIG/frozen/$1/$dir.pt
  data_pull "$DATA/tasks/$CONFIG/rlt_checkpoints/$1/$dir/checkpoint.pt"
  "${UVRUN[@]}" rlt-freeze-checkpoint --config-name "$CONFIG" --run-id "$1" --step "$2" \
    --output "$frozen" \
    --armnet-volume-path "rlt/frozen/$CONFIG/$1/$dir/frozen_actor.pt" --overwrite >&2 \
    && data_push "$frozen" >&2 \
    && echo "volume://rlt/frozen/$CONFIG/$1/$dir/frozen_actor.pt"
}

# Rollout counts per condition and scene seed over the job logs of an evaluation group:
# "<condition> <seed> <rollouts> <successes> <trips>" lines, at most <cap> rollouts in all.
# A rollout's condition is the one its scene recorded, else <default>.
_tally() {  # <logs dir> <cap> <default condition>
  { cat "$1"/job*.log 2>/dev/null || true; } | sed -E 's/\x1b\[[0-9;]*m//g' | python3 -c '
import collections, json, sys
cap, default = int(sys.argv[1]), sys.argv[2]
counts = collections.OrderedDict(); total = 0
for line in sys.stdin:
    tag = next((k for k in ("[rlt rollout]", "[act rollout]") if k in line), None)
    if tag is None or total >= cap:
        continue
    try:
        r = json.loads(line.split(tag, 1)[1])
    except ValueError:
        continue
    if r.get("policy_variant") == "base":
        continue
    v = r.get("variation") or {}
    c = "%s %s" % (v.get("condition") or default, v.get("seed", -1))
    n, s, t = counts.get(c, (0, 0, 0))
    counts[c] = (n + 1, s + bool(r.get("success")), t + (r.get("outcome") == "safety_trip"))
    total += 1
print(total)
for c, (n, s, t) in counts.items():
    print(c, n, s, t)' "$2" "$3"
}

# OOD rail ranges (0.60-0.65 and 0.75-0.80) shifted by RAIL_SHIFT.
ood_rails() {
  python3 -c "import sys; s = float(sys.argv[1]); print([[round(a + s, 4), round(b + s, 4)] for a, b in ((0.60, 0.65), (0.75, 0.80))])" "$RAIL_SHIFT"
}

# Runs the evaluation protocol on one checkpoint and prints its summary. One cell job walks
# through the levels and seeds with --eval-schedule, setting rail, light and top tilt per
# episode; EVAL_ONE_JOB=0 runs one job per level, aimed at its start.
#   kind lerobot: <checkpoint> is a pretrained_model dir. With <learner key> (a running
#     LeRobot learner) the actor sends it nothing (--eval-only); without one, a
#     checkpoint-eval learner is started per job.
#   kind rlt: <checkpoint> is a volume:// frozen actor of an rlt/dsrl/rl run; the policy
#     server loads its local copy (--actor-checkpoint) on the π0.5 <base>.
# Results: experiment-data/eval/<label>/results.jsonl, one row per condition and seed; a group
# whose conditions are all there is skipped. Uses POLICY_PORT and, without a learner key,
# LEARNER_PORT. With modal the server runs in its own container under the policy key
# <RLT_POLICY_INSTANCE>-eval, next to a run's training server; a lerobot checkpoint without
# a learner key gets a container per job, with its checkpoint-eval learner.
eval_checkpoint() {  # <kind> <checkpoint> <label> <base or learner key>
  local kind=$1 ckpt=$2 label=$3 extra=${4:-} out=$DATA/eval/$3 server ps="" key ev group n conds
  local id=$EVAL_ID_ROLLOUTS ood=$EVAL_OOD_ROLLOUTS image=rl instance=$RLT_POLICY_INSTANCE
  mkdir -p "$out"
  if [ "$kind" = lerobot ]; then
    image=lerobot
    server=($LEROBOT_BIN/rlt-local policy-server $GPU_ARG --config-name eval --policy-type lerobot
      --checkpoint "$ckpt" $POLICY_TUNNEL)
    [ -z "$extra" ] && server+=($STEERING)
  else
    # volume://rlt/frozen/<config>/<run>/<step>/frozen_actor.pt -> local copy from freeze_rlt.
    local frozen=${ckpt#volume://rlt/frozen/$CONFIG/}
    frozen=$DATA/tasks/$CONFIG/frozen/${frozen%/frozen_actor.pt}.pt
    [ -f "$frozen" ] || die "no local copy of $ckpt at $frozen"
    server=($OPENPI_BIN/rlt-local policy-server $GPU_ARG --config-name "$CONFIG" --checkpoint "$extra"
      --prompt "$PROMPT" --actor-checkpoint "$frozen" $POLICY_TUNNEL)
  fi
  if [ "$BACKEND" = modal ]; then
    instance=$RLT_POLICY_INSTANCE-eval
    if [ "$kind" != lerobot ] || [ -n "$extra" ]; then
      rm -f "$out/modal-server.jsonl"
      modal_proc "$out/modal-server.jsonl" policy-server "$out/policy-server.log" "${server[@]}"
      EVAL_CALLS+=("$out/policy-server.call")
      RLT_POLICY_INSTANCE=$instance modal_start "$image" "$out/modal-server.jsonl" "$out/policy-server.call"
    fi
  else
    "${server[@]}" > "$out/policy-server.log" 2>&1 &
    ps=$!; EVAL_PIDS+=($ps)
    wait_for $ps "$out/policy-server.log" "policy server" "eval policy server"
  fi
  log "evaluating $ckpt on ${CELL:-any cell} -> $out"
  # ID top aim of the cell from RLT_CELL_CAMERA_AIM; OOD tilt is 0.1358 above it.
  local id_tilt
  id_tilt=$(echo "${RLT_CELL_CAMERA_AIM:-}" | tr ';' '\n' | grep "^${CELL:-none}=" | grep -oE "top\.tilt=[0-9.]+" | cut -d= -f2 || true)
  local id_pan ood_tilt=0.3019
  id_pan=$(echo "${RLT_CELL_CAMERA_AIM:-}" | tr ';' '\n' | grep "^${CELL:-none}=" | grep -oE "top\.pan=[0-9.]+" | cut -d= -f2 || true)
  if [ -n "$id_tilt" ]; then
    ood_tilt=$(python3 -c "print(round($id_tilt + 0.1358, 4))")
  fi
  local groups=all
  [ "${EVAL_ONE_JOB:-1}" = 1 ] || groups="id ood"
  # A per-episode tilt recentres the pan, so a cell aimed with a pan gets one job per level.
  [ -z "$id_pan" ] || groups="id ood"
  # EVAL_LEVELS="id" or "ood" evaluates only that level; the other can be added later.
  [ -z "${EVAL_LEVELS:-}" ] || groups=$EVAL_LEVELS
  for group in $groups; do
    local entries="" seed level brightness ranges count tilt
    for level in $([ "$group" = all ] && echo id ood || echo "$group"); do
      case "$level" in
        id) lights="30:$id" ranges=null count=$id tilt=$id_tilt ;;
        ood) lights="40:$(((ood + 1) / 2)) 20:$((ood / 2))" ranges=$(ood_rails) count=$ood tilt=$ood_tilt ;;
      esac
      for seed in $EVAL_SEEDS; do
        for part in $lights; do
          brightness=${part%%:*}
          [ "${part##*:}" -gt 0 ] || continue
          entries="$entries${entries:+, }{\"condition\": \"$level\", \"seed\": $seed, \"count\": ${part##*:}, \"brightness\": $brightness, \"rail_ranges\": $ranges$([ "$group" = all ] && [ -n "$tilt" ] && echo ", \"top_tilt\": $tilt" || true)}"
        done
      done
    done
    local seeds; seeds=$(echo $EVAL_SEEDS | wc -w)
    if [ "$group" = all ]; then
      n=$(((id + ood) * seeds)); conds="id ood"
      local group_args=(--eval-schedule "[$entries]" $ID_ARGS)
    else
      n=$((count * seeds)); conds=$group
      # Without a per-episode tilt the job's camera aim sets it.
      local group_args=(--eval-schedule "[$entries]" $ID_ARGS ${tilt:+--armnet.camera-aim ${id_pan:+top.pan=$id_pan,}top.tilt=$tilt})
    fi
    local missing=""
    for c in $conds; do grep -q "\"condition\": \"$c\"" "$out/results.jsonl" 2>/dev/null || missing="$missing $c"; done
    [ -n "$missing" ] || continue
    mkdir -p "$out/$group"
    local jobs=0 done
    while :; do
      done=$(_tally "$out/$group" "$n" "$group" | sed -n 1p)
      [ "$done" -ge "$n" ] && break
      [ "$jobs" -ge 4 ] && { log "$group: stopping after $jobs jobs at $done/$n"; break; }
      jobs=$((jobs + 1))
      EVAL_JOB_LOG=$out/$group/job$(date +%s).log
      wait_cell
      local first=${EVAL_SEEDS%% *}
      local seeded=(--armnet.variation-seed "$first" --armnet.variation-episode-offset "$done"
        --action-noise-seed "$first" --timeout-seconds 7200 "${group_args[@]}")
      # A run.sh evaluation (<run>-ep<N>) records to the run's dataset, eval.sh to <label>.
      case "$label" in
        *-ep[0-9]*) record_args "${label%-ep*}" phase=eval "checkpoint_episodes=${label##*-ep}" "job=$group" ;;
        *) record_args "$label" phase=eval "checkpoint=$label" "job=$group" ;;
      esac
      seeded+=("${RECORD_ARGS[@]}")
      if [ "$kind" = lerobot ] && [ -n "$extra" ]; then
        RLT_POLICY_INSTANCE=$instance "${UVRUN[@]}" rlt-submit-actor "${ACTOR[@]}" "${seeded[@]}" --policy-type lerobot \
          --config-name eval --learner-key "$extra" --num-rollouts "$((n - done))" \
          --exploration-scale 0 --eval-only > "$EVAL_JOB_LOG" 2>&1 || true
      elif [ "$kind" = lerobot ]; then
        key=$label-$group-$(date +%s)
        local learner=($LEROBOT_BIN/rlt-local checkpoint-eval $GPU_ARG --config-name eval --run-id "$key"
          --checkpoint "$ckpt" --rollouts-per-checkpoint "$((n - done))" $LEARNER_TUNNEL)
        if [ "$BACKEND" = modal ]; then
          rm -f "$out/$group/modal.jsonl"
          modal_proc "$out/$group/modal.jsonl" checkpoint-eval "$out/$group/learner.log" "${learner[@]}"
          modal_proc "$out/$group/modal.jsonl" policy-server "$out/policy-server.log" "${server[@]}"
          EVAL_CALLS+=("$out/$group/eval.call")
          RLT_POLICY_INSTANCE=$instance modal_start lerobot "$out/$group/modal.jsonl" "$out/$group/eval.call"
        else
          "${learner[@]}" > "$out/$group/learner.log" 2>&1 &
          ev=$!; EVAL_PIDS+=($ev)
          wait_for $ev "$out/$group/learner.log" "\[ckpt eval\] serving" checkpoint-eval
        fi
        RLT_POLICY_INSTANCE=$instance "${UVRUN[@]}" rlt-submit-actor "${ACTOR[@]}" "${seeded[@]}" --policy-type lerobot \
          --config-name eval --learner-key "eval:$key" --num-rollouts "$((n - done + 2))" \
          --exploration-scale 0 > "$EVAL_JOB_LOG" 2>&1 || true
        if [ "$BACKEND" = modal ]; then modal_stop "$out/$group/eval.call"; else kill $ev 2>/dev/null || true; fi
      else
        RLT_POLICY_INSTANCE=$instance "${UVRUN[@]}" rlt-submit-eval "${ACTOR[@]}" "${seeded[@]}" --config-name "$CONFIG" \
          --checkpoint-dir "$CELL_CHECKPOINT" --frozen-rlt-checkpoint "$ckpt" \
          --num-rollouts "$((n - done))" --no-include-base > "$EVAL_JOB_LOG" 2>&1 || true
      fi
      stop_cell_job "$EVAL_JOB_LOG"; EVAL_JOB_LOG=""
      sleep 5
    done
    done=$(_tally "$out/$group" "$n" "$group" | sed -n 1p)
    [ "$done" -ge "$n" ] || { log "$group incomplete ($done/$n); not recorded"; continue; }
    _tally "$out/$group" "$n" "$group" | tail -n +2 | while read -r c seed rollouts succ trips; do
      python3 -c 'import json, sys; print(json.dumps({"checkpoint": sys.argv[1], "condition": sys.argv[2], "seed": int(sys.argv[3]), "rollouts": int(sys.argv[4]), "successes": int(sys.argv[5]), "safety_trips": int(sys.argv[6])}))' \
        "$ckpt" "$c" "$seed" "$rollouts" "$succ" "$trips" >> "$out/results.jsonl"
      log "$c seed $seed: $succ/$rollouts ($trips trips)"
    done
  done
  if [ "$BACKEND" = modal ]; then
    modal_stop "$out/policy-server.call"
    data_pull "$out/policy-server.log"
  else
    kill $ps 2>/dev/null || true
    wait $ps 2>/dev/null || true
  fi
  eval_summary "$out"
}

# ID and OOD totals of a results.jsonl.
eval_summary() {  # <eval dir>
  python3 - "$1/results.jsonl" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])] if __import__("os").path.exists(sys.argv[1]) else []
def total(rs):
    return sum(r["successes"] for r in rs), sum(r["rollouts"] for r in rs), sum(r["safety_trips"] for r in rs)
for c in dict.fromkeys(r["condition"] for r in rows):
    s, n, t = total([r for r in rows if r["condition"] == c])
    seeds = " ".join(f"{r['successes']}/{r['rollouts']}" for r in rows if r["condition"] == c)
    print(f"  {c:<10} {s}/{n}  trips {t}  (per seed {seeds})")
i = total([r for r in rows if r["condition"] == "id"]); o = total([r for r in rows if r["condition"] != "id"])
print(f"  ID {i[0]}/{i[1]}  OOD {o[0]}/{o[1]}  trips {i[2] + o[2]}")
PY
}
