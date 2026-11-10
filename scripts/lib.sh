# Shared helpers for setup.sh, run.sh and eval.sh. Source it; do not run it.
#
# Settings (environment or .env):
#   CELL              Armnet cell to run on (e.g. cell-01); empty lets the queue pick
#   IMAGE_REF         actor image (setup.sh writes it to .env)
#   GPU               GPU index for the learner and policy server (unset: visible GPU)
#   RLT_AUTH_SECRET   Armnet secret with RLT_LEARNER_AUTH_TOKEN (default armnet-rlt-auth)
#   RLT_HF_SECRET     Armnet secret with HF_TOKEN (default huggingface-token)
#   PUBLIC_HOST       your own TLS ingress instead of ngrok; then also TLS_CERT, TLS_KEY,
#                     PUBLIC_LEARNER_PORT (443) and PUBLIC_POLICY_PORT (8443)
#   LEARNER_PORT, POLICY_PORT   local ports (50051, 50052)
#   WANDB_PROJECT     log learners to this Weights & Biases project

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
RLT_AUTH_SECRET=${RLT_AUTH_SECRET:-armnet-rlt-auth}
RLT_HF_SECRET=${RLT_HF_SECRET:-huggingface-token}
LEARNER_PORT=${LEARNER_PORT:-50051}
POLICY_PORT=${POLICY_PORT:-50052}
WANDB_ARG=${WANDB_PROJECT:+--wandb-project $WANDB_PROJECT}
BASE_REPO=AutoRLBench/pi05_busybox_minmax_step5k_rltoken
BASE_DIR=$ROOT/checkpoints/pi05_rltoken
# Checkpoint path passed to the cell; unused while the policy runs on your GPU.
CELL_CHECKPOINT=${CELL_CHECKPOINT:-volume://openpi/checkpoints/pi05_rltoken}
LEARNER_BIN=envs/venv-learner/bin
OPENPI_BIN=envs/venv-openpi/bin
LEROBOT_BIN=envs/venv-lerobot/bin

log() { echo "[$(basename "$0" .sh) $(date -u +%H:%M:%S)] $*"; }
die() { log "$*"; exit 1; }
py() { "${UVRUN[@]}" python - "$@"; }

# Tunnel flags for the learner (cell -> learner) and the policy server (cell -> policy).
if [ -n "${PUBLIC_HOST:-}" ]; then
  : "${TLS_CERT:?PUBLIC_HOST needs TLS_CERT}" "${TLS_KEY:?PUBLIC_HOST needs TLS_KEY}"
  LEARNER_TUNNEL="--tunnel none --public-endpoint $PUBLIC_HOST:${PUBLIC_LEARNER_PORT:-443} --port $LEARNER_PORT --tls-cert $TLS_CERT --tls-key $TLS_KEY"
  POLICY_TUNNEL="--tunnel none --public-endpoint $PUBLIC_HOST:${PUBLIC_POLICY_PORT:-8443} --port $POLICY_PORT --tls-cert $TLS_CERT --tls-key $TLS_KEY"
  STEERING="--steering-from-learner 127.0.0.1:$LEARNER_PORT --steering-learner-tls-name $PUBLIC_HOST"
else
  LEARNER_TUNNEL="--tunnel ngrok-tcp --port $LEARNER_PORT"
  POLICY_TUNNEL="--tunnel ngrok --port $POLICY_PORT"
  STEERING="--steering-from-learner 127.0.0.1:$LEARNER_PORT"
fi

# Common actor flags in ACTOR: fixed scene (rail 0.7, light 30), 30 s episodes,
# no recording, policy on your GPU.
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

# Scene variation of the ID condition and the six OOD conditions used in evaluation.
ID_ARGS="--armnet.variation-rail --no-armnet.variation-lighting --no-armnet.variation-cameras --variation-curriculum --variation-scale-start 0.2 --variation-scale-min 0.2 --variation-scale-max 0.2 --variation-frontier-fraction 0"
ALL_CONDITIONS="id rail-low rail-high bright15 bright5 tilt1550 tilt1750"
condition_args() {
  case "$1" in
    id) echo "$ID_ARGS" ;;
    rail-low) echo "$ID_ARGS --variation-rail-range 0.60,0.65" ;;
    rail-high) echo "$ID_ARGS --variation-rail-range 0.75,0.80" ;;
    bright15) echo "$ID_ARGS --armnet.lightbox-brightness 15" ;;
    bright5) echo "$ID_ARGS --armnet.lightbox-brightness 5" ;;
    tilt1550) echo "$ID_ARGS --armnet.camera-aim top.tilt=0.2567" ;;
    tilt1750) echo "$ID_ARGS --armnet.camera-aim top.tilt=0.3472" ;;
    *) die "unknown condition $1 (one of: $ALL_CONDITIONS)" ;;
  esac
}

# 0 when CELL is available (or no CELL is set).
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

# Stops the cell job a submit log reports, if it is still queued or running.
stop_cell_job() {
  local id
  id=$(grep -oE "job [0-9a-f-]{36} dispatched|job queued: [0-9a-f-]{36}" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9a-f-]{36}" || true)
  [ -n "$id" ] || return 0
  py "$id" <<'PY'
import sys
from armnet_client import OrchestratorClient
client = OrchestratorClient()
try:
    if client.get(sys.argv[1]).status.value in ("running", "queued", "dispatched", "submitted"):
        client.stop_job(sys.argv[1], reason="finished")
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

# A local OpenPI checkpoint dir for --model: a directory as is, otherwise a
# Hugging Face repo downloaded under checkpoints/.
resolve_model() {
  local m=${1:-$BASE_REPO}
  case "$m" in /*) ;; *) [ -d "$CALLER_PWD/$m" ] && m=$CALLER_PWD/$m ;; esac
  if [ -d "$m" ]; then (cd "$m" && pwd); return; fi
  local dir=$ROOT/checkpoints/$(basename "$m")
  [ "$m" = "$BASE_REPO" ] && dir=$BASE_DIR
  [ -d "$dir" ] || "${UVRUN[@]}" hf download "$m" --local-dir "$dir" >&2
  echo "$dir"
}
