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
  if [ ! -d "$dir" ]; then
    $LEARNER_BIN/hf download "$m" --local-dir "$dir" >&2 || { rm -rf "$dir"; die "could not download $m"; }
  fi
  echo "$dir"
}

# Evaluation protocol: EVAL_ID_ROLLOUTS (30) ID rollouts and EVAL_OOD_ROLLOUTS (5) for each
# of the six OOD conditions (30 OOD in all), deterministic, all with variation seed
# EVAL_SEED (1042) so every policy sees the same scenes. A safety trip is a failure.
EVAL_SEED=${EVAL_SEED:-1042}
EVAL_ID_ROLLOUTS=${EVAL_ID_ROLLOUTS:-30}
EVAL_OOD_ROLLOUTS=${EVAL_OOD_ROLLOUTS:-5}
EVAL_PIDS=()
EVAL_JOB_LOG=""

# Freezes step STEP of an rlt/dsrl run, uploads it, and prints its volume path.
freeze_rlt() {  # <run> <step>
  local dir; dir=$(printf step_%06d "$2")
  "${UVRUN[@]}" rlt-freeze-checkpoint --local --config-name "$CONFIG" --run-id "$1" --step "$2" \
    --output "$DATA/tasks/$CONFIG/frozen/$1/$dir.pt" \
    --armnet-volume-path "rlt/frozen/$CONFIG/$1/$dir/frozen_actor.pt" --overwrite >&2 \
    && echo "volume://rlt/frozen/$CONFIG/$1/$dir/frozen_actor.pt"
}

# "<rollouts> <successes> <trips>" over the rollout records in a directory's job logs.
_tally() {  # <logs dir> <cap>
  cat "$1"/job*.log 2>/dev/null | sed -E 's/\x1b\[[0-9;]*m//g' | python3 -c '
import json, sys
cap = int(sys.argv[1]); n = s = t = 0
for line in sys.stdin:
    tag = next((k for k in ("[rlt rollout]", "[act rollout]") if k in line), None)
    if tag is None or n >= cap:
        continue
    try:
        r = json.loads(line.split(tag, 1)[1])
    except ValueError:
        continue
    if r.get("policy_variant") == "base":
        continue
    n += 1; s += bool(r.get("success")); t += r.get("outcome") == "safety_trip"
print(n, s, t)' "$2"
}

# Runs the evaluation protocol on one checkpoint and prints its summary.
#   kind lerobot: <checkpoint> is a pretrained_model dir. With <learner key> (a running
#     LeRobot learner), the actor sends it nothing (--eval-only); without, a
#     checkpoint-eval learner is started for each job.
#   kind rlt: <checkpoint> is a volume:// frozen actor of an rlt/dsrl run, served on <base>.
# Results: experiment-data/eval/<label>/results.jsonl, one row per condition; conditions
# already there are skipped. Uses POLICY_PORT and, without a learner key, LEARNER_PORT.
eval_checkpoint() {  # <kind> <checkpoint> <label> <base or learner key>
  local kind=$1 ckpt=$2 label=$3 extra=${4:-} out=$DATA/eval/$3 server c n done succ trips jobs ps key ev
  mkdir -p "$out"
  if [ "$kind" = lerobot ]; then
    server=($LEROBOT_BIN/rlt-local policy-server $GPU_ARG --config-name eval --policy-type lerobot
      --checkpoint "$ckpt" $POLICY_TUNNEL)
    [ -z "$extra" ] && server+=($STEERING)
  else
    server=($OPENPI_BIN/rlt-local policy-server $GPU_ARG --config-name "$CONFIG" --checkpoint "$extra"
      --prompt "$PROMPT" $POLICY_TUNNEL)
  fi
  "${server[@]}" > "$out/policy-server.log" 2>&1 &
  ps=$!; EVAL_PIDS+=($ps)
  wait_for $ps "$out/policy-server.log" "policy server" "eval policy server"
  log "evaluating $ckpt on ${CELL:-any cell} -> $out"
  for c in $ALL_CONDITIONS; do
    n=$([ "$c" = id ] && echo "$EVAL_ID_ROLLOUTS" || echo "$EVAL_OOD_ROLLOUTS")
    grep -q "\"condition\": \"$c\"" "$out/results.jsonl" 2>/dev/null && continue
    mkdir -p "$out/$c"
    jobs=0
    while :; do
      read -r done succ trips <<<"$(_tally "$out/$c" "$n")"
      [ "$done" -ge "$n" ] && break
      [ "$jobs" -ge 4 ] && { log "$c: stopping after $jobs jobs at $done/$n"; break; }
      jobs=$((jobs + 1))
      EVAL_JOB_LOG=$out/$c/job$(date +%s).log
      wait_cell
      local seeded=(--armnet.variation-seed "$EVAL_SEED" --armnet.variation-episode-offset "$done"
        --action-noise-seed "$EVAL_SEED" --timeout-seconds 7200 $(condition_args "$c"))
      if [ "$kind" = lerobot ] && [ -n "$extra" ]; then
        "${UVRUN[@]}" rlt-submit-actor "${ACTOR[@]}" "${seeded[@]}" --policy-type lerobot \
          --config-name eval --learner-key "$extra" --num-rollouts "$((n - done))" \
          --exploration-scale 0 --eval-only > "$EVAL_JOB_LOG" 2>&1 || true
      elif [ "$kind" = lerobot ]; then
        key=$label-$c-$(date +%s)
        $LEROBOT_BIN/rlt-local checkpoint-eval $GPU_ARG --config-name eval --run-id "$key" \
          --checkpoint "$ckpt" --rollouts-per-checkpoint "$((n - done))" $LEARNER_TUNNEL \
          > "$out/$c/learner.log" 2>&1 &
        ev=$!; EVAL_PIDS+=($ev)
        wait_for $ev "$out/$c/learner.log" "\[ckpt eval\] serving" checkpoint-eval
        "${UVRUN[@]}" rlt-submit-actor "${ACTOR[@]}" "${seeded[@]}" --policy-type lerobot \
          --config-name eval --learner-key "eval:$key" --num-rollouts "$((n - done + 2))" \
          --exploration-scale 0 > "$EVAL_JOB_LOG" 2>&1 || true
        kill $ev 2>/dev/null || true
      else
        "${UVRUN[@]}" rlt-submit-eval "${ACTOR[@]}" "${seeded[@]}" --config-name "$CONFIG" \
          --checkpoint-dir "$CELL_CHECKPOINT" --frozen-rlt-checkpoint "$ckpt" \
          --num-rollouts "$((n - done))" --no-include-base > "$EVAL_JOB_LOG" 2>&1 || true
      fi
      stop_cell_job "$EVAL_JOB_LOG"; EVAL_JOB_LOG=""
      sleep 30
    done
    read -r done succ trips <<<"$(_tally "$out/$c" "$n")"
    [ "$done" -ge "$n" ] || { log "$c incomplete ($done/$n); not recorded"; continue; }
    python3 -c 'import json, sys; print(json.dumps({"checkpoint": sys.argv[1], "condition": sys.argv[2], "seed": int(sys.argv[3]), "rollouts": int(sys.argv[4]), "successes": int(sys.argv[5]), "safety_trips": int(sys.argv[6])}))' \
      "$ckpt" "$c" "$EVAL_SEED" "$done" "$succ" "$trips" >> "$out/results.jsonl"
    log "$c: $succ/$done ($trips trips)"
  done
  kill $ps 2>/dev/null || true
  wait $ps 2>/dev/null || true
  eval_summary "$out"
}

# ID and OOD totals of a results.jsonl.
eval_summary() {  # <eval dir>
  python3 - "$1/results.jsonl" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])] if __import__("os").path.exists(sys.argv[1]) else []
def total(rs):
    return sum(r["successes"] for r in rs), sum(r["rollouts"] for r in rs), sum(r["safety_trips"] for r in rs)
for r in rows:
    print(f"  {r['condition']:<10} {r['successes']}/{r['rollouts']}  trips {r['safety_trips']}")
i = total([r for r in rows if r["condition"] == "id"]); o = total([r for r in rows if r["condition"] != "id"])
print(f"  ID {i[0]}/{i[1]}  OOD {o[0]}/{o[1]}  trips {i[2] + o[2]}")
PY
}
