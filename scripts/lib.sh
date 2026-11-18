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

# Scene variation of the ID condition: the rail is drawn per episode around 0.7 (0.65-0.75).
ID_ARGS="--armnet.variation-rail --no-armnet.variation-lighting --no-armnet.variation-cameras --variation-curriculum --variation-scale-start 0.2 --variation-scale-min 0.2 --variation-scale-max 0.2 --variation-frontier-fraction 0"

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

# Evaluation protocol, per scene seed in EVAL_SEEDS (1042 2042 3042): EVAL_ID_ROLLOUTS (10)
# ID rollouts and EVAL_OOD_ROLLOUTS (5) each of OOD-easy and OOD-hard, so 30 ID and 30 OOD
# in all. OOD-easy stacks a rail outside the ID range (0.60-0.65 or 0.75-0.80), brightness 15
# and top tilt 0.2567; OOD-hard the same rail ranges, brightness 5 and tilt 0.3472. Rollouts
# are deterministic and a safety trip is a failure.
EVAL_SEEDS=${EVAL_SEEDS:-1042 2042 3042}
EVAL_ID_ROLLOUTS=${EVAL_ID_ROLLOUTS:-10}
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
    c = f"{v.get('condition') or default} {v.get('seed', -1)}"
    n, s, t = counts.get(c, (0, 0, 0))
    counts[c] = (n + 1, s + bool(r.get("success")), t + (r.get("outcome") == "safety_trip"))
    total += 1
print(total)
for c, (n, s, t) in counts.items():
    print(c, n, s, t)' "$2" "$3"
}

# Runs the evaluation protocol on one checkpoint and prints its summary, in one cell job
# that walks through the levels and scene seeds with --eval-schedule, setting the rail, the
# light and the top camera tilt per episode (EVAL_ONE_JOB=0: one job per level, tilted at
# its start).
#   kind lerobot: <checkpoint> is a pretrained_model dir. With <learner key> (a running
#     LeRobot learner), the actor sends it nothing (--eval-only); without, a
#     checkpoint-eval learner is started for each job.
#   kind rlt: <checkpoint> is a volume:// frozen actor of an rlt/dsrl run, served on <base>.
# Results: experiment-data/eval/<label>/results.jsonl, one row per condition; a group whose
# conditions are all there is skipped. Uses POLICY_PORT and, without a learner key, LEARNER_PORT.
eval_checkpoint() {  # <kind> <checkpoint> <label> <base or learner key>
  local kind=$1 ckpt=$2 label=$3 extra=${4:-} out=$DATA/eval/$3 server ps key ev group n conds
  local id=$EVAL_ID_ROLLOUTS ood=$EVAL_OOD_ROLLOUTS
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
  # The cell's own top tilt for ID scenes (RLT_CELL_CAMERA_AIM), when it sets one.
  local id_tilt
  id_tilt=$(echo "${RLT_CELL_CAMERA_AIM:-}" | tr ';' '\n' | grep "^${CELL:-none}=" | grep -oE "top\.tilt=[0-9.]+" | cut -d= -f2 || true)
  # One job for all three levels, tilting the top camera per episode; EVAL_ONE_JOB=0 runs
  # one job per level instead, each tilted once at its start.
  local groups=all
  [ "${EVAL_ONE_JOB:-1}" = 1 ] || groups="id ood-easy ood-hard"
  for group in $groups; do
    local entries="" seed level brightness ranges count tilt
    for level in $([ "$group" = all ] && echo id ood-easy ood-hard || echo "$group"); do
      case "$level" in
        id) brightness=30 ranges=null count=$id tilt=$id_tilt ;;
        ood-easy) brightness=15 ranges="[[0.60, 0.65], [0.75, 0.80]]" count=$ood tilt=0.2567 ;;
        ood-hard) brightness=5 ranges="[[0.60, 0.65], [0.75, 0.80]]" count=$ood tilt=0.3472 ;;
      esac
      for seed in $EVAL_SEEDS; do
        entries="$entries${entries:+, }{\"condition\": \"$level\", \"seed\": $seed, \"count\": $count, \"brightness\": $brightness, \"rail_ranges\": $ranges$([ "$group" = all ] && [ -n "$tilt" ] && echo ", \"top_tilt\": $tilt" || true)}"
      done
    done
    local seeds; seeds=$(echo $EVAL_SEEDS | wc -w)
    if [ "$group" = all ]; then
      n=$(((id + 2 * ood) * seeds)); conds="id ood-easy ood-hard"
      local group_args=(--eval-schedule "[$entries]" $ID_ARGS)
    else
      n=$((count * seeds)); conds=$group
      # Without per-episode tilt the job's own aim sets it.
      local group_args=(--eval-schedule "[$entries]" $ID_ARGS ${tilt:+--armnet.camera-aim top.tilt=$tilt})
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
      # Each schedule entry sets its own scene seed; the job-level one is only a default.
      local first=${EVAL_SEEDS%% *}
      local seeded=(--armnet.variation-seed "$first" --armnet.variation-episode-offset "$done"
        --action-noise-seed "$first" --timeout-seconds 7200 "${group_args[@]}")
      if [ "$kind" = lerobot ] && [ -n "$extra" ]; then
        "${UVRUN[@]}" rlt-submit-actor "${ACTOR[@]}" "${seeded[@]}" --policy-type lerobot \
          --config-name eval --learner-key "$extra" --num-rollouts "$((n - done))" \
          --exploration-scale 0 --eval-only > "$EVAL_JOB_LOG" 2>&1 || true
      elif [ "$kind" = lerobot ]; then
        key=$label-$group-$(date +%s)
        $LEROBOT_BIN/rlt-local checkpoint-eval $GPU_ARG --config-name eval --run-id "$key" \
          --checkpoint "$ckpt" --rollouts-per-checkpoint "$((n - done))" $LEARNER_TUNNEL \
          > "$out/$group/learner.log" 2>&1 &
        ev=$!; EVAL_PIDS+=($ev)
        wait_for $ev "$out/$group/learner.log" "\[ckpt eval\] serving" checkpoint-eval
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
for c in dict.fromkeys(r["condition"] for r in rows):
    s, n, t = total([r for r in rows if r["condition"] == c])
    seeds = " ".join(f"{r['successes']}/{r['rollouts']}" for r in rows if r["condition"] == c)
    print(f"  {c:<10} {s}/{n}  trips {t}  (per seed {seeds})")
i = total([r for r in rows if r["condition"] == "id"]); o = total([r for r in rows if r["condition"] != "id"])
print(f"  ID {i[0]}/{i[1]}  OOD {o[0]}/{o[1]}  trips {i[2] + o[2]}")
PY
}
