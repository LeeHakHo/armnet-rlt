#!/usr/bin/env bash
# Evaluates one checkpoint on a cell: for every condition and seed, ROLLOUTS
# deterministic 30 s rollouts with the seed as the scene-variation seed. A safety trip
# counts as a failed rollout. Seeds already in a condition's results.jsonl are skipped,
# so an interrupted evaluation continues where it stopped.
#
#   CELL=<cell-id> scripts/eval.sh <checkpoint> [options]
#
#   <checkpoint>           an rlt/dsrl run id (newest step, or --step), or a LeRobot
#                          pretrained_model directory (dit and other LeRobot policies)
#   --step N               rlt/dsrl step to evaluate (default: newest saved step)
#   --model <dir|HF repo>  rlt/dsrl: the π0.5 base the run trained on (default as run.sh)
#   --conditions "..."     subset of: id rail-low rail-high bright15 bright5 tilt1550 tilt1750
#   --seeds "..."          default "1042 2042 3042"
#   --rollouts N           rollouts per seed (default 5)
#   --label NAME           results go to experiment-data/eval/<NAME>-<condition>
set -euo pipefail
TARGET=${1:-}
case "$TARGET" in ""|-h|--help) sed -n '2,17p' "$0"; exit 0 ;; esac
shift
STEP="" MODEL="" CONDITIONS="" SEEDS="1042 2042 3042" ROLLOUTS=5 LABEL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --step) STEP=$2 ;; --model) MODEL=$2 ;; --conditions) CONDITIONS=$2 ;;
    --seeds) SEEDS=$2 ;; --rollouts) ROLLOUTS=$2 ;; --label) LABEL=$2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift 2
done
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
actor_common
CONDITIONS=${CONDITIONS:-$ALL_CONDITIONS}
MAX_JOBS=4
case "$TARGET" in /*) ;; *) [ -e "$CALLER_PWD/$TARGET" ] && TARGET=$CALLER_PWD/$TARGET ;; esac

if [ -f "$TARGET/config.json" ]; then
  KIND=lerobot
  CKPT=$(cd "$TARGET" && pwd)
  # <run>/checkpoints/<step>/pretrained_model -> <run>-<step>
  LABEL=${LABEL:-$(basename "$(dirname "$(dirname "$(dirname "$CKPT")")")")-$(basename "$(dirname "$CKPT")")}
  SERVER=($LEROBOT_BIN/rlt-local policy-server $GPU_ARG --config-name eval --policy-type lerobot
    --checkpoint "$CKPT" $STEERING $POLICY_TUNNEL)
else
  KIND=rlt
  RUN=$(basename "$TARGET")
  RUN_DIR=$DATA/tasks/$CONFIG/rlt_checkpoints/$RUN
  [ -d "$RUN_DIR" ] || die "no LeRobot checkpoint or rlt/dsrl run at $TARGET (looked in $RUN_DIR)"
  STEP=${STEP:-$(ls "$RUN_DIR" | grep -oE '^step_[0-9]+$' | sort -V | tail -1 | sed 's/step_0*//' || true)}
  [ -n "$STEP" ] || die "$RUN has no saved step"
  STEP_DIR=$(printf step_%06d "$STEP")
  VOLUME=rlt/frozen/$CONFIG/$RUN/$STEP_DIR/frozen_actor.pt
  log "freezing $RUN $STEP_DIR -> volume://$VOLUME"
  "${UVRUN[@]}" rlt-freeze-checkpoint --local --config-name "$CONFIG" --run-id "$RUN" \
    --step "$STEP" --output "$DATA/tasks/$CONFIG/frozen/$RUN/$STEP_DIR.pt" \
    --armnet-volume-path "$VOLUME" --overwrite
  CKPT=volume://$VOLUME
  LABEL=${LABEL:-$RUN-$STEP_DIR}
  BASE=$(resolve_model "$MODEL")
  SERVER=($OPENPI_BIN/rlt-local policy-server $GPU_ARG --config-name "$CONFIG" --checkpoint "$BASE"
    --prompt "$PROMPT" $POLICY_TUNNEL)
fi

PIDS=()
cleanup() {
  local status=$?
  kill "${PIDS[@]}" 2>/dev/null || true
  [ -z "${JOB_LOG:-}" ] || stop_cell_job "$JOB_LOG"
  exit $status
}
trap cleanup EXIT
trap 'exit 130' INT TERM
mkdir -p "$DATA/eval"
"${SERVER[@]}" > "$DATA/eval/$LABEL.policy-server.log" 2>&1 &
PIDS+=($!)
wait_for "${PIDS[0]}" "$DATA/eval/$LABEL.policy-server.log" "policy server" "policy server"
log "evaluating $CKPT ($KIND) on ${CELL:-any cell}: $CONDITIONS"

# "<rollouts> <successes> <trips>" over the rollout records in a seed's job logs.
tally() {  # <logs dir> <seed>
  cat "$1"/s$2.job*.log 2>/dev/null | sed -E 's/\x1b\[[0-9;]*m//g' | python3 -c '
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
print(n, s, t)' "$ROLLOUTS"
}

for C in $CONDITIONS; do
  OUT=$DATA/eval/$LABEL-$C
  RESULTS=$OUT/results.jsonl
  mkdir -p "$OUT/logs"
  for SEED in $SEEDS; do
    grep -q "\"seed\": $SEED," "$RESULTS" 2>/dev/null && continue
    JOBS=0
    while :; do
      read -r DONE SUCC TRIPS <<<"$(tally "$OUT/logs" "$SEED")"
      [ "$DONE" -ge "$ROLLOUTS" ] && break
      [ "$JOBS" -ge "$MAX_JOBS" ] && { log "$C seed $SEED: stopping after $JOBS jobs at $DONE/$ROLLOUTS"; break; }
      JOBS=$((JOBS + 1)); LEFT=$((ROLLOUTS - DONE))
      JOB_LOG=$OUT/logs/s$SEED.job$(date +%s).log
      wait_cell
      SEEDED=(--armnet.variation-seed "$SEED" --armnet.variation-episode-offset "$DONE" --action-noise-seed "$SEED")
      if [ "$KIND" = lerobot ]; then
        KEY=$LABEL-$C-s$SEED-$(date +%s)
        $LEROBOT_BIN/rlt-local checkpoint-eval $GPU_ARG --config-name eval --run-id "$KEY" \
          --checkpoint "$CKPT" --rollouts-per-checkpoint "$LEFT" $LEARNER_TUNNEL \
          > "$OUT/logs/s$SEED.learner.log" 2>&1 &
        EVAL_PID=$!; PIDS+=($EVAL_PID)
        wait_for $EVAL_PID "$OUT/logs/s$SEED.learner.log" "\[ckpt eval\] serving" checkpoint-eval
        "${UVRUN[@]}" rlt-submit-actor "${ACTOR[@]}" "${SEEDED[@]}" --policy-type lerobot \
          --config-name eval --learner-key "eval:$KEY" --num-rollouts "$((LEFT + 2))" \
          --exploration-scale 0 --timeout-seconds 3600 $(condition_args "$C") > "$JOB_LOG" 2>&1 || true
        kill $EVAL_PID 2>/dev/null || true
      else
        "${UVRUN[@]}" rlt-submit-eval "${ACTOR[@]}" "${SEEDED[@]}" --config-name "$CONFIG" \
          --checkpoint-dir "$CELL_CHECKPOINT" --frozen-rlt-checkpoint "$CKPT" \
          --num-rollouts "$LEFT" --no-include-base --timeout-seconds 7200 $(condition_args "$C") \
          > "$JOB_LOG" 2>&1 || true
      fi
      stop_cell_job "$JOB_LOG"; JOB_LOG=""
      sleep 30
    done
    read -r DONE SUCC TRIPS <<<"$(tally "$OUT/logs" "$SEED")"
    [ "$DONE" -ge "$ROLLOUTS" ] || { log "$C seed $SEED incomplete ($DONE/$ROLLOUTS); not recorded"; continue; }
    python3 -c 'import json, sys; print(json.dumps({"checkpoint": sys.argv[1], "condition": sys.argv[2], "seed": int(sys.argv[3]), "rollouts": int(sys.argv[4]), "successes": int(sys.argv[5]), "safety_trips": int(sys.argv[6])}))' \
      "$CKPT" "$C" "$SEED" "$DONE" "$SUCC" "$TRIPS" >> "$RESULTS"
    log "$C seed $SEED: $SUCC/$DONE ($TRIPS trips)"
  done
done

python3 - "$DATA/eval" "$LABEL" $CONDITIONS <<'PY'
import json, sys
from pathlib import Path
root, label, conditions = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
total = [0, 0, 0]
print(f"{'condition':<10} {'success':>9} {'trips':>6}")
for c in conditions:
    f = root / f"{label}-{c}" / "results.jsonl"
    rows = [json.loads(l) for l in f.open()] if f.exists() else []
    s, n, t = (sum(r[k] for r in rows) for k in ("successes", "rollouts", "safety_trips"))
    total = [total[0] + s, total[1] + n, total[2] + t]
    print(f"{c:<10} {f'{s}/{n}':>9} {t:>6}")
print(f"{'total':<10} {f'{total[0]}/{total[1]}':>9} {total[2]:>6}")
PY
