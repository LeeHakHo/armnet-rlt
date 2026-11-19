#!/usr/bin/env bash
# Evaluates one checkpoint on a cell. For each of the scene seeds 1042, 2042 and 3042: 10 ID
# rollouts, 5 OOD-easy (rail outside the ID range, brightness 15, top tilt 0.2567) and 5
# OOD-hard (rail outside the ID range, brightness 5, top tilt 0.3472); 30 ID and 30 OOD in
# all, deterministic, with the same scenes for every policy. A safety trip counts as a
# failed rollout. Conditions already in results.jsonl are skipped, so an interrupted
# evaluation continues where it stopped. scripts/run.sh runs the same protocol during
# training.
#
#   CELL=<cell-id> scripts/eval.sh <checkpoint> [options]
#
#   <checkpoint>           an rlt/dsrl run id (newest step, or --step), or a LeRobot
#                          pretrained_model directory (dit and other LeRobot policies)
#   --step N               rlt/dsrl step to evaluate (default: newest saved step)
#   --model <dir|HF repo>  rlt/dsrl: the π0.5 base the run trained on (default as run.sh)
#   --label NAME           results go to experiment-data/eval/<NAME>/results.jsonl
# Environment: EVAL_SEEDS ("1042 2042 3042"), EVAL_ID_ROLLOUTS (10 per seed),
# EVAL_OOD_ROLLOUTS (5 per seed for each of OOD-easy and OOD-hard). Rollouts are recorded
# to RECORD_REPO (see scripts/lib.sh); RECORD=0 turns that off.
set -euo pipefail
TARGET=${1:-}
case "$TARGET" in ""|-h|--help) sed -n '2,19p' "$0"; exit 0 ;; esac
shift
STEP="" MODEL="" LABEL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --step) STEP=$2 ;; --model) MODEL=$2 ;; --label) LABEL=$2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift 2
done
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
actor_common
case "$TARGET" in /*) ;; *) [ -e "$CALLER_PWD/$TARGET" ] && TARGET=$CALLER_PWD/$TARGET ;; esac

cleanup() {
  local status=$?
  kill "${EVAL_PIDS[@]}" 2>/dev/null || true
  [ -z "$EVAL_JOB_LOG" ] || stop_cell_job "$EVAL_JOB_LOG"
  exit $status
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if [ -f "$TARGET/config.json" ]; then
  CKPT=$(cd "$TARGET" && pwd)
  # <run>/checkpoints/<step>/pretrained_model -> <run>-<step>
  LABEL=${LABEL:-$(basename "$(dirname "$(dirname "$(dirname "$CKPT")")")")-$(basename "$(dirname "$CKPT")")}
  eval_checkpoint lerobot "$CKPT" "$LABEL"
else
  RUN=$(basename "$TARGET")
  RUN_DIR=$DATA/tasks/$CONFIG/rlt_checkpoints/$RUN
  [ -d "$RUN_DIR" ] || die "no LeRobot checkpoint or rlt/dsrl run at $TARGET (looked in $RUN_DIR)"
  STEP=${STEP:-$(ls "$RUN_DIR" | grep -oE '^step_[0-9]+$' | sort -V | tail -1 | sed 's/step_0*//' || true)}
  [ -n "$STEP" ] || die "$RUN has no saved step"
  CKPT=$(freeze_rlt "$RUN" "$STEP") || die "could not freeze $RUN step $STEP"
  eval_checkpoint rlt "$CKPT" "${LABEL:-$RUN-$(printf step_%06d "$STEP")}" "$(resolve_model "$MODEL")"
fi
