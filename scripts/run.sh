#!/usr/bin/env bash
# Trains one method on a cell, end to end: starts the policy server and the learner on
# this machine's GPU, then submits the actor (again after a cell or safety stop) until
# the learner reaches its last step. Ctrl-C stops everything, the cell job included.
#
#   CELL=<cell-id> scripts/run.sh <rlt|dsrl|dit> [options]
#
#   --model <dir|HF repo>  starting policy: an OpenPI π0.5 checkpoint for rlt/dsrl
#                          (default AutoRLBench/pi05_busybox_minmax_step5k_rltoken), a
#                          LeRobot pretrained_model for dit (default: trained from the
#                          demo dataset with --policy-type)
#   --policy-type <type>   dit only: LeRobot policy to train from the demos
#                          (multi_task_dit, diffusion, smolvla, ...; default multi_task_dit)
#   --steps N              learner steps (rlt 30000, dsrl 400000, dit 1800)
#   --std X                exploration std, rlt/dsrl (0.5)
#   --trip-penalty X       reward of a safety-tripped episode, rlt/dsrl (rlt 5, dsrl 2)
#   --run-id ID            run name (default <method>-<UTC time>)
#   --resume ID            continue run ID from its newest checkpoint
#   --learner-args "..."   extra learner flags (rlt-local online / lerobot-streaming-bc)
#   --actor-args "..."     extra rlt-submit-actor flags
#
# Logs: experiment-data/logs/<run id>/. Checkpoints: experiment-data/tasks/<config>/
# rlt_checkpoints/<run id> (rlt, dsrl) or experiment-data/lerobot-sbc/<run id> (dit).
set -euo pipefail
METHOD=${1:-}
case "$METHOD" in rlt|dsrl|dit) shift ;; -h|--help|"") sed -n '2,24p' "$0"; exit 0 ;;
  *) echo "method must be rlt, dsrl or dit" >&2; exit 2 ;; esac
MODEL="" POLICY_TYPE=multi_task_dit STEPS="" STD=0.5 PENALTY="" RUN="" RESUME=""
LEARNER_ARGS="" ACTOR_ARGS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL=$2 ;; --policy-type) POLICY_TYPE=$2 ;; --steps) STEPS=$2 ;;
    --std) STD=$2 ;; --trip-penalty) PENALTY=$2 ;; --run-id) RUN=$2 ;;
    --resume) RUN=$2; RESUME=--resume ;;
    --learner-args) LEARNER_ARGS=$2 ;; --actor-args) ACTOR_ARGS=$2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift 2
done
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
actor_common
RUN=${RUN:-$METHOD-$(date -u +%Y%m%d-%H%M%S)}
LOGS=$DATA/logs/$RUN
mkdir -p "$LOGS"
DATASET=${DATASET:-AutoRLBench/busybox_push_green_button_426x240}
PIDS=()
cleanup() {
  local status=$?
  kill "${PIDS[@]}" 2>/dev/null || true
  [ -z "${ACTOR_LOG:-}" ] || stop_cell_job "$ACTOR_LOG"
  exit $status
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if [ "$METHOD" = dit ]; then
  STEPS=${STEPS:-1800}
  BIN=$LEROBOT_BIN
  if [ -n "$MODEL" ]; then
    POLICY=$(resolve_model "$MODEL")
  else
    PRE_DIR=$DATA/dit/demo-$POLICY_TYPE-s300
    POLICY=$PRE_DIR/checkpoints/000300/pretrained_model
    if [ ! -f "$POLICY/config.json" ]; then
      log "training a $POLICY_TYPE on $DATASET for 300 steps -> $PRE_DIR"
      EXTRA=(); [ "$POLICY_TYPE" = multi_task_dit ] && EXTRA=(--policy.image_resize_shape='[240, 240]')
      env ${GPU:+CUDA_VISIBLE_DEVICES=$GPU} $BIN/lerobot-train --dataset.repo_id="$DATASET" \
        --policy.type="$POLICY_TYPE" --policy.device=cuda --policy.push_to_hub=false "${EXTRA[@]}" \
        --output_dir="$PRE_DIR" --job_name="demo-$POLICY_TYPE" --steps=300 --batch_size=32 \
        --num_workers=8 --save_freq=300 --eval_freq=0 --wandb.enable=false > "$LOGS/pretrain.log" 2>&1
    fi
  fi
  log "starting policy $POLICY"
  $BIN/rlt-local policy-server $GPU_ARG --config-name eval --policy-type lerobot \
    --checkpoint "$POLICY" $STEERING $POLICY_TUNNEL > "$LOGS/policy-server.log" 2>&1 &
  PIDS+=($!)
  $BIN/rlt-local lerobot-streaming-bc $GPU_ARG --run-id "$RUN" --policy-path "$POLICY" \
    --base-dataset "$DATASET" --base-batch-fraction 0.5 --min-shaped-reward 1.0 \
    --online-steps "$STEPS" --idle-timeout-s 1800 $LEARNER_TUNNEL $WANDB_ARG $LEARNER_ARGS \
    > "$LOGS/learner.log" 2>&1 &
  LEARNER=$!; PIDS+=($LEARNER)
  wait_for $LEARNER "$LOGS/learner.log" "\[lerobot sbc\] serving" learner
  DONE_PATTERN="\[lerobot sbc\] complete"
  SUBMIT=(--policy-type lerobot --config-name eval --learner-key "eval:$RUN" --num-rollouts 180
    --exploration-scale 0 --episodes-per-round 30,30,20 --steps-per-round 300 --round-wait-timeout-s 2400)
  OUT=$DATA/lerobot-sbc/$RUN/checkpoints
else
  STEPS=${STEPS:-$([ "$METHOD" = rlt ] && echo 30000 || echo 400000)}
  PENALTY=${PENALTY:-$([ "$METHOD" = rlt ] && echo 5 || echo 2)}
  BASE=$(resolve_model "$MODEL")
  if [ "$METHOD" = rlt ]; then
    ALGO=(--algorithm rlt --residual-actor --residual-scale 0.1 --critic-action-residual)
    [ -f "$DATA/tasks/$CONFIG/demo/rlt_demo_cache.pt" ] \
      || { log "no demo cache: RLT starts without demo pretraining"; ALGO+=(--pretrain-steps 0 --online-batch-fraction 1.0); }
    SERVER_STEERING=""
    ACTOR_ALGO=(--algorithm rlt)
  else
    ALGO=(--algorithm dsrl --dsrl-noise-bound 1.5 --dsrl-noise-steps 5 --pretrain-steps 0)
    SERVER_STEERING=$STEERING
    ACTOR_ALGO=(--algorithm dsrl --dsrl-noise-bound 1.5 --dsrl-noise-steps 5)
  fi
  log "$METHOD run $RUN on base $BASE"
  $OPENPI_BIN/rlt-local policy-server $GPU_ARG --config-name "$CONFIG" --checkpoint "$BASE" \
    --prompt "$PROMPT" $SERVER_STEERING $POLICY_TUNNEL > "$LOGS/policy-server.log" 2>&1 &
  PIDS+=($!)
  $LEARNER_BIN/rlt-local online $GPU_ARG --config-name "$CONFIG" --run-id "$RUN" $RESUME \
    "${ALGO[@]}" --safety-trip-penalty "$PENALTY" --online-steps "$STEPS" --save-freq 10000 \
    --idle-timeout-s 0 --push-interval-s 5 $LEARNER_TUNNEL $WANDB_ARG $LEARNER_ARGS \
    >> "$LOGS/learner.log" 2>&1 &
  LEARNER=$!; PIDS+=($LEARNER)
  wait_for $LEARNER "$LOGS/learner.log" "rendezvous" learner
  DONE_PATTERN=""  # the online learner exits when it reaches --steps
  SUBMIT=(--config-name "$CONFIG" --checkpoint-dir "$CELL_CHECKPOINT" --learner-key "$CONFIG:$RUN"
    --num-rollouts 1200 --exploration-scale "$STD" --scene-retry-delay 3 --scene-retry-limit 1
    "${ACTOR_ALGO[@]}")
  OUT=$DATA/tasks/$CONFIG/rlt_checkpoints/$RUN
fi
wait_for "${PIDS[0]}" "$LOGS/policy-server.log" "policy server" "policy server"
log "learner and policy server up; logs in $LOGS"

# Sleeps up to N seconds, returning early once the learner has exited.
pause() { local i; for ((i = 0; i < $1; i += 10)); do kill -0 "$LEARNER" 2>/dev/null || return 0; sleep 10; done; }
ATTEMPT=$(find "$LOGS" -maxdepth 1 -name "actor.attempt*.log" | wc -l)
QUICK=0 TRIPS=0
while kill -0 "$LEARNER" 2>/dev/null && { [ -z "$DONE_PATTERN" ] || ! grep -q "$DONE_PATTERN" "$LOGS/learner.log"; }; do
  wait_cell
  ATTEMPT=$((ATTEMPT + 1))
  ACTOR_LOG=$LOGS/actor.attempt$ATTEMPT.log
  log "actor attempt $ATTEMPT${CELL:+ on $CELL}"
  START=$(date +%s)
  "${UVRUN[@]}" rlt-submit-actor "${ACTOR[@]}" "${SUBMIT[@]}" --timeout-seconds 72000 \
    $ID_ARGS $ACTOR_ARGS > "$ACTOR_LOG" 2>&1 || true
  stop_cell_job "$ACTOR_LOG"
  ELAPSED=$(( $(date +%s) - START ))
  if grep -q "gate tripped" "$ACTOR_LOG"; then
    # A safety trip ends the cell job; the episode reached the learner as a failure.
    if [ "$ELAPSED" -lt 300 ]; then TRIPS=$((TRIPS + 1)); else TRIPS=0; fi
    [ "$TRIPS" -lt 10 ] || die "10 safety trips in a row within 5 minutes; check the cell"
    pause 60
  elif [ "$ELAPSED" -lt 300 ]; then
    QUICK=$((QUICK + 1))
    [ "$QUICK" -lt 20 ] || die "the actor failed $QUICK times within 5 minutes; see $ACTOR_LOG"
    log "actor ended after ${ELAPSED}s; retrying in 10 minutes"
    pause 600
  else
    QUICK=0 TRIPS=0
    pause 60
  fi
done
ACTOR_LOG=""
log "learner finished; newest checkpoint: $(ls "$OUT" 2>/dev/null | grep -E '^(step_)?[0-9]+$' | sort -V | tail -1 || true)"
log "checkpoints in $OUT"
log "evaluate with: CELL=$CELL scripts/eval.sh $([ "$METHOD" = dit ] && echo "$OUT/<step>/pretrained_model" || echo "$RUN")"
