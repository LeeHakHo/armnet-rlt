#!/usr/bin/env bash
# Trains one method on a cell for a fixed number of robot episodes and evaluates it
# during training: starts the policy server and the learner on this machine's GPU, and
# submits the actor in segments of --eval-every episodes. After each segment the learner
# saves a checkpoint and the scripts/eval.sh protocol (30 ID + 30 OOD rollouts) runs on
# it; evaluation rollouts are not training episodes. Ctrl-C stops everything, the cell
# job included.
#
#   CELL=<cell-id> scripts/run.sh <rlt|dsrl|dit> [options]
#   dit trains any LeRobot policy by iterative BC: multi-task DiT by default, or
#   --policy-type, including your own plugin under plugins/ (see the README).
#
#   --episodes N           training episodes (400)
#   --eval-every N         evaluate after every N training episodes (100); 0 disables
#   --model <dir|HF repo>  starting policy: an OpenPI π0.5 checkpoint for rlt/dsrl
#                          (default AutoRLBench/pi05_busybox_minmax_step5k_rltoken), a
#                          LeRobot pretrained_model for dit (default: trained from the
#                          demo dataset with --policy-type)
#   --policy-type <type>   dit only: LeRobot policy to train from the demos
#                          (multi_task_dit, diffusion, smolvla, ...; default multi_task_dit)
#   --steps N              rlt/dsrl: learner step cap, which also sets the learning-rate
#                          schedule (rlt 30000, dsrl 400000)
#   --std X                exploration std, rlt/dsrl (0.5)
#   --trip-penalty X       reward of a safety-tripped episode, rlt/dsrl (rlt 5, dsrl 2)
#   --run-id ID            run name (default <method>-<UTC time>)
#   --resume ID            continue rlt/dsrl run ID from its newest checkpoint
#   --learner-args "..."   extra learner flags (rlt-local online / lerobot-streaming-bc)
#   --actor-args "..."     extra rlt-submit-actor flags
#   --leaderboard NAME     after the last evaluation, submit the run to the leaderboard as
#                          method NAME (needs CELL and recorded evaluations; see
#                          scripts/leaderboard.py). Listed once both leaderboard cells are in.
#   --leaderboard-tag TAG  separate entry of the same method (default none: a new submission
#                          replaces the method's entry on that cell)
#   --leaderboard-code URL, --leaderboard-notes TEXT   shown with the method
#
# Logs: experiment-data/logs/<run id>/. Checkpoints: experiment-data/tasks/<config>/
# rlt_checkpoints/<run id> (rlt, dsrl) or experiment-data/lerobot-sbc/<run id> (dit).
# Evaluations: experiment-data/eval/<run id>-ep<episodes>/results.jsonl.
set -euo pipefail
METHOD=${1:-}
case "$METHOD" in rlt|dsrl|dit) shift ;; -h|--help|"") sed -n '2,38p' "$0"; exit 0 ;;
  *) echo "method must be rlt, dsrl or dit" >&2; exit 2 ;; esac
EPISODES=400 EVAL_EVERY=100 MODEL="" POLICY_TYPE=multi_task_dit STEPS="" STD=0.5 PENALTY=""
RUN="" RESUME="" LEARNER_ARGS="" ACTOR_ARGS="" BOARD_NAME="" BOARD_TAG="" BOARD_CODE="" BOARD_NOTES=""
while [ $# -gt 0 ]; do
  case "$1" in
    --episodes) EPISODES=$2 ;; --eval-every) EVAL_EVERY=$2 ;;
    --model) MODEL=$2 ;; --policy-type) POLICY_TYPE=$2 ;; --steps) STEPS=$2 ;;
    --std) STD=$2 ;; --trip-penalty) PENALTY=$2 ;; --run-id) RUN=$2 ;;
    --resume) RUN=$2; RESUME=--resume ;;
    --learner-args) LEARNER_ARGS=$2 ;; --actor-args) ACTOR_ARGS=$2 ;;
    --leaderboard) BOARD_NAME=$2 ;; --leaderboard-tag) BOARD_TAG=$2 ;;
    --leaderboard-code) BOARD_CODE=$2 ;; --leaderboard-notes) BOARD_NOTES=$2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift 2
done
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
actor_common
[ -z "$RESUME" ] || [ "$METHOD" != dit ] || die "--resume is for rlt and dsrl runs"
[ -z "$BOARD_NAME" ] || [ -n "$CELL" ] || die "--leaderboard needs CELL"
[ -z "$BOARD_NAME" ] || { [ "${RECORD:-1}" = 1 ] && [ -n "${RECORD_REPO:-}" ]; } \
  || die "--leaderboard needs recorded evaluations: set RECORD_REPO (scripts/setup.sh does)"
RUN=${RUN:-$METHOD-$(date -u +%Y%m%d-%H%M%S)}
LOGS=$DATA/logs/$RUN
mkdir -p "$LOGS"
DATASET=${DATASET:-AutoRLBench/busybox_push_green_button_426x240}
# Episode counts at which to evaluate: every EVAL_EVERY, and the last one.
if [ "$EVAL_EVERY" -gt 0 ]; then
  MARKS=$(seq "$EVAL_EVERY" "$EVAL_EVERY" "$EPISODES" | tr '\n' ' ')
  [ $((EPISODES % EVAL_EVERY)) -eq 0 ] || MARKS="$MARKS$EPISODES"
else
  MARKS=$EPISODES
fi
PIDS=() SERVER_PID="" ACTOR_LOG="" SEEN_SERVERS=0
cleanup() {
  local status=$?
  kill "${PIDS[@]}" "${EVAL_PIDS[@]}" $SERVER_PID 2>/dev/null || true
  [ -z "$ACTOR_LOG" ] || stop_cell_job "$ACTOR_LOG"
  [ -z "$EVAL_JOB_LOG" ] || stop_cell_job "$EVAL_JOB_LOG"
  exit $status
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if [ "$METHOD" = dit ]; then
  BIN=$LEROBOT_BIN
  ROUND=20  # episodes per learner round; one round = 300 learner steps
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
  log "dit run $RUN from $POLICY"
  SERVER=($BIN/rlt-local policy-server $GPU_ARG --config-name eval --policy-type lerobot
    --checkpoint "$POLICY" $STEERING $POLICY_TUNNEL)
  # run.sh stops the learner after the last evaluation; the step cap is never reached.
  $BIN/rlt-local lerobot-streaming-bc $GPU_ARG --run-id "$RUN" --policy-path "$POLICY" \
    --base-dataset "$DATASET" --base-batch-fraction 0.5 --min-shaped-reward 1.0 \
    --episodes-per-round $ROUND --steps-per-round 300 --online-steps 1000000000 \
    --idle-timeout-s 86400 $LEARNER_TUNNEL $WANDB_ARG $LEARNER_ARGS > "$LOGS/learner.log" 2>&1 &
  LEARNER=$!; PIDS+=($LEARNER)
  wait_for $LEARNER "$LOGS/learner.log" "\[lerobot sbc\] serving" learner
  SUBMIT=(--policy-type lerobot --config-name eval --learner-key "eval:$RUN" --exploration-scale 0
    --episodes-per-round $ROUND --steps-per-round 300 --round-wait-timeout-s 2400)
  OUT=$DATA/lerobot-sbc/$RUN
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
  SERVER=($OPENPI_BIN/rlt-local policy-server $GPU_ARG --config-name "$CONFIG" --checkpoint "$BASE"
    --prompt "$PROMPT" $SERVER_STEERING $POLICY_TUNNEL)
  EVERY=$([ "$EVAL_EVERY" -gt 0 ] && echo "$EVAL_EVERY" || echo "$EPISODES")
  $LEARNER_BIN/rlt-local online $GPU_ARG --config-name "$CONFIG" --run-id "$RUN" $RESUME \
    "${ALGO[@]}" --safety-trip-penalty "$PENALTY" --online-steps "$STEPS" --save-freq 10000 \
    --max-episodes "$EPISODES" --checkpoint-every-episodes "$EVERY" \
    --idle-timeout-s 0 --push-interval-s 5 $LEARNER_TUNNEL $WANDB_ARG $LEARNER_ARGS \
    >> "$LOGS/learner.log" 2>&1 &
  LEARNER=$!; PIDS+=($LEARNER)
  wait_for $LEARNER "$LOGS/learner.log" "rendezvous" learner
  SUBMIT=(--config-name "$CONFIG" --checkpoint-dir "$CELL_CHECKPOINT" --learner-key "$CONFIG:$RUN"
    --exploration-scale "$STD" --scene-retry-delay 3 --scene-retry-limit 1 "${ACTOR_ALGO[@]}")
  OUT=$DATA/tasks/$CONFIG/rlt_checkpoints/$RUN
fi

# The training policy server; it is stopped while an evaluation serves its checkpoint.
start_server() {
  "${SERVER[@]}" >> "$LOGS/policy-server.log" 2>&1 &
  SERVER_PID=$!
  until [ "$(grep -c "policy server" "$LOGS/policy-server.log" || true)" -gt "$SEEN_SERVERS" ]; do
    kill -0 "$SERVER_PID" 2>/dev/null || die "policy server exited; see $LOGS/policy-server.log"
    sleep 5
  done
  SEEN_SERVERS=$(grep -c "policy server" "$LOGS/policy-server.log" || true)
}
stop_server() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; SERVER_PID=""; }
episodes_done() { if [ -f "$OUT/rollout_metrics.jsonl" ]; then wc -l < "$OUT/rollout_metrics.jsonl"; else echo 0; fi; }
learner_alive() { kill -0 "$LEARNER" 2>/dev/null; }

# Prints the learner's checkpoint after <mark> training episodes, once it is saved.
checkpoint_at() {  # <mark>
  if [ "$METHOD" = dit ]; then
    # The round that ends at <mark> episodes; after its 300 steps the learner logs
    # "0 steps queued ... saved <dir>". A round without successes trains nothing.
    local dir
    until dir=$(python3 - "$LOGS/learner.log" "$(($1 / ROUND))" <<'PY'
import re, sys
lines = open(sys.argv[1]).read().splitlines()
mark = next((i for i, l in enumerate(lines) if re.match(rf"\[lerobot sbc\] round {sys.argv[2]}:", l)), None)
if mark is None:
    sys.exit(1)
if "no successes" in lines[mark]:
    print("newest"); sys.exit(0)
saved = [l for l in lines[mark:] if " 0 steps queued" in l and "saved " in l]
if not saved:
    sys.exit(1)
print(saved[0].split("saved ", 1)[1].strip())
PY
    ); do
      learner_alive || die "learner exited before episode $1; see $LOGS/learner.log"
      sleep 10
    done
    # No successes yet: the newest checkpoint, or the starting policy.
    [ "$dir" = newest ] && dir=$(ls -d "$OUT"/checkpoints/[0-9]* 2>/dev/null | sort -V | tail -1 || true)
    dir=${dir%/}; dir=${dir:-$POLICY}
    case "$dir" in */pretrained_model) echo "$dir" ;; *) echo "$dir/pretrained_model" ;; esac
  else
    local step
    until step=$(python3 -c '
import json, sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r["episodes"] == int(sys.argv[2]):
        print(r["step"]); sys.exit(0)
sys.exit(1)' "$OUT/episode_checkpoints.jsonl" "$1" 2>/dev/null); do
      learner_alive || die "learner exited before episode $1; see $LOGS/learner.log"
      sleep 10
    done
    freeze_rlt "$RUN" "$step"
  fi
}

# Sleeps up to N seconds, returning early once the learner has exited.
pause() { local i; for ((i = 0; i < $1; i += 10)); do learner_alive || return 0; sleep 10; done; }
ATTEMPT=$(find "$LOGS" -maxdepth 1 -name "actor.attempt*.log" | wc -l)
QUICK=0 TRIPS=0
start_server
log "learner and policy server up; logs in $LOGS; evaluations after episodes: $MARKS"
for MARK in $MARKS; do
  while [ "$(episodes_done)" -lt "$MARK" ]; do
    learner_alive || die "learner exited at $(episodes_done) episodes; see $LOGS/learner.log"
    wait_cell
    ATTEMPT=$((ATTEMPT + 1))
    ACTOR_LOG=$LOGS/actor.attempt$ATTEMPT.log
    LEFT=$((MARK - $(episodes_done)))
    log "actor attempt $ATTEMPT${CELL:+ on $CELL}: $LEFT episodes to $MARK"
    START=$(date +%s) BEFORE=$(episodes_done)
    record_args "${RUN}_train_$(episodes_done)"
    "${UVRUN[@]}" rlt-submit-actor "${ACTOR[@]}" "${SUBMIT[@]}" "${RECORD_ARGS[@]}" --num-rollouts "$LEFT" \
      --timeout-seconds 72000 $ID_ARGS $ACTOR_ARGS > "$ACTOR_LOG" 2>&1 || true
    stop_cell_job "$ACTOR_LOG"
    ELAPSED=$(( $(date +%s) - START ))
    # The last episodes reach the learner a little after the job ends.
    SENT=$(grep -c "rollout\] {" "$ACTOR_LOG" || true)
    for _ in $(seq 24); do [ "$(episodes_done)" -ge $((BEFORE + SENT)) ] && break; sleep 5; done
    if [ "$(episodes_done)" -ge "$MARK" ]; then
      QUICK=0 TRIPS=0
    elif grep -q "gate tripped" "$ACTOR_LOG"; then
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
      pause 5
    fi
    ACTOR_LOG=""
  done
  [ "$EVAL_EVERY" -gt 0 ] || break
  CKPT=$(checkpoint_at "$MARK")
  log "$MARK episodes: checkpoint $CKPT"
  stop_server
  if [ "$METHOD" = dit ]; then
    eval_checkpoint lerobot "$CKPT" "$RUN-ep$MARK" "eval:$RUN"
  else
    eval_checkpoint rlt "$CKPT" "$RUN-ep$MARK" "$BASE"
  fi
  [ "$MARK" = "$EPISODES" ] || start_server
done
stop_server
if [ "$METHOD" = dit ]; then
  kill "$LEARNER" 2>/dev/null || true
else
  # The online learner stops by itself after its last episode checkpoint.
  for _ in $(seq 60); do learner_alive || break; sleep 10; done
fi
log "done: $(episodes_done) training episodes; checkpoints in $OUT"
for MARK in $MARKS; do
  if [ -f "$DATA/eval/$RUN-ep$MARK/results.jsonl" ]; then
    echo "after $MARK episodes:"
    eval_summary "$DATA/eval/$RUN-ep$MARK"
  fi
done
if [ -n "$BOARD_NAME" ]; then
  $LEARNER_BIN/python scripts/leaderboard.py submit --name "$BOARD_NAME" --tag "$BOARD_TAG" --run "$RUN" --cell "$CELL" \
    --code "$BOARD_CODE" --notes "$BOARD_NOTES" || die "leaderboard submission failed"
fi
