#!/usr/bin/env bash
# Iterative BC of a multi-task DiT on one Armnet cell, end to end:
#   1. trains the starting policy on the demo dataset with lerobot-train
#      (PRETRAIN_STEPS steps) unless that checkpoint already exists,
#   2. starts the LeRobot policy server and the streaming-BC learner on the GPU,
#   3. submits the actor, and submits it again after a failure, until the learner
#      reaches its last step.
# Env: IMAGE_REF (required), RUN_ID, GPU (index; unset uses the visible GPU),
#      DATASET, PRETRAIN_STEPS (300), ROLLOUTS (180),
#      LEARNER_TUNNEL ("--tunnel ngrok-tcp"), POLICY_TUNNEL ("--tunnel ngrok"),
#      ACTOR_ARGS (extra rlt-submit-actor arguments, e.g. --armnet.require-cell
#      and the ID flags), MAX_ATTEMPTS (20).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -f .env ] && { set -a; . ./.env; set +a; }
: "${IMAGE_REF:?set IMAGE_REF to the actor image ref}"
RUN_ID=${RUN_ID:-idit-$(date -u +%Y%m%d-%H%M%S)}
GPU=${GPU:-}
GPU_ARG=${GPU:+--gpu $GPU}
DATASET=${DATASET:-AutoRLBench/busybox_push_green_button_426x240}
PRETRAIN_STEPS=${PRETRAIN_STEPS:-300}
ROLLOUTS=${ROLLOUTS:-180}
LEARNER_TUNNEL=${LEARNER_TUNNEL:---tunnel ngrok-tcp}
POLICY_TUNNEL=${POLICY_TUNNEL:---tunnel ngrok}
ACTOR_ARGS=${ACTOR_ARGS:-}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-20}
DATA=${RLT_DATA_ROOT:-$ROOT/experiment-data}
LOGS=$DATA/logs/$RUN_ID
BIN=envs/venv-lerobot/bin
mkdir -p "$LOGS"
log() { echo "[iterative-dit $(date -u +%H:%M:%S)] $*"; }

# 1. Starting policy.
PRE_DIR=$DATA/dit/demo-s$PRETRAIN_STEPS
POLICY=$PRE_DIR/checkpoints/$(printf %06d "$PRETRAIN_STEPS")/pretrained_model
if [ ! -f "$POLICY/config.json" ]; then
  log "training the starting policy on $DATASET for $PRETRAIN_STEPS steps -> $PRE_DIR"
  env ${GPU:+CUDA_VISIBLE_DEVICES=$GPU} $BIN/lerobot-train \
    --dataset.repo_id="$DATASET" \
    --policy.type=multi_task_dit --policy.device=cuda --policy.push_to_hub=false \
    --policy.image_resize_shape='[240, 240]' \
    --output_dir="$PRE_DIR" --job_name="demo-s$PRETRAIN_STEPS" \
    --steps="$PRETRAIN_STEPS" --batch_size=32 --num_workers=8 \
    --save_freq="$PRETRAIN_STEPS" --eval_freq=0 --wandb.enable=false \
    > "$LOGS/pretrain.log" 2>&1
fi
log "starting policy $POLICY"

# 2. Policy server and learner.
PIDS=()
trap 'kill "${PIDS[@]}" 2>/dev/null || true' EXIT
$BIN/rlt-local policy-server $GPU_ARG --config-name eval --policy-type lerobot \
  --checkpoint "$POLICY" --steering-from-learner 127.0.0.1:50051 $POLICY_TUNNEL \
  > "$LOGS/policy-server.log" 2>&1 &
PIDS+=($!)
$BIN/rlt-local lerobot-streaming-bc $GPU_ARG --run-id "$RUN_ID" --policy-path "$POLICY" \
  --base-dataset "$DATASET" --base-batch-fraction 0.5 --min-shaped-reward 1.0 \
  --idle-timeout-s 1800 $LEARNER_TUNNEL > "$LOGS/learner.log" 2>&1 &
LEARNER=$!
PIDS+=($LEARNER)
until grep -q "\[lerobot sbc\] serving" "$LOGS/learner.log" 2>/dev/null \
  && grep -q "policy server" "$LOGS/policy-server.log" 2>/dev/null; do
  kill -0 "$LEARNER" 2>/dev/null || { log "learner exited; see $LOGS/learner.log"; exit 1; }
  sleep 10
done
log "learner and policy server up (logs in $LOGS)"

# 3. Actor.
ATTEMPT=0
while kill -0 "$LEARNER" 2>/dev/null && ! grep -q "\[lerobot sbc\] complete" "$LOGS/learner.log"; do
  ATTEMPT=$((ATTEMPT + 1))
  [ "$ATTEMPT" -le "$MAX_ATTEMPTS" ] || { log "stopping after $MAX_ATTEMPTS actor attempts"; exit 1; }
  log "actor attempt $ATTEMPT"
  uv run --env-file .env rlt-submit-actor --policy-type lerobot \
    --task push_green_button --config-name eval \
    --language-instruction "push the green button" --learner-key "eval:$RUN_ID" \
    --num-rollouts "$ROLLOUTS" --exploration-scale 0 \
    --episodes-per-round 30,30,20 --steps-per-round 300 --round-wait-timeout-s 2400 \
    --timeout-seconds 72000 --image-ref "$IMAGE_REF" $ACTOR_ARGS \
    > "$LOGS/actor.attempt$ATTEMPT.log" 2>&1 || true
  sleep 60
done
log "learner finished: $(grep -o "\[lerobot sbc\] complete.*" "$LOGS/learner.log" | cut -c1-200)"
log "checkpoints in $DATA/lerobot-sbc/$RUN_ID/checkpoints"
