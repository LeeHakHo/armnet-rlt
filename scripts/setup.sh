#!/usr/bin/env bash
# One-time setup; safe to run again (finished steps are skipped).
#   scripts/setup.sh [--model <HF repo or dir>] [--skip-image] [--skip-cache]
# 1. .env from .env.example, asking for ARMNET_API_KEY and HF_TOKEN
# 2. uv sync and the three envs/ environments (learner, openpi, lerobot)
# 3. the learner auth token and the two Armnet secrets
# 4. the base π0.5 checkpoint (default AutoRLBench/pi05_busybox_minmax_step5k_rltoken),
#    its normalization stats and the RLT demo cache (needs a GPU; --skip-cache skips it)
# 5. the actor image, written to .env as IMAGE_REF (needs Docker; --skip-image skips it)
set -euo pipefail
MODEL="" SKIP_IMAGE="" SKIP_CACHE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL=$2; shift 2 ;;
    --skip-image) SKIP_IMAGE=1; shift ;;
    --skip-cache) SKIP_CACHE=1; shift ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
command -v uv >/dev/null || die "install uv first: https://docs.astral.sh/uv/"

# Sets KEY=VALUE in .env, replacing an existing line.
set_env() {
  if grep -q "^$1=" .env; then sed -i.bak "s|^$1=.*|$1=$2|" .env && rm -f .env.bak
  else echo "$1=$2" >> .env; fi
  export "$1=$2"
}
ask() {  # <name> <prompt>
  local v="${!1:-}"
  case "$v" in ""|"<"*) ;; *) return 0 ;; esac
  [ -t 0 ] || die "$1 is not set in .env"
  read -r -s -p "$2: " v; echo
  [ -n "$v" ] || die "$1 is required"
  set_env "$1" "$v"
}

log "1/5 .env"
[ -f .env ] || cp .env.example .env
UVRUN=(uv run --quiet --env-file .env)
ask ARMNET_API_KEY "Armnet API key"
ask HF_TOKEN "Hugging Face token (read access to AutoRLBench)"
grep -q "^RLT_HF_SECRET=" .env || set_env RLT_HF_SECRET "$RLT_HF_SECRET"
grep -q "^RLT_AUTH_SECRET=" .env || set_env RLT_AUTH_SECRET "$RLT_AUTH_SECRET"

log "2/5 environments"
uv sync --locked --quiet
for env in learner openpi lerobot; do
  [ -x "envs/venv-$env/bin/rlt-local" ] && continue
  log "installing envs/venv-$env"
  scripts/setup_local.sh "$env"
done

log "3/5 secrets"
if [ -z "${RLT_LEARNER_AUTH_TOKEN:-}" ]; then
  TOKEN=$(openssl rand -hex 32)
  "${UVRUN[@]}" armnet secret create "$RLT_AUTH_SECRET" "$TOKEN"
  set_env RLT_LEARNER_AUTH_TOKEN "$TOKEN"
  "${UVRUN[@]}" armnet secret create "$RLT_HF_SECRET" "$HF_TOKEN" \
    || log "could not create $RLT_HF_SECRET; it may already exist"
fi
if [ -z "${PUBLIC_HOST:-}" ] && ! ngrok config check >/dev/null 2>&1; then
  log "ngrok is not configured: install it and run 'ngrok config add-authtoken <token>'"
  log "(or set PUBLIC_HOST, TLS_CERT and TLS_KEY in .env to use your own TLS ingress)"
fi

log "4/5 base checkpoint, normalization stats and demo cache"
BASE=$(resolve_model "$MODEL")
log "base checkpoint $BASE"
[ -d "$DATA/tasks/$CONFIG/ckpt/assets" ] || $LEARNER_BIN/rlt-local assets --config-name "$CONFIG"
CACHE=$DATA/tasks/$CONFIG/demo/rlt_demo_cache.pt
if [ -f "$CACHE" ]; then
  log "demo cache $CACHE"
elif [ -n "$SKIP_CACHE" ]; then
  log "skipping the demo cache; RLT then starts without demo pretraining"
else
  $OPENPI_BIN/rlt-local cache $GPU_ARG --preset multitask --checkpoint "$BASE"
fi

log "5/5 actor image"
if [ -n "$SKIP_IMAGE" ]; then
  log "skipping the image build"
elif ! docker info >/dev/null 2>&1; then
  log "Docker is not available here. On a machine with Docker and this checkout, run"
  log "  uv run python scripts/build_actor_image.py"
  log "and put the printed ref in .env as IMAGE_REF=<ref>"
elif [ -n "${IMAGE_REF:-}" ] && [ "$("${UVRUN[@]}" python scripts/build_actor_image.py --dry-run | tail -1 || true)" = "$IMAGE_REF" ]; then
  log "image $IMAGE_REF is current"
else
  REF=$("${UVRUN[@]}" python scripts/build_actor_image.py | tail -1)
  set_env IMAGE_REF "$REF"
  log "image $REF"
fi
log "done. Next: CELL=<cell-id> scripts/run.sh dsrl"
