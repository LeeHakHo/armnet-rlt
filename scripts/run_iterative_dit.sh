#!/usr/bin/env bash
# Same as `scripts/run.sh dit`; RUN_ID and ACTOR_ARGS map to --run-id and --actor-args.
exec "$(dirname "${BASH_SOURCE[0]}")/run.sh" dit ${RUN_ID:+--run-id "$RUN_ID"} \
  ${ACTOR_ARGS:+--actor-args "$ACTOR_ARGS"} "$@"
