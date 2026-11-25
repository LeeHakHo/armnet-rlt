#!/usr/bin/env bash
# Installs the plugins under plugins/ into the envs/ environments: LeRobot policies
# (plugins/lerobot_policy_*) into lerobot, RL algorithms (plugins/rl_algorithm_*) into
# learner and openpi. Environments that do not exist are skipped.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
install() {  # <plugin> <env>
  [ -f "$1/pyproject.toml" ] && [ -x "envs/venv-$2/bin/python" ] || return 0
  uv pip install --quiet --python "envs/venv-$2" --no-deps -e "$1"
}
for plugin in plugins/lerobot_policy_*/; do install "$plugin" lerobot; done
for plugin in plugins/rl_algorithm_*/; do
  for env in learner openpi; do install "$plugin" "$env"; done
done
