#!/usr/bin/env bash
# Build the `rlt-local` environments under tools/.
#   scripts/setup_local.sh learner   torch + LeRobot transport (bc, streaming-bc)
#   scripts/setup_local.sh openpi    OpenPI/JAX (cache, bc-cache)
#   scripts/setup_local.sh act       LeRobot ACT/SmolVLA training and armnet-lerobot-eval
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="$ROOT/tools"
LEROBOT="$ROOT/third_party/lerobot"
OPENPI_REVISION=90490b9c42a1accafde514f7ee610fbd7bc53376
TORCH_INDEX=https://download.pytorch.org/whl/cu129
export UV_HTTP_TIMEOUT=600

mkdir -p "$TOOLS"

setup_learner() {
    local venv="$TOOLS/venv-learner" overrides="$TOOLS/learner-overrides.txt"
    uv venv --python 3.12 --allow-existing "$venv"
    printf '%s\n' \
        'pynput>=0.0.0; sys_platform == "never"' \
        'evdev>=0.0.0; sys_platform == "never"' \
        'torchcodec>=0.0.0; sys_platform == "never"' \
        'torch==2.8.0' \
        'torchvision==0.23.0' \
        'numpy>=2.0,<2.3' \
        'grpcio==1.73.1' \
        'protobuf>=6.31.1,<6.32.0' \
        'setuptools>=71,<81' \
        > "$overrides"
    uv pip install --python "$venv" --override "$overrides" \
        grpcio==1.73.1 'protobuf>=6.31.1,<6.32.0' \
        'numpy>=2.0,<2.3' tqdm 'wandb>=0.24,<0.25' \
        'huggingface-hub>=1,<2' python-dotenv
    uv pip install --python "$venv" --no-deps --override "$overrides" \
        -e "$LEROBOT"
    uv pip install --python "$venv" --no-deps -e "$ROOT"
    # Install the CUDA torch wheels.
    uv pip install --python "$venv" --upgrade --override "$overrides" \
        --extra-index-url "$TORCH_INDEX" \
        torch==2.8.0 torchvision==0.23.0
    "$venv/bin/python" -c "import torch; from lerobot.rl.learner_service import LearnerService; from armnet_rlt.learner import run_streaming_bc; print('RLT learner env OK', torch.__version__, 'cuda', torch.cuda.is_available())"
}

setup_openpi() {
    local venv="$TOOLS/venv-openpi" overrides="$TOOLS/openpi-overrides.txt"
    local openpi="$TOOLS/openpi"
    uv venv --python 3.12 --seed --allow-existing "$venv"
    printf '%s\n' \
        'pynput>=0.0.0; sys_platform == "never"' \
        'evdev>=0.0.0; sys_platform == "never"' \
        'torchcodec>=0.0.0; sys_platform == "never"' \
        'torch==2.8.0' \
        'torchvision==0.23.0' \
        'transformers>=4.57,<5.0' \
        'huggingface-hub>=0.30,<1.0' \
        'numpy>=2.0,<2.3' \
        'fsspec>=2024.6.0,<=2026.2.0' \
        > "$overrides"
    uv pip install --python "$venv" --override "$overrides" \
        --extra-index-url "$TORCH_INDEX" \
        --index-strategy unsafe-best-match \
        'transformers[vision]>=4.57,<5.0' accelerate diffusers einops \
        'huggingface-hub>=0.30,<1.0' numpy pandas pyarrow scipy \
        opencv-python-headless safetensors av \
        augmax dm-tree equinox flatbuffers flax==0.10.2 \
        'fsspec>=2024.6.0,<=2026.2.0' gcsfs imageio 'jaxlib==0.5.3' \
        'jax[cuda12]==0.5.3' jaxtyping==0.2.36 ml_collections==1.0.0 \
        'numpydantic>=1.6.6' orbax-checkpoint==0.11.13 sentencepiece \
        'tqdm-loggable>=0.2' 'tyro>=0.9.5' wandb beartype==0.19.0 \
        'treescope>=0.1.7' rich chex python-dotenv \
        -e "$LEROBOT"
    if [ ! -d "$openpi/.git" ]; then
        git clone --recurse-submodules https://github.com/pravsels/openpi "$openpi"
    fi
    git -C "$openpi" fetch --quiet origin
    git -C "$openpi" checkout --quiet "$OPENPI_REVISION"
    git -C "$openpi" submodule update --init --recursive
    uv pip install --python "$venv" --no-deps \
        "$openpi/packages/openpi-client" "$openpi"
    uv pip install --python "$venv" --no-deps -e "$ROOT"
    # Install the CUDA torch wheels.
    uv pip install --python "$venv" --upgrade --override "$overrides" \
        --extra-index-url "$TORCH_INDEX" \
        --index-strategy unsafe-best-match \
        torch==2.8.0 torchvision==0.23.0
    "$venv/bin/python" -c "from armnet_rlt.openpi_rlt import _install_training_stubs; _install_training_stubs(); from openpi.training.config import get_config; get_config('pi05_rlt_busybox_multitask_singlearm_minmax'); from lerobot.datasets.lerobot_dataset import LeRobotDataset; print('RLT cache env OK')"
}

setup_act() {
    local venv="$TOOLS/venv-act" overrides="$TOOLS/act-overrides.txt"
    uv venv --python 3.12 --allow-existing "$venv"
    printf '%s\n' \
        'pynput>=0.0.0; sys_platform == "never"' \
        'evdev>=0.0.0; sys_platform == "never"' \
        'torchcodec>=0.0.0; sys_platform == "never"' \
        'torch==2.8.0' \
        'torchvision==0.23.0' \
        > "$overrides"
    uv pip install --python "$venv" --override "$overrides" \
        --extra-index-url "$TORCH_INDEX" \
        --index-strategy unsafe-best-match \
        -e "$LEROBOT[smolvla]" armnet-client==0.3.5 'wandb>=0.24,<0.25' python-dotenv \
        torch==2.8.0 torchvision==0.23.0
    "$venv/bin/python" -c "import torch; from lerobot.policies.act.modeling_act import ACTPolicy; from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; import armnet_client.lerobot_eval; print('ACT env OK', torch.__version__)"
}

case "${1:-}" in
    learner) setup_learner ;;
    openpi) setup_openpi ;;
    act) setup_act ;;
    *) echo "usage: $0 learner|openpi|act" >&2; exit 2 ;;
esac
