"""Build the single-arm RLT demonstration cache on Modal."""

from __future__ import annotations

import modal

from armnet_rlt.resources import VOLUME_MOUNT, VOLUME_NAME, task_paths


OPENPI_REVISION = "e0eb11f08ffcef37809ba72d72b4ce5fb68ca0f7"
LEROBOT_VERSION = "0.5.1"
CONFIG_NAME = "pi05_rlt_busybox_push_green_button"
CHECKPOINT_REPO = "pravsels/pi05_rlt_busybox_push_green_button"
DATASET_REPO = "villekuosmanen/busybox_push_green_button"
PROMPT = "push the green button"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04", add_python="3.12"
    )
    .env(
        {
            "PYTHONPATH": "/root",
            "HF_HOME": f"{VOLUME_MOUNT}/hf-cache",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
        }
    )
    .apt_install(
        "git",
        "ffmpeg",
        "ca-certificates",
        "libgl1",
        "libglib2.0-0",
    )
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel uv",
        "printf '%s\\n' "
        "'pynput>=0.0.0; sys_platform == \"never\"' "
        "'evdev>=0.0.0; sys_platform == \"never\"' "
        "'torchcodec>=0.0.0; sys_platform == \"never\"' "
        "'torch==2.8.0' "
        "'torchvision==0.23.0' "
        "'transformers>=4.57,<5.0' "
        "'huggingface-hub>=0.30,<1.0' "
        "'numpy>=2.0,<2.3' "
        "'fsspec>=2024.6.0,<=2026.2.0' "
        "> /tmp/overrides.txt",
        "uv pip install --system --override /tmp/overrides.txt "
        "--extra-index-url https://download.pytorch.org/whl/cu129 "
        "--index-strategy unsafe-best-match "
        "'transformers[vision]>=4.57,<5.0' accelerate diffusers einops "
        "'huggingface-hub>=0.30,<1.0' numpy pandas pyarrow scipy "
        "opencv-python-headless safetensors av "
        "augmax dm-tree equinox flatbuffers flax==0.10.2 "
        "'fsspec>=2024.6.0,<=2026.2.0' gcsfs imageio 'jaxlib==0.5.3' "
        "'jax[cuda12]==0.5.3' jaxtyping==0.2.36 ml_collections==1.0.0 "
        "'numpydantic>=1.6.6' orbax-checkpoint==0.11.13 sentencepiece "
        "'tqdm-loggable>=0.2' 'tyro>=0.9.5' wandb beartype==0.19.0 "
        "'treescope>=0.1.7' rich chex "
        f"lerobot=={LEROBOT_VERSION}",
        "git clone --recurse-submodules https://github.com/pravsels/openpi "
        "/opt/openpi && "
        f"git -C /opt/openpi checkout {OPENPI_REVISION} && "
        "git -C /opt/openpi submodule update --init --recursive && "
        "python -m pip install --no-deps "
        "/opt/openpi/packages/openpi-client /opt/openpi",
        # Torch last so OpenPI/LeRobot cannot leave a CPU wheel behind.
        "uv pip install --upgrade --system --override /tmp/overrides.txt "
        "--extra-index-url https://download.pytorch.org/whl/cu129 "
        "--index-strategy unsafe-best-match "
        "torch==2.8.0 torchvision==0.23.0",
    )
    .add_local_dir(
        "src/armnet_rlt",
        remote_path="/root/armnet_rlt",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc"],
    )
    .run_commands(
        "python -c \"from armnet_rlt.openpi_rlt import _install_training_stubs; "
        "_install_training_stubs(); "
        "from openpi.training.config import get_config; "
        f"get_config('{CONFIG_NAME}'); "
        "from lerobot.datasets.lerobot_dataset import LeRobotDataset; "
        "print('RLT cache image imports OK')\""
    )
)

app = modal.App("armnet-rlt-cache")


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={str(VOLUME_MOUNT): volume},
    secrets=[hf_secret],
    timeout=4 * 60 * 60,
)
def build_green_button_cache(inference_batch_size: int = 4) -> dict:
    """Build and persist the cache and matching normalization assets."""
    from huggingface_hub import snapshot_download
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from armnet_rlt.cache_builder import build_demo_cache, copy_norm_stats
    from armnet_rlt.openpi_rlt import OpenPIRLTPolicy

    checkpoint = snapshot_download(CHECKPOINT_REPO)
    print(f"[RLT cache] checkpoint={checkpoint}", flush=True)
    policy = OpenPIRLTPolicy.from_checkpoint(
        checkpoint,
        CONFIG_NAME,
        default_prompt=PROMPT,
    )
    dataset = LeRobotDataset(DATASET_REPO, video_backend="pyav")
    cache_path, assets_dir, _output_dir = task_paths(CONFIG_NAME)
    summary = build_demo_cache(
        dataset=dataset,
        policy=policy,
        output_path=cache_path,
        prompt=PROMPT,
        inference_batch_size=inference_batch_size,
    )
    copy_norm_stats(checkpoint, assets_dir)
    volume.commit()
    result = {
        **summary,
        "config_name": CONFIG_NAME,
        "dataset": DATASET_REPO,
        "checkpoint": CHECKPOINT_REPO,
        "cache_path": cache_path,
        "assets_dir": assets_dir,
    }
    print(f"[RLT cache] complete: {result}", flush=True)
    return result
