#!/usr/bin/env bash
# usage: reward_models/robometer/train.sh <robometer checkout> <work dir>
# Builds AutoRLBench/robometer-4b-lora-busybox-demo39 from scratch:
# 1. the 39 demos (AutoRLBench/busybox_push_green_button_426x240), every frame labelled a success;
# 2. tiled into a Robometer dataset and pushed to <HF_USERNAME>/so101_busybox_demo39_tiled;
# 3. preprocessed;
# 4. LoRA fine-tune of robometer/Robometer-4B: 500 steps, batch 4 x 4, lr 2e-5, 8 frames of
#    640x540, discrete progress in 10 bins. About 5.5 h on one RTX A6000.
# The checkpoint is written to <work dir>/runs. Needs HF_TOKEN and HF_USERNAME, and the
# armnet-tiled branch of https://github.com/jesbu1/robometer with robometer.patch applied.
set -euo pipefail
RBM=$(cd "$1" && pwd)
mkdir -p "$2"
WORK=$(cd "$2" && pwd)
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=$RBM/.venv/bin/python
DS=${HF_USERNAME}_so101_busybox_demo39_tiled_so101_busybox_demo39_lerobot_tiled

# Short name busybox_demo39 for <HF_USERNAME>'s dataset; robometer.patch maps only AutoRLBench's.
MAP=$RBM/robometer/data/datasets/name_mapping.py
PATCHED='    "AutoRLBench_so101_busybox_demo39_tiled_so101_busybox_demo39_lerobot_tiled": "busybox_demo39",'
grep -qF "\"$DS\"" "$MAP" || sed -i "/^${PATCHED}\$/a\\    \"$DS\": \"busybox_demo39\"," "$MAP"
grep -qF "\"$DS\": \"busybox_demo39\"" "$MAP" || { echo "apply robometer.patch first" >&2; exit 1; }

# 1. The demos with a per-frame success column.
$PY - "$WORK/demo39-src" <<'PY'
import pathlib, shutil, sys
import pandas as pd
from huggingface_hub import snapshot_download
src = pathlib.Path(snapshot_download("AutoRLBench/busybox_push_green_button_426x240", repo_type="dataset"))
dst = pathlib.Path(sys.argv[1])
for name in ("meta", "videos"):
    if not (dst / name).exists():
        shutil.copytree(src / name, dst / name)
for path in (src / "data").rglob("*.parquet"):
    frame = pd.read_parquet(path)
    frame["success"] = True
    out = dst / "data" / path.relative_to(src / "data")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out)
PY

# 2. Tiled dataset.
cd "$WORK"
$PY -m dataset_upload.generate_hf_dataset --config_path "$HERE/demo39_tiled.yaml"
mkdir -p "$WORK/datasets"
ln -sfn "$WORK/demo39-tiled" "$WORK/datasets/so101_busybox_demo39_tiled"

# 3. Preprocessing.
cd "$RBM"
sed -e "s|AutoRLBench/so101_busybox_demo39_tiled|$HF_USERNAME/so101_busybox_demo39_tiled|" \
  -e "s|^cache_dir: .*|cache_dir: $WORK/processed|" "$HERE/preprocess_busybox_demo39.yaml" > "$WORK/preprocess.yaml"
ROBOMETER_DATASET_PATH=$WORK/datasets $PY -m robometer.data.scripts.preprocess_datasets --config "$WORK/preprocess.yaml"

# 4. Fine-tuning.
ROBOMETER_DATASET_PATH=$WORK/datasets ROBOMETER_PROCESSED_DATASETS_PATH=$WORK/processed WANDB_MODE=disabled $PY train.py \
  model.base_model_id=Qwen/Qwen3-VL-4B-Instruct model.use_peft=true model.use_unsloth=false \
  model.train_progress_head=true model.train_preference_head=true model.train_success_head=true \
  model.average_temporal_patches=true model.use_multi_image=true \
  model.progress_discrete_bins=10 model.progress_loss_type=discrete \
  loss.progress_discrete_bins=10 loss.progress_loss_type=discrete loss.success_positive_weight=1.0 \
  data.train_datasets=[$DS] data.eval_datasets=[$DS] data.max_frames=8 data.max_trajectories=-1 \
  data.resized_height=540 data.resized_width=640 data.use_per_frame_progress_token=true \
  data.dataset_success_cutoff_file=robometer/data/dataset_success_cutoff.txt \
  training.load_from_checkpoint=robometer/Robometer-4B \
  training.per_device_train_batch_size=4 training.gradient_accumulation_steps=4 \
  training.learning_rate=2e-5 training.warmup_ratio=0.1 training.weight_decay=0.01 \
  training.max_steps=500 training.output_dir=$WORK/runs \
  training.exp_name=rbm4b_lora_busybox_demo39 training.overwrite_output_dir=true \
  training.eval_steps=100 training.custom_eval_steps=100 \
  custom_eval.eval_types=[reward_alignment] custom_eval.reward_alignment=[$DS] custom_eval.pad_frames=true \
  logging.log_to=[] logging.save_best.metric_names=[eval_rew_align/pearson_busybox_demo39] \
  logging.save_best.greater_is_better=[true] logging.save_best.save_every=100 logging.save_best.upload_to_hub=false
