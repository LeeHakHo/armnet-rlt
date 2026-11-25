# Robometer reward model

`AutoRLBench/robometer-4b-lora-busybox-demo39`, the reward model of the "DSRL + Robometer"
baseline: [Robometer-4B](https://huggingface.co/robometer/Robometer-4B) tuned with LoRA on
the 39 teleoperated demonstrations of the task, the same ones the other baselines start from.
No robot rollouts go into it.

To rebuild it (about 5.5 h on one RTX A6000):

```bash
git clone -b armnet-tiled https://github.com/jesbu1/robometer && cd robometer && uv sync
uv pip install --python .venv/bin/python transformers==5.3.0
git apply <armnet-rlt>/reward_models/robometer/robometer.patch
export HF_TOKEN=<token> HF_USERNAME=<your HF user or org>
<armnet-rlt>/reward_models/robometer/train.sh . <work dir>
```

`train.sh`
1. labels every frame of `AutoRLBench/busybox_push_green_button_426x240` a success;
2. tiles the top, front and wrist views into one 640x540 frame (64 frames at 10 fps,
   `demo39_tiled.yaml`) and pushes the result to `<HF_USERNAME>/so101_busybox_demo39_tiled`;
3. preprocesses it (`preprocess_busybox_demo39.yaml`);
4. fine-tunes Robometer-4B with LoRA (rank 32, alpha 64): 500 steps, batch 4 x gradient
   accumulation 4, lr 2e-5 with 10 % warmup, 8 frames per trajectory, discrete progress in
   10 bins, progress, preference and success heads trained.

`robometer.patch` registers the dataset with Robometer: success cutoff 0.9 and a short
name for the evaluation metrics. The checkpoint lands in `<work dir>/runs`; serve it as in
the main README.
