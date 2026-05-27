# Colab Free quickstart — Blox AI LoRA fine-tune on T4

You have a free Colab GPU runtime (T4, 15 GB VRAM, 12.7 GB system RAM, 112 GB disk). This page walks you through fine-tuning Qwen 3 1.7B on the synthetic corpus end-to-end. Estimated time: **~2-3 hours** including model download.

Each cell below is a separate Colab notebook cell. Paste, hit ▶ (or Shift+Enter), wait for the green ✓, then move to the next.

## Cell 1 — Confirm GPU is allocated

```
!nvidia-smi
```

Expect to see something like `Tesla T4` or `NVIDIA L4` with 15 GB total memory. If it says "No GPU available", go to **Runtime → Change runtime type → T4 GPU** and try again.

## Cell 2 — Clone the repo on the dataset branch

```
!git clone --branch qwen3-1.7b-synthetic-dataset https://github.com/functionland/fula-ai-training.git
%cd fula-ai-training
```

That puts you inside `fula-ai-training/` with the 174 synthetic training files already in `corpus/labelled/`.

## Cell 3 — Install the Python deps

```
!pip install -q transformers peft datasets trl pyyaml accelerate
```

`torch` is pre-installed on Colab; the rest pull from PyPI. Takes ~30 seconds.

## Cell 4 — Mount Google Drive so the trained model survives

```
from google.colab import drive
drive.mount('/content/drive')
```

A popup asks for permission. Click through, sign in, paste the auth code back. The LoRA adapter weights will save to `/content/drive/MyDrive/blox-ai-training/` so you keep them after the Colab session ends.

## Cell 5 — Dry-run to confirm the corpus parses

```
!python -m training.train_qwen3_messages \
    --config training/configs/qwen3_1_7b_colab_t4.yaml \
    --dry-run
```

Expected output ends with: `INFO loaded 174 messages-format examples (0 skipped)`. If you see `min_transcripts` errors, the corpus didn't clone properly — re-run Cell 2.

## Cell 6 — Verify the loss mask before the multi-hour run

```
!python -m training.verify_mask \
    --config training/configs/qwen3_1_7b_colab_t4.yaml \
    --sample 0
```

First run downloads the Qwen3-1.7B tokenizer (~10 MB). The output shows the full rendered prompt and what the mask should isolate. Just confirms the pipeline works end-to-end before you commit to the long training run.

## Cell 7 — Pilot training: 1 epoch first

Per `corpus/labelled/SUCCESS_METRICS.md`, run **1 epoch first** so you catch problems quickly. Edit the config to `num_epochs: 1`:

```
import yaml
with open('training/configs/qwen3_1_7b_colab_t4.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['training']['num_epochs'] = 1
with open('training/configs/qwen3_1_7b_colab_t4_pilot.yaml', 'w') as f:
    yaml.safe_dump(cfg, f)
print("Wrote pilot config with num_epochs=1")
```

Then run:

```
!python -m training.train_qwen3_messages \
    --config training/configs/qwen3_1_7b_colab_t4_pilot.yaml
```

First run downloads Qwen3-1.7B weights (~3.4 GB) — takes 3-10 minutes depending on Colab's network. Then 1 epoch trains in **~25-40 minutes** on T4. Watch the loss go down in the log output.

## Cell 8 — Confirm the pilot adapter saved

```
!ls -la /content/drive/MyDrive/blox-ai-training/
```

You should see a `<YYYYMMDD>-<gitsha>/` dir containing `adapter/` (the LoRA weights), `training_args.json`, and `manifest.json`. If the dir exists, the pilot worked.

## Cell 9 — Full training: 3 epochs

If the pilot's loss dropped reasonably (final loss < initial × 0.7 is a good sign), commit to the full 3-epoch run:

```
!python -m training.train_qwen3_messages \
    --config training/configs/qwen3_1_7b_colab_t4.yaml
```

**~1.5-2 hours** on T4. Don't close the laptop. (Colab disconnects idle sessions after ~90 minutes; the active training keeps it alive.)

## Cell 10 — When training finishes

```
!ls -la /content/drive/MyDrive/blox-ai-training/
!du -sh /content/drive/MyDrive/blox-ai-training/*/adapter/
```

The newest `<date>-<sha>/adapter/` directory has your LoRA weights (~50 MB).

To download to your local machine: open Google Drive in a browser, find the `blox-ai-training` folder, right-click the relevant `<date>-<sha>` subfolder → Download. Or use the Colab file browser (folder icon, left sidebar) → right-click → Download.

## What to do next

The adapter alone isn't ready to deploy to a Blox device — you need to merge it into the base model + convert to RKLLM W8A8 + canary on the lab. That flow is on a different machine (the Rockchip RKLLM toolkit is finicky; do it on your own Linux box). See:

- `corpus/labelled/SUCCESS_METRICS.md` — pilot path + eval thresholds
- `OPERATIONS.md` — full cycle walkthrough (step 4 onwards: merge → quantize → canary → publish)

## Troubleshooting

**"CUDA out of memory" mid-training**
- Lower `per_device_train_batch_size` from `2` to `1` in the config, and bump `gradient_accumulation_steps` from `16` to `32` (keeps effective batch = 32). Re-run.

**"Repository not found" cloning the repo**
- The dataset branch hasn't been merged yet. Either clone main + checkout the branch manually:
  ```
  !git clone https://github.com/functionland/fula-ai-training.git
  %cd fula-ai-training
  !git checkout qwen3-1.7b-synthetic-dataset
  ```

**Session disconnects mid-training**
- Colab Free has a ~12-hour hard cap and ~90-minute idle disconnect. Active training keeps it alive, but if you close the laptop or wifi blips for 90+ minutes, the session dies and the run is lost. Mitigations:
  - Don't close the laptop during training
  - Save checkpoints frequently (already set: `save_steps: 50` in the config — every 50 steps writes to Drive)
  - If a checkpoint is in Drive, you can resume by setting `--resume_from_checkpoint <path>` (TODO: wire into the trainer if it becomes a frequent problem)

**Training never finishes / hangs**
- Press **Runtime → Interrupt execution**, then check the log output. The most common cause is the Qwen tokenizer download timing out behind a corporate firewall — re-run Cell 7 (it resumes the download).

**Loss not decreasing**
- After 50-100 steps, the training loss should drop noticeably (say from 1.5 → 0.8 range, depending on init). If it's stuck at 1.5+, your learning rate may be too low — bump `learning_rate: 1.0e-4` to `2.0e-4` and try again. If it's NaN or extremely high, the gradient checkpointing + bf16 combo may be misbehaving — set `bf16: false` (falls back to fp16) and re-run.
