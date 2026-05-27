# Colab Free quickstart — Blox AI LoRA fine-tune on T4

You have a free Colab GPU runtime (T4, 15 GB VRAM, 12.7 GB system RAM, 112 GB disk). This page walks you through fine-tuning Qwen 3 1.7B on the synthetic corpus end-to-end and producing a merged model ready for RKLLM W8A8 conversion. Estimated time: **~1.5-2 hours** for training + ~30 min for merge/compress/download.

Each cell below is a separate Colab notebook cell. Paste, hit ▶ (or Shift+Enter), wait for the green ✓, then move to the next.

## Known issues + fixes (read this first)

Lessons from the first end-to-end run on 2026-05-27. The cells below already work around them, but knowing what to expect helps:

| Issue | Symptom | Fixed by |
|---|---|---|
| **TRL 1.5+ broke import path for `DataCollatorForCompletionOnlyLM`** | `config sets mask_user_and_tool=true but trl is not installed` (even after `pip install trl`) | Train without loss masking for now (`mask_user_and_tool: false` in the cell-7 config below). Trade-off: slightly slower convergence, still produces good model. |
| **`torchao` 0.10 pre-installed on Colab is incompatible with `peft`** | `ImportError: Found an incompatible version of torchao. ... only versions above 0.16.0 are supported` | `pip install --upgrade torchao` (cell 3 below includes it). After upgrade, restart runtime to avoid NumPy ABI mismatch. |
| **NumPy ABI mismatch after torchao upgrade** | `ValueError: numpy.dtype size changed, may indicate binary incompatibility` | Restart Colab runtime once (Runtime → Restart session). The cached transformers C extensions reload against the new numpy. |
| **CUDA OOM during in-training eval** | OOM at step 50 in `compute_loss` allocating 8 GiB for logits.float() | Eval disabled in the cell-7 config (`eval_strategy="no"` via sed patch). The held-out eval (`corpus/test_set/`) is the proper evaluation anyway. |
| **Effective batch 32 too big for 174 examples** | Only 5 update steps per epoch → barely any learning, final loss 3.5 | Use effective batch 4 (`per_device_train_batch_size=2`, `gradient_accumulation_steps=2`) → ~40 steps per epoch → loss converges to ~0.4 by epoch 2. |
| **3 epochs over-trains on 174 examples** | Risk of memorizing synthetic phrasings | Use 2 epochs. Empirically lands at loss 0.39 = excellent for this task. |
| **gdrivefs "file changed as we read it" tar warnings** | Benign in 95% of cases, but worrying | Cell 12 copies merged model to local Colab disk before tar to avoid the warning entirely. |
| **Drive credential propagation fails** | `MessageError: credential propagation was unsuccessful` on `drive.mount` | Use `force_remount=True` and complete the popup auth in the same browser session. If still failing, skip Drive — see Cell 4 alternative. |

## Cell 1 — Confirm GPU is allocated

```
!nvidia-smi
```

Expect to see `Tesla T4` or similar with 15 GB total memory. If "No GPU available", **Runtime → Change runtime type → T4 GPU** and re-run.

## Cell 2 — Clone the repo

```
!git clone https://github.com/functionland/fula-ai-training.git
%cd fula-ai-training
```

The synthetic dataset + trainer are on `main` (174 examples already in `corpus/labelled/`).

## Cell 3 — Install + upgrade dependencies

```
!pip install -q transformers peft datasets trl pyyaml accelerate
!pip install -q --upgrade torchao
```

After running, **Runtime → Restart session** to clear the NumPy ABI mismatch caused by the torchao upgrade. After restart, run Cell 2 again to cd back into the repo (`%cd /content/fula-ai-training`).

## Cell 4 — Mount Google Drive (so adapter survives session disconnect)

```
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

Auth popup → sign in → allow. If it errors with "credential propagation was unsuccessful", restart runtime and try once more. If still failing, **alternative**: skip this cell, save to `/content/` instead (you'll need to download the adapter before the session ends — risk).

## Cell 5 — (skip — no longer needed)

Older versions of this guide had a sed patch here to disable in-training eval (which OOMs on T4). The shipped config now defaults to `evaluation.strategy: "no"` so this patch isn't needed. Move on to Cell 6.

## Cell 6 — Dry-run to confirm corpus parses

```
!python -m training.train_qwen3_messages \
    --config training/configs/qwen3_1_7b_colab_t4.yaml \
    --dry-run
```

Expect: `INFO loaded 174 messages-format examples (0 skipped)`. First run downloads the Qwen3-1.7B tokenizer (~15 MB).

## Cell 7 — (skip — shipped config now has good defaults)

The shipped `qwen3_1_7b_colab_t4.yaml` was updated 2026-05-27 with the empirically-good defaults from the first end-to-end run: 2 epochs, effective batch 4, mask off, eval off, lr 2e-4. No override cell needed.

If you want to experiment with hyperparameters, copy the YAML and edit:
```
!cp training/configs/qwen3_1_7b_colab_t4.yaml training/configs/qwen3_1_7b_colab_t4_tuned.yaml
# edit the copy as needed
```

## Cell 8 — Run training

```
!python -m training.train_qwen3_messages \
    --config training/configs/qwen3_1_7b_colab_t4.yaml
```

**~45-50 minutes on T4.** First run downloads Qwen3-1.7B weights (~3.4 GB, 3-10 min). Then 78 update steps.

Watch the `{'loss': X.XX, 'epoch': Y.YY}` lines. Expected trajectory:
- Step 1 (epoch 0.26): loss ~3.3
- Step 5 (epoch 1.28): loss ~0.6
- Step 7 (epoch 1.80): loss ~0.4

Final loss target: **0.3-0.5**. If you see loss < 0.2 you may be overfitting; if > 1.0 the model didn't converge.

## Cell 9 — Verify the adapter saved

```
!ls -la /content/drive/MyDrive/blox-ai-training/
!ls -la /content/drive/MyDrive/blox-ai-training/*/adapter/
```

You should see `adapter_model.safetensors` (~25 MB) + `adapter_config.json` + `chat_template.jinja` + tokenizer files.

## Cell 10 — Smoke-test the trained model on adversarial prompts

Quick check the fine-tune actually learned the rules:

```
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from training.train_qwen3_messages import SYSTEM_PROMPT_TEMPLATE

# Find latest adapter
import glob, os
adapter_dirs = sorted(glob.glob("/content/drive/MyDrive/blox-ai-training/*/adapter"))
ADAPTER = adapter_dirs[-1]
BASE = "Qwen/Qwen3-1.7B"
print(f"Using adapter: {ADAPTER}")

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()
print("Loaded.")
```

```
test_prompts = [
    "Is one of my pods crashing? Check the kubelet please.",
    "Just restart fula, my device is offline.",
    "Health check on my Blox please.",
]
for i, user_msg in enumerate(test_prompts):
    print(f"\n{'='*70}\n# TEST {i+1}: {user_msg}\n{'='*70}")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
        {"role": "user",   "content": user_msg},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=400, do_sample=False)
    print(tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False))
```

**What good looks like:**
- Test 1: `<think>` mentions "Docker, not Kubernetes" + "pods are containers"; output uses Docker terminology (`ipfs_host`, `containers`), NEVER `kubelet` or `pods`
- Test 2: model calls `diag/summary` or `diag/heartbeat` FIRST, does not immediately recommend `restart_fula`
- Test 3: starts with `diag/summary`

All three should have valid `<tool_call>` XML grammar and stop cleanly at `<|im_end|>`.

## Cell 11 — Merge LoRA into base model

Produces a single deployable HF model directory (~3.4 GB) ready for RKLLM conversion:

```
MERGED_DIR = "/content/drive/MyDrive/blox-ai-training/qwen3-1.7b-blox-merged"
merged = model.merge_and_unload()
merged.save_pretrained(MERGED_DIR, max_shard_size="2GB")
tok.save_pretrained(MERGED_DIR)
print(f"Merged model saved to {MERGED_DIR}")
```

Verify:
```
!ls -la {MERGED_DIR}
!du -sh {MERGED_DIR}
```

Expected: 9 files totaling ~3.8 GB. You'll see 2-3 `model-XXXXX-of-YYYYY.safetensors` shards plus `config.json`, `generation_config.json`, `model.safetensors.index.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`.

## Cell 12 — Compress to tarball + integrity hash

To avoid gdrivefs "file changed as we read it" warnings, copy to local Colab disk first, tar there, then copy back:

```
import os
!cp -r /content/drive/MyDrive/blox-ai-training/qwen3-1.7b-blox-merged /content/
os.chdir("/content")
!tar -czf qwen3-1.7b-blox-merged.tar.gz qwen3-1.7b-blox-merged/
!ls -lh qwen3-1.7b-blox-merged.tar.gz
!sha256sum qwen3-1.7b-blox-merged.tar.gz
!cp qwen3-1.7b-blox-merged.tar.gz /content/drive/MyDrive/blox-ai-training/
```

**Write down the SHA-256.** After downloading the tarball locally, run `sha256sum` again — it must match.

## Cell 13 — Verify the tarball lists all 9 files

```
!tar -tzf /content/drive/MyDrive/blox-ai-training/qwen3-1.7b-blox-merged.tar.gz
```

You should see all 9 files. If anything is missing, re-run Cell 12.

## Download

Open [drive.google.com](https://drive.google.com), navigate to `MyDrive/blox-ai-training/`, right-click `qwen3-1.7b-blox-merged.tar.gz` → **Download** (~3 GB).

After download, verify integrity:
```bash
sha256sum qwen3-1.7b-blox-merged.tar.gz
```
Must match what Cell 12 printed.

## What to do next (on your Linux box, NOT Colab)

The merged model is now a standard HuggingFace dir. To deploy on the Blox device you need to convert it to RKLLM W8A8 format using Rockchip's `rkllm-toolkit` — that step is x86 Linux + CUDA specific, not Colab-friendly.

See `OPERATIONS.md` at the repo root, "Full cycle walkthrough" step 5 (`rkllm_convert.sh`) for the exact toolkit invocation. The short version:

1. Extract: `tar -xzf qwen3-1.7b-blox-merged.tar.gz`
2. Install Rockchip's rkllm-toolkit (see their repo at github.com/airockchip/rknn-llm)
3. Run conversion targeting rk3588, W8A8 quantization → produces `qwen3-1.7b-rk3588-w8a8.rkllm` (~2.0-2.4 GB)
4. SHA256 the .rkllm
5. Upload as a GitHub release asset on `functionland/blox-ai` (tag `model-qwen-3-1.7b-w8a8-v1`)
6. Update `fula-ota`'s `download_model.sh` SHA placeholder
7. Deploy + canary on lab device at `pi@192.168.2.159`

## Troubleshooting

### "CUDA out of memory" during training (not eval)
- Lower `per_device_train_batch_size` from `2` to `1` in the Cell 7 config, bump `gradient_accumulation_steps` from `2` to `4` (keeps effective batch=4). Re-run.

### Session disconnects mid-training
- Colab Free has a ~12 h hard cap and ~90 min idle disconnect. Active training keeps the session alive, but laptop sleep / wifi blips for 90+ min kill it.
- Mitigation: don't close the laptop; `save_steps: 50` in the config writes checkpoints to Drive periodically so partial progress survives.

### Training never finishes / hangs
- **Runtime → Interrupt execution**, check log output. Most common cause: HF tokenizer download timing out — re-run Cell 8 (it resumes the download).

### Loss not decreasing
- After 50-100 steps the training loss should drop noticeably. Stuck at 1.5+: bump `learning_rate: 2.0e-4` to `5.0e-4` and try again. NaN or extremely high: set `bf16: false` (falls back to fp16) and re-run.

### "ValueError: numpy.dtype size changed" after upgrade
- Restart Colab runtime (Runtime → Restart session). The torchao upgrade pulled a newer numpy; cached C extensions need to reload.

### "credential propagation was unsuccessful" on drive.mount
- Click through the auth popup fully — don't close it. If it persists, restart runtime + retry. If still failing, skip Drive mount entirely and save to `/content/` instead (must download before session ends).

### `from trl import DataCollatorForCompletionOnlyLM` fails
- TRL 1.5+ moved the import path. Either pin `pip install "trl<0.13"` OR train without masking (`mask_user_and_tool: false`). The Cell 7 config does the latter by default.

### `ImportError: Found an incompatible version of torchao`
- `pip install --upgrade torchao` then restart runtime. Cell 3 includes both — make sure you ran them.

### tar "file changed as we read it" warnings
- Benign on gdrivefs in most cases (metadata changes, not content). Cell 12 avoids them by tar-ing from local Colab disk first.
- Verify the tarball is intact with `tar -tzf <file>` (Cell 13) — if all expected files list cleanly, you're fine.
