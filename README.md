# fula-ai-training

Off-device infrastructure for the Blox AI model improvement loop. Two siblings live in this repo:

1. **Transcript intake server** (`server/`) — Phase 20 of the parent plan; receives opt-in anonymized troubleshooting transcripts uploaded by the Blox phone app.
2. **Training pipeline** (`corpus/`, `labeller/`, `training/`, `compilation/`, `publish/`) — Phase 19 sub-plan; turns intake transcripts into new `.rkllm` model files + bumps the device-side `/etc/fula/ai-manifest.json` so the fleet upgrades.

This README covers the training pipeline. The intake server has its own contract in `server/README.md` (TLS posture, schema, PII scanner).

## Privacy contract (inherited from Phase 20)

Transcripts in `corpus/raw/` are **already anonymized on the phone before upload** (peerIds, IPs, paths, SSIDs, BSSIDs, timestamps stripped per Phase 21's `anonymizeTranscript.ts`). The intake server runs a defense-in-depth PII scanner before persisting. The training pipeline inherits this: it never reattaches identifiers, never re-runs the model on raw transcripts before the labeler reviewed them.

If you find PII in `corpus/raw/`, that's an intake-server failure — fix the scanner and re-evaluate every affected transcript before training on it.

## The 9 sub-phases

| Sub-phase | Module | Purpose |
|---|---|---|
| 19.1 | `corpus/sync_corpus.py` | Pull raw transcripts from intake-server object storage; idempotent + schema-validating |
| 19.2 | `labeller/app.py` | Tiny FastAPI tool: developer reviews transcripts + flags verdict/action correctness |
| 19.3 | `corpus/test_set/` | Hand-authored canonical scenarios that the eval gate runs against |
| 19.4 | `training/lora_train.py` | PEFT-driven LoRA fine-tune on labelled transcripts |
| 19.5 | `training/eval_held_out.py` | Hard-gate scorer (≥85% verdict / ≥90% root-cause / ≥80% action / ZERO whitelist violations) |
| 19.6 | `compilation/rkllm_convert.sh` | Wrapper around Rockchip's RKLLM-Toolkit; converts merged HF weights to W8A8 `.rkllm` |
| 19.7 | `compilation/lab_canary.sh` | A/B test on the lab device before any CDN upload |
| 19.8 | `publish/update_manifest.py` | Bump `current`/`rollback` entries in `ai-manifest.json`; uploads model + manifest to CDN |
| 19.9 | `OPERATIONS.md` | Triage runbook: how to interpret eval reports, roll back, add test scenarios |

## Cadence + operational ingredients

A full cycle (transcripts → new published model) requires:

| Ingredient | Where it lives | Who provides |
|---|---|---|
| ≥500 labelled transcripts | `corpus/labelled/` | accumulates from intake-server over weeks; manual labelling pass via `labeller/app.py` |
| Base Qwen 2.5 3B Instruct weights | Hugging Face: `Qwen/Qwen2.5-3B-Instruct` | downloaded by `training/lora_train.py` |
| GPU box with CUDA 11.8+ + ≥24 GB VRAM | external | operator |
| Rockchip RKLLM-Toolkit | installed on the GPU box; pinned version in `compilation/pin_rkllm_toolkit.md` | operator (Rockchip developer account) |
| Lab RK3588 device for A/B canary | `pi@192.168.68.107` | shared lab device |
| CDN upload credentials | env vars / config — see `publish/README.md` | operator |

**Target cycle time**: 4-8 hours of operator time per cycle once the pipeline is operational. 1-2 cycles per month after the first 500 transcripts accumulate.

## Quick start (once you have a labelled corpus)

```bash
# On the GPU box
pip install -e ".[train,eval]"

# 1. Sync transcripts from intake server's S3 bucket to corpus/raw/
python -m corpus.sync_corpus --bucket s3://fula-ai-training-prod-transcripts

# 2. (Manual) Label transcripts via the labeller UI
uvicorn labeller.app:app --host 127.0.0.1 --port 8765
# Open http://127.0.0.1:8765/ in a browser and label each transcript.

# 3. Fine-tune LoRA on the labelled set
python -m training.lora_train --config training/configs/qwen3b_lora.yaml

# 4. Evaluate against held-out test scenarios (HARD GATE — exits non-zero on failure)
python -m training.eval_held_out --adapter training/output/<date>/adapter

# 5. Merge LoRA + convert to RKLLM W8A8
python -m compilation.merge_lora --adapter training/output/<date>/adapter
bash compilation/rkllm_convert.sh training/output/<date>/merged

# 6. A/B canary on the lab device (deploys to :test tag first)
bash compilation/lab_canary.sh training/output/<date>/merged.rkllm

# 7. After canary passes (manual sign-off), publish + bump manifest
python -m publish.update_manifest \
    --new-current-model-version $(date +%Y-%m-%d) \
    --new-current-url https://functionyard.fx.land/qwen-3b-$(date +%Y-%m-%d).rkllm \
    --new-current-sha256 $(sha256sum training/output/<date>/merged.rkllm | awk '{print $1}') \
    --new-current-size-bytes $(stat -c%s training/output/<date>/merged.rkllm)
```

## What this pipeline does NOT do

- **On-device fine-tuning**: RK3588 is inference-only; LoRA happens off-device.
- **Continuous fine-tuning without human-in-the-loop**: every cycle goes through a manual labelling pass.
- **Auto-promotion**: the eval gate is advisory until a human reviews the report. Final publication is always an operator-driven step.
- **Cross-org transcript sharing**: each org's transcripts stay in that org's bucket; the pipeline doesn't aggregate across deployments.

## Tests

```bash
pip install -e ".[dev,eval]"
pytest -ra -q
```

Server tests live in `server/tests/`; pipeline tests live in `tests/`. Both run from one `pytest` invocation.
