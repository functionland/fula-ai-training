# corpus/

Training + evaluation data for the Blox AI Qwen 3 1.7B fine-tune.

## Where the instructions live

Quick navigation — open the file that matches what you're trying to do.

| File | What it tells you |
|---|---|
| [`synthetic/README.md`](synthetic/README.md) | How the synthetic dataset was built, composition stats, how to regenerate, how to add new scenarios |
| [`labelled/README.md`](labelled/README.md) | How to train (commands, dependency list, dry-run + verify-mask steps) |
| [`labelled/SUCCESS_METRICS.md`](labelled/SUCCESS_METRICS.md) | Hard-gate thresholds the fine-tuned model must hit + the recommended pilot path (1-epoch smoke test before the full 3-epoch run) |
| [`test_set/README.md`](test_set/README.md) | Format spec for the 15 hand-authored YAML eval scenarios + how to add new ones |
| [`../OPERATIONS.md`](../OPERATIONS.md) | Full operational workflow: real transcripts → labeller → trainer → eval → publish → rollback. Includes the "Iteration loop" section covering how Claude can help when you share bad transcripts / eval reports / feedback clusters. |
| [`../training/train_qwen3_messages.py`](../training/train_qwen3_messages.py) (docstring) | What the trainer does, how it differs from the legacy `lora_train.py` |
| [`../training/verify_mask.py`](../training/verify_mask.py) (docstring) | How to confirm the loss mask isolates only assistant tokens BEFORE the multi-hour GPU run |

## Subdirectory map

```
corpus/
├── README.md             (this file)
├── synthetic/            bootstrap dataset generator + scenario catalog + 2026-05-27 lab snapshot
├── labelled/             *.labelled.json files the trainer consumes (synthetic + real, both)
├── test_set/             15 hand-authored YAML scenarios the eval gate runs against
├── sync_corpus.py        sub-phase 19.1 — pull real transcripts from intake server (corpus/raw/)
└── validation_report.txt auto-generated stats from the last synthetic-dataset generation
```

`corpus/raw/` (intake server output) is created on first `sync_corpus.py` run; not version-controlled.

## How the pieces fit together

1. **Bootstrap** — `synthetic/generate.py` produces ~170 `synthetic_*.labelled.json` files in `labelled/` so the LoRA pipeline can be exercised before real transcripts accumulate.
2. **Real data** — users opt-in to share anonymized transcripts → `corpus/raw/` → operator labels them with `labeller/app.py` → `labelled/<upload_id>.labelled.json`. Same schema as synthetic.
3. **Train** — `training/train_qwen3_messages.py` reads ALL `*.labelled.json` (synthetic + real). One trainer, one corpus dir.
4. **Eval** — `training/eval_held_out.py` scores the trained model against the 15 scenarios in `test_set/`. Hard gates in `labelled/SUCCESS_METRICS.md`.
5. **Iterate** — see `OPERATIONS.md` "Iteration loop" for how new logs → new scenarios → new model, including how Claude can help between cycles.
