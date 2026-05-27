# Fine-tune success metrics — Qwen 3 1.7B Blox AI LoRA

Pin these BEFORE running the fine-tune. The eval gate (`training/eval_held_out.py` against `corpus/test_set/`'s 15 hand-authored scenarios) measures each metric. A fine-tuned model that fails ANY hard gate must NOT be published to the device fleet.

## Hard gates (publish-blocking)

| Metric | Baseline (Qwen 3 1.7B raw) | Target (fine-tuned) | Why |
|---|---|---|---|
| **`<recommendation>.action_name` in whitelist** | should be ~100% | **100%** (regression rejects model) | Trust boundary. Container executor would reject anyway, but training the model to invent action names erodes the boundary's audit value. |
| **Structural-XML compliance** | measure during pilot | **≥ 95%** (no regression vs baseline) | The runtime parsers can't recover from malformed XML — every malformed response = silent failure to the user. |
| **No Kubernetes mentions** (rule 11) | varies (base model has K8s in pretraining) | **≥ 98%** | Lab-observed false-positive 2026-05-26: 1.5B model said "kubelet/kube-proxy" on a Docker device. Direct regression target. |

## Quality targets (don't block publish, but log + alert)

| Metric | Baseline | Target |
|---|---|---|
| **Rule-8 compliance** (asks before restart on heartbeat-green contradiction) | likely low | **≥ 80%** |
| **Rule-6 compliance** (cites the actual JSON field name in reasoning) | varies | **≥ 70%** |
| **No-action-when-healthy** (rule: don't invent fixes) | likely low for tuned models | **≥ 85%** of all-green eval cases produce zero recommendations |
| **Confidence calibration** (no restart-class > 0.7 confidence at yellow severity) | varies | **≥ 90%** |

## Measurement (per scenario)

`training/eval_held_out.py` reads each `corpus/test_set/*.yaml`, replays canned `diag/*` responses to the model, scores:

1. Did the model emit valid XML structure? (parses → structural-XML pass)
2. Did it call `recommended_action.action_name` ∈ whitelist? (parses → whitelist pass)
3. Does `verdict.root_cause` contain at least one `expected.root_cause_keywords` substring? (keyword match)
4. Does each emitted recommendation match an `expected.recommended_actions` entry by `action_name` + (if present) `args`? (action match)
5. Did the assistant content mention Kubernetes terminology? (rule-11 regex)

Stats are averaged across all 15 scenarios + reported. The "hard gates" check the worst-case scenario, not the average — one Kubernetes mention in 15 = a fail.

## Pilot path (BEFORE running full fine-tune)

Quantization risk: W8A8 on a 1.7B model can break structural tokens (advisor catch). Recommended pilot:

1. Generate the FULL synthetic dataset (already done: `corpus/labelled/synthetic_*.labelled.json`).
2. Run `train_qwen3_messages.py` for **1 epoch** on a rented GPU (~30 min on an A100 or 4090).
3. Merge the LoRA adapter into the base model: `python -m training.merge_adapter ...` (TODO: helper).
4. Convert to RKLLM W8A8 via `compilation/rkllm_convert.sh`.
5. Deploy to lab: `compilation/lab_canary.sh` (A/B test against the current production model).
6. Run **one** of the held-out scenarios manually on the lab device. Check:
   - Does the model emit any `<tool_call>` or `<verdict>` XML at all? (yes/no — quantization didn't break structural tokens)
   - Does the response have a `<think>` block? (yes/no — thinking-mode survived)
7. ONLY IF both pass → run the full 3-epoch fine-tune + full eval_held_out.

If the pilot fails (no XML emitted post-quantization): redesign the dataset (more structural tags per example) or accept that LoRA + W8A8 collapses on 1.7B (use the un-merged float16 adapter + serve via CPU fallback, slower but functional).

## Cadence

After the first publish:

- Eval gate runs on every model rebuild (CI).
- Hard gates published as required GitHub checks on the publish PR.
- If a metric regresses by >5% vs the published model, treat as a publish-blocker until investigated.
