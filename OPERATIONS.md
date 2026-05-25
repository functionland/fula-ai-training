# Phase 19 — Operations runbook

Internal triage runbook for the fula-ai-training fine-tune pipeline. Lives alongside `README.md` (which is user-facing). When in doubt, read `eval_report.json` first — most operational questions ("can I publish?", "did this cycle regress?") are answered there.

## Full cycle walkthrough

A complete fine-tune cycle takes 4-8 hours of operator time + several hours of GPU time.

```
1. sync transcripts            (5 min)
2. label ≥100 new transcripts (60-120 min, depends on operator)
3. lora_train.py             (2-4 hours on a 24GB A10G)
4. merge_lora.py             (5 min)
5. rkllm_convert.sh         (30-90 min on the same GPU box)
6. lab_canary.sh            (15 min including model load + eval)
7. update_manifest.py       (2 min)
8. publish manifest to CDN  (operator-driven; whatever tooling we use)
9. observe for 2 weeks      (canary roll-out per parent plan Phase 22)
10. promote to :release if clean
```

## Reading an eval report

`eval_report.json` from `training/eval_held_out.py`:

```json
{
  "n_scenarios": 15,
  "verdict_severity_correct": 14,
  "root_cause_correct": 13,
  "recommended_action_set_match": 12,
  "whitelist_violations": 0,
  "hard_rule_violations": 0,
  "per_scenario": [ ... ],
  "promoted": true,
  "reasons": []
}
```

- `promoted: true` + `reasons: []` → safe to publish (subject to manual review).
- `promoted: false` → DON'T PUBLISH. Read `per_scenario` for which scenarios failed + why.

### Per-scenario debugging

If `kubo_hang_001` failed `actions_match`:

1. Look at `per_scenario[i].actions_actual` vs `actions_expected`.
2. If actions_actual is empty: the model didn't recommend anything. Either the runbook excerpt didn't include enough about kubo restart, OR the training corpus didn't have similar examples. Both fixable in next cycle.
3. If actions_actual has WRONG args: the model proposed `docker.restart {container: ipfs_cluster}` instead of `ipfs_host`. Check the labelled corpus for this scenario type — labels should clearly say which container goes with which symptom.
4. If actions_actual has an action NOT in the whitelist: `whitelist_violations` will list it. This is a HARD GATE failure — the model is hallucinating actions and would be rejected by the container's executor anyway.

## Common failure modes

### Eval gate red but model "feels fine" interactively

The held-out scenarios stress specific edge cases. A model that handles 80% of real user prompts well can still flunk the gate if those user prompts are easier than the scenarios. **The gate is the authority.** If you disagree with a scenario's expected behaviour, edit the YAML — but only AFTER reviewing whether your disagreement reflects the actual user need.

### Toolkit upgraded without runtime upgrade

If `rkllm_convert.sh` succeeds but the lab device's `librkllmrt.so` can't load the resulting `.rkllm`:

1. Check `compilation/pin_rkllm_toolkit.md` — toolkit and runtime must be the same major.minor.
2. If the toolkit was bumped intentionally, the on-device runtime must be bumped in the same cycle (or one ahead). Coordinate with the fula-ota OTA team.

### "Not enough labelled transcripts"

`lora_train.py` refuses to train on fewer than `min_transcripts` (default 100). Either:
- Wait for more uploads (the intake server's `ai-feedback.jsonl` shows opt-in rates).
- Manually craft training examples that match common failure patterns observed in production audit logs. Mark them with `label_decision=synthetic` so they're tagged in the corpus for later review.

### Lab canary passes eval gate but real fleet regresses

The lab device has one specific environment (this kernel, this fula version, this network). Real fleet has variance. **Wait the full 2 weeks of canary observation** before promoting. Use `ai-feedback.jsonl` thumbs-down rate as the primary signal; investigate any spike.

## Rollback

`publish/update_manifest.py --rollback-required` flips devices to the prior model on next plugin restart. Use when:

- A new model ships and audit logs show repeated `executor_busy` from confused-decoy tool calls (model keeps re-calling tools after results — a real regression observed in our 1.1.4 testing).
- Eval-gate green but real fleet shows >2x thumbs-down rate vs prior model.
- Any whitelist violations observed in production audit logs (the container's executor rejects them, but the model proposing off-whitelist actions is a regression signal).

After rolling back, the offending model stays in the manifest's `current` field with `rollback_required: true`. Devices use `rollback` until the operator publishes a new manifest with the failed version demoted to `rollback` (or removed entirely) and a known-good model in `current`.

## When to bump the eval gate's threshold

The default thresholds (`THRESHOLDS` in `eval_held_out.py`) are conservative for v1. As we accumulate cycles + the model improves:

- If 3+ consecutive cycles ship at ≥95% on a metric, consider raising that metric's threshold to lock in the gain.
- NEVER lower a threshold without a documented decision in this file (search-grep "threshold-lowered" entries).

## Logs + evidence

- Per-cycle: `training/output/<date-sha>/manifest.json` carries the git sha + config + start time.
- Canary: `E:/fxblox/evidence/phase-19-canary/<date>/run.log` + `eval_report.json`.
- Production: device-side `/var/log/fula/ai-actions.jsonl` (HMAC audit) + `/var/log/fula/ai-feedback.jsonl` (user ratings).

Pull all of these when post-morteming a bad cycle.

## When to escalate

- Any whitelist violation observed in PRODUCTION audit logs (not just eval): page the on-call.
- Eval gate green but real fleet thumbs-down rate >2x vs prior cycle: page the on-call.
- Toolkit + runtime drift detected in PLAYBOOK.md (e.g., devices on 1.1.4 but we just shipped a 1.2.x model): page the on-call.

## What's NOT in this runbook (intentional)

- Cross-org transcript sharing: out of scope; each org keeps its own bucket.
- On-device fine-tuning: not happening; RK3588 is inference-only.
- Auto-promotion: every cycle ships through human review.
- Anonymizer changes: the Phase 21 anonymizer lives in `fx-components`; if you find PII in the corpus, file an issue there, fix the anonymizer, re-pull transcripts before training.
