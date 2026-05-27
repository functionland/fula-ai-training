# Synthetic Blox AI training corpus

Generator that produces `*.labelled.json` files for `corpus/labelled/` — the corpus the LoRA trainer (`training/train_qwen3_messages.py`) consumes.

Built 2026-05-27 to bootstrap fine-tuning of Qwen 3 1.7B for the on-device Blox AI use case before real anonymized user transcripts accumulate via the Phase 21 opt-in upload path.

## What it ships

- **174 training examples** covering all 9 runbook sections + 5 adversarial categories.
- Each example is one multi-turn troubleshooting session: `session_started` → `thought`/`tool_call`/`tool_result` (1-3 turns) → final `thought` + `verdict` + 0+ `recommended_action`.
- Reasoning noise distribution: 30% terse / 50% normal / 20% verbose `<think>` blocks (per Gemini-advisor sign-off: prevents template-recital collapse).
- Severity distribution: ~30% green / ~30% yellow / ~40% red (matches operational reality).
- All examples pre-labelled `accept` (synthetic = correct by construction).

## Composition

| Category | Examples | What it trains |
|---|---|---|
| `real-lab-anchored` | 14 | The 2026-05-27 lab snapshot (discovery red + heartbeat green = false-positive trap). Authentic JSON shapes. |
| `runbook-section-containers` | 20 | kubo OOM (legitimate `docker.restart`); cluster pebble corruption (multi-turn drill-in). |
| `runbook-section-time` | 10 | Clock unsynced + heartbeat 401 → `ntp.resync` (rule-9 high confidence). |
| `runbook-section-wireguard` | 10 | Handshake stale + WAN OK → `wireguard.bounce`. |
| `runbook-section-internet` | 10 | Captive portal — NO action (user must auth at portal). |
| `runbook-section-storage` | 10 | ext4 errors + smartctl FAILED — NO action (hardware replacement). |
| `runbook-section-power` | 10 | Undervoltage events + reboots — NO action (PSU/cable issue). |
| `runbook-section-relay` | 10 | Zero reservations + internet OK → `restart_fula` (medium confidence per runbook). |
| `csv-not-earning` | 20 | The user CSV's two not-earning paths: ipfs_cluster down + cannot-join-pool. |
| `adversarial-k8s` | 14 | User uses Kubernetes terminology (pod / kubelet / kubectl); model corrects in `<think>` and uses correct stack names. |
| `adversarial-rule-8` | 10 | User asks for restart while heartbeat is green — model refuses + cites field. |
| `adversarial-rule-6` | 8 | Model must cite the exact JSON field name (`internet.latency_ms_avg` is network, NOT clock). |
| `adversarial-rule-10` | 8 | Relay yellow alone is informational, not actionable. |
| `null-action` | 12 | Fully healthy device — verdict green + ZERO recommendation (critical: prevents always-invent-fix). |
| `user-question` | 8 | Vague prompt → model asks one clarifying `user_question` with choice options. |
| **TOTAL** | **174** | |

## Format

Each `synthetic_<scenario>_<idx>.labelled.json` matches:
- The `*.labelled.json` wrapper schema from `labeller/app.py:LabelRequest`
- The nested `transcript` matches `server/anonymized_transcript.schema.json` (closed schema, additionalProperties:false at every level)

`session_relative_start: "+0s"` and `consent.anonymizer_version: "1.0.0"` are present so synthetic records validate exactly like real anonymized uploads.

## Validation (executed at generation)

Every file passes through `validators.py:validate_labelled_record()` before write. Validation rules (per advisor sign-off):

1. XML tags lowercase strict (`<tool_call>` never `<Tool_Call>`)
2. Never `<verdict>` and `<tool_call>` in the same assistant turn
3. `<recommendation>.action_name` in `action_whitelist.json`
4. `<recommendation>.args` matches `argument_constraints` for that action
5. `<verdict>.severity` ∈ {green, yellow, red}
6. `<think>` blocks open/close balanced; no nested `<think>`
7. No Kubernetes terms in assistant output (unless k8s-correction scenario)
8. `tool_call.payload.tool` in diag/* enum
9. `tool_result.call_id` matches preceding `tool_call.call_id`
10. `session_started` always first event
11. Events count ≤ 200 (Phase 20 schema cap)
12. Approx token count ≤ 4096

A validation failure logs + skips that example — never silently shipped.

## Regenerating

```bash
# Wipe + regenerate
rm corpus/labelled/synthetic_*.labelled.json
python -m corpus.synthetic.generate --out corpus/labelled

# Dry-run (validate only, no writes)
python -m corpus.synthetic.generate --out corpus/labelled --dry-run

# Pilot (first N for toolchain validation)
python -m corpus.synthetic.generate --out corpus/labelled --max-files 10

# Reproducible (default seed=42)
python -m corpus.synthetic.generate --out corpus/labelled --seed 17
```

Report goes to `corpus/validation_report.txt`.

## Extending

To add a new scenario:

1. Open `scenarios.py` (main) or `scenarios_extended.py` (overflow).
2. Add a new `Scenario(...)` instance to `SCENARIOS` or `EXTENDED`.
3. Provide `user_prompts` (5-10 phrasings), `tool_call_sequence` (list of `(tool, args, response)` tuples), `verdict_summary` + `verdict_severity` + `verdict_root_cause`, `recommendations` (or empty for null-action), `think_texts` (per-turn `(terse, normal, verbose)` tuples).
4. Choose `variations` count (10-15 is typical).
5. Re-run the generator.

The validator catches whitelist violations, schema mismatches, and the rule-7 markdown-list anti-pattern automatically.

## Composition advisor sign-off (2026-05-27)

- **Built-in advisor**: SOUND. Flagged: validate LoRA → RKLLM-W8A8 quantization preserves XML grammar before scaling (see SUCCESS_METRICS.md pilot path). Multiple lab states (only 1 captured; rest schema-valid-synthesized — acceptable for v1).
- **Gemini advisor**: SOUND. Flagged: quantization-collapse risk + system-prompt saturation + thinking-loop infinity. Added k8s-correction scenario per Gemini missing-piece catch.
- **Cursor advisor**: unavailable (Free quota exhausted 2026-05-27).
- **Codex advisor**: not consulted (Free rate limit not yet reset).

## Source attribution

- Real device state: captured from `pi@192.168.2.159` on 2026-05-27 (lab RK3588).
- User-reported issue patterns: `E:/Blox Issues - Sheet1.csv` (user's runbook).
- Schema invariants: `fula-ota/docker/fxsupport/linux/plugins/blox-ai/api/*.schema.json` + `action_whitelist.json` + `runbook.md`.
- HuggingFace `mecha-org/linux-command-dataset` — **excluded**. Generic shell command pairs would dilute the tool-calling XML signal.
