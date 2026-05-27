"""Synthetic Blox AI training corpus generator.

CLI:
    python -m corpus.synthetic.generate --out corpus/labelled
    python -m corpus.synthetic.generate --out corpus/labelled --max-files 50  # pilot
    python -m corpus.synthetic.generate --out corpus/labelled --dry-run       # validate only

Output: one `synthetic_<scenario>_<idx>.labelled.json` per generated
example. Each matches the schema enforced by `labeller/app.py` AND the
nested `transcript` matches `server/anonymized_transcript.schema.json`,
so `training/lora_train.py` consumes them without modification.

Composition (per dataset_plan_v2.md):
  - Per-scenario `variations` count from scenarios.py
  - Each variation samples (prompt_phrasing, reasoning_noise, severity_jitter)
  - Reasoning-noise distribution: 30% terse / 50% normal / 20% verbose
    (Gemini advisor: small models in template-driven generation can
    "recite" rather than reason; mixing noise prevents that)
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from corpus.synthetic.scenarios import (
    SCENARIOS, Scenario, Recommendation, expand_prompt,
)
from corpus.synthetic.scenarios_extended import EXTENDED
from corpus.synthetic.validators import (
    ValidationError, validate_labelled_record, lint_thought_content,
)

# Extend the master scenario list with the supplementary scenarios.
# (Done at module load so the rest of generate.py only sees SCENARIOS.)
SCENARIOS.extend(EXTENDED)


logger = logging.getLogger("synthetic-generator")

DEFAULT_OUT = Path("corpus/labelled")
SCHEMA_VERSION = 1
PROTOCOL_VERSION = 3
SESSION_TTL_S = 1800
PLACEHOLDER_TOKEN = "f" * 64  # 64-char placeholder; real one minted by executor
ANONYMIZER_VERSION = "1.0.0"  # matches phone-side anonymizer at this date


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _pick_noise(rng: random.Random) -> int:
    """Return 0 (terse) / 1 (normal) / 2 (verbose) with weights 30/50/20."""
    r = rng.random()
    if r < 0.30:
        return 0
    if r < 0.80:
        return 1
    return 2


def _think_text(scenario: Scenario, turn_idx: int, noise: int) -> str:
    """Pull the noise-level appropriate think text for a turn. Falls
    back gracefully if scenario has fewer think_texts than turns."""
    if turn_idx >= len(scenario.think_texts):
        if scenario.think_texts:
            terse, normal, verbose = scenario.think_texts[-1]
        else:
            return "Continuing analysis."
    else:
        terse, normal, verbose = scenario.think_texts[turn_idx]
    return [terse, normal, verbose][noise]


def _build_events(
    scenario: Scenario,
    session_id: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Build the SSE event stream for one variation of a scenario."""
    events: list[dict[str, Any]] = []

    # 1. session_started always first
    events.append({
        "type": "session_started",
        "session_id": session_id,
        "protocol_version": PROTOCOL_VERSION,
        "ttl_seconds": SESSION_TTL_S,
    })

    # 2. Special-case: vague-user scenarios end on user_question (no tool calls / verdict)
    if scenario.category == "user-question":
        noise = _pick_noise(rng)
        events.append({
            "type": "thought",
            "payload": (
                f"<think>{_think_text(scenario, 0, noise)}</think>\n"
                "I need a bit more detail to know which diagnostic to run first."
            ),
        })
        events.append({
            "type": "user_question",
            "question_id": _new_uuid(),
            "payload": {
                "question": "Which of these best describes what you're seeing?",
                "expected_response_type": "choice",
                "options": [
                    "App shows my Blox as disconnected",
                    "Rewards not increasing / not earning",
                    "Can't join a pool",
                    "Something else",
                ],
            },
        })
        return events

    # 3. Standard scenarios — walk the tool_call_sequence, then verdict + recs
    n_steps = len(scenario.tool_call_sequence)
    for i, (tool, args, response) in enumerate(scenario.tool_call_sequence):
        noise = _pick_noise(rng)
        # Assistant turn: thought (with <think>...</think>) + tool_call
        think_body = _think_text(scenario, i, noise)
        thought_payload = (
            f"<think>{think_body}</think>\n"
            f"Calling {tool} to gather data."
        )
        events.append({"type": "thought", "payload": thought_payload})
        call_id = f"tc-{i + 1}"
        events.append({
            "type": "tool_call",
            "call_id": call_id,
            "payload": {"tool": tool, "args": args},
        })
        events.append({
            "type": "tool_result",
            "call_id": call_id,
            "ok": True,
            "payload": response,
        })

    # 4. Final assistant turn: thought + verdict + 0+ recommendations
    final_noise = _pick_noise(rng)
    final_think = _think_text(scenario, n_steps, final_noise)
    final_thought = (
        f"<think>{final_think}</think>\n"
        f"{scenario.verdict_summary}"
    )
    events.append({"type": "thought", "payload": final_thought})
    events.append({
        "type": "verdict",
        "payload": {
            "summary": scenario.verdict_summary,
            "severity": scenario.verdict_severity,
            "root_cause": scenario.verdict_root_cause,
        },
    })
    for j, rec in enumerate(scenario.recommendations):
        events.append({
            "type": "recommended_action",
            "action_id": f"ra-{j + 1}",
            "action_name": rec.action_name,
            "args": rec.args,
            "reasoning": rec.reasoning,
            "confidence": rec.confidence,
            "tier": rec.tier,
            "approval_token": PLACEHOLDER_TOKEN,
        })
    return events


def _build_record(
    scenario: Scenario,
    variation_idx: int,
    rng: random.Random,
) -> dict[str, Any]:
    """Build one full `*.labelled.json` record."""
    upload_id = _new_uuid()
    session_id = _new_uuid()
    prompt_seed = variation_idx * 7 + sum(ord(c) for c in scenario.id) % 13
    base_prompt = scenario.user_prompts[variation_idx % len(scenario.user_prompts)]
    user_prompt = expand_prompt(base_prompt, prompt_seed)

    events = _build_events(scenario, session_id, rng)

    transcript = {
        "schema_version": SCHEMA_VERSION,
        "upload_id": upload_id,
        "session_relative_start": "+0s",
        "events": events,
        "user_rating": 1,  # synthetic examples are "correct" by construction
        "user_prompt": user_prompt,
        "scenario_id": scenario.scenario_id,
        "consent": {
            "explicit_opt_in": True,
            "preview_shown": True,
            "anonymizer_version": ANONYMIZER_VERSION,
        },
        "device_class": "rk3588",
    }

    notes_parts = [
        f"synthetic v1",
        f"scenario={scenario.id}",
        f"category={scenario.category}",
        f"variation={variation_idx}",
    ]
    if scenario.tags:
        notes_parts.append(f"tags={','.join(scenario.tags)}")

    record = {
        "upload_id": upload_id,
        "label_decision": "accept",
        "verdict_correct": True,
        "actions_correct": True,
        "root_cause_correct": True,
        "notes": "; ".join(notes_parts),
        "labelled_at": _iso_now(),
        "transcript": transcript,
    }
    return record


def generate_all(
    out_dir: Path,
    max_files: Optional[int] = None,
    dry_run: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate all configured scenarios. Returns stats dict."""
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "scenarios_processed": 0,
        "examples_generated": 0,
        "examples_written": 0,
        "examples_rejected": 0,
        "rejection_reasons": {},
        "per_scenario_counts": {},
        "per_category_counts": {},
        "severity_distribution": {"green": 0, "yellow": 0, "red": 0},
        "action_distribution": {},
        "files": [],
    }

    for scenario in SCENARIOS:
        stats["scenarios_processed"] += 1
        per_scenario = 0
        for var_idx in range(scenario.variations):
            stats["examples_generated"] += 1
            try:
                record = _build_record(scenario, var_idx, rng)
                validate_labelled_record(record)
            except ValidationError as e:
                stats["examples_rejected"] += 1
                reason = str(e)
                stats["rejection_reasons"][reason] = (
                    stats["rejection_reasons"].get(reason, 0) + 1
                )
                logger.warning(
                    "REJECT scenario=%s var=%d reason=%s",
                    scenario.id, var_idx, reason,
                )
                continue

            # Lint (non-fatal)
            for ev in record["transcript"]["events"]:
                if ev.get("type") == "thought":
                    for w in lint_thought_content(ev.get("payload", "")):
                        logger.debug(
                            "LINT scenario=%s var=%d: %s",
                            scenario.id, var_idx, w,
                        )

            # Stats
            stats["severity_distribution"][scenario.verdict_severity] += 1
            for rec in scenario.recommendations:
                stats["action_distribution"][rec.action_name] = (
                    stats["action_distribution"].get(rec.action_name, 0) + 1
                )

            # Write
            if not dry_run:
                fname = f"synthetic_{scenario.id}_{var_idx:03d}.labelled.json"
                path = out_dir / fname
                path.write_text(
                    json.dumps(record, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                stats["files"].append(str(path.relative_to(out_dir.parent)))
            stats["examples_written"] += 1
            per_scenario += 1

            if max_files is not None and stats["examples_written"] >= max_files:
                stats["per_scenario_counts"][scenario.id] = per_scenario
                stats["per_category_counts"][scenario.category] = (
                    stats["per_category_counts"].get(scenario.category, 0)
                    + per_scenario
                )
                return stats

        stats["per_scenario_counts"][scenario.id] = per_scenario
        stats["per_category_counts"][scenario.category] = (
            stats["per_category_counts"].get(scenario.category, 0) + per_scenario
        )

    return stats


def _write_validation_report(out_dir: Path, stats: dict[str, Any]) -> Path:
    report_path = out_dir.parent / "validation_report.txt"
    lines = [
        "# Synthetic dataset generation report",
        f"Generated at: {_iso_now()}",
        f"Output dir: {out_dir}",
        "",
        "## Counts",
        f"  Scenarios processed: {stats['scenarios_processed']}",
        f"  Examples generated:  {stats['examples_generated']}",
        f"  Examples written:    {stats['examples_written']}",
        f"  Examples rejected:   {stats['examples_rejected']}",
        "",
        "## Per-scenario counts",
    ]
    for sid, n in sorted(stats["per_scenario_counts"].items()):
        lines.append(f"  {sid:50s} {n}")
    lines.extend([
        "",
        "## Per-category counts",
    ])
    for cat, n in sorted(stats["per_category_counts"].items()):
        lines.append(f"  {cat:30s} {n}")
    lines.extend([
        "",
        "## Severity distribution",
    ])
    for sev, n in stats["severity_distribution"].items():
        lines.append(f"  {sev:10s} {n}")
    lines.extend([
        "",
        "## Recommendation-action distribution",
    ])
    if not stats["action_distribution"]:
        lines.append("  (no recommendations issued — null-action-heavy dataset)")
    else:
        for action, n in sorted(stats["action_distribution"].items()):
            lines.append(f"  {action:30s} {n}")
    if stats["rejection_reasons"]:
        lines.extend([
            "",
            "## Rejection reasons (validator failures)",
        ])
        for reason, n in stats["rejection_reasons"].items():
            lines.append(f"  [{n}] {reason}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT), type=Path,
        help="Output directory for *.labelled.json files",
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Cap on files written (for pilot runs)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate but don't write files",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for reproducible generation",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        help="Logging level (DEBUG/INFO/WARNING)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")

    out_dir = args.out.resolve()
    logger.info("Generating into %s (max_files=%s, dry_run=%s)",
                 out_dir, args.max_files, args.dry_run)
    stats = generate_all(
        out_dir, max_files=args.max_files, dry_run=args.dry_run, seed=args.seed,
    )
    report_path = _write_validation_report(out_dir, stats)
    logger.info("Written: %d, Rejected: %d. Report at %s",
                 stats["examples_written"], stats["examples_rejected"], report_path)

    if stats["examples_rejected"] > 0:
        logger.error("Validation failures present — see %s", report_path)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
