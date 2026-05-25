"""19.5 — Held-out eval gate. HARD PROMOTION CRITERIA per the Phase 19 sub-plan:
  - >= 85% verdict severity correctness
  - >= 90% root cause correctness (keyword match)
  - >= 80% recommended-action set match
  - ZERO whitelist violations (model proposes off-whitelist action_name)
  - ZERO hard_rules violations (max_tool_calls exceeded)

Exits non-zero on failure so CI can gate model publication.

Decouples from the model implementation via a Backend protocol:
  - InProcessRKLLMBackend: real Qwen via blox-ai's RKLLMRuntime (lab-like)
  - ScriptedBackend: returns canned model outputs (unit-testable; lets
    us prove the SCORER is correct without needing a real model)
  - HTTPBackend: posts to a running blox-ai container's /troubleshoot
    (recommended for cycles where the new .rkllm is already loaded
    into a container; uses the actual production code path)
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Protocol


logger = logging.getLogger("fula-ai-training.eval")


# ---------------------------------------------------------------------------
# Promotion thresholds (PUBLIC — read by tooling)
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "verdict_severity_correct_pct": 85.0,
    "root_cause_correct_pct": 90.0,
    "recommended_action_set_match_pct": 80.0,
    "whitelist_violations": 0,
    "hard_rule_violations": 0,
}


SEVERITY_VALUES = ("green", "yellow", "red")


# ---------------------------------------------------------------------------
# Test-scenario loading
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    id: str
    description: str
    user_prompt: str
    canned_diag_responses: dict[str, Any]
    expected_severity: str
    expected_root_cause_keywords: list[str]
    expected_recommended_actions: list[dict]
    max_tool_calls: int
    no_whitelist_violations: bool
    phone_context: Optional[dict] = None


def load_scenarios(test_set_dir: Path) -> list[Scenario]:
    """Load all *.yaml from the test_set_dir into Scenario dataclasses."""
    import yaml
    out: list[Scenario] = []
    for p in sorted(test_set_dir.glob("*.yaml")):
        with open(p, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            logger.warning("skip %s: not a YAML mapping", p)
            continue
        out.append(Scenario(
            id=str(raw["id"]),
            description=str(raw.get("description", "")),
            user_prompt=str(raw["user_prompt"]),
            canned_diag_responses=dict(raw.get("canned_diag_responses") or {}),
            expected_severity=raw["expected"]["severity"],
            expected_root_cause_keywords=list(raw["expected"].get("root_cause_keywords") or []),
            expected_recommended_actions=list(raw["expected"].get("recommended_actions") or []),
            max_tool_calls=int(raw["hard_rules"].get("max_tool_calls", 4)),
            no_whitelist_violations=bool(raw["hard_rules"].get("no_whitelist_violations", True)),
            phone_context=raw.get("phone_context"),
        ))
    return out


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

class Backend(Protocol):
    """A backend takes a Scenario + returns the model's complete SSE
    event sequence as a list of dicts (one per SSE event)."""
    def run(self, scenario: Scenario) -> list[dict]: ...


class ScriptedBackend:
    """Test-only backend: returns a pre-recorded event sequence per id.
    Lets us prove the scorer's correctness without needing a real model."""

    def __init__(self, scripts: dict[str, list[dict]]):
        self.scripts = scripts

    def run(self, scenario: Scenario) -> list[dict]:
        events = self.scripts.get(scenario.id)
        if events is None:
            raise KeyError(f"no script for scenario id={scenario.id!r}")
        return events


# ---------------------------------------------------------------------------
# Whitelist loading (cross-validate recommended_action names)
# ---------------------------------------------------------------------------

def load_whitelist_names(whitelist_path: Path) -> set[str]:
    """Return the set of valid tier_2 + tier_3 action_names from
    action_whitelist.json (mirrors what the container's executor allows)."""
    with open(whitelist_path, encoding="utf-8") as f:
        raw = json.load(f)
    t2 = (raw.get("tier_2_idempotent") or {}).get("actions") or {}
    t3 = (raw.get("tier_3_destructive") or {}).get("actions") or {}
    if isinstance(t2, dict) and isinstance(t3, dict):
        return set(t2.keys()) | set(t3.keys())
    return set()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class ScenarioScore:
    id: str
    severity_correct: bool
    severity_expected: str
    severity_actual: Optional[str]
    root_cause_correct: bool
    root_cause_text: str
    actions_match: bool
    actions_expected: list[dict]
    actions_actual: list[dict]
    tool_calls_made: int
    max_tool_calls: int
    hard_rule_violations: list[str] = field(default_factory=list)
    whitelist_violations: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    n_scenarios: int
    verdict_severity_correct: int
    root_cause_correct: int
    recommended_action_set_match: int
    whitelist_violations: int
    hard_rule_violations: int
    per_scenario: list[ScenarioScore] = field(default_factory=list)
    promoted: bool = False
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["per_scenario"] = [asdict(s) for s in self.per_scenario]
        return d


def _root_cause_match(text: str, keywords: list[str]) -> bool:
    """Case-insensitive substring match; the verdict's root_cause OR
    summary may carry the keyword."""
    if not keywords:
        return True
    low = (text or "").lower()
    for kw in keywords:
        if kw.lower() in low:
            return True
    return False


def _normalise_action(a: dict) -> tuple:
    """Stable hashable form: (action_name, sorted-args-tuple). args
    comparison is loose — order-insensitive and ignores tier (since
    tier is enforced by the whitelist independently)."""
    name = a.get("action_name") or a.get("action") or ""
    args = a.get("args") or {}
    args_tuple = tuple(sorted(args.items())) if isinstance(args, dict) else ()
    return (str(name), args_tuple)


def _action_set_match(expected: list[dict], actual: list[dict]) -> bool:
    """Equal as multisets. Empty == empty is correct (some scenarios
    expect NO recommendation)."""
    exp = sorted(_normalise_action(a) for a in expected)
    act = sorted(_normalise_action(a) for a in actual)
    return exp == act


def _extract_verdict(events: list[dict]) -> Optional[dict]:
    for e in events:
        if e.get("type") == "verdict":
            return e
    return None


def _extract_recommended_actions(events: list[dict]) -> list[dict]:
    return [
        {"action_name": e["action_name"], "args": e.get("args") or {}}
        for e in events
        if e.get("type") == "recommended_action"
    ]


def _count_tool_calls(events: list[dict]) -> int:
    return sum(1 for e in events if e.get("type") == "tool_call")


def score_scenario(
    scenario: Scenario,
    events: list[dict],
    whitelist_names: set[str],
) -> ScenarioScore:
    verdict = _extract_verdict(events)
    actual_severity = (verdict or {}).get("payload", {}).get("severity")
    severity_correct = actual_severity == scenario.expected_severity

    rc_text = " ".join([
        (verdict or {}).get("payload", {}).get("root_cause", ""),
        (verdict or {}).get("payload", {}).get("summary", ""),
    ])
    root_cause_correct = _root_cause_match(rc_text, scenario.expected_root_cause_keywords)

    actual_actions = _extract_recommended_actions(events)
    actions_match = _action_set_match(scenario.expected_recommended_actions, actual_actions)

    tool_calls = _count_tool_calls(events)

    hard_violations: list[str] = []
    if tool_calls > scenario.max_tool_calls:
        hard_violations.append(
            f"max_tool_calls exceeded: {tool_calls} > {scenario.max_tool_calls}"
        )

    whitelist_violations: list[str] = []
    if scenario.no_whitelist_violations:
        for act in actual_actions:
            name = act.get("action_name", "")
            if name and name not in whitelist_names:
                whitelist_violations.append(name)

    return ScenarioScore(
        id=scenario.id,
        severity_correct=severity_correct,
        severity_expected=scenario.expected_severity,
        severity_actual=actual_severity,
        root_cause_correct=root_cause_correct,
        root_cause_text=rc_text[:300],
        actions_match=actions_match,
        actions_expected=scenario.expected_recommended_actions,
        actions_actual=actual_actions,
        tool_calls_made=tool_calls,
        max_tool_calls=scenario.max_tool_calls,
        hard_rule_violations=hard_violations,
        whitelist_violations=whitelist_violations,
    )


def aggregate_report(per_scenario: list[ScenarioScore]) -> EvalReport:
    n = len(per_scenario)
    report = EvalReport(
        n_scenarios=n,
        verdict_severity_correct=sum(1 for s in per_scenario if s.severity_correct),
        root_cause_correct=sum(1 for s in per_scenario if s.root_cause_correct),
        recommended_action_set_match=sum(1 for s in per_scenario if s.actions_match),
        whitelist_violations=sum(len(s.whitelist_violations) for s in per_scenario),
        hard_rule_violations=sum(len(s.hard_rule_violations) for s in per_scenario),
        per_scenario=per_scenario,
    )
    if n == 0:
        report.reasons.append("no scenarios loaded")
        return report
    sev_pct = 100.0 * report.verdict_severity_correct / n
    rc_pct = 100.0 * report.root_cause_correct / n
    act_pct = 100.0 * report.recommended_action_set_match / n
    reasons: list[str] = []
    if sev_pct < THRESHOLDS["verdict_severity_correct_pct"]:
        reasons.append(
            f"verdict severity {sev_pct:.1f}% < {THRESHOLDS['verdict_severity_correct_pct']}%"
        )
    if rc_pct < THRESHOLDS["root_cause_correct_pct"]:
        reasons.append(
            f"root cause {rc_pct:.1f}% < {THRESHOLDS['root_cause_correct_pct']}%"
        )
    if act_pct < THRESHOLDS["recommended_action_set_match_pct"]:
        reasons.append(
            f"action set match {act_pct:.1f}% < {THRESHOLDS['recommended_action_set_match_pct']}%"
        )
    if report.whitelist_violations > THRESHOLDS["whitelist_violations"]:
        reasons.append(f"whitelist_violations={report.whitelist_violations}")
    if report.hard_rule_violations > THRESHOLDS["hard_rule_violations"]:
        reasons.append(f"hard_rule_violations={report.hard_rule_violations}")
    report.promoted = not reasons
    report.reasons = reasons
    return report


def run_eval(
    scenarios: list[Scenario],
    backend: Backend,
    whitelist_names: set[str],
) -> EvalReport:
    per_scenario: list[ScenarioScore] = []
    for sc in scenarios:
        try:
            events = backend.run(sc)
        except Exception as e:  # noqa: BLE001
            logger.warning("backend failed for %s: %s", sc.id, e)
            events = [{"type": "error", "code": "BACKEND_FAILURE",
                       "message": str(e), "recoverable": False}]
        per_scenario.append(score_scenario(sc, events, whitelist_names))
    return aggregate_report(per_scenario)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-set",
                        default="corpus/test_set",
                        help="Directory with scenario *.yaml files")
    parser.add_argument("--whitelist",
                        default="../fula-ota/docker/fxsupport/linux/plugins/blox-ai/action_whitelist.json",
                        help="Path to action_whitelist.json (cross-validates "
                             "recommended_action names)")
    parser.add_argument("--backend",
                        choices=("scripted", "http"),
                        default="http",
                        help="scripted: only for testing; http: POST to a running blox-ai container")
    parser.add_argument("--http-url",
                        default="http://127.0.0.1:8083",
                        help="Base URL of the running blox-ai container (HTTP backend)")
    parser.add_argument("--output",
                        default="-",
                        help="Where to write the JSON report (default stdout)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level)

    test_set_dir = Path(args.test_set).resolve()
    scenarios = load_scenarios(test_set_dir)
    logger.info("loaded %d scenarios from %s", len(scenarios), test_set_dir)

    whitelist_names = load_whitelist_names(Path(args.whitelist).resolve())
    logger.info("loaded %d whitelist names", len(whitelist_names))

    if args.backend == "http":
        backend = HTTPBackend(args.http_url)
    else:
        # Scripted backend isn't useful from CLI; CI uses HTTP. Stub.
        raise SystemExit("--backend=scripted is for unit tests only; use http")

    report = run_eval(scenarios, backend, whitelist_names)

    report_json = json.dumps(report.to_dict(), indent=2)
    if args.output == "-":
        print(report_json)
    else:
        Path(args.output).write_text(report_json, encoding="utf-8")

    if not report.promoted:
        logger.error("PROMOTION REJECTED: %s", "; ".join(report.reasons))
        return 1
    logger.info("PROMOTION PASSED")
    return 0


# ---------------------------------------------------------------------------
# HTTPBackend — talks to a running blox-ai container
# ---------------------------------------------------------------------------

class HTTPBackend:
    """Streams /troubleshoot from a running blox-ai container. The
    container's tool_executor must be wired to return the scenario's
    canned_diag_responses (use the included `mock_diag_server.py`
    sidecar; or run the eval against a dev container with a mock
    backend wired)."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def run(self, scenario: Scenario) -> list[dict]:
        import httpx
        events: list[dict] = []
        body = {"prompt": scenario.user_prompt}
        with httpx.stream(
            "POST", f"{self.base_url}/troubleshoot",
            json=body, timeout=300.0,
        ) as resp:
            resp.raise_for_status()
            buf = ""
            for chunk in resp.iter_text():
                buf += chunk
                while "\n\n" in buf:
                    record, buf = buf.split("\n\n", 1)
                    record = record.strip()
                    if not record.startswith("data: "):
                        continue
                    try:
                        events.append(json.loads(record[len("data: "):]))
                    except json.JSONDecodeError:
                        continue
        return events


if __name__ == "__main__":
    sys.exit(main())
