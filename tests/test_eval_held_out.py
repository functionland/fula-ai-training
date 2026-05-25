"""19.5 — eval gate tests. The scorer's correctness is load-bearing —
a bug here would either ship a bad model or block a good one. Cover
both directions."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.eval_held_out import (
    EvalReport, Scenario, ScenarioScore, ScriptedBackend, THRESHOLDS,
    _action_set_match, _root_cause_match, aggregate_report, load_scenarios,
    load_whitelist_names, run_eval, score_scenario,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _ev_session_started():
    return {"type": "session_started", "session_id": "s",
            "protocol_version": 3, "ttl_seconds": 1800}


def _ev_tool_call(call_id, tool="diag/summary"):
    return {"type": "tool_call", "call_id": call_id,
            "payload": {"tool": tool, "args": {}}}


def _ev_tool_result(call_id, ok=True, payload=None):
    return {"type": "tool_result", "call_id": call_id, "ok": ok,
            "payload": payload or {}}


def _ev_verdict(severity, summary="ok", root_cause="x"):
    return {"type": "verdict", "payload": {
        "summary": summary, "severity": severity, "root_cause": root_cause,
    }}


def _ev_recommendation(action_name, args=None, tier=2):
    return {"type": "recommended_action", "action_id": "a1",
            "action_name": action_name, "args": args or {},
            "reasoning": "r", "confidence": 0.8, "tier": tier,
            "approval_token": "a" * 64}


def _scen(**overrides) -> Scenario:
    base = dict(
        id="test-1",
        description="",
        user_prompt="x",
        canned_diag_responses={},
        expected_severity="red",
        expected_root_cause_keywords=["kubo"],
        expected_recommended_actions=[
            {"action_name": "docker.restart", "args": {"container": "ipfs_host"}}
        ],
        max_tool_calls=4,
        no_whitelist_violations=True,
    )
    base.update(overrides)
    return Scenario(**base)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def test_root_cause_match_case_insensitive():
    assert _root_cause_match("Kubo container HUNG", ["kubo", "hang"])
    assert not _root_cause_match("everything fine", ["kubo"])


def test_root_cause_match_empty_keywords_passes():
    """No expected keywords = no check needed."""
    assert _root_cause_match("anything", [])


def test_action_set_match_order_insensitive():
    a = [{"action_name": "x", "args": {"a": 1}}, {"action_name": "y", "args": {}}]
    b = [{"action_name": "y", "args": {}}, {"action_name": "x", "args": {"a": 1}}]
    assert _action_set_match(a, b)


def test_action_set_match_args_must_match():
    a = [{"action_name": "x", "args": {"k": "ipfs_host"}}]
    b = [{"action_name": "x", "args": {"k": "evil_container"}}]
    assert not _action_set_match(a, b)


def test_action_set_match_empty_equals_empty():
    """Some scenarios expect NO recommendation (e.g., all_green)."""
    assert _action_set_match([], [])


# ---------------------------------------------------------------------------
# Per-scenario scoring
# ---------------------------------------------------------------------------

def test_score_perfect_run(tmp_path):
    sc = _scen()
    events = [
        _ev_session_started(),
        _ev_tool_call("c1", "diag/summary"),
        _ev_tool_result("c1"),
        _ev_verdict("red", root_cause="kubo wedged"),
        _ev_recommendation("docker.restart", {"container": "ipfs_host"}),
    ]
    score = score_scenario(sc, events, whitelist_names={"docker.restart"})
    assert score.severity_correct
    assert score.root_cause_correct
    assert score.actions_match
    assert score.hard_rule_violations == []
    assert score.whitelist_violations == []


def test_score_wrong_severity():
    sc = _scen()
    events = [_ev_verdict("green", root_cause="kubo OK")]
    score = score_scenario(sc, events, whitelist_names=set())
    assert not score.severity_correct


def test_score_missing_verdict_treated_as_wrong():
    sc = _scen()
    events = [_ev_tool_call("c1"), _ev_tool_result("c1")]
    score = score_scenario(sc, events, whitelist_names=set())
    assert not score.severity_correct
    assert score.severity_actual is None


def test_score_whitelist_violation():
    sc = _scen(expected_recommended_actions=[])
    events = [
        _ev_verdict("red"),
        _ev_recommendation("rm_-rf_root"),  # not whitelisted
    ]
    score = score_scenario(sc, events,
                           whitelist_names={"docker.restart", "ntp.resync"})
    assert "rm_-rf_root" in score.whitelist_violations


def test_score_tool_call_overflow():
    sc = _scen(max_tool_calls=2)
    events = [
        _ev_tool_call("c1"), _ev_tool_result("c1"),
        _ev_tool_call("c2"), _ev_tool_result("c2"),
        _ev_tool_call("c3"), _ev_tool_result("c3"),  # one too many
        _ev_verdict("red", root_cause="kubo"),
    ]
    score = score_scenario(sc, events, whitelist_names=set())
    assert score.hard_rule_violations
    assert "max_tool_calls" in score.hard_rule_violations[0]


# ---------------------------------------------------------------------------
# aggregate_report + promotion gate
# ---------------------------------------------------------------------------

def _all_perfect_scores(n):
    return [
        ScenarioScore(
            id=f"s{i}",
            severity_correct=True, severity_expected="red", severity_actual="red",
            root_cause_correct=True, root_cause_text="",
            actions_match=True, actions_expected=[], actions_actual=[],
            tool_calls_made=1, max_tool_calls=4,
        )
        for i in range(n)
    ]


def test_promotion_passes_when_all_perfect():
    rpt = aggregate_report(_all_perfect_scores(15))
    assert rpt.promoted
    assert rpt.reasons == []


def test_promotion_fails_when_verdict_pct_low():
    scores = _all_perfect_scores(10)
    # 4 of 10 severity wrong → 60% < 85%
    for s in scores[:4]:
        s.severity_correct = False
    rpt = aggregate_report(scores)
    assert not rpt.promoted
    assert any("verdict severity" in r for r in rpt.reasons)


def test_promotion_fails_on_any_whitelist_violation():
    scores = _all_perfect_scores(15)
    scores[0].whitelist_violations.append("rm_rf_root")
    rpt = aggregate_report(scores)
    assert not rpt.promoted
    assert any("whitelist_violations" in r for r in rpt.reasons)


def test_promotion_fails_on_any_hard_rule_violation():
    scores = _all_perfect_scores(15)
    scores[0].hard_rule_violations.append("max_tool_calls exceeded: 5 > 4")
    rpt = aggregate_report(scores)
    assert not rpt.promoted
    assert any("hard_rule_violations" in r for r in rpt.reasons)


def test_thresholds_match_subplan_constants():
    """These thresholds are the parent plan's PROMOTION CRITERIA.
    Don't lower them without an explicit decision."""
    assert THRESHOLDS["verdict_severity_correct_pct"] == 85.0
    assert THRESHOLDS["root_cause_correct_pct"] == 90.0
    assert THRESHOLDS["recommended_action_set_match_pct"] == 80.0
    assert THRESHOLDS["whitelist_violations"] == 0
    assert THRESHOLDS["hard_rule_violations"] == 0


# ---------------------------------------------------------------------------
# end-to-end with ScriptedBackend
# ---------------------------------------------------------------------------

def test_e2e_perfect_model_promoted():
    """A scripted backend that gives the perfect answer for every
    scenario must pass the gate."""
    scenarios = [
        _scen(id="kubo", expected_severity="red",
              expected_root_cause_keywords=["kubo"],
              expected_recommended_actions=[
                  {"action_name": "docker.restart", "args": {"container": "ipfs_host"}}
              ]),
        _scen(id="all_green", expected_severity="green",
              expected_root_cause_keywords=["healthy"],
              expected_recommended_actions=[]),
    ]
    scripts = {
        "kubo": [
            _ev_session_started(),
            _ev_tool_call("c1"), _ev_tool_result("c1"),
            _ev_verdict("red", root_cause="kubo wedged"),
            _ev_recommendation("docker.restart", {"container": "ipfs_host"}),
        ],
        "all_green": [
            _ev_session_started(),
            _ev_tool_call("c1"), _ev_tool_result("c1"),
            _ev_verdict("green", root_cause="device is healthy"),
        ],
    }
    rpt = run_eval(scenarios, ScriptedBackend(scripts),
                   whitelist_names={"docker.restart"})
    assert rpt.promoted
    assert rpt.n_scenarios == 2
    assert rpt.verdict_severity_correct == 2


def test_e2e_broken_model_rejected():
    """A scripted backend that always says everything is green when
    the truth is red must fail the gate."""
    scenarios = [_scen(id=f"s{i}", expected_severity="red",
                       expected_root_cause_keywords=["kubo"],
                       expected_recommended_actions=[])
                 for i in range(10)]
    scripts = {sc.id: [
        _ev_session_started(),
        _ev_verdict("green", root_cause="device is healthy"),
    ] for sc in scenarios}
    rpt = run_eval(scenarios, ScriptedBackend(scripts), whitelist_names=set())
    assert not rpt.promoted
    assert rpt.verdict_severity_correct == 0


# ---------------------------------------------------------------------------
# Scenario YAML loading
# ---------------------------------------------------------------------------

def test_load_real_test_set():
    """The 15 canonical scenarios in corpus/test_set/ load cleanly."""
    scenarios = load_scenarios(_REPO_ROOT / "corpus" / "test_set")
    assert len(scenarios) >= 15
    # Every scenario has a unique id
    ids = [s.id for s in scenarios]
    assert len(set(ids)) == len(ids)
    # Severity is in the closed enum
    for s in scenarios:
        assert s.expected_severity in ("green", "yellow", "red")


def test_load_whitelist_returns_known_actions():
    """Cross-validates against the real fula-ota whitelist if checked
    out as a sibling; otherwise skips."""
    sibling = _REPO_ROOT.parent / "fula-ota" / "docker" / "fxsupport" / "linux" / "plugins" / "blox-ai" / "action_whitelist.json"
    if not sibling.is_file():
        pytest.skip("fula-ota sibling checkout not present")
    names = load_whitelist_names(sibling)
    assert "docker.restart" in names
    assert "ntp.resync" in names
