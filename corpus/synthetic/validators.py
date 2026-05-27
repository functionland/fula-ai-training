"""Post-generation validation for synthetic Blox AI training data.

Every generated `*.labelled.json` file passes through these checks
BEFORE being written. A failing example is logged + skipped (never
silently shipped). Stats reported at end of generation.

Validation rules (per advisor sign-off):
  1. XML tags lowercase strict
  2. Never <verdict> + <tool_call> in same assistant turn
  3. <recommendation>.action_name must be in action_whitelist
  4. <recommendation>.args match argument_constraints
  5. <verdict>.severity in {green,yellow,red}
  6. <think> blocks open + close balanced
  7. No Kubernetes terms in assistant output (unless k8s-correction scenario)
  8. tool_call.payload.tool in diag/* enum
  9. tool_result.call_id matches preceding tool_call.call_id
  10. session_started always first event
  11. events count <= 200 (Phase 20 schema cap)
  12. Token estimate <= 4096 (matches qwen3b_lora.yaml:tokenizer.max_length)
"""
from __future__ import annotations

import json
import re
from typing import Any

from .scenarios import (
    DIAG_TOOLS, TIER_2_ACTIONS, TIER_3_ACTIONS, ARG_CONSTRAINTS,
)


# Kubernetes terms banned per system-prompt rule 11
K8S_BANNED_RE = re.compile(
    r"\b(kubelet|kube-?proxy|kubectl|k8s|kubeadm|kustomize|helm|"
    r"crashloopbackoff|kube-apiserver|kube-scheduler|kube-controller-manager|"
    r"etcd-cluster|coredns)\b",
    re.IGNORECASE,
)
# Note: "kubo" is NOT banned (kubo is the IPFS daemon rename, NOT kubernetes)
# Note: "pod" is allowed in user prompts (we WANT k8s-correction examples)
#   but not in assistant output. Caller toggles via `allow_k8s_in_user`.

# XML tag detector — strict lowercase
XML_TAG_RE = re.compile(r"</?(tool_call|verdict|recommendation|think|tool_response)>")
XML_TAG_BAD_CASE_RE = re.compile(
    r"</?(Tool_Call|Verdict|Recommendation|Think|TOOL_CALL|VERDICT|RECOMMENDATION|THINK)>"
)


class ValidationError(Exception):
    pass


def validate_assistant_content(content: str, *, is_k8s_correction_scenario: bool = False) -> None:
    """Validate one assistant message's content string."""
    # Rule 1: lowercase XML
    if XML_TAG_BAD_CASE_RE.search(content):
        raise ValidationError("Non-lowercase XML tag found")

    # Rule 2: never <verdict> + <tool_call> in same turn
    has_verdict = "<verdict>" in content
    has_tool_call = "<tool_call>" in content
    if has_verdict and has_tool_call:
        raise ValidationError("<verdict> and <tool_call> in same assistant turn")

    # Rule 6: <think> open/close balanced
    n_think_open = content.count("<think>")
    n_think_close = content.count("</think>")
    if n_think_open != n_think_close:
        raise ValidationError(
            f"<think> blocks unbalanced: {n_think_open} open vs {n_think_close} close"
        )
    # No nested think
    if "<think>" in content:
        inner = content.split("<think>", 1)[1].split("</think>", 1)[0]
        if "<think>" in inner:
            raise ValidationError("Nested <think> block")

    # Rule 7: no Kubernetes terms in assistant output (unless k8s-correction)
    if not is_k8s_correction_scenario and K8S_BANNED_RE.search(content):
        match = K8S_BANNED_RE.search(content)
        raise ValidationError(f"Kubernetes term '{match.group()}' in assistant output")

    # Sanity: no markdown numbered-action lists (rule 7 inverse)
    # Detect "1. <some-action>" or "- restart_fula" patterns at line start
    md_action_re = re.compile(
        r"^\s*(?:\d+\.\s*\*?\*?|[-*]\s*)(?:restart_fula|docker\.restart|"
        r"systemctl\.restart|ntp\.resync|wireguard\.bounce|reset|"
        r"partition|node_delete|ipfs_delete|force_update)\b",
        re.MULTILINE,
    )
    if md_action_re.search(content):
        raise ValidationError(
            "Markdown numbered/bulleted action list (should be <recommendation> XML)"
        )


def validate_recommendation(rec_dict: dict[str, Any]) -> None:
    """Validate one recommendation event's structured fields."""
    name = rec_dict.get("action_name")
    if name not in TIER_2_ACTIONS and name not in TIER_3_ACTIONS:
        raise ValidationError(
            f"recommendation.action_name {name!r} not in whitelist"
        )
    args = rec_dict.get("args") or {}
    if not isinstance(args, dict):
        raise ValidationError("recommendation.args must be a dict")
    # Check argument_constraints
    constraints = ARG_CONSTRAINTS.get(name)
    if constraints:
        for arg_key, allowed in constraints.items():
            if arg_key in args and args[arg_key] not in allowed:
                raise ValidationError(
                    f"recommendation.args.{arg_key}={args[arg_key]!r} not in "
                    f"allowed {allowed}"
                )
    # Severity check via tier
    tier = rec_dict.get("tier")
    if tier not in (2, 3):
        raise ValidationError(f"recommendation.tier must be 2 or 3, got {tier}")
    if tier == 2 and name not in TIER_2_ACTIONS:
        raise ValidationError(f"tier-2 mismatch for {name}")
    if tier == 3 and name not in TIER_3_ACTIONS:
        raise ValidationError(f"tier-3 mismatch for {name}")
    conf = rec_dict.get("confidence")
    if not isinstance(conf, (int, float)) or not 0.0 <= conf <= 1.0:
        raise ValidationError(f"recommendation.confidence {conf!r} out of [0,1]")


def validate_verdict(verdict_dict: dict[str, Any]) -> None:
    sev = verdict_dict.get("severity")
    if sev not in ("green", "yellow", "red"):
        raise ValidationError(f"verdict.severity {sev!r} not in green/yellow/red")
    summary = verdict_dict.get("summary", "")
    if not isinstance(summary, str) or len(summary) < 1 or len(summary) > 500:
        raise ValidationError(f"verdict.summary length {len(summary)} out of [1, 500]")


def validate_tool_call(tc_dict: dict[str, Any]) -> None:
    tool = tc_dict.get("payload", {}).get("tool")
    if tool not in DIAG_TOOLS:
        raise ValidationError(f"tool_call.tool {tool!r} not in diag/* enum")


def validate_labelled_record(record: dict[str, Any]) -> None:
    """Validate one full `*.labelled.json` record before writing.

    Raises ValidationError on any failure. Caller catches + logs +
    skips that example."""
    transcript = record.get("transcript", {})
    events = transcript.get("events", [])

    # Rule 11: events count cap (Phase 20 schema)
    if len(events) > 200:
        raise ValidationError(f"events count {len(events)} exceeds 200")
    if len(events) < 1:
        raise ValidationError("events empty")

    # Rule 10: session_started always first
    if events[0].get("type") != "session_started":
        raise ValidationError(
            f"first event must be session_started, got {events[0].get('type')!r}"
        )

    # Validate each event type
    is_k8s_scenario = "k8s" in record.get("notes", "").lower() or \
                      "kubernetes" in record.get("notes", "").lower()
    open_tool_calls: dict[str, dict] = {}
    for ev in events:
        etype = ev.get("type")
        if etype == "tool_call":
            validate_tool_call(ev)
            open_tool_calls[ev["call_id"]] = ev
        elif etype == "tool_result":
            cid = ev.get("call_id")
            if cid not in open_tool_calls:
                raise ValidationError(
                    f"tool_result.call_id {cid!r} has no preceding tool_call"
                )
            del open_tool_calls[cid]
            if ev.get("ok") is False and not ev.get("error"):
                raise ValidationError(
                    "tool_result with ok=false must include error field"
                )
        elif etype == "verdict":
            validate_verdict(ev.get("payload", {}))
        elif etype == "recommended_action":
            validate_recommendation(ev)
        elif etype == "thought":
            payload = ev.get("payload", "")
            if not isinstance(payload, str) or len(payload) < 1 or len(payload) > 4000:
                raise ValidationError(
                    f"thought.payload length {len(payload)} out of [1, 4000]"
                )
            validate_assistant_content(
                payload, is_k8s_correction_scenario=is_k8s_scenario,
            )

    # Token estimate (~4 chars/token average is fine for our short XML)
    total_chars = sum(
        len(str(ev.get("payload") or "")) + 100
        for ev in events
    )
    approx_tokens = total_chars // 4
    if approx_tokens > 4096:
        raise ValidationError(
            f"approx token count {approx_tokens} exceeds 4096 cap"
        )


def lint_thought_content(text: str) -> list[str]:
    """Non-fatal style lints. Caller decides whether to skip or just warn."""
    warnings = []
    if text.endswith("..."):
        warnings.append("thought ends with '...' (likely truncated)")
    if "```" in text:
        warnings.append("thought contains markdown code fence (use XML tags instead)")
    return warnings
