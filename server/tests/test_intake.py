"""Phase 20 — intake server tests.

Coverage:
- Happy path: well-formed, anonymized transcript → 200, persisted.
- Schema violations: missing fields, wrong types → 400 with generic error.
- Anonymization scanner catches IPv4, IPv6, peerId, home path, wall-clock ts.
- Idempotency: same upload_id twice → second write is a no-op.
- Rate limit: per-IP cap blocks the (N+1)th request from same IP.
- Source IP never persisted alongside the transcript.
- Error responses contain NO field-level detail (no schema fingerprinting).
"""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _reset_storage_and_limits(monkeypatch, tmp_path_factory):
    # Isolate each test's storage dir.
    storage_dir = tmp_path_factory.mktemp("intake")
    monkeypatch.setenv("BLOX_AI_STORAGE_DIR", str(storage_dir))
    # Re-import app fresh so the new storage dir is picked up.
    import importlib, sys
    for mod in ("app", "anonymization_check", "storage"):
        sys.modules.pop(mod, None)
    import app as appmod
    importlib.reload(appmod)
    appmod._reset_rate_limit_for_tests()
    yield appmod
    if Path(storage_dir).exists():
        shutil.rmtree(storage_dir, ignore_errors=True)


@pytest.fixture
def client(_reset_storage_and_limits):
    return TestClient(_reset_storage_and_limits.app)


def _good_transcript():
    return {
        "schema_version": 1,
        "upload_id": str(uuid.uuid4()),
        "session_relative_start": "+0s",
        "events": [
            {"type": "session_started", "relative_ts": "+0s", "payload": {}},
            {"type": "thought", "relative_ts": "+1s",
             "payload": "checking discovery reachability"},
            {"type": "tool_call", "relative_ts": "+2s",
             "payload": {"tool": "diag/internet", "args": {}}},
            {"type": "tool_result", "relative_ts": "+3s",
             "payload": {"dns_ok": True, "https_discovery_ok": False}},
            {"type": "verdict", "relative_ts": "+5s",
             "payload": {"summary": "discovery unreachable",
                         "severity": "yellow"}},
        ],
        "user_rating": 1,
        "consent": {
            "explicit_opt_in": True,
            "preview_shown": True,
            "anonymizer_version": "0.1.0",
        },
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_happy_path_persists(client):
    body = _good_transcript()
    r = client.post("/transcripts", json=body)
    assert r.status_code == 200
    assert r.json() == {}, "response MUST be empty (no echo, no fingerprint)"
    # Verify the file landed in storage with the expected content.
    import os
    storage_dir = Path(os.environ["BLOX_AI_STORAGE_DIR"])
    uid = body["upload_id"]
    written = storage_dir / uid[:2] / uid[2:4] / f"{uid}.json"
    assert written.exists(), "transcript should have been persisted"
    persisted = json.loads(written.read_text())
    assert persisted == body
    # Sanity: nothing in storage names the client IP.
    for p in storage_dir.rglob("*"):
        if p.is_file():
            assert "testclient" not in p.name.lower()
            assert "127.0.0.1" not in p.read_text()


def test_idempotent_on_same_upload_id(client):
    body = _good_transcript()
    r1 = client.post("/transcripts", json=body)
    r2 = client.post("/transcripts", json=body)
    assert r1.status_code == 200
    assert r2.status_code == 200
    # Both 200, single file on disk.
    import os
    storage_dir = Path(os.environ["BLOX_AI_STORAGE_DIR"])
    files = list(storage_dir.rglob("*.json"))
    assert len(files) == 1


# ---------------------------------------------------------------------------
# Schema violations — must NOT leak field details
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mutate,_desc", [
    (lambda b: b.pop("schema_version"), "missing schema_version"),
    (lambda b: b.update({"schema_version": 2}), "unsupported schema_version"),
    (lambda b: b.pop("upload_id"), "missing upload_id"),
    (lambda b: b.update({"upload_id": "not-a-uuid"}), "malformed upload_id"),
    (lambda b: b.update({"session_relative_start": "+5s"}), "tripwire failed"),
    (lambda b: b.pop("consent"), "missing consent"),
    (lambda b: b["consent"].update({"explicit_opt_in": False}), "consent tripwire"),
    (lambda b: b["consent"].update({"preview_shown": False}), "preview tripwire"),
    (lambda b: b.update({"user_rating": 5}), "rating enum"),
    (lambda b: b.update({"events": []}), "events min"),
    (lambda b: b.update({"events": [{"type": "thought"}]}), "event missing relative_ts"),
    (lambda b: b.update({"surprise_field": True}), "additional property"),
])
def test_schema_violation_returns_generic_400(client, mutate, _desc):
    body = _good_transcript()
    mutate(body)
    r = client.post("/transcripts", json=body)
    assert r.status_code == 400
    j = r.json()
    assert j == {"error": "body_invalid"}, (
        "error body MUST be {'error':'body_invalid'} — no field-level detail")


def test_non_json_body_returns_400(client):
    r = client.post("/transcripts", data="not json at all")
    assert r.status_code == 400
    assert r.json() == {"error": "body_invalid"}


def test_non_object_body_returns_400(client):
    r = client.post("/transcripts", json=[1, 2, 3])
    assert r.status_code == 400
    assert r.json() == {"error": "body_invalid"}


# ---------------------------------------------------------------------------
# Anonymization scanner (defense-in-depth)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inject,scanner_name", [
    (lambda b: b["events"].append({"type": "thought", "relative_ts": "+1s",
                                   "payload": "saw IP 192.168.1.55 in logs"}),
     "ipv4_literal"),
    (lambda b: b["events"].append({"type": "thought", "relative_ts": "+1s",
                                   "payload": "fe80::1ff:fe23:4567:890a peer"}),
     "ipv6_literal"),
    (lambda b: b["events"].append({"type": "thought", "relative_ts": "+1s",
                                   "payload": "peer 12D3KooWBLAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}),
     "peerid_libp2p"),
    (lambda b: b["events"].append({"type": "thought", "relative_ts": "+1s",
                                   "payload": "ipfs CID QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG"}),
     "peerid_legacy"),
    (lambda b: b["events"].append({"type": "thought", "relative_ts": "+1s",
                                   "payload": "in /home/ehsan/.fula/config"}),
     "home_directory"),
    (lambda b: b["events"].append({"type": "thought", "relative_ts": "+1s",
                                   "payload": "happened at 2026-05-23T12:34:56Z"}),
     "wallclock_ts"),
])
def test_anonymization_scanner_rejects(client, inject, scanner_name):
    body = _good_transcript()
    inject(body)
    r = client.post("/transcripts", json=body)
    assert r.status_code == 400
    assert r.json() == {"error": "anonymization_check_failed"}


def test_anonymization_scanner_allows_clean_transcript(client):
    """A transcript with NO PII patterns must pass."""
    body = _good_transcript()
    r = client.post("/transcripts", json=body)
    assert r.status_code == 200


def test_anonymization_response_does_not_leak_matched_value(client):
    """Hard rule: even if we reject, we MUST NOT echo the matched substring
    or the scanner name back to the client (only generic error)."""
    body = _good_transcript()
    body["events"].append({"type": "thought", "relative_ts": "+1s",
                           "payload": "leaked-ip-here 10.0.0.42"})
    r = client.post("/transcripts", json=body)
    assert r.status_code == 400
    text = r.text
    assert "10.0.0.42" not in text
    assert "ipv4_literal" not in text  # don't even leak the scanner name


# ---------------------------------------------------------------------------
# Rate limit
# ---------------------------------------------------------------------------

def test_rate_limit_per_ip_kicks_in(client, monkeypatch):
    """Lower the cap to 3 for this test; submit 4 transcripts from same IP."""
    import app as appmod
    monkeypatch.setattr(appmod, "RATE_LIMIT_PER_IP_PER_WINDOW", 3)
    for _ in range(3):
        r = client.post("/transcripts", json=_good_transcript())
        assert r.status_code == 200
    r4 = client.post("/transcripts", json=_good_transcript())
    assert r4.status_code == 429
    assert r4.json() == {"error": "rate_limit"}


def test_rate_limit_global_kicks_in(client, monkeypatch):
    import app as appmod
    monkeypatch.setattr(appmod, "RATE_LIMIT_PER_IP_PER_WINDOW", 100)
    monkeypatch.setattr(appmod, "GLOBAL_RATE_LIMIT_PER_WINDOW", 2)
    r1 = client.post("/transcripts", json=_good_transcript())
    r2 = client.post("/transcripts", json=_good_transcript())
    r3 = client.post("/transcripts", json=_good_transcript())
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 429


# ---------------------------------------------------------------------------
# Privacy guarantees
# ---------------------------------------------------------------------------

def test_response_body_is_empty_on_success(client):
    """No echo, no derived fingerprint, no upload_id reflection."""
    body = _good_transcript()
    r = client.post("/transcripts", json=body)
    assert r.status_code == 200
    j = r.json()
    assert j == {}
    # Specifically — must NOT echo the upload_id (a fingerprint-correlation surface).
    assert body["upload_id"] not in r.text


def test_persisted_file_has_no_request_metadata(client):
    """The on-disk transcript must equal the request body exactly. No
    client IP, no headers, no request_id added on the server side."""
    body = _good_transcript()
    client.post("/transcripts", json=body)
    import os
    storage_dir = Path(os.environ["BLOX_AI_STORAGE_DIR"])
    files = list(storage_dir.rglob("*.json"))
    assert len(files) == 1
    persisted = json.loads(files[0].read_text())
    assert persisted == body, "server MUST persist payload byte-for-byte; no augmentation"
