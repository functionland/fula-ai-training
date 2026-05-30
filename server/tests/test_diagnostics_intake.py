"""Tests for the POST /diagnostics intake endpoint.

/diagnostics is the user-shared "Send to Support" path. It deliberately
differs from /transcripts:
  - NO anonymization / find_pii gate. The bundle is SUPPOSED to carry the
    blox kubo/cluster peer ids + app peer id so support can correlate the
    device. The whole point is identifiability, so a PII scanner here would
    be wrong.
  - No transcript JSON-Schema. Only minimal structural checks:
    kind == "diagnostics" and a canonical-lowercase-uuid upload_id.
  - Size-capped at DIAGNOSTICS_MAX_BYTES (256 KiB default).

It SHARES with /transcripts: the per-IP/global rate limiter, the
UUID-keyed storage tree, and the admin inbox (rows labelled
source='diagnostics').
"""
from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _reset_storage_and_limits(monkeypatch, tmp_path_factory):
    # Isolate each test's storage dir, mirroring test_intake.py.
    storage_dir = tmp_path_factory.mktemp("diag_intake")
    monkeypatch.setenv("BLOX_AI_STORAGE_DIR", str(storage_dir))
    import importlib
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


def _good_diagnostics():
    """Mirror what the app's buildDiagnosticsPayload() emits (see
    apps/box/src/utils/diagnosticsUpload.ts). The server only enforces
    kind + canonical upload_id; the rest is free-form by design."""
    return {
        "kind": "diagnostics",
        "upload_id": str(uuid.uuid4()),  # canonical lowercase v4
        "generated_at": "2026-05-29T00:00:00.000Z",
        "phone": {
            "blox_kubo_peer_id": "12D3KooWBLAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "blox_cluster_peer_id": "12D3KooWCLUaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "app_peer_id": "12D3KooWAPPaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "phone_internet": "ok",
            "discovery_service": "failed",
            "relays": [{"dns_name": "relay1.fx.land", "status": "ok"}],
            "transport_used": "lan-http",
            "app_platform": "android",
        },
        "blox": {
            "generated_at": "2026-05-29T00:00:00.000Z",
            "tools": {"internet": {"dns_ok": True, "https_discovery_ok": False}},
        },
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_happy_path_persists(client):
    body = _good_diagnostics()
    r = client.post("/diagnostics", json=body)
    assert r.status_code == 200
    assert r.json() == {}, "response MUST be empty (no echo, no fingerprint)"
    # File landed under the same UUID-keyed tree /transcripts uses.
    import os
    storage_dir = Path(os.environ["BLOX_AI_STORAGE_DIR"])
    uid = body["upload_id"]
    written = storage_dir / uid[:2] / uid[2:4] / f"{uid}.json"
    assert written.exists(), "diagnostics bundle should have been persisted"
    persisted = json.loads(written.read_text())
    assert persisted == body, "server MUST persist payload byte-for-byte"


def test_response_body_is_empty_and_does_not_echo_upload_id(client):
    body = _good_diagnostics()
    r = client.post("/diagnostics", json=body)
    assert r.status_code == 200
    assert r.json() == {}
    assert body["upload_id"] not in r.text


def test_idempotent_on_same_upload_id(client):
    body = _good_diagnostics()
    r1 = client.post("/diagnostics", json=body)
    r2 = client.post("/diagnostics", json=body)
    assert r1.status_code == 200
    assert r2.status_code == 200
    import os
    storage_dir = Path(os.environ["BLOX_AI_STORAGE_DIR"])
    files = list(storage_dir.rglob("*.json"))
    assert len(files) == 1


# ---------------------------------------------------------------------------
# NO PII gate — the defining difference from /transcripts
# ---------------------------------------------------------------------------

def test_peer_ids_and_ips_are_accepted_not_rejected(client):
    """A diagnostics bundle full of identifiers (peer ids, IPs, home paths,
    wall-clock timestamps) MUST be accepted. The same content posted to
    /transcripts would be rejected by the anonymization scanner — that
    asymmetry is the whole point of a separate endpoint."""
    body = _good_diagnostics()
    body["blox"]["tools"]["wireguard"] = {
        "endpoint": "192.168.1.55:51820",
        "config_path": "/home/pi/.fula/wg.conf",
        "last_handshake": "2026-05-23T12:34:56Z",
    }
    r = client.post("/diagnostics", json=body)
    assert r.status_code == 200, (
        "diagnostics intake must NOT run the PII scanner; identifiers are "
        "the point of a support bundle"
    )


# ---------------------------------------------------------------------------
# Structural validation — minimal, generic errors only
# ---------------------------------------------------------------------------

def test_non_json_body_returns_400(client):
    r = client.post("/diagnostics", content=b"not json at all",
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    assert r.json() == {"error": "body_invalid"}


def test_non_object_body_returns_400(client):
    r = client.post("/diagnostics", json=[1, 2, 3])
    assert r.status_code == 400
    assert r.json() == {"error": "body_invalid"}


def test_wrong_kind_returns_400(client):
    body = _good_diagnostics()
    body["kind"] = "transcript"  # must be exactly "diagnostics"
    r = client.post("/diagnostics", json=body)
    assert r.status_code == 400
    assert r.json() == {"error": "body_invalid"}


def test_missing_kind_returns_400(client):
    body = _good_diagnostics()
    del body["kind"]
    r = client.post("/diagnostics", json=body)
    assert r.status_code == 400
    assert r.json() == {"error": "body_invalid"}


@pytest.mark.parametrize("bad_uid,_desc", [
    ("not-a-uuid", "free text"),
    ("a1b2c3d4-e5f6-4789-abcd-ef012345678G", "non-hex char"),
    ("a1b2c3d4e5f64789abcdef0123456789", "hyphenless"),
    # Has hex LETTERS so uppercasing genuinely de-canonicalises it (an
    # all-zeros uuid would be a no-op under .upper()).
    ("a1b2c3d4-e5f6-4789-abcd-ef0123456789".upper(), "uppercase (non-canonical)"),
    ("", "empty"),
])
def test_non_canonical_upload_id_returns_400(client, bad_uid, _desc):
    """upload_id must be a canonical lowercase 36-char UUID string
    (str(uuid.UUID(s)) == s). Anything else → clean 400, never a 500
    from the storage layer."""
    body = _good_diagnostics()
    body["upload_id"] = bad_uid
    r = client.post("/diagnostics", json=body)
    assert r.status_code == 400
    assert r.json() == {"error": "body_invalid"}


def test_missing_upload_id_returns_400(client):
    body = _good_diagnostics()
    del body["upload_id"]
    r = client.post("/diagnostics", json=body)
    assert r.status_code == 400
    assert r.json() == {"error": "body_invalid"}


def test_non_string_upload_id_returns_400(client):
    body = _good_diagnostics()
    body["upload_id"] = 12345
    r = client.post("/diagnostics", json=body)
    assert r.status_code == 400
    assert r.json() == {"error": "body_invalid"}


# ---------------------------------------------------------------------------
# Size cap
# ---------------------------------------------------------------------------

def test_oversize_body_returns_413(client):
    """A bundle larger than DIAGNOSTICS_MAX_BYTES is rejected. The
    Content-Length precheck fires before the body is buffered."""
    import app as appmod
    body = _good_diagnostics()
    # Pad well past the 256 KiB cap.
    body["blox"]["tools"]["padding"] = "x" * (appmod.DIAGNOSTICS_MAX_BYTES + 1024)
    r = client.post("/diagnostics", json=body)
    assert r.status_code == 413
    assert r.json() == {"error": "body_too_large"}


def test_cap_is_independently_tunable_from_transcripts(client, monkeypatch):
    """Lower the diagnostics cap; a body that would otherwise be fine is
    now rejected. Confirms DIAGNOSTICS_MAX_BYTES is the gate (not the
    transcript path's limits)."""
    import app as appmod
    monkeypatch.setattr(appmod, "DIAGNOSTICS_MAX_BYTES", 256)
    body = _good_diagnostics()  # comfortably larger than 256 bytes
    r = client.post("/diagnostics", json=body)
    assert r.status_code == 413
    assert r.json() == {"error": "body_too_large"}


# ---------------------------------------------------------------------------
# Rate limit (shared limiter with /transcripts)
# ---------------------------------------------------------------------------

def test_rate_limit_per_ip_kicks_in(client, monkeypatch):
    import app as appmod
    monkeypatch.setattr(appmod, "RATE_LIMIT_PER_IP_PER_WINDOW", 3)
    for _ in range(3):
        r = client.post("/diagnostics", json=_good_diagnostics())
        assert r.status_code == 200
    r4 = client.post("/diagnostics", json=_good_diagnostics())
    assert r4.status_code == 429
    assert r4.json() == {"error": "rate_limit"}


def test_rate_limit_is_shared_with_transcripts(client, monkeypatch):
    """The per-IP window counts /diagnostics and /transcripts together —
    one shared limiter, so a flood on either path throttles both."""
    import app as appmod
    monkeypatch.setattr(appmod, "RATE_LIMIT_PER_IP_PER_WINDOW", 2)
    r1 = client.post("/diagnostics", json=_good_diagnostics())
    assert r1.status_code == 200
    # _good_diagnostics is not a valid transcript, but the rate-limit check
    # runs BEFORE body parsing, so this still consumes a token.
    r2 = client.post("/transcripts", json={"anything": True})
    assert r2.status_code in (400, 200)  # consumed a token regardless
    r3 = client.post("/diagnostics", json=_good_diagnostics())
    assert r3.status_code == 429
    assert r3.json() == {"error": "rate_limit"}


# ---------------------------------------------------------------------------
# Privacy: source IP never persisted
# ---------------------------------------------------------------------------

def test_persisted_file_has_no_request_metadata(client):
    body = _good_diagnostics()
    client.post("/diagnostics", json=body)
    import os
    storage_dir = Path(os.environ["BLOX_AI_STORAGE_DIR"])
    files = list(storage_dir.rglob("*.json"))
    assert len(files) == 1
    persisted = json.loads(files[0].read_text())
    assert persisted == body
    for p in storage_dir.rglob("*"):
        if p.is_file():
            assert "testclient" not in p.name.lower()
            assert "127.0.0.1" not in p.read_text()


# ---------------------------------------------------------------------------
# Admin inbox: diagnostics rows are labelled source='diagnostics'
# (lifespan-driven so the issue DB is initialised — mirrors test_admin.py)
# ---------------------------------------------------------------------------

ADMIN_TOKEN = "test-admin-token-not-real"


def test_diagnostics_post_inserts_issue_with_source_diagnostics(
    monkeypatch, tmp_path_factory,
):
    storage_dir = tmp_path_factory.mktemp("diag_admin_storage")
    db_dir = tmp_path_factory.mktemp("diag_admin_db")
    monkeypatch.setenv("BLOX_AI_STORAGE_DIR", str(storage_dir))
    monkeypatch.setenv("BLOX_AI_DB_PATH", str(db_dir / "issues.sqlite"))
    monkeypatch.setenv("BLOX_AI_ADMIN_TOKEN", ADMIN_TOKEN)
    for mod in ("app", "admin", "issue_db", "anonymization_check", "storage"):
        sys.modules.pop(mod, None)
    import importlib
    import app as appmod
    importlib.reload(appmod)
    appmod._reset_rate_limit_for_tests()
    import admin as adminmod
    adminmod._reset_auth_state_for_tests()

    auth = {"Authorization": f"Bearer {ADMIN_TOKEN}"}
    with TestClient(appmod.app) as client:
        body = _good_diagnostics()
        r = client.post("/diagnostics", json=body)
        assert r.status_code == 200
        # Appears in the admin inbox, labelled source='diagnostics'.
        r2 = client.get("/admin/issues?status=new", headers=auth)
        assert r2.status_code == 200
        j = r2.json()
        assert j["total"] == 1
        assert j["items"][0]["id"] == body["upload_id"]
        assert j["items"][0]["source"] == "diagnostics"
        assert j["items"][0]["status"] == "new"
        # Detail inlines the bundle payload.
        r3 = client.get(f"/admin/issues/{body['upload_id']}", headers=auth)
        assert r3.status_code == 200
        assert r3.json()["payload"] == body
