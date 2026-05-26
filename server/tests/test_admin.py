"""Tests for the status-tracked-inbox admin endpoints.

Covers:
- Bearer auth required on every /admin/* endpoint (no token, wrong token)
- Auth-failure rate limit (Gemini's defense-in-depth)
- List with filtering (status, since, limit, offset, pagination metadata)
- Detail endpoint inlines transcript payload
- Status update (with history written)
- Admin-filed issue with envelope schema + 1 MB body cap
- Delete (DB row + transcript file)
- POST /transcripts side-effect: issue row inserted with status='new'
- Background migration picks up existing transcript files
- Cache-Control: no-store on every admin response
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


ADMIN_TOKEN = "test-admin-token-not-real"


def _good_transcript():
    return {
        "schema_version": 1,
        "upload_id": str(uuid.uuid4()),
        "session_relative_start": "+0s",
        "events": [
            {"type": "session_started", "relative_ts": "+0s", "payload": {}},
            {"type": "thought", "relative_ts": "+1s",
             "payload": "checking discovery reachability"},
            {"type": "verdict", "relative_ts": "+5s",
             "payload": {"summary": "ok", "severity": "green"}},
        ],
        "user_rating": 1,
        "consent": {
            "explicit_opt_in": True,
            "preview_shown": True,
            "anonymizer_version": "0.1.0",
        },
    }


@pytest.fixture
def app_and_client(monkeypatch, tmp_path_factory):
    """Fresh app + storage + DB per test. Drives lifespan startup via
    TestClient's context-manager protocol so the issue DB is initialised
    AND the background migration has a chance to run."""
    storage_dir = tmp_path_factory.mktemp("admin_test_storage")
    db_dir = tmp_path_factory.mktemp("admin_test_db")
    monkeypatch.setenv("BLOX_AI_STORAGE_DIR", str(storage_dir))
    monkeypatch.setenv("BLOX_AI_DB_PATH", str(db_dir / "issues.sqlite"))
    monkeypatch.setenv("BLOX_AI_ADMIN_TOKEN", ADMIN_TOKEN)

    # Drop cached modules so the env var is read fresh.
    for mod in ("app", "admin", "issue_db",
                "anonymization_check", "storage"):
        sys.modules.pop(mod, None)
    import importlib
    import app as appmod
    importlib.reload(appmod)
    appmod._reset_rate_limit_for_tests()

    import admin as adminmod
    adminmod._reset_auth_state_for_tests()

    # `with TestClient(...)` triggers FastAPI lifespan — DB initialises here.
    with TestClient(appmod.app) as client:
        yield appmod, client

    if storage_dir.exists():
        shutil.rmtree(storage_dir, ignore_errors=True)
    if db_dir.exists():
        shutil.rmtree(db_dir, ignore_errors=True)


def auth():
    return {"Authorization": f"Bearer {ADMIN_TOKEN}"}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "/admin/issues",
    "/admin/issues/00000000-0000-4000-8000-000000000000",
    "/admin/issues/00000000-0000-4000-8000-000000000000/history",
])
def test_admin_data_endpoints_require_bearer(app_and_client, path):
    """JSON-returning endpoints require bearer. Browser hits without
    a token get 401. (The HTML shell itself is intentionally public —
    see test_admin_html_is_public_for_token_prompt below.)"""
    _, client = app_and_client
    r = client.get(path)
    assert r.status_code == 401
    assert r.json() == {"detail": {"error": "auth_required"}}


def test_admin_html_is_public_for_token_prompt(app_and_client):
    """GET /admin/ and /admin must serve the HTML shell without auth,
    so the user can SEE the token-prompt UI on first load.

    Bug regression guard 2026-05-26: previously _admin_root_no_slash
    + admin_ui both called _require_bearer, which made the browser
    receive a JSON `{"detail":{"error":"auth_required"}}` body
    instead of the HTML page. Users had no way to enter the token
    (the prompt itself lives in the HTML).

    The HTML is empty shell + token-prompt JS only — no sensitive
    data. All data-bearing endpoints stay auth'd."""
    _, client = app_and_client
    for path in ("/admin/", "/admin"):
        r = client.get(path)
        assert r.status_code == 200, (
            f"{path} returned {r.status_code}; HTML shell must be public "
            f"so the token-prompt UI can render"
        )
        assert "html" in r.headers.get("content-type", "").lower()
        assert "Blox AI inbox" in r.text, (
            "expected admin_ui.html body to be served"
        )
        # The shell tells the user about the token (visible in HTML).
        assert "BLOX_AI_ADMIN_TOKEN" in r.text, (
            "HTML should reference the token env var so users know where "
            "to look for it"
        )
        # No-store still applies via middleware.
        assert r.headers.get("cache-control") == "no-store"


def test_admin_rejects_wrong_token(app_and_client):
    _, client = app_and_client
    r = client.get("/admin/issues", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_admin_accepts_right_token(app_and_client):
    _, client = app_and_client
    r = client.get("/admin/issues", headers=auth())
    assert r.status_code == 200


def test_admin_auth_failure_rate_limit_returns_429(app_and_client):
    """11 wrong-token attempts in <60s -> 429 instead of 401."""
    _, client = app_and_client
    # 10 allowed (auth-fail bucket fills), 11th triggers 429.
    for _ in range(10):
        r = client.get("/admin/issues", headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401
    r11 = client.get("/admin/issues", headers={"Authorization": "Bearer wrong"})
    assert r11.status_code == 429
    assert r11.json() == {"detail": {"error": "auth_rate_limit"}}


def test_admin_missing_token_env_returns_503(monkeypatch, tmp_path_factory):
    """If the operator forgets to set BLOX_AI_ADMIN_TOKEN, admin endpoints
    must refuse rather than silently allowing every request."""
    storage_dir = tmp_path_factory.mktemp("s")
    db_dir = tmp_path_factory.mktemp("d")
    monkeypatch.setenv("BLOX_AI_STORAGE_DIR", str(storage_dir))
    monkeypatch.setenv("BLOX_AI_DB_PATH", str(db_dir / "issues.sqlite"))
    monkeypatch.delenv("BLOX_AI_ADMIN_TOKEN", raising=False)
    for mod in ("app", "admin", "issue_db",
                "anonymization_check", "storage"):
        sys.modules.pop(mod, None)
    import importlib
    import app as appmod
    importlib.reload(appmod)
    appmod._reset_rate_limit_for_tests()
    with TestClient(appmod.app) as client:
        # Even with a "correct-looking" header, server has no token configured.
        r = client.get("/admin/issues", headers={"Authorization": "Bearer anything"})
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# Cache-Control: no-store on every admin response
# ---------------------------------------------------------------------------

def test_admin_responses_carry_no_store(app_and_client):
    _, client = app_and_client
    r = client.get("/admin/issues", headers=auth())
    assert r.headers.get("cache-control") == "no-store"
    # HTML shell is public now (no auth header) but no-store still applies.
    r2 = client.get("/admin/")
    assert r2.headers.get("cache-control") == "no-store"


def test_admin_error_responses_also_carry_no_store(app_and_client):
    """Codex review fix: 401/422/413/404 must also be no-store, not just
    successes. Middleware enforces this on every /admin/* response."""
    _, client = app_and_client
    # 401 — no token
    r = client.get("/admin/issues")
    assert r.status_code == 401
    assert r.headers.get("cache-control") == "no-store"
    # 404 — missing issue
    r2 = client.get("/admin/issues/00000000-0000-4000-8000-000000000000",
                    headers=auth())
    assert r2.status_code == 404
    assert r2.headers.get("cache-control") == "no-store"
    # 413 — oversize admin issue
    huge = "x" * (1024 * 1024 + 100)
    r3 = client.post("/admin/issues",
                     json={"summary": "x", "source": "admin",
                           "payload": {"data": huge}},
                     headers=auth())
    assert r3.status_code == 413
    assert r3.headers.get("cache-control") == "no-store"
    # 422 — pydantic rejected the status value
    body = _good_transcript()
    client.post("/transcripts", json=body)
    r4 = client.patch(f"/admin/issues/{body['upload_id']}",
                      json={"status": "banana"}, headers=auth())
    assert r4.status_code == 422
    assert r4.headers.get("cache-control") == "no-store"


def test_admin_500_also_carries_no_store(app_and_client):
    """Copilot review fix: even when a handler raises an unhandled
    exception, the resulting 500 must carry Cache-Control: no-store
    for /admin/* paths."""
    appmod, client = app_and_client
    # Close the DB connection mid-flight to force a sqlite3.ProgrammingError
    # on the next query. Caught by the middleware's except clause.
    appmod._db_conn.close()
    try:
        r = client.get("/admin/issues", headers=auth())
        assert r.status_code == 500
        assert r.headers.get("cache-control") == "no-store", (
            "500 responses on /admin/* must also be no-store"
        )
        # The middleware also masks the underlying error message.
        assert r.json() == {"detail": "internal_error"}
    finally:
        # Re-open so the fixture teardown doesn't choke.
        appmod._db_conn = appmod.issue_db.open_db(
            __import__("pathlib").Path(os.environ["BLOX_AI_DB_PATH"])
        )


def test_admin_content_length_precheck_413(app_and_client):
    """A truthful oversized Content-Length should be rejected BEFORE
    buffering the body. Cheap reject; defense in depth on the after-read
    length check that already covers lying-CL + chunked transfer."""
    _, client = app_and_client
    big_payload = json.dumps({"summary": "x", "source": "admin",
                              "payload": {"data": "x" * (1024 * 1024 + 50)}})
    r = client.post("/admin/issues",
                    content=big_payload.encode("utf-8"),
                    headers={**auth(),
                             "Content-Type": "application/json",
                             "Content-Length": str(len(big_payload))})
    assert r.status_code == 413


# ---------------------------------------------------------------------------
# Transcript-side hook: POST /transcripts inserts an issue row
# ---------------------------------------------------------------------------

def test_transcript_post_inserts_issue(app_and_client):
    _, client = app_and_client
    body = _good_transcript()
    r = client.post("/transcripts", json=body)
    assert r.status_code == 200
    # The issue should now appear in the list with status='new'.
    r2 = client.get("/admin/issues?status=new", headers=auth())
    assert r2.status_code == 200
    j = r2.json()
    assert j["total"] == 1
    assert j["items"][0]["id"] == body["upload_id"]
    assert j["items"][0]["source"] == "transcript"
    assert j["items"][0]["status"] == "new"


def test_transcript_post_is_idempotent_for_issue_row(app_and_client):
    _, client = app_and_client
    body = _good_transcript()
    client.post("/transcripts", json=body)
    client.post("/transcripts", json=body)
    r = client.get("/admin/issues?status=new", headers=auth())
    assert r.json()["total"] == 1


# ---------------------------------------------------------------------------
# List filtering + pagination metadata
# ---------------------------------------------------------------------------

def test_list_returns_pagination_metadata(app_and_client):
    _, client = app_and_client
    for _ in range(5):
        client.post("/transcripts", json=_good_transcript())
    r = client.get("/admin/issues?status=new&limit=2&offset=0", headers=auth())
    j = r.json()
    assert j["total"] == 5
    assert j["limit"] == 2
    assert j["offset"] == 0
    assert len(j["items"]) == 2

    r2 = client.get("/admin/issues?status=new&limit=2&offset=4", headers=auth())
    j2 = r2.json()
    assert j2["total"] == 5
    assert len(j2["items"]) == 1


def test_list_filter_by_multiple_statuses(app_and_client):
    appmod, client = app_and_client
    body = _good_transcript()
    client.post("/transcripts", json=body)
    # Move to reviewed
    client.patch(f"/admin/issues/{body['upload_id']}",
                 json={"status": "reviewed"}, headers=auth())
    # New transcript stays 'new'
    other = _good_transcript()
    client.post("/transcripts", json=other)

    r = client.get("/admin/issues?status=new&status=reviewed", headers=auth())
    j = r.json()
    assert j["total"] == 2


def test_list_rejects_invalid_status(app_and_client):
    _, client = app_and_client
    r = client.get("/admin/issues?status=banana", headers=auth())
    assert r.status_code == 400
    assert r.json() == {"detail": {"error": "invalid_status"}}


def test_list_rejects_invalid_since(app_and_client):
    _, client = app_and_client
    r = client.get("/admin/issues?since=not-a-date", headers=auth())
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Detail endpoint
# ---------------------------------------------------------------------------

def test_detail_inlines_transcript_payload(app_and_client):
    _, client = app_and_client
    body = _good_transcript()
    client.post("/transcripts", json=body)
    r = client.get(f"/admin/issues/{body['upload_id']}", headers=auth())
    assert r.status_code == 200
    j = r.json()
    assert j["id"] == body["upload_id"]
    assert j["source"] == "transcript"
    assert j["payload"] == body, "detail must inline the full transcript"


def test_detail_404_for_missing(app_and_client):
    _, client = app_and_client
    r = client.get("/admin/issues/00000000-0000-4000-8000-000000000000",
                   headers=auth())
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Admin-filed issues (POST /admin/issues)
# ---------------------------------------------------------------------------

def test_create_admin_issue_happy_path(app_and_client):
    _, client = app_and_client
    body = {"summary": "PII scanner flagged something",
            "source": "admin",
            "payload": {"observed_at": "2026-05-26T12:00:00Z",
                        "scanner": "ipv4_literal"}}
    r = client.post("/admin/issues", json=body, headers=auth())
    assert r.status_code == 201
    j = r.json()
    assert "id" in j and "created_at" in j

    # Appears in the new tab
    r2 = client.get("/admin/issues?status=new", headers=auth())
    j2 = r2.json()
    assert j2["total"] == 1
    assert j2["items"][0]["source"] == "admin"
    assert j2["items"][0]["summary"] == "PII scanner flagged something"

    # Detail returns the payload inline
    r3 = client.get(f"/admin/issues/{j['id']}", headers=auth())
    assert r3.status_code == 200
    assert r3.json()["payload"] == body["payload"]


def test_create_admin_issue_rejects_source_eq_transcript(app_and_client):
    """Mechanically prevent server-filed issues from posing as transcripts."""
    _, client = app_and_client
    body = {"summary": "sneaky", "source": "transcript", "payload": {}}
    r = client.post("/admin/issues", json=body, headers=auth())
    assert r.status_code == 400


def test_create_admin_issue_rejects_extra_fields(app_and_client):
    _, client = app_and_client
    body = {"summary": "x", "source": "admin", "payload": {}, "extra": "no"}
    r = client.post("/admin/issues", json=body, headers=auth())
    assert r.status_code == 400


def test_create_admin_issue_size_cap_413(app_and_client):
    """Gemini-flagged: 1 MB hard cap on admin-filed payloads to keep the UI
    from blowing up + bound disk-exhaustion."""
    _, client = app_and_client
    huge = "x" * (1024 * 1024 + 100)
    body = {"summary": "huge", "source": "admin", "payload": {"data": huge}}
    r = client.post("/admin/issues", json=body, headers=auth())
    assert r.status_code == 413
    assert r.json() == {"detail": {"error": "body_too_large"}}


# ---------------------------------------------------------------------------
# Status update + history
# ---------------------------------------------------------------------------

def test_patch_status_writes_history(app_and_client):
    _, client = app_and_client
    body = _good_transcript()
    client.post("/transcripts", json=body)
    uid = body["upload_id"]

    r = client.patch(f"/admin/issues/{uid}",
                     json={"status": "reviewed"}, headers=auth())
    assert r.status_code == 200
    assert r.json()["status"] == "reviewed"
    assert r.json()["changed"] is True

    h = client.get(f"/admin/issues/{uid}/history", headers=auth())
    rows = h.json()["history"]
    # Initial NULL -> 'new', then 'new' -> 'reviewed'. Newest first.
    assert len(rows) == 2
    assert rows[0]["from_status"] == "new"
    assert rows[0]["to_status"] == "reviewed"
    assert rows[1]["from_status"] is None
    assert rows[1]["to_status"] == "new"


def test_patch_status_same_value_is_idempotent(app_and_client):
    _, client = app_and_client
    body = _good_transcript()
    client.post("/transcripts", json=body)
    uid = body["upload_id"]
    # Already 'new' — patching to 'new' should be rejected by the regex
    # (we explicitly disallow API-side regression to 'new').
    r = client.patch(f"/admin/issues/{uid}",
                     json={"status": "new"}, headers=auth())
    assert r.status_code == 422  # pydantic validation


def test_patch_status_rejects_invalid_value(app_and_client):
    _, client = app_and_client
    body = _good_transcript()
    client.post("/transcripts", json=body)
    r = client.patch(f"/admin/issues/{body['upload_id']}",
                     json={"status": "banana"}, headers=auth())
    assert r.status_code == 422


def test_patch_status_404_for_missing(app_and_client):
    _, client = app_and_client
    r = client.patch(
        "/admin/issues/00000000-0000-4000-8000-000000000000",
        json={"status": "reviewed"}, headers=auth(),
    )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

def test_delete_removes_db_row_and_file(app_and_client):
    appmod, client = app_and_client
    body = _good_transcript()
    client.post("/transcripts", json=body)
    uid = body["upload_id"]

    storage_dir = Path(os.environ["BLOX_AI_STORAGE_DIR"])
    file_path = storage_dir / uid[:2] / uid[2:4] / f"{uid}.json"
    assert file_path.exists()

    r = client.delete(f"/admin/issues/{uid}", headers=auth())
    assert r.status_code == 204

    # Both file and row gone.
    assert not file_path.exists()
    r2 = client.get(f"/admin/issues/{uid}", headers=auth())
    assert r2.status_code == 404


def test_delete_404_for_missing(app_and_client):
    _, client = app_and_client
    r = client.delete(
        "/admin/issues/00000000-0000-4000-8000-000000000000",
        headers=auth(),
    )
    assert r.status_code == 404


def test_delete_refuses_path_traversal_in_storage_key(
    monkeypatch, tmp_path_factory,
):
    """Codex review fix: if a DB row's storage_key contains a traversal
    sequence (corrupted/malicious row), the DELETE handler must NOT
    unlink outside storage root. DB row is still deleted; filesystem
    is untouched."""
    storage_dir = tmp_path_factory.mktemp("trav_storage")
    db_dir = tmp_path_factory.mktemp("trav_db")
    sentinel = tmp_path_factory.mktemp("trav_sibling") / "do-not-delete.txt"
    sentinel.write_text("guard")
    monkeypatch.setenv("BLOX_AI_STORAGE_DIR", str(storage_dir))
    monkeypatch.setenv("BLOX_AI_DB_PATH", str(db_dir / "issues.sqlite"))
    monkeypatch.setenv("BLOX_AI_ADMIN_TOKEN", ADMIN_TOKEN)
    for mod in ("app", "admin", "issue_db",
                "anonymization_check", "storage"):
        sys.modules.pop(mod, None)
    import importlib
    import app as appmod
    importlib.reload(appmod)
    appmod._reset_rate_limit_for_tests()

    with TestClient(appmod.app) as client:
        # Inject a malicious DB row directly (simulates a corrupted row /
        # future-bug import). We bypass POST /transcripts to control the
        # storage_key value exactly.
        bad_uid = "11111111-1111-4111-8111-111111111111"
        traversal_key = "../trav_sibling/do-not-delete.txt"
        appmod._db_conn.execute(
            "INSERT INTO issues (id, source, status, created_at, "
            "updated_at, storage_key) "
            "VALUES (?, 'transcript', 'new', '2026-01-01T00:00:00Z', "
            "'2026-01-01T00:00:00Z', ?)",
            (bad_uid, traversal_key),
        )

        r = client.delete(f"/admin/issues/{bad_uid}", headers=auth())
        # DB row still goes (it's not the file's fault), but the sentinel
        # outside root MUST still exist.
        assert r.status_code == 204
        assert sentinel.exists(), (
            "path traversal defense failed: file outside storage root was deleted"
        )
        # And the issue is gone from the DB.
        r2 = client.get(f"/admin/issues/{bad_uid}", headers=auth())
        assert r2.status_code == 404


def test_detail_refuses_path_traversal_in_storage_key(
    monkeypatch, tmp_path_factory,
):
    """Sibling test for GET: a traversal storage_key must yield payload=None,
    not a leaked file from outside storage root."""
    storage_dir = tmp_path_factory.mktemp("trav2_storage")
    db_dir = tmp_path_factory.mktemp("trav2_db")
    secret = tmp_path_factory.mktemp("trav2_sibling") / "secret.txt"
    secret.write_text('{"top_secret": "do-not-leak"}')
    monkeypatch.setenv("BLOX_AI_STORAGE_DIR", str(storage_dir))
    monkeypatch.setenv("BLOX_AI_DB_PATH", str(db_dir / "issues.sqlite"))
    monkeypatch.setenv("BLOX_AI_ADMIN_TOKEN", ADMIN_TOKEN)
    for mod in ("app", "admin", "issue_db",
                "anonymization_check", "storage"):
        sys.modules.pop(mod, None)
    import importlib
    import app as appmod
    importlib.reload(appmod)
    appmod._reset_rate_limit_for_tests()

    with TestClient(appmod.app) as client:
        bad_uid = "22222222-2222-4222-8222-222222222222"
        traversal_key = "../trav2_sibling/secret.txt"
        appmod._db_conn.execute(
            "INSERT INTO issues (id, source, status, created_at, "
            "updated_at, storage_key) "
            "VALUES (?, 'transcript', 'new', '2026-01-01T00:00:00Z', "
            "'2026-01-01T00:00:00Z', ?)",
            (bad_uid, traversal_key),
        )
        r = client.get(f"/admin/issues/{bad_uid}", headers=auth())
        assert r.status_code == 200
        # Critical: the leaked secret must NOT appear in the response body.
        assert r.json()["payload"] is None
        assert "do-not-leak" not in r.text


# ---------------------------------------------------------------------------
# Background migration
# ---------------------------------------------------------------------------

def test_background_migration_picks_up_existing_files(
    monkeypatch, tmp_path_factory,
):
    """Pre-populate storage dir with files BEFORE the app starts; expect
    the background migration to insert one issue row per file."""
    storage_dir = tmp_path_factory.mktemp("preexisting_storage")
    db_dir = tmp_path_factory.mktemp("preexisting_db")
    # Lay down 3 pre-existing transcripts using the same path convention
    # LocalDirStorage uses.
    pre_ids = []
    for _ in range(3):
        uid = str(uuid.uuid4())
        pre_ids.append(uid)
        p = storage_dir / uid[:2] / uid[2:4] / f"{uid}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"upload_id": uid, "events": []}),
                     encoding="utf-8")

    monkeypatch.setenv("BLOX_AI_STORAGE_DIR", str(storage_dir))
    monkeypatch.setenv("BLOX_AI_DB_PATH", str(db_dir / "issues.sqlite"))
    monkeypatch.setenv("BLOX_AI_ADMIN_TOKEN", ADMIN_TOKEN)
    for mod in ("app", "admin", "issue_db", "anonymization_check", "storage"):
        sys.modules.pop(mod, None)
    import importlib
    import app as appmod
    importlib.reload(appmod)
    appmod._reset_rate_limit_for_tests()

    with TestClient(appmod.app) as client:
        # Migration is async — give it a beat to land. Few hundred ms is
        # plenty for 3 files; tighten if flaky.
        for _ in range(20):
            r = client.get("/admin/issues?status=new&limit=10", headers=auth())
            if r.json()["total"] == 3:
                break
            time.sleep(0.1)
        r = client.get("/admin/issues?status=new&limit=10", headers=auth())
        assert r.json()["total"] == 3
        ids_in_db = sorted(it["id"] for it in r.json()["items"])
        assert ids_in_db == sorted(pre_ids)
