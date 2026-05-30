"""Admin endpoints for the status-tracked inbox.

Auth: Bearer token via BLOX_AI_ADMIN_TOKEN env var. All `/admin/*`
endpoints require it; the intake POST endpoint stays unauth'd because
phones don't know it.

Threat model (operator-accepted): the data behind this auth is
already-anonymized transcripts (PII-scanned twice — once on the
device, once on intake). The token is for casual operator access
control, not high-value secret protection. Bearer in .env is fine.

Defense in depth:
- HTTPS-only (nginx + Let's Encrypt; install.sh)
- secrets.compare_digest for timing-safe compare
- Generic 401 (no token-vs-no-token disambiguation)
- Per-IP rate limit on auth failures (slows brute-force)
- Cache-Control: no-store on every response
- 1 MB body cap on POST /admin/issues
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

import issue_db


logger = logging.getLogger("blox-ai-intake.admin")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ADMIN_TOKEN_ENV = "BLOX_AI_ADMIN_TOKEN"
ADMIN_BODY_CAP_BYTES = 1024 * 1024  # 1 MB per Gemini's recommendation
AUTH_FAIL_WINDOW_SEC = 60
AUTH_FAIL_MAX_PER_WINDOW = 10  # per-IP

NO_STORE = {"Cache-Control": "no-store"}


# ---------------------------------------------------------------------------
# Auth-failure rate limit (separate from intake's; this gate slows brute force)
# ---------------------------------------------------------------------------

_auth_fail_buckets: dict[str, deque] = defaultdict(deque)


def _client_ip_for_auth(request: Request) -> str:
    # Reuse intake's XFF-trust posture so behaviour is consistent behind nginx.
    if os.environ.get("BLOX_AI_TRUST_XFF") == "1":
        xff = request.headers.get("x-forwarded-for")
        if xff:
            return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _auth_failure_rate_check(ip: str, now: float) -> bool:
    """True iff the caller is within budget. Records a fresh failure when
    called. False => return 429 instead of 401."""
    bucket = _auth_fail_buckets[ip]
    cutoff = now - AUTH_FAIL_WINDOW_SEC
    while bucket and bucket[0] < cutoff:
        bucket.popleft()
    if len(bucket) >= AUTH_FAIL_MAX_PER_WINDOW:
        return False
    bucket.append(now)
    return True


def _reset_auth_state_for_tests() -> None:
    _auth_fail_buckets.clear()


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

def _require_bearer(request: Request) -> None:
    expected = os.environ.get(ADMIN_TOKEN_ENV, "").strip()
    if not expected:
        # Admin disabled (no token configured). Generic 503 — don't reveal
        # the cause.
        raise HTTPException(status_code=503,
                            detail={"error": "admin_unavailable"})

    provided = ""
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        provided = header[len("bearer "):].strip()

    ok = bool(provided) and secrets.compare_digest(
        provided.encode("utf-8"), expected.encode("utf-8")
    )
    if not ok:
        ip = _client_ip_for_auth(request)
        if not _auth_failure_rate_check(ip, time.time()):
            raise HTTPException(status_code=429,
                                detail={"error": "auth_rate_limit"})
        # Generic 401 — DO NOT disambiguate missing vs wrong token.
        raise HTTPException(status_code=401,
                            detail={"error": "auth_required"})


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AdminIssueCreate(BaseModel):
    """Envelope for server-filed issues. The `payload` field is free-form
    (any JSON object) so internal callers can stash whatever they need.
    Total request body is hard-capped at ADMIN_BODY_CAP_BYTES by middleware
    before this model is even constructed."""
    model_config = {"extra": "forbid"}
    summary: str = Field(min_length=1, max_length=1024)
    source: str = Field(pattern="^admin$")  # mechanically restricts to 'admin'
    payload: dict = Field(default_factory=dict)


class StatusUpdate(BaseModel):
    model_config = {"extra": "forbid"}
    status: str = Field(pattern="^(reviewed|fixed|dismissed)$")
    # Note: 'new' is intentionally not in the regex. You can't move a
    # finished issue back to 'new' via the API — that would erase work
    # signal. Use DELETE if you want it gone, or file a new one.


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def make_admin_router(get_db, get_storage, ui_html_path: Path) -> APIRouter:
    """Build the admin router with dependency-injected accessors. Keeps
    admin.py free of module-globals so tests can swap in their own DB.

    Parameters:
      get_db()        -> sqlite3.Connection
      get_storage()   -> Storage (LocalDirStorage)
      ui_html_path    -> path to admin_ui.html
    """
    router = APIRouter(prefix="/admin")

    @router.get("", include_in_schema=False)
    def _admin_root_no_slash(request: Request) -> Response:
        # The HTML page itself is PUBLIC by design — it's an empty
        # vanilla-JS shell that PROMPTS the user for the bearer token
        # and stashes it in localStorage. If we gated this route too,
        # the browser would get the raw 401 JSON body and the user
        # would never see a token input field (chicken-and-egg bug
        # observed 2026-05-26). All JSON endpoints below remain
        # auth'd; only the shell HTML is public.
        #
        # FastAPI's APIRouter doesn't auto-redirect /admin -> /admin/
        # when we set prefix='/admin' + route='/'. Provide both
        # explicitly so nginx + browsers see a stable URL.
        return _serve_ui(ui_html_path)

    @router.get("/", include_in_schema=False)
    def admin_ui(_request: Request) -> Response:
        # See _admin_root_no_slash above — HTML shell is public; only
        # the data-bearing endpoints require the bearer token.
        return _serve_ui(ui_html_path)

    @router.get("/issues")
    def list_issues_endpoint(
        request: Request,
        status: list[str] = Query(default=None),
        since: str = Query(default=None),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ) -> JSONResponse:
        _require_bearer(request)
        statuses = status or ["new"]
        for s in statuses:
            if s not in issue_db.VALID_STATUSES:
                raise HTTPException(status_code=400,
                                    detail={"error": "invalid_status"})
        if since is not None:
            # Loose but non-bogus check: must parse as ISO-8601 date or datetime.
            try:
                _parse_iso(since)
            except ValueError:
                raise HTTPException(status_code=400,
                                    detail={"error": "invalid_since"})
        items, total = issue_db.list_issues(
            get_db(), statuses=statuses, since=since,
            limit=limit, offset=offset,
        )
        return JSONResponse(
            {"items": items, "total": total, "limit": limit, "offset": offset},
            headers=NO_STORE,
        )

    @router.get("/issues/{issue_id}")
    def get_issue_endpoint(request: Request, issue_id: str) -> JSONResponse:
        _require_bearer(request)
        issue = issue_db.get_issue(get_db(), issue_id)
        if issue is None:
            raise HTTPException(status_code=404,
                                detail={"error": "not_found"})
        # Inline the payload so the UI / operator-script can consume the
        # whole thing in one round trip. Key off the populated column, not
        # the source label: file-backed sources (transcript, diagnostics)
        # carry a storage_key; admin-filed issues carry inline payload_json.
        payload = None
        if issue["storage_key"]:
            payload = _read_transcript(get_storage(), issue["storage_key"])
        elif issue["payload_json"]:
            try:
                payload = json.loads(issue["payload_json"])
            except json.JSONDecodeError:
                payload = None
        return JSONResponse(
            {
                "id": issue["id"],
                "source": issue["source"],
                "status": issue["status"],
                "created_at": issue["created_at"],
                "updated_at": issue["updated_at"],
                "summary": issue["summary"],
                "payload": payload,
            },
            headers=NO_STORE,
        )

    @router.post("/issues")
    async def create_admin_issue_endpoint(request: Request) -> JSONResponse:
        _require_bearer(request)
        # Body-size cap BEFORE buffering. Two layers:
        #  1. Cheap Content-Length precheck (caller-stated size). Caller
        #     can lie, but a correct Content-Length costs nothing to check.
        #  2. After-read length check. Catches the lie-about-Content-Length
        #     case AND missing-Content-Length case (e.g., chunked transfer).
        # nginx upstream also caps at 1 MB but we don't depend on that.
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > ADMIN_BODY_CAP_BYTES:
                    raise HTTPException(status_code=413,
                                        detail={"error": "body_too_large"})
            except ValueError:
                raise HTTPException(status_code=400,
                                    detail={"error": "body_invalid"})
        raw = await request.body()
        if len(raw) > ADMIN_BODY_CAP_BYTES:
            raise HTTPException(status_code=413,
                                detail={"error": "body_too_large"})
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400,
                                detail={"error": "body_invalid"})
        try:
            model = AdminIssueCreate.model_validate(data)
        except Exception:
            # Generic 400 — match intake's no-fingerprint posture.
            raise HTTPException(status_code=400,
                                detail={"error": "body_invalid"})
        issue_id, created_at = issue_db.create_admin_issue(
            get_db(),
            summary=model.summary,
            payload_json=json.dumps(model.payload, separators=(",", ":")),
        )
        return JSONResponse(
            {"id": issue_id, "created_at": created_at},
            status_code=201,
            headers=NO_STORE,
        )

    @router.patch("/issues/{issue_id}")
    def patch_status_endpoint(
        request: Request, issue_id: str, payload: StatusUpdate,
    ) -> JSONResponse:
        _require_bearer(request)
        try:
            result = issue_db.update_status(
                get_db(), issue_id, payload.status
            )
        except ValueError:
            raise HTTPException(status_code=400,
                                detail={"error": "invalid_status"})
        if result is None:
            raise HTTPException(status_code=404,
                                detail={"error": "not_found"})
        return JSONResponse(result, headers=NO_STORE)

    @router.delete("/issues/{issue_id}")
    def delete_issue_endpoint(request: Request, issue_id: str) -> Response:
        _require_bearer(request)
        # Look up first so we know if there's a backing transcript file to
        # remove from storage. Hard-delete is intentional — DB + file go
        # together.
        issue = issue_db.get_issue(get_db(), issue_id)
        if issue is None:
            raise HTTPException(status_code=404,
                                detail={"error": "not_found"})
        if issue["storage_key"]:
            storage = get_storage()
            fpath = _safe_storage_path(storage, issue["storage_key"])
            if fpath is not None:
                try:
                    if fpath.exists():
                        fpath.unlink()
                except OSError as e:
                    # Storage delete failure shouldn't strand the DB row;
                    # log and continue with the DB delete.
                    logger.warning("delete: storage unlink failed id=%s: %s",
                                    issue_id, e)
            else:
                # storage_key would have escaped root — log + don't touch
                # the filesystem. DB row still gets deleted below.
                logger.warning("delete: refusing to unlink (traversal) id=%s",
                                issue_id)
        issue_db.delete_issue(get_db(), issue_id)
        return Response(status_code=204, headers=NO_STORE)

    @router.get("/issues/{issue_id}/history")
    def get_history_endpoint(request: Request, issue_id: str) -> JSONResponse:
        _require_bearer(request)
        if issue_db.get_issue(get_db(), issue_id) is None:
            raise HTTPException(status_code=404,
                                detail={"error": "not_found"})
        return JSONResponse(
            {"history": issue_db.get_history(get_db(), issue_id)},
            headers=NO_STORE,
        )

    return router


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serve_ui(path: Path) -> HTMLResponse:
    if not path.is_file():
        raise HTTPException(status_code=500,
                            detail={"error": "ui_missing"})
    return HTMLResponse(
        path.read_text(encoding="utf-8"),
        headers={**NO_STORE, "X-Content-Type-Options": "nosniff"},
    )


def _safe_storage_path(storage, storage_key: str) -> Optional[Path]:
    """Resolve storage.root/storage_key and verify the result stays under
    storage.root. Defense against a corrupted/malicious DB row whose
    storage_key contains '..' or absolute path segments.

    Returns the safe path on success, None if traversal was attempted
    (caller treats as not-found, never reads/deletes outside root).
    """
    try:
        root = Path(storage.root).resolve(strict=False)
        target = (root / storage_key).resolve(strict=False)
    except (OSError, AttributeError):
        return None
    # Path.is_relative_to (3.9+) is more readable than relative_to-with-try.
    try:
        target.relative_to(root)
    except ValueError:
        logger.warning("storage path traversal blocked: key=%r", storage_key)
        return None
    return target


def _read_transcript(storage, storage_key: str) -> Optional[dict]:
    """Read a transcript file via the Storage adapter's root path. Returns
    the parsed JSON or None if the file is missing / malformed / would
    have escaped root."""
    path = _safe_storage_path(storage, storage_key)
    if path is None:
        return None
    try:
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _parse_iso(s: str):
    """Accept '2026-05-26' or '2026-05-26T12:00:00Z' (Python's
    fromisoformat doesn't accept the trailing Z natively until 3.11)."""
    import datetime as _dt
    return _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
