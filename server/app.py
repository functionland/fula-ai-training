"""Phase 20 — transcript intake server (FastAPI).

POST /transcripts:
  1. Validate request body against anonymized_transcript.schema.json.
  2. Run the defense-in-depth PII scanner.
  3. Enforce per-IP rate limit.
  4. Persist via the Storage adapter.
  5. Return {} (or 204).

The server NEVER:
  - Persists the client IP alongside the transcript.
  - Echoes any field-level validation detail in error responses
    (prevents schema fingerprinting).
  - Reads or lists prior uploads (bucket is write-only from the server
    IAM identity in prod).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import jsonschema
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse

import issue_db
from admin import make_admin_router
from anonymization_check import find_pii
from storage import LocalDirStorage, Storage, StorageError


logger = logging.getLogger("blox-ai-intake")
logging.basicConfig(level=os.environ.get("BLOX_AI_LOG_LEVEL", "INFO"))


_SCHEMA_PATH = Path(__file__).parent / "anonymized_transcript.schema.json"
_SCHEMA: dict = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
_VALIDATOR = jsonschema.Draft202012Validator(_SCHEMA)

# Rate limiter: per-IP sliding window. Defaults are conservative and
# can be overridden by env var for canary tuning.
RATE_LIMIT_WINDOW_SEC = int(os.environ.get("BLOX_AI_RATE_WINDOW_SEC", "86400"))
RATE_LIMIT_PER_IP_PER_WINDOW = int(os.environ.get("BLOX_AI_RATE_PER_IP", "50"))
GLOBAL_RATE_LIMIT_PER_WINDOW = int(os.environ.get("BLOX_AI_GLOBAL_RATE", "10000"))


_ip_buckets: dict[str, deque] = defaultdict(deque)
_global_bucket: deque = deque()


def _make_storage() -> Storage:
    """Pick a storage adapter based on env. Defaults to a local dir so
    devs can run the server without any S3 wiring."""
    storage_dir = os.environ.get("BLOX_AI_STORAGE_DIR")
    if storage_dir:
        return LocalDirStorage(storage_dir)
    # Production would return S3Storage(...) here; for the Phase 20
    # scaffold we default to a tmp dir so the server still boots and
    # tests can exercise the full path.
    fallback = Path(os.environ.get("TMPDIR", "/tmp")) / "blox-ai-intake-default"
    logger.warning(
        "BLOX_AI_STORAGE_DIR not set; using %s as fallback. Set env var in prod.",
        fallback,
    )
    return LocalDirStorage(str(fallback))


_storage: Storage = _make_storage()


# ---------------------------------------------------------------------------
# Issue DB (status-tracked inbox sidecar) — lifespan-managed
# ---------------------------------------------------------------------------

def _db_path() -> Path:
    """Resolve the issue DB path. Defaults to a sibling of the storage dir
    so backups + log rotation paths line up naturally."""
    explicit = os.environ.get("BLOX_AI_DB_PATH")
    if explicit:
        return Path(explicit)
    # Sibling: /var/lib/blox-ai-intake/transcripts/ -> /var/lib/blox-ai-intake/db/issues.sqlite
    storage_root = Path(getattr(_storage, "root", "/tmp/blox-ai-intake"))
    return storage_root.parent / "db" / "issues.sqlite"


_db_conn = None  # opened in lifespan startup


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Open SQLite WAL connection + spawn background migration that walks
    pre-existing transcripts and inserts status='new' rows. The migration
    yields to the event loop so request handling is responsive even on
    large existing trees."""
    global _db_conn
    path = _db_path()
    logger.info("opening issue DB at %s", path)
    _db_conn = issue_db.open_db(path)
    issue_db.init_schema(_db_conn)
    # Spawn migration as a background task so startup doesn't block.
    storage_root = Path(getattr(_storage, "root", ""))
    if storage_root.exists():
        asyncio.create_task(_run_migration(storage_root))
    yield
    logger.info("closing issue DB")
    if _db_conn is not None:
        _db_conn.close()
        _db_conn = None


async def _run_migration(storage_root: Path) -> None:
    try:
        n = await issue_db.background_migration(_db_conn, storage_root)
        logger.info("migration: scanned %d existing transcripts", n)
    except Exception as e:  # noqa: BLE001
        logger.warning("migration failed: %s", e)


def _get_db():
    """Accessor passed to the admin router; raises if startup hasn't run
    (e.g., tests that bypass lifespan)."""
    if _db_conn is None:
        raise RuntimeError("issue DB not initialized; ensure lifespan ran")
    return _db_conn


def _get_storage():
    return _storage


app = FastAPI(
    title="Blox AI intake server",
    description="Receives opt-in anonymized troubleshooting transcripts.",
    version="0.1.0",
    lifespan=_lifespan,
)

# Mount the admin router (status-tracked inbox + web UI).
_UI_HTML = Path(__file__).parent / "admin_ui.html"
app.include_router(make_admin_router(_get_db, _get_storage, _UI_HTML))


# Codex review fix: HTTPException responses (401/422/413/etc) don't pass
# through our per-endpoint NO_STORE header dict, so error bodies could end
# up in browser/proxy caches. Middleware enforces no-store on EVERY
# response under /admin/*, regardless of status code or source.
#
# Copilot review fix: middleware must also cover the unhandled-exception
# path. If a handler raises before producing a response, the default 500
# response would leak without no-store. We catch the exception, log it,
# and return a manual 500 with the header attached. Non-admin paths
# re-raise so default error handling applies as usual.
@app.middleware("http")
async def _admin_no_store(request: Request, call_next):
    path = request.url.path or ""
    is_admin = path == "/admin" or path.startswith("/admin/")
    try:
        response = await call_next(request)
    except Exception:
        if is_admin:
            logger.exception("admin endpoint raised; returning 500 with no-store")
            return JSONResponse(
                {"detail": "internal_error"},
                status_code=500,
                headers={"Cache-Control": "no-store"},
            )
        raise
    if is_admin:
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/transcripts")
async def post_transcript(request: Request) -> Response:
    client_ip = _client_ip(request)
    now = time.time()

    # 1. Rate-limit BEFORE parsing the body. Cheap reject for abuse.
    if not _check_rate_limit(client_ip, now):
        # 429; don't leak which limit hit.
        logger.info("rate_limit ip=%s", _hash_ip(client_ip))
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"error": "rate_limit"},
        )

    # 2. Parse body.
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "body_invalid"},
        )
    if not isinstance(payload, dict):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "body_invalid"},
        )

    # 3. Schema validation.
    try:
        _VALIDATOR.validate(payload)
    except jsonschema.ValidationError:
        # Generic error — do NOT leak field-level details.
        logger.info("schema_violation ip=%s", _hash_ip(client_ip))
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "body_invalid"},
        )

    # 4. Defense-in-depth anonymization scan.
    hit = find_pii(payload)
    if hit is not None:
        logger.warning("anonymization_check_failed scanner=%s ip=%s anonymizer_version=%s",
                       hit, _hash_ip(client_ip),
                       payload.get("consent", {}).get("anonymizer_version", "?"))
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "anonymization_check_failed"},
        )

    # 5. Persist (idempotent on upload_id).
    upload_id = payload["upload_id"]
    try:
        result = _storage.write_transcript(upload_id, payload)
    except StorageError:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "storage_unavailable"},
        )

    # 6. Track in the issue DB so it shows up in the admin inbox.
    # Failure here MUST NOT 5xx the upload — the transcript is already on
    # disk and the background migration will pick it up next restart.
    if _db_conn is not None:
        try:
            issue_db.upsert_transcript_issue(
                _db_conn, upload_id, storage_key=result.storage_key,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("issue_db upsert failed (non-fatal) upload_id=%s: %s",
                            upload_id, e)

    logger.info("persisted upload_id=%s written=%s key=%s",
                upload_id, result.written, result.storage_key)
    # Return empty body — no echoed transcript, no derived fingerprint.
    return JSONResponse(status_code=status.HTTP_200_OK, content={})


# ---------------------------------------------------------------------------
# Rate-limit + IP-handling helpers
# ---------------------------------------------------------------------------

def _client_ip(request: Request) -> str:
    """Best-effort client IP for rate-limit accounting.

    SECURITY NOTE: X-Forwarded-For is trusted ONLY when the env var
    BLOX_AI_TRUST_XFF is set. If the server is deployed directly
    exposed (no LB in front), an attacker could spoof XFF and bypass
    the per-IP rate limit. Default behaviour: ignore XFF, use the raw
    TCP peer. Deployers behind a TLS-terminating LB should set
    BLOX_AI_TRUST_XFF=1 explicitly.
    """
    if os.environ.get("BLOX_AI_TRUST_XFF") == "1":
        xff = request.headers.get("x-forwarded-for")
        if xff:
            return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _hash_ip(ip: str) -> str:
    """Stable short hash for log correlation. We never log the raw IP."""
    import hashlib
    return hashlib.sha256(ip.encode("utf-8")).hexdigest()[:12]


_sweep_counter = 0
_SWEEP_INTERVAL = 1000  # every Nth request, walk + drop empty per-IP deques


def _check_rate_limit(ip: str, now: float) -> bool:
    """Sliding window per-IP + global. Returns True iff the call is allowed
    and records it. Returns False to indicate rejection (NOT recorded)."""
    global _sweep_counter
    cutoff = now - RATE_LIMIT_WINDOW_SEC
    # Trim
    bucket = _ip_buckets[ip]
    while bucket and bucket[0] < cutoff:
        bucket.popleft()
    while _global_bucket and _global_bucket[0] < cutoff:
        _global_bucket.popleft()

    # Periodic sweep so the per-IP dict doesn't grow unbounded over months
    # of unique source IPs. Cheap amortised cost.
    _sweep_counter += 1
    if _sweep_counter >= _SWEEP_INTERVAL:
        _sweep_counter = 0
        empty = [k for k, dq in _ip_buckets.items() if not dq]
        for k in empty:
            del _ip_buckets[k]

    if len(bucket) >= RATE_LIMIT_PER_IP_PER_WINDOW:
        return False
    if len(_global_bucket) >= GLOBAL_RATE_LIMIT_PER_WINDOW:
        return False
    bucket.append(now)
    _global_bucket.append(now)
    return True


def _reset_rate_limit_for_tests() -> None:
    """Test-only helper. Production code should never call this."""
    global _sweep_counter
    _ip_buckets.clear()
    _global_bucket.clear()
    _sweep_counter = 0
