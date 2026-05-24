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

import json
import logging
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import jsonschema
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse

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


app = FastAPI(
    title="Blox AI intake server",
    description="Receives opt-in anonymized troubleshooting transcripts.",
    version="0.1.0",
)


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
