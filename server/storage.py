"""Phase 20 — pluggable object-storage adapter.

Two backends:
- LocalDirStorage: writes transcripts under a configurable host directory.
  Used in dev + tests.
- S3Storage: writes to S3-compatible object storage. Used in prod. NOT
  implemented in this phase scaffold — the IAM policy + bucket name are
  separate infra repo decisions.

Idempotency: write_transcript() takes an upload_id and is a no-op if a
prior write with the same upload_id already exists. Both backends MUST
provide this guarantee so client-side retries don't duplicate writes.

What this module does NOT do:
- Verify the transcript schema (caller's job).
- Run the anonymization PII scanner (caller's job).
- Persist client IP, request headers, or anything else not in the
  caller-supplied payload.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class StorageError(RuntimeError):
    pass


@dataclass(frozen=True)
class WriteResult:
    upload_id: str
    written: bool  # False when the prior write was a hit (idempotency)
    storage_key: str


class Storage(Protocol):
    def write_transcript(self, upload_id: str, payload: dict) -> WriteResult: ...


class LocalDirStorage:
    """Writes one JSON file per upload_id under `root`.

    Filename convention: `<upload_id>.json`. Subdirectories by upload_id
    prefix to avoid bottoming out at huge flat directories on long-lived
    deployments. With UUIDv4 inputs this gives roughly even fan-out:
        <root>/<first-2>/<next-2>/<upload_id>.json
    """

    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _key_for(self, upload_id: str) -> Path:
        # upload_id is server-validated as a UUID; this routine assumes
        # it's safe for filesystem use. If you call into here from
        # elsewhere, validate first.
        if not _looks_like_uuid(upload_id):
            raise StorageError("upload_id must look like a UUID")
        return self.root / upload_id[:2] / upload_id[2:4] / f"{upload_id}.json"

    def write_transcript(self, upload_id: str, payload: dict) -> WriteResult:
        path = self._key_for(upload_id)
        rel = str(path.relative_to(self.root)).replace(os.sep, "/")
        if path.exists():
            return WriteResult(upload_id=upload_id, written=False, storage_key=rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic-ish: write to a sibling .partial then rename. Two concurrent
        # writes of the same upload_id will both land cleanly (the second
        # rename either succeeds-and-overwrites or loses race; either way
        # the contents are byte-identical because we validated the same
        # payload).
        partial = path.with_suffix(".partial")
        with open(partial, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"), sort_keys=True)
        os.replace(partial, path)
        return WriteResult(upload_id=upload_id, written=True, storage_key=rel)


def _looks_like_uuid(s: str) -> bool:
    if not isinstance(s, str) or len(s) != 36:
        return False
    if s[8] != "-" or s[13] != "-" or s[18] != "-" or s[23] != "-":
        return False
    hex_chars = set("0123456789abcdef")
    for ch in s:
        if ch == "-":
            continue
        if ch not in hex_chars:
            return False
    return True
