"""SQLite issue tracker for the admin inbox.

WAL mode enables concurrent reads while a write is in progress, so
POST /transcripts (phones) and PATCH /admin/issues (operator) don't
collide as `database is locked`. Critical for production load.

Schema:
  issues          (id, source, status, created_at, updated_at,
                   storage_key, payload_json, summary)
  status_history  (issue_id, from_status, to_status, changed_at)

Sources:   'transcript' | 'admin'
Statuses:  'new' | 'reviewed' | 'fixed' | 'dismissed'

Transcripts are stored as JSON files via LocalDirStorage; the DB
holds only metadata + status. Admin-filed issues store their payload
inline as `payload_json` because they don't have a backing file.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


logger = logging.getLogger("blox-ai-intake.issue_db")


VALID_STATUSES = ("new", "reviewed", "fixed", "dismissed")
VALID_SOURCES = ("transcript", "admin", "diagnostics")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _looks_like_uuid(s: str) -> bool:
    """Lenient UUID validation: accepts any valid UUID string (any case)
    via stdlib parsing. We don't normalize to lowercase here because the
    upstream schema already enforces lowercase canonical form on
    transcript upload_ids; this guard exists for defense in depth on
    the migration path."""
    if not isinstance(s, str):
        return False
    try:
        uuid.UUID(s)
        return True
    except ValueError:
        return False


def open_db(path: Path) -> sqlite3.Connection:
    """Open a connection with WAL + sensible defaults.

    check_same_thread=False is safe here because SQLite (with WAL) supports
    one writer + many concurrent readers, and uvicorn's worker model means
    requests can hop threads. busy_timeout retries internal locking briefly
    before raising, smoothing over write contention without us having to
    code retries in every endpoint.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS issues (
          id           TEXT PRIMARY KEY,
          source       TEXT NOT NULL,
          status       TEXT NOT NULL DEFAULT 'new',
          created_at   TEXT NOT NULL,
          updated_at   TEXT NOT NULL,
          storage_key  TEXT,
          payload_json TEXT,
          summary      TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_status_created
          ON issues(status, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_created
          ON issues(created_at DESC);
        CREATE TABLE IF NOT EXISTS status_history (
          issue_id    TEXT NOT NULL,
          from_status TEXT,
          to_status   TEXT NOT NULL,
          changed_at  TEXT NOT NULL,
          FOREIGN KEY (issue_id) REFERENCES issues(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_history_issue
          ON status_history(issue_id, changed_at DESC);
    """)


def _begin_immediate(conn: sqlite3.Connection) -> None:
    """Acquire a SQLite write lock. With WAL, this lets one writer proceed
    while readers continue unblocked; without it, two compound writes on
    the same Python connection in autocommit mode can interleave
    (SELECT-A, SELECT-B, UPDATE-A, UPDATE-B both racing on stale state).
    busy_timeout (5s) handles wait-for-lock retries internally."""
    conn.execute("BEGIN IMMEDIATE")


def _commit(conn: sqlite3.Connection) -> None:
    conn.execute("COMMIT")


def _rollback(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("ROLLBACK")
    except sqlite3.OperationalError:
        # No active transaction (e.g., BEGIN failed); nothing to roll back.
        pass


def _upsert_transcript_issue_inner(
    conn: sqlite3.Connection,
    upload_id: str,
    storage_key: str,
    summary: Optional[str],
    now: str,
    source: str = "transcript",
) -> bool:
    """Inner upsert WITHOUT transaction control. Used by the public
    functions (which wrap in BEGIN/COMMIT) and by background_migration
    (which batches many of these inside one transaction).

    `source` defaults to 'transcript' so background_migration keeps
    labelling files it scans as transcripts. Diagnostics intake passes
    source='diagnostics'; the ON CONFLICT(id) DO NOTHING below protects
    that label on restart (the migration's later insert is a no-op for an
    already-present row)."""
    cur = conn.execute(
        "INSERT INTO issues (id, source, status, created_at, updated_at, "
        "storage_key, summary) "
        "VALUES (?, ?, 'new', ?, ?, ?, ?) "
        "ON CONFLICT(id) DO NOTHING",
        (upload_id, source, now, now, storage_key, summary),
    )
    inserted = cur.rowcount > 0
    if inserted:
        conn.execute(
            "INSERT INTO status_history (issue_id, from_status, to_status, "
            "changed_at) VALUES (?, NULL, 'new', ?)",
            (upload_id, now),
        )
    return inserted


def upsert_transcript_issue(
    conn: sqlite3.Connection,
    upload_id: str,
    storage_key: str,
    summary: Optional[str] = None,
    created_at: Optional[str] = None,
) -> bool:
    """Insert a new transcript-issue if it doesn't already exist.

    Idempotent: re-inserting the same upload_id is a no-op. Returns True
    if a row was actually inserted (status_history also gets a NULL -> 'new'
    entry on insert). Atomic via BEGIN IMMEDIATE.
    """
    if not _looks_like_uuid(upload_id):
        raise ValueError("upload_id must look like a UUID")
    now = created_at or _utcnow_iso()
    _begin_immediate(conn)
    try:
        inserted = _upsert_transcript_issue_inner(
            conn, upload_id, storage_key, summary, now,
        )
        _commit(conn)
        return inserted
    except Exception:
        _rollback(conn)
        raise


def upsert_diagnostics_issue(
    conn: sqlite3.Connection,
    upload_id: str,
    storage_key: str,
    summary: Optional[str] = None,
    created_at: Optional[str] = None,
) -> bool:
    """Insert a new diagnostics-issue if it doesn't already exist.

    Same idempotent file-backed shape as upsert_transcript_issue, but the
    row is labelled source='diagnostics' so the admin inbox can tell a
    user-shared diagnostics bundle apart from an AI training transcript.
    Returns True if a row was actually inserted. Atomic via BEGIN IMMEDIATE.
    """
    if not _looks_like_uuid(upload_id):
        raise ValueError("upload_id must look like a UUID")
    now = created_at or _utcnow_iso()
    _begin_immediate(conn)
    try:
        inserted = _upsert_transcript_issue_inner(
            conn, upload_id, storage_key, summary, now, source="diagnostics",
        )
        _commit(conn)
        return inserted
    except Exception:
        _rollback(conn)
        raise


def create_admin_issue(
    conn: sqlite3.Connection,
    summary: str,
    payload_json: str,
) -> tuple[str, str]:
    """Insert a server-filed issue. Returns (issue_id, created_at).

    Atomic via BEGIN IMMEDIATE so the issues row and its first history
    entry land together (or not at all).
    """
    issue_id = str(uuid.uuid4())
    now = _utcnow_iso()
    _begin_immediate(conn)
    try:
        conn.execute(
            "INSERT INTO issues (id, source, status, created_at, updated_at, "
            "payload_json, summary) "
            "VALUES (?, 'admin', 'new', ?, ?, ?, ?)",
            (issue_id, now, now, payload_json, summary),
        )
        conn.execute(
            "INSERT INTO status_history (issue_id, from_status, to_status, "
            "changed_at) VALUES (?, NULL, 'new', ?)",
            (issue_id, now),
        )
        _commit(conn)
        return issue_id, now
    except Exception:
        _rollback(conn)
        raise


def list_issues(
    conn: sqlite3.Connection,
    statuses: Optional[Iterable[str]] = None,
    since: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """Filtered list. Returns (items_for_page, total_matching)."""
    where_clauses: list[str] = []
    args: list = []
    if statuses:
        statuses = list(statuses)
        placeholders = ",".join("?" * len(statuses))
        where_clauses.append(f"status IN ({placeholders})")
        args.extend(statuses)
    if since:
        where_clauses.append("created_at >= ?")
        args.append(since)
    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    total = conn.execute(
        f"SELECT COUNT(*) FROM issues {where}", args
    ).fetchone()[0]

    rows = conn.execute(
        f"SELECT id, source, status, created_at, updated_at, summary "
        f"FROM issues {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
        args + [limit, offset],
    ).fetchall()
    items = [
        {
            "id": r[0],
            "source": r[1],
            "status": r[2],
            "created_at": r[3],
            "updated_at": r[4],
            "summary": r[5],
        }
        for r in rows
    ]
    return items, total


def get_issue(conn: sqlite3.Connection, issue_id: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT id, source, status, created_at, updated_at, storage_key, "
        "payload_json, summary FROM issues WHERE id = ?",
        (issue_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "id": row[0],
        "source": row[1],
        "status": row[2],
        "created_at": row[3],
        "updated_at": row[4],
        "storage_key": row[5],
        "payload_json": row[6],
        "summary": row[7],
    }


def update_status(
    conn: sqlite3.Connection,
    issue_id: str,
    new_status: str,
) -> Optional[dict]:
    """Compare-update + history. Returns None if issue not found, otherwise
    {status, updated_at, changed} where changed=False means status was
    already at new_status (no history row written, response still 200).

    Optimization (per copilot review): cheap not-found / same-status check
    OUTSIDE the write transaction; only escalate to BEGIN IMMEDIATE when
    we're actually going to write. Avoids unnecessary lock contention
    when an operator double-clicks the same status button.

    Correctness: the post-BEGIN re-check makes this race-safe. If a
    competing PATCH lands between our outside-SELECT and our BEGIN,
    we'll re-read the now-stale state inside the transaction and either
    bail (same-status) or write history with the correct from_status.
    """
    if new_status not in VALID_STATUSES:
        raise ValueError("invalid status")
    # Cheap pre-check — short-circuit common no-op paths without holding
    # a write lock.
    pre = conn.execute(
        "SELECT status FROM issues WHERE id = ?", (issue_id,)
    ).fetchone()
    if pre is None:
        return None
    if pre[0] == new_status:
        return {"status": new_status, "updated_at": _utcnow_iso(),
                "changed": False}

    # Real change — acquire write lock + re-check + write.
    now = _utcnow_iso()
    _begin_immediate(conn)
    try:
        row = conn.execute(
            "SELECT status FROM issues WHERE id = ?", (issue_id,)
        ).fetchone()
        if row is None:
            # Race: issue was deleted between our pre-check and the
            # BEGIN IMMEDIATE. Release lock + report not-found.
            _commit(conn)
            return None
        from_status = row[0]
        if from_status == new_status:
            # Race: another PATCH already moved it to new_status.
            _commit(conn)
            return {"status": new_status, "updated_at": now,
                    "changed": False}
        conn.execute(
            "UPDATE issues SET status = ?, updated_at = ? WHERE id = ?",
            (new_status, now, issue_id),
        )
        conn.execute(
            "INSERT INTO status_history (issue_id, from_status, to_status, "
            "changed_at) VALUES (?, ?, ?, ?)",
            (issue_id, from_status, new_status, now),
        )
        _commit(conn)
        return {"status": new_status, "updated_at": now, "changed": True}
    except Exception:
        _rollback(conn)
        raise


def get_history(conn: sqlite3.Connection, issue_id: str) -> list[dict]:
    # Tiebreak by rowid (monotonic per-row) for the common case where two
    # inserts land in the same second; _utcnow_iso() is second-resolution.
    rows = conn.execute(
        "SELECT from_status, to_status, changed_at FROM status_history "
        "WHERE issue_id = ? ORDER BY changed_at DESC, rowid DESC",
        (issue_id,),
    ).fetchall()
    return [
        {"from_status": r[0], "to_status": r[1], "changed_at": r[2]}
        for r in rows
    ]


def delete_issue(conn: sqlite3.Connection, issue_id: str) -> bool:
    """Hard delete the DB row + history (FK cascade). Caller is responsible
    for cleaning up the on-disk transcript file via the Storage adapter."""
    cur = conn.execute("DELETE FROM issues WHERE id = ?", (issue_id,))
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Migration: walk an existing LocalDirStorage tree on first DB init
# ---------------------------------------------------------------------------

async def background_migration(
    conn: sqlite3.Connection,
    storage_root: Path,
    batch_size: int = 500,
) -> int:
    """Walk `storage_root` for *.json files and INSERT-OR-IGNORE one issue
    row per file with status='new'. Runs as an asyncio background task so
    the FastAPI startup isn't blocked while a 100K-file tree is scanned.

    Uses chunked transactions of `batch_size` rows (per codex's review):
    one BEGIN IMMEDIATE per batch instead of one per row. With WAL,
    this dramatically reduces fsync pressure on big migrations while
    keeping each batch atomic and bounded.

    Yields back to the event loop between batches so the admin endpoints
    stay responsive during migration. Returns the number of rows scanned
    (not the number inserted — some may already exist).
    """
    scanned = 0
    in_batch = 0
    if not storage_root.exists():
        logger.info("migration: storage root missing, nothing to scan: %s",
                    storage_root)
        return 0

    _begin_immediate(conn)
    try:
        for json_path in storage_root.rglob("*.json"):
            try:
                upload_id = json_path.stem
                if not _looks_like_uuid(upload_id):
                    continue
                rel = str(
                    json_path.relative_to(storage_root)
                ).replace(os.sep, "/")
                mtime = datetime.fromtimestamp(
                    json_path.stat().st_mtime, tz=timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
                _upsert_transcript_issue_inner(
                    conn, upload_id,
                    storage_key=rel, summary=None, now=mtime,
                )
                scanned += 1
                in_batch += 1
                if in_batch >= batch_size:
                    _commit(conn)
                    await asyncio.sleep(0)
                    _begin_immediate(conn)
                    in_batch = 0
            except OSError as e:
                logger.warning("migration: skip %s: %s", json_path, e)
        _commit(conn)
    except Exception:
        _rollback(conn)
        raise
    logger.info("migration: completed, scanned=%d", scanned)
    return scanned
