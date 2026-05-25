"""19.1 — sync_corpus tests with mocked source backends."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from corpus.sync_corpus import (
    LocalSource, TranscriptItem, _dest_path, parse_src, sync,
    SyncStats, _load_schema, _make_validator,
)


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def validator(repo_root):
    schema = _load_schema(repo_root)
    return _make_validator(schema)


def _make_valid_transcript(upload_id: str) -> dict:
    """A minimal transcript that satisfies the Phase 20 schema."""
    return {
        "schema_version": 1,
        "upload_id": upload_id,
        "session_relative_start": "+0s",
        "events": [
            {"type": "session_started", "relative_ts": "+0s", "payload": {}}
        ],
        "user_rating": 1,
        "consent": {
            "explicit_opt_in": True,
            "preview_shown": True,
            "anonymizer_version": "0.1.0",
        },
    }


def test_local_source_finds_json_files(tmp_path):
    f = tmp_path / "a.json"
    f.write_text('{"upload_id":"x"}')
    src = LocalSource(str(tmp_path))
    items = list(src.iter_items())
    assert len(items) == 1
    assert items[0].payload == {"upload_id": "x"}


def test_local_source_walks_subdirs(tmp_path):
    sub = tmp_path / "2026" / "05" / "24"
    sub.mkdir(parents=True)
    (sub / "x.json").write_text('{"upload_id":"x"}')
    src = LocalSource(str(tmp_path))
    items = list(src.iter_items())
    assert len(items) == 1


def test_local_source_skips_malformed_json(tmp_path):
    (tmp_path / "good.json").write_text('{"upload_id":"g"}')
    (tmp_path / "bad.json").write_text("{not json")
    items = list(LocalSource(str(tmp_path)).iter_items())
    assert len(items) == 1
    assert items[0].payload == {"upload_id": "g"}


def test_local_source_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        LocalSource(str(tmp_path / "no-such-dir"))


def test_parse_src_local(tmp_path):
    src = parse_src(f"local://{tmp_path}")
    assert isinstance(src, LocalSource)


def test_parse_src_invalid_scheme():
    with pytest.raises(ValueError, match="unrecognised --src scheme"):
        parse_src("ftp://nope")


def test_dest_path_fans_out_by_date(tmp_path):
    item = TranscriptItem(
        key="k",
        payload={"upload_id": "12345678-90ab-cdef-1234-567890abcdef"},
        received_at=datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc),
    )
    dest = _dest_path(tmp_path, item)
    assert dest.parts[-4:] == ("2026", "05", "24", "12345678-90ab-cdef-1234-567890abcdef.json")


def test_dest_path_sanitises_evil_upload_id(tmp_path):
    item = TranscriptItem(
        key="k",
        payload={"upload_id": "../../etc/passwd"},
        received_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
    )
    dest = _dest_path(tmp_path, item)
    # No ".." in the leaf; sanitized to underscores
    assert ".." not in dest.name
    assert "passwd" in dest.name  # the alphanum survives


def test_sync_writes_valid_transcripts(tmp_path, validator):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "a.json").write_text(json.dumps(_make_valid_transcript(
        "11111111-1111-4111-8111-111111111111"
    )))
    out_dir = tmp_path / "out"
    src = LocalSource(str(src_dir))
    stats = sync(src, out_dir, validator)
    assert stats.seen == 1
    assert stats.written == 1
    assert stats.rejected_schema == 0
    # File landed
    files = list(out_dir.rglob("*.json"))
    assert len(files) == 1


def test_sync_rejects_schema_violations(tmp_path, validator):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    bad = _make_valid_transcript("22222222-2222-4222-8222-222222222222")
    bad["session_relative_start"] = "+5s"  # tripwire: MUST be "+0s"
    (src_dir / "bad.json").write_text(json.dumps(bad))
    stats = sync(LocalSource(str(src_dir)), tmp_path / "out", validator)
    assert stats.rejected_schema == 1
    assert stats.written == 0


def test_sync_is_idempotent(tmp_path, validator):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    payload = _make_valid_transcript("33333333-3333-4333-8333-333333333333")
    (src_dir / "a.json").write_text(json.dumps(payload))
    out_dir = tmp_path / "out"
    src = LocalSource(str(src_dir))
    s1 = sync(src, out_dir, validator)
    s2 = sync(src, out_dir, validator)
    assert s1.written == 1
    assert s2.written == 0
    assert s2.skipped_existing == 1


def test_sync_handles_mixed_valid_invalid_corpus(tmp_path, validator):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "good.json").write_text(json.dumps(
        _make_valid_transcript("44444444-4444-4444-8444-444444444444")
    ))
    bad = _make_valid_transcript("55555555-5555-4555-8555-555555555555")
    del bad["consent"]
    (src_dir / "bad.json").write_text(json.dumps(bad))
    stats = sync(LocalSource(str(src_dir)), tmp_path / "out", validator)
    assert stats.seen == 2
    assert stats.written == 1
    assert stats.rejected_schema == 1
