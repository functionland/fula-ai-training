"""19.2 — labeller UI tests with FastAPI TestClient."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest


@pytest.fixture
def labeller_dirs(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    labelled = tmp_path / "labelled"
    raw.mkdir()
    labelled.mkdir()
    monkeypatch.setenv("BLOX_AI_RAW_DIR", str(raw))
    monkeypatch.setenv("BLOX_AI_LABELLED_DIR", str(labelled))
    # Force fresh import so the new env vars are picked up at module load.
    # `pop` + `import_module` is safer than `from x import y` + `reload`, which
    # can fail when another test has cached a stale parent-package attribute.
    import importlib, sys
    sys.modules.pop("labeller.app", None)
    sys.modules.pop("labeller", None)
    labmod = importlib.import_module("labeller.app")
    return raw, labelled, labmod


def _make_raw(raw_dir: Path, upload_id: str):
    (raw_dir / f"{upload_id}.json").write_text(json.dumps({
        "schema_version": 1,
        "upload_id": upload_id,
        "session_relative_start": "+0s",
        "events": [{"type": "session_started", "relative_ts": "+0s",
                    "payload": {}}],
        "user_rating": 1,
        "user_comment": "test comment",
        "consent": {"explicit_opt_in": True, "preview_shown": True,
                    "anonymizer_version": "0.1.0"},
    }))


def test_index_returns_html(labeller_dirs):
    _, _, mod = labeller_dirs
    from fastapi.testclient import TestClient
    c = TestClient(mod.app)
    r = c.get("/")
    assert r.status_code == 200
    assert "Blox AI labeller" in r.text


def test_transcripts_lists_only_unlabelled(labeller_dirs):
    raw, _, mod = labeller_dirs
    _make_raw(raw, "11111111-1111-4111-8111-111111111111")
    _make_raw(raw, "22222222-2222-4222-8222-222222222222")
    from fastapi.testclient import TestClient
    c = TestClient(mod.app)
    r = c.get("/transcripts")
    assert r.status_code == 200
    data = r.json()
    assert len(data["unlabelled"]) == 2
    assert data["total_labelled"] == 0


def test_get_transcript_returns_full_data(labeller_dirs):
    raw, _, mod = labeller_dirs
    _make_raw(raw, "33333333-3333-4333-8333-333333333333")
    from fastapi.testclient import TestClient
    c = TestClient(mod.app)
    r = c.get("/transcripts/33333333-3333-4333-8333-333333333333")
    assert r.status_code == 200
    data = r.json()
    assert data["upload_id"] == "33333333-3333-4333-8333-333333333333"


def test_get_transcript_404_on_unknown(labeller_dirs):
    _, _, mod = labeller_dirs
    from fastapi.testclient import TestClient
    c = TestClient(mod.app)
    r = c.get("/transcripts/nope")
    assert r.status_code == 404


def test_post_label_writes_labelled_file(labeller_dirs):
    raw, labelled, mod = labeller_dirs
    _make_raw(raw, "44444444-4444-4444-8444-444444444444")
    from fastapi.testclient import TestClient
    c = TestClient(mod.app)
    r = c.post("/label", json={
        "upload_id": "44444444-4444-4444-8444-444444444444",
        "label_decision": "accept",
        "verdict_correct": True,
        "actions_correct": True,
        "root_cause_correct": True,
        "notes": "looks good",
    })
    assert r.status_code == 200
    files = list(labelled.glob("*.labelled.json"))
    assert len(files) == 1
    saved = json.loads(files[0].read_text())
    assert saved["label_decision"] == "accept"
    assert saved["verdict_correct"] is True
    assert "labelled_at" in saved
    assert saved["transcript"]["upload_id"] == "44444444-4444-4444-8444-444444444444"


def test_post_label_rejects_invalid_decision(labeller_dirs):
    _, _, mod = labeller_dirs
    from fastapi.testclient import TestClient
    c = TestClient(mod.app)
    r = c.post("/label", json={
        "upload_id": "x",
        "label_decision": "maybe",  # not in {accept, partial, reject}
        "verdict_correct": True,
        "actions_correct": True,
        "root_cause_correct": True,
    })
    assert r.status_code == 422


def test_post_label_404_on_unknown_transcript(labeller_dirs):
    _, _, mod = labeller_dirs
    from fastapi.testclient import TestClient
    c = TestClient(mod.app)
    r = c.post("/label", json={
        "upload_id": "nope-such-id",
        "label_decision": "accept",
        "verdict_correct": True,
        "actions_correct": True,
        "root_cause_correct": True,
    })
    assert r.status_code == 404


def test_labelled_transcripts_disappear_from_list(labeller_dirs):
    raw, _, mod = labeller_dirs
    _make_raw(raw, "55555555-5555-4555-8555-555555555555")
    from fastapi.testclient import TestClient
    c = TestClient(mod.app)
    # Label it
    c.post("/label", json={
        "upload_id": "55555555-5555-4555-8555-555555555555",
        "label_decision": "accept",
        "verdict_correct": True,
        "actions_correct": True,
        "root_cause_correct": True,
    })
    # No longer in the unlabelled list
    r = c.get("/transcripts")
    data = r.json()
    assert len(data["unlabelled"]) == 0
    assert data["total_labelled"] == 1


def test_post_label_rejects_extra_fields(labeller_dirs):
    """Closed schema discipline: extra fields → 422."""
    _, _, mod = labeller_dirs
    from fastapi.testclient import TestClient
    c = TestClient(mod.app)
    r = c.post("/label", json={
        "upload_id": "x",
        "label_decision": "accept",
        "verdict_correct": True,
        "actions_correct": True,
        "root_cause_correct": True,
        "surprise": "field",
    })
    assert r.status_code == 422
