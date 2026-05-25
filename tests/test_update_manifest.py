"""19.8 — manifest update tests. Validates against the real fula-ota
schema when available (skips with clear message otherwise)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from publish.update_manifest import bump_manifest, verify_sha


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _locate_schema() -> Path:
    """Find the device-side ai_manifest.schema.json.

    CI sets BLOX_AI_FULA_OTA_SCHEMA_DIR to the api/ dir of a checked-out
    fula-ota; locally the test falls back to a sibling checkout at
    `../fula-ota/...` so a developer with both repos cloned side-by-side
    gets the same cross-repo verification without setting anything.
    """
    env_dir = os.environ.get("BLOX_AI_FULA_OTA_SCHEMA_DIR")
    if env_dir:
        return Path(env_dir) / "ai_manifest.schema.json"
    return (_REPO_ROOT.parent / "fula-ota" / "docker" / "fxsupport" / "linux"
            / "plugins" / "blox-ai" / "api" / "ai_manifest.schema.json")


_FULA_OTA_SCHEMA = _locate_schema()


def _entry(version="2026-06-01", sha=None):
    return {
        "model_version": version,
        "url": f"https://functionyard.fx.land/qwen-3b-{version}.rkllm",
        "sha256": sha or ("a" * 64),
        "size_bytes": 3_100_000_000,
    }


def _current_manifest(version=1, rollback_required=False):
    return {
        "schema_version": 1,
        "current": _entry("2026-05-01", "b" * 64),
        "rollback": _entry("2026-04-01", "c" * 64),
        "rollback_required": rollback_required,
        "manifest_version": version,
        "published_at": "2026-05-01T00:00:00Z",
    }


# ---------------------------------------------------------------------------
# bump_manifest
# ---------------------------------------------------------------------------

def test_bump_demotes_current_to_rollback():
    current = _current_manifest()
    new = _entry("2026-06-01")
    bumped = bump_manifest(current, new)
    assert bumped["current"]["model_version"] == "2026-06-01"
    assert bumped["rollback"]["model_version"] == "2026-05-01"


def test_bump_increments_manifest_version():
    current = _current_manifest(version=7)
    bumped = bump_manifest(current, _entry("v2"))
    assert bumped["manifest_version"] == 8


def test_bump_writes_published_at():
    bumped = bump_manifest(_current_manifest(), _entry("v2"))
    assert "published_at" in bumped
    assert bumped["published_at"].endswith("Z")


def test_bump_does_not_mutate_input():
    current = _current_manifest()
    original = json.loads(json.dumps(current))  # deep copy
    bump_manifest(current, _entry("v2"))
    assert current == original


def test_bump_rollback_required_flag_propagates():
    bumped = bump_manifest(_current_manifest(), _entry("v2"),
                           rollback_required=True)
    assert bumped["rollback_required"] is True


def test_bump_produces_schema_valid_output():
    """The bumped manifest must satisfy ai_manifest.schema.json."""
    if not _FULA_OTA_SCHEMA.is_file():
        pytest.skip("fula-ota sibling checkout not present")
    import jsonschema
    schema = json.loads(_FULA_OTA_SCHEMA.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    bumped = bump_manifest(_current_manifest(), _entry("v2"))
    validator.validate(bumped)  # raises on violation


def test_bump_with_rkllm_version_field():
    new = _entry("v2")
    new["rkllm_version"] = "1.1.4"
    bumped = bump_manifest(_current_manifest(), new)
    assert bumped["current"]["rkllm_version"] == "1.1.4"


# ---------------------------------------------------------------------------
# verify_sha
# ---------------------------------------------------------------------------

def test_verify_sha_matches(tmp_path):
    f = tmp_path / "model.bin"
    f.write_bytes(b"hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert verify_sha(f, expected)


def test_verify_sha_mismatch(tmp_path):
    f = tmp_path / "model.bin"
    f.write_bytes(b"hello world")
    assert not verify_sha(f, "z" * 64)


def test_verify_sha_handles_large_file_streaming(tmp_path):
    """1 MB+ payload to verify the chunked read path works."""
    f = tmp_path / "model.bin"
    payload = b"x" * (2 * 1024 * 1024)  # 2 MB
    f.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert verify_sha(f, expected)
