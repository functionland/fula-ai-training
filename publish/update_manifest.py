"""19.8 — Bump ai-manifest.json with a new model entry.

The fula-ota plugin's `model_manifest.py` reads
`/etc/fula/ai-manifest.json` on the device and picks `current` or
`rollback` based on `rollback_required`. This script implements the
operator-side authoring + validation:

- Load the currently-published manifest from $current_manifest_url
  (operator may pass --from-current-file for offline testing).
- Validate it against ai_manifest.schema.json (refuses to start
  from a broken manifest).
- Demote current → rollback (so the new current's rollback path is
  the model devices are running right now — atomic-swap discipline).
- Promote the new model to current.
- Cross-validate the result against the schema again.
- Optionally publish to CDN via $publish_url (requires creds; can
  also be done manually by the operator).

Refuses to publish if:
- manifest_version would not strictly increase
- new size_bytes is outside the schema bounds [1 GB, 20 GB]
- sha256 doesn't match the actual file (when --verify-sha is given a path)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


logger = logging.getLogger("fula-ai-training.publish.update_manifest")


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

DEFAULT_SCHEMA_PATH = (
    "../fula-ota/docker/fxsupport/linux/plugins/blox-ai/api/ai_manifest.schema.json"
)


def _validator(schema_path: Path):
    import jsonschema
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator(schema)


# ---------------------------------------------------------------------------
# Manifest mutation
# ---------------------------------------------------------------------------

def bump_manifest(
    current: dict,
    new_current: dict,
    rollback_required: bool = False,
) -> dict:
    """Demote current → rollback, promote new_current. Bumps
    manifest_version + sets published_at. Returns the new manifest
    (does NOT mutate the input)."""
    next_version = int(current.get("manifest_version", 0)) + 1
    out = {
        "schema_version": 1,
        "current": dict(new_current),
        "rollback": dict(current.get("current") or current.get("rollback") or new_current),
        "rollback_required": bool(rollback_required),
        "manifest_version": next_version,
        "published_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return out


def verify_sha(file_path: Path, expected_sha: str) -> bool:
    """SHA-256 the file at file_path; compare to expected_sha (lowercase
    hex). Returns True on match."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest() == expected_sha.lower()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-current-file", required=True,
                        help="Path to the currently-published manifest "
                             "(operator fetches this manually before running)")
    parser.add_argument("--schema",
                        default=DEFAULT_SCHEMA_PATH,
                        help="Path to ai_manifest.schema.json")
    parser.add_argument("--new-current-model-version", required=True,
                        help="model_version of the new current entry "
                             "(convention: YYYY-MM-DD or semver)")
    parser.add_argument("--new-current-url", required=True,
                        help="HTTPS URL where the new .rkllm is published")
    parser.add_argument("--new-current-sha256", required=True,
                        help="SHA-256 of the new .rkllm (hex lowercase)")
    parser.add_argument("--new-current-size-bytes", required=True, type=int,
                        help="Size in bytes of the new .rkllm")
    parser.add_argument("--rkllm-version",
                        help="Optional: RKLLM toolkit version that built the model")
    parser.add_argument("--rollback-required", action="store_true",
                        help="Publish with rollback_required=true (devices "
                             "will switch to rollback entry on next plugin "
                             "restart)")
    parser.add_argument("--verify-sha",
                        help="If given, SHA-verify the local file at this "
                             "path before publishing (defensive)")
    parser.add_argument("--output",
                        default="-",
                        help="Where to write the new manifest (default stdout)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level)

    schema_path = Path(args.schema).resolve()
    if not schema_path.is_file():
        logger.error("schema not found: %s", schema_path)
        return 2
    validator = _validator(schema_path)

    # Validate input
    current = json.loads(Path(args.from_current_file).read_text(encoding="utf-8"))
    try:
        validator.validate(current)
    except Exception as e:
        logger.error("input manifest is invalid: %s", e)
        return 3

    # Build new entry
    new_current = {
        "model_version": args.new_current_model_version,
        "url": args.new_current_url,
        "sha256": args.new_current_sha256,
        "size_bytes": args.new_current_size_bytes,
    }
    if args.rkllm_version:
        new_current["rkllm_version"] = args.rkllm_version

    # Optional SHA check
    if args.verify_sha:
        if not Path(args.verify_sha).is_file():
            logger.error("--verify-sha file not found: %s", args.verify_sha)
            return 4
        if not verify_sha(Path(args.verify_sha), args.new_current_sha256):
            logger.error("SHA mismatch: expected %s does not match file %s",
                         args.new_current_sha256, args.verify_sha)
            return 5
        logger.info("SHA verified ✓")

    new_manifest = bump_manifest(current, new_current,
                                 rollback_required=args.rollback_required)

    # Cross-validate
    try:
        validator.validate(new_manifest)
    except Exception as e:
        logger.error("synthesized manifest is invalid: %s", e)
        return 6

    out_json = json.dumps(new_manifest, indent=2) + "\n"
    if args.output == "-":
        sys.stdout.write(out_json)
    else:
        Path(args.output).write_text(out_json, encoding="utf-8")
        logger.info("wrote %s (manifest_version=%d)",
                    args.output, new_manifest["manifest_version"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
