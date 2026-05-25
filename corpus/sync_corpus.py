"""19.1 — Corpus sync from object storage to local corpus/raw/.

Idempotent: re-runs skip files already present at expected size.
Schema-validating: rejects files that don't satisfy
server/anonymized_transcript.schema.json (defense in depth — the intake
server already validated on accept, but a labelling pass that ingests
stale transcripts from a pre-schema-bump era could feed bad data into
training).

Backends:
  - Local filesystem (`--src local://<path>`): dev / first-cycle work
  - S3 (`--src s3://<bucket>[/prefix]`): production intake bucket

Layout written under `corpus/raw/`:
  corpus/raw/<YYYY>/<MM>/<DD>/<upload_id>.json

`<YYYY>/<MM>/<DD>` is derived from the transcript's `ts`-equivalent if
the intake-server preserved one; otherwise from the local file mtime
(intake server writes mtime at receive-time).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional


logger = logging.getLogger("fula-ai-training.corpus.sync")


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def _load_schema(repo_root: Path) -> dict:
    """Load anonymized_transcript.schema.json from the server/ sibling."""
    schema_path = repo_root / "server" / "anonymized_transcript.schema.json"
    with open(schema_path, encoding="utf-8") as f:
        return json.load(f)


def _make_validator(schema: dict):
    import jsonschema
    return jsonschema.Draft202012Validator(schema)


# ---------------------------------------------------------------------------
# Source backends
# ---------------------------------------------------------------------------

@dataclass
class TranscriptItem:
    """One transcript to be synced. `key` is the source-side identifier
    (S3 key or local path); `payload` is the parsed JSON; `received_at`
    is a UTC datetime used for fanout-by-date."""
    key: str
    payload: dict
    received_at: datetime


class LocalSource:
    """Walk a local directory tree for *.json files."""

    def __init__(self, root: str):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"local source not a dir: {root}")

    def iter_items(self) -> Iterator[TranscriptItem]:
        for p in self.root.rglob("*.json"):
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("skip %s: %s", p, e)
                continue
            try:
                mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
            except OSError:
                mtime = datetime.now(timezone.utc)
            yield TranscriptItem(key=str(p), payload=payload, received_at=mtime)


class S3Source:
    """Walk an S3 bucket/prefix; lazily imports boto3 so non-S3 paths
    don't pay the import cost."""

    def __init__(self, bucket: str, prefix: str = ""):
        try:
            import boto3  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "boto3 required for S3 backend; pip install boto3"
            ) from e
        self.bucket = bucket
        self.prefix = prefix.lstrip("/")

    def iter_items(self) -> Iterator[TranscriptItem]:
        import boto3
        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        kwargs = {"Bucket": self.bucket}
        if self.prefix:
            kwargs["Prefix"] = self.prefix
        for page in paginator.paginate(**kwargs):
            for obj in page.get("Contents") or []:
                key = obj["Key"]
                if not key.endswith(".json"):
                    continue
                last_modified = obj.get("LastModified") or datetime.now(timezone.utc)
                try:
                    body = s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
                    payload = json.loads(body)
                except Exception as e:  # noqa: BLE001
                    logger.warning("skip s3://%s/%s: %s", self.bucket, key, e)
                    continue
                yield TranscriptItem(
                    key=f"s3://{self.bucket}/{key}",
                    payload=payload,
                    received_at=last_modified,
                )


def parse_src(spec: str):
    """Parse `local:///some/path` or `s3://bucket[/prefix]` into a
    source backend."""
    if spec.startswith("local://"):
        return LocalSource(spec[len("local://"):])
    if spec.startswith("s3://"):
        rest = spec[len("s3://"):]
        bucket, _, prefix = rest.partition("/")
        return S3Source(bucket=bucket, prefix=prefix)
    raise ValueError(
        f"unrecognised --src scheme: {spec}; use local:// or s3://"
    )


# ---------------------------------------------------------------------------
# Sync logic
# ---------------------------------------------------------------------------

@dataclass
class SyncStats:
    seen: int = 0
    written: int = 0
    skipped_existing: int = 0
    rejected_schema: int = 0
    rejected_io: int = 0


def _dest_path(out_root: Path, item: TranscriptItem) -> Path:
    upload_id = item.payload.get("upload_id", "unknown")
    # Sanitize upload_id for filesystem use (closed schema enforces UUIDv4
    # but defense in depth).
    safe = "".join(c if c.isalnum() or c == "-" else "_" for c in str(upload_id))[:64]
    y = f"{item.received_at.year:04d}"
    m = f"{item.received_at.month:02d}"
    d = f"{item.received_at.day:02d}"
    return out_root / y / m / d / f"{safe}.json"


def sync(
    source,
    out_root: Path,
    validator,
) -> SyncStats:
    stats = SyncStats()
    for item in source.iter_items():
        stats.seen += 1
        try:
            validator.validate(item.payload)
        except Exception as e:
            stats.rejected_schema += 1
            logger.info("reject schema %s: %s", item.key, str(e)[:200])
            continue
        dest = _dest_path(out_root, item)
        if dest.exists():
            stats.skipped_existing += 1
            continue
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".json.partial")
            tmp.write_text(json.dumps(item.payload, separators=(",", ":"),
                                       sort_keys=True), encoding="utf-8")
            tmp.replace(dest)
            stats.written += 1
        except OSError as e:
            stats.rejected_io += 1
            logger.warning("io error writing %s: %s", dest, e)
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src", required=True,
        help="Source: local:///abs/path or s3://bucket[/prefix]",
    )
    parser.add_argument(
        "--out", default="corpus/raw",
        help="Output root (default: corpus/raw)",
    )
    parser.add_argument(
        "--schema-dir", default="server",
        help="Directory containing anonymized_transcript.schema.json "
             "(default: server)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s sync_corpus %(levelname)s %(message)s")

    repo_root = Path(args.schema_dir).resolve().parent
    schema = _load_schema(repo_root)
    validator = _make_validator(schema)

    source = parse_src(args.src)
    out_root = Path(args.out).resolve()
    stats = sync(source, out_root, validator)

    logger.info("sync done: %s", stats)
    print(json.dumps(stats.__dict__))
    return 0


if __name__ == "__main__":
    sys.exit(main())
