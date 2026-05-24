# fula-ai-training

Off-device infrastructure for the Blox AI model improvement loop.

Two related but separately-deployed pieces live here:

1. **Transcript intake server** (Phase 20 of [`i-am-running-fula-ota-snuggly-finch.md`](https://github.com/anthropics/claude-internal-plans/blob/main/i-am-running-fula-ota-snuggly-finch.md)) — public-facing FastAPI service that receives opt-in anonymized troubleshooting transcripts uploaded by the Blox phone app. Lives in [`server/`](./server/).
2. **Training pipeline** (Phase 19 sub-plan: see [`fula-ai-training-pipeline.md`](https://github.com/anthropics/claude-internal-plans/blob/main/fula-ai-training-pipeline.md)) — offline pipeline that takes intake-server transcripts, labels them, fine-tunes Qwen2.5-3B via LoRA, validates against a held-out test set, and publishes new `.rkllm` files + manifest updates. Lives in `corpus/`, `labeller/`, `training/`, `compilation/`, `publish/` (per the sub-plan).

This README covers the intake server. The training pipeline is documented in its own sub-plan and gets fleshed out in a separate session sequence.

## Security posture (intake server)

The intake server is **the only new central channel** in the entire Blox AI plan. The whole architecture defaults to local-only; transcripts are uploaded only when the user explicitly taps "Upload to help improve AI" in the phone app, *after seeing the exact payload that will be sent*. The server's job is to receive that payload, validate it, and persist it for the training pipeline — and to do so without leaking anything back.

Hard rules (enforced in code; tested):

- **TLS-only ingress.** HTTP requests are refused (production deployment terminates TLS at the load balancer; the server itself binds to a TLS port in production, plaintext only inside the cluster).
- **Source IP is ephemeral.** The server logs the source IP exactly once (for rate-limit accounting + abuse triage) and never persists it alongside the transcript. The transcript file written to object storage has no client-IP field.
- **Schema validation BEFORE persist.** Every uploaded payload is validated against [`server/anonymized_transcript.schema.json`](./server/anonymized_transcript.schema.json). Anything failing validation is rejected with a generic 400 — no field-level error echo (which would leak structure to an attacker probing the schema).
- **Anonymization sanity at server side.** Even though the phone-side anonymizer strips PII before upload, the server runs a defense-in-depth scan for IPv4/IPv6 literals, peerId-like base58 strings, and SSID-looking values in any text field. Hits → reject with `anonymization_check_failed` + alert.
- **No reply payload.** The server returns `{}` on success (HTTP 204 also acceptable). It does NOT echo the transcript back. It does NOT return anything derived from the transcript that could fingerprint the device. Idempotency is keyed on a client-supplied opaque `upload_id` (UUID), not on transcript content.
- **No central telemetry from the device.** This server is the only ingress; there is no `POST /heartbeat` or `POST /telemetry` etc.
- **Bucket isolation.** The S3 bucket the server writes to is write-only-from-server (IAM policy enforces — write but no list, no read). The training pipeline reads via a separate read-only IAM identity. This means a server-side compromise can write but cannot exfiltrate prior uploads.

**Server-side log content note.** When the defense-in-depth scanner rejects a transcript, the *log line* records the scanner name (`anonymization_check_failed scanner=ipv4_literal ip=<hashed>`). This is intentional for operator forensics — when a wave of rejections hits a single scanner, the operator can correlate to a specific app version via the `anonymizer_version` field. The scanner name and the matched substring are NEVER returned to the client; only the operator-side log sees them. If your deployment has a stricter no-leak-to-logs posture, set the log level to ERROR (rejections log at WARNING).

What this server does NOT defend against:

- An attacker on the user's device modifying the anonymizer to leak PII into a "looks fine" transcript that passes both client + server scans. That's an endpoint-compromise risk; this server can't fix it.
- A user who deliberately uploads a transcript containing data they want shared. The "you'll see exactly what gets uploaded" preview in the app is the gate for that.
- Bulk submission abuse from a single source. Rate limits help but a well-resourced attacker can rotate IPs. Mitigation: cap per-IP at 50/day and globally at 10000/day; alert on bursts.

## Repo layout (current state)

```
fula-ai-training/
├── README.md                              (this file)
├── server/                                 — Phase 20 intake server
│   ├── anonymized_transcript.schema.json   — strict closed schema
│   ├── app.py                              — FastAPI app
│   ├── anonymization_check.py              — defense-in-depth PII scanner
│   ├── storage.py                          — object-storage adapter (local + S3)
│   ├── requirements.txt
│   └── tests/
│       ├── conftest.py
│       └── test_intake.py
└── (training pipeline: scaffolded in a later sub-plan session)
```

## Phase 20 deliverable (this PR)

- The schema file above.
- `app.py` implementing `POST /transcripts` with the validation chain documented above.
- `anonymization_check.py` with the defense-in-depth regex scanner.
- `storage.py` with a `LocalDirStorage` (dev) + `S3Storage` (prod) adapter sharing one `Storage` protocol.
- `tests/` covering: happy path; schema violations; anonymization-check rejections; idempotent retry on same upload_id; rate-limit behavior (mocked); source-IP not persisted.

## Phase 20 does NOT yet include

- TLS termination (handled by deployment ingress).
- Production S3 bucket policy (lives in an infra repo).
- The training-pipeline modules (see the Phase 19 sub-plan).
- Per-user rate limiting beyond per-IP (we don't have user identity — by design).
- Bucket retention policy (separate decision; suggest 12 months default with operator override).

## Running locally (dev)

```bash
cd server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
BLOX_AI_STORAGE_DIR=/tmp/blox-ai-intake \
  uvicorn app:app --host 127.0.0.1 --port 8765 --reload
```

Then:

```bash
curl -X POST http://127.0.0.1:8765/transcripts \
     -H 'Content-Type: application/json' \
     -H 'X-Upload-Id: 11111111-1111-1111-1111-111111111111' \
     -d @example_transcript.json
```

## Tests

```bash
cd server
pip install -r requirements.txt pytest httpx
pytest tests/ -q
```
