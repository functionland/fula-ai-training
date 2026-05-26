# Blox AI transcript intake server — minimal, hardened, multi-arch.
#
# Hardening choices (all enforced at runtime by docker-compose.yml as well):
#   - Non-root user (uid 1000) baked in; matches the host-side storage dir
#     ownership the install.sh sets up.
#   - Read-only root fs at runtime (compose sets read_only: true) — storage
#     mounts an explicit writable volume.
#   - No build-time secrets. All config via env vars.
#   - Single-stage: this app is ~50 MB, multi-stage adds no real benefit.

FROM python:3.12-slim-bookworm

# Pin OS deps to what's needed. No build-essential, no curl, no compilers.
# python:3.12-slim already has pip; we only need to install jsonschema's
# C-accelerated bits (which ship as wheels for amd64+arm64 — no compile).
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system --gid 1000 blox \
 && useradd  --system --uid 1000 --gid 1000 --home-dir /app --shell /usr/sbin/nologin blox

WORKDIR /app

# Install Python deps first — better layer caching.
COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

# Copy ONLY the server source. The training pipeline and tests don't ship.
# This includes admin.py + issue_db.py + admin_ui.html for the inbox UI.
COPY server/ /app/server/

# Default storage + DB dirs (compose mounts host volumes here so the
# read-only rootfs doesn't block writes). BLOX_AI_ADMIN_TOKEN is intentionally
# NOT set in the image — operators provide it via .env so the token never
# bakes into a build artifact.
ENV BLOX_AI_STORAGE_DIR=/var/lib/blox-ai-intake/transcripts \
    BLOX_AI_DB_PATH=/var/lib/blox-ai-intake/db/issues.sqlite \
    BLOX_AI_TRUST_XFF=1 \
    BLOX_AI_LOG_LEVEL=INFO \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Switch CWD to where the modules live so bare imports resolve.
# server/app.py uses flat imports (`import issue_db`, `from
# anonymization_check import find_pii`, etc.) which expect the
# server/ directory to be on sys.path. Python always adds CWD to
# sys.path, so making /app/server the CWD is the simplest fix — no
# PYTHONPATH gymnastics, no `server.` prefix rewrites, and the test
# suite's conftest.py already uses the same layout.
WORKDIR /app/server

USER blox

EXPOSE 8000

# uvicorn binds 0.0.0.0:8000 INSIDE the container — docker-compose maps
# that to 127.0.0.1:8090 on the host (never publicly exposed; nginx
# proxies it). 1 worker is plenty for this load profile; intake server
# is I/O bound on file writes.
# CMD uses `app:app` (not `server.app:app`) because WORKDIR is now
# /app/server, so `app` resolves to /app/server/app.py directly.
CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "127.0.0.1"]
