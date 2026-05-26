#!/usr/bin/env bash
# Blox AI transcript intake server — installer for an Ubuntu/Debian VPS
# that ALREADY runs functionland/pinning-service and/or functionland/fula-api.
#
# Coexistence guarantees (audited against the existing install scripts):
#   - NEVER runs `ufw --force reset` (fula-api install.sh:639 does that,
#     it would wipe pinning-service's deny rules for 5432/5001/9094/9095).
#     We only ADD ufw rules, and only if they're not already present.
#   - NEVER edits an existing nginx site-available file. We write
#     /etc/nginx/sites-available/ai-training.fx.land (unique name).
#   - NEVER restarts the Docker daemon. Container deps for other
#     services stay up.
#   - NEVER changes the binding of existing containers. We only
#     manage `blox-ai-intake` (unique container name).
#   - Our container publishes 127.0.0.1:8090 ONLY — same defense-in-depth
#     posture pinning-service's `harden_infrastructure_bindings`
#     enforces for postgres and kubo API. Public traffic reaches us
#     only through nginx + Let's Encrypt TLS on 443.
#
# Required: Ubuntu 22.04+ or Debian 12+, root (sudo), DNS A-record
# already pointing $DOMAIN at this host.
#
# Usage:
#   sudo bash install.sh
#     (interactive — prompts for domain + SSL email)
#   sudo DOMAIN=ai-training.fx.land SSL_EMAIL=admin@fx.land bash install.sh
#     (non-interactive)
#   sudo bash install.sh --check
#     (dry-run: print what would change, change nothing)
#
# Re-run safety: idempotent. Re-running upgrades the image, refreshes
# the cert if needed, leaves all your other services untouched.

set -euo pipefail

# ────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────

DOMAIN="${DOMAIN:-}"
SSL_EMAIL="${SSL_EMAIL:-}"
DRY_RUN=0
ASSUME_YES=0
CONFLICTS_FOUND=0  # incremented by coexistence_preflight()
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Stable filenames so re-runs are idempotent + collision-free with
# pinning-service / fula-api conventions.
NGINX_SITE_NAME="ai-training-intake"
NGINX_SITE_AVAILABLE="/etc/nginx/sites-available/${NGINX_SITE_NAME}"
NGINX_SITE_ENABLED="/etc/nginx/sites-enabled/${NGINX_SITE_NAME}"
STORAGE_DIR="/var/lib/blox-ai-intake/transcripts"
DB_DIR="/var/lib/blox-ai-intake/db"
LOG_DIR="/var/log/blox-ai-intake"
SERVICE_NAME="blox-ai-intake"
HOST_BIND_PORT=8090
ENV_FILE="${INSTALL_DIR}/.env"

# Args
for arg in "$@"; do
  case "$arg" in
    --check) DRY_RUN=1 ;;
    -y|--yes) ASSUME_YES=1 ;;
    -h|--help)
      sed -n '2,28p' "$0"
      echo
      echo "Flags:"
      echo "  --check     Dry-run: print what would change, change nothing"
      echo "  -y, --yes   Skip the 'continue despite conflicts' prompt"
      echo "  -h, --help  This message"
      exit 0
      ;;
  esac
done

# ────────────────────────────────────────────────────────────────────
# Pretty printing
# ────────────────────────────────────────────────────────────────────

c_red()    { printf '\033[31m%s\033[0m\n' "$*"; }
c_green()  { printf '\033[32m%s\033[0m\n' "$*"; }
c_yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
c_blue()   { printf '\033[34m%s\033[0m\n' "$*"; }

info()    { c_blue   "  → $*"; }
ok()      { c_green  "  ✓ $*"; }
warn()    { c_yellow "  ⚠ $*"; }
err()     { c_red    "  ✗ $*" >&2; }
section() { echo; c_blue "═══ $* ═══"; }

would() {
  if [ "$DRY_RUN" = 1 ]; then
    c_yellow "  [dry-run] would: $*"
    return 0
  fi
  return 1
}

# ────────────────────────────────────────────────────────────────────
# Preflight
# ────────────────────────────────────────────────────────────────────

require_root() {
  if [ "$(id -u)" -ne 0 ]; then
    err "Run as root: sudo bash install.sh"
    exit 1
  fi
}

require_distro() {
  if ! grep -qE 'Ubuntu|Debian' /etc/os-release 2>/dev/null; then
    err "Only Ubuntu/Debian are supported (detected: $(. /etc/os-release; echo "$PRETTY_NAME"))"
    exit 1
  fi
}

prompt_domain_email() {
  if [ -z "$DOMAIN" ]; then
    read -rp "Domain (e.g. ai-training.fx.land): " DOMAIN
  fi
  if [ -z "$DOMAIN" ]; then
    err "Domain is required (or set DOMAIN env var)"
    exit 1
  fi

  if [ -z "$SSL_EMAIL" ]; then
    read -rp "Email for Let's Encrypt notifications: " SSL_EMAIL
  fi
  if [ -z "$SSL_EMAIL" ]; then
    err "SSL email is required (or set SSL_EMAIL env var). Let's Encrypt sends expiry warnings here."
    exit 1
  fi
}

check_dns_or_warn() {
  local resolved
  resolved=$(getent hosts "$DOMAIN" 2>/dev/null | awk '{print $1; exit}' || true)
  if [ -z "$resolved" ]; then
    warn "DNS resolution for $DOMAIN returned nothing."
    warn "Add an A record pointing $DOMAIN at this server's public IP before SSL will succeed."
    warn "Continuing — certbot will fail loudly if DNS isn't ready yet."
    sleep 2
  else
    info "DNS: $DOMAIN → $resolved"
  fi
}

# ────────────────────────────────────────────────────────────────────
# Coexistence pre-flight — verify the host is in a state we can safely
# add to, BEFORE we change anything. Catches:
#   - port 8090 already in use by some OTHER service (would block
#     `docker compose up` later; better to fail loudly NOW)
#   - existing nginx configuration is broken (we'd see `nginx -t` fail
#     after writing our site, which would correctly NOT enable us, but
#     also tells you something is already wrong on the host)
#   - existing nginx site already serves the same server_name we want
#     (also caught by nginx -t, but better to warn early)
# ────────────────────────────────────────────────────────────────────

PORT_8090_PRE_USE=0  # set by coexistence_preflight() if something else holds the port

# ────────────────────────────────────────────────────────────────────
# Read-only diagnostic — show the operator what's CURRENTLY on the
# host and exactly what we plan to add, BEFORE any mutating step
# (including apt installs). Lets the operator see at a glance whether
# we'd step on any existing service.
# ────────────────────────────────────────────────────────────────────

show_host_state() {
  section "0a. current host state (read-only diagnostic)"

  # --- UFW: status + rules ---
  echo
  info "UFW status (current rules + default policies):"
  if command -v ufw >/dev/null 2>&1; then
    ufw status verbose 2>/dev/null | sed 's/^/    /' || echo "    (ufw status failed)"
  else
    echo "    (ufw not installed — will be installed)"
  fi

  # --- Listening TCP ports + binding process ---
  echo
  info "Listening TCP ports (ss -lntp):"
  if command -v ss >/dev/null 2>&1; then
    ss -lntp 2>/dev/null | sed 's/^/    /'
  elif command -v netstat >/dev/null 2>&1; then
    netstat -lntp 2>/dev/null | sed 's/^/    /'
  else
    echo "    (neither ss nor netstat available)"
  fi

  # --- Running Docker containers ---
  echo
  info "Running Docker containers:"
  if command -v docker >/dev/null 2>&1; then
    docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Ports}}' 2>/dev/null \
      | sed 's/^/    /' || echo "    (docker ps failed; daemon may be down)"
  else
    echo "    (docker not installed — will be installed)"
  fi

  # --- Enabled nginx sites ---
  echo
  info "Enabled nginx sites:"
  if [ -d /etc/nginx/sites-enabled ]; then
    ls -1 /etc/nginx/sites-enabled/ 2>/dev/null | sed 's/^/    /' \
      || echo "    (sites-enabled dir empty or unreadable)"
  else
    echo "    (no /etc/nginx/sites-enabled — nginx not installed yet)"
  fi

  # --- Existing certbot certs (we'll only renew, never delete) ---
  echo
  info "Existing Let's Encrypt certificates (certbot manages renewal cron):"
  if command -v certbot >/dev/null 2>&1; then
    certbot certificates 2>/dev/null \
      | grep -E '^(  Certificate Name:|    Domains:|    Expiry Date:)' \
      | sed 's/^/    /' || echo "    (none issued)"
  else
    echo "    (certbot not installed — will be installed)"
  fi

  # --- What we plan to add ---
  echo
  info "What this installer will ADD to your host:"
  echo "    UFW rules (added only if NOT already present):"
  echo "      ufw allow 80/tcp     # for nginx (HTTP, redirected to HTTPS by certbot)"
  echo "      ufw allow 443/tcp    # for nginx (HTTPS — our public endpoint)"
  echo "      ufw deny  ${HOST_BIND_PORT}/tcp   # SKIPPED if anything is already on ${HOST_BIND_PORT}"
  echo "    Docker container: ${SERVICE_NAME}"
  echo "      Image: blox-ai-intake:local (built locally from ./Dockerfile)"
  echo "      Bind:  127.0.0.1:${HOST_BIND_PORT}:8000  (LOCAL-ONLY, no public exposure)"
  echo "    nginx site: ${NGINX_SITE_AVAILABLE}"
  echo "      server_name: ${DOMAIN}"
  echo "      proxies to:  127.0.0.1:${HOST_BIND_PORT}"
  echo "    Host directories (owner 1000:1000, mode 0750):"
  echo "      ${STORAGE_DIR}/    (transcripts — bind-mounted RW)"
  echo "      ${DB_DIR}/    (SQLite WAL files — bind-mounted RW)"
  echo "      ${LOG_DIR}/    (operator-readable log copies)"
  echo "    Let's Encrypt cert for: ${DOMAIN}  (via certbot --nginx)"
  echo "    Admin bearer token: written to ${ENV_FILE} (auto-generated if missing)"
}

coexistence_preflight() {
  section "0b. coexistence checks (detect interference)"

  # ── Port 8090 in use? ──
  # `ss -lnt` reads /proc/net/tcp; works without root for listening sockets.
  # We accept binds on 0.0.0.0:8090, [::]:8090, 127.0.0.1:8090, or any
  # specific interface — any of those would collide with our compose bind.
  if command -v ss >/dev/null 2>&1; then
    if ss -lnt 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)${HOST_BIND_PORT}\$"; then
      PORT_8090_PRE_USE=1
      CONFLICTS_FOUND=$((CONFLICTS_FOUND + 1))
      warn "CONFLICT: port ${HOST_BIND_PORT}/tcp is already listening on this host"
      warn "  Identified above in the 'Listening TCP ports' diagnostic."
      warn "  Will SKIP our ufw deny rule on ${HOST_BIND_PORT}/tcp (could disrupt them)."
      warn "  \`docker compose up\` will likely fail with 'address already in use'."
      warn "  Fix options:"
      warn "    a) If this is an OLD blox-ai-intake instance:"
      warn "         sudo docker rm -f ${SERVICE_NAME}"
      warn "    b) If it's a different service, edit docker-compose.yml +"
      warn "       this script to use a different HOST_BIND_PORT (e.g. 8091)."
    else
      ok "Port ${HOST_BIND_PORT}/tcp is free"
    fi
  else
    info "ss(8) not available — skipping port-in-use pre-check"
  fi

  # ── nginx state currently valid? ──
  # Pre-existing broken config in OTHER sites would make our later
  # `nginx -t` fail — better to surface that now rather than after
  # we've changed things. This is read-only (just `nginx -t`).
  if command -v nginx >/dev/null 2>&1; then
    if ! nginx -t >/dev/null 2>&1; then
      err "BLOCKER: nginx -t already fails BEFORE our changes — your other site"
      err "  configs have an error. Run \`sudo nginx -t\` and fix what it reports"
      err "  FIRST, then re-run install.sh. We are aborting now so we don't make"
      err "  a bad situation harder to debug."
      exit 2
    fi
    ok "nginx config currently passes -t"
  else
    info "nginx not installed yet — will be installed + tested"
  fi

  # ── Existing site with the same server_name? ──
  if [ -d /etc/nginx/sites-enabled ] && command -v grep >/dev/null 2>&1; then
    local matches
    matches=$(grep -rlE "server_name[[:space:]]+[^;]*\b${DOMAIN//./\\.}\b" \
                /etc/nginx/sites-enabled/ 2>/dev/null \
                | grep -v "/${NGINX_SITE_NAME}\$" || true)
    if [ -n "$matches" ]; then
      CONFLICTS_FOUND=$((CONFLICTS_FOUND + 1))
      warn "CONFLICT: another enabled nginx site already references server_name '${DOMAIN}':"
      while IFS= read -r f; do warn "    $f"; done <<<"$matches"
      warn "  nginx -t will fail with 'conflicting server name' when we add ours."
      warn "  Fix: move the existing reference to a different vhost FIRST,"
      warn "  or pick a different domain via DOMAIN env var when re-running."
    else
      ok "No other site claims server_name '${DOMAIN}'"
    fi
  fi

  # ── Container name collision? ──
  if command -v docker >/dev/null 2>&1 \
     && docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "$SERVICE_NAME"; then
    # An existing container with our name is FINE for re-runs (docker compose
    # will recreate it). Only flag if it's NOT compose-managed (compose sets a
    # label "com.docker.compose.service" we can check) AND not at our path.
    local managed_path
    managed_path=$(docker inspect --format '{{ index .Config.Labels "com.docker.compose.project.working_dir" }}' "$SERVICE_NAME" 2>/dev/null || true)
    if [ -n "$managed_path" ] && [ "$managed_path" != "$INSTALL_DIR" ]; then
      CONFLICTS_FOUND=$((CONFLICTS_FOUND + 1))
      warn "CONFLICT: a Docker container named '${SERVICE_NAME}' already exists,"
      warn "  managed by a compose project at: $managed_path"
      warn "  (this installer is in: $INSTALL_DIR)"
      warn "  Re-running here will recreate the container using OUR compose file."
    else
      ok "Existing container '${SERVICE_NAME}' will be recreated by our compose (safe re-run)"
    fi
  else
    ok "No existing container named '${SERVICE_NAME}'"
  fi

  echo
  if [ "$CONFLICTS_FOUND" -eq 0 ]; then
    ok "No coexistence conflicts detected."
  else
    warn "$CONFLICTS_FOUND coexistence conflict(s) flagged above."
  fi
}

# ────────────────────────────────────────────────────────────────────
# Confirmation gate — if conflicts were detected, prompt the operator
# to either acknowledge or abort. --yes skips the prompt.
# ────────────────────────────────────────────────────────────────────

confirm_or_abort_on_conflicts() {
  if [ "$CONFLICTS_FOUND" -eq 0 ]; then
    return 0
  fi
  if [ "$DRY_RUN" = 1 ]; then
    info "[dry-run] would prompt the operator to continue despite ${CONFLICTS_FOUND} conflict(s)."
    return 0
  fi
  if [ "$ASSUME_YES" = 1 ]; then
    warn "--yes flag set; continuing despite ${CONFLICTS_FOUND} conflict(s)."
    return 0
  fi
  if [ ! -t 0 ]; then
    err "Non-interactive shell with ${CONFLICTS_FOUND} conflict(s) detected."
    err "Re-run with -y to acknowledge, or fix the conflicts above first."
    exit 1
  fi
  echo
  local reply=""
  read -rp "Continue with install despite ${CONFLICTS_FOUND} conflict(s)? [y/N] " reply
  case "$reply" in
    y|Y|yes|YES) ok "Operator acknowledged. Continuing." ;;
    *) err "Aborting at operator request. No changes made yet."; exit 1 ;;
  esac
}

# ────────────────────────────────────────────────────────────────────
# OS packages — install nginx, certbot, docker if missing
# Never re-install if already present (don't disturb other services).
# ────────────────────────────────────────────────────────────────────

ensure_apt_pkg() {
  local pkg="$1"
  if dpkg -s "$pkg" >/dev/null 2>&1; then
    info "$pkg already installed — skipping"
    return 0
  fi
  if would "apt-get install -y $pkg"; then return 0; fi
  apt-get install -y -qq "$pkg"
  ok "$pkg installed"
}

ensure_docker() {
  if command -v docker >/dev/null 2>&1; then
    info "docker already installed: $(docker --version)"
  else
    if would "install docker via get.docker.com"; then return 0; fi
    info "Installing docker (fula-api install.sh:102 uses the same one-liner)..."
    curl -fsSL https://get.docker.com | sh
    ok "docker installed"
  fi

  # Docker Compose plugin (v2) — preferred; fall back to standalone
  # `docker-compose` only if the plugin really isn't there.
  if docker compose version >/dev/null 2>&1; then
    info "docker compose (v2 plugin) present"
  elif command -v docker-compose >/dev/null 2>&1; then
    info "docker-compose (v1 standalone) present — will use that"
  else
    if would "apt-get install -y docker-compose-plugin"; then return 0; fi
    apt-get install -y -qq docker-compose-plugin || apt-get install -y -qq docker-compose
    ok "docker compose installed"
  fi
}

install_prereqs() {
  section "1. system packages"
  if would "apt-get update"; then :
  else apt-get update -qq; fi

  ensure_apt_pkg nginx
  ensure_apt_pkg certbot
  ensure_apt_pkg python3-certbot-nginx
  ensure_apt_pkg ufw
  ensure_apt_pkg curl
  ensure_docker
}

# ────────────────────────────────────────────────────────────────────
# Storage directory — host-side persistence for transcripts
# Owned by uid 1000 to match the container's `blox` user.
# ────────────────────────────────────────────────────────────────────

setup_storage() {
  section "2. storage + db directories"
  if would "mkdir -p $STORAGE_DIR $DB_DIR (mode 0750, owner 1000:1000)"; then return 0; fi
  mkdir -p "$STORAGE_DIR" "$DB_DIR" "$LOG_DIR"
  chown -R 1000:1000 "$STORAGE_DIR" "$DB_DIR" "$LOG_DIR"
  chmod 0750 "$STORAGE_DIR" "$DB_DIR" "$LOG_DIR"
  ok "Storage: $STORAGE_DIR (1000:1000, 0750)"
  ok "DB:      $DB_DIR     (1000:1000, 0750) — SQLite WAL files live here"
  ok "Logs:    $LOG_DIR"
}

# ────────────────────────────────────────────────────────────────────
# Admin token bootstrap — generate BLOX_AI_ADMIN_TOKEN if .env doesn't
# have one yet. Persisted to .env so docker-compose's ${VAR:-} default
# substitution picks it up.
# ────────────────────────────────────────────────────────────────────

setup_admin_token() {
  section "2b. admin web-UI token"
  if [ "$DRY_RUN" = 1 ]; then
    if [ -f "$ENV_FILE" ] && grep -q '^BLOX_AI_ADMIN_TOKEN=' "$ENV_FILE"; then
      info "[dry-run] would: keep existing BLOX_AI_ADMIN_TOKEN in .env"
    else
      info "[dry-run] would: generate new BLOX_AI_ADMIN_TOKEN, append to .env"
    fi
    return 0
  fi

  # Create .env if missing
  if [ ! -f "$ENV_FILE" ]; then
    touch "$ENV_FILE"
    chmod 0640 "$ENV_FILE"
  fi

  if grep -q '^BLOX_AI_ADMIN_TOKEN=' "$ENV_FILE"; then
    # Don't print the token (it's already set; printing would leak it
    # to anything tailing the install log).
    info "BLOX_AI_ADMIN_TOKEN already set in $ENV_FILE — keeping"
    GENERATED_TOKEN=""
    return 0
  fi

  # openssl produces a URL-safe-ish base64; trim '/+=' to avoid quoting
  # surprises inside docker-compose env interpolation.
  local token
  token=$(openssl rand -base64 33 | tr -d '/+=' | head -c 40)
  if [ -z "$token" ]; then
    err "openssl rand failed — cannot generate admin token"
    exit 1
  fi
  printf 'BLOX_AI_ADMIN_TOKEN=%s\n' "$token" >> "$ENV_FILE"
  chmod 0640 "$ENV_FILE"
  GENERATED_TOKEN="$token"
  ok "Generated BLOX_AI_ADMIN_TOKEN (printed at the end of this script)"
}

# ────────────────────────────────────────────────────────────────────
# UFW — ADD ONLY. Never reset, never deny something that's already
# allowed. Existing pinning-service/fula-api rules survive untouched.
# ────────────────────────────────────────────────────────────────────

ufw_has_rule() {
  # Returns 0 if a rule allowing the given port already exists.
  local port="$1"
  ufw status 2>/dev/null | grep -qE "^${port}/tcp\s+ALLOW" \
   || ufw status 2>/dev/null | grep -qE "^${port}\s+ALLOW"
}

setup_firewall() {
  section "3. firewall (additive only — never resets)"
  if ! command -v ufw >/dev/null 2>&1; then
    warn "ufw not installed; skipping"
    return 0
  fi
  if ! ufw status 2>/dev/null | grep -q "^Status: active"; then
    warn "ufw is installed but inactive. Leaving it that way — enabling it"
    warn "  here could lock you out if your SSH rule isn't in place."
    warn "  If you want it active: \`sudo ufw allow ssh && sudo ufw enable\`"
    return 0
  fi

  for port in 80 443; do
    if ufw_has_rule "$port"; then
      info "ufw: ${port}/tcp already allowed — no change"
    else
      if would "ufw allow ${port}/tcp comment 'blox-ai-intake nginx (added by ai-training install.sh)'"; then continue; fi
      ufw allow "${port}/tcp" comment 'blox-ai-intake nginx (added by ai-training install.sh)' >/dev/null
      ok "ufw: allowed ${port}/tcp"
    fi
  done

  # Defense in depth: our container is bound to 127.0.0.1:8090 by docker,
  # so 8090 is NEVER routable from the public internet. Adding a `deny`
  # rule here is pinning-service-style cosmetic (Docker writes its own
  # iptables) but documents intent + protects against the case where
  # someone later flips the binding to 0.0.0.0 manually.
  #
  # IMPORTANT: skip this if SOMETHING ELSE on the host is already using
  # 8090 (detected by coexistence_preflight). A deny rule could block
  # that other service. We prefer to fail loudly at `docker compose up`
  # later (port collision) than silently break whatever was already there.
  if [ "$PORT_8090_PRE_USE" = 1 ]; then
    warn "ufw: skipping deny on ${HOST_BIND_PORT}/tcp because some other"
    warn "     service is already listening there (would disrupt them)."
  elif ufw status | grep -qE "^${HOST_BIND_PORT}/tcp\s+DENY"; then
    info "ufw: ${HOST_BIND_PORT}/tcp deny already present"
  else
    if ! would "ufw deny ${HOST_BIND_PORT}/tcp comment 'blox-ai-intake container must stay bound to 127.0.0.1'"; then
      ufw deny "${HOST_BIND_PORT}/tcp" comment 'blox-ai-intake container must stay bound to 127.0.0.1' >/dev/null
      ok "ufw: explicit deny on ${HOST_BIND_PORT}/tcp (defense in depth — Docker binding is 127.0.0.1 only)"
    fi
  fi
}

# ────────────────────────────────────────────────────────────────────
# Docker container — build, run, hardened.
# ────────────────────────────────────────────────────────────────────

compose_cmd() {
  # Echo the right compose invocation for this host.
  if docker compose version >/dev/null 2>&1; then
    echo "docker compose"
  else
    echo "docker-compose"
  fi
}

build_and_start() {
  section "4. docker image + container"

  cd "$INSTALL_DIR"
  if [ ! -f Dockerfile ] || [ ! -f docker-compose.yml ]; then
    err "Dockerfile or docker-compose.yml missing from $INSTALL_DIR"
    err "Run install.sh from the repo root (where docker-compose.yml lives)."
    exit 1
  fi

  local COMPOSE
  COMPOSE=$(compose_cmd)

  if would "$COMPOSE build && $COMPOSE up -d"; then return 0; fi

  info "Building blox-ai-intake:local..."
  $COMPOSE build --pull

  info "(Re)starting container..."
  $COMPOSE up -d --force-recreate

  # Wait for /health
  info "Waiting for container to report healthy..."
  local tries=0
  until curl -sS --max-time 3 "http://127.0.0.1:${HOST_BIND_PORT}/health" 2>/dev/null | grep -q '"ok":true'; do
    tries=$((tries + 1))
    if [ "$tries" -gt 30 ]; then
      err "Container didn't respond on /health within 90s. Recent logs:"
      docker logs --tail 50 "$SERVICE_NAME" >&2 || true
      exit 1
    fi
    sleep 3
  done
  ok "Container healthy on 127.0.0.1:${HOST_BIND_PORT}"
}

# ────────────────────────────────────────────────────────────────────
# nginx site — written under a UNIQUE filename so existing sites stay
# untouched. Only enables our site; existing enabled sites stay enabled.
# ────────────────────────────────────────────────────────────────────

write_nginx_site() {
  section "5. nginx site for $DOMAIN"

  # Compose the site config in-memory; only write if different. Avoids
  # marking the file modified on every re-run.
  local desired
  desired=$(cat <<EOF
# Managed by fula-ai-training/install.sh — safe to edit but re-running
# install.sh will overwrite changes here.
server {
    listen 80;
    listen [::]:80;
    server_name ${DOMAIN};

    # certbot --nginx will rewrite this to redirect to https + add the
    # ACME challenge location. Keeping the plain block here means
    # certbot has something to attach to on the FIRST run.

    # Defense-in-depth headers (nginx-level; the app also enforces
    # privacy via no-echo error bodies).
    server_tokens off;

    # Strip any client-supplied X-Forwarded-* before we inject our own.
    # The app trusts XFF (BLOX_AI_TRUST_XFF=1) ONLY because we're in
    # front of it; if the client could spoof XFF, the per-IP rate
    # limiter would be defeated.
    proxy_set_header X-Real-IP \$remote_addr;
    proxy_set_header X-Forwarded-For \$remote_addr;
    proxy_set_header X-Forwarded-Proto \$scheme;
    proxy_set_header X-Forwarded-Host \$host;
    proxy_set_header Host \$host;

    # Conservative client_max_body_size — anonymized transcripts cap
    # out under ~256 KB per upload (schema enforces). 1 MB gives 4x
    # headroom for future schema growth.
    client_max_body_size 1m;

    # POST /transcripts is the only real endpoint; GET /health is a
    # cheap probe. Pass everything to the FastAPI app.
    location / {
        proxy_pass http://127.0.0.1:${HOST_BIND_PORT};
        proxy_http_version 1.1;
        proxy_read_timeout 30s;
        proxy_connect_timeout 5s;
        proxy_send_timeout 10s;
        # Defense in depth: nginx-level limit_req would go here if we
        # see abuse. App-level rate limit (50/IP/day) is the first
        # line.
    }

    # Drop anything that looks like a probe for unrelated paths fast.
    location ~ /\\.(env|git) { return 404; }
}
EOF
)

  # Re-run safety: once certbot has issued a cert for this domain, it
  # has ALREADY edited this file to add the HTTPS server block +
  # listen 443 ssl directives. Rewriting from our HTTP-only template
  # would silently erase those edits. The next configure_ssl call only
  # runs `certbot renew` (idempotent — won't re-add the HTTPS block
  # if the cert isn't expiring) so HTTPS would break until manual
  # intervention. Preserve certbot's edits when a cert exists.
  local cert_exists=0
  if [ "$DRY_RUN" = 0 ] \
     && command -v certbot >/dev/null 2>&1 \
     && certbot certificates 2>/dev/null | grep -q "Domains: ${DOMAIN}"; then
    cert_exists=1
  fi

  if [ "$cert_exists" = 1 ] && [ -f "$NGINX_SITE_AVAILABLE" ]; then
    info "nginx site exists + cert issued — preserving certbot's edits (re-run)"
  elif [ -f "$NGINX_SITE_AVAILABLE" ] && diff -q <(echo "$desired") "$NGINX_SITE_AVAILABLE" >/dev/null 2>&1; then
    info "nginx site already at desired state — no rewrite"
  else
    if would "write $NGINX_SITE_AVAILABLE"; then return 0; fi
    echo "$desired" > "$NGINX_SITE_AVAILABLE"
    ok "Wrote $NGINX_SITE_AVAILABLE"
  fi

  if [ ! -L "$NGINX_SITE_ENABLED" ]; then
    if would "ln -sf $NGINX_SITE_AVAILABLE $NGINX_SITE_ENABLED"; then return 0; fi
    ln -sf "$NGINX_SITE_AVAILABLE" "$NGINX_SITE_ENABLED"
    ok "Enabled site (symlink)"
  fi

  if [ "$DRY_RUN" = 0 ]; then
    if ! nginx -t 2>/dev/null; then
      err "nginx -t failed after writing our site. Showing the test output:"
      nginx -t || true
      err "Our site only — REMOVING the symlink so your other sites keep working:"
      rm -f "$NGINX_SITE_ENABLED"
      exit 1
    fi
    systemctl reload nginx
    ok "nginx reloaded (gentle — no restart, your other vhosts unaffected)"
  fi
}

# ────────────────────────────────────────────────────────────────────
# SSL via Let's Encrypt. certbot --nginx mode mutates our site file
# to add HTTPS server block + cert paths. Idempotent: --keep tells it
# not to re-issue if the cert is still valid.
# ────────────────────────────────────────────────────────────────────

configure_ssl() {
  section "6. SSL cert (Let's Encrypt)"
  if would "certbot --nginx -d $DOMAIN -m $SSL_EMAIL"; then return 0; fi

  if certbot certificates 2>/dev/null | grep -q "Domains: ${DOMAIN}"; then
    info "Cert for $DOMAIN already issued — checking renewal only"
    certbot renew --quiet --no-self-upgrade || warn "certbot renew exited non-zero"
  else
    info "Requesting cert for $DOMAIN..."
    if ! certbot --nginx \
        -d "$DOMAIN" \
        -m "$SSL_EMAIL" \
        --agree-tos \
        --no-eff-email \
        --redirect \
        --non-interactive; then
      err "certbot failed. Common causes:"
      err "  - $DOMAIN doesn't resolve to this server's public IP yet (DNS lag)"
      err "  - Port 80 isn't reachable from the internet (check ufw + cloud-provider firewall)"
      err "  - Let's Encrypt rate limit hit (5 failures/hour; wait an hour)"
      err "After fixing: re-run \`sudo bash install.sh\`. The container is already up."
      return 1
    fi
    ok "Cert issued + nginx redirected to https"
  fi

  systemctl enable certbot.timer >/dev/null 2>&1 || true
  systemctl start  certbot.timer >/dev/null 2>&1 || true
  ok "Auto-renewal timer enabled"
}

# ────────────────────────────────────────────────────────────────────
# Final smoke test — hit the public URL via DNS just like a phone would
# ────────────────────────────────────────────────────────────────────

smoke_test() {
  section "7. smoke test"
  if [ "$DRY_RUN" = 1 ]; then
    info "[dry-run] would curl https://${DOMAIN}/health"
    return 0
  fi

  info "Probing https://${DOMAIN}/health (from this host's network egress)..."
  if curl -sSf --max-time 10 "https://${DOMAIN}/health" 2>/dev/null | grep -q '"ok":true'; then
    ok "https://${DOMAIN}/health → 200 {ok:true}"
  else
    warn "Public probe failed from this host."
    warn "  Try from your laptop: curl https://${DOMAIN}/health"
    warn "  If that also fails: check the DNS A record + cloud-provider firewall."
  fi

  # Admin endpoint: 401 without token, 200 with it. Probes locally so
  # the test runs even before DNS settles.
  info "Probing admin endpoint auth (local-only)..."
  local code401
  code401=$(curl -sS -o /dev/null -w "%{http_code}" --max-time 5 \
             "http://127.0.0.1:${HOST_BIND_PORT}/admin/issues" 2>/dev/null || echo "")
  if [ "$code401" = "401" ] || [ "$code401" = "503" ]; then
    ok "admin endpoint properly rejects no-token (HTTP ${code401})"
  else
    warn "admin endpoint returned HTTP ${code401} for no-token — expected 401 or 503"
  fi
}

# ────────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────────

print_summary() {
  echo
  c_green "══════════════════════════════════════════════════════════════"
  c_green " Blox AI intake server — installed"
  c_green "══════════════════════════════════════════════════════════════"
  echo
  echo "  Domain:        https://${DOMAIN}/"
  echo "  Health:        https://${DOMAIN}/health"
  echo "  Upload path:   POST https://${DOMAIN}/transcripts (no auth — phones)"
  echo "  Admin UI:      https://${DOMAIN}/admin/  (bearer-token auth)"
  echo "  Admin API:     /admin/issues, /admin/issues/{id}, etc."
  echo
  echo "  Container:     ${SERVICE_NAME} (bound 127.0.0.1:${HOST_BIND_PORT})"
  echo "  Storage:       ${STORAGE_DIR}/"
  echo "  DB:            ${DB_DIR}/issues.sqlite (WAL mode)"
  echo "  Logs:          docker logs ${SERVICE_NAME}"
  echo "  Nginx site:    ${NGINX_SITE_AVAILABLE}"
  echo "  Env file:      ${ENV_FILE} (chmod 0640; contains admin token)"
  echo
  if [ -n "${GENERATED_TOKEN:-}" ]; then
    c_yellow "  ⚠  Admin token (NEW, generated this run — copy it now):"
    echo "      ${GENERATED_TOKEN}"
    echo "      Paste this into the admin UI on first load."
    echo "      Rotate later by editing ${ENV_FILE} and restarting the container."
    echo
  else
    echo "  Admin token:   (kept existing value from ${ENV_FILE} — not printed)"
    echo "      To rotate: edit ${ENV_FILE}, then \`docker compose restart\`."
    echo
  fi
  echo "  Operations:"
  echo "    Update image:   cd ${INSTALL_DIR} && git pull && sudo bash install.sh"
  echo "    Restart only:   cd ${INSTALL_DIR} && sudo docker compose restart"
  echo "    Tail logs:      docker logs -f ${SERVICE_NAME}"
  echo "    Disk usage:     du -sh ${STORAGE_DIR} ${DB_DIR}"
  echo "    Cert renewal:   systemctl status certbot.timer"
  echo
  echo "  Coexistence — what we touched / what we left alone:"
  echo "    We OWN (created/managed by this install):"
  echo "      - Docker container: ${SERVICE_NAME}"
  echo "      - Docker image:     blox-ai-intake:local"
  echo "      - Storage:          ${STORAGE_DIR}, ${DB_DIR}, ${LOG_DIR}"
  echo "      - nginx site file:  ${NGINX_SITE_AVAILABLE} (+ matching enabled symlink)"
  echo "      - .env file:        ${ENV_FILE}"
  echo "      - UFW rules added:  allow 80/tcp, allow 443/tcp, deny ${HOST_BIND_PORT}/tcp"
  echo "                          (each rule added ONLY if not already present;"
  echo "                          ${HOST_BIND_PORT}-deny skipped if something else uses that port)"
  echo "    We NEVER touched:"
  echo "      - Any other Docker container (compose only knows about ours)"
  echo "      - Any other nginx site file or sites-enabled symlink"
  echo "      - The Docker daemon (no restart)"
  echo "      - nginx.conf or any global nginx state (only \`systemctl reload\`)"
  echo "      - Any existing UFW rule (no resets, no deletions, no default-policy change)"
  echo "      - Anything under /var/lib outside /var/lib/blox-ai-intake/"
  echo "      - Anything under /var/log outside ${LOG_DIR}"
  echo "      - System certbot.timer is enabled (idempotent — was likely already on)"
  echo
  echo "  Phone-app side: app already hardcodes the upload URL to"
  echo "    https://ai-training.fx.land/transcripts (uploadTranscriptUrl.ts)."
  echo "  If your domain differs, update that constant in apps/box/src/utils/"
  echo "  and ship a new app build."
}

main() {
  # ── Read-only phase (no host mutation) ──────────────────────────────
  require_root
  require_distro
  prompt_domain_email
  check_dns_or_warn
  show_host_state             # prints current ufw / ports / containers / nginx
  coexistence_preflight       # detects conflicts; sets CONFLICTS_FOUND
  confirm_or_abort_on_conflicts  # operator decides: continue or abort

  # ── Mutating phase (everything below changes host state) ────────────
  install_prereqs
  setup_storage
  setup_admin_token
  setup_firewall              # respects PORT_8090_PRE_USE
  build_and_start
  write_nginx_site            # preserves certbot's HTTPS edits on re-runs
  configure_ssl
  smoke_test
  print_summary
}

main "$@"
