#!/usr/bin/env bash
# Deploy one immutable backend/frontend/Caddy image set on the stateful VM.
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 BACKEND_IMAGE FRONTEND_IMAGE CADDY_IMAGE APP_DOMAIN RELEASE_ID" >&2
  exit 64
fi

readonly APP_DIR="/opt/gyaan-sarthi"
readonly RELEASE_DIR="$APP_DIR/releases"
readonly CURRENT_RELEASE="$RELEASE_DIR/current.env"
readonly LOCK_FILE="/var/lock/gyaan-sarthi-deploy.lock"
readonly BACKEND_IMAGE="$1"
readonly FRONTEND_IMAGE="$2"
readonly CADDY_IMAGE="$3"
readonly APP_DOMAIN="$4"
readonly RELEASE_ID="$5"

case "$RELEASE_ID" in
  *[!a-zA-Z0-9._-]*|'')
    echo "Release ID contains unsupported characters." >&2
    exit 64
    ;;
esac

case "$APP_DOMAIN" in
  *[!a-zA-Z0-9.-]*|''|.*|*.)
    echo "App domain contains unsupported characters." >&2
    exit 64
    ;;
esac

mkdir -p "$RELEASE_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another deployment is already running." >&2
  exit 75
fi

write_release() {
  local path="$1"
  umask 077
  {
    printf 'BACKEND_IMAGE=%q\n' "$BACKEND_IMAGE"
    printf 'FRONTEND_IMAGE=%q\n' "$FRONTEND_IMAGE"
    printf 'CADDY_IMAGE=%q\n' "$CADDY_IMAGE"
    printf 'RELEASE_ID=%q\n' "$RELEASE_ID"
  } > "$path"
}

run_compose() {
  # Pass dotenv files directly to Compose. They are configuration data, not
  # shell scripts; sourcing them would execute malformed/unquoted values.
  # The release file is last so its immutable image references override any
  # defaults from `.env.production`.
  docker compose --env-file "$APP_DIR/.env.production" \
    --env-file "$1" \
    -f "$APP_DIR/docker-compose.production.yml" "${@:2}"
}

rollback() {
  local failed_status="$1"
  local rollback_release
  if [[ ! -f "$CURRENT_RELEASE" ]]; then
    echo "Deployment failed and no prior release is recorded." >&2
    exit "$failed_status"
  fi
  # Releases created before Caddy was versioned have no CADDY_IMAGE entry.
  # Keep the validated candidate proxy while restoring their app images.
  rollback_release="$RELEASE_DIR/rollback-$RELEASE_ID.env"
  cp "$CURRENT_RELEASE" "$rollback_release"
  if ! grep -q '^CADDY_IMAGE=' "$rollback_release"; then
    printf 'CADDY_IMAGE=%q\n' "$CADDY_IMAGE" >> "$rollback_release"
  fi
  echo "Deployment failed; restoring the previous healthy image set." >&2
  run_compose "$rollback_release" pull backend frontend caddy
  run_compose "$rollback_release" up -d --no-build --wait --wait-timeout 120
  exit "$failed_status"
}

candidate="$RELEASE_DIR/$RELEASE_ID.env"
write_release "$candidate"

if ! run_compose "$candidate" pull backend frontend caddy; then
  rollback 1
fi
if ! run_compose "$candidate" up -d --no-build --wait --wait-timeout 120; then
  rollback 1
fi
# The backend port is intentionally private to the Compose network; check it
# from inside the backend container instead of assuming a host port mapping.
if ! run_compose "$candidate" exec -T backend python -c \
  'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8000/health")' >/dev/null; then
  rollback 1
fi
# Verify the browser-facing route through Caddy. ``--resolve`` keeps the check
# on this VM while preserving the production hostname for TLS and routing.
if ! curl --fail --silent --show-error --max-time 15 \
  --retry 5 --retry-connrefused --retry-delay 2 \
  --resolve "$APP_DOMAIN:443:127.0.0.1" \
  "https://$APP_DOMAIN/api/health" >/dev/null; then
  rollback 1
fi

cp "$candidate" "$CURRENT_RELEASE"
echo "Release $RELEASE_ID is healthy."
