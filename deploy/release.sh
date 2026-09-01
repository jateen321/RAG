#!/usr/bin/env bash
# Deploy one immutable backend/frontend image pair on the stateful VM.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 BACKEND_IMAGE FRONTEND_IMAGE RELEASE_ID" >&2
  exit 64
fi

readonly APP_DIR="/opt/gyaan-sarthi"
readonly RELEASE_DIR="$APP_DIR/releases"
readonly CURRENT_RELEASE="$RELEASE_DIR/current.env"
readonly LOCK_FILE="/var/lock/gyaan-sarthi-deploy.lock"
readonly BACKEND_IMAGE="$1"
readonly FRONTEND_IMAGE="$2"
readonly RELEASE_ID="$3"

case "$RELEASE_ID" in
  *[!a-zA-Z0-9._-]*|'')
    echo "Release ID contains unsupported characters." >&2
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
  if [[ ! -f "$CURRENT_RELEASE" ]]; then
    echo "Deployment failed and no prior release is recorded." >&2
    exit "$failed_status"
  fi
  echo "Deployment failed; restoring the previous healthy image pair." >&2
  run_compose "$CURRENT_RELEASE" pull backend frontend
  run_compose "$CURRENT_RELEASE" up -d --no-build --wait --wait-timeout 120
  exit "$failed_status"
}

candidate="$RELEASE_DIR/$RELEASE_ID.env"
write_release "$candidate"

if ! run_compose "$candidate" pull backend frontend; then
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

cp "$candidate" "$CURRENT_RELEASE"
echo "Release $RELEASE_ID is healthy."
