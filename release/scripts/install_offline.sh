#!/usr/bin/env bash
# First-time installation on the offline Debian server. Uses only local
# files -- never contacts the Internet.
#
# What this does:
#   1. Verifies Docker is available.
#   2. Verifies (or helps create) the .env file.
#   3. Loads the release's Docker images.
#   4. Extracts the Speech-to-Text model asset (assets/whisper-model-medium.zip).
#   5. Verifies the organizational-unit/user provisioning artifact's checksum
#      (fails immediately if missing or corrupted -- this release must never
#      complete an install with zero organizational units and no accounts).
#   6. Starts the full stack (SQL Server -> db-init [schema + provisioning] ->
#      backend -> frontend).
#   7. Waits for the backend to report healthy and prints the result.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$RELEASE_ROOT/compose/docker-compose.yml"
ENV_FILE="$RELEASE_ROOT/.env"

echo "=== Patient Feedback System - Offline Installation ==="
echo "Release root: $RELEASE_ROOT"
echo ""

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker is not installed. Install it from the Offline Debian"
    echo "       Server Kit before running this script."
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "ERROR: the 'docker compose' plugin is not available."
    exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
    echo "No .env file found at $ENV_FILE"
    echo "Creating one from .env.offline.template ..."
    cp "$RELEASE_ROOT/.env.offline.template" "$ENV_FILE"
    echo ""
    echo "ACTION REQUIRED: edit $ENV_FILE now and fill in every __SET_ME__"
    echo "value (SQL Server password, edition, app version). Then run this"
    echo "script again."
    exit 1
fi

if grep -q "__SET_ME__" "$ENV_FILE"; then
    echo "ERROR: $ENV_FILE still contains __SET_ME__ placeholders."
    echo "       Edit it and fill in every required value before continuing."
    exit 1
fi

echo "[1/5] Loading Docker images ..."
"$SCRIPT_DIR/load_images.sh"

echo ""
echo "[2/5] Extracting the Speech-to-Text model asset ..."
MODEL_ZIP="$RELEASE_ROOT/assets/whisper-model-medium.zip"
MODEL_DIR="$RELEASE_ROOT/assets/whisper-model-medium"
if [ ! -f "$MODEL_ZIP" ]; then
    echo "ERROR: $MODEL_ZIP not found. The release package is incomplete -- the"
    echo "       Speech-to-Text model asset is required; the backend has no"
    echo "       other way to obtain it on an offline server."
    exit 1
fi
if [ -d "$MODEL_DIR" ] && [ -n "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    echo "  Already extracted at $MODEL_DIR, skipping."
else
    mkdir -p "$RELEASE_ROOT/assets"
    unzip -q -o "$MODEL_ZIP" -d "$RELEASE_ROOT/assets"
    echo "  Extracted to $MODEL_DIR"
fi

echo ""
echo "[3/6] Creating backup directory (for future updates) ..."
mkdir -p "$RELEASE_ROOT/backups"

echo ""
echo "[4/6] Verifying the organizational-unit/user provisioning artifact ..."
PROVISION_DIR="$RELEASE_ROOT/database/sqlserver/seed"
PROVISION_JSON="$PROVISION_DIR/provisioning.v1.json"
PROVISION_SHA="$PROVISION_DIR/provisioning.v1.json.sha256"
if [ ! -f "$PROVISION_JSON" ] || [ ! -f "$PROVISION_SHA" ]; then
    echo "ERROR: $PROVISION_JSON or its .sha256 checksum file is missing."
    echo "       This release package is incomplete -- a fresh install must"
    echo "       never complete with zero organizational units and no usable"
    echo "       accounts. Installation aborted."
    exit 1
fi
if ! (cd "$PROVISION_DIR" && sha256sum -c provisioning.v1.json.sha256 >/dev/null); then
    echo "ERROR: provisioning.v1.json failed checksum verification."
    echo "       The release bundle may be corrupted or tampered with in"
    echo "       transit. Installation aborted -- do not proceed with a"
    echo "       re-copy of this release before investigating."
    exit 1
fi
echo "  Checksum OK."

echo ""
echo "[5/6] Starting the stack (schema install + organizational/user provisioning"
echo "      run automatically as part of db-init) ..."
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d

echo "  Waiting for db-init (schema install + provisioning) to finish ..."
for _ in $(seq 1 60); do
    db_init_status="$(docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" ps -a --format '{{.Name}} {{.State}} {{.ExitCode}}' db-init 2>/dev/null || true)"
    case "$db_init_status" in
        *"exited 0"*) echo "  db-init completed successfully."; break ;;
        *"exited"*)
            echo ""
            echo "ERROR: db-init failed (schema install or provisioning). Logs:"
            docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" logs db-init
            echo ""
            echo "Installation aborted. Fix the issue above before re-running."
            exit 1
            ;;
    esac
    sleep 5
done

echo ""
echo "[6/6] Waiting for the backend to become healthy ..."

attempt=0
max_attempts=30  # ~7.5 minutes at 15s intervals -- the model is already
                 # local (no download), so this should be quick.
while [ "$attempt" -lt "$max_attempts" ]; do
    status="$(docker inspect --format='{{.State.Health.Status}}' "$(docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" ps -q backend)" 2>/dev/null || echo "starting")"
    if [ "$status" = "healthy" ]; then
        echo ""
        echo "=== Installation complete. Backend is healthy. ==="
        break
    fi
    attempt=$((attempt + 1))
    sleep 15
done

if [ "$status" != "healthy" ]; then
    echo ""
    echo "WARNING: backend did not report healthy within the wait window."
    echo "         Check logs with: scripts/show_logs.sh backend"
    exit 1
fi

BACKEND_PORT="$(grep -E '^BACKEND_HOST_PORT=' "$ENV_FILE" | cut -d= -f2 || echo 8100)"
FRONTEND_PORT="$(grep -E '^FRONTEND_HOST_PORT=' "$ENV_FILE" | cut -d= -f2 || echo 8101)"

echo ""
echo "Application URL:  http://<server-ip>:${FRONTEND_PORT}"
echo "Backend API docs: http://<server-ip>:${BACKEND_PORT}/docs"
echo ""
echo "Run scripts/verify_installation.sh for a full validation pass."
