#!/usr/bin/env bash
# Updates an existing installation to a new release version.
#
# IMPORTANT: this release (1.0.0) is the baseline install -- there are no
# database migrations to apply yet (database/sqlserver/migrations/ is a
# placeholder, see its README.md). A future release's update procedure would
# add a step here to run pending migration scripts against the existing
# database. For now, db-init's install scripts are re-run, which is safe
# because every install script uses IF NOT EXISTS / IF OBJECT_ID IS NULL
# guards -- they will not recreate or touch existing data.
#
# Steps: backup -> load new images -> recreate app containers -> verify.
# The sqlserver container and its data volume are NOT recreated.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$RELEASE_ROOT/.env"
COMPOSE_FILE="$RELEASE_ROOT/compose/docker-compose.yml"

# shellcheck disable=SC1090
set -a; source "$ENV_FILE"; set +a

# Explicit -p so Compose's project name doesn't fall back to the basename of
# the directory containing docker-compose.yml ("compose").
COMPOSE_ARGS=(--env-file "$ENV_FILE" -f "$COMPOSE_FILE" -p "${PROJECT_NAME:-pfms}")

echo "=== Patient Feedback System - Offline Update ==="
echo ""
echo "This will:"
echo "  1. Back up the current database."
echo "  2. Load the new release's Docker images."
echo "  3. Recreate db-init, backend, and frontend containers."
echo "  4. Leave the SQL Server container and its data untouched."
echo ""
read -r -p "Continue? Type YES to proceed: " confirm
if [ "$confirm" != "YES" ]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "[1/4] Backing up the database ..."
"$SCRIPT_DIR/backup_database.sh"

echo ""
echo "[2/4] Loading new images ..."
"$SCRIPT_DIR/load_images.sh"

echo ""
echo "[3/4] Recreating db-init, backend, and frontend (sqlserver is untouched) ..."
docker compose "${COMPOSE_ARGS[@]}" up -d --force-recreate db-init backend frontend

echo ""
echo "[4/4] Verifying ..."
"$SCRIPT_DIR/verify_installation.sh" || {
    echo ""
    echo "Verification reported failures. To roll back:"
    echo "  1. Reload the previous release's images (docker load -i <old .tar files>)"
    echo "  2. Restore the pre-update backup: scripts/restore_database.sh <backup file printed above>"
    echo "  3. Recreate containers: docker compose --env-file .env -f compose/docker-compose.yml -p \${PROJECT_NAME:-pfms} up -d --force-recreate"
    exit 1
}

echo ""
echo "=== Update complete ==="
