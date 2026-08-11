#!/usr/bin/env bash
# Backs up the IncidentManager database to a timestamped .bak file, written
# through the sqlserver container's own backup volume mount (../backups on
# the host -- outside the container, per the org's backup standard).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$RELEASE_ROOT/.env"
COMPOSE_FILE="$RELEASE_ROOT/compose/docker-compose.yml"

# shellcheck disable=SC1090
set -a; source "$ENV_FILE"; set +a

CONTAINER="${PROJECT_NAME:-pfms}-sqlserver"
DB_NAME="${DB_DATABASE:-IncidentManager}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_FILE="${DB_NAME}_${TIMESTAMP}.bak"

echo "=== Backing up ${DB_NAME} ==="
mkdir -p "$RELEASE_ROOT/backups"

# MSYS_NO_PATHCONV scoped to this one command (no-op on the real Debian
# target): prevents Git-Bash-on-Windows from mangling the absolute
# /opt/mssql-tools18/... container path when tested on a dev machine.
MSYS_NO_PATHCONV=1 docker exec "$CONTAINER" /opt/mssql-tools18/bin/sqlcmd \
    -S localhost -U sa -P "$MSSQL_SA_PASSWORD" -C \
    -Q "BACKUP DATABASE [${DB_NAME}] TO DISK = N'/var/opt/mssql/backup/${BACKUP_FILE}' WITH FORMAT, INIT, STATS=10"

if [ -f "$RELEASE_ROOT/backups/${BACKUP_FILE}" ]; then
    echo ""
    echo "=== Backup complete ==="
    echo "File: $RELEASE_ROOT/backups/${BACKUP_FILE}"
    ls -lh "$RELEASE_ROOT/backups/${BACKUP_FILE}"
else
    echo "ERROR: expected backup file not found on host -- check the volume"
    echo "       mount in $COMPOSE_FILE (sqlserver service: ../backups)."
    exit 1
fi
