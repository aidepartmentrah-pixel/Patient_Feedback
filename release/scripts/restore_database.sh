#!/usr/bin/env bash
# Restores IncidentManager from a .bak file produced by backup_database.sh.
# Usage: restore_database.sh <backup_filename.bak>
#
# WITH REPLACE means this overwrites the current database contents. This is
# a destructive, deliberate recovery action -- it prompts for confirmation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$RELEASE_ROOT/.env"

if [ $# -ne 1 ]; then
    echo "Usage: $0 <backup_filename.bak>"
    echo ""
    echo "Available backups in $RELEASE_ROOT/backups:"
    ls -1 "$RELEASE_ROOT/backups" 2>/dev/null || echo "  (none found)"
    exit 1
fi

BACKUP_FILE="$1"

if [ ! -f "$RELEASE_ROOT/backups/$BACKUP_FILE" ]; then
    echo "ERROR: $RELEASE_ROOT/backups/$BACKUP_FILE not found."
    exit 1
fi

# shellcheck disable=SC1090
set -a; source "$ENV_FILE"; set +a

CONTAINER="${PROJECT_NAME:-pfms}-sqlserver"
DB_NAME="${DB_DATABASE:-IncidentManager}"

echo "=== RESTORE ${DB_NAME} from ${BACKUP_FILE} ==="
echo "WARNING: this REPLACES all current data in ${DB_NAME}. This cannot be"
echo "         undone unless you have a separate, more recent backup."
read -r -p "Type YES to continue: " confirm
if [ "$confirm" != "YES" ]; then
    echo "Aborted."
    exit 1
fi

echo "Stopping the backend so it doesn't hold connections open during restore ..."
docker stop "${PROJECT_NAME:-pfms}-backend" >/dev/null 2>&1 || true

# MSYS_NO_PATHCONV scoped to this one command (no-op on the real Debian
# target): prevents Git-Bash-on-Windows from mangling the absolute
# /opt/mssql-tools18/... container path when tested on a dev machine.
MSYS_NO_PATHCONV=1 docker exec "$CONTAINER" /opt/mssql-tools18/bin/sqlcmd \
    -S localhost -U sa -P "$MSSQL_SA_PASSWORD" -C \
    -Q "ALTER DATABASE [${DB_NAME}] SET SINGLE_USER WITH ROLLBACK IMMEDIATE; \
        RESTORE DATABASE [${DB_NAME}] FROM DISK = N'/var/opt/mssql/backup/${BACKUP_FILE}' WITH REPLACE, STATS=10; \
        ALTER DATABASE [${DB_NAME}] SET MULTI_USER;"

echo ""
echo "Restarting the backend ..."
docker start "${PROJECT_NAME:-pfms}-backend" >/dev/null 2>&1 || true

echo ""
echo "=== Restore complete. Run scripts/verify_installation.sh to confirm. ==="
