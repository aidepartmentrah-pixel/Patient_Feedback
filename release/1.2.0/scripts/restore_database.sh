#!/usr/bin/env bash
# Restores IncidentManager from a .bak file produced by backup_database.sh.
# Usage: restore_database.sh <backup_filename.bak>
#
# WITH REPLACE means this overwrites the current database contents. This is
# a destructive, deliberate recovery action -- it prompts for confirmation
# on a manual run.
#
# The RAH Offline Installation Platform, not only a human operator, may
# invoke this script. When it does, it passes the real source artifact
# location via RAH_BACKUP_SOURCE_PATH (Integration Guide §11a) rather than
# a positional argument, and cannot answer an interactive prompt (it runs
# scripts non-interactively -- a script that blocks on `read` here would
# simply hang forever instead of failing loudly).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

if [ -n "${RAH_BACKUP_SOURCE_PATH:-}" ]; then
    # Platform-driven restore: the source artifact may not already be
    # visible under $LIVE_ROOT/backups at all -- don't assume it is.
    BACKUP_FILE="$(basename "$RAH_BACKUP_SOURCE_PATH")"
    SOURCE_PATH="$RAH_BACKUP_SOURCE_PATH"
    if [ ! -f "$SOURCE_PATH" ]; then
        echo "ERROR: RAH_BACKUP_SOURCE_PATH ($SOURCE_PATH) not found."
        exit 1
    fi
elif [ $# -eq 1 ]; then
    BACKUP_FILE="$1"
    SOURCE_PATH="$LIVE_ROOT/backups/$BACKUP_FILE"
    if [ ! -f "$SOURCE_PATH" ]; then
        echo "ERROR: $SOURCE_PATH not found."
        exit 1
    fi
else
    echo "Usage: $0 <backup_filename.bak>"
    echo ""
    echo "Available backups in $LIVE_ROOT/backups:"
    ls -1 "$LIVE_ROOT/backups" 2>/dev/null || echo "  (none found)"
    exit 1
fi

load_env

CONTAINER="${PROJECT_NAME:-pfms}-sqlserver"
DB_NAME="${DATABASE_NAME}"

echo "=== RESTORE ${DB_NAME} from ${BACKUP_FILE} ==="
echo "WARNING: this REPLACES all current data in ${DB_NAME}. This cannot be"
echo "         undone unless you have a separate, more recent backup."
if [ -n "${RAH_BACKUP_SOURCE_PATH:-}" ]; then
    echo "Platform-driven restore -- skipping the interactive confirmation"
    echo "(non-interactive invocation, per Integration Guide's own convention)."
else
    read -r -p "Type YES to continue: " confirm
    if [ "$confirm" != "YES" ]; then
        echo "Aborted."
        exit 1
    fi
fi

# Docker auto-creates ../backups as root-owned; SQL Server's own container
# process runs as a fixed non-root user (UID 10001/mssql). Fixed from
# inside the container, not on the host (Integration Guide §11b).
compose exec -T -u root sqlserver chown -R 10001:0 /var/opt/mssql/backup

# Ensure the container actually has the file at the path it expects --
# under a Platform-driven restore, the source lives wherever
# RAH_BACKUP_SOURCE_PATH pointed, not necessarily already inside the
# container's own bind-mounted backup directory. mkdir -p first -- same
# real gap found in backup_database.sh during Pass 5 (HCAT) Phase 5,
# 2026-08-26: a fresh deployment that's never run a local backup yet
# won't have $LIVE_ROOT/backups created at all.
if [ ! -f "$LIVE_ROOT/backups/$BACKUP_FILE" ]; then
    mkdir -p "$LIVE_ROOT/backups"
    cp "$SOURCE_PATH" "$LIVE_ROOT/backups/$BACKUP_FILE"
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
