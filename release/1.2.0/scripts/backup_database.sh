#!/usr/bin/env bash
# Backs up the IncidentManager database to a timestamped .bak file.
#
# Two ways this can be invoked:
#   - Manually, by a human on the real host: the backup lands in
#     $LIVE_ROOT/backups (the sqlserver container's own backup volume
#     mount, ../backups relative to the live Compose file), per the org's
#     backup standard.
#   - By the RAH Offline Installation Platform: Platform passes the real,
#     durable artifact location explicitly via RAH_BACKUP_OUTPUT_PATH, a
#     location outside this app's own replaceable deployment (see RAH
#     Packager & Platform Integration Guide, §11a). When set, the backup
#     is additionally copied there via `docker compose cp`, not assumed
#     reachable through the bind mount alone -- a live deployment's own
#     Compose convention can differ from whatever this script's author
#     expected, and Platform needs the artifact at the exact path it named.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"
load_env

CONTAINER="${PROJECT_NAME:-pfms}-sqlserver"
DB_NAME="${DB_DATABASE:-IncidentManager}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_FILE="${DB_NAME}_${TIMESTAMP}.bak"

echo "=== Backing up ${DB_NAME} ==="
mkdir -p "$LIVE_ROOT/backups"

# Docker auto-creates ../backups as root-owned; SQL Server's own container
# process runs as a fixed non-root user (UID 10001/mssql) and can't write
# into a root-owned directory. Fixed from inside the container, not on the
# host -- the host-side resolved path isn't guaranteed stable across a
# Compose file's own revisions (see Integration Guide §11b).
compose exec -T -u root sqlserver chown -R 10001:0 /var/opt/mssql/backup

# MSYS_NO_PATHCONV scoped to this one command (no-op on the real Debian
# target): prevents Git-Bash-on-Windows from mangling the absolute
# /opt/mssql-tools18/... container path when tested on a dev machine.
MSYS_NO_PATHCONV=1 docker exec "$CONTAINER" /opt/mssql-tools18/bin/sqlcmd \
    -S localhost -U sa -P "$MSSQL_SA_PASSWORD" -C \
    -Q "BACKUP DATABASE [${DB_NAME}] TO DISK = N'/var/opt/mssql/backup/${BACKUP_FILE}' WITH FORMAT, INIT, STATS=10"

if [ -f "$LIVE_ROOT/backups/${BACKUP_FILE}" ]; then
    echo ""
    echo "=== Backup complete ==="
    echo "File: $LIVE_ROOT/backups/${BACKUP_FILE}"
    ls -lh "$LIVE_ROOT/backups/${BACKUP_FILE}"
else
    echo "ERROR: expected backup file not found on host -- check the volume"
    echo "       mount in $COMPOSE_FILE (sqlserver service: ../backups)."
    exit 1
fi

# Platform's own backup contract: when it's the caller, it names the real,
# durable destination explicitly via RAH_BACKUP_OUTPUT_PATH -- copy the
# artifact there too via `docker compose cp` against the container's own
# well-known internal path, not a computed host path (Integration Guide
# §11a). A manual run (this variable unset) is unaffected.
if [ -n "${RAH_BACKUP_OUTPUT_PATH:-}" ]; then
    echo ""
    echo "Copying backup to Platform's declared output path ..."
    # docker compose cp requires the destination directory to already exist
    # on the host -- it does not create parent directories itself. Found
    # live during Pass 5 (HCAT) Phase 5, 2026-08-26: PLT-BACKUP-003,
    # "invalid output path: directory ... does not exist" -- confirmed
    # against HCopilot's own already-proven backup_database.sh, which
    # already does this same mkdir -p first.
    mkdir -p "$(dirname "$RAH_BACKUP_OUTPUT_PATH")"
    compose cp "sqlserver:/var/opt/mssql/backup/${BACKUP_FILE}" "$RAH_BACKUP_OUTPUT_PATH"
    echo "  Copied to $RAH_BACKUP_OUTPUT_PATH"
fi
