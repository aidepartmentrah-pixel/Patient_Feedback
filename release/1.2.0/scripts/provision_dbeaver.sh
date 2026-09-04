#!/usr/bin/env bash
# Auto-provisions (writes/merges) a DBeaver connection for this deployment's
# database into the operator's own DBeaver config, so they don't have to
# follow DBEAVER_GUIDE.md by hand every time.
#
# IMPORTANT -- schema confidence note: DBeaver's connection format
# (data-sources.json) is internal/undocumented and can vary between
# versions. This script's field names and SQL Server driver ID ("sqlserver")
# reflect DBeaver's modern connection-config format as of the CE 6.x+ series
# (confirmed CE 26.1.2 is what's actually installed in the lab, per
# 8 RAH-OIP Lab Environment Reference.md) but have NOT been verified against
# a real running DBeaver instance from this engineering session. Verify once
# against the real file (create one connection by hand in DBeaver's UI, then
# compare) and correct this script if the schema differs -- see the plan's
# own note: "confirm/adjust the exact file format against the real thing
# once Phase C starts."
#
# Safety posture given that uncertainty: this script ONLY touches the config
# file through `jq` (a real JSON tool) -- never a blind text/sed merge into a
# file that may already hold the operator's own unrelated connections. If
# `jq` isn't available, or anything about the existing file looks
# unexpected, it skips cleanly and points at DBEAVER_GUIDE.md instead of
# risking corruption. A backup of the file is always taken before writing.
#
# No password is stored (matches standard practice for a hospital database
# and avoids depending on DBeaver's machine-specific password-encryption
# internals) -- the operator enters it once per DBeaver session.

set -uo pipefail  # not -e: this is a best-effort convenience step, never
                   # allowed to fail the install/update it's called from

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"
set +e; set -uo pipefail
load_env

DBEAVER_CONFIG_DIR="$HOME/.local/share/DBeaverData/workspace6/General/.dbeaver"
DBEAVER_DATA_SOURCES="$DBEAVER_CONFIG_DIR/data-sources.json"
CONNECTION_ID="pfms-${PROJECT_NAME:-pfms}"

echo "=== DBeaver connection auto-provisioning ==="

if ! command -v jq >/dev/null 2>&1; then
    echo "  SKIPPED: 'jq' is not installed on this server -- required to safely"
    echo "  edit DBeaver's config without risking the operator's other saved"
    echo "  connections. Install jq and re-run, or add the connection by hand"
    echo "  via documentation/DBEAVER_GUIDE.md."
    exit 0
fi

if [ ! -d "$HOME/.local/share/DBeaverData" ]; then
    echo "  SKIPPED: DBeaver does not appear to have been run yet on this"
    echo "  machine ($HOME/.local/share/DBeaverData not found). Run DBeaver"
    echo "  once first (so it creates its own config), then re-run this"
    echo "  script, or add the connection by hand via"
    echo "  documentation/DBEAVER_GUIDE.md."
    exit 0
fi

mkdir -p "$DBEAVER_CONFIG_DIR"

if [ -f "$DBEAVER_DATA_SOURCES" ]; then
    if ! jq empty "$DBEAVER_DATA_SOURCES" >/dev/null 2>&1; then
        echo "  SKIPPED: $DBEAVER_DATA_SOURCES exists but is not valid JSON --"
        echo "  refusing to touch it. Add the connection by hand via"
        echo "  documentation/DBEAVER_GUIDE.md."
        exit 0
    fi
    cp "$DBEAVER_DATA_SOURCES" "$DBEAVER_DATA_SOURCES.bak-$(date -u +%Y%m%dT%H%M%SZ)"
else
    echo '{"folders":{},"connections":{}}' > "$DBEAVER_DATA_SOURCES"
fi

CONNECTION_JSON=$(jq -n \
    --arg name "PFMS - ${DATABASE_NAME} (${PROJECT_NAME:-pfms})" \
    --arg host "localhost" \
    --arg port "${SQLSERVER_HOST_PORT:-1433}" \
    --arg database "${DATABASE_NAME}" \
    --arg username "${DATABASE_USER}" \
    '{
        "provider": "sqlserver",
        "driver": "sqlserver",
        "name": $name,
        "save-password": false,
        "configuration": {
            "host": $host,
            "port": $port,
            "database": $database,
            "configurationType": "MANUAL",
            "type": "dev",
            "auth-model": "native",
            "provider-properties": {
                "sqlserver.encrypt": "false",
                "sqlserver.trustServerCertificate": "true"
            },
            "credentials": {
                "user-name": $username
            }
        }
    }')

if jq --argjson conn "$CONNECTION_JSON" --arg id "$CONNECTION_ID" \
    '.connections[$id] = $conn' \
    "$DBEAVER_DATA_SOURCES" > "$DBEAVER_DATA_SOURCES.tmp" \
    && jq empty "$DBEAVER_DATA_SOURCES.tmp" >/dev/null 2>&1; then
    mv "$DBEAVER_DATA_SOURCES.tmp" "$DBEAVER_DATA_SOURCES"
    echo "  Connection '$CONNECTION_ID' written to $DBEAVER_DATA_SOURCES"
    echo "  (no password stored -- enter DATABASE_PASSWORD from .env once per"
    echo "  DBeaver session). Restart DBeaver if it's currently running."
else
    rm -f "$DBEAVER_DATA_SOURCES.tmp"
    echo "  SKIPPED: failed to merge the connection safely -- left the"
    echo "  existing file untouched (backup was still taken). Add the"
    echo "  connection by hand via documentation/DBEAVER_GUIDE.md."
    exit 0
fi
