#!/usr/bin/env bash
# Loads the release's Docker images from local tar files. Uses only local
# files -- never contacts the Internet or a registry.
#
# The SQL Server image (mcr.microsoft.com/mssql/server:2022-latest) is NOT
# included here -- it is expected to already be present on the offline
# server via the Offline Debian Server Kit. This script checks for it and
# fails clearly if it's missing, rather than silently proceeding.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGES_DIR="$RELEASE_ROOT/docker-images"

echo "=== Loading Patient Feedback release images ==="

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker is not installed or not on PATH. Install it from the"
    echo "       Offline Debian Server Kit before running this script."
    exit 1
fi

if ! docker image inspect mcr.microsoft.com/mssql/server:2022-latest >/dev/null 2>&1; then
    echo "ERROR: SQL Server image (mcr.microsoft.com/mssql/server:2022-latest)"
    echo "       is not present on this server."
    echo "       This image ships with the Offline Debian Server Kit, not this"
    echo "       application release. Load it from the Server Kit first, then"
    echo "       re-run this script."
    exit 1
fi
echo "  [OK] SQL Server image already present."

for tar_file in backend frontend db-init; do
    path="$IMAGES_DIR/${tar_file}.tar"
    if [ ! -f "$path" ]; then
        echo "ERROR: missing $path"
        exit 1
    fi
    echo "  Loading ${tar_file}.tar ..."
    docker load -i "$path"
done

echo ""
echo "=== All images loaded successfully ==="
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" | grep -E "rah-pfms|REPOSITORY" || true
