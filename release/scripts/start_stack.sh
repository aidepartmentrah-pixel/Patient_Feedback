#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
docker compose --env-file "$RELEASE_ROOT/.env" -f "$RELEASE_ROOT/compose/docker-compose.yml" up -d
echo "Stack started. Run scripts/verify_installation.sh to confirm health."
