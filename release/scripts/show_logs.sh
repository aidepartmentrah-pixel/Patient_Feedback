#!/usr/bin/env bash
# Usage: show_logs.sh [service-name]
# With no argument, shows logs for all services. Otherwise, e.g.:
#   show_logs.sh backend
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
docker compose --env-file "$RELEASE_ROOT/.env" -f "$RELEASE_ROOT/compose/docker-compose.yml" logs -f --tail=200 "$@"
