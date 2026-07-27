#!/usr/bin/env bash
# Usage: show_logs.sh [service-name]
# With no argument, shows logs for all services. Otherwise, e.g.:
#   show_logs.sh backend
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$RELEASE_ROOT/.env"

# shellcheck disable=SC1090
set -a; source "$ENV_FILE"; set +a

# Explicit -p so Compose's project name doesn't fall back to the basename of
# the directory containing docker-compose.yml ("compose").
docker compose --env-file "$ENV_FILE" -f "$RELEASE_ROOT/compose/docker-compose.yml" -p "${PROJECT_NAME:-pfms}" logs -f --tail=200 "$@"
