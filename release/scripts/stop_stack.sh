#!/usr/bin/env bash
# Stops all containers without deleting volumes (database data, config, and
# the model cache are preserved).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
docker compose --env-file "$RELEASE_ROOT/.env" -f "$RELEASE_ROOT/compose/docker-compose.yml" stop
echo "Stack stopped. Data is preserved. Run scripts/start_stack.sh to resume."
