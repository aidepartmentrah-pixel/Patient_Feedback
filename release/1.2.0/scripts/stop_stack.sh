#!/usr/bin/env bash
# Stops all containers without deleting volumes (database data, config, and
# the model cache are preserved).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

compose stop
echo "Stack stopped. Data is preserved. Run scripts/start_stack.sh to resume."
