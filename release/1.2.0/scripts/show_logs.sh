#!/usr/bin/env bash
# Usage: show_logs.sh [service-name]
# With no argument, shows logs for all services. Otherwise, e.g.:
#   show_logs.sh backend
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

compose logs -f --tail=200 "$@"
