#!/usr/bin/env bash
# Shared deployment context, sourced by every release lifecycle script.
#
# Resolves release/live paths and application identity in exactly one
# place, so install, update, backup, restore, start/stop/logs, and
# verification scripts can't independently guess paths and silently
# diverge (see RAH Application Release Engineering, "Implement Shared
# Deployment-Script Context").
#
# Two roots exist and are deliberately different things:
#   RELEASE_ROOT -- this release's own staging copy (wherever it was
#                   extracted/mounted: DVD, /tmp, /mnt/usb, ...). Disposable
#                   once install/update finishes; version-specific.
#   LIVE_ROOT     -- the canonical, persistent deployment. Every release of
#                   this application targets the same path; the release
#                   version is never encoded into it. Survives regardless of
#                   what happens to RELEASE_ROOT afterward.
set -euo pipefail

APP_SLUG="pfms"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Each release version lives in its own release/<version>/ folder (e.g.
# release/1.2.0/) -- the folder name IS the version identifier, so derive it
# rather than hardcoding it in multiple scripts.
RELEASE_VERSION="$(basename "$RELEASE_ROOT")"

# Deliberately NOT the same thing as RELEASE_VERSION. RELEASE_VERSION is the
# package's own folder/identity; IMAGE_VERSION is the Docker image tag
# actually baked into docker-images/*.tar right now -- these can differ
# mid-development (e.g. release/1.2.0/ under active work while its images
# are still tagged 1.1.1, not yet rebuilt/retagged). Compose selects images
# by IMAGE_VERSION (APP_VERSION in .env), so getting this wrong means
# "pull access denied" for a tag that was never built. Update the
# IMAGE_VERSION file in lockstep whenever images are rebuilt/retagged.
IMAGE_VERSION="$(cat "$RELEASE_ROOT/IMAGE_VERSION" 2>/dev/null || echo "$RELEASE_VERSION")"

# Overridable via PFMS_LIVE_ROOT for engineering-side testing on machines
# where /opt/ isn't writable (e.g. Windows dev boxes) -- the real offline
# server always uses the default, since this variable is never set there.
LIVE_ROOT="${PFMS_LIVE_ROOT:-/opt/rah/apps/${APP_SLUG}}"
ENV_FILE="$LIVE_ROOT/.env"
COMPOSE_FILE="$LIVE_ROOT/compose/docker-compose.yml"

# Canonical operational documentation root -- shared across ALL RAH
# applications on this server (an Obsidian-compatible vault), deliberately
# separate from LIVE_ROOT (which is this one app's own deployment). Also
# overridable for Windows engineering-side testing.
DOCS_ROOT="${PFMS_DOCS_ROOT:-/opt/rah/documentation}"
APP_DOCS_DIR="$DOCS_ROOT/Applications/${APP_SLUG}"

# True once a live deployment has been established (install_offline.sh has
# run successfully at least once). Scripts use this to refuse to "update"
# a nonexistent deployment or to refuse to "install" over an existing one.
live_deployment_exists() {
    [ -f "$COMPOSE_FILE" ]
}

# Loads the persistent production .env from the live deployment -- never
# from the release's own .env.offline.template. Fails clearly instead of
# silently running against an empty environment if install hasn't happened
# yet.
load_env() {
    if [ ! -f "$ENV_FILE" ]; then
        echo "ERROR: $ENV_FILE not found. Run install_offline.sh first." >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    set -a; source "$ENV_FILE"; set +a
}

# Copies this release's version-owned resources (Compose definition, DB
# seed/provisioning data) into the live deployment, overwriting whatever the
# previous release left there. Deliberately never touches LIVE_ROOT/.env,
# LIVE_ROOT/assets, or LIVE_ROOT/backups -- those are deployment-owned
# persistent state, not release contents (see "Replace what belongs to the
# software version. Preserve what belongs to the deployment.").
#
# The whole database/sqlserver/ tree is copied, not just seed/ -- install and
# migration SQL are baked into the db-init image and aren't read from disk at
# runtime, but verify_installation.sh and qualify_offline_installation.sh
# both read validation/*.sql and the seed/ manifest directly off disk, and
# must keep working even after the release folder is gone (release-folder
# independence). Cheap: these are small text files.
sync_version_owned_resources() {
    mkdir -p "$LIVE_ROOT"
    rm -rf "$LIVE_ROOT/compose"
    cp -r "$RELEASE_ROOT/compose" "$LIVE_ROOT/compose"
    rm -rf "$LIVE_ROOT/database"
    mkdir -p "$LIVE_ROOT/database"
    cp -r "$RELEASE_ROOT/database/sqlserver" "$LIVE_ROOT/database/sqlserver"
}

# Explicit -p so Compose's project name doesn't fall back to the basename of
# the directory containing docker-compose.yml -- also set as compose.yml's
# own top-level `name:` key, this is belt-and-suspenders.
compose() {
    load_env
    docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" -p "${PROJECT_NAME:-pfms}" "$@"
}

# Updates the operational documentation vault: overwrites a "current state"
# snapshot (always reflects right now) and appends one entry to an
# append-only history log (never reflects a past state as if it were
# current). Called by install_offline.sh and update_offline.sh on success.
# Assumes load_env has already been called (needs .env values).
#
# $1 = action label, e.g. "Installed" or "Updated"
update_operational_docs() {
    local action="$1"
    mkdir -p "$APP_DOCS_DIR"
    local now
    now="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

    cat > "$APP_DOCS_DIR/current-state.md" <<EOF
# Patient Feedback System (pfms) -- Current Deployment State

**Last updated:** $now
**Last action:** $action

- **Release version:** ${IMAGE_VERSION}
- **Compose project name:** ${PROJECT_NAME:-pfms}
- **Live deployment path:** $LIVE_ROOT
- **Database:** ${DB_DATABASE:-IncidentManager}
- **SQL Server edition (MSSQL_PID):** ${MSSQL_PID:-unset}
- **Ports:** backend=${BACKEND_HOST_PORT:-8100} frontend=${FRONTEND_HOST_PORT:-8101} sqlserver=${SQLSERVER_HOST_PORT:-1433}
- **Application URL:** http://<server-ip>:${FRONTEND_HOST_PORT:-8101}
- **Backend API docs:** http://<server-ip>:${BACKEND_HOST_PORT:-8100}/docs

See \`deployment-history.md\` in this same folder for the full install/update
history. This file always reflects the current state only -- it is
overwritten on every install/update, never appended to.
EOF

    # Checked BEFORE the `>>` redirect below opens/creates the file -- bash
    # sets up redirections before running a compound command's body, so a
    # `[ ! -f "$file" ]` test *inside* a block already redirected with
    # `>> "$file"` would always see the file as existing (just-created,
    # empty), and the header would never be written even on first creation.
    local history_is_new=0
    [ -f "$APP_DOCS_DIR/deployment-history.md" ] || history_is_new=1

    {
        if [ "$history_is_new" = "1" ]; then
            echo "# Patient Feedback System (pfms) -- Deployment History"
            echo ""
            echo "Append-only log. Never edit past entries -- add a new one."
            echo ""
        fi
        echo "## $now -- $action version ${IMAGE_VERSION}"
        echo ""
        echo "- Compose project name: ${PROJECT_NAME:-pfms}"
        echo "- Database: ${DB_DATABASE:-IncidentManager}"
        echo "- SQL Server edition: ${MSSQL_PID:-unset}"
        echo "- Ports: backend=${BACKEND_HOST_PORT:-8100} frontend=${FRONTEND_HOST_PORT:-8101} sqlserver=${SQLSERVER_HOST_PORT:-1433}"
        echo ""
    } >> "$APP_DOCS_DIR/deployment-history.md"

    echo "  Operational docs updated at $APP_DOCS_DIR"
}
