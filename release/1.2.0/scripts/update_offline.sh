#!/usr/bin/env bash
# Updates an existing installation (at /opt/rah/apps/pfms/) to this release's
# version.
#
# IMPORTANT: there are no database migrations to apply yet
# (database/sqlserver/migrations/ is a placeholder, see its README.md). db-init's
# install scripts are re-run, which is safe because every install script uses
# IF NOT EXISTS / IF OBJECT_ID IS NULL guards -- they will not recreate or
# touch existing data.
#
# Steps: backup -> load new images -> sync the live deployment to this
# release's Compose/DB-seed definitions -> recreate app containers -> verify.
# The sqlserver container and its data volume are NOT recreated. The live
# deployment's .env, assets/, and backups/ are NOT touched -- only the
# version-owned Compose/DB-seed resources are replaced.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

if ! live_deployment_exists; then
    echo "ERROR: no existing live deployment found at $LIVE_ROOT."
    echo "       update_offline.sh only updates an existing installation --"
    echo "       run install_offline.sh first for a first-time install."
    exit 1
fi

load_env

# This release introduced HTTPS (see RELEASE_NOTES.md 1.2.0) -- an existing
# pre-1.2.0 .env won't have SERVER_HOSTNAME_OR_IP set, and update_offline.sh
# deliberately never touches .env itself (see below). Without it, the
# frontend's nginx config would require a certificate that was never
# generated, and the frontend container would fail to start after
# --force-recreate. Fail clearly and BEFORE touching anything, rather than
# leaving the operator with a broken frontend mid-update.
if [ -z "${SERVER_HOSTNAME_OR_IP:-}" ]; then
    echo "ERROR: SERVER_HOSTNAME_OR_IP is not set in $ENV_FILE."
    echo "       This release added HTTPS support, which requires this value"
    echo "       (the address clinical workstations use to reach this server)"
    echo "       to generate a certificate. Add a line to $ENV_FILE:"
    echo "         SERVER_HOSTNAME_OR_IP=<this server's real LAN IP or hostname>"
    echo "       then re-run this script. See RELEASE_NOTES.md 'Upgrade note'"
    echo "       for the 1.2.0 release."
    exit 1
fi

echo "=== Patient Feedback System - Offline Update ==="
echo "Release root: $RELEASE_ROOT"
echo "Live deployment path: $LIVE_ROOT"
echo ""
echo "This will:"
echo "  1. Back up the current database."
echo "  2. Load this release's Docker images."
echo "  3. Replace the live deployment's Compose definition and DB seed data"
echo "     with this release's versions, and bump APP_VERSION in .env to"
echo "     ${IMAGE_VERSION} (everything else in .env is left untouched)."
echo "  4. Generate a TLS certificate if this deployment doesn't have one yet"
echo "     (first update from a pre-HTTPS release), otherwise leave it untouched."
echo "  5. Recreate db-init, backend, and frontend containers."
echo "  6. Leave the SQL Server container and its data untouched."
echo "  7. Update the operational documentation vault and DBeaver connection."
echo ""
read -r -p "Continue? Type YES to proceed: " confirm
if [ "$confirm" != "YES" ]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "[1/6] Backing up the database ..."
"$SCRIPT_DIR/backup_database.sh"

echo ""
echo "[2/6] Loading new images ..."
"$SCRIPT_DIR/load_images.sh"

echo ""
echo "[3/6] Syncing the live deployment to this release's Compose/DB-seed ..."
sync_version_owned_resources
echo "  Done -- $LIVE_ROOT/compose and $LIVE_ROOT/database/sqlserver/seed now"
echo "  match this release. Persistent state (.env, assets/, backups/) untouched."

# APP_VERSION is the one .env value that MUST change on update (it's what
# Compose uses to pick the image tag) -- everything else in .env is
# deployment-owned and stays put. Without this, --force-recreate below would
# just recreate containers from the OLD tag forever.
sed -i "s|^APP_VERSION=.*|APP_VERSION=${IMAGE_VERSION}|" "$ENV_FILE"
echo "  APP_VERSION set to ${IMAGE_VERSION} in $ENV_FILE."

# Covers upgrading a pre-1.2.0 deployment that has never generated a
# certificate before (same "generate once, persists" logic as
# install_offline.sh's step 6). A no-op on any deployment that already has
# one -- this is not regenerated on every update.
CERT_FILE="$LIVE_ROOT/certs/cert.pem"
if [ -f "$CERT_FILE" ]; then
    echo "  Certificate already present at $CERT_FILE, unchanged."
else
    echo "  No certificate found -- generating one now (first time this"
    echo "  deployment has had HTTPS support) ..."
    "$SCRIPT_DIR/generate_certificate.sh" "$SERVER_HOSTNAME_OR_IP" localhost 127.0.0.1
fi

echo ""
echo "[4/6] Recreating db-init, backend, and frontend (sqlserver is untouched) ..."
compose up -d --force-recreate db-init backend frontend

echo "  Waiting for backend and frontend to report healthy ..."
for svc in backend frontend; do
    attempt=0
    status="starting"
    while [ "$attempt" -lt 30 ]; do
        status="$(docker inspect --format='{{.State.Health.Status}}' "$(compose ps -q "$svc")" 2>/dev/null || echo "starting")"
        [ "$status" = "healthy" ] && break
        attempt=$((attempt + 1))
        sleep 5
    done
    if [ "$status" != "healthy" ]; then
        echo "  WARNING: $svc did not report healthy within the wait window (last status: $status)."
    fi
done

echo ""
echo "[5/6] Verifying ..."
"$SCRIPT_DIR/verify_installation.sh" || {
    echo ""
    echo "Verification reported failures. To roll back:"
    echo "  1. Restore the previous release's Compose/DB-seed at $LIVE_ROOT"
    echo "     (from the previous release's own folder, if still available)."
    echo "  2. Reload the previous release's images (docker load -i <old .tar files>)."
    echo "  3. Restore the pre-update backup: scripts/restore_database.sh <backup file printed above>"
    echo "  4. Recreate containers: scripts/start_stack.sh"
    exit 1
}

# Refresh the "genuinely installed" marker to the new version now that
# verification has actually passed -- see _common.sh's
# live_deployment_exists(). Kept in sync with install_offline.sh's own
# write of the same file.
echo "$IMAGE_VERSION" > "$INSTALLED_VERSION_FILE"

echo ""
echo "[6/6] Updating operational documentation and DBeaver connection ..."
load_env
update_operational_docs "Updated"
"$SCRIPT_DIR/provision_dbeaver.sh"

echo ""
echo "=== Update complete ==="
