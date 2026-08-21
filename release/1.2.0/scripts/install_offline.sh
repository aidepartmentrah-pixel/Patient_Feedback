#!/usr/bin/env bash
# First-time installation on the offline Debian server. Uses only local
# files -- never contacts the Internet.
#
# What this does:
#   1. Verifies Docker is available.
#   2. Verifies (or helps create) the .env file in the canonical live
#      deployment (/opt/rah/apps/pfms/), not in this release's own folder.
#   3. Loads the release's Docker images.
#   4. Establishes the canonical live deployment -- copies this release's
#      Compose definition and DB seed data into /opt/rah/apps/pfms/, which
#      is what actually runs from here on, independent of this release
#      folder's location.
#   5. Extracts the Speech-to-Text model asset (assets/whisper-model-medium.zip).
#   6. Verifies the organizational-unit/user provisioning artifact's checksum
#      (fails immediately if missing or corrupted -- this release must never
#      complete an install with zero organizational units and no accounts).
#   7. Generates a self-signed TLS certificate for SERVER_HOSTNAME_OR_IP
#      (persists across updates -- see scripts/generate_certificate.sh).
#   8. Starts the full stack (SQL Server -> db-init [schema + provisioning] ->
#      backend -> frontend).
#   9. Waits for the backend to report healthy, updates the operational
#      documentation vault (/opt/rah/documentation/Applications/pfms/),
#      auto-provisions a DBeaver connection if DBeaver is present, and
#      prints the result.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

echo "=== Patient Feedback System - Offline Installation ==="
echo "Release root: $RELEASE_ROOT"
echo "Live deployment path: $LIVE_ROOT"
echo ""

if live_deployment_exists; then
    echo "ERROR: a live deployment already exists at $LIVE_ROOT."
    echo "       This script is for first-time installation only -- it will"
    echo "       not overwrite an existing deployment's persistent state."
    echo "       To install a new release over an existing deployment, use"
    echo "       update_offline.sh instead."
    exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker is not installed. Install it from the Offline Debian"
    echo "       Server Kit before running this script."
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "ERROR: the 'docker compose' plugin is not available."
    exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
    echo "No .env file found at $ENV_FILE"
    echo "Creating one from .env.offline.template ..."
    mkdir -p "$LIVE_ROOT"
    cp "$RELEASE_ROOT/.env.offline.template" "$ENV_FILE"

    echo "Auto-generating credentials ..."
    # MSSQL_SA_PASSWORD is a fixed, deliberately-chosen value (not random) --
    # see RELEASE_NOTES.md "Password handling" for why. The two encryption
    # keys MUST be valid Fernet keys (32 random bytes, url-safe base64) and
    # MUST be unique per deployment (they key independent encrypted secrets --
    # sharing one across deployments would mean a leak of one deployment's
    # key exposes another's data). Generated here with openssl, which
    # produces the exact same format as Python's Fernet.generate_key()
    # without needing python/cryptography on the host -- confirmed against
    # the real backend image during engineering testing.
    generate_fernet_key() {
        openssl rand 32 | base64 -w0 | tr '+/' '-_'
    }
    sed -i "s|^MSSQL_SA_PASSWORD=.*|MSSQL_SA_PASSWORD=NewPassword2004|" "$ENV_FILE"
    sed -i "s|^SETTINGS_ENCRYPTION_KEY=.*|SETTINGS_ENCRYPTION_KEY=$(generate_fernet_key)|" "$ENV_FILE"
    sed -i "s|^PASSWORD_EXPORT_ENCRYPTION_KEY=.*|PASSWORD_EXPORT_ENCRYPTION_KEY=$(generate_fernet_key)|" "$ENV_FILE"
    # Matches whatever's actually baked into docker-images/*.tar right now
    # (IMAGE_VERSION, from _common.sh) -- NOT necessarily the release
    # folder's own name (RELEASE_VERSION), which can be ahead of it
    # mid-development. Using the wrong one here means Compose tries to pull
    # a tag that was never built.
    sed -i "s|^APP_VERSION=.*|APP_VERSION=${IMAGE_VERSION}|" "$ENV_FILE"
    echo "  Done -- SQL Server password, both encryption keys, and app version are set."

    if grep -q "__SET_ME__" "$ENV_FILE"; then
        echo ""
        echo "ACTION REQUIRED: $ENV_FILE still needs MSSQL_PID set to your SQL"
        echo "Server license edition (Express / Standard / Enterprise /"
        echo "EnterpriseCore -- Developer is not licensed for production)."
        echo "This is a licensing decision only the operator can make, so it's"
        echo "not auto-filled. Edit it, then run this script again."
        exit 1
    fi
fi

if grep -q "__SET_ME__" "$ENV_FILE"; then
    echo "ERROR: $ENV_FILE still contains __SET_ME__ placeholders."
    echo "       Edit it and fill in every required value before continuing."
    exit 1
fi

load_env

echo "[1/9] Loading Docker images ..."
"$SCRIPT_DIR/load_images.sh"

echo ""
echo "[2/9] Establishing the canonical live deployment at $LIVE_ROOT ..."
sync_version_owned_resources
echo "  Compose definition and DB seed data copied."

echo ""
echo "[3/9] Extracting the Speech-to-Text model asset ..."
MODEL_ZIP="$RELEASE_ROOT/assets/whisper-model-medium.zip"
MODEL_DIR="$LIVE_ROOT/assets/whisper-model-medium"
if [ ! -f "$MODEL_ZIP" ]; then
    echo "ERROR: $MODEL_ZIP not found. The release package is incomplete -- the"
    echo "       Speech-to-Text model asset is required; the backend has no"
    echo "       other way to obtain it on an offline server."
    exit 1
fi
# The 4 files a CTranslate2 Faster-Whisper model actually needs to load
# (see scripts/export_whisper_model.sh) -- a directory can be non-empty
# (e.g. just the .cache/huggingface/ download metadata, or a truncated
# extraction) without actually being loadable. Check each required file
# individually, not just "the directory has something in it."
REQUIRED_MODEL_FILES="config.json model.bin tokenizer.json vocabulary.txt"

model_dir_complete() {
    for f in $REQUIRED_MODEL_FILES; do
        if [ ! -s "$MODEL_DIR/$f" ]; then
            return 1
        fi
    done
    return 0
}

if [ -d "$MODEL_DIR" ] && model_dir_complete; then
    echo "  Already extracted and complete at $MODEL_DIR, skipping."
else
    mkdir -p "$LIVE_ROOT/assets"
    unzip -q -o "$MODEL_ZIP" -d "$LIVE_ROOT/assets"
    if ! model_dir_complete; then
        echo "ERROR: extraction completed but $MODEL_DIR is missing one or more of:"
        echo "       $REQUIRED_MODEL_FILES"
        echo "       The release's whisper-model-medium.zip is incomplete or"
        echo "       corrupted. Speech-to-Text cannot work without these files"
        echo "       present -- re-copy the release bundle before retrying."
        exit 1
    fi
    echo "  Extracted and verified complete at $MODEL_DIR"
fi

echo ""
echo "[4/9] Creating backup directory (for future updates) ..."
mkdir -p "$LIVE_ROOT/backups"

echo ""
echo "[5/9] Verifying the organizational-unit/user provisioning artifact ..."
PROVISION_DIR="$LIVE_ROOT/database/sqlserver/seed"
PROVISION_JSON="$PROVISION_DIR/provisioning.v1.json"
PROVISION_SHA="$PROVISION_DIR/provisioning.v1.json.sha256"
if [ ! -f "$PROVISION_JSON" ] || [ ! -f "$PROVISION_SHA" ]; then
    echo "ERROR: $PROVISION_JSON or its .sha256 checksum file is missing."
    echo "       This release package is incomplete -- a fresh install must"
    echo "       never complete with zero organizational units and no usable"
    echo "       accounts. Installation aborted."
    exit 1
fi
if ! (cd "$PROVISION_DIR" && sha256sum -c provisioning.v1.json.sha256 >/dev/null); then
    echo "ERROR: provisioning.v1.json failed checksum verification."
    echo "       The release bundle may be corrupted or tampered with in"
    echo "       transit. Installation aborted -- do not proceed with a"
    echo "       re-copy of this release before investigating."
    exit 1
fi
echo "  Checksum OK."

echo ""
echo "  Checking for the optional ML historical training-data seed ..."
ML_TRAINING_JSON="$PROVISION_DIR/ml_training_data.v1.json"
ML_TRAINING_SHA="$PROVISION_DIR/ml_training_data.v1.json.sha256"
if [ -f "$ML_TRAINING_JSON" ] && [ -f "$ML_TRAINING_SHA" ]; then
    if (cd "$PROVISION_DIR" && sha256sum -c ml_training_data.v1.json.sha256 >/dev/null); then
        echo "  Found and verified -- 'Train All Models' will have a real historical"
        echo "  baseline from day one instead of starting empty."
    else
        echo "ERROR: ml_training_data.v1.json failed checksum verification."
        echo "       The release bundle may be corrupted or tampered with in"
        echo "       transit. Installation aborted -- do not proceed with a"
        echo "       re-copy of this release before investigating."
        exit 1
    fi
else
    echo "  Not present -- this is optional, not an error. The system is fully"
    echo "  functional without it: training and the ML dashboards will simply"
    echo "  start empty and grow as real incidents are processed (see"
    echo "  database/sqlserver/seed/extract_ml_training_data.py to produce this"
    echo "  artifact from an engineering database that already has historical data)."
fi

echo ""
echo "[6/9] Generating the TLS certificate ..."
CERT_FILE="$LIVE_ROOT/certs/cert.pem"
if [ -f "$CERT_FILE" ]; then
    echo "  Already present at $CERT_FILE, skipping (persists across updates --"
    echo "  re-run scripts/generate_certificate.sh by hand if the server's"
    echo "  address ever changes)."
else
    "$SCRIPT_DIR/generate_certificate.sh" "$SERVER_HOSTNAME_OR_IP" localhost 127.0.0.1
fi

echo ""
echo "[7/9] Starting the stack (schema install + organizational/user provisioning"
echo "      run automatically as part of db-init) ..."
compose up -d

echo "  Waiting for db-init (schema install + provisioning) to finish ..."
for _ in $(seq 1 60); do
    db_init_status="$(compose ps -a --format '{{.Name}} {{.State}} {{.ExitCode}}' db-init 2>/dev/null || true)"
    case "$db_init_status" in
        *"exited 0"*) echo "  db-init completed successfully."; break ;;
        *"exited"*)
            echo ""
            echo "ERROR: db-init failed (schema install or provisioning). Logs:"
            compose logs db-init
            echo ""
            echo "Installation aborted. Fix the issue above before re-running."
            exit 1
            ;;
    esac
    sleep 5
done

echo ""
echo "[8/9] Waiting for the backend to become healthy ..."

attempt=0
max_attempts=30  # ~7.5 minutes at 15s intervals -- the model is already
                 # local (no download), so this should be quick.
while [ "$attempt" -lt "$max_attempts" ]; do
    status="$(docker inspect --format='{{.State.Health.Status}}' "$(compose ps -q backend)" 2>/dev/null || echo "starting")"
    if [ "$status" = "healthy" ]; then
        echo ""
        echo "=== Installation complete. Backend is healthy. ==="
        break
    fi
    attempt=$((attempt + 1))
    sleep 15
done

if [ "$status" != "healthy" ]; then
    echo ""
    echo "WARNING: backend did not report healthy within the wait window."
    echo "         Check logs with: scripts/show_logs.sh backend"
    exit 1
fi

# Written only now that the backend has actually reported healthy -- this
# is the authoritative "installation genuinely completed" signal
# live_deployment_exists() checks (see _common.sh). Must not be written any
# earlier: the Platform pre-renders compose/ and other resources into
# LIVE_ROOT before this script even starts, so anything written before a
# real success confirmation would falsely mark an incomplete/failed
# attempt as installed.
echo "$IMAGE_VERSION" > "$INSTALLED_VERSION_FILE"

echo ""
echo "[9/9] Updating operational documentation and DBeaver connection ..."
load_env
update_operational_docs "Installed"
"$SCRIPT_DIR/provision_dbeaver.sh"

BACKEND_PORT="$(grep -E '^BACKEND_HOST_PORT=' "$ENV_FILE" | cut -d= -f2 || echo 8100)"
FRONTEND_HTTPS_PORT_VAL="$(grep -E '^FRONTEND_HTTPS_PORT=' "$ENV_FILE" | cut -d= -f2 || echo 8102)"

echo ""
echo "Live deployment:  $LIVE_ROOT"
echo "Application URL:  https://${SERVER_HOSTNAME_OR_IP}:${FRONTEND_HTTPS_PORT_VAL}"
echo "                  (self-signed certificate -- see"
echo "                  documentation/HTTPS_CLIENT_TRUST_GUIDE.md for how to"
echo "                  trust it on clinical workstations)"
echo "Backend API docs: http://<server-ip>:${BACKEND_PORT}/docs"
echo ""
echo "This release folder ($RELEASE_ROOT) can now be safely moved or removed"
echo "-- the live deployment no longer depends on it."
echo ""
echo "Run scripts/verify_installation.sh for a full validation pass."
