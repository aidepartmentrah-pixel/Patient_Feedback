#!/usr/bin/env bash
# Generates a self-signed TLS cert/key for the frontend's nginx, pinned to
# whatever IP/hostname the deployment target actually uses. Adapted from the
# proven pattern in voice-project_Deployment/nginx/generate_self_signed_cert.sh.
#
# Re-run this whenever the stack moves to a new host or IP -- a cert
# generated for one address will not validate for another. install_offline.sh
# calls this automatically on first install only (persists across updates,
# like .env/assets/backups); run it by hand afterward if the server's
# network identity ever changes.
#
# Usage: ./generate_certificate.sh <IP-or-hostname> [more IPs/hostnames...]
# Example: ./generate_certificate.sh 150.50.10.30 localhost 127.0.0.1
#
# Note for testing on Windows Git Bash: MSYS mangles a leading "/" in -subj
# into a Windows path. Run with `MSYS2_ARG_CONV_EXCL="/CN=" ./generate_certificate.sh ...`
# there (scoped to just that argument). The real target (Debian) has no such
# quirk.

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <IP-or-hostname> [more IPs/hostnames...]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

CERT_DIR="$LIVE_ROOT/certs"
mkdir -p "$CERT_DIR"

SAN=""
for entry in "$@"; do
    if [[ "$entry" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        SAN="${SAN}IP:${entry},"
    else
        SAN="${SAN}DNS:${entry},"
    fi
done
SAN="${SAN%,}"

openssl req -x509 -nodes -days 825 \
    -newkey rsa:2048 \
    -keyout "$CERT_DIR/key.pem" \
    -out "$CERT_DIR/cert.pem" \
    -subj "/CN=$1/O=RAH/C=LB" \
    -addext "subjectAltName=$SAN"

echo "Generated $CERT_DIR/cert.pem and $CERT_DIR/key.pem for: $SAN"
echo "Restart the frontend container to pick up the new cert:"
echo "  scripts/stop_stack.sh && scripts/start_stack.sh"
