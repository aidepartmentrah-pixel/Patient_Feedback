#!/usr/bin/env bash
# Generates a self-signed TLS cert/key for the dev frontend (docker-compose.dev.yml
# mounts ./.dev-certs-root/certs into nginx). Re-run if the dev host's IP changes.
#
# Usage: scripts/generate_dev_cert.sh [IP-or-hostname ...]   (default: localhost 127.0.0.1)
#
# On Windows Git Bash, MSYS mangles the leading "/" in -subj; run with
#   MSYS2_ARG_CONV_EXCL="/CN=" scripts/generate_dev_cert.sh
set -euo pipefail

[ "$#" -ge 1 ] || set -- localhost 127.0.0.1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CERT_DIR="$REPO_ROOT/.dev-certs-root/certs"
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

echo "Generated $CERT_DIR/cert.pem and key.pem for: $SAN"
