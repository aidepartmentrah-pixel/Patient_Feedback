#!/usr/bin/env bash
# Engineering-side tool -- NOT part of the operator-facing install/update
# flow. Run this on the build machine right before shipping, after all
# images are finalized/tagged and checksums/release_hashes.txt is
# regenerated, to produce a machine-readable snapshot of what's actually
# packaged in this release (as opposed to RELEASE_NOTES.md, which is the
# human-readable "what's new and what's known-broken" narrative -- this
# script never duplicates that content, only structural facts).
#
# Usage: ./generate_release_manifest.sh   (run from anywhere; writes
#         release-manifest.json to the release root)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

MANIFEST_PATH="$RELEASE_ROOT/release-manifest.json"

echo "=== Generating release manifest for $RELEASE_ROOT ==="

GIT_COMMIT="unknown"
if command -v git >/dev/null 2>&1 && git -C "$RELEASE_ROOT" rev-parse HEAD >/dev/null 2>&1; then
    GIT_COMMIT="$(git -C "$RELEASE_ROOT" rev-parse HEAD)"
fi

image_entry() {
    local name="$1"
    local tar="$RELEASE_ROOT/docker-images/${name}.tar"
    local image_ref="rah-pfms-${name}:${IMAGE_VERSION}"
    local digest="not-loaded-locally"
    if docker image inspect "$image_ref" >/dev/null 2>&1; then
        digest="$(docker image inspect "$image_ref" --format '{{.Id}}')"
    fi
    local tar_size="0"
    local tar_sha="not-found"
    if [ -f "$tar" ]; then
        tar_size="$(wc -c < "$tar" | tr -d ' ')"
        tar_sha="$(sha256sum "$tar" | cut -d' ' -f1)"
    fi
    printf '    {\n      "image": "%s",\n      "docker_image_id": "%s",\n      "tar_file": "docker-images/%s.tar",\n      "tar_size_bytes": %s,\n      "tar_sha256": "%s"\n    }' \
        "$image_ref" "$digest" "$name" "$tar_size" "$tar_sha"
}

TOTAL_SIZE_BYTES="$(find "$RELEASE_ROOT" -type f -exec du -cb {} + 2>/dev/null | tail -1 | cut -f1 || echo 0)"
FILE_COUNT="$(find "$RELEASE_ROOT" -type f | wc -l | tr -d ' ')"

cat > "$MANIFEST_PATH" <<EOF
{
  "release_version": "${RELEASE_VERSION}",
  "image_version": "${IMAGE_VERSION}",
  "generated_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "source_git_commit": "${GIT_COMMIT}",
  "package": {
    "total_files": ${FILE_COUNT},
    "total_size_bytes": ${TOTAL_SIZE_BYTES}
  },
  "images": [
$(image_entry backend),
$(image_entry frontend),
$(image_entry db-init)
  ],
  "checksums_file": "checksums/release_hashes.txt",
  "release_notes": "documentation/RELEASE_NOTES.md",
  "qualification_status": "Engineered Release -- Awaiting Offline Qualification"
}
EOF

echo "Wrote $MANIFEST_PATH"
# Checks functionality, not just PATH presence -- on Windows, `python3` can
# resolve to the Microsoft Store's non-functional execution-alias stub (see
# qualify_offline_installation.sh's python_or_die for the same fix).
PY=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "1" >/dev/null 2>&1; then
        PY="$candidate"
        break
    fi
done
if [ -n "$PY" ]; then
    "$PY" -c "import json; json.load(open('$MANIFEST_PATH'))" && echo "  Valid JSON, confirmed."
else
    echo "  (no working python available to validate JSON syntax here -- inspect manually)"
fi
