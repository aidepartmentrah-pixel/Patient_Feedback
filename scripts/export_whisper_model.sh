#!/usr/bin/env bash
# export_whisper_model.sh
# Runs on the ONLINE engineering machine (needs internet). Pre-downloads the
# CTranslate2-format Faster-Whisper model files into assets/whisper-model-<size>/,
# which then ships as a release asset and gets mounted into the backend
# container at runtime (WHISPER_MODEL_PATH) -- no huggingface.co access is
# ever needed on the offline server. Adapted from the same pattern already
# proven in voice-project_Deployment/scripts/export_whisper_model.sh.
#
# Usage: ./export_whisper_model.sh [model-size]   (default: medium)
set -euo pipefail

MODEL_SIZE="${1:-medium}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$REPO_ROOT/assets/whisper-model-${MODEL_SIZE}"

echo "==> Downloading '${MODEL_SIZE}' model inside a scratch container..."
mkdir -p "$OUT_DIR"

# Downloads inside the container's own filesystem (not a bind mount) then
# docker cp's the result out -- bind-mounting the output directly can
# silently produce an empty directory under Docker Desktop's Windows/WSL2
# file sharing (rename-across-mount-boundary issue), so this avoids that
# entirely. Uses the already-built backend image so this always matches
# exactly what the container will load with.
CONTAINER_NAME="whisper-model-export-$$"
docker run --name "$CONTAINER_NAME" \
  --user root \
  rah-pfms-backend:1.0.0 \
  python -c "
from faster_whisper.utils import download_model
path = download_model('${MODEL_SIZE}', output_dir='/tmp/model-out', local_files_only=False)
print(f'Downloaded to: {path}')
"
docker cp "$CONTAINER_NAME:/tmp/model-out/." "$OUT_DIR/"
docker rm "$CONTAINER_NAME" >/dev/null

echo "==> Done. Contents of ${OUT_DIR}:"
ls -lh "$OUT_DIR"
echo
echo "==> Zipping for release packaging..."
mkdir -p "$REPO_ROOT/release/assets"
ZIP_PATH="$REPO_ROOT/release/assets/whisper-model-${MODEL_SIZE}.zip"

# Deliberately using Python's zipfile module (via the already-built backend
# image) rather than `zip` or PowerShell's Compress-Archive. Compress-Archive
# writes backslash path separators inside the zip's internal entry names --
# technically non-conformant (the ZIP spec mandates forward slashes) -- and
# standard Linux `unzip` doesn't reliably convert them, so the archive would
# extract flat instead of into a whisper-model-<size>/ subdirectory on the
# actual Debian target. zipfile always writes forward slashes regardless of
# host OS, sidestepping this entirely.
ZIP_CONTAINER="whisper-zip-export-$$"
MSYS_NO_PATHCONV=1 docker run --name "$ZIP_CONTAINER" \
  -v "$OUT_DIR:/src:ro" \
  --entrypoint python \
  rah-pfms-backend:1.0.0 \
  -c "
import zipfile, os
src = '/src'
with zipfile.ZipFile('/tmp/out.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(src):
        for f in files:
            full = os.path.join(root, f)
            arcname = 'whisper-model-${MODEL_SIZE}/' + os.path.relpath(full, src).replace(os.sep, '/')
            zf.write(full, arcname)
print('zip built inside container')
"
docker cp "$ZIP_CONTAINER:/tmp/out.zip" "$ZIP_PATH"
docker rm "$ZIP_CONTAINER" >/dev/null

echo "==> Wrote $ZIP_PATH"
echo "==> Verifying archive uses forward slashes and the expected top-level folder..."
MSYS_NO_PATHCONV=1 docker run --rm -v "$REPO_ROOT/release/assets:/z:ro" --entrypoint python rah-pfms-backend:1.0.0 -c "
import zipfile
with zipfile.ZipFile('/z/whisper-model-${MODEL_SIZE}.zip') as zf:
    names = zf.namelist()
    assert all('\\\\' not in n for n in names), 'backslash found in zip entry names!'
    assert all(n.startswith('whisper-model-${MODEL_SIZE}/') for n in names), 'unexpected top-level entry'
    print(f'OK - {len(names)} entries, all forward-slash, all under whisper-model-${MODEL_SIZE}/')
"

echo
echo "NOTE: ship ONLY $ZIP_PATH in the release package -- do NOT also commit"
echo "      or ship the extracted $OUT_DIR directory. install_offline.sh"
echo "      extracts the zip on the target server; shipping both roughly"
echo "      doubles this asset's size for no benefit."
