"""
ML Architecture Consolidation — Stage 13: Final SQLite Archive

Takes the last checksummed, read-only archival copy of the legacy ML store
(models_directory/patient_feedback_ml.db) before it is removed from the
active application path — matching the exact pattern already used for the
Stage 1 and Stage 8 baselines (C:\\SQLBackup\\ml_stage1_archive_*,
ml_stage8_freeze_baseline_*), so this final copy sits alongside them as
migration evidence / emergency-recovery material.

Marks the archive copy read-only on disk (icacls) after the checksum
verification, since this is meant to be the permanent historical record,
not a working copy.

Does NOT touch backend/data/training_metadata.db (a separate, small
training-run-history store, out of scope for this retirement — see
ML_ARCHITECTURE_DECISION_RECORD.md) or the live patient_feedback_ml.db file
itself; that removal is a separate, explicit step (ml_stage13_retire_sqlite.py).

Run from the backend/ directory:
    python -m scripts.ml_stage13_final_archive
"""

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARCHIVE_ROOT = r"C:\SQLBackup"
SOURCE_PATH = os.path.join(WORKSPACE_ROOT, "models_directory", "patient_feedback_ml.db")


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 70)
    print("ML ARCHITECTURE CONSOLIDATION — STAGE 13: FINAL SQLITE ARCHIVE")
    print("=" * 70)

    if not os.path.exists(SOURCE_PATH):
        print(f"\n[ERROR] Source file not found: {SOURCE_PATH}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join(ARCHIVE_ROOT, f"ml_stage13_final_archive_{timestamp}")
    os.makedirs(archive_dir, exist_ok=True)
    dest_path = os.path.join(archive_dir, "patient_feedback_ml.db")

    print(f"\nSource: {SOURCE_PATH}")
    print(f"Archive destination: {dest_path}")

    source_size = os.path.getsize(SOURCE_PATH)
    print(f"\n[1] Checksumming source ({source_size:,} bytes)...")
    source_checksum = sha256_of(SOURCE_PATH)
    print(f"    SHA-256: {source_checksum}")

    print("\n[2] Copying to archive...")
    shutil.copy2(SOURCE_PATH, dest_path)
    dest_checksum = sha256_of(dest_path)

    if source_checksum != dest_checksum:
        raise RuntimeError("Checksum mismatch after copy — archive copy does not match source!")
    print(f"    Copied and verified: SHA-256 matches ({dest_checksum})")

    print("\n[3] Marking archive copy read-only...")
    os.chmod(dest_path, 0o444)
    try:
        subprocess.run(["icacls", dest_path, "/inheritance:r", "/grant:r", "Everyone:(R)"],
                        capture_output=True, text=True, check=False)
    except Exception as e:
        print(f"    [WARNING] icacls step failed (os.chmod read-only flag still applied): {e}")
    print(f"    {dest_path} is now read-only.")

    manifest = {
        "timestamp": timestamp,
        "source_path": SOURCE_PATH,
        "archive_path": dest_path,
        "sha256": source_checksum,
        "size_bytes": source_size,
        "note": (
            "Final archival copy of the legacy ML SQLite store, taken immediately "
            "before its removal from the active application path (Stage 13 of the "
            "ML architecture consolidation). All valuable data was migrated into "
            "SQL Server's ml schema in Stages 1-11; see "
            "ML_ARCHITECTURE_DECISION_RECORD.md for the full retirement record."
        ),
    }
    manifest_path = os.path.join(archive_dir, "stage13_final_archive_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[4] Manifest written to: {manifest_path}")

    print("\n" + "=" * 70)
    print("FINAL ARCHIVE COMPLETE")
    print(f"Archive dir: {archive_dir}")
    print(f"SHA-256: {source_checksum}")
    print("=" * 70)


if __name__ == "__main__":
    main()
