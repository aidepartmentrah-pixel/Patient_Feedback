"""
One-time merge step: adds the custom_views domain into an existing, already-
validated provisioning.v1.json without re-running the full Stage B build.

Why this exists: Stage B's raw_extract/user_credentials.json is deleted after
a successful build (by design -- it holds plaintext test passwords). Adding a
new, unrelated data domain (custom views) later must not require re-extracting
real user credentials all over again just to rebuild org_units/users
identically. This script loads the EXISTING provisioning.v1.json (leaving its
org_units/users untouched, byte-for-byte), adds custom_views from
raw_extract/custom_views.json, and rewrites the artifact + manifest + checksum
the same way Stage B does.

Usage:
    python database/sqlserver/seed/merge_custom_views_into_artifact.py
"""
import hashlib
import json
import sys
from pathlib import Path

SEED_DIR = Path(__file__).resolve().parent
RAW_DIR = SEED_DIR / "raw_extract"

sys.path.insert(0, str(SEED_DIR))
from build_provisioning_artifact import load_custom_views  # noqa: E402


def main():
    final_path = SEED_DIR / "provisioning.v1.json"
    manifest_path = SEED_DIR / "provisioning.v1.manifest.json"

    if not final_path.exists():
        print(f"ERROR: {final_path} not found -- nothing to merge into.")
        sys.exit(1)

    artifact = json.load(open(final_path, encoding="utf-8"))
    if "custom_views" in artifact:
        print("custom_views already present in provisioning.v1.json -- overwriting with fresh extract.")

    custom_views = load_custom_views()
    artifact["custom_views"] = custom_views
    print(f"Merging {len(custom_views)} custom views into existing artifact "
          f"({len(artifact['org_units'])} org units, {len(artifact['users'])} users unchanged).")

    tmp_path = final_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)
    tmp_path.replace(final_path)

    # Validate
    reloaded = json.load(open(final_path, encoding="utf-8"))
    assert len(reloaded["custom_views"]) == len(custom_views)
    assert all(v["view_name"] for v in reloaded["custom_views"])
    print("Validation: structure OK.")

    # Recompute checksum + manifest
    file_bytes = final_path.read_bytes()
    checksum = hashlib.sha256(file_bytes).hexdigest()

    manifest = json.load(open(manifest_path, encoding="utf-8"))
    manifest["checksum_sha256"] = checksum
    manifest["record_counts"]["custom_views_total"] = len(custom_views)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Updated {manifest_path}")

    sha256_path = SEED_DIR / "provisioning.v1.json.sha256"
    with open(sha256_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{checksum}  provisioning.v1.json\n")
    print(f"Updated {sha256_path}")

    print(f"\nDone. New checksum: {checksum}")


if __name__ == "__main__":
    main()
