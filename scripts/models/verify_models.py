"""
Verifies downloaded/local model files against the SHA256 manifest recorded
during upload, and does a basic load smoke-test per file type (catches
corruption or truncation, not just presence).

Usage:
    python scripts/models/verify_models.py
    python scripts/models/verify_models.py --only pfms-classification-models
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lock_utils import REPO_ROOT, load_lock, sha256_of, read_manifest


def smoke_test_load(path: Path) -> str:
    """Returns 'ok', 'skipped:<reason>', or 'FAILED:<error>'."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".pkl":
            import joblib
            joblib.load(path)
            return "ok"
        if suffix == ".json":
            with open(path, encoding="utf-8") as f:
                json.load(f)
            return "ok"
        if suffix == ".safetensors":
            try:
                from safetensors import safe_open
            except ImportError:
                return "skipped:safetensors package not installed"
            with safe_open(str(path), framework="pt") as f:
                list(f.keys())
            return "ok"
        return "skipped:no smoke-test defined for this file type"
    except Exception as e:
        return f"FAILED:{type(e).__name__}: {e}"


def verify_one(entry: dict) -> bool:
    model_id = entry["id"]
    target_dir = REPO_ROOT / entry["target_directory"]
    manifest_path = REPO_ROOT / "scripts" / "models" / entry["sha256_manifest"]

    if not manifest_path.exists():
        print(f"[SKIP] {model_id}: no manifest at {manifest_path} -- run upload_models.py first")
        return True

    expected = read_manifest(manifest_path)
    all_ok = True
    print(f"[{model_id}] verifying {len(expected)} files against {manifest_path.name} ...")

    for rel_path, expected_hash in expected.items():
        local_path = target_dir / rel_path
        if not local_path.exists():
            print(f"    MISSING: {rel_path}")
            all_ok = False
            continue

        actual_hash = sha256_of(local_path)
        if actual_hash != expected_hash:
            print(f"    CHECKSUM MISMATCH: {rel_path}")
            print(f"        expected {expected_hash}")
            print(f"        actual   {actual_hash}")
            all_ok = False
            continue

        smoke = smoke_test_load(local_path)
        if smoke.startswith("FAILED"):
            print(f"    LOAD FAILED: {rel_path} -- {smoke}")
            all_ok = False
        elif smoke.startswith("skipped"):
            print(f"    ok (checksum), {smoke}: {rel_path}")
        else:
            print(f"    ok: {rel_path}")

    print(f"[{model_id}] {'ALL PASSED' if all_ok else 'FAILURES FOUND -- see above'}")
    return all_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="Only verify this model id from models.lock.yaml")
    args = parser.parse_args()

    lock = load_lock()
    results = []
    for entry in lock["models"]:
        if args.only and entry["id"] != args.only:
            continue
        results.append(verify_one(entry))

    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
