"""
Stage 11 End-to-End Test — Training Run -> Graph Generation -> ZIP Download

Genuinely new coverage: no existing test touches training_service.py,
training_router.py, or training_run_artifacts_service.py at all. This
test runs the REAL end-to-end path a user triggers via POST
/api/settings/training/run (train_all()), then exercises Stage 10's
listing/detail/ZIP-download service functions (already unit-verified in
Stage 10) against the resulting real run.

Deletes its own run folder at teardown so it doesn't clutter the small
set of genuine historical runs kept after Stage 10's cleanup.

Run from the backend/ directory:
    python -m tests.test_ml_e2e_training_run_zip_download
"""

import hashlib
import os
import shutil
import sys
import zipfile
from io import BytesIO

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models_directory.split_data import split_data_from_sql_server
from models_directory.Classification_Models.Maintainance.train_all import train_all
from models_directory.Classification_Models.Maintainance import run_versioning
from api.services.training_run_artifacts_service import (
    list_versioned_runs,
    get_versioned_run_detail,
    build_run_zip,
)


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main():
    print("=" * 70)
    print("STAGE 11 E2E — Training Run -> Graph Generation -> ZIP Download")
    print("=" * 70)

    print("\n[1] Splitting data via split_data_from_sql_server() — the real POST /run "
          "pipeline (training_service._run_split_data(), Stage 13) always does this "
          "immediately before train_all(); this test must too, rather than relying on "
          "table_feedback_train/test happening to already exist from an earlier run...")
    split_result = split_data_from_sql_server()
    print(f"    {split_result}")

    print("\n[2] Running a REAL training run (train_all()) — this is what POST /run triggers...")
    result = train_all()
    run_id = result["run_id"]
    print(f"    run_id={run_id}, total_models={result['summary']['total_models']}")

    run_dir = run_versioning.RUNS_ROOT / run_id
    assert run_dir.is_dir(), f"Expected run folder to exist at {run_dir}"
    assert (run_dir / run_versioning.SUMMARY_FILENAME).is_file()

    try:
        print("\n[3] Confirming the run appears in list_versioned_runs()...")
        runs = list_versioned_runs()
        run_ids_listed = [r["run_id"] for r in runs]
        print(f"    Most recent listed run_id: {run_ids_listed[0] if run_ids_listed else None}")
        assert run_id in run_ids_listed, "New run did not appear in the versioned-runs list"

        print("\n[4] Confirming get_versioned_run_detail() returns the full summary...")
        detail = get_versioned_run_detail(run_id)
        assert detail["run_id"] == run_id
        assert len(detail["models"]) == 18
        models_with_artifacts = [name for name, m in detail["models"].items() if m.get("artifacts")]
        print(f"    Models with graph/artifact output: {len(models_with_artifacts)} of 18")
        assert len(models_with_artifacts) >= 1, "Expected at least some models to have produced graph artifacts"

        print("\n[5] Building and verifying the ZIP download...")
        zip_result = build_run_zip(run_id)
        print(f"    filename={zip_result['filename']}, size={len(zip_result['content'])} bytes")
        assert zip_result["content_type"] == "application/zip"

        zf = zipfile.ZipFile(BytesIO(zip_result["content"]))
        names = set(zf.namelist())
        assert "run_summary.json" in names

        checked = 0
        for model_name, m in detail["models"].items():
            for artifact in m.get("artifacts", []):
                rel = artifact["relative_path"]
                assert rel in names, f"Manifest artifact {rel} missing from ZIP"
                actual_sha = sha256_of_bytes(zf.read(rel))
                assert actual_sha == artifact["sha256"], f"Checksum mismatch for {rel}"
                checked += 1
        print(f"    Verified {checked} artifact checksums match the manifest exactly")
        assert checked > 0

        print("\n" + "=" * 70)
        print("ALL E2E ASSERTIONS PASSED — Training Run -> Graph Generation -> ZIP Download")
        print("=" * 70)

    finally:
        print(f"\n[Cleanup] Removing test run folder {run_dir}...")
        if run_dir.is_dir():
            shutil.rmtree(run_dir)
        print("    Cleanup complete.")


if __name__ == "__main__":
    main()
