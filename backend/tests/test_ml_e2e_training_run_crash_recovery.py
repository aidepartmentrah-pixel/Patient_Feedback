"""
Stage 12 Rollback/Recovery Test — Training Run Crash Recovery

train_all() runs on a daemon thread inside the same backend process (see
training_service.run_training_pipeline()) — there is no separate process to
SIGKILL to simulate a real crash. This test reproduces the on-disk state a
killed-mid-run process WOULD leave by calling run_versioning.create_run_summary()
and one update_run_summary(), then deliberately never calling
finalize_run_summary() (exactly what a killed process could never do either),
plus writing training_progress.json's is_running=True the same way
initialize_training_progress() does.

Confirms two things:
  1. list_versioned_runs()/get_versioned_run_detail() handle the permanently
     "running" run gracefully (no crash, correct status reported), and a
     separate genuinely-completed run in the same listing is unaffected.
  2. reconcile_stuck_training_runs() (called from backend/main.py's startup,
     mirroring embedding_worker.sweep_stuck_jobs_startup()'s pattern) correctly
     resets training_progress.json's is_running flag and marks the stuck run's
     own run_summary.json as "failed" — closing the real availability bug
     found during Stage 12 planning (a crashed run previously blocked ALL
     future training runs forever via the 409 guard, with no automatic
     recovery, unlike the embedding worker).

Run from the backend/ directory:
    python -m tests.test_ml_e2e_training_run_crash_recovery
"""

import json
import os
import shutil
import sys
from datetime import datetime

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models_directory.Classification_Models.Maintainance import run_versioning
from api.services.training_run_artifacts_service import list_versioned_runs, get_versioned_run_detail
from api.services import training_service


def main():
    print("=" * 70)
    print("STAGE 12 — Training Run Crash Recovery")
    print("=" * 70)

    stuck_run_id = run_versioning.generate_run_id()
    good_run_id = run_versioning.generate_run_id()
    stuck_run_dir = None
    good_run_dir = None
    saved_progress_before_test = training_service._read_progress_file()

    try:
        print(f"\n[1] Simulating a killed-mid-run training process (run_id={stuck_run_id})...")
        stuck_run_dir = run_versioning.get_run_dir(stuck_run_id)
        run_versioning.create_run_summary(
            stuck_run_dir, stuck_run_id, datetime.now().isoformat(),
            dataset_info={"train_rows": 10}, embedding_model_info={"name": "test"},
        )
        run_versioning.update_run_summary(stuck_run_dir, "Fake_Model_One", {"accuracy": 0.9, "warnings": []})
        # Deliberately never call finalize_run_summary() — this is exactly
        # the state a killed process leaves: status stays "running" forever.
        training_service.initialize_training_progress(stuck_run_id, total_steps=18)
        print(f"    Wrote run_summary.json with status='running' (never finalized) and "
              f"training_progress.json's is_running=True.")

        print(f"\n[2] Creating a genuinely completed run for comparison (run_id={good_run_id})...")
        good_run_dir = run_versioning.get_run_dir(good_run_id)
        run_versioning.create_run_summary(
            good_run_dir, good_run_id, datetime.now().isoformat(),
            dataset_info={"train_rows": 10}, embedding_model_info={"name": "test"},
        )
        run_versioning.update_run_summary(good_run_dir, "Fake_Model_One", {"accuracy": 0.95, "warnings": []})
        run_versioning.finalize_run_summary(good_run_dir, datetime.now().isoformat(), "completed")
        print(f"    Wrote a properly finalized run_summary.json with status='completed'.")

        print("\n[3] Confirming list_versioned_runs()/get_versioned_run_detail() handle the "
              "stuck run gracefully, without corrupting or hiding the good run...")
        runs = list_versioned_runs()
        by_id = {r["run_id"]: r for r in runs}
        assert stuck_run_id in by_id, "Stuck run should still be listed"
        assert by_id[stuck_run_id]["status"] == "running"
        assert good_run_id in by_id, "Good run should still be listed"
        assert by_id[good_run_id]["status"] == "completed"
        print(f"    stuck run status={by_id[stuck_run_id]['status']!r}, "
              f"good run status={by_id[good_run_id]['status']!r} — both correct, no crash.")

        stuck_detail = get_versioned_run_detail(stuck_run_id)
        assert stuck_detail["status"] == "running"
        assert stuck_detail["finished_at"] is None
        good_detail = get_versioned_run_detail(good_run_id)
        assert good_detail["status"] == "completed"
        assert good_detail["finished_at"] is not None
        print("    get_versioned_run_detail() confirms both independently, unaffected by each other.")

        print("\n[4] Confirming the bug: is_training_running() currently reports True "
              "(would block all new training runs with a 409)...")
        assert training_service.is_training_running() is True, (
            "Expected the simulated crash to leave is_running=True before reconciliation"
        )
        print("    Confirmed: is_training_running() == True (the bug, before reconciliation).")

        print("\n[5] Running reconcile_stuck_training_runs() (the Stage 12 fix, normally called "
              "from backend/main.py's startup)...")
        result = training_service.reconcile_stuck_training_runs()
        print(f"    Result: {result}")
        assert result.get("reconciled") is True
        assert result.get("run_id") == stuck_run_id

        assert training_service.is_training_running() is False, (
            "is_training_running() should be False after reconciliation"
        )
        print("    is_training_running() is now False — new training runs are no longer blocked.")

        reconciled_detail = get_versioned_run_detail(stuck_run_id)
        assert reconciled_detail["status"] == "failed", (
            f"Expected the stuck run's summary to be finalized as 'failed', got {reconciled_detail['status']!r}"
        )
        assert reconciled_detail["finished_at"] is not None
        print(f"    Stuck run's own run_summary.json is now status={reconciled_detail['status']!r} "
              f"with finished_at set — no longer permanently 'running'.")

        good_detail_after = get_versioned_run_detail(good_run_id)
        assert good_detail_after == good_detail, "Reconciliation must not touch an already-completed run"
        print("    Genuinely completed run is byte-for-byte unaffected by reconciliation.")

        print("\n[6] Confirming reconciliation is a no-op on a clean process state "
              "(nothing left to reconcile after step 5)...")
        second_result = training_service.reconcile_stuck_training_runs()
        assert second_result == {"reconciled": False}
        print(f"    Result: {second_result} — correct, idempotent no-op.")

        print("\n" + "=" * 70)
        print("ALL E2E ASSERTIONS PASSED — Training Run Crash Recovery")
        print("=" * 70)

    finally:
        print("\n[Cleanup] Removing test run folders and restoring training_progress.json...")
        for run_dir in (stuck_run_dir, good_run_dir):
            if run_dir and run_dir.is_dir():
                shutil.rmtree(run_dir)
        training_service._write_progress_file(saved_progress_before_test)
        print("    Cleanup complete.")


if __name__ == "__main__":
    main()
