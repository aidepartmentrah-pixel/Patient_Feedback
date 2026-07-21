"""
Training Run Artifacts Service

Stage 10 of the ML architecture consolidation. Exposes the versioned
per-run folders Stage 9 introduced (models_directory/Classification_Models/
Maintainance/training_runs/{run_id}/run_summary.json) over HTTP: listing,
single-run detail, and ZIP download of a run's artifacts.

Deliberately separate from training_service.py, which owns the older,
unrelated SQLite-backed history (training_runs/model_metrics tables) that
powers the existing performance/timeline charts — that system is untouched.
"""

import json
import os
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

from models_directory.Classification_Models.Maintainance import run_versioning


class RunNotFoundError(Exception):
    pass


def _read_run_summary(run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / run_versioning.SUMMARY_FILENAME
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_versioned_runs(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Lightweight summary per run, newest first. run_id's YYYY_MM_DD_HHMMSS_...
    format sorts correctly as a plain string, so no date parsing is needed.
    Any run folder with a missing/corrupt run_summary.json is skipped rather
    than failing the whole list — one bad run should never hide the rest.
    """
    runs_root = run_versioning.RUNS_ROOT
    if not runs_root.is_dir():
        return []

    run_ids = sorted(
        (p.name for p in runs_root.iterdir() if p.is_dir()),
        reverse=True,
    )

    results = []
    for run_id in run_ids:
        if len(results) >= limit:
            break
        try:
            summary = _read_run_summary(runs_root / run_id)
        except Exception as e:
            print(f"[TRAINING RUN ARTIFACTS] Skipping {run_id}: could not read run_summary.json ({e})")
            continue

        models = summary.get("models", {})
        warnings_count = sum(1 for m in models.values() if m.get("warnings"))
        results.append({
            "run_id": summary.get("run_id", run_id),
            "status": summary.get("status"),
            "started_at": summary.get("started_at"),
            "finished_at": summary.get("finished_at"),
            "model_count": len(models),
            "families_with_warnings": warnings_count,
        })

    return results


def get_versioned_run_detail(run_id: str) -> Dict[str, Any]:
    """Full run_summary.json for one run. Raises RunNotFoundError if the
    run directory or its summary file doesn't exist."""
    run_versioning.validate_run_id(run_id)
    run_dir = run_versioning.RUNS_ROOT / run_id
    if not run_dir.is_dir():
        raise RunNotFoundError(f"Run not found: {run_id}")

    try:
        return _read_run_summary(run_dir)
    except FileNotFoundError:
        raise RunNotFoundError(f"Run {run_id} has no run_summary.json")


def build_run_zip(run_id: str) -> Dict[str, Any]:
    """
    Zips every artifact listed in a run's manifest (not "whatever's in the
    folder") — mirrors multi_report_export_service.py's exact BytesIO +
    zipfile.ZIP_DEFLATED pattern, reading real files off disk via
    zip_file.write() instead of that service's in-memory writestr().
    """
    summary = get_versioned_run_detail(run_id)
    run_dir = run_versioning.RUNS_ROOT / run_id

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("run_summary.json", json.dumps(summary, indent=2, default=str))
        for model_name, model_result in summary.get("models", {}).items():
            for artifact in model_result.get("artifacts", []):
                relative_path = artifact["relative_path"]
                full_path = run_dir / relative_path
                if full_path.is_file():
                    zip_file.write(full_path, arcname=relative_path)
                else:
                    print(f"[TRAINING RUN ARTIFACTS] {run_id}: manifest lists missing file {relative_path}")

    zip_buffer.seek(0)
    return {
        "filename": f"training_run_{run_id}.zip",
        "content": zip_buffer.getvalue(),
        "content_type": "application/zip",
    }
