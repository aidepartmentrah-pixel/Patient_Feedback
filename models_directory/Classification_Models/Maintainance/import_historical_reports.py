"""
Migration Script: Import Historical Training Reports to Database

Reads all classification_training_report_*.txt files from Performance_Reporting
and imports them into the training_metadata.db database.
"""

import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from backend.api.db_layer.training_db import (
    _init_training_db,
    _get_training_connection,
    store_training_run
)


REPORT_DIR = Path(__file__).parent / "Performance_Reporting"


def parse_report_file(filepath: str) -> dict:
    """
    Parse a training report text file and extract metrics.
    
    Returns:
        dict with run_date, models list
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Extract date from "Generated: DD_MM_YYYY"
    date_match = re.search(r"Generated:\s*(\d{2}_\d{2}_\d{4})", content)
    if not date_match:
        print(f"  [SKIP] Could not parse date from {filepath}")
        return None
    
    date_str = date_match.group(1)  # e.g., "26_02_2026"
    
    # Convert to run_id format: YYYY_MM_DD_0000
    day, month, year = date_str.split("_")
    run_id = f"{year}_{month}_{day}_0000"
    
    # Parse ISO date for started_at
    try:
        run_date = datetime.strptime(date_str, "%d_%m_%Y")
        started_at = run_date.isoformat()
    except:
        started_at = f"{year}-{month}-{day}T00:00:00"
    
    # Parse each model block
    models = []
    
    # Pattern to match model blocks
    model_pattern = re.compile(
        r"Model:\s*(\S+)\s*\n"
        r"\s*Training Records:\s*(\d+)\s*\n"
        r"\s*Classes:.*?\n"
        r"\s*Metrics:\s*\n"
        r"\s*Accuracy:\s*([\d.]+)\s*\n"
        r"\s*Precision:\s*([\d.]+)\s*\n"
        r"\s*Recall:\s*([\d.]+)\s*\n"
        r"\s*F1-Score:\s*([\d.]+)",
        re.MULTILINE
    )
    
    for match in model_pattern.finditer(content):
        model_name = match.group(1)
        num_records = int(match.group(2))
        accuracy = float(match.group(3))
        precision = float(match.group(4))
        recall = float(match.group(5))
        f1 = float(match.group(6))
        
        models.append({
            "model_name": model_name,
            "num_records": num_records,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "last_trained": started_at
        })
    
    return {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": started_at,  # Approximate
        "models": models
    }


def import_reports():
    """Import all historical report files into database."""
    
    print("=" * 60)
    print("IMPORTING HISTORICAL TRAINING REPORTS")
    print("=" * 60)
    
    # Initialize database
    _init_training_db()
    
    # Find all report files
    report_files = sorted(REPORT_DIR.glob("classification_training_report_*.txt"))
    
    if not report_files:
        print(f"No report files found in {REPORT_DIR}")
        return
    
    print(f"Found {len(report_files)} report files")
    
    imported = 0
    skipped = 0
    
    for filepath in report_files:
        print(f"\nProcessing: {filepath.name}")
        
        # Check if already imported
        conn = _get_training_connection()
        cursor = conn.cursor()
        
        parsed = parse_report_file(str(filepath))
        
        if not parsed:
            skipped += 1
            continue
        
        # Check if run_id exists
        cursor.execute(
            "SELECT run_id FROM training_runs WHERE run_id = ?",
            (parsed["run_id"],)
        )
        existing = cursor.fetchone()
        conn.close()
        
        if existing:
            print(f"  [SKIP] Run {parsed['run_id']} already exists in database")
            skipped += 1
            continue
        
        # Import the run
        try:
            store_training_run(
                run_id=parsed["run_id"],
                started_at=parsed["started_at"],
                finished_at=parsed["finished_at"],
                status="completed",
                models=parsed["models"]
            )
            print(f"  [OK] Imported {parsed['run_id']} with {len(parsed['models'])} models")
            imported += 1
        except Exception as e:
            print(f"  [ERROR] Failed to import: {e}")
            skipped += 1
    
    print("\n" + "=" * 60)
    print(f"IMPORT COMPLETE: {imported} imported, {skipped} skipped")
    print("=" * 60)


if __name__ == "__main__":
    import_reports()
