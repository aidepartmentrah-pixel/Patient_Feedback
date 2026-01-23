"""
Training Database Layer
Stores and retrieves training run metadata and ML database size history.
"""

import sqlite3
import os
from datetime import datetime, date
from typing import Dict, List, Any
from pathlib import Path

# Get workspace root (3 levels up from this file: db_layer -> api -> backend -> root)
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Store training metadata in SQLite
TRAINING_DB_PATH = os.path.join(
    str(WORKSPACE_ROOT),
    "backend",
    "data",
    "training_metadata.db"
)

# ML Database path
ML_DB_PATH = os.path.join(
    str(WORKSPACE_ROOT),
    "models_directory",
    "patient_feedback_ml.db"
)

# Ensure data directory exists
os.makedirs(os.path.dirname(TRAINING_DB_PATH), exist_ok=True)


def _get_training_connection():
    """Get SQLite connection for training metadata."""
    conn = sqlite3.connect(TRAINING_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_training_db():
    """Initialize training metadata database schema."""
    conn = _get_training_connection()
    cursor = conn.cursor()
    
    # Training runs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_runs (
            run_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT NOT NULL,
            models_trained INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Model metrics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            num_records INTEGER,
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1 REAL,
            last_trained TEXT,
            FOREIGN KEY (run_id) REFERENCES training_runs(run_id)
        )
    """)
    
    # ML database size history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ml_db_size_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_date TEXT NOT NULL,
            record_count INTEGER NOT NULL,
            recorded_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(record_date)
        )
    """)
    
    conn.commit()
    conn.close()


def store_training_run(run_id: str, started_at: str, finished_at: str, status: str, models: List[Dict[str, Any]]):
    """
    Store a training run and its model metrics.
    
    Args:
        run_id: Unique run identifier (format: YYYY_MM_DD_HHMM)
        started_at: ISO format timestamp when training started
        finished_at: ISO format timestamp when training finished
        status: "completed" or "failed"
        models: List of model metric dicts
    """
    _init_training_db()
    conn = _get_training_connection()
    cursor = conn.cursor()
    
    try:
        # Insert training run
        cursor.execute("""
            INSERT OR REPLACE INTO training_runs 
            (run_id, started_at, finished_at, status, models_trained)
            VALUES (?, ?, ?, ?, ?)
        """, (run_id, started_at, finished_at, status, len(models)))
        
        # Insert model metrics
        for model in models:
            cursor.execute("""
                INSERT INTO model_metrics 
                (run_id, model_name, num_records, accuracy, precision, recall, f1, last_trained)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                model.get('model_name'),
                model.get('num_records'),
                model.get('accuracy'),
                model.get('precision'),
                model.get('recall'),
                model.get('f1'),
                model.get('last_trained')
            ))
        
        conn.commit()
    finally:
        conn.close()


def get_latest_training_status() -> Dict[str, Any]:
    """
    Get the latest completed training run with all model metrics.
    
    Returns:
        Dict with last_run, status, and models list
    """
    _init_training_db()
    conn = _get_training_connection()
    cursor = conn.cursor()
    
    try:
        # Get latest run
        cursor.execute("""
            SELECT run_id, started_at, finished_at, status, models_trained
            FROM training_runs
            ORDER BY started_at DESC
            LIMIT 1
        """)
        
        run_row = cursor.fetchone()
        
        if not run_row:
            return {
                "last_run": None,
                "status": "never_run",
                "models": []
            }
        
        run_id = run_row['run_id']
        
        # Get metrics for this run
        cursor.execute("""
            SELECT model_name, num_records, accuracy, precision, recall, f1, last_trained
            FROM model_metrics
            WHERE run_id = ?
            ORDER BY model_name
        """, (run_id,))
        
        models = [dict(row) for row in cursor.fetchall()]
        
        return {
            "last_run": run_row['started_at'],
            "status": run_row['status'],
            "models": models
        }
    finally:
        conn.close()


def get_training_history() -> List[Dict[str, Any]]:
    """
    Get all training runs ordered by most recent first.
    
    Returns:
        List of training run dicts with metadata
    """
    _init_training_db()
    conn = _get_training_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT run_id, started_at, finished_at, status, models_trained
            FROM training_runs
            ORDER BY started_at DESC
            LIMIT 50
        """)
        
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def record_ml_db_size(record_count: int, record_date: str = None):
    """
    Record current ML database size for history tracking.
    
    Args:
        record_count: Number of records in ML database
        record_date: Date in YYYY-MM-DD format (default: today)
    """
    if record_date is None:
        record_date = date.today().isoformat()
    
    if record_count < 0:
        print(f"[ML DB SIZE] Warning: Invalid record count {record_count}, skipping")
        return
    
    _init_training_db()
    conn = _get_training_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO ml_db_size_history (record_date, record_count)
            VALUES (?, ?)
        """, (record_date, record_count))
        conn.commit()
        print(f"[ML DB SIZE] Recorded {record_count} records for date {record_date}")
    except Exception as e:
        print(f"[ML DB SIZE ERROR] Failed to record: {str(e)}")
        raise
    finally:
        conn.close()


def get_ml_db_size_history(days: int = 90) -> List[Dict[str, Any]]:
    """
    Get ML database size history for the last N days.
    
    Args:
        days: Number of days of history to return
    
    Returns:
        List of {date, records} dicts
    """
    _init_training_db()
    conn = _get_training_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT record_date as date, record_count as records
            FROM ml_db_size_history
            ORDER BY record_date DESC
            LIMIT ?
        """, (days,))
        
        history = [dict(row) for row in cursor.fetchall()]
        # Reverse to get chronological order
        return list(reversed(history))
    finally:
        conn.close()


def get_current_ml_db_size() -> int:
    """
    Get current number of records in ML database.
    
    Returns:
        Total record count from patient_feedback_encoded table
    """
    try:
        if not os.path.exists(ML_DB_PATH):
            print(f"[ML DB SIZE] Database not found at: {ML_DB_PATH}")
            return 0
        
        conn = sqlite3.connect(ML_DB_PATH)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='patient_feedback_encoded'
        """)
        
        if not cursor.fetchone():
            print(f"[ML DB SIZE] Table 'patient_feedback_encoded' not found")
            conn.close()
            return 0
        
        # Get count
        cursor.execute("SELECT COUNT(*) FROM patient_feedback_encoded")
        count = cursor.fetchone()[0]
        conn.close()
        
        print(f"[ML DB SIZE] Current size: {count} records")
        return count
        
    except Exception as e:
        print(f"[ML DB SIZE ERROR] Could not get ML DB size: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0
