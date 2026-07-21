"""
ML Architecture Consolidation — Stage 1: Backup and Inventory

Read-only against the live system. Creates a fresh, checksummed archive copy of
every current ML SQLite file plus a fresh SQL Server .bak, and records a row/column
inventory baseline that every later migration-verification step compares against.

Run from the backend/ directory:
    python -m scripts.ml_stage1_backup_and_inventory

Does not modify patient_feedback_ml.db, training_metadata.db, or IncidentManager.
"""

import hashlib
import json
import os
import shutil
import sqlite3
from datetime import datetime

from core.database import get_connection

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARCHIVE_ROOT = r"C:\SQLBackup"

SQLITE_SOURCES = {
    "patient_feedback_ml.db": os.path.join(WORKSPACE_ROOT, "models_directory", "patient_feedback_ml.db"),
    "training_metadata.db": os.path.join(WORKSPACE_ROOT, "backend", "data", "training_metadata.db"),
}

EMBEDDING_COLUMNS = {
    "embedding_text1", "embedding_text2", "embedding_text3",
    "embedding_text123", "embedding_text23",
    "sentence_1_embedding", "sentence_2_embedding", "sentence_3_embedding",
    "sentence_4_embedding", "sentence_5_embedding", "sentence_6_embedding",
}


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory_sqlite(path: str) -> dict:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cur.fetchall()]

    result = {}
    for table in tables:
        cur.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cur.fetchall()]

        cur.execute(f"SELECT COUNT(*) FROM {table}")
        row_count = cur.fetchone()[0]

        embed_cols_present = [c for c in columns if c in EMBEDDING_COLUMNS]
        non_null_embed_counts = {}
        for col in embed_cols_present:
            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} IS NOT NULL")
            non_null_embed_counts[col] = cur.fetchone()[0]

        sample_text = None
        if "complaint_text" in columns:
            cur.execute(f"SELECT complaint_text FROM {table} WHERE complaint_text IS NOT NULL LIMIT 1")
            row = cur.fetchone()
            if row:
                sample_text = (row[0] or "")[:80]

        result[table] = {
            "columns": columns,
            "row_count": row_count,
            "non_null_embedding_counts": non_null_embed_counts,
            "sample_complaint_text_prefix": sample_text,
        }

    conn.close()
    return result


def backup_sql_server(archive_dir: str, timestamp: str) -> dict:
    bak_filename = f"IncidentManager_ml_stage1_{timestamp}.bak"
    bak_path = os.path.join(ARCHIVE_ROOT, bak_filename)

    conn = get_connection()
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(
        "BACKUP DATABASE IncidentManager TO DISK = ? WITH INIT, CHECKSUM",
        (bak_path,),
    )
    while cur.nextset():
        pass

    cur.execute("SELECT COUNT(*) FROM dbo.APP_IncidentCase")
    case_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM sys.tables")
    table_count = cur.fetchone()[0]
    conn.close()

    return {
        "backup_file": bak_path,
        "app_incident_case_row_count": case_count,
        "table_count": table_count,
    }


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join(ARCHIVE_ROOT, f"ml_stage1_archive_{timestamp}")
    os.makedirs(archive_dir, exist_ok=True)

    print("=" * 70)
    print("ML ARCHITECTURE CONSOLIDATION — STAGE 1: BACKUP AND INVENTORY")
    print("=" * 70)

    manifest = {"timestamp": timestamp, "archive_dir": archive_dir, "sqlite_files": {}, "sql_server": None}

    # --- SQLite: copy + checksum + inventory ---
    for label, source_path in SQLITE_SOURCES.items():
        if not os.path.exists(source_path):
            print(f"\n[SKIP] {label} not found at {source_path}")
            manifest["sqlite_files"][label] = {"status": "not_found", "source_path": source_path}
            continue

        dest_path = os.path.join(archive_dir, label)
        print(f"\n[{label}]")
        print(f"  Source: {source_path}")

        source_checksum = sha256_of(source_path)
        shutil.copy2(source_path, dest_path)
        dest_checksum = sha256_of(dest_path)

        if source_checksum != dest_checksum:
            raise RuntimeError(f"Checksum mismatch after copying {label} — archive copy does not match source!")

        print(f"  Archived to: {dest_path}")
        print(f"  SHA-256: {source_checksum}")

        inventory = inventory_sqlite(source_path)
        for table, info in inventory.items():
            print(f"    Table {table}: {info['row_count']} rows")
            if info["non_null_embedding_counts"]:
                for col, cnt in info["non_null_embedding_counts"].items():
                    print(f"      {col}: {cnt} non-null")

        manifest["sqlite_files"][label] = {
            "status": "archived",
            "source_path": source_path,
            "archive_path": dest_path,
            "sha256": source_checksum,
            "inventory": inventory,
        }

    # --- SQL Server: fresh .bak + baseline counts ---
    print(f"\n[SQL Server — IncidentManager]")
    try:
        sql_info = backup_sql_server(archive_dir, timestamp)
        print(f"  Backup written to: {sql_info['backup_file']}")
        print(f"  Tables: {sql_info['table_count']}")
        print(f"  APP_IncidentCase rows: {sql_info['app_incident_case_row_count']}")
        manifest["sql_server"] = {"status": "backed_up", **sql_info}
    except Exception as e:
        print(f"  [ERROR] SQL Server backup failed: {e}")
        manifest["sql_server"] = {"status": "failed", "error": str(e)}

    # --- Write manifest ---
    manifest_path = os.path.join(archive_dir, "stage1_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"Manifest written to: {manifest_path}")
    print("Stage 1 complete. Original files untouched — this was a read-only, additive operation.")
    print("=" * 70)


if __name__ == "__main__":
    main()
