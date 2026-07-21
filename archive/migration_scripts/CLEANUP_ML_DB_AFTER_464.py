"""
CLEANUP_ML_DB_AFTER_464.py
==========================
Deletes all records with ID > 464 from both ML databases.
Record 464 (موسى حسين كنعان) is the last valid record — everything after is bad data.

HOW TO USE:
  Step 1 — Run in PREVIEW mode first:
      python CLEANUP_ML_DB_AFTER_464.py

  Step 2 — If preview looks correct, run in DELETE mode:
      python CLEANUP_ML_DB_AFTER_464.py --delete
"""

import sqlite3
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent

# Both databases to clean
DATABASES = [
    {
        "path": WORKSPACE / "models_directory" / "patient_feedback_ml.db",
        "tables": ["patient_feedback_encoded"],
    },
]

LAST_VALID_ID = 464
DELETE_MODE = "--delete" in sys.argv


def get_existing_tables(conn):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return {row[0] for row in cur.fetchall()}


def preview_db(db_info):
    path = db_info["path"]
    if not path.exists():
        print(f"  [SKIP] File not found: {path}")
        return

    conn = sqlite3.connect(str(path))
    existing = get_existing_tables(conn)
    cur = conn.cursor()

    for table in db_info["tables"]:
        if table not in existing:
            print(f"  [SKIP] Table '{table}' not in {path.name}")
            continue

        # Show the anchor record (464) to confirm it is correct
        cur.execute(f"SELECT id, complaint_text FROM {table} WHERE id = ?", (LAST_VALID_ID,))
        anchor = cur.fetchone()
        print(f"\n  Table: {table}")
        if anchor:
            text_preview = (anchor[1] or "")[:100]
            print(f"  Anchor record 464: {text_preview}")
        else:
            print(f"  [WARNING] Record 464 NOT FOUND in {table}")

        # Count and list records that will be deleted
        cur.execute(f"SELECT id, complaint_text FROM {table} WHERE id > ? ORDER BY id", (LAST_VALID_ID,))
        rows = cur.fetchall()
        print(f"  Records to delete (id > 464): {len(rows)}")
        for row in rows:
            text_preview = (row[1] or "")[:80]
            print(f"    ID {row[0]}: {text_preview}")

    conn.close()


def delete_db(db_info):
    path = db_info["path"]
    if not path.exists():
        print(f"  [SKIP] File not found: {path}")
        return

    conn = sqlite3.connect(str(path))
    existing = get_existing_tables(conn)
    cur = conn.cursor()

    for table in db_info["tables"]:
        if table not in existing:
            print(f"  [SKIP] Table '{table}' not in {path.name}")
            continue

        cur.execute(f"SELECT COUNT(*) FROM {table} WHERE id > ?", (LAST_VALID_ID,))
        count = cur.fetchone()[0]

        cur.execute(f"DELETE FROM {table} WHERE id > ?", (LAST_VALID_ID,))
        conn.commit()
        print(f"  [DELETED] {count} rows from {table} in {path.name}")

        cur.execute(f"SELECT COUNT(*) FROM {table}")
        remaining = cur.fetchone()[0]
        print(f"  [OK] Rows remaining in {table}: {remaining}")

    conn.close()


print("=" * 60)
print("ML DATABASE CLEANUP — Records after ID 464")
print("=" * 60)

if not DELETE_MODE:
    print("\nMODE: PREVIEW (no changes made)")
    print("Run with --delete to actually delete\n")
    for db in DATABASES:
        print(f"\nDatabase: {db['path'].name}")
        preview_db(db)
    print("\n\nIf the above looks correct, run:")
    print("  python CLEANUP_ML_DB_AFTER_464.py --delete")
else:
    print("\nMODE: DELETE — this is permanent")
    for db in DATABASES:
        print(f"\nDatabase: {db['path'].name}")
        delete_db(db)
    print("\nDone.")
