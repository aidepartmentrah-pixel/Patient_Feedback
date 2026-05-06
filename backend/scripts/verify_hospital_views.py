#!/usr/bin/env python3
"""
Verify HIS-facing views referenced in db_settings.json exist on SQL Server.

Uses the same TCP connection pattern as test_db_full.py.
After editing backend/config/db_settings.json, restart the NSSM backend service
so deployment_port reloads (imports happen at process start).

Usage (from repository root):
    python backend/scripts/verify_hospital_views.py

Exit code 0 if every configured view resolves with OBJECT_ID; 1 otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _connection_string(db: dict) -> str:
    host = db.get("host") or db.get("server", "127.0.0.1")
    port = db.get("port", 1433)
    return (
        f"Driver={{{db['driver']}}};"
        f"Server=tcp:{host},{port};"
        f"Database={db['database']};"
        f"UID={db['username']};"
        f"PWD={db['password']};"
        f"Encrypt=yes;"
        f"TrustServerCertificate=yes;"
    )


def main() -> int:
    try:
        import pyodbc
    except ImportError:
        print("ERROR: pyodbc is required. Install dependencies for this environment.")
        return 1

    cfg_path = ROOT / "backend" / "config" / "db_settings.json"
    if not cfg_path.is_file():
        print(f"ERROR: Missing config: {cfg_path}")
        return 1

    with open(cfg_path, encoding="utf-8") as f:
        config = json.load(f)

    db = config["database"]
    views_map = config.get("views") or {}

    print("=" * 60)
    print("HOSPITAL VIEW CHECK (OBJECT_ID)")
    print("=" * 60)
    print(f"Server: tcp:{db.get('host') or db.get('server')}:{db.get('port', 1433)}")
    print(f"Database: {db['database']}")
    print()

    conn_str = _connection_string(db)
    try:
        conn = pyodbc.connect(conn_str, timeout=15)
    except Exception as e:
        print(f"CONNECTION FAILED: {e}")
        return 1

    cursor = conn.cursor()
    failed = False

    for logical_name, view_name in sorted(views_map.items()):
        obj = f"dbo.{view_name}"
        cursor.execute("SELECT OBJECT_ID(?)", (obj,))
        oid = cursor.fetchone()[0]
        if oid is None:
            print(f"  FAIL  {logical_name} -> {obj}  (not found)")
            failed = True
        else:
            print(f"  OK    {logical_name} -> {obj}  (object_id={oid})")

    cursor.close()
    conn.close()

    print()
    if failed:
        print("At least one view is missing. Create/restore the view on this server,")
        print("or set views.* in backend/config/db_settings.json to the real names.")
        return 1

    print("All configured views exist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
