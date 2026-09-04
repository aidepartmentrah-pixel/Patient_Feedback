"""
Runs the fresh-install pipeline (database/sqlserver/install/001-011) against
a target SQL Server instance, in order, in one shot.

Usage:
    python database/sqlserver/scripts/install_database.py

Connection comes from backend/config/db_settings.json (+ env var overrides),
same as the running application -- point that config at your target SQL
Server (e.g. a local Docker container) before running this.

Idempotent: every generated install script uses IF NOT EXISTS / IF OBJECT_ID
IS NULL guards, so re-running after a partial failure is safe.
"""
import os
import sys
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
INSTALL_DIR = REPO_ROOT / "database" / "sqlserver" / "install"

sys.path.insert(0, str(BACKEND_DIR))
from core.deployment_port import (  # noqa: E402
    DB_SERVER, DB_DATABASE, DB_DRIVER,
    USE_WINDOWS_AUTH, DB_USERNAME, DB_PASSWORD,
    TRUST_SERVER_CERTIFICATE,
)
import pyodbc  # noqa: E402


def _conn_string(database: str, *, username: str = None, password: str = None) -> str:
    parts = [f"DRIVER={{{DB_DRIVER}}};", f"SERVER={DB_SERVER};", f"DATABASE={database};"]
    if USE_WINDOWS_AUTH:
        parts.append("Trusted_Connection=yes;")
    else:
        parts.append(f"UID={username if username is not None else DB_USERNAME};")
        parts.append(f"PWD={password if password is not None else DB_PASSWORD};")
    if TRUST_SERVER_CERTIFICATE:
        parts.append("TrustServerCertificate=yes;")
    return "".join(parts)


def ensure_database_exists():
    # Deliberately not DB_USERNAME/DB_PASSWORD: CREATE DATABASE is a
    # server-level operation only `sa` can perform, regardless of which
    # login this app's own runtime code otherwise connects as (see
    # ensure_login_exists.py, which creates that dedicated,
    # database-scoped login right after this function runs).
    print(f"[1/2] Connecting to 'master' on {DB_SERVER} to ensure database '{DB_DATABASE}' exists...")
    conn = pyodbc.connect(
        _conn_string("master", username="sa", password=os.environ.get("MSSQL_SA_PASSWORD", "")),
        timeout=15, autocommit=True,
    )
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM sys.databases WHERE name = ?", DB_DATABASE)
    if cur.fetchone():
        print(f"      Database '{DB_DATABASE}' already exists.")
        conn.close()
        return

    print(f"      Creating database '{DB_DATABASE}'...")
    try:
        cur.execute(f"CREATE DATABASE [{DB_DATABASE}]")
    except pyodbc.ProgrammingError as e:
        if "permission denied" in str(e).lower():
            conn.close()
            print(
                f"\nERROR: the configured login lacks CREATE DATABASE permission on {DB_SERVER}.\n"
                f"Either grant it dbcreator/sysadmin, or create the database yourself first\n"
                f"(e.g. connect as 'sa' on a fresh SQL Server container and run:\n"
                f"  CREATE DATABASE [{DB_DATABASE}];\n"
                f"), then re-run this script -- it will detect the existing database and skip\n"
                f"straight to the install scripts."
            )
            sys.exit(1)
        raise
    print(f"      Created.")
    conn.close()


def _split_statements(batch: str):
    """
    Splits a GO-batch into individual ;-terminated statements, executed one
    at a time. Needed because very large multi-statement batches (hundreds
    of IF/INSERT pairs in one execute() call) have been observed to silently
    fail to apply a trailing SET IDENTITY_INSERT OFF, even though the batch
    itself reports no error -- a pyodbc/ODBC driver quirk with big batches,
    not a SQL syntax problem. One statement per execute() avoids it entirely.
    Accumulates lines until one ends with ';', so a two-line
    "IF NOT EXISTS (...)\\n    INSERT ...;" pair stays together as one statement.

    BEGIN/END-aware: a "IF NOT EXISTS (...) BEGIN ... END" block can contain
    several ;-terminated statements (DECLARE/INSERT/SET/UPDATE) that share
    local variables and MUST run together in one execute() call, or the
    variables are out of scope by the second statement. Lines are only
    treated as statement boundaries while BEGIN/END nesting depth is 0 --
    matched as whole trimmed lines (this codebase's seed scripts always put
    BEGIN/END alone on their own line), not as a general T-SQL parser.
    """
    statements = []
    buf = []
    depth = 0
    for line in batch.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        buf.append(line)
        upper = stripped.upper()
        if upper == "BEGIN" or upper.endswith(" BEGIN"):
            depth += 1
        elif upper == "END":
            depth -= 1
        if depth == 0 and stripped.endswith(";"):
            statements.append("\n".join(buf))
            buf = []
    if buf:
        statements.append("\n".join(buf))
    return statements


def run_install_scripts():
    files = sorted(INSTALL_DIR.glob("0*.sql"))
    files = [f for f in files if not f.name.startswith("001_")]  # 001 is the placeholder handled above
    print(f"[2/2] Running {len(files)} install scripts against '{DB_DATABASE}' in order...")

    conn = pyodbc.connect(_conn_string(DB_DATABASE), timeout=30, autocommit=True)
    cur = conn.cursor()

    for f in files:
        sql = f.read_text(encoding="utf-8")
        batches = [b for b in re.split(r"(?m)^\s*GO\s*$", sql) if b.strip()]
        statement_count = sum(len(_split_statements(b)) for b in batches)
        print(f"    {f.name} ({len(batches)} batches, {statement_count} statements)...", end=" ")
        try:
            for batch in batches:
                for stmt in _split_statements(batch):
                    cur.execute(stmt)
            print("OK")
        except Exception as e:
            print(f"FAILED\n\n{f.name} failed on a statement:\n{e}\n")
            conn.close()
            sys.exit(1)

    conn.close()
    print("\nAll install scripts completed successfully.")


if __name__ == "__main__":
    print(f"Target: {DB_SERVER} / {DB_DATABASE} (driver: {DB_DRIVER})\n")
    ensure_database_exists()
    # PFMS's own dedicated SQL Server login (db_owner on its own database
    # only) must exist before run_install_scripts() connects as
    # DB_USERNAME/DB_PASSWORD below -- see ensure_login_exists.py's own
    # docstring. Imported rather than shelled out to as a separate process
    # step so it runs strictly between database creation and schema
    # install, sharing this script's own sys.path setup.
    import ensure_login_exists
    ensure_login_exists.main()
    run_install_scripts()
