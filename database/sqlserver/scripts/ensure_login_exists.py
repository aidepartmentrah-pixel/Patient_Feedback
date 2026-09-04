"""
Creates PFMS's own dedicated SQL Server login (DATABASE_USER, e.g.
"pfms_user") the first time the database exists but the login doesn't yet
-- scoped to db_owner on PFMS's own database only, no server-level rights,
no CREATE DATABASE. Matches the shape already used for the real production
login on this app's legacy source VM (HCAT_Insight, per
database/docs/DATABASE_INSTALL_GUIDE.md), just expressed here as a real,
repeatable script instead of one-off manual provisioning.

Must run as `sa` (the only login that can CREATE LOGIN) -- after
install_database.py's own ensure_database_exists() (the target database
must already exist) and before run_install_scripts()/provision.py, both of
which connect as DB_USERNAME/DB_PASSWORD (the new dedicated login).

Connection comes from backend/config/db_settings.json (+ env var
overrides) for server/driver/database, same as install_database.py --
but username/password are deliberately NOT DB_USERNAME/DB_PASSWORD (those
name the new login this script is about to create, which doesn't exist
yet): sa's own credentials, read directly from MSSQL_SA_PASSWORD.

Usage:
    python database/sqlserver/scripts/ensure_login_exists.py
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BACKEND_DIR = REPO_ROOT / "backend"

sys.path.insert(0, str(BACKEND_DIR))
from core.deployment_port import DB_SERVER, DB_DATABASE, DB_DRIVER, TRUST_SERVER_CERTIFICATE  # noqa: E402
import pyodbc  # noqa: E402

_SA_PASSWORD = os.environ.get("MSSQL_SA_PASSWORD", "")
_NEW_LOGIN = os.environ.get("DATABASE_USER", "")
_NEW_LOGIN_PASSWORD = os.environ.get("DATABASE_PASSWORD", "")


def _conn_string(database: str) -> str:
    parts = [f"DRIVER={{{DB_DRIVER}}};", f"SERVER={DB_SERVER};", f"DATABASE={database};", f"UID=sa;PWD={_SA_PASSWORD};"]
    if TRUST_SERVER_CERTIFICATE:
        parts.append("TrustServerCertificate=yes;")
    return "".join(parts)


def main():
    if not _NEW_LOGIN or not _NEW_LOGIN_PASSWORD:
        sys.exit("DATABASE_USER and DATABASE_PASSWORD must both be set -- cannot create a login with no name/password.")

    master_conn = pyodbc.connect(_conn_string("master"), timeout=15, autocommit=True)
    try:
        cur = master_conn.cursor()
        cur.execute("SELECT name FROM sys.server_principals WHERE name = ?", _NEW_LOGIN)
        if cur.fetchone() is None:
            print(f"Creating SQL Server login '{_NEW_LOGIN}'...")
            # Real, live-found bug (confirmed against HCopilot's identical
            # script): CREATE LOGIN's WITH PASSWORD clause does not accept
            # a parameterized value at all -- SQL Server rejects a
            # `?`/sp_executesql placeholder here with "Incorrect syntax
            # near '@P1'" (102), unlike an ordinary DML statement. The
            # password must be a literal in the SQL text; single quotes are
            # doubled (T-SQL's own literal-escaping convention) rather than
            # trusted as injection-safe just because this password is
            # always alphanumeric today. CHECK_POLICY = OFF: a service
            # login never interactively changes its password, so Windows
            # password-expiration policy would eventually lock it out for
            # no real reason -- same real fleet precedent as Voice
            # Project's own 001_create_database.sql.
            escaped_password = _NEW_LOGIN_PASSWORD.replace("'", "''")
            cur.execute(f"CREATE LOGIN [{_NEW_LOGIN}] WITH PASSWORD = '{escaped_password}', CHECK_POLICY = OFF")
            print(f"Login '{_NEW_LOGIN}' created.")
        else:
            print(f"Login '{_NEW_LOGIN}' already exists, skipping.")
    finally:
        master_conn.close()

    db_conn = pyodbc.connect(_conn_string(DB_DATABASE), timeout=15, autocommit=True)
    try:
        cur = db_conn.cursor()
        cur.execute("SELECT name FROM sys.database_principals WHERE name = ?", _NEW_LOGIN)
        if cur.fetchone() is None:
            print(f"Creating database user '{_NEW_LOGIN}' in '{DB_DATABASE}', granting db_owner...")
            cur.execute(f"CREATE USER [{_NEW_LOGIN}] FOR LOGIN [{_NEW_LOGIN}]")
            # db_owner, not a narrower role: this same login runs the DDL
            # install pipeline (install_database.py) as well as the
            # backend's own runtime CRUD -- see
            # docs/decisions/database-identity-convention.md in the
            # Air-Gapped-System-Platform repo.
            cur.execute(f"ALTER ROLE db_owner ADD MEMBER [{_NEW_LOGIN}]")
            print(f"User '{_NEW_LOGIN}' created and granted db_owner on '{DB_DATABASE}'.")
        else:
            print(f"Database user '{_NEW_LOGIN}' already exists in '{DB_DATABASE}', skipping.")
    finally:
        db_conn.close()


if __name__ == "__main__":
    main()
