"""
Stage A (authenticated part) -- logs into the old HCAT system's own normal
login flow (session cookie, same as the browser) using an existing
SOFTWARE_ADMIN/COMPLAINT_SUPERVISOR account, and pulls the full user/role/
org-unit-scope list via the app's own admin-only, read-only endpoint. Logs
out immediately after. Performs no write, update, delete, migration, or
password-reset action against the source system at any point.

Credential handling (see the approved migration plan):
  - Username/password are read ONLY from HCAT_EXTRACT_USERNAME /
    HCAT_EXTRACT_PASSWORD environment variables, or an interactive
    non-echoing prompt (getpass) if either is unset.
  - Never accepted as a CLI argument (would appear in shell history/process
    list), never logged, never written to any file by this script.
  - This script is meant to be run BY YOU, not relayed through an AI tool
    call, so the plaintext password never appears in any transcript.

Usage (run this yourself, in your own terminal):
    python database/sqlserver/seed/extract_source_data.py

    # or, to avoid the interactive prompt:
    export HCAT_EXTRACT_USERNAME=software_admin
    export HCAT_EXTRACT_PASSWORD='...'
    python database/sqlserver/seed/extract_source_data.py
    unset HCAT_EXTRACT_PASSWORD
"""
import getpass
import json
import os
import sys
import urllib3
from datetime import datetime, timezone
from pathlib import Path

import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SOURCE_BASE = "https://170.70.32.34"
OUT_DIR = Path(__file__).resolve().parent / "raw_extract"


def main():
    username = os.environ.get("HCAT_EXTRACT_USERNAME") or input("HCAT username: ").strip()
    password = os.environ.get("HCAT_EXTRACT_PASSWORD") or getpass.getpass("HCAT password (not echoed): ")

    session = requests.Session()
    session.verify = False

    print(f"Logging in to {SOURCE_BASE} as {username!r} ...")
    login_resp = session.post(
        f"{SOURCE_BASE}/api/auth/login",
        json={"username": username, "password": password},
        timeout=15,
    )
    # Drop the password from local scope immediately; nothing below needs it again.
    password = None  # noqa: F841

    if login_resp.status_code != 200:
        print(f"ERROR: login failed with HTTP {login_resp.status_code}")
        print("(No source-system data was modified. Nothing further will be attempted.)")
        sys.exit(1)

    login_role = login_resp.json().get("user", {}).get("role", "?")
    print(f"  Logged in OK (role: {login_role})")

    print("Fetching /api/admin/testing/user-credentials ...")
    cred_resp = session.get(f"{SOURCE_BASE}/api/admin/testing/user-credentials", timeout=30)

    if cred_resp.status_code != 200:
        print(f"ERROR: fetch failed with HTTP {cred_resp.status_code}: {cred_resp.text[:500]}")
        # Still attempt logout below before exiting.
        users = None
    else:
        users = cred_resp.json()
        print(f"  OK ({cred_resp.status_code}) -- {len(users)} accounts returned")

    print("Logging out (discarding session) ...")
    logout_resp = session.post(f"{SOURCE_BASE}/api/auth/logout", timeout=10)
    print(f"  Logout HTTP {logout_resp.status_code}")
    session.close()

    if users is None:
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "user_credentials.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

    with_password = sum(1 for u in users if u.get("test_password"))
    without_password = len(users) - with_password
    active = sum(1 for u in users if u.get("active"))

    manifest_path = OUT_DIR / "user_credentials_manifest.json"
    manifest = {
        "source_url": SOURCE_BASE,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "extraction_account_role": login_role,
        "endpoint": "/api/admin/testing/user-credentials",
        "http_status": cred_resp.status_code,
        "total_accounts": len(users),
        "accounts_with_recoverable_password": with_password,
        "accounts_without_recoverable_password": without_password,
        "active_accounts": active,
        "inactive_accounts": len(users) - active,
        "source_write_operations_performed": False,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {out_path}")
    print(f"Wrote {manifest_path}")
    print(f"\nSummary: {len(users)} accounts, {active} active, {with_password} with a "
          f"recoverable test password, {without_password} without (real bcrypt-secured "
          f"accounts -- will need fresh temporary passwords in Stage B).")
    print("\nNo write, update, delete, migration, or password-reset action was performed "
          "against the source system.")


if __name__ == "__main__":
    main()
