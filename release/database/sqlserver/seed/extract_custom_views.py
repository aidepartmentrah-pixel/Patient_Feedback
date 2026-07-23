"""
Stage A (authenticated part) -- pulls the real, currently-active Custom Table
Views from the old HCAT system's own /api/custom-views endpoint (requires a
logged-in session, unlike the org-unit endpoints). Logs out immediately
after. Performs no write, update, delete, or migration action against the
source system at any point -- GET only.

Only ACTIVE views are pulled (active_only=true, the default) -- confirmed via
manual inspection that the source system also has ~21 deactivated
test/experimental views (names like "j", "all", "Koussa", duplicated
"Abbass" entries, gibberish text) mixed in with the 11 real ones staff
actually use. Migrating the deactivated rows would pollute a clean hospital
deployment with test cruft, so this intentionally does not pass
active_only=false.

Credential handling matches extract_source_data.py:
  - Username/password from HCAT_EXTRACT_USERNAME / HCAT_EXTRACT_PASSWORD env
    vars, or an interactive non-echoing prompt (getpass) if unset.
  - Never a CLI argument, never logged, never written to any file here.

Usage:
    python database/sqlserver/seed/extract_custom_views.py
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
    password = None  # noqa: F841

    if login_resp.status_code != 200:
        print(f"ERROR: login failed with HTTP {login_resp.status_code}")
        print("(No source-system data was modified. Nothing further will be attempted.)")
        sys.exit(1)

    login_role = login_resp.json().get("user", {}).get("scopes", [{}])[0].get("role_code", "?")
    print(f"  Logged in OK (role: {login_role})")

    print("Fetching /api/custom-views (active_only=true) ...")
    views_resp = session.get(f"{SOURCE_BASE}/api/custom-views", timeout=30)

    if views_resp.status_code != 200:
        print(f"ERROR: fetch failed with HTTP {views_resp.status_code}: {views_resp.text[:500]}")
        views_data = None
    else:
        views_data = views_resp.json()
        print(f"  OK ({views_resp.status_code}) -- {views_data.get('total', '?')} active views returned")

    print("Logging out (discarding session) ...")
    logout_resp = session.post(f"{SOURCE_BASE}/api/auth/logout", headers={"Content-Length": "0"}, timeout=10)
    print(f"  Logout HTTP {logout_resp.status_code}")
    session.close()

    if views_data is None:
        sys.exit(1)

    views = views_data.get("views", [])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "custom_views.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(views, f, ensure_ascii=False, indent=2)

    manifest_path = OUT_DIR / "custom_views_manifest.json"
    manifest = {
        "source_url": SOURCE_BASE,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "extraction_account_role": login_role,
        "endpoint": "/api/custom-views",
        "http_status": views_resp.status_code,
        "active_only": True,
        "total_views": len(views),
        "view_names": [v.get("ViewName") for v in views],
        "source_write_operations_performed": False,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {out_path}")
    print(f"Wrote {manifest_path}")
    print(f"\nSummary: {len(views)} active custom views extracted.")
    print("\nNo write, update, delete, or migration action was performed against the source system.")


if __name__ == "__main__":
    main()
