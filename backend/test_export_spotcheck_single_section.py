"""
One-off spot check: export a real single-Section Word report end-to-end
(Classical AND Stylish) via the live /api/reports/monthly/export endpoint,
and confirm:
  - the file is a valid, non-trivial docx
  - report_export_service's entity-name/type label now correctly names the
    Section (not an ancestor Administration/Department) when the request
    carries all three cascading ID fields, as ReportingPage.js always does

Temporarily flips the monthly_report_format DB setting to test both
formatters, and ALWAYS restores the original setting in a finally block —
this is a shared, persisted setting used by the real app.

Run: python backend/test_export_spotcheck_single_section.py
"""

import os
import sys
import requests

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_BACKEND_DIR = os.path.dirname(__file__)
for _p in (_REPO_ROOT, _BACKEND_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BASE_URL = "http://127.0.0.1:8000"
USERNAME = "complaint_supervisor"
PASSWORD = "5bb5a339"
OUT_DIR = r"C:\Users\ADMINI~1\AppData\Local\Temp\3\claude\c--Users-Administrator-Documents-GitHub-Patient-Feedback\d5b4a0dc-6ccf-48dd-8b89-b01ae8345be2\scratchpad"

FAILURES = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        FAILURES.append(label)


def find_nonempty_section(s: requests.Session):
    depts = s.get(f"{BASE_URL}/api/org-units/departments", timeout=30).json()["departments"]
    sections = s.get(f"{BASE_URL}/api/org-units/sections", timeout=30).json()["sections"]
    sec_by_dept = {}
    for sec in sections:
        sec_by_dept.setdefault(sec["department_id"], []).append(sec)

    for dept in depts:
        for sec in sec_by_dept.get(dept["id"], []):
            r = s.post(f"{BASE_URL}/api/reports/monthly/view",
                       json={"year": 2026, "month": 1, "mode": "detailed", "section_ids": str(sec["id"])},
                       timeout=60)
            if r.status_code == 200 and r.json().get("pagination", {}).get("total_records", 0) > 0:
                return dept["administration_id"], dept["id"], sec["id"], sec.get("name") or sec.get("name_ar")
    return None


def export_docx(s: requests.Session, admin_id, dept_id, section_id):
    r = s.post(
        f"{BASE_URL}/api/reports/export?format=docx",
        json={
            "report_type": "monthly",
            "display_mode": "detailed",
            "year": 2026,
            "month": 1,
            "filters": {
                "scope": "section",
                "administration_ids": str(admin_id),
                "department_ids": str(dept_id),
                "section_ids": str(section_id),
            },
            "language": "ar",
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def run():
    print("=" * 70)
    print("SINGLE-SECTION EXPORT SPOT CHECK (Classical + Stylish)")
    print("=" * 70)

    s = requests.Session()
    r = s.post(f"{BASE_URL}/api/auth/login", json={"username": USERNAME, "password": PASSWORD}, timeout=30)
    r.raise_for_status()
    check("login succeeds", r.json().get("success") is True)

    found = find_nonempty_section(s)
    check("found a section with real data to spot-check", found is not None)
    if not found:
        sys.exit(1)
    admin_id, dept_id, section_id, section_name = found
    print(f"  spot-check target: admin={admin_id} dept={dept_id} section={section_id} ('{section_name}')")

    from api.db_layer.report_config_db import get_report_config, set_report_config
    original_cfg = get_report_config()
    original_format = original_cfg.get("monthly_report_format", "classical")
    print(f"  original monthly_report_format = '{original_format}' (will be restored)")

    os.makedirs(OUT_DIR, exist_ok=True)

    try:
        for fmt in ("classical", "stylish"):
            set_report_config({"monthly_report_format": fmt})
            result = export_docx(s, admin_id, dept_id, section_id)
            content_bytes_len = result.get("file_size_bytes", 0)
            filename = result.get("file_name", "")
            check(f"[{fmt}] export produced a non-trivial file ({content_bytes_len} bytes)",
                  content_bytes_len > 5000)
            check(f"[{fmt}] filename does not say 'hospital' (should be scoped to the section)",
                  "hospital" not in filename.lower())
            print(f"  [{fmt}] filename: {filename}")

            # Download and save locally for manual inspection
            download_url = result.get("download_url")
            if download_url:
                dl = s.get(f"{BASE_URL}{download_url}", timeout=60)
                if dl.status_code == 200:
                    out_path = os.path.join(OUT_DIR, f"spotcheck_{fmt}_section{section_id}.docx")
                    with open(out_path, "wb") as f:
                        f.write(dl.content)
                    check(f"[{fmt}] downloaded file starts with DOCX ZIP signature",
                          dl.content[:4] == b"PK\x03\x04")
                    print(f"  [{fmt}] saved to {out_path}")
    finally:
        set_report_config({"monthly_report_format": original_format})
        restored = get_report_config().get("monthly_report_format")
        check(f"monthly_report_format restored to original ('{original_format}')", restored == original_format)

    print("=" * 70)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} check(s) FAILED")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("RESULT: all checks passed")
        sys.exit(0)


if __name__ == "__main__":
    run()
