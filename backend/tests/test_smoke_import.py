"""
Smoke test for the Import Pipeline.

Queries the DB for real lookup values, builds a test Excel with:
  - Group IMP-SMOKE-001: 2 valid rows → should import
  - Group IMP-SMOKE-002: 1 row with invalid classification → should reject

Run from backend/ directory:
    python -m tests.test_smoke_import

Cleanup (roll back test data):
    DELETE FROM APP_AdministrativeSubcase WHERE CreatedByUserID = 1 AND CreatedAt > GETDATE()-0.01
    DELETE FROM APP_IncidentCaseTargetDepartment WHERE AssignedByUserID = 1
    DELETE FROM APP_IncidentCase WHERE CreatedByUserID = 1 AND incident_id IN (SELECT incident_id FROM APP_Incident WHERE created_by_user_id = 1 AND patient_name LIKE '%SMOKE TEST%')
    DELETE FROM APP_Incident WHERE created_by_user_id = 1 AND patient_name LIKE '%SMOKE TEST%'
    DELETE FROM APP_RESERVE_PATIENT WHERE FullName LIKE '%SMOKE TEST%'
"""

import sys
import os
import json
from io import BytesIO
from datetime import date

from openpyxl import Workbook

# ---- path setup so we can import from backend ----
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import get_connection
from api.services.import_service import process_upload, TEMPLATE_COLUMNS


# ============================================================
# STEP 1 — Fetch real lookup values from DB
# ============================================================

def fetch_test_values() -> dict:
    """Grab one valid value for each mandatory lookup from the live DB."""
    conn = get_connection()
    cur = conn.cursor()
    vals = {}

    try:
        cur.execute("SELECT TOP 1 NameAr FROM dbo.APP_LOOKUP_FEEDBACK_INTENT_TYPE WHERE IsActive=1 ORDER BY DisplayOrder")
        r = cur.fetchone(); vals["feedback_type"] = r[0] if r else None

        cur.execute("SELECT TOP 1 SourceNameAr FROM dbo.APP_LOOKUP_SOURCE WHERE IsActive=1 ORDER BY DisplayOrder")
        r = cur.fetchone(); vals["source"] = r[0] if r else None

        cur.execute("SELECT TOP 1 Name FROM dbo.AdminsrationUnit WHERE Frozen=0 ORDER BY Name")
        r = cur.fetchone(); vals["org_unit"] = r[0] if r else None

        cur.execute("""
            SELECT TOP 1 c.Classification_AR
            FROM dbo.APP_LOOKUP_CLASSIFICATION c
            WHERE c.IsActive = 1
            ORDER BY c.Classification_AR
        """)
        r = cur.fetchone(); vals["classification"] = r[0] if r else None

        cur.execute("SELECT TOP 1 SeverityName FROM dbo.APP_LOOKUP_SEVERITY ORDER BY SeverityID")
        r = cur.fetchone(); vals["severity"] = r[0] if r else None

        cur.execute("SELECT TOP 1 StageName FROM dbo.APP_LOOKUP_CASE_STAGE ORDER BY StageOrder")
        r = cur.fetchone(); vals["stage"] = r[0] if r else None

    finally:
        cur.close()
        conn.close()

    return vals


# ============================================================
# STEP 2 — Build test Excel in memory
# ============================================================

def build_test_excel(tv: dict) -> bytes:
    """
    Build a minimal valid-format Excel file matching the Import Template structure.
    Group IMP-SMOKE-001: 2 valid rows
    Group IMP-SMOKE-002: 1 row with broken classification (should reject)
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Import Template"

    headers = [col[0] for col in TEMPLATE_COLUMNS]
    ws.append(headers)

    today = date.today().isoformat()
    patient = "SMOKE TEST Patient"
    patient2 = "SMOKE TEST Patient Two"

    def row(**kwargs):
        """Build a row dict aligned to TEMPLATE_COLUMNS headers."""
        return {h: kwargs.get(h) for h in headers}

    rows = [
        # --- Group 1, Row 1 (valid) ---
        row(**{
            "Incident Group Key": "IMP-SMOKE-001",
            "Patient Name": patient,
            "Complaint Date (YYYY-MM-DD)": today,
            "Feedback Type (النوع)": tv["feedback_type"],
            "Source (المصدر)": tv["source"],
            "Issuing Dept (قسم الصادر)": tv["org_unit"],
            "Domain": None,
            "Category": None,
            "Subcategory": None,
            "Classification": tv["classification"],
            "Severity": tv["severity"],
            "Stage": tv["stage"],
            "Harm Level": None,
            "Feedback Risk Type": None,
            "Building": None,
            "Is Inpatient": "Yes",
            "Complaint Text": "Test complaint text — row 1 of group 001",
            "Immediate Action": "Immediate response taken",
            "Taken Action (الإجراءات المتخذة)": "Follow-up action completed",
            "Target Dept 1": tv["org_unit"],
            "Target Dept 2": None,
            "Target Dept 3": None,
            "Doctor 1": None,
            "Doctor 2": None,
            "Doctor 3": None,
            "Worker 1 (Full Name)": None,
            "Worker 2 (Full Name)": None,
            "Worker 3 (Full Name)": None,
        }),
        # --- Group 1, Row 2 (valid second case for same incident) ---
        row(**{
            "Incident Group Key": "IMP-SMOKE-001",
            "Patient Name": patient,
            "Complaint Date (YYYY-MM-DD)": today,
            "Feedback Type (النوع)": tv["feedback_type"],
            "Source (المصدر)": tv["source"],
            "Issuing Dept (قسم الصادر)": tv["org_unit"],
            "Domain": None,
            "Category": None,
            "Subcategory": None,
            "Classification": tv["classification"],
            "Severity": None,
            "Stage": None,
            "Harm Level": None,
            "Feedback Risk Type": None,
            "Building": None,
            "Is Inpatient": "No",
            "Complaint Text": "Test complaint text — row 2 of group 001",
            "Immediate Action": None,
            "Taken Action (الإجراءات المتخذة)": None,
            "Target Dept 1": tv["org_unit"],
            "Target Dept 2": None,
            "Target Dept 3": None,
            "Doctor 1": None,
            "Doctor 2": None,
            "Doctor 3": None,
            "Worker 1 (Full Name)": None,
            "Worker 2 (Full Name)": None,
            "Worker 3 (Full Name)": None,
        }),
        # --- Group 2, Row 1 (invalid classification → should reject) ---
        row(**{
            "Incident Group Key": "IMP-SMOKE-002",
            "Patient Name": patient2,
            "Complaint Date (YYYY-MM-DD)": today,
            "Feedback Type (النوع)": tv["feedback_type"],
            "Source (المصدر)": tv["source"],
            "Issuing Dept (قسم الصادر)": tv["org_unit"],
            "Domain": None,
            "Category": None,
            "Subcategory": None,
            "Classification": "INVALID_CLASSIFICATION_THAT_DOES_NOT_EXIST_XYZ",
            "Severity": None,
            "Stage": None,
            "Harm Level": None,
            "Feedback Risk Type": None,
            "Building": None,
            "Is Inpatient": "No",
            "Complaint Text": "This group should be rejected due to invalid classification",
            "Immediate Action": None,
            "Taken Action (الإجراءات المتخذة)": None,
            "Target Dept 1": tv["org_unit"],
            "Target Dept 2": None,
            "Target Dept 3": None,
            "Doctor 1": None,
            "Doctor 2": None,
            "Doctor 3": None,
            "Worker 1 (Full Name)": None,
            "Worker 2 (Full Name)": None,
            "Worker 3 (Full Name)": None,
        }),
    ]

    for r in rows:
        ws.append([r.get(h) for h in headers])

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ============================================================
# STEP 3 — Run the import and print the report
# ============================================================

def main():
    print("\n" + "="*65)
    print("  IMPORT PIPELINE SMOKE TEST")
    print("="*65)

    print("\n[1] Fetching real lookup values from DB...")
    tv = fetch_test_values()
    for k, v in tv.items():
        print(f"    {k:20s} = {v}")

    missing = [k for k, v in tv.items() if v is None]
    if missing:
        print(f"\nERROR: Missing lookup values: {missing}")
        print("DB may not have data in required lookup tables.")
        sys.exit(1)

    print("\n[2] Building test Excel...")
    excel_bytes = build_test_excel(tv)
    print(f"    Excel size: {len(excel_bytes):,} bytes")

    print("\n[3] Running import pipeline (created_by_user_id=1)...")
    report = process_upload(excel_bytes, created_by_user_id=1)

    print("\n" + "="*65)
    print("  IMPORT REPORT")
    print("="*65)

    s = report["summary"]
    print(f"\n  Total Groups      : {s['total_groups']}")
    print(f"  Imported Groups   : {s['imported_groups']}")
    print(f"  Rejected Groups   : {s['rejected_groups']}")
    print(f"  Total Rows        : {s['total_rows']}")
    print(f"  Imported Rows     : {s['imported_rows']}")
    print(f"  Rejected Rows     : {s['rejected_rows']}")
    print(f"  New Patients      : {s['new_patients_created']}")
    print(f"  Warnings          : {s['warnings_count']}")

    if report["imported"]:
        print("\n  [IMPORTED GROUPS]")
        for g in report["imported"]:
            print(f"    {g['group_key']} -> incident_id={g['incident_id']}, "
                  f"rows={g['rows_imported']}, subcases={g['subcases_created']}, "
                  f"new_patient={g['new_patient']}")

    if report["rejected"]:
        print("\n  [REJECTED GROUPS]")
        for r in report["rejected"]:
            reason_safe = r['reason'].encode('cp1256', errors='replace').decode('cp1256')
            print(f"    {r['group_key']} -> {reason_safe}")

    if report["warnings"]:
        print("\n  [WARNINGS]")
        for w in report["warnings"]:
            msg_safe = w['message'].encode('cp1256', errors='replace').decode('cp1256')
            print(f"    {w['group_key']} -> {msg_safe}")

    if report["rejected_excel_b64"]:
        out_path = os.path.join(os.path.dirname(__file__), "rejected_smoke.xlsx")
        with open(out_path, "wb") as f:
            import base64
            f.write(base64.b64decode(report["rejected_excel_b64"]))
        print(f"\n  Rejected Excel saved to: {out_path}")

    # Assertions
    print("\n[4] Checking assertions...")
    assert s["imported_groups"] == 1, f"Expected 1 imported group, got {s['imported_groups']}"
    assert s["rejected_groups"] == 1, f"Expected 1 rejected group, got {s['rejected_groups']}"
    assert s["imported_rows"] == 2, f"Expected 2 imported rows, got {s['imported_rows']}"
    assert s["rejected_rows"] == 1, f"Expected 1 rejected row, got {s['rejected_rows']}"
    assert report["imported"][0]["subcases_created"] == 2, \
        f"Expected 2 subcases (one per row), got {report['imported'][0]['subcases_created']}"

    print("    All assertions passed.")

    print("\n" + "="*65)
    print("  SMOKE TEST COMPLETE")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
