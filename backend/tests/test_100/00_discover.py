"""
00_discover.py — Verify all prerequisites and lock in IDs for the 100-record test.

Checks:
  - 5 test doctors (IDs 101-105)
  - 5 test workers (IDs 6-10)
  - 8 target sections (departments)
  - Classification chain: domain=3, category=1, subcat=1, class=78
  - Building ID=1 (RAH)
  - Issuing department ID=43 (cardiac 1)

Outputs: data/config.json
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.database import get_connection

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

PASS = "  [PASS]"
FAIL = "  [FAIL]"


def check(label, ok, detail=""):
    status = PASS if ok else FAIL
    print(f"{status}  {label}{(' — ' + detail) if detail else ''}")
    return ok


def main():
    print("=" * 60)
    print("00_DISCOVER — Prerequisites Check")
    print("=" * 60)

    conn = get_connection()
    cur = conn.cursor()
    all_ok = True

    # ── Doctors 101-105 ──────────────────────────────────────────
    print("\n[Doctors]")
    doctor_ids = [101, 102, 103, 104, 105]
    doctor_names = {}
    for did in doctor_ids:
        cur.execute("SELECT DoctorName FROM dbo.APP_LOOKUP_DOCTOR WHERE DoctorID=? AND IsActive=1", did)
        row = cur.fetchone()
        ok = row is not None
        all_ok &= ok
        name = row[0] if row else "NOT FOUND"
        doctor_names[did] = name
        check(f"Doctor {did}", ok, name)

    # ── Workers 6-10 ─────────────────────────────────────────────
    print("\n[Workers]")
    worker_ids = [6, 7, 8, 9, 10]
    worker_names = {}
    for wid in worker_ids:
        cur.execute("SELECT FullName FROM dbo.VW_HrEmployeeProfileView WHERE EmployeeID=?", wid)
        row = cur.fetchone()
        ok = row is not None
        all_ok &= ok
        name = row[0] if row else "NOT FOUND"
        worker_names[wid] = name
        check(f"Worker {wid}", ok, name)

    # ── 8 Target Sections ─────────────────────────────────────────
    print("\n[Target Sections]")
    section_ids = [43, 95, 93, 60, 72, 98, 309, 83]
    section_names = {}
    for sid in section_ids:
        cur.execute("SELECT Name FROM dbo.AdminsrationUnit WHERE UniqueID=?", sid)
        row = cur.fetchone()
        ok = row is not None
        all_ok &= ok
        name = row[0] if row else "NOT FOUND"
        section_names[sid] = name
        check(f"Section {sid}", ok, name)

    # ── Issuing department 43 ─────────────────────────────────────
    print("\n[Issuing Department]")
    cur.execute("SELECT Name FROM dbo.AdminsrationUnit WHERE UniqueID=43")
    row = cur.fetchone()
    ok = row is not None
    all_ok &= ok
    check("Issuing dept ID=43", ok, row[0] if row else "NOT FOUND")

    # ── Classification chain ──────────────────────────────────────
    print("\n[Classification Chain]")
    cur.execute("SELECT DomainName FROM dbo.APP_LOOKUP_DOMAIN WHERE DomainID=3")
    row = cur.fetchone()
    ok = row is not None; all_ok &= ok
    check("Domain ID=3", ok, row[0] if row else "NOT FOUND")

    cur.execute("SELECT CategoryName FROM dbo.APP_LOOKUP_CATEGORY WHERE CategoryID=1 AND DomainID=3")
    row = cur.fetchone()
    ok = row is not None; all_ok &= ok
    check("Category ID=1 under Domain 3", ok, row[0] if row else "NOT FOUND")

    cur.execute("SELECT SubCategoryName FROM dbo.APP_LOOKUP_SUBCATEGORY WHERE SubCategoryID=1 AND CategoryID=1")
    row = cur.fetchone()
    ok = row is not None; all_ok &= ok
    check("SubCategory ID=1 under Cat 1", ok, row[0] if row else "NOT FOUND")

    cur.execute("SELECT Classification_EN FROM dbo.APP_LOOKUP_CLASSIFICATION WHERE ClassificationID=78 AND SubCategoryID=1")
    row = cur.fetchone()
    ok = row is not None; all_ok &= ok
    check("Classification ID=78 under SubCat 1", ok, row[0] if row else "NOT FOUND")

    # ── Lookup IDs ────────────────────────────────────────────────
    print("\n[Lookup IDs]")
    lookups = [
        ("dbo.APP_LOOKUP_SEVERITY",        "SeverityID",        2, "Medium"),
        ("dbo.APP_LOOKUP_CASE_STAGE",       "StageID",           1, "Examination & Diagnosis"),
        ("dbo.APP_LOOKUP_HARM_LEVEL",       "HarmID",            1, "No Harm"),
        ("dbo.APP_LOOKUP_SOURCE",           "SourceID",          1, "Tours"),
        ("dbo.APP_LOOKUP_BUILDING",         "BuildingID",        1, "RAH"),
        ("dbo.APP_LOOKUP_CLINICAL_RISK_TYPE","ClinicalRiskTypeID",1,"ORDINARY"),
        ("dbo.APP_LOOKUP_CLINICAL_RISK_TYPE","ClinicalRiskTypeID",2,"RED_FLAG"),
        ("dbo.APP_LOOKUP_CLINICAL_RISK_TYPE","ClinicalRiskTypeID",3,"NEVER_EVENT"),
    ]
    for table, col, val, label in lookups:
        cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {col}=?", val)
        ok = cur.fetchone()[0] > 0
        all_ok &= ok
        check(f"{label} (ID={val})", ok)

    cur.close()
    conn.close()

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_ok:
        print("RESULT: ALL CHECKS PASSED — Writing config.json")
    else:
        print("RESULT: SOME CHECKS FAILED — Fix before proceeding")

    config = {
        "doctor_ids":     doctor_ids,
        "doctor_names":   doctor_names,
        "worker_ids":     worker_ids,
        "worker_names":   worker_names,
        "section_ids":    section_ids,
        "section_names":  section_names,
        "issuing_dept_id": 43,
        "domain_id":       3,
        "category_id":     1,
        "subcategory_id":  1,
        "classification_id": 78,
        "severity_id":     2,
        "stage_id":        1,
        "harm_id":         1,
        "source_id":       1,
        "building_id":     1,
        "month1_date":    "2025-01-15",
        "month5_date":    "2025-05-15",
        "patient_prefix": "T100_P",
        "all_checks_passed": all_ok
    }

    config_path = os.path.join(DATA_DIR, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    print(f"Config saved -> {config_path}")
    print("=" * 60)
    return all_ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
