"""
=============================================================================
CLOSED-LOOP EXPORT ENDPOINT VERIFICATION TEST
=============================================================================
Purpose: Verify the monthly export endpoint (POST /api/reports/monthly/export)
         returns correct, consistent data across all supported formats and
         matches the known test bench ground truth.

Endpoint Under Test:
    POST /api/reports/monthly/export?year=...&month=...&format=...&display_mode=...

Test Design:
  - Same test bench as the 78/78 view test:
    10 cases (IDs 492-501), 8 sections, 30 subcases
    5 cases in January 2026, 5 in December 2025
    Triangular distribution across sections

Validation Levels:
  Phase 0: Database ground truth (prerequisite)
  Phase 1: Format validity — csv, pdf, docx produce valid files
  Phase 2: View ↔ Export cross-validation — CSV row count = view total
  Phase 3: CSV test case presence — all 10 test cases found in exports
  Phase 4: CSV subcase integrity — section/dept/admin counts match ground truth
  Phase 5: Numeric mode DOCX validity

Org Hierarchy:
  Section (ID)              → Department (ID)                    → Administration (ID)
  cardiac 1 (43)            → دائرة العناية الفائقة (28)         → الادارة التمريضية (3)
  قسم بنك الدم (95)          → دائرة التحاليل المخبرية (16)      → الادارة الطبية (4)
  التدريب و التقييم (60)    → الموارد البشرية (9)               → الموارد البشرية (9)
  المباني (72)              → الإدارة الإدارية (10)             → الإدارة الإدارية (10)
  قسم ضبط العدوى (98)       → الجودة والسلامة (11)              → الجودة والسلامة (11)
  الجراحة القلبية (42)      → الادارة الطبية لمركز القلب (13)   → الادارة الطبية لمركز القلب (13)
  AI Section (309)          → دائرة المعلوماتية (21)            → الادارة العامة (1)
  قسم الفوترة (93)          → دائرة شؤون المرضى (6)            → الادارة المالية (2)

=============================================================================
"""

import requests
import json
import sys
import csv
import ast
import zipfile
from io import BytesIO, StringIO
from collections import defaultdict

BASE_URL = "http://localhost:8000"

# ========================================================================
# SECTION 1: HIERARCHY & GROUND TRUTH DEFINITIONS
# ========================================================================

SECTIONS = {
    43:  "cardiac 1",
    95:  "قسم بنك الدم",
    60:  "التدريب و التقييم",
    72:  "المباني",
    98:  "قسم ضبط العدوى",
    42:  "الجراحة القلبية",
    309: "AI Section",
    93:  "قسم الفوترة",
}

DEPARTMENTS = {
    28: "دائرة العناية الفائقة والأقسام المتخصصة",
    16: "دائرة التحاليل المخبرية والانسجة",
    9:  "الموارد البشرية",
    10: "الإدارة الإدارية",
    11: "الجودة والسلامة",
    13: "الادارة الطبية لمركز القلب",
    21: "دائرة المعلوماتية",
    6:  "دائرة شؤون المرضى",
}

ADMINISTRATIONS = {
    3:  "الادارة التمريضية",
    4:  "الادارة الطبية",
    9:  "الموارد البشرية",
    10: "الإدارة الإدارية",
    11: "الجودة والسلامة",
    13: "الادارة الطبية لمركز القلب",
    1:  "الادارة العامة",
    2:  "الادارة المالية",
}

SECTION_TO_DEPT = {43: 28, 95: 16, 60: 9, 72: 10, 98: 11, 42: 13, 309: 21, 93: 6}
SECTION_TO_ADMIN = {43: 3, 95: 4, 60: 9, 72: 10, 98: 11, 42: 13, 309: 1, 93: 2}

# Case IDs
MONTH1_CASE_IDS = [492, 493, 494, 495, 496]   # January 2026
MONTH12_CASE_IDS = [497, 498, 499, 500, 501]   # December 2025
ALL_CASE_IDS = MONTH1_CASE_IDS + MONTH12_CASE_IDS

# Section-level expected subcase counts
SECTION_EXPECTED_M1 = {43: 5, 95: 4, 60: 3, 72: 2, 98: 1, 42: 0, 309: 0, 93: 0}
SECTION_EXPECTED_M12 = {43: 0, 95: 0, 60: 0, 72: 1, 98: 2, 42: 3, 309: 4, 93: 5}
SECTION_EXPECTED_COMBINED = {43: 5, 95: 4, 60: 3, 72: 3, 98: 3, 42: 3, 309: 4, 93: 5}


def rollup(section_counts, mapping):
    """Roll up section-level counts to a higher level using the mapping."""
    result = defaultdict(int)
    for sec_id, count in section_counts.items():
        if count > 0:
            parent_id = mapping[sec_id]
            result[parent_id] += count
    return dict(result)


DEPT_EXPECTED_M1 = rollup(SECTION_EXPECTED_M1, SECTION_TO_DEPT)
DEPT_EXPECTED_M12 = rollup(SECTION_EXPECTED_M12, SECTION_TO_DEPT)
DEPT_EXPECTED_COMBINED = rollup(SECTION_EXPECTED_COMBINED, SECTION_TO_DEPT)

ADMIN_EXPECTED_M1 = rollup(SECTION_EXPECTED_M1, SECTION_TO_ADMIN)
ADMIN_EXPECTED_M12 = rollup(SECTION_EXPECTED_M12, SECTION_TO_ADMIN)
ADMIN_EXPECTED_COMBINED = rollup(SECTION_EXPECTED_COMBINED, SECTION_TO_ADMIN)

# Period definitions for iteration
PERIODS = [
    {
        "name": "January 2026",
        "short": "M1",
        "params": {"year": 2026, "month": 1},
        "range_params": None,
        "expected_case_ids": MONTH1_CASE_IDS,
        "expected_subcases": 15,
        "section_expected": SECTION_EXPECTED_M1,
        "dept_expected": DEPT_EXPECTED_M1,
        "admin_expected": ADMIN_EXPECTED_M1,
    },
    {
        "name": "December 2025",
        "short": "M12",
        "params": {"year": 2025, "month": 12},
        "range_params": None,
        "expected_case_ids": MONTH12_CASE_IDS,
        "expected_subcases": 15,
        "section_expected": SECTION_EXPECTED_M12,
        "dept_expected": DEPT_EXPECTED_M12,
        "admin_expected": ADMIN_EXPECTED_M12,
    },
    {
        "name": "Combined (Dec 2025 + Jan 2026)",
        "short": "COMB",
        "params": {"year": 2025},  # year for range queries
        "range_params": {"start_date": "2025-12-01", "end_date": "2026-01-31"},
        "expected_case_ids": ALL_CASE_IDS,
        "expected_subcases": 30,
        "section_expected": SECTION_EXPECTED_COMBINED,
        "dept_expected": DEPT_EXPECTED_COMBINED,
        "admin_expected": ADMIN_EXPECTED_COMBINED,
    },
]


# ========================================================================
# SECTION 2: HELPERS
# ========================================================================

def login():
    """Login and return session cookies."""
    resp = requests.post(f"{BASE_URL}/api/auth/login", json={
        "username": "software_admin",
        "password": "admin123"
    })
    if resp.status_code != 200:
        print(f"  LOGIN FAILED: {resp.status_code} {resp.text[:200]}")
        sys.exit(1)
    print(f"  Login successful")
    return resp.cookies


def build_export_params(period, fmt, display_mode="detailed"):
    """Build query params for the export endpoint."""
    params = {"format": fmt, "display_mode": display_mode}
    if period["range_params"]:
        params["year"] = period["params"]["year"]
        params["start_date"] = period["range_params"]["start_date"]
        params["end_date"] = period["range_params"]["end_date"]
    else:
        params["year"] = period["params"]["year"]
        params["month"] = period["params"]["month"]
    return params


def call_export(cookies, params):
    """Call POST /api/reports/monthly/export with given params."""
    resp = requests.post(
        f"{BASE_URL}/api/reports/monthly/export",
        params=params,
        cookies=cookies,
        timeout=120  # exports can be slow
    )
    return resp


def call_view_numeric(cookies, period, group_by="section"):
    """Call the view endpoint in numeric mode to get reference counts."""
    body = {
        "mode": "numeric",
        "group_by": group_by,
    }
    if period["range_params"]:
        body["year"] = period["params"]["year"]
        body["month"] = None
        body["start_date"] = period["range_params"]["start_date"]
        body["end_date"] = period["range_params"]["end_date"]
    else:
        body["year"] = period["params"]["year"]
        body["month"] = period["params"]["month"]

    resp = requests.post(
        f"{BASE_URL}/api/reports/monthly/view",
        json=body,
        cookies=cookies,
        timeout=60
    )
    if resp.status_code != 200:
        return None
    return resp.json()


def parse_csv_content(content_bytes):
    """Parse CSV bytes into a list of dicts."""
    text = content_bytes.decode("utf-8-sig")  # handle BOM
    reader = csv.DictReader(StringIO(text))
    return list(reader)


def parse_target_departments(td_str):
    """Parse the stringified target_departments list from CSV."""
    if not td_str or td_str.strip() in ("", "[]", "None"):
        return []
    try:
        return ast.literal_eval(td_str)
    except (ValueError, SyntaxError):
        return []


def is_valid_pdf(content):
    """Check if content starts with PDF signature."""
    return content[:5] == b"%PDF-"


def is_valid_zip(content):
    """Check if content is a valid ZIP (docx/xlsx are ZIP-based)."""
    return content[:4] == b"PK\x03\x04"


# ========================================================================
# SECTION 3: DATABASE GROUND TRUTH
# ========================================================================

def verify_database_ground_truth():
    """Directly query the database to confirm test data matches expectations."""
    print("\n" + "=" * 70)
    print("PHASE 0: DATABASE GROUND TRUTH VERIFICATION")
    print("=" * 70)

    try:
        sys.path.insert(0, ".")
        from core.database import get_connection
        conn = get_connection()
        cursor = conn.cursor()

        all_pass = True

        # Check all 10 test cases exist
        print("\n  [0.1] Test cases exist in APP_IncidentCase...")
        cursor.execute("""
            SELECT IncidentRequestCaseID, FeedbackRecievedDate
            FROM dbo.APP_IncidentCase
            WHERE IncidentRequestCaseID IN (492,493,494,495,496,497,498,499,500,501)
            ORDER BY IncidentRequestCaseID
        """)
        rows = cursor.fetchall()
        found_ids = {r[0] for r in rows}
        missing = set(ALL_CASE_IDS) - found_ids
        status = "PASS" if not missing else "FAIL"
        if missing:
            all_pass = False
        print(f"    Found {len(found_ids)}/10 test cases -> {status}")
        if missing:
            print(f"    Missing: {sorted(missing)}")

        # Check target departments total
        print("\n  [0.2] Total subcases in APP_IncidentCaseTargetDepartment...")
        cursor.execute("""
            SELECT COUNT(*)
            FROM dbo.APP_IncidentCaseTargetDepartment
            WHERE IncidentRequestCaseID IN (492,493,494,495,496,497,498,499,500,501)
        """)
        total = cursor.fetchone()[0]
        status = "PASS" if total == 30 else "FAIL"
        if total != 30:
            all_pass = False
        print(f"    Subcases: {total} (expected 30) -> {status}")

        conn.close()

        if all_pass:
            print("\n  PHASE 0 RESULT: ALL DATABASE CHECKS PASSED")
        else:
            print("\n  PHASE 0 RESULT: SOME CHECKS FAILED - export tests may fail")

        return all_pass

    except Exception as e:
        print(f"\n  DATABASE CHECK ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


# ========================================================================
# SECTION 4: PHASE 1 — FORMAT VALIDITY
# ========================================================================

def run_phase1_format_validity(cookies):
    """Test that each format produces a valid response for each period."""
    print("\n" + "=" * 70)
    print("PHASE 1: EXPORT FORMAT VALIDITY")
    print("  Test: Each format returns HTTP 200 with valid, non-empty content")
    print("=" * 70)

    FORMATS = {
        "csv":  {"content_type": "text/csv",         "validator": lambda c: len(c) > 0},
        "pdf":  {"content_type": "application/pdf",   "validator": is_valid_pdf},
        "docx": {
            "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "validator": is_valid_zip
        },
    }

    passes = 0
    fails = 0

    for period in PERIODS:
        for fmt, spec in FORMATS.items():
            label = f"{period['short']}/{fmt.upper()}"
            params = build_export_params(period, fmt, display_mode="detailed")

            try:
                resp = call_export(cookies, params)

                checks = []
                # Check 1: HTTP 200
                checks.append(("HTTP 200", resp.status_code == 200))
                # Check 2: Non-empty
                checks.append(("Non-empty", len(resp.content) > 0))
                # Check 3: Valid file signature
                if resp.status_code == 200 and len(resp.content) > 0:
                    checks.append(("Valid file", spec["validator"](resp.content)))
                else:
                    checks.append(("Valid file", False))

                all_ok = all(ok for _, ok in checks)
                if all_ok:
                    passes += 1
                    print(f"  {label:<18} -> PASS  ({len(resp.content):>8} bytes)")
                else:
                    fails += 1
                    failed = [n for n, ok in checks if not ok]
                    print(f"  {label:<18} -> FAIL  (failed: {', '.join(failed)}, "
                          f"status={resp.status_code}, size={len(resp.content)})")

            except Exception as e:
                fails += 1
                print(f"  {label:<18} -> FAIL  (exception: {e})")

    print(f"\n  PHASE 1 SUBTOTAL: {passes} passed, {fails} failed (of {passes + fails})")
    return passes, fails


# ========================================================================
# SECTION 5: PHASE 2 — VIEW ↔ EXPORT CROSS-VALIDATION
# ========================================================================

def run_phase2_cross_validation(cookies):
    """Verify CSV export row count matches the view endpoint's total_records."""
    print("\n" + "=" * 70)
    print("PHASE 2: VIEW ↔ EXPORT CROSS-VALIDATION")
    print("  Test: CSV row count == view endpoint total_records")
    print("=" * 70)

    passes = 0
    fails = 0

    for period in PERIODS:
        label = period["short"]

        # Step A: Get view total
        view_data = call_view_numeric(cookies, period, group_by="section")
        if not view_data:
            print(f"  {label:<10} -> FAIL  (view endpoint returned no data)")
            fails += 1
            continue
        view_total = view_data.get("summary", {}).get("total_complaints", -1)

        # Step B: Get CSV export
        params = build_export_params(period, "csv", display_mode="detailed")
        resp = call_export(cookies, params)
        if resp.status_code != 200:
            print(f"  {label:<10} -> FAIL  (export returned {resp.status_code})")
            fails += 1
            continue

        csv_rows = parse_csv_content(resp.content)
        csv_count = len(csv_rows)

        # Step C: Compare
        if csv_count == view_total:
            passes += 1
            print(f"  {label:<10} -> PASS  (view={view_total}, csv={csv_count})")
        else:
            fails += 1
            print(f"  {label:<10} -> FAIL  (view={view_total}, csv={csv_count})")

    print(f"\n  PHASE 2 SUBTOTAL: {passes} passed, {fails} failed (of {passes + fails})")
    return passes, fails


# ========================================================================
# SECTION 6: PHASE 3 — TEST CASE PRESENCE IN CSV
# ========================================================================

def run_phase3_case_presence(cookies):
    """Verify all 10 test case IDs appear in the correct CSV exports."""
    print("\n" + "=" * 70)
    print("PHASE 3: TEST CASE PRESENCE IN CSV EXPORTS")
    print("  Test: Each expected case ID appears in the exported CSV")
    print("=" * 70)

    passes = 0
    fails = 0

    for period in PERIODS:
        label = period["short"]
        expected_ids = set(period["expected_case_ids"])

        params = build_export_params(period, "csv", display_mode="detailed")
        resp = call_export(cookies, params)
        if resp.status_code != 200:
            print(f"  {label:<10} -> FAIL  (export returned {resp.status_code})")
            fails += 1
            continue

        csv_rows = parse_csv_content(resp.content)
        found_ids = set()
        for row in csv_rows:
            try:
                case_id = int(row.get("id", 0))
                if case_id in expected_ids:
                    found_ids.add(case_id)
            except (ValueError, TypeError):
                pass

        missing = expected_ids - found_ids
        if not missing:
            passes += 1
            print(f"  {label:<10} -> PASS  (all {len(expected_ids)} test cases found)")
        else:
            fails += 1
            print(f"  {label:<10} -> FAIL  (missing case IDs: {sorted(missing)})")

    print(f"\n  PHASE 3 SUBTOTAL: {passes} passed, {fails} failed (of {passes + fails})")
    return passes, fails


# ========================================================================
# SECTION 7: PHASE 4 — CSV SUBCASE INTEGRITY (mirrors 78/78 test)
# ========================================================================

def extract_subcases_from_csv(csv_rows, test_case_ids):
    """
    Extract subcase counts from CSV target_departments for our test cases.
    Returns dicts: {section_id: count}, {dept_id: count}, {admin_id: count}
    """
    section_counts = defaultdict(int)
    dept_counts = defaultdict(int)
    admin_counts = defaultdict(int)

    for row in csv_rows:
        try:
            case_id = int(row.get("id", 0))
        except (ValueError, TypeError):
            continue

        if case_id not in test_case_ids:
            continue

        td_str = row.get("target_departments", "")
        targets = parse_target_departments(td_str)

        for td in targets:
            sec_id = td.get("section_id")
            dept_id = td.get("department_id")
            admin_id = td.get("administration_id")

            if sec_id is not None:
                section_counts[sec_id] += 1
            if dept_id is not None:
                dept_counts[dept_id] += 1
            if admin_id is not None:
                admin_counts[admin_id] += 1

    return dict(section_counts), dict(dept_counts), dict(admin_counts)


def compare_counts(actual, expected, names_map, level_name):
    """Compare actual vs expected counts. Return (passes, fails, details)."""
    p = 0
    f = 0
    details = []

    for unit_id, exp_count in expected.items():
        act_count = actual.get(unit_id, 0)
        if exp_count == 0 and act_count == 0:
            # Both zero — skip (not in the CSV is fine)
            p += 1
            details.append((unit_id, names_map.get(unit_id, "?"), exp_count, act_count, "PASS"))
            continue
        if act_count == exp_count:
            p += 1
            details.append((unit_id, names_map.get(unit_id, "?"), exp_count, act_count, "PASS"))
        else:
            f += 1
            details.append((unit_id, names_map.get(unit_id, "?"), exp_count, act_count, "FAIL"))

    return p, f, details


def run_phase4_subcase_integrity(cookies):
    """
    Parse CSV target_departments for test cases and verify subcase counts
    match the mathematical ground truth at all 3 hierarchy levels.
    """
    print("\n" + "=" * 70)
    print("PHASE 4: CSV SUBCASE INTEGRITY (3 LEVELS)")
    print("  Test: target_departments subcases match ground truth")
    print("=" * 70)

    total_passes = 0
    total_fails = 0

    LEVEL_CONFIGS = [
        ("Section", SECTIONS, "section"),
        ("Department", DEPARTMENTS, "department"),
        ("Administration", ADMINISTRATIONS, "administration"),
    ]

    for period in PERIODS:
        label = period["short"]
        test_ids = set(period["expected_case_ids"])

        # Get CSV
        params = build_export_params(period, "csv", display_mode="detailed")
        resp = call_export(cookies, params)
        if resp.status_code != 200:
            print(f"\n  {label}: EXPORT FAILED ({resp.status_code})")
            total_fails += len(SECTIONS) + len(DEPARTMENTS) + len(ADMINISTRATIONS)
            continue

        csv_rows = parse_csv_content(resp.content)
        sec_counts, dept_counts, admin_counts = extract_subcases_from_csv(csv_rows, test_ids)
        actual_maps = {
            "section": sec_counts,
            "department": dept_counts,
            "administration": admin_counts,
        }

        for level_name, names_map, level_key in LEVEL_CONFIGS:
            expected_key = f"{level_key[0:4] if level_key != 'section' else 'section'}_expected"
            # Get expected from period
            if level_key == "section":
                expected = period["section_expected"]
            elif level_key == "department":
                expected = period["dept_expected"]
            else:
                expected = period["admin_expected"]

            actual = actual_maps[level_key]
            p, f, details = compare_counts(actual, expected, names_map, level_name)
            total_passes += p
            total_fails += f

            # Print table
            print(f"\n  --- {label} / {level_name} ---")
            print(f"    {'Name':<48} {'Exp':>5} {'Act':>5} {'Status':>6}")
            print(f"    {'-'*48} {'-'*5} {'-'*5} {'-'*6}")
            for uid, name, exp, act, status in details:
                marker = "  OK" if status == "PASS" else "FAIL"
                display = f"{name[:40]} (ID={uid})"
                if exp > 0 or act > 0:  # only show non-zero for clarity
                    print(f"    {display:<48} {exp:>5} {act:>5} {marker:>6}")

    print(f"\n  PHASE 4 SUBTOTAL: {total_passes} passed, {total_fails} failed "
          f"(of {total_passes + total_fails})")
    return total_passes, total_fails


# ========================================================================
# SECTION 8: PHASE 5 — NUMERIC MODE DOCX EXPORT
# ========================================================================

def run_phase5_numeric_docx(cookies):
    """Verify numeric mode DOCX exports produce valid Word documents."""
    print("\n" + "=" * 70)
    print("PHASE 5: NUMERIC MODE DOCX EXPORT VALIDITY")
    print("  Test: display_mode=numeric with format=docx produces valid DOCX")
    print("=" * 70)

    passes = 0
    fails = 0

    for period in PERIODS:
        label = period["short"]
        params = build_export_params(period, "docx", display_mode="numeric")

        try:
            resp = call_export(cookies, params)

            checks = []
            checks.append(("HTTP 200", resp.status_code == 200))
            checks.append(("Non-empty", len(resp.content) > 10))
            if resp.status_code == 200 and len(resp.content) > 10:
                checks.append(("Valid DOCX", is_valid_zip(resp.content)))
            else:
                checks.append(("Valid DOCX", False))

            all_ok = all(ok for _, ok in checks)
            if all_ok:
                passes += 1
                print(f"  {label:<10} -> PASS  ({len(resp.content):>8} bytes)")
            else:
                fails += 1
                failed = [n for n, ok in checks if not ok]
                print(f"  {label:<10} -> FAIL  (failed: {', '.join(failed)}, "
                      f"status={resp.status_code}, size={len(resp.content)})")

        except Exception as e:
            fails += 1
            print(f"  {label:<10} -> FAIL  (exception: {e})")

    print(f"\n  PHASE 5 SUBTOTAL: {passes} passed, {fails} failed (of {passes + fails})")
    return passes, fails


# ========================================================================
# SECTION 9: PHASE 6 — TOTAL SUBCASE SUM VERIFICATION
# ========================================================================

def run_phase6_subcase_totals(cookies):
    """
    Verify that the total number of subcases extracted from CSV matches
    the expected 15 (per month) or 30 (combined).
    """
    print("\n" + "=" * 70)
    print("PHASE 6: TOTAL SUBCASE COUNT VERIFICATION")
    print("  Test: Sum of all section subcases from CSV = expected total")
    print("=" * 70)

    passes = 0
    fails = 0

    for period in PERIODS:
        label = period["short"]
        test_ids = set(period["expected_case_ids"])
        expected_total = period["expected_subcases"]

        params = build_export_params(period, "csv", display_mode="detailed")
        resp = call_export(cookies, params)
        if resp.status_code != 200:
            print(f"  {label:<10} -> FAIL  (export returned {resp.status_code})")
            fails += 1
            continue

        csv_rows = parse_csv_content(resp.content)
        sec_counts, _, _ = extract_subcases_from_csv(csv_rows, test_ids)
        actual_total = sum(sec_counts.values())

        if actual_total == expected_total:
            passes += 1
            print(f"  {label:<10} -> PASS  (subcases: {actual_total} == {expected_total})")
        else:
            fails += 1
            print(f"  {label:<10} -> FAIL  (subcases: {actual_total} != {expected_total})")

    print(f"\n  PHASE 6 SUBTOTAL: {passes} passed, {fails} failed (of {passes + fails})")
    return passes, fails


# ========================================================================
# SECTION 10: MAIN
# ========================================================================

def main():
    print("=" * 70)
    print("  CLOSED-LOOP EXPORT ENDPOINT VERIFICATION TEST")
    print("  Endpoint: POST /api/reports/monthly/export")
    print("  Test Bench: 10 Cases | 8 Sections | 30 Subcases | 3 Levels")
    print("=" * 70)

    # Phase 0: Database ground truth
    db_ok = verify_database_ground_truth()
    if not db_ok:
        print("\n  WARNING: Database ground truth failed.")
        print("  Continuing anyway for diagnostics...\n")

    # Login
    print("\n" + "=" * 70)
    print("AUTHENTICATION")
    print("=" * 70)
    cookies = login()

    # Run all phases
    results = {}

    p, f = run_phase1_format_validity(cookies)
    results["Phase 1: Format Validity"] = (p, f)

    p, f = run_phase2_cross_validation(cookies)
    results["Phase 2: View↔Export Match"] = (p, f)

    p, f = run_phase3_case_presence(cookies)
    results["Phase 3: Case Presence"] = (p, f)

    p, f = run_phase4_subcase_integrity(cookies)
    results["Phase 4: Subcase Integrity"] = (p, f)

    p, f = run_phase5_numeric_docx(cookies)
    results["Phase 5: Numeric DOCX"] = (p, f)

    p, f = run_phase6_subcase_totals(cookies)
    results["Phase 6: Subcase Totals"] = (p, f)

    # ======================================================================
    # FINAL REPORT
    # ======================================================================
    total_passes = sum(r[0] for r in results.values())
    total_fails = sum(r[1] for r in results.values())
    total_tests = total_passes + total_fails
    pass_rate = (total_passes / total_tests * 100) if total_tests > 0 else 0

    print("\n" + "=" * 70)
    print("FINAL REPORT: CLOSED-LOOP EXPORT VERIFICATION")
    print("=" * 70)

    print(f"\n  {'Phase':<35} {'Passed':>8} {'Failed':>8} {'Total':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")
    for phase_name, (p, f) in results.items():
        total = p + f
        print(f"  {phase_name:<35} {p:>8} {f:>8} {total:>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'TOTAL':<35} {total_passes:>8} {total_fails:>8} {total_tests:>8}")
    print()

    print(f"  Database Ground Truth:  {'VALID' if db_ok else 'INVALID'}")
    print(f"  Tests Passed:           {total_passes}/{total_tests} ({pass_rate:.1f}%)")
    print(f"  Tests Failed:           {total_fails}/{total_tests}")

    if db_ok and total_fails == 0:
        print(f"""
  +----------------------------------------------------------------+
  |  CONCLUSION: {total_passes}/{total_tests} PASS — ALL PHASES VERIFIED                  |
  |                                                                |
  |  The monthly export endpoint correctly:                        |
  |    1. Produces valid CSV, PDF, DOCX files for all periods      |
  |    2. CSV row count matches view endpoint total_records        |
  |    3. All 10 test cases present in exported CSVs               |
  |    4. Subcase counts match ground truth at 3 hierarchy levels  |
  |       (Section / Department / Administration)                  |
  |    5. Numeric mode DOCX exports produce valid Word documents   |
  |    6. Total subcase sums are correct (15+15=30)                |
  |                                                                |
  |  Mathematical guarantee: Same test bench as 78/78 view test.   |
  |  Export data is consistent with view data.                     |
  +----------------------------------------------------------------+
        """)
    else:
        print(f"""
  +----------------------------------------------------------------+
  |  CONCLUSION: SOME TESTS FAILED                                 |
  |                                                                |
  |  Review FAIL entries above to identify the issue.              |
  |  The export endpoint may not be producing correct data or      |
  |  files for all format/period combinations.                     |
  +----------------------------------------------------------------+
        """)

    sys.exit(0 if (db_ok and total_fails == 0) else 1)


if __name__ == "__main__":
    main()
