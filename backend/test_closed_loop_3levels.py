"""
=============================================================================
CLOSED-LOOP REPORTING VERIFICATION TEST - 3 LEVELS
=============================================================================
Purpose: Verify the monthly reporting endpoint returns correct
         section-level subcase counts at ALL THREE org hierarchy levels:
         1. Section level (group_by=section)
         2. Department level (group_by=department)
         3. Administration level (group_by=administration)

Test Design:
  - 10 cases across 2 months (Jan 2026 & Dec 2025)
  - 8 target sections with a triangular distribution pattern
  - 30 total subcases (15 per month)
  - Each section maps to a unique department AND unique administration
  - Ground truth is computed for all 3 levels

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
from collections import defaultdict

BASE_URL = "http://localhost:8000"

# ========================================================================
# SECTION 1: HIERARCHY & GROUND TRUTH DEFINITIONS
# ========================================================================

# Section level (Type=324) - the leaf nodes
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

# Department level (parent of section)
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

# Administration level (grandparent, or parent if section→admin directly)
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

# Mapping: section_id → department_id → administration_id
SECTION_TO_DEPT = {43: 28, 95: 16, 60: 9, 72: 10, 98: 11, 42: 13, 309: 21, 93: 6}
SECTION_TO_ADMIN = {43: 3, 95: 4, 60: 9, 72: 10, 98: 11, 42: 13, 309: 1, 93: 2}

# Case IDs
MONTH1_CASE_IDS = [492, 493, 494, 495, 496]  # January 2026
MONTH12_CASE_IDS = [497, 498, 499, 500, 501]  # December 2025

# -----------------------------------------------------------------------
# Section-level expected counts (from test bench)
# -----------------------------------------------------------------------
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

# Department-level expected (rollup from sections)
DEPT_EXPECTED_M1 = rollup(SECTION_EXPECTED_M1, SECTION_TO_DEPT)
DEPT_EXPECTED_M12 = rollup(SECTION_EXPECTED_M12, SECTION_TO_DEPT)
DEPT_EXPECTED_COMBINED = rollup(SECTION_EXPECTED_COMBINED, SECTION_TO_DEPT)

# Administration-level expected (rollup from sections)
ADMIN_EXPECTED_M1 = rollup(SECTION_EXPECTED_M1, SECTION_TO_ADMIN)
ADMIN_EXPECTED_M12 = rollup(SECTION_EXPECTED_M12, SECTION_TO_ADMIN)
ADMIN_EXPECTED_COMBINED = rollup(SECTION_EXPECTED_COMBINED, SECTION_TO_ADMIN)


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


def get_monthly_report(cookies, year, month, group_by="section"):
    """Call POST /api/reports/monthly/view in numeric mode."""
    resp = requests.post(
        f"{BASE_URL}/api/reports/monthly/view",
        json={
            "year": year,
            "month": month,
            "mode": "numeric",
            "group_by": group_by,
        },
        cookies=cookies
    )
    if resp.status_code != 200:
        print(f"  REPORT FAILED: {resp.status_code} {resp.text[:500]}")
        return None
    return resp.json()


def get_monthly_report_range(cookies, start_date, end_date, group_by="section"):
    """Call POST /api/reports/monthly/view with date range in numeric mode."""
    resp = requests.post(
        f"{BASE_URL}/api/reports/monthly/view",
        json={
            "year": int(start_date[:4]),
            "month": None,
            "mode": "numeric",
            "start_date": start_date,
            "end_date": end_date,
            "group_by": group_by,
        },
        cookies=cookies
    )
    if resp.status_code != 200:
        print(f"  RANGE REPORT FAILED: {resp.status_code} {resp.text[:500]}")
        return None
    return resp.json()


def extract_dept_counts(report_data):
    """Extract {dept_id: count} map from by_department array."""
    dept_map = {}
    for dept in report_data.get("by_department", []):
        dept_id = dept.get("dayra_id")
        count = dept.get("count", 0)
        dept_map[dept_id] = count
    return dept_map


def compare_counts(actual_map, expected, names_map, label):
    """Compare actual vs expected counts. Return (pass_count, fail_count, details)."""
    passes = 0
    fails = 0
    details = []
    
    for unit_id, expected_count in expected.items():
        actual_count = actual_map.get(unit_id, 0)
        unit_name = names_map.get(unit_id, f"Unknown({unit_id})")
        
        if actual_count == expected_count:
            passes += 1
            status = "PASS"
        else:
            fails += 1
            status = "FAIL"
        
        details.append({
            "id": unit_id,
            "name": unit_name,
            "expected": expected_count,
            "actual": actual_count,
            "status": status
        })
    
    # Check for unexpected units
    expected_ids = set(expected.keys())
    unexpected = {k: v for k, v in actual_map.items() if k not in expected_ids}
    
    return passes, fails, details, unexpected


def print_table(details, label, level_name):
    """Print a formatted comparison table."""
    print(f"\n  {level_name + ' Name':<50} {'Expected':>8} {'Actual':>8} {'Status':>8}")
    print(f"  {'-'*50} {'-'*8} {'-'*8} {'-'*8}")
    for d in details:
        marker = "  OK" if d["status"] == "PASS" else " FAIL"
        name = f"{d['name']} (ID={d['id']})"
        print(f"  {name:<50} {d['expected']:>8} {d['actual']:>8} {marker:>8}")


# ========================================================================
# SECTION 3: DATABASE GROUND TRUTH
# ========================================================================

def verify_database_ground_truth():
    """Directly query the database to confirm test data matches expectations."""
    print("\n" + "="*70)
    print("PHASE 0: DATABASE GROUND TRUTH VERIFICATION")
    print("="*70)
    
    try:
        sys.path.insert(0, ".")
        from core.database import get_connection
        conn = get_connection()
        cursor = conn.cursor()
        
        all_pass = True
        
        # Check target departments per case
        print("\n[Check 0.1] Target departments per case...")
        cursor.execute("""
            SELECT IncidentRequestCaseID, DepartmentID
            FROM dbo.APP_IncidentCaseTargetDepartment
            WHERE IncidentRequestCaseID IN (492,493,494,495,496,497,498,499,500,501)
            ORDER BY IncidentRequestCaseID, DepartmentID
        """)
        case_depts = defaultdict(set)
        for r in cursor.fetchall():
            case_depts[r[0]].add(r[1])
        
        expected_case_depts = {
            492: {43}, 493: {43, 95}, 494: {43, 95, 60},
            495: {43, 95, 60, 72}, 496: {43, 95, 60, 72, 98},
            497: {93}, 498: {93, 309}, 499: {93, 309, 42},
            500: {93, 309, 42, 98}, 501: {93, 309, 42, 98, 72},
        }
        
        for case_id in sorted(expected_case_depts.keys()):
            actual = case_depts.get(case_id, set())
            expected = expected_case_depts[case_id]
            status = "PASS" if actual == expected else "FAIL"
            if status == "FAIL": all_pass = False
            print(f"  Case {case_id}: {sorted(actual)} -> {status}")
        
        # Check hierarchy mapping (handles 2-level and 3-level trees)
        print("\n[Check 0.2] Hierarchy mapping (Section -> Dept -> Admin)...")
        for sid in SECTIONS:
            cursor.execute("""
                SELECT s.ParentID as parent_id, p.Type as parent_type,
                       p.ParentID as grandparent_id
                FROM dbo.AdminsrationUnit s
                LEFT JOIN dbo.AdminsrationUnit p ON s.ParentID = p.UniqueID
                WHERE s.UniqueID = ?
            """, sid)
            r = cursor.fetchone()
            parent_id = r[0]
            parent_type = r[1]
            grandparent_id = r[2]
            
            # "Department" level = always the parent
            actual_dept = parent_id
            # "Administration" level = parent if it's Type=323, else grandparent
            actual_admin = parent_id if parent_type == 323 else grandparent_id
            
            exp_dept = SECTION_TO_DEPT[sid]
            exp_admin = SECTION_TO_ADMIN[sid]
            
            dept_ok = actual_dept == exp_dept
            admin_ok = actual_admin == exp_admin
            status = "PASS" if (dept_ok and admin_ok) else "FAIL"
            if not dept_ok or not admin_ok: all_pass = False
            
            hier = "2-lvl" if parent_type == 323 else "3-lvl"
            print(f"  Section {sid} ({SECTIONS[sid][:20]}): "
                  f"Dept={actual_dept}(exp={exp_dept}) Admin={actual_admin}(exp={exp_admin}) "
                  f"[{hier}] -> {status}")
        
        # Total subcases
        print("\n[Check 0.3] Total subcases...")
        cursor.execute("""
            SELECT COUNT(*) FROM dbo.APP_AdministrativeSubcase
            WHERE IncidentRequestCaseID IN (492,493,494,495,496,497,498,499,500,501)
        """)
        total = cursor.fetchone()[0]
        status = "PASS" if total == 30 else "FAIL"
        if total != 30: all_pass = False
        print(f"  Total subcases: {total} (expected 30) -> {status}")
        
        conn.close()
        
        if all_pass:
            print("\n  PHASE 0 RESULT: ALL DATABASE CHECKS PASSED")
        else:
            print("\n  PHASE 0 RESULT: SOME CHECKS FAILED")
        
        return all_pass
        
    except Exception as e:
        print(f"\n  DATABASE CHECK ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


# ========================================================================
# SECTION 4: API TESTS - ONE LEVEL
# ========================================================================

def run_level_test(cookies, level_name, group_by, names_map,
                   expected_m1, expected_m12, expected_combined,
                   test_prefix):
    """Run tests for one grouping level. Returns (passes, fails)."""
    
    total_passes = 0
    total_fails = 0
    
    # ------------------------------------------------------------------
    # Month 1 (January 2026)
    # ------------------------------------------------------------------
    print(f"\n  --- {test_prefix}.1: January 2026 (group_by={group_by}) ---")
    
    report = get_monthly_report(cookies, year=2026, month=1, group_by=group_by)
    if report:
        dept_map = extract_dept_counts(report)
        passes, fails, details, unexpected = compare_counts(
            dept_map, expected_m1, names_map, f"Month 1 - {level_name}")
        total_passes += passes
        total_fails += fails
        print_table(details, "Month 1", level_name)
        
        if unexpected:
            print(f"\n    UNEXPECTED entries: {len(unexpected)}")
            for uid, cnt in sorted(unexpected.items(), key=lambda x: -x[1])[:5]:
                print(f"      ID={uid}, count={cnt}")
            total_fails += 1
        else:
            print(f"\n    No unexpected entries -> PASS")
            total_passes += 1
        
        total_subcases = sum(dept_map.values())
        status = "PASS" if total_subcases == 15 else "FAIL"
        print(f"    Total target dept entries: {total_subcases} (expected 15) -> {status}")
        if status == "PASS": total_passes += 1
        else: total_fails += 1
    else:
        total_fails += len(expected_m1) + 2
    
    # ------------------------------------------------------------------
    # Month 12 (December 2025)
    # ------------------------------------------------------------------
    print(f"\n  --- {test_prefix}.2: December 2025 (group_by={group_by}) ---")
    
    report = get_monthly_report(cookies, year=2025, month=12, group_by=group_by)
    if report:
        dept_map = extract_dept_counts(report)
        passes, fails, details, unexpected = compare_counts(
            dept_map, expected_m12, names_map, f"Month 12 - {level_name}")
        total_passes += passes
        total_fails += fails
        print_table(details, "Month 12", level_name)
        
        if unexpected:
            print(f"\n    UNEXPECTED entries: {len(unexpected)}")
            for uid, cnt in sorted(unexpected.items(), key=lambda x: -x[1])[:5]:
                print(f"      ID={uid}, count={cnt}")
            total_fails += 1
        else:
            print(f"\n    No unexpected entries -> PASS")
            total_passes += 1
        
        total_subcases = sum(dept_map.values())
        status = "PASS" if total_subcases == 15 else "FAIL"
        print(f"    Total target dept entries: {total_subcases} (expected 15) -> {status}")
        if status == "PASS": total_passes += 1
        else: total_fails += 1
    else:
        total_fails += len(expected_m12) + 2
    
    # ------------------------------------------------------------------
    # Combined Range (Dec 2025 + Jan 2026) = Reference Result 7
    # ------------------------------------------------------------------
    print(f"\n  --- {test_prefix}.3: Combined Range (group_by={group_by}) ---")
    
    report = get_monthly_report_range(cookies, "2025-12-01", "2026-01-31", group_by=group_by)
    if report:
        dept_map = extract_dept_counts(report)
        passes, fails, details, unexpected = compare_counts(
            dept_map, expected_combined, names_map, f"Combined - {level_name}")
        total_passes += passes
        total_fails += fails
        print_table(details, "Combined", level_name)
        
        if unexpected:
            print(f"\n    UNEXPECTED entries: {len(unexpected)}")
            for uid, cnt in sorted(unexpected.items(), key=lambda x: -x[1])[:5]:
                print(f"      ID={uid}, count={cnt}")
            total_fails += 1
        else:
            print(f"\n    No unexpected entries -> PASS")
            total_passes += 1
        
        total_subcases = sum(dept_map.values())
        status = "PASS" if total_subcases == 30 else "FAIL"
        print(f"    Total target dept entries: {total_subcases} (expected 30) -> {status}")
        if status == "PASS": total_passes += 1
        else: total_fails += 1
    else:
        total_fails += len(expected_combined) + 2
    
    return total_passes, total_fails


# ========================================================================
# SECTION 5: MAIN
# ========================================================================

def main():
    print("="*70)
    print("  CLOSED-LOOP REPORTING VERIFICATION TEST - 3 LEVELS")
    print("  Test Bench: 10 Cases | 8 Sections | 30 Subcases")
    print("  Levels: Section | Department | Administration")
    print("="*70)
    
    # Phase 0: Verify database ground truth
    db_ok = verify_database_ground_truth()
    if not db_ok:
        print("\n  WARNING: Database ground truth failed.")
        print("  Continuing anyway for diagnostics...\n")
    
    # Login
    print("\n" + "="*70)
    print("PHASE 1: API LOGIN")
    print("="*70)
    cookies = login()
    
    total_passes = 0
    total_fails = 0
    
    # ======================================================================
    # LEVEL 1: Section (group_by=section)
    # ======================================================================
    print("\n" + "="*70)
    print("LEVEL 1: SECTION-LEVEL REPORTING (group_by=section)")
    print("="*70)
    
    p, f = run_level_test(
        cookies, "Section", "section", SECTIONS,
        SECTION_EXPECTED_M1, SECTION_EXPECTED_M12, SECTION_EXPECTED_COMBINED,
        "L1"
    )
    total_passes += p
    total_fails += f
    print(f"\n  LEVEL 1 SUBTOTAL: {p} passed, {f} failed")
    
    # ======================================================================
    # LEVEL 2: Department (group_by=department)
    # ======================================================================
    print("\n" + "="*70)
    print("LEVEL 2: DEPARTMENT-LEVEL REPORTING (group_by=department)")
    print("="*70)
    
    p, f = run_level_test(
        cookies, "Department", "department", DEPARTMENTS,
        DEPT_EXPECTED_M1, DEPT_EXPECTED_M12, DEPT_EXPECTED_COMBINED,
        "L2"
    )
    total_passes += p
    total_fails += f
    print(f"\n  LEVEL 2 SUBTOTAL: {p} passed, {f} failed")
    
    # ======================================================================
    # LEVEL 3: Administration (group_by=administration)
    # ======================================================================
    print("\n" + "="*70)
    print("LEVEL 3: ADMINISTRATION-LEVEL REPORTING (group_by=administration)")
    print("="*70)
    
    p, f = run_level_test(
        cookies, "Administration", "administration", ADMINISTRATIONS,
        ADMIN_EXPECTED_M1, ADMIN_EXPECTED_M12, ADMIN_EXPECTED_COMBINED,
        "L3"
    )
    total_passes += p
    total_fails += f
    print(f"\n  LEVEL 3 SUBTOTAL: {p} passed, {f} failed")
    
    # ======================================================================
    # FINAL REPORT
    # ======================================================================
    total_tests = total_passes + total_fails
    pass_rate = (total_passes / total_tests * 100) if total_tests > 0 else 0
    
    print("\n" + "="*70)
    print("FINAL REPORT: CLOSED-LOOP 3-LEVEL VERIFICATION")
    print("="*70)
    print(f"""
  Database Ground Truth:  {"VALID" if db_ok else "INVALID"}
  Tests Passed:           {total_passes}/{total_tests} ({pass_rate:.1f}%)
  Tests Failed:           {total_fails}/{total_tests}
    """)
    
    if db_ok and total_fails == 0:
        print("""
  +----------------------------------------------------------------+
  |  CONCLUSION: ALL 3 LEVELS PASSED                               |
  |                                                                |
  |  The monthly reporting endpoint correctly:                     |
  |    1. Section level:  Counts per individual section            |
  |    2. Department level: Rolls up sections to departments       |
  |    3. Administration level: Rolls up to administrations        |
  |    4. No leaking of unrelated org units at any level           |
  |    5. Date isolation works at all levels                       |
  |    6. Combined ranges aggregate correctly at all levels        |
  |                                                                |
  |  Mathematical guarantee: 10 cases, 8 sections, 30 subcases,   |
  |  3 hierarchy levels, deterministic triangular distribution.    |
  +----------------------------------------------------------------+
        """)
    else:
        failed_msg = []
        print("""
  +----------------------------------------------------------------+
  |  CONCLUSION: SOME TESTS FAILED                                 |
  |                                                                |
  |  Review FAIL entries above to identify the issue.              |
  |  The group_by parameter may not be working correctly at        |
  |  all hierarchy levels.                                         |
  +----------------------------------------------------------------+
        """)
    
    sys.exit(0 if (db_ok and total_fails == 0) else 1)


if __name__ == "__main__":
    main()
