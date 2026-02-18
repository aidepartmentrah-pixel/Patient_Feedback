"""
=============================================================================
CLOSED-LOOP REPORTING VERIFICATION TEST
=============================================================================
Purpose: Verify the monthly/seasonal reporting endpoints return correct
         section-level subcase counts against a deterministic test bench.

Test Design:
  - 10 cases across 2 months (Jan 2026 & Dec 2025)
  - 8 target sections with a triangular distribution pattern
  - 30 total subcases (15 per month)
  - Ground truth is mathematically defined and compared to API output

Degrees of Freedom Tested:
  1. Sections (by_department counts)
  2. Monthly isolation (Month 1 vs Month 12)
  3. Seasonal aggregation (Q4-2025 + Q1-2026)
  4. Total case counts
  5. Subcase counts per section

=============================================================================
"""

import requests
import json
import sys
from collections import defaultdict

BASE_URL = "http://localhost:8000"

# ========================================================================
# SECTION 1: GROUND TRUTH DEFINITIONS
# ========================================================================

# The 8 test sections (OrgUnit IDs)
SECTIONS = {
    43:  "Cardiac 1",
    95:  "قسم بنك الدم",
    60:  "قسم التدريب والتقييم",
    72:  "قسم المباني",
    98:  "قسم ضبط العدوى",
    42:  "قسم الجراحة القلبية",
    309: "قسم الذكاء الاصطناعي",
    93:  "قسم الفوترة",
}

# Case IDs
MONTH1_CASE_IDS = [492, 493, 494, 495, 496]  # January 2026
MONTH12_CASE_IDS = [497, 498, 499, 500, 501]  # December 2025
ALL_CASE_IDS = MONTH1_CASE_IDS + MONTH12_CASE_IDS

# -----------------------------------------------------------------------
# Reference Result 5: Month 1 (January 2026) - by_department expected
# -----------------------------------------------------------------------
# Case 1 (492): targets [43]
# Case 2 (493): targets [43, 95]
# Case 3 (494): targets [43, 95, 60]
# Case 4 (495): targets [43, 95, 60, 72]
# Case 5 (496): targets [43, 95, 60, 72, 98]
EXPECTED_MONTH1 = {
    43:  5,  # Cardiac 1
    95:  4,  # بنك الدم
    60:  3,  # التدريب
    72:  2,  # المباني
    98:  1,  # ضبط العدوى
    42:  0,  # الجراحة القلبية
    309: 0,  # الذكاء الاصطناعي
    93:  0,  # الفوترة
}
EXPECTED_MONTH1_TOTAL_SUBCASES = 15
EXPECTED_MONTH1_TOTAL_CASES = 5

# -----------------------------------------------------------------------
# Reference Result 6: Month 12 (December 2025) - by_department expected
# -----------------------------------------------------------------------
# Case 6  (497): targets [93]
# Case 7  (498): targets [93, 309]
# Case 8  (499): targets [93, 309, 42]
# Case 9  (500): targets [93, 309, 42, 98]
# Case 10 (501): targets [93, 309, 42, 98, 72]
EXPECTED_MONTH12 = {
    43:  0,  # Cardiac 1
    95:  0,  # بنك الدم
    60:  0,  # التدريب
    72:  1,  # المباني
    98:  2,  # ضبط العدوى
    42:  3,  # الجراحة القلبية
    309: 4,  # الذكاء الاصطناعي
    93:  5,  # الفوترة
}
EXPECTED_MONTH12_TOTAL_SUBCASES = 15
EXPECTED_MONTH12_TOTAL_CASES = 5

# -----------------------------------------------------------------------
# Reference Result 7: Combined (Seasonal Q4-2025 + Q1-2026)
# -----------------------------------------------------------------------
EXPECTED_COMBINED = {
    43:  5,  # 5 + 0
    95:  4,  # 4 + 0
    60:  3,  # 3 + 0
    72:  3,  # 2 + 1
    98:  3,  # 1 + 2
    42:  3,  # 0 + 3
    309: 4,  # 0 + 4
    93:  5,  # 0 + 5
}
EXPECTED_COMBINED_TOTAL_SUBCASES = 30
EXPECTED_COMBINED_TOTAL_CASES = 10


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


def get_monthly_report(cookies, year, month):
    """Call POST /api/reports/monthly/view in numeric mode."""
    resp = requests.post(
        f"{BASE_URL}/api/reports/monthly/view",
        json={
            "year": year,
            "month": month,
            "mode": "numeric"
        },
        cookies=cookies
    )
    if resp.status_code != 200:
        print(f"  REPORT FAILED: {resp.status_code} {resp.text[:300]}")
        return None
    return resp.json()


def get_monthly_report_range(cookies, start_date, end_date):
    """Call POST /api/reports/monthly/view with date range in numeric mode."""
    resp = requests.post(
        f"{BASE_URL}/api/reports/monthly/view",
        json={
            "year": int(start_date[:4]),
            "month": None,
            "mode": "numeric",
            "start_date": start_date,
            "end_date": end_date,
        },
        cookies=cookies
    )
    if resp.status_code != 200:
        print(f"  RANGE REPORT FAILED: {resp.status_code} {resp.text[:300]}")
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


def compare_section_counts(actual_map, expected, label):
    """Compare actual vs expected section counts. Return (pass_count, fail_count, details)."""
    passes = 0
    fails = 0
    details = []
    
    for section_id, expected_count in expected.items():
        actual_count = actual_map.get(section_id, 0)
        section_name = SECTIONS.get(section_id, f"Unknown({section_id})")
        
        if actual_count == expected_count:
            passes += 1
            status = "PASS"
        else:
            fails += 1
            status = "FAIL"
        
        details.append({
            "section_id": section_id,
            "section_name": section_name,
            "expected": expected_count,
            "actual": actual_count,
            "status": status
        })
    
    # Check for unexpected sections (sections in actual but not in our 8)
    unexpected = {k: v for k, v in actual_map.items() if k not in expected}
    
    return passes, fails, details, unexpected


def print_section_table(details, label):
    """Print a formatted comparison table."""
    print(f"\n  {'Section':<30} {'Expected':>8} {'Actual':>8} {'Status':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
    for d in details:
        marker = "  ✓" if d["status"] == "PASS" else " ✗✗"
        print(f"  {d['section_name']:<30} {d['expected']:>8} {d['actual']:>8} {marker:>8}")


# ========================================================================
# SECTION 3: DATABASE GROUND TRUTH VERIFICATION
# ========================================================================

def verify_database_ground_truth():
    """Directly query the database to confirm our test data matches expectations."""
    print("\n" + "="*70)
    print("PHASE 0: DATABASE GROUND TRUTH VERIFICATION")
    print("="*70)
    
    try:
        sys.path.insert(0, ".")
        from core.database import get_connection
        conn = get_connection()
        cursor = conn.cursor()
        
        all_pass = True
        
        # Check 1: Case dates
        print("\n[Check 0.1] Case dates...")
        cursor.execute("""
            SELECT IncidentRequestCaseID, 
                YEAR(FeedbackRecievedDate) as yr, MONTH(FeedbackRecievedDate) as mo
            FROM dbo.APP_IncidentCase
            WHERE IncidentRequestCaseID IN (492,493,494,495,496,497,498,499,500,501)
            ORDER BY IncidentRequestCaseID
        """)
        for r in cursor.fetchall():
            case_id, yr, mo = r
            if case_id in MONTH1_CASE_IDS:
                if yr == 2026 and mo == 1:
                    print(f"  Case {case_id}: {yr}-{mo:02d} → PASS (Month 1)")
                else:
                    print(f"  Case {case_id}: {yr}-{mo:02d} → FAIL (expected 2026-01)")
                    all_pass = False
            elif case_id in MONTH12_CASE_IDS:
                if yr == 2025 and mo == 12:
                    print(f"  Case {case_id}: {yr}-{mo:02d} → PASS (Month 12)")
                else:
                    print(f"  Case {case_id}: {yr}-{mo:02d} → FAIL (expected 2025-12)")
                    all_pass = False
        
        # Check 2: Target departments match subcases
        print("\n[Check 0.2] Target departments per case...")
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
            492: {43},
            493: {43, 95},
            494: {43, 95, 60},
            495: {43, 95, 60, 72},
            496: {43, 95, 60, 72, 98},
            497: {93},
            498: {93, 309},
            499: {93, 309, 42},
            500: {93, 309, 42, 98},
            501: {93, 309, 42, 98, 72},
        }
        
        for case_id in sorted(expected_case_depts.keys()):
            actual = case_depts.get(case_id, set())
            expected = expected_case_depts[case_id]
            if actual == expected:
                print(f"  Case {case_id}: {sorted(actual)} → PASS")
            else:
                print(f"  Case {case_id}: {sorted(actual)} → FAIL (expected {sorted(expected)})")
                all_pass = False
        
        # Check 3: Subcase count per section per month
        print("\n[Check 0.3] Subcase counts per section...")
        cursor.execute("""
            SELECT s.TargetOrgUnitID, 
                MONTH(ic.FeedbackRecievedDate) as mo,
                YEAR(ic.FeedbackRecievedDate) as yr,
                COUNT(*) as cnt
            FROM dbo.APP_AdministrativeSubcase s
            JOIN dbo.APP_IncidentCase ic ON s.IncidentRequestCaseID = ic.IncidentRequestCaseID
            WHERE s.IncidentRequestCaseID IN (492,493,494,495,496,497,498,499,500,501)
            GROUP BY s.TargetOrgUnitID, MONTH(ic.FeedbackRecievedDate), YEAR(ic.FeedbackRecievedDate)
            ORDER BY yr, mo, s.TargetOrgUnitID
        """)
        subcase_counts = {}
        for r in cursor.fetchall():
            org_id, mo, yr, cnt = r
            key = (yr, mo, org_id)
            subcase_counts[key] = cnt
        
        # Verify Jan 2026
        for section_id, expected_count in EXPECTED_MONTH1.items():
            actual = subcase_counts.get((2026, 1, section_id), 0)
            status = "PASS" if actual == expected_count else "FAIL"
            if status == "FAIL":
                all_pass = False
            name = SECTIONS[section_id]
            print(f"  Jan 2026 | {name:<30} expected={expected_count}, actual={actual} → {status}")
        
        # Verify Dec 2025
        for section_id, expected_count in EXPECTED_MONTH12.items():
            actual = subcase_counts.get((2025, 12, section_id), 0)
            status = "PASS" if actual == expected_count else "FAIL"
            if status == "FAIL":
                all_pass = False
            name = SECTIONS[section_id]
            print(f"  Dec 2025 | {name:<30} expected={expected_count}, actual={actual} → {status}")
        
        # Check 4: Total subcases
        cursor.execute("""
            SELECT COUNT(*) FROM dbo.APP_AdministrativeSubcase
            WHERE IncidentRequestCaseID IN (492,493,494,495,496,497,498,499,500,501)
        """)
        total_subcases = cursor.fetchone()[0]
        print(f"\n[Check 0.4] Total subcases: {total_subcases} (expected 30) → {'PASS' if total_subcases == 30 else 'FAIL'}")
        if total_subcases != 30:
            all_pass = False
        
        conn.close()
        
        if all_pass:
            print("\n→ PHASE 0 RESULT: ALL DATABASE CHECKS PASSED ✓")
        else:
            print("\n→ PHASE 0 RESULT: SOME DATABASE CHECKS FAILED ✗")
            print("  Fix the test data before proceeding with API tests.")
        
        return all_pass
        
    except Exception as e:
        print(f"\n  DATABASE CHECK ERROR: {e}")
        return False


# ========================================================================
# SECTION 4: API TESTS
# ========================================================================

def run_api_tests():
    """Run the full closed-loop API verification."""
    
    print("\n" + "="*70)
    print("PHASE 1: API LOGIN")
    print("="*70)
    cookies = login()
    
    total_passes = 0
    total_fails = 0
    
    # ------------------------------------------------------------------
    # TEST 1: January 2026 Monthly Report
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("TEST 1: Monthly Report - January 2026 (Month 1)")
    print("="*70)
    
    report = get_monthly_report(cookies, year=2026, month=1)
    if report:
        # 1a. Total cases
        total = report.get("summary", {}).get("total_complaints", 0)
        status = "PASS" if total == EXPECTED_MONTH1_TOTAL_CASES else "FAIL"
        print(f"\n  [1a] Total cases: {total} (expected {EXPECTED_MONTH1_TOTAL_CASES}) → {status}")
        if status == "PASS": total_passes += 1
        else: total_fails += 1
        
        # 1b. Section counts
        dept_map = extract_dept_counts(report)
        passes, fails, details, unexpected = compare_section_counts(dept_map, EXPECTED_MONTH1, "Month 1")
        total_passes += passes
        total_fails += fails
        
        print(f"\n  [1b] Section-level counts (Reference Result 5):")
        print_section_table(details, "Month 1")
        
        if unexpected:
            print(f"\n  [1c] UNEXPECTED SECTIONS IN OUTPUT: {len(unexpected)}")
            for uid, cnt in sorted(unexpected.items(), key=lambda x: -x[1])[:10]:
                print(f"       Section ID={uid}, count={cnt}")
            total_fails += 1  # Unexpected sections = fail
        else:
            print(f"\n  [1c] No unexpected sections → PASS")
            total_passes += 1
        
        # 1d. Total subcases (sum of by_department counts)
        total_dept_count = sum(dept_map.values())
        status = "PASS" if total_dept_count == EXPECTED_MONTH1_TOTAL_SUBCASES else "FAIL"
        print(f"\n  [1d] Total target dept entries: {total_dept_count} (expected {EXPECTED_MONTH1_TOTAL_SUBCASES}) → {status}")
        if status == "PASS": total_passes += 1
        else: total_fails += 1
        
        # 1e. Number of sections in output
        num_sections = len(dept_map)
        expected_non_zero = sum(1 for v in EXPECTED_MONTH1.values() if v > 0)
        status = "PASS" if num_sections == expected_non_zero else "FAIL"
        print(f"  [1e] Sections in output: {num_sections} (expected {expected_non_zero}) → {status}")
        if status == "PASS": total_passes += 1
        else: total_fails += 1
    
    else:
        print("  SKIPPED - Report request failed")
        total_fails += 12
    
    # ------------------------------------------------------------------
    # TEST 2: December 2025 Monthly Report
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("TEST 2: Monthly Report - December 2025 (Month 12)")
    print("="*70)
    
    report = get_monthly_report(cookies, year=2025, month=12)
    if report:
        # 2a. Total cases
        total = report.get("summary", {}).get("total_complaints", 0)
        status = "PASS" if total == EXPECTED_MONTH12_TOTAL_CASES else "FAIL"
        print(f"\n  [2a] Total cases: {total} (expected {EXPECTED_MONTH12_TOTAL_CASES}) → {status}")
        if status == "PASS": total_passes += 1
        else: total_fails += 1
        
        # 2b. Section counts
        dept_map = extract_dept_counts(report)
        passes, fails, details, unexpected = compare_section_counts(dept_map, EXPECTED_MONTH12, "Month 12")
        total_passes += passes
        total_fails += fails
        
        print(f"\n  [2b] Section-level counts (Reference Result 6):")
        print_section_table(details, "Month 12")
        
        if unexpected:
            print(f"\n  [2c] UNEXPECTED SECTIONS IN OUTPUT: {len(unexpected)}")
            for uid, cnt in sorted(unexpected.items(), key=lambda x: -x[1])[:10]:
                print(f"       Section ID={uid}, count={cnt}")
            total_fails += 1
        else:
            print(f"\n  [2c] No unexpected sections → PASS")
            total_passes += 1
        
        # 2d. Total subcases
        total_dept_count = sum(dept_map.values())
        status = "PASS" if total_dept_count == EXPECTED_MONTH12_TOTAL_SUBCASES else "FAIL"
        print(f"\n  [2d] Total target dept entries: {total_dept_count} (expected {EXPECTED_MONTH12_TOTAL_SUBCASES}) → {status}")
        if status == "PASS": total_passes += 1
        else: total_fails += 1
        
        # 2e. Number of sections in output
        num_sections = len(dept_map)
        expected_non_zero = sum(1 for v in EXPECTED_MONTH12.values() if v > 0)
        status = "PASS" if num_sections == expected_non_zero else "FAIL"
        print(f"  [2e] Sections in output: {num_sections} (expected {expected_non_zero}) → {status}")
        if status == "PASS": total_passes += 1
        else: total_fails += 1
    
    else:
        print("  SKIPPED - Report request failed")
        total_fails += 12
    
    # ------------------------------------------------------------------
    # TEST 3: Combined Range (Dec 2025 + Jan 2026) - Seasonal
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("TEST 3: Combined Range Report (Dec 2025 - Jan 2026)")
    print("           This simulates seasonal aggregation (Reference Result 7)")
    print("="*70)
    
    report = get_monthly_report_range(cookies, "2025-12-01", "2026-01-31")
    if report:
        # 3a. Total cases
        total = report.get("summary", {}).get("total_complaints", 0)
        status = "PASS" if total == EXPECTED_COMBINED_TOTAL_CASES else "FAIL"
        print(f"\n  [3a] Total cases: {total} (expected {EXPECTED_COMBINED_TOTAL_CASES}) → {status}")
        if status == "PASS": total_passes += 1
        else: total_fails += 1
        
        # 3b. Section counts
        dept_map = extract_dept_counts(report)
        passes, fails, details, unexpected = compare_section_counts(dept_map, EXPECTED_COMBINED, "Combined")
        total_passes += passes
        total_fails += fails
        
        print(f"\n  [3b] Section-level counts (Reference Result 7):")
        print_section_table(details, "Combined")
        
        if unexpected:
            print(f"\n  [3c] UNEXPECTED SECTIONS IN OUTPUT: {len(unexpected)}")
            for uid, cnt in sorted(unexpected.items(), key=lambda x: -x[1])[:10]:
                print(f"       Section ID={uid}, count={cnt}")
            total_fails += 1
        else:
            print(f"\n  [3c] No unexpected sections → PASS")
            total_passes += 1
        
        # 3d. Total subcases
        total_dept_count = sum(dept_map.values())
        status = "PASS" if total_dept_count == EXPECTED_COMBINED_TOTAL_SUBCASES else "FAIL"
        print(f"\n  [3d] Total target dept entries: {total_dept_count} (expected {EXPECTED_COMBINED_TOTAL_SUBCASES}) → {status}")
        if status == "PASS": total_passes += 1
        else: total_fails += 1
        
        # 3e. Number of sections (should be 8 - all have data)
        num_sections = len(dept_map)
        expected_non_zero = sum(1 for v in EXPECTED_COMBINED.values() if v > 0)
        status = "PASS" if num_sections == expected_non_zero else "FAIL"
        print(f"  [3e] Sections in output: {num_sections} (expected {expected_non_zero}) → {status}")
        if status == "PASS": total_passes += 1
        else: total_fails += 1
        
        # 3f. Symmetry check - the combined result should be symmetric
        # Sections with count 5: Cardiac 1, الفوترة
        # Sections with count 4: بنك الدم, الذكاء الاصطناعي
        # Sections with count 3: التدريب, المباني, ضبط العدوى, الجراحة القلبية
        count_5 = [sid for sid, cnt in dept_map.items() if cnt == 5 and sid in SECTIONS]
        count_4 = [sid for sid, cnt in dept_map.items() if cnt == 4 and sid in SECTIONS]
        count_3 = [sid for sid, cnt in dept_map.items() if cnt == 3 and sid in SECTIONS]
        symmetry_ok = len(count_5) == 2 and len(count_4) == 2 and len(count_3) == 4
        status = "PASS" if symmetry_ok else "FAIL"
        print(f"\n  [3f] Symmetry check (2×5, 2×4, 4×3): → {status}")
        print(f"       Count=5: {[SECTIONS[s] for s in count_5]}")
        print(f"       Count=4: {[SECTIONS[s] for s in count_4]}")
        print(f"       Count=3: {[SECTIONS[s] for s in count_3]}")
        if status == "PASS": total_passes += 1
        else: total_fails += 1
    
    else:
        print("  SKIPPED - Report request failed")
        total_fails += 12
    
    return total_passes, total_fails


# ========================================================================
# SECTION 5: MATHEMATICAL CONCLUSION
# ========================================================================

def print_conclusion(db_ok, api_passes, api_fails):
    """Print the final mathematical conclusion."""
    print("\n" + "="*70)
    print("FINAL REPORT: CLOSED-LOOP VERIFICATION RESULTS")
    print("="*70)
    
    total_tests = api_passes + api_fails
    pass_rate = (api_passes / total_tests * 100) if total_tests > 0 else 0
    
    print(f"""
  Database Ground Truth:  {"VALID" if db_ok else "INVALID"}
  API Tests Passed:       {api_passes}/{total_tests} ({pass_rate:.1f}%)
  API Tests Failed:       {api_fails}/{total_tests}
    """)
    
    if db_ok and api_fails == 0:
        print("""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  CONCLUSION: ALL TESTS PASSED                                  ║
  ║                                                                ║
  ║  The monthly reporting endpoint correctly:                     ║
  ║    1. Counts cases per section for single-month queries        ║
  ║    2. Returns correct section breakdown (by_department)        ║
  ║    3. Aggregates correctly across date ranges                  ║
  ║    4. Does NOT leak unrelated sections into results            ║
  ║    5. Maintains triangular distribution symmetry               ║
  ║                                                                ║
  ║  Mathematical guarantee: With 10 cases, 8 sections, 30        ║
  ║  subcases, and a deterministic triangular pattern, the         ║
  ║  probability of false-positive is effectively 0.               ║
  ╚══════════════════════════════════════════════════════════════════╝
        """)
    else:
        print("""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  CONCLUSION: TESTS FAILED - REPORTING BUG DETECTED             ║
  ║                                                                ║
  ║  The reporting endpoint does NOT correctly match the ground    ║
  ║  truth test bench. Review the FAIL entries above to identify   ║
  ║  which sections have incorrect counts.                         ║
  ║                                                                ║
  ║  Common issues:                                                ║
  ║    - Leaking unrelated sections (allowed_unit_ids too broad)   ║
  ║    - Counting wrong table (subcases vs target departments)     ║
  ║    - Date filter not isolating the month correctly             ║
  ║    - IssuingOrgUnitID filter pulling in wrong cases            ║
  ╚══════════════════════════════════════════════════════════════════╝
        """)


# ========================================================================
# MAIN
# ========================================================================

if __name__ == "__main__":
    print("="*70)
    print("  CLOSED-LOOP REPORTING VERIFICATION TEST")
    print("  Test Bench: 10 Cases | 8 Sections | 30 Subcases")
    print("  Months: January 2026 (Q1-2026) & December 2025 (Q4-2025)")
    print("="*70)
    
    # Phase 0: Verify database ground truth
    db_ok = verify_database_ground_truth()
    
    if not db_ok:
        print("\n⚠ Database ground truth failed. Fix data before API testing.")
        print("  Continuing with API tests anyway for diagnostic purposes...\n")
    
    # Phase 1-3: API tests
    api_passes, api_fails = run_api_tests()
    
    # Final conclusion
    print_conclusion(db_ok, api_passes, api_fails)
    
    # Exit code
    sys.exit(0 if (db_ok and api_fails == 0) else 1)
