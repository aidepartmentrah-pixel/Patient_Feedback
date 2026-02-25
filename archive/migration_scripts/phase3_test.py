"""
Phase 3 Test: Service Layer + Classification

Tests:
1. WorkerReportingService.get_worker_profile() returns correct response model
2. Response includes severity breakdown (high/medium/low) in metrics
3. Response includes intent classification (good/bad/neutral) in metrics
4. Response includes incidents detail list
5. All new schema fields are present and correctly typed
6. Seasonal reporting service includes new fields
7. Zero-incident employee returns graceful defaults (0 counts, empty list)
8. Date filtering propagates through service layer
"""
import sys
import os
sys.path.insert(0, '.')
sys.path.insert(0, os.path.join('.', 'backend'))

from backend.api.services.worker_reporting_service import WorkerReportingService
from backend.api.services.worker_seasonal_reporting_service import WorkerSeasonalReportingService
from backend.api.schemas.worker_reporting_schema import (
    WorkerProfileResponse,
    WorkerIdentityBlock,
    WorkerMetricBlock
)
from datetime import date

PASS = 0
FAIL = 0

def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        icon = "✅"
    else:
        FAIL += 1
        icon = "❌"
    print(f"  {icon} {name}" + (f" — {detail}" if detail else ""))


def run_all_tests():
    global PASS, FAIL

    print("=" * 70)
    print("PHASE 3 TEST SUITE: Service Layer + Classification")
    print("=" * 70)

    # =========================================================================
    # Find a valid employee that has incidents linked via junction table
    # =========================================================================
    from backend.core.database import get_connection
    conn = get_connection()
    cursor = conn.cursor()

    # Find employee linked to incidents
    cursor.execute("""
        SELECT TOP 1 ice.EmployeeID
        FROM dbo.APP_IncidentCaseEmployee ice
        ORDER BY ice.EmployeeID
    """)
    row = cursor.fetchone()
    if not row:
        print("❌ FATAL: No employees linked to incidents in APP_IncidentCaseEmployee")
        return

    linked_employee_id = row.EmployeeID
    print(f"\n--- Using linked employee: {linked_employee_id} ---")

    # Count how many incidents this employee has
    cursor.execute("""
        SELECT COUNT(*) as cnt FROM dbo.APP_IncidentCaseEmployee
        WHERE EmployeeID = ?
    """, (linked_employee_id,))
    expected_count = cursor.fetchone().cnt
    print(f"   Expected total incidents: {expected_count}")

    # Get severity breakdown for validation
    cursor.execute("""
        SELECT
            SUM(CASE WHEN ic.SeverityID = 3 THEN 1 ELSE 0 END) as high_cnt,
            SUM(CASE WHEN ic.SeverityID = 2 THEN 1 ELSE 0 END) as med_cnt,
            SUM(CASE WHEN ic.SeverityID = 1 THEN 1 ELSE 0 END) as low_cnt
        FROM dbo.APP_IncidentCaseEmployee ice
        INNER JOIN dbo.APP_IncidentCase ic ON ice.IncidentRequestCaseID = ic.IncidentRequestCaseID
        WHERE ice.EmployeeID = ?
    """, (linked_employee_id,))
    sev_row = cursor.fetchone()
    expected_high = sev_row.high_cnt or 0
    expected_med = sev_row.med_cnt or 0
    expected_low = sev_row.low_cnt or 0
    print(f"   Expected severity: H={expected_high}, M={expected_med}, L={expected_low}")

    # Get intent breakdown for validation
    cursor.execute("""
        SELECT
            SUM(CASE WHEN ic.FeedbackIntentTypeID = 2 THEN 1 ELSE 0 END) as good_cnt,
            SUM(CASE WHEN ic.FeedbackIntentTypeID = 3 THEN 1 ELSE 0 END) as bad_cnt,
            SUM(CASE WHEN ic.FeedbackIntentTypeID IN (1, 4) OR ic.FeedbackIntentTypeID IS NULL THEN 1 ELSE 0 END) as neutral_cnt
        FROM dbo.APP_IncidentCaseEmployee ice
        INNER JOIN dbo.APP_IncidentCase ic ON ice.IncidentRequestCaseID = ic.IncidentRequestCaseID
        WHERE ice.EmployeeID = ?
    """, (linked_employee_id,))
    intent_row = cursor.fetchone()
    expected_good = intent_row.good_cnt or 0
    expected_bad = intent_row.bad_cnt or 0
    expected_neutral = intent_row.neutral_cnt or 0
    print(f"   Expected intent: Good={expected_good}, Bad={expected_bad}, Neutral={expected_neutral}")

    # Find an employee with NO incidents for zero-test
    cursor.execute("""
        SELECT TOP 1 EmployeeID FROM dbo.APP_VIEWTABLE_HR_EMPLOYEES
        WHERE EmployeeID NOT IN (SELECT EmployeeID FROM dbo.APP_IncidentCaseEmployee)
        ORDER BY EmployeeID
    """)
    zero_row = cursor.fetchone()
    zero_employee_id = zero_row.EmployeeID if zero_row else None
    if zero_employee_id:
        print(f"   Zero-incident employee: {zero_employee_id}")
    else:
        print("   ⚠ No zero-incident employee found, will skip zero tests")

    conn.close()

    # =========================================================================
    # TEST GROUP 1: Worker Profile Service — Basic Response
    # =========================================================================
    print("\n--- TEST GROUP 1: WorkerReportingService.get_worker_profile() ---")

    try:
        profile = WorkerReportingService.get_worker_profile(employee_id=linked_employee_id)
        test("get_worker_profile returns WorkerProfileResponse",
             isinstance(profile, WorkerProfileResponse),
             f"type={type(profile).__name__}")
    except Exception as e:
        test("get_worker_profile returns WorkerProfileResponse", False, f"ERROR: {e}")
        print(f"\n❌ FATAL: Cannot proceed with remaining tests. Error: {e}")
        print(f"\n{'='*70}")
        print(f"RESULTS: {PASS} passed, {FAIL} failed, {PASS+FAIL} total")
        print(f"{'='*70}")
        return

    # =========================================================================
    # TEST GROUP 2: Response Structure — Identity Block
    # =========================================================================
    print("\n--- TEST GROUP 2: Identity Block ---")

    test("worker block is WorkerIdentityBlock",
         isinstance(profile.worker, WorkerIdentityBlock))

    test("employee_id matches request",
         profile.worker.employee_id == linked_employee_id,
         f"got={profile.worker.employee_id}")

    test("full_name is a non-empty string",
         isinstance(profile.worker.full_name, str) and len(profile.worker.full_name) > 0,
         f"name={profile.worker.full_name}")

    # =========================================================================
    # TEST GROUP 3: Response Structure — Metrics Block
    # =========================================================================
    print("\n--- TEST GROUP 3: Metrics Block ---")

    test("metrics block is WorkerMetricBlock",
         isinstance(profile.metrics, WorkerMetricBlock))

    test("total_incidents matches expected",
         profile.metrics.total_incidents == expected_count,
         f"got={profile.metrics.total_incidents}, expected={expected_count}")

    # =========================================================================
    # TEST GROUP 4: Severity Breakdown
    # =========================================================================
    print("\n--- TEST GROUP 4: Severity Breakdown ---")

    test("high_severity field exists and is int",
         isinstance(profile.metrics.high_severity, int))

    test("medium_severity field exists and is int",
         isinstance(profile.metrics.medium_severity, int))

    test("low_severity field exists and is int",
         isinstance(profile.metrics.low_severity, int))

    test("high_severity matches expected",
         profile.metrics.high_severity == expected_high,
         f"got={profile.metrics.high_severity}, expected={expected_high}")

    test("medium_severity matches expected",
         profile.metrics.medium_severity == expected_med,
         f"got={profile.metrics.medium_severity}, expected={expected_med}")

    test("low_severity matches expected",
         profile.metrics.low_severity == expected_low,
         f"got={profile.metrics.low_severity}, expected={expected_low}")

    test("severity sum equals total_incidents",
         (profile.metrics.high_severity + profile.metrics.medium_severity + profile.metrics.low_severity) == profile.metrics.total_incidents,
         f"sum={profile.metrics.high_severity + profile.metrics.medium_severity + profile.metrics.low_severity}, total={profile.metrics.total_incidents}")

    # =========================================================================
    # TEST GROUP 5: Intent Classification (Good/Bad/Neutral)
    # =========================================================================
    print("\n--- TEST GROUP 5: Intent Classification ---")

    test("good_feedback_count field exists and is int",
         isinstance(profile.metrics.good_feedback_count, int))

    test("bad_feedback_count field exists and is int",
         isinstance(profile.metrics.bad_feedback_count, int))

    test("neutral_feedback_count field exists and is int",
         isinstance(profile.metrics.neutral_feedback_count, int))

    test("good_feedback_count matches expected",
         profile.metrics.good_feedback_count == expected_good,
         f"got={profile.metrics.good_feedback_count}, expected={expected_good}")

    test("bad_feedback_count matches expected",
         profile.metrics.bad_feedback_count == expected_bad,
         f"got={profile.metrics.bad_feedback_count}, expected={expected_bad}")

    test("neutral_feedback_count matches expected",
         profile.metrics.neutral_feedback_count == expected_neutral,
         f"got={profile.metrics.neutral_feedback_count}, expected={expected_neutral}")

    test("intent sum equals total_incidents",
         (profile.metrics.good_feedback_count + profile.metrics.bad_feedback_count + profile.metrics.neutral_feedback_count) == profile.metrics.total_incidents,
         f"sum={profile.metrics.good_feedback_count + profile.metrics.bad_feedback_count + profile.metrics.neutral_feedback_count}, total={profile.metrics.total_incidents}")

    # =========================================================================
    # TEST GROUP 6: Incidents Detail List
    # =========================================================================
    print("\n--- TEST GROUP 6: Incidents Detail List ---")

    test("incidents field exists and is list",
         isinstance(profile.incidents, list))

    test("incidents count matches total_incidents",
         len(profile.incidents) == expected_count,
         f"got={len(profile.incidents)}, expected={expected_count}")

    if len(profile.incidents) > 0:
        first_inc = profile.incidents[0]
        test("incident has 'id' field",
             'id' in first_inc, f"keys={list(first_inc.keys())}")
        test("incident has 'severity' field",
             'severity' in first_inc)
        test("incident has 'classification' field",
             'classification' in first_inc)
        test("classification is one of good/bad/neutral",
             first_inc.get('classification') in ('good', 'bad', 'neutral'),
             f"got={first_inc.get('classification')}")
        test("incident has 'intent_type_ar' field",
             'intent_type_ar' in first_inc)
        test("incident has 'intent_type_en' field",
             'intent_type_en' in first_inc)
    else:
        # Skip detail field checks if no incidents
        for _ in range(6):
            test("(skipped — no incidents to check detail fields)", True)

    # =========================================================================
    # TEST GROUP 7: Zero-Incident Employee
    # =========================================================================
    print("\n--- TEST GROUP 7: Zero-Incident Employee ---")

    if zero_employee_id:
        try:
            zero_profile = WorkerReportingService.get_worker_profile(employee_id=zero_employee_id)
            test("zero-incident profile returns successfully",
                 isinstance(zero_profile, WorkerProfileResponse))
            test("zero total_incidents",
                 zero_profile.metrics.total_incidents == 0,
                 f"got={zero_profile.metrics.total_incidents}")
            test("zero high_severity",
                 zero_profile.metrics.high_severity == 0)
            test("zero good_feedback_count",
                 zero_profile.metrics.good_feedback_count == 0)
            test("zero bad_feedback_count",
                 zero_profile.metrics.bad_feedback_count == 0)
            test("zero incidents list is empty",
                 len(zero_profile.incidents) == 0,
                 f"got={len(zero_profile.incidents)}")
        except Exception as e:
            test("zero-incident profile returns successfully", False, f"ERROR: {e}")
            for _ in range(4):
                test("(skipped due to error)", False)
    else:
        for _ in range(5):
            test("(skipped — no zero-incident employee available)", True)

    # =========================================================================
    # TEST GROUP 8: Seasonal Reporting Service
    # =========================================================================
    print("\n--- TEST GROUP 8: Seasonal Reporting Service ---")

    try:
        seasonal_data = WorkerSeasonalReportingService.build_worker_seasonal_report_data(
            employee_id=linked_employee_id,
            season_start=date(2020, 1, 1),
            season_end=date(2030, 12, 31)
        )
        test("seasonal service returns dict",
             isinstance(seasonal_data, dict))

        # Check metrics contain new fields
        m = seasonal_data.get('metrics', {})
        test("seasonal metrics has high_severity",
             'high_severity' in m, f"keys={list(m.keys())}")
        test("seasonal metrics has medium_severity",
             'medium_severity' in m)
        test("seasonal metrics has low_severity",
             'low_severity' in m)
        test("seasonal metrics has good_feedback_count",
             'good_feedback_count' in m)
        test("seasonal metrics has bad_feedback_count",
             'bad_feedback_count' in m)
        test("seasonal metrics has neutral_feedback_count",
             'neutral_feedback_count' in m)

        # Check incidents_details is in the payload
        test("seasonal payload has incidents_details",
             'incidents_details' in seasonal_data,
             f"keys={list(seasonal_data.keys())}")

        test("seasonal incidents_details is list",
             isinstance(seasonal_data.get('incidents_details', None), list))

        # Check performance block still works
        perf = seasonal_data.get('performance', {})
        test("seasonal performance has score",
             'score' in perf)
        test("seasonal performance has praise_level",
             'praise_level' in perf)

    except Exception as e:
        test("seasonal service returns dict", False, f"ERROR: {e}")
        for _ in range(9):
            test("(skipped — seasonal service error)", False)

    # =========================================================================
    # TEST GROUP 9: Date Filtering Through Service Layer
    # =========================================================================
    print("\n--- TEST GROUP 9: Date Filtering ---")

    try:
        # Use a very old date range — should return 0 incidents
        old_profile = WorkerReportingService.get_worker_profile(
            employee_id=linked_employee_id,
            date_from=date(2000, 1, 1),
            date_to=date(2000, 12, 31)
        )
        test("date filter: old range returns 0 incidents",
             old_profile.metrics.total_incidents == 0,
             f"got={old_profile.metrics.total_incidents}")
        test("date filter: old range returns empty incidents list",
             len(old_profile.incidents) == 0,
             f"got={len(old_profile.incidents)}")
        test("date filter: period_from is set correctly",
             old_profile.period_from == date(2000, 1, 1))
        test("date filter: period_to is set correctly",
             old_profile.period_to == date(2000, 12, 31))
    except Exception as e:
        test("date filter: old range returns 0 incidents", False, f"ERROR: {e}")
        for _ in range(3):
            test("(skipped — date filter error)", False)

    # =========================================================================
    # FINAL RESULTS
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 3 RESULTS: {PASS} passed, {FAIL} failed, {PASS+FAIL} total")
    print(f"{'='*70}")

    if FAIL == 0:
        print("🎉 ALL TESTS PASSED! Phase 3 complete.")
    else:
        print(f"⚠ {FAIL} test(s) failed. Review output above.")


if __name__ == "__main__":
    run_all_tests()
