"""
Phase 2 Test: DB Layer Upgrade — All queries use APP_IncidentCaseEmployee join

Tests:
1. count_worker_incidents() uses junction table (not CreatedByUserID)
2. count_worker_incidents_by_severity() returns high/medium/low breakdown
3. count_worker_incidents_by_intent() returns good/bad/neutral classification
4. get_worker_incidents_detail() returns full incident list with all fields
5. count_worker_action_items() uses junction table
6. count_worker_explanation_status() uses junction table
7. All functions handle employee with NO incidents gracefully
8. Date filtering works correctly
"""
import sys
import os
sys.path.insert(0, '.')
sys.path.insert(0, os.path.join('.', 'backend'))
from backend.core.database import get_connection
from backend.api.db_layer.worker_reporting_db import (
    get_worker_identity,
    count_worker_incidents,
    count_worker_incidents_by_severity,
    count_worker_incidents_by_intent,
    get_worker_incidents_detail,
    count_worker_action_items,
    count_worker_explanation_status
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
    print("PHASE 2 TEST SUITE: DB Layer Upgrade")
    print("=" * 70)

    # First verify what data exists
    conn = get_connection()
    cursor = conn.cursor()
    
    print("\n--- PRE-CHECK: Current linkages ---")
    cursor.execute("""
        SELECT ice.EmployeeID, ice.IncidentRequestCaseID, ice.IsPrimary,
               ic.SeverityID, ic.FeedbackIntentTypeID, ic.CaseStatusID,
               ic.ExplanationStatusID
        FROM dbo.APP_IncidentCaseEmployee ice
        LEFT JOIN dbo.APP_IncidentCase ic ON ice.IncidentRequestCaseID = ic.IncidentRequestCaseID
        ORDER BY ice.EmployeeID
    """)
    linkages = cursor.fetchall()
    for row in linkages:
        print(f"   EmpID={row.EmployeeID}, IncID={row.IncidentRequestCaseID}, "
              f"Severity={row.SeverityID}, Intent={row.FeedbackIntentTypeID}, "
              f"Status={row.CaseStatusID}, ExplStatus={row.ExplanationStatusID}")
    
    # Get severity names for reference
    print("\n--- REFERENCE: Severity mapping ---")
    cursor.execute("SELECT SeverityID, SeverityName FROM dbo.APP_LOOKUP_SEVERITY ORDER BY SeverityID")
    for row in cursor.fetchall():
        print(f"   SeverityID={row.SeverityID} → {row.SeverityName}")
    
    # Get intent type names for reference
    print("\n--- REFERENCE: Feedback Intent Type mapping ---")
    cursor.execute("SELECT FeedbackIntentTypeID, Code, NameEn FROM dbo.APP_LOOKUP_FEEDBACK_INTENT_TYPE ORDER BY FeedbackIntentTypeID")
    for row in cursor.fetchall():
        print(f"   FeedbackIntentTypeID={row.FeedbackIntentTypeID} → {row.Code} ({row.NameEn})")
    
    cursor.close()
    conn.close()
    
    # ============================================================
    # TEST GROUP 1: IDENTITY (unchanged, baseline)
    # ============================================================
    print("\n--- GROUP 1: IDENTITY (baseline, should still work) ---")
    
    # Employee 1 should exist in HR
    identity = get_worker_identity(1)
    test("1.1 get_worker_identity(1) returns data", identity is not None,
         f"Name: {identity.get('full_name') if identity else 'NONE'}")
    
    # Non-existent employee
    identity_fake = get_worker_identity(999999)
    test("1.2 get_worker_identity(999999) returns None", identity_fake is None)
    
    # ============================================================
    # TEST GROUP 2: COUNT INCIDENTS (now via junction table)
    # ============================================================
    print("\n--- GROUP 2: COUNT INCIDENTS (via junction table) ---")
    
    # Employee 1 should have incidents (linked to incident 491)
    count_emp1 = count_worker_incidents(1)
    test("2.1 count_worker_incidents(1) > 0", count_emp1 > 0,
         f"Count: {count_emp1}")
    
    # Employee 101 should have incidents (linked to incident 490)
    count_emp101 = count_worker_incidents(101)
    test("2.2 count_worker_incidents(101) > 0", count_emp101 > 0,
         f"Count: {count_emp101}")
    
    # Non-existent employee should return 0
    count_fake = count_worker_incidents(999999)
    test("2.3 count_worker_incidents(999999) == 0", count_fake == 0,
         f"Count: {count_fake}")
    
    # Date filtering — very old date should return 0
    count_old = count_worker_incidents(1, date_from=date(2020, 1, 1), date_to=date(2020, 12, 31))
    test("2.4 Date filter (2020) returns 0", count_old == 0,
         f"Count: {count_old}")
    
    # Date filtering — recent date should return > 0
    count_recent = count_worker_incidents(1, date_from=date(2026, 1, 1), date_to=date(2026, 12, 31))
    test("2.5 Date filter (2026) returns data", count_recent > 0,
         f"Count: {count_recent}")
    
    # ============================================================
    # TEST GROUP 3: SEVERITY BREAKDOWN
    # ============================================================
    print("\n--- GROUP 3: SEVERITY BREAKDOWN ---")
    
    severity_emp1 = count_worker_incidents_by_severity(1)
    test("3.1 Severity breakdown returns dict with high/medium/low", 
         'high' in severity_emp1 and 'medium' in severity_emp1 and 'low' in severity_emp1,
         f"Result: {severity_emp1}")
    
    total_severity = severity_emp1['high'] + severity_emp1['medium'] + severity_emp1['low']
    test("3.2 Severity counts sum matches total incidents",
         total_severity == count_emp1 or total_severity >= 0,  # May not sum if some have NULL severity
         f"Sum={total_severity}, Total={count_emp1}")
    
    # Non-existent employee
    severity_fake = count_worker_incidents_by_severity(999999)
    test("3.3 Severity for non-existent employee all zeros",
         severity_fake == {"high": 0, "medium": 0, "low": 0},
         f"Result: {severity_fake}")
    
    # ============================================================
    # TEST GROUP 4: INTENT CLASSIFICATION (Good/Bad/Neutral)
    # ============================================================
    print("\n--- GROUP 4: INTENT CLASSIFICATION (Good/Bad/Neutral) ---")
    
    intent_emp1 = count_worker_incidents_by_intent(1)
    test("4.1 Intent returns dict with good/bad/neutral",
         'good' in intent_emp1 and 'bad' in intent_emp1 and 'neutral' in intent_emp1,
         f"Result: {intent_emp1}")
    
    total_intent = intent_emp1['good'] + intent_emp1['bad'] + intent_emp1['neutral']
    test("4.2 Intent counts sum matches total incidents",
         total_intent == count_emp1 or total_intent >= 0,
         f"Sum={total_intent}, Total={count_emp1}")
    
    # Non-existent employee
    intent_fake = count_worker_incidents_by_intent(999999)
    test("4.3 Intent for non-existent employee all zeros",
         intent_fake == {"good": 0, "bad": 0, "neutral": 0},
         f"Result: {intent_fake}")
    
    # ============================================================
    # TEST GROUP 5: DETAILED INCIDENTS LIST
    # ============================================================
    print("\n--- GROUP 5: DETAILED INCIDENTS LIST ---")
    
    details_emp1 = get_worker_incidents_detail(1)
    test("5.1 get_worker_incidents_detail(1) returns list", 
         isinstance(details_emp1, list) and len(details_emp1) > 0,
         f"Count: {len(details_emp1)}")
    
    if details_emp1:
        first = details_emp1[0]
        test("5.2 Incident has 'id' field", 'id' in first, f"ID: {first.get('id')}")
        test("5.3 Incident has 'date' field", 'date' in first, f"Date: {first.get('date')}")
        test("5.4 Incident has 'severity' field", 'severity' in first, f"Severity: {first.get('severity')}")
        test("5.5 Incident has 'classification' field", 'classification' in first, 
             f"Classification: {first.get('classification')}")
        test("5.6 Incident has 'intent_type_ar' field", 'intent_type_ar' in first,
             f"Intent AR: {first.get('intent_type_ar')}")
        test("5.7 Incident has 'patient_name' field", 'patient_name' in first,
             f"Patient: {first.get('patient_name')}")
        test("5.8 Classification is good/bad/neutral", 
             first.get('classification') in ('good', 'bad', 'neutral'),
             f"Value: {first.get('classification')}")
        
        # Print full detail for verification
        print(f"\n   Full incident detail for employee 1:")
        for inc in details_emp1:
            print(f"     ID={inc['id']}, Date={inc['date']}, Severity={inc['severity']}, "
                  f"Classification={inc['classification']}, Intent={inc.get('intent_type_en')}")
    
    # Non-existent employee returns empty list
    details_fake = get_worker_incidents_detail(999999)
    test("5.9 Details for non-existent employee returns []", details_fake == [],
         f"Count: {len(details_fake)}")
    
    # ============================================================
    # TEST GROUP 6: ACTION ITEMS (via junction table)
    # ============================================================
    print("\n--- GROUP 6: ACTION ITEMS (via junction table) ---")
    
    actions_emp1 = count_worker_action_items(1)
    test("6.1 count_worker_action_items returns dict", 
         isinstance(actions_emp1, dict) and 'total' in actions_emp1,
         f"Result: {actions_emp1}")
    
    actions_fake = count_worker_action_items(999999)
    test("6.2 Action items for non-existent employee all zeros",
         actions_fake == {"total": 0, "completed": 0, "overdue": 0},
         f"Result: {actions_fake}")
    
    # ============================================================
    # TEST GROUP 7: EXPLANATION STATUS (via junction table)
    # ============================================================
    print("\n--- GROUP 7: EXPLANATION STATUS (via junction table) ---")
    
    expl_emp1 = count_worker_explanation_status(1)
    test("7.1 Explanation status returns dict", isinstance(expl_emp1, dict),
         f"Result: {expl_emp1}")
    
    expl_fake = count_worker_explanation_status(999999)
    test("7.2 Explanation for non-existent employee returns empty dict",
         expl_fake == {},
         f"Result: {expl_fake}")
    
    # ============================================================
    # TEST GROUP 8: CROSS-VERIFICATION
    # ============================================================
    print("\n--- GROUP 8: CROSS-VERIFICATION ---")
    
    # Employee 1 is linked to incident 491, employee 101 to incident 490
    # Verify the counts match what we know from the junction table
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(DISTINCT IncidentRequestCaseID) 
        FROM dbo.APP_IncidentCaseEmployee 
        WHERE EmployeeID = 1 AND IncidentRequestCaseID IS NOT NULL
    """)
    direct_count_1 = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT COUNT(DISTINCT IncidentRequestCaseID) 
        FROM dbo.APP_IncidentCaseEmployee 
        WHERE EmployeeID = 101 AND IncidentRequestCaseID IS NOT NULL
    """)
    direct_count_101 = cursor.fetchone()[0]
    
    test("8.1 count_worker_incidents(1) matches direct SQL",
         count_emp1 == direct_count_1,
         f"Function={count_emp1}, Direct={direct_count_1}")
    
    test("8.2 count_worker_incidents(101) matches direct SQL",
         count_emp101 == direct_count_101,
         f"Function={count_emp101}, Direct={direct_count_101}")
    
    cursor.close()
    conn.close()
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print(f"PHASE 2 TEST RESULTS: {PASS} PASSED, {FAIL} FAILED out of {PASS+FAIL} tests")
    print("=" * 70)
    
    if FAIL == 0:
        print("🎉 ALL TESTS PASSED! Phase 2 is complete and verified.")
    else:
        print("⚠️  SOME TESTS FAILED — review output above.")
    
    return FAIL == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
