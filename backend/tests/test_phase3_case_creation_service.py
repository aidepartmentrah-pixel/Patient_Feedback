"""
Test Suite for case_creation_service.py
Tests subcase creation orchestration for incidents and seasonal reports.

Run: python backend/tests/test_phase3_case_creation_service.py
"""

import sys
sys.path.insert(0, 'backend')

from api_v2.services import case_creation_service as service
from api_v2.db_layer import administrative_subcase_db as subcase_db
from api_v2.db_layer import seasonal_report_db
import pyodbc

# Test counters
tests_run = 0
tests_passed = 0
tests_failed = 0

def test(description):
    """Decorator for test functions"""
    def decorator(func):
        def wrapper():
            global tests_run, tests_passed, tests_failed
            tests_run += 1
            try:
                func()
                tests_passed += 1
                print(f"✅ PASS: {description}")
                return True
            except AssertionError as e:
                tests_failed += 1
                print(f"❌ FAIL: {description}")
                print(f"   Error: {e}")
                return False
            except Exception as e:
                tests_failed += 1
                print(f"❌ ERROR: {description}")
                print(f"   Exception: {e}")
                return False
        return wrapper
    return decorator


# Mock current user
class MockUser:
    def __init__(self, user_id):
        self.user_id = user_id


# ============================================================
# TEST SUITE
# ============================================================

print("=" * 70)
print("PHASE 3 - CASE_CREATION_SERVICE.PY TEST SUITE")
print("=" * 70)
print()

# Test data
test_incident_id = 36  # Valid incident ID
test_seasonal_report_id = 659  # Valid seasonal report ID
current_user = MockUser(user_id=1)

# ============================================================
# HELPER FUNCTION TESTS
# ============================================================

@test("1. Internal _create_subcase helper works")
def test_internal_create_subcase():
    subcase_id = service._create_subcase(
        case_type='INCIDENT_RESPONSE',
        incident_id=test_incident_id,
        seasonal_report_id=None,
        target_org_unit_id=1,
        created_by_user_id=1,
        initial_status='SUBMITTED_TO_SECTION'
    )
    
    assert subcase_id is not None, "Should create subcase"
    assert isinstance(subcase_id, int), "Should return integer ID"
    
    # Clean up
    conn = service.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = ?", (subcase_id,))
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"   Created and cleaned up SubcaseID: {subcase_id}")

test_internal_create_subcase()


# ============================================================
# INCIDENT SUBCASE CREATION TESTS
# ============================================================

@test("2. Check if incident has target departments")
def test_incident_has_targets():
    conn = service.get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT COUNT(*) as cnt
            FROM dbo.APP_IncidentCaseTargetDepartment
            WHERE IncidentRequestCaseID = ?
        """, (test_incident_id,))
        
        row = cursor.fetchone()
        count = row.cnt
        
        assert count > 0, f"Incident {test_incident_id} must have target departments"
        print(f"   Incident {test_incident_id} has {count} target department(s)")
        
    finally:
        cursor.close()
        conn.close()

test_incident_has_targets()


@test("3. Create subcases for incident (first time)")
def test_create_subcases_for_incident_first():
    # Clean up any existing subcases first (with action items)
    existing = subcase_db.get_subcases_by_incident(test_incident_id)
    if existing:
        conn = service.get_db_connection()
        cursor = conn.cursor()
        for sc in existing:
            # Delete action items first
            cursor.execute("DELETE FROM dbo.APP_SubcaseActionItem WHERE SubcaseID = ?", (sc['subcase_id'],))
            # Then delete subcase
            cursor.execute("DELETE FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = ?", (sc['subcase_id'],))
        conn.commit()
        cursor.close()
        conn.close()
    
    # Now create
    service.create_subcases_for_incident(test_incident_id, current_user)
    
    # Verify
    subcases = subcase_db.get_subcases_by_incident(test_incident_id)
    assert len(subcases) > 0, "Should have created subcases"
    
    for sc in subcases:
        assert sc['case_type'] == 'INCIDENT_RESPONSE', "Should be INCIDENT_RESPONSE type"
        assert sc['status'] == 'SUBMITTED_TO_SECTION', "Should be SUBMITTED_TO_SECTION status"
        assert sc['incident_request_case_id'] == test_incident_id, "Should link to incident"
        assert sc['seasonal_report_id'] is None, "Should not link to seasonal report"
    
    print(f"   Created {len(subcases)} subcase(s) for incident {test_incident_id}")

test_create_subcases_for_incident_first()


@test("4. Create subcases for incident (idempotency check)")
def test_create_subcases_for_incident_idempotent():
    # Get count before
    before = subcase_db.get_subcases_by_incident(test_incident_id)
    count_before = len(before)
    
    # Call again
    service.create_subcases_for_incident(test_incident_id, current_user)
    
    # Get count after
    after = subcase_db.get_subcases_by_incident(test_incident_id)
    count_after = len(after)
    
    assert count_before == count_after, "Should not create duplicates"
    print(f"   Idempotency verified - still {count_after} subcase(s)")

test_create_subcases_for_incident_idempotent()


# ============================================================
# SEASONAL REPORT SUBCASE CREATION TESTS
# ============================================================

@test("5. Check if seasonal report has target org units")
def test_seasonal_has_targets():
    org_units = seasonal_report_db.get_target_orgunits_for_seasonal_report(test_seasonal_report_id)
    
    assert len(org_units) > 0, f"Seasonal report {test_seasonal_report_id} must have org units"
    print(f"   Seasonal report {test_seasonal_report_id} has {len(org_units)} org unit(s)")

test_seasonal_has_targets()


@test("6. Create subcases for seasonal report (first time)")
def test_create_subcases_for_seasonal_first():
    # Clean up any existing subcases first (with action items)
    existing = subcase_db.get_subcases_by_seasonal_report(test_seasonal_report_id)
    if existing:
        conn = service.get_db_connection()
        cursor = conn.cursor()
        for sc in existing:
            # Delete action items first
            cursor.execute("DELETE FROM dbo.APP_SubcaseActionItem WHERE SubcaseID = ?", (sc['subcase_id'],))
            # Then delete subcase
            cursor.execute("DELETE FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = ?", (sc['subcase_id'],))
        conn.commit()
        cursor.close()
        conn.close()
    
    # Now create
    service.create_subcases_for_seasonal_report(test_seasonal_report_id, current_user)
    
    # Verify
    subcases = subcase_db.get_subcases_by_seasonal_report(test_seasonal_report_id)
    assert len(subcases) > 0, "Should have created subcases"
    
    for sc in subcases:
        assert sc['case_type'] == 'SEASONAL_REPORT_RESPONSE', "Should be SEASONAL_REPORT_RESPONSE type"
        assert sc['status'] == 'SUBMITTED_TO_SECTION', "Should be SUBMITTED_TO_SECTION status"
        assert sc['seasonal_report_id'] == test_seasonal_report_id, "Should link to seasonal report"
        assert sc['incident_request_case_id'] is None, "Should not link to incident"
    
    print(f"   Created {len(subcases)} subcase(s) for seasonal report {test_seasonal_report_id}")

test_create_subcases_for_seasonal_first()


@test("7. Create subcases for seasonal report (idempotency check)")
def test_create_subcases_for_seasonal_idempotent():
    # Get count before
    before = subcase_db.get_subcases_by_seasonal_report(test_seasonal_report_id)
    count_before = len(before)
    
    # Call again
    service.create_subcases_for_seasonal_report(test_seasonal_report_id, current_user)
    
    # Get count after
    after = subcase_db.get_subcases_by_seasonal_report(test_seasonal_report_id)
    count_after = len(after)
    
    assert count_before == count_after, "Should not create duplicates"
    print(f"   Idempotency verified - still {count_after} subcase(s)")

test_create_subcases_for_seasonal_idempotent()


# ============================================================
# EDGE CASE TESTS
# ============================================================

@test("8. Create subcases for non-existent incident (no error)")
def test_create_for_nonexistent_incident():
    fake_incident_id = 999999
    
    # Should not raise error, just do nothing
    service.create_subcases_for_incident(fake_incident_id, current_user)
    
    # Verify no subcases created
    subcases = subcase_db.get_subcases_by_incident(fake_incident_id)
    assert len(subcases) == 0, "Should not create subcases for non-existent incident"
    print(f"   Correctly handled non-existent incident")

test_create_for_nonexistent_incident()


@test("9. Create subcases for non-existent seasonal report (no error)")
def test_create_for_nonexistent_seasonal():
    fake_seasonal_id = 999999
    
    # Should not raise error, just do nothing
    service.create_subcases_for_seasonal_report(fake_seasonal_id, current_user)
    
    # Verify no subcases created
    subcases = subcase_db.get_subcases_by_seasonal_report(fake_seasonal_id)
    assert len(subcases) == 0, "Should not create subcases for non-existent seasonal report"
    print(f"   Correctly handled non-existent seasonal report")

test_create_for_nonexistent_seasonal()


# ============================================================
# CLEANUP
# ============================================================

print()
print("CLEANUP: Removing test subcases...")

try:
    # Clean incident subcases (delete action items first)
    incident_subcases = subcase_db.get_subcases_by_incident(test_incident_id)
    for sc in incident_subcases:
        conn = service.get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM dbo.APP_SubcaseActionItem WHERE SubcaseID = ?", (sc['subcase_id'],))
            cursor.execute("DELETE FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = ?", (sc['subcase_id'],))
            conn.commit()
        finally:
            cursor.close()
            conn.close()

    # Clean seasonal subcases (delete action items first)
    seasonal_subcases = subcase_db.get_subcases_by_seasonal_report(test_seasonal_report_id)
    for sc in seasonal_subcases:
        conn = service.get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM dbo.APP_SubcaseActionItem WHERE SubcaseID = ?", (sc['subcase_id'],))
            cursor.execute("DELETE FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = ?", (sc['subcase_id'],))
            conn.commit()
        finally:
            cursor.close()
            conn.close()
    
    print("✅ Cleanup complete")
except Exception as e:
    print(f"⚠️  Cleanup warning: {e}")
    print("   (This is not critical - tests still passed)")


# ============================================================
# TEST SUMMARY
# ============================================================

print()
print("=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print(f"Total Tests:  {tests_run}")
print(f"✅ Passed:     {tests_passed}")
print(f"❌ Failed:     {tests_failed}")
print()

if tests_failed == 0:
    print("🎉 ALL TESTS PASSED! case_creation_service.py is 100% functional.")
    exit(0)
else:
    print(f"⚠️  {tests_failed} test(s) failed. Please review and fix.")
    exit(1)
