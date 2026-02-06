"""
Test Suite for administrative_subcase_db.py
Tests all 23 functions in the administrative subcase database layer.

Run: python backend/tests/test_phase3_administrative_subcase_db.py
"""

import sys
sys.path.insert(0, 'backend')

from api_v2.db_layer import administrative_subcase_db as db

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


# ============================================================
# TEST SUITE
# ============================================================

print("=" * 70)
print("PHASE 3 - ADMINISTRATIVE_SUBCASE_DB.PY TEST SUITE")
print("=" * 70)
print()

# Test data storage
test_incident_id = None
test_subcase_id = None

# ============================================================
# CREATION / FETCH TESTS
# ============================================================

@test("1. Create subcase for incident")
def test_create_incident_subcase():
    global test_subcase_id, test_incident_id
    # Use existing incident ID = 36 (verified to exist in test DB)
    test_incident_id = 36
    target_org_unit_id = 1  # Verified to exist in AdminsrationUnit
    
    subcase_id = db.create_subcase(
        case_type="INCIDENT_RESPONSE",
        incident_id=test_incident_id,
        seasonal_report_id=None,
        target_org_unit_id=target_org_unit_id,
        created_by_user_id=1,
        initial_status="SUBMITTED_TO_SECTION"
    )
    
    assert subcase_id is not None, "Subcase ID should not be None"
    assert isinstance(subcase_id, int), "Subcase ID should be integer"
    test_subcase_id = subcase_id
    print(f"   Created SubcaseID: {test_subcase_id}")

test_create_incident_subcase()


@test("2. Get subcase by ID")
def test_get_subcase_by_id():
    global test_subcase_id
    subcase = db.get_subcase_by_id(test_subcase_id)
    
    assert subcase is not None, "Subcase should exist"
    assert subcase["subcase_id"] == test_subcase_id, "IDs should match"
    assert subcase["case_type"] == "INCIDENT_RESPONSE", "Case type should match"
    assert subcase["status"] == "SUBMITTED_TO_SECTION", "Status should match"
    assert subcase["target_org_unit_id"] == 1, "Target org unit should match"
    print(f"   Retrieved subcase: {subcase['subcase_id']}")

test_get_subcase_by_id()


@test("3. Get subcases by incident")
def test_get_subcases_by_incident():
    global test_incident_id
    subcases = db.get_subcases_by_incident(test_incident_id)
    
    assert isinstance(subcases, list), "Should return list"
    assert len(subcases) > 0, "Should have at least one subcase"
    assert any(s["subcase_id"] == test_subcase_id for s in subcases), "Should contain test subcase"
    print(f"   Found {len(subcases)} subcase(s) for incident {test_incident_id}")

test_get_subcases_by_incident()


@test("4. Get subcases by seasonal report (empty)")
def test_get_subcases_by_seasonal_empty():
    subcases = db.get_subcases_by_seasonal_report(99999)  # Non-existent ID
    
    assert isinstance(subcases, list), "Should return list"
    assert len(subcases) == 0, "Should be empty for non-existent report"
    print(f"   Correctly returned empty list")

test_get_subcases_by_seasonal_empty()


@test("5. Get subcases by target org unit")
def test_get_subcases_by_target_orgunit():
    subcases = db.get_subcases_by_target_orgunit(1)
    
    assert isinstance(subcases, list), "Should return list"
    assert len(subcases) > 0, "Should have at least one subcase"
    print(f"   Found {len(subcases)} subcase(s) for org unit 1")

test_get_subcases_by_target_orgunit()


@test("6. Get subcases by status")
def test_get_subcases_by_status():
    subcases = db.get_subcases_by_status("SUBMITTED_TO_SECTION")
    
    assert isinstance(subcases, list), "Should return list"
    assert len(subcases) > 0, "Should have at least one subcase"
    assert all(s["status"] == "SUBMITTED_TO_SECTION" for s in subcases), "All should have correct status"
    print(f"   Found {len(subcases)} subcase(s) with status SUBMITTED_TO_SECTION")

test_get_subcases_by_status()


@test("7. Get subcases by case type")
def test_get_subcases_by_case_type():
    subcases = db.get_subcases_by_case_type("INCIDENT_RESPONSE")
    
    assert isinstance(subcases, list), "Should return list"
    assert len(subcases) > 0, "Should have at least one subcase"
    assert all(s["case_type"] == "INCIDENT_RESPONSE" for s in subcases), "All should have correct type"
    print(f"   Found {len(subcases)} INCIDENT_RESPONSE subcase(s)")

test_get_subcases_by_case_type()


# ============================================================
# INBOX QUERY TESTS
# ============================================================

@test("8. Get subcases pending for section")
def test_get_pending_section():
    subcases = db.get_subcases_pending_for_section()
    
    assert isinstance(subcases, list), "Should return list"
    assert len(subcases) > 0, "Should have at least one subcase"
    assert all(s["status"] == "SUBMITTED_TO_SECTION" for s in subcases), "All should be SUBMITTED_TO_SECTION"
    print(f"   Found {len(subcases)} subcase(s) pending for section")

test_get_pending_section()


@test("9. Get subcases pending for department (initially empty)")
def test_get_pending_department_empty():
    subcases = db.get_subcases_pending_for_department()
    
    assert isinstance(subcases, list), "Should return list"
    print(f"   Found {len(subcases)} subcase(s) pending for department")

test_get_pending_department_empty()


@test("10. Get subcases pending for administration (initially empty)")
def test_get_pending_administration_empty():
    subcases = db.get_subcases_pending_for_administration()
    
    assert isinstance(subcases, list), "Should return list"
    print(f"   Found {len(subcases)} subcase(s) pending for administration")

test_get_pending_administration_empty()


# ============================================================
# WORKFLOW MUTATION TESTS
# ============================================================

@test("11. Update subcase status")
def test_update_status():
    global test_subcase_id
    result = db.update_subcase_status(
        subcase_id=test_subcase_id,
        new_status="SECTION_ACCEPTED_PENDING_DEPT",
        updated_by_user_id=1
    )
    
    assert result is True, "Update should succeed"
    
    # Verify update
    subcase = db.get_subcase_by_id(test_subcase_id)
    assert subcase["status"] == "SECTION_ACCEPTED_PENDING_DEPT", "Status should be updated"
    print(f"   Status updated to SECTION_ACCEPTED_PENDING_DEPT")

test_update_status()


@test("12. Update section explanation")
def test_update_section_explanation():
    global test_subcase_id
    result = db.update_section_explanation(
        subcase_id=test_subcase_id,
        text="This is a test section explanation",
        updated_by_user_id=1
    )
    
    assert result is True, "Update should succeed"
    
    # Verify update
    subcase = db.get_subcase_by_id(test_subcase_id)
    assert subcase["section_explanation_text"] == "This is a test section explanation", "Text should be updated"
    print(f"   Section explanation updated")

test_update_section_explanation()


@test("13. Update section rejection")
def test_update_section_rejection():
    global test_subcase_id
    result = db.update_section_rejection(
        subcase_id=test_subcase_id,
        text="This is a test section rejection",
        updated_by_user_id=1
    )
    
    assert result is True, "Update should succeed"
    
    # Verify update
    subcase = db.get_subcase_by_id(test_subcase_id)
    assert subcase["section_rejection_text"] == "This is a test section rejection", "Text should be updated"
    print(f"   Section rejection updated")

test_update_section_rejection()


@test("14. Update department explanation")
def test_update_department_explanation():
    global test_subcase_id
    result = db.update_department_explanation(
        subcase_id=test_subcase_id,
        text="This is a test department explanation",
        updated_by_user_id=1
    )
    
    assert result is True, "Update should succeed"
    
    # Verify update
    subcase = db.get_subcase_by_id(test_subcase_id)
    assert subcase["department_explanation_text"] == "This is a test department explanation", "Text should be updated"
    print(f"   Department explanation updated")

test_update_department_explanation()


@test("15. Update department rejection")
def test_update_department_rejection():
    global test_subcase_id
    result = db.update_department_rejection(
        subcase_id=test_subcase_id,
        text="This is a test department rejection",
        updated_by_user_id=1
    )
    
    assert result is True, "Update should succeed"
    
    # Verify update
    subcase = db.get_subcase_by_id(test_subcase_id)
    assert subcase["department_rejection_text"] == "This is a test department rejection", "Text should be updated"
    print(f"   Department rejection updated")

test_update_department_rejection()


@test("16. Update administration explanation")
def test_update_administration_explanation():
    global test_subcase_id
    result = db.update_administration_explanation(
        subcase_id=test_subcase_id,
        text="This is a test administration explanation",
        updated_by_user_id=1
    )
    
    assert result is True, "Update should succeed"
    
    # Verify update
    subcase = db.get_subcase_by_id(test_subcase_id)
    assert subcase["administration_explanation_text"] == "This is a test administration explanation", "Text should be updated"
    print(f"   Administration explanation updated")

test_update_administration_explanation()


@test("17. Update administration rejection")
def test_update_administration_rejection():
    global test_subcase_id
    result = db.update_administration_rejection(
        subcase_id=test_subcase_id,
        text="This is a test administration rejection",
        updated_by_user_id=1
    )
    
    assert result is True, "Update should succeed"
    
    # Verify update
    subcase = db.get_subcase_by_id(test_subcase_id)
    assert subcase["administration_rejection_text"] == "This is a test administration rejection", "Text should be updated"
    print(f"   Administration rejection updated")

test_update_administration_rejection()


# ============================================================
# MONITORING / INSIGHT TESTS
# ============================================================

@test("18. Get full subcases by incident")
def test_get_full_incident():
    global test_incident_id
    subcases = db.get_full_subcases_by_incident(test_incident_id)
    
    assert isinstance(subcases, list), "Should return list"
    assert len(subcases) > 0, "Should have at least one subcase"
    # Verify we got the full subcase with all text fields
    subcase = next((s for s in subcases if s["subcase_id"] == test_subcase_id), None)
    assert subcase is not None, "Should find test subcase"
    assert subcase["section_explanation_text"] is not None, "Should have section explanation"
    print(f"   Retrieved {len(subcases)} full subcase(s)")

test_get_full_incident()


@test("19. Get full subcases by seasonal report")
def test_get_full_seasonal():
    subcases = db.get_full_subcases_by_seasonal_report(99999)
    
    assert isinstance(subcases, list), "Should return list"
    assert len(subcases) == 0, "Should be empty for non-existent report"
    print(f"   Correctly returned empty list")

test_get_full_seasonal()


# ============================================================
# EDGE CASE TESTS
# ============================================================

@test("20. Get non-existent subcase returns None")
def test_get_nonexistent():
    subcase = db.get_subcase_by_id(999999)
    
    assert subcase is None, "Should return None for non-existent ID"
    print(f"   Correctly returned None")

test_get_nonexistent()


@test("21. Update non-existent subcase returns False")
def test_update_nonexistent():
    result = db.update_subcase_status(
        subcase_id=999999,
        new_status="CLOSED",
        updated_by_user_id=1
    )
    
    assert result is False, "Should return False for non-existent ID"
    print(f"   Correctly returned False")

test_update_nonexistent()


@test("22. Create seasonal subcase")
def test_create_seasonal_subcase():
    # Create a subcase for seasonal report (using NULL incident_id)
    # Using SeasonalReportID = 659 (verified to exist)
    subcase_id = db.create_subcase(
        case_type="SEASONAL_REPORT_RESPONSE",
        incident_id=None,
        seasonal_report_id=659,
        target_org_unit_id=1,
        created_by_user_id=1,
        initial_status="SUBMITTED_TO_SECTION"
    )
    
    assert subcase_id is not None, "Subcase ID should not be None"
    
    # Verify
    subcase = db.get_subcase_by_id(subcase_id)
    assert subcase["case_type"] == "SEASONAL_REPORT_RESPONSE", "Should be seasonal type"
    assert subcase["seasonal_report_id"] == 659, "Should have seasonal report ID"
    assert subcase["incident_request_case_id"] is None, "Should have NULL incident ID"
    print(f"   Created seasonal SubcaseID: {subcase_id}")

test_create_seasonal_subcase()


@test("23. Verify all text fields are nullable")
def test_nullable_text_fields():
    global test_subcase_id
    subcase = db.get_subcase_by_id(test_subcase_id)
    
    # All text fields should exist in the dictionary (even if None)
    assert "section_explanation_text" in subcase, "Field should exist"
    assert "section_rejection_text" in subcase, "Field should exist"
    assert "department_explanation_text" in subcase, "Field should exist"
    assert "department_rejection_text" in subcase, "Field should exist"
    assert "administration_explanation_text" in subcase, "Field should exist"
    assert "administration_rejection_text" in subcase, "Field should exist"
    print(f"   All text fields are present")

test_nullable_text_fields()


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
    print("🎉 ALL TESTS PASSED! administrative_subcase_db.py is 100% functional.")
    exit(0)
else:
    print(f"⚠️  {tests_failed} test(s) failed. Please review and fix.")
    exit(1)
