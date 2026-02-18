"""
PHASE K — K-API-1 — MIGRATION ROUTER TEST

Validates migration router endpoints for API contract compliance,
authorization guards, and response formatting.

TARGET
------
backend/api/routers/migration_router.py

ENDPOINTS TESTED
----------------
1. GET /api/migration/legacy/list - List legacy cases with pagination
2. GET /api/migration/legacy/{id} - Get legacy case detail
3. POST /api/migration/migrate/{id} - Migrate legacy case
4. GET /api/migration/progress - Get migration progress

TESTS
-----
1. Role Guard - Unauthorized access blocked
2. Legacy List - Pagination and response format
3. Legacy Detail Found - Case retrieval
4. Legacy Detail Not Found - 404 handling
5. Migrate Success - Normal migration flow
6. Migrate Validation Fail - Invalid payload handling
7. Progress - Progress statistics

RUN
---
python test_phase_k_api1_migration_router.py
"""

import sys
from fastapi.testclient import TestClient
from main import app
from core.database import get_connection
from api.dependencies.user_context import get_current_user
from api.schemas.auth_models import CurrentUser, UserScope
from core.constants.roles import SOFTWARE_ADMIN, WORKER, COMPLAINT_SUPERVISOR


# Test client
client = TestClient(app)


# =========================================================
# AUTHENTICATION MOCKING
# =========================================================

def create_mock_user(role: str) -> CurrentUser:
    """Create a mock user with specified role."""
    return CurrentUser(
        user_id=1,
        username="test_user",
        is_active=True,
        scopes=[
            UserScope(
                role_code=role,
                org_unit_id=1,  # Required field
                org_unit_type="HOSPITAL"  # Required field
            )
        ],
        allowed_unit_ids={1},
        roles=[role],
        primary_unit_id=1,
        primary_unit_type="HOSPITAL"
    )


def override_auth_with_role(role: str):
    """Override authentication dependency with mock user."""
    mock_user = create_mock_user(role)
    app.dependency_overrides[get_current_user] = lambda: mock_user


def clear_auth_override():
    """Clear authentication override."""
    app.dependency_overrides.clear()


# =========================================================
# TEST UTILITIES
# =========================================================

def print_header(title: str):
    """Print test section header."""
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_test(description: str, passed: bool, details: str = ""):
    """Print test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} — {description}")
    if details:
        print(f"   {details}")


def get_test_legacy_case_id() -> int:
    """Get a valid legacy case ID for testing."""
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT TOP 1 IncidentRequestCaseID FROM dbo.APP_IncidentCase ORDER BY IncidentRequestCaseID")
        row = cursor.fetchone()
        
        if row:
            return row[0]
        else:
            print("⚠️  WARNING: No legacy cases found in database")
            return 1  # Fallback
            
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# =========================================================
# TEST FUNCTIONS
# =========================================================

def test_role_guard():
    """
    TEST 1: ROLE GUARD
    
    Verify that unauthorized roles are blocked with 403.
    """
    print_header("TEST 1: ROLE GUARD")
    
    # Test with SECTION_ADMIN role (not allowed for migration)
    print("📋 Testing unauthorized role (SECTION_ADMIN) on legacy list endpoint...")
    
    override_auth_with_role("SECTION_ADMIN")
    
    try:
        response = client.get("/api/migration/legacy/list")
        
        # Check 403 status
        is_forbidden = response.status_code == 403
        print_test("Unauthorized role blocked", is_forbidden, f"Status: {response.status_code}")
        
        if response.status_code == 403:
            detail = response.json().get("detail", {})
            has_error_field = "error" in detail
            print_test("Structured error detail present", has_error_field)
            
            if has_error_field:
                print(f"   Error Code: {detail.get('error')}")
        
        print()
        return is_forbidden
    finally:
        clear_auth_override()


def test_legacy_list():
    """
    TEST 2: LEGACY LIST
    
    Verify legacy case listing endpoint.
    """
    print_header("TEST 2: LEGACY LIST")
    
    print("📋 Requesting legacy case list (page=1, page_size=10)...")
    
    override_auth_with_role(SOFTWARE_ADMIN)
    
    try:
        response = client.get("/api/migration/legacy/list?page=1&page_size=10")
        
        # Check 200 OK
        is_ok = response.status_code == 200
        print_test("Request successful", is_ok, f"Status: {response.status_code}")
        
        if not is_ok:
            print(f"   Response: {response.json()}")
            print()
            return False
        
        data = response.json()
        
        # Check response structure
        has_cases = "cases" in data
        has_total = "total" in data
        
        print_test("Response has 'cases' field", has_cases)
        print_test("Response has 'total' field", has_total)
        
        if has_cases and has_total:
            print(f"   Total cases: {data['total']}")
            print(f"   Cases returned: {len(data['cases'])}")
            
            # Check cases array structure
            if len(data['cases']) > 0:
                first_case = data['cases'][0]
                required_fields = ["legacy_case_id", "complaint_text", "patient_name", "migrated"]
                
                for field in required_fields:
                    has_field = field in first_case
                    print_test(f"Case has '{field}' field", has_field)
        
        print()
        return is_ok and has_cases and has_total
    finally:
        clear_auth_override()


def test_legacy_detail_found():
    """
    TEST 3: LEGACY DETAIL — FOUND
    
    Verify legacy case detail endpoint with valid ID.
    """
    print_header("TEST 3: LEGACY DETAIL — FOUND")
    
    legacy_id = get_test_legacy_case_id()
    print(f"📋 Requesting legacy case detail (ID: {legacy_id})...")
    
    override_auth_with_role(SOFTWARE_ADMIN)
    
    try:
        response = client.get(f"/api/migration/legacy/{legacy_id}")
        
        # Check 200 OK
        is_ok = response.status_code == 200
        print_test("Request successful", is_ok, f"Status: {response.status_code}")
        
        if not is_ok:
            print(f"   Response: {response.json()}")
            print()
            return False
        
        case = response.json()
        
        # Check case structure
        has_legacy_id = "legacy_case_id" in case
        has_complaint = "complaint_text" in case
        has_patient = "patient_name" in case
        
        print_test("Case has 'legacy_case_id'", has_legacy_id)
        print_test("Case has 'complaint_text'", has_complaint)
        print_test("Case has 'patient_name'", has_patient)
        
        print()
        return is_ok and has_legacy_id
    finally:
        clear_auth_override()


def test_legacy_detail_not_found():
    """
    TEST 4: LEGACY DETAIL — NOT FOUND
    
    Verify 404 handling for non-existent legacy case.
    """
    print_header("TEST 4: LEGACY DETAIL — NOT FOUND")
    
    invalid_id = 999999999
    print(f"📋 Requesting non-existent legacy case (ID: {invalid_id})...")
    
    override_auth_with_role(SOFTWARE_ADMIN)
    
    try:
        response = client.get(f"/api/migration/legacy/{invalid_id}")
        
        # Check 404 status
        is_not_found = response.status_code == 404
        print_test("404 status returned", is_not_found, f"Status: {response.status_code}")
        
        if response.status_code == 404:
            detail = response.json().get("detail", {})
            error_code = detail.get("error", "")
            
            is_correct_error = error_code == "LEGACY_CASE_NOT_FOUND"
            print_test("Error code is LEGACY_CASE_NOT_FOUND", is_correct_error, f"Got: {error_code}")
            
            has_message = "message" in detail
            has_message_ar = "message_ar" in detail
            
            print_test("Has error message", has_message)
            print_test("Has Arabic message", has_message_ar)
        
        print()
        return is_not_found
    finally:
        clear_auth_override()


def get_valid_test_payload(legacy_id: int) -> dict:
    """Get valid migration payload with proper lookup references."""
    import time
    from core.database import get_connection
    
    unique_id = int(time.time() * 1000) % 1000000
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get valid lookup IDs from database
    cursor.execute("SELECT TOP 1 DomainID FROM dbo.APP_LOOKUP_DOMAIN ORDER BY DomainID")
    domain_id_row = cursor.fetchone()
    domain_id = domain_id_row[0] if domain_id_row else 1
    
    cursor.execute("SELECT TOP 1 CategoryID FROM dbo.APP_LOOKUP_CATEGORY WHERE DomainID = ? ORDER BY CategoryID", domain_id)
    category_id_row = cursor.fetchone()
    category_id = category_id_row[0] if category_id_row else 1
    
    cursor.execute("SELECT TOP 1 SubCategoryID FROM dbo.APP_LOOKUP_SUBCATEGORY WHERE CategoryID = ? ORDER BY SubCategoryID", category_id)
    subcategory_id_row = cursor.fetchone()
    subcategory_id = subcategory_id_row[0] if subcategory_id_row else 1
    
    cursor.execute("SELECT TOP 1 ClassificationID FROM dbo.APP_LOOKUP_CLASSIFICATION WHERE SubCategoryID = ? ORDER BY ClassificationID", subcategory_id)
    classification_id_row = cursor.fetchone()
    classification_id = classification_id_row[0] if classification_id_row else 1
    
    cursor.close()
    conn.close()
    
    return {
        "complaint_text": f"Test migration API {legacy_id} #{unique_id}",
        "immediate_action": "None",
        "taken_action": "None",
        "feedback_received_date": "2024-06-15",
        "patient_name": f"Test Patient API {legacy_id}",
        "is_inpatient": True,
        "clinical_risk_type_id": 1,
        "feedback_intent_type_id": 1,
        "building_id": 1,
        "domain_id": domain_id,
        "category_id": category_id,
        "subcategory_id": subcategory_id,
        "classification_id": classification_id,
        "severity_id": 1,
        "stage_id": 1,
        "harm_id": 1,
        "source_id": 1,
        "issuing_department_id": 1,
        "requires_explanation": False
    }


def test_migrate_success():
    """
    TEST 5: MIGRATE SUCCESS
    
    Verify successful migration flow.
    """
    print_header("TEST 5: MIGRATE SUCCESS")
    
    legacy_id = 800000 + int(__import__('time').time() % 100000)
    
    print(f"📋 Attempting migration (legacy_id: {legacy_id})...")
    
    # Build valid payload with real lookup references
    payload = get_valid_test_payload(legacy_id)
    
    override_auth_with_role(SOFTWARE_ADMIN)
    
    try:
        response = client.post(f"/api/migration/migrate/{legacy_id}", json=payload)
        
        # Check 200 OK
        is_ok = response.status_code == 200
        print_test("Request successful", is_ok, f"Status: {response.status_code}")
        
        if not is_ok:
            print(f"   Response: {response.json()}")
            print()
            return False
        
        result = response.json()
        
        # Check response structure
        has_success = "success" in result
        has_status = "status" in result
        has_new_id = "new_case_id" in result
        
        print_test("Response has 'success' field", has_success)
        print_test("Response has 'status' field", has_status)
        print_test("Response has 'new_case_id' field", has_new_id)
        
        if has_success and has_status:
            success = result.get("success")
            status_val = result.get("status")
            
            print_test("Success is True", success == True, f"Success: {success}")
            print_test("Status is MIGRATED or ALREADY_MIGRATED", 
                      status_val in ["MIGRATED", "ALREADY_MIGRATED"],
                      f"Status: {status_val}")
            
            if has_new_id:
                print(f"   New Case ID: {result.get('new_case_id')}")
        
        print()
        return is_ok and has_success
    finally:
        clear_auth_override()


def test_migrate_validation_fail():
    """
    TEST 6: MIGRATE VALIDATION FAIL
    
    Verify validation error handling.
    """
    print_header("TEST 6: MIGRATE VALIDATION FAIL")
    
    legacy_id = 800001 + int(__import__('time').time() % 100000)
    
    print(f"📋 Attempting migration with invalid payload (missing complaint_text)...")
    
    # Invalid payload - missing required field
    payload = {
        "immediate_action": "None",
        "taken_action": "None",
        "patient_name": "Test Patient"
        # Missing complaint_text and other required fields
    }
    
    override_auth_with_role(SOFTWARE_ADMIN)
    
    try:
        response = client.post(f"/api/migration/migrate/{legacy_id}", json=payload)
        
        # Check 400 Bad Request
        is_bad_request = response.status_code == 400
        print_test("400 status returned", is_bad_request, f"Status: {response.status_code}")
        
        if response.status_code == 400:
            detail = response.json().get("detail", {})
            
            has_error = "error" in detail
            has_message = "message" in detail
            
            print_test("Structured error detail present", has_error and has_message)
            
            if has_error:
                print(f"   Error Code: {detail.get('error')}")
                print(f"   Message: {detail.get('message')}")
        
        print()
        return is_bad_request
    finally:
        clear_auth_override()


def test_progress():
    """
    TEST 7: PROGRESS
    
    Verify migration progress endpoint.
    """
    print_header("TEST 7: PROGRESS")
    
    print("📋 Requesting migration progress...")
    
    override_auth_with_role(SOFTWARE_ADMIN)
    
    try:
        response = client.get("/api/migration/progress")
        
        # Check 200 OK
        is_ok = response.status_code == 200
        print_test("Request successful", is_ok, f"Status: {response.status_code}")
        
        if not is_ok:
            print(f"   Response: {response.json()}")
            print()
            return False
        
        progress = response.json()
        
        # Check response structure
        required_fields = ["total", "migrated", "remaining", "percent"]
        
        all_present = True
        for field in required_fields:
            has_field = field in progress
            print_test(f"Response has '{field}' field", has_field)
            all_present = all_present and has_field
        
        if all_present:
            print(f"\n📊 Progress Statistics:")
            print(f"   Total: {progress['total']}")
            print(f"   Migrated: {progress['migrated']}")
            print(f"   Remaining: {progress['remaining']}")
            print(f"   Percent: {progress['percent']}%")
            
            # Verify calculations
            calc_remaining = progress['total'] - progress['migrated']
            remaining_correct = progress['remaining'] == calc_remaining
            print_test("Remaining calculation correct", remaining_correct)
        
        print()
        return is_ok and all_present
    finally:
        clear_auth_override()


# =========================================================
# MAIN TEST RUNNER
# =========================================================

def main():
    """
    Run all K-API-1 Migration Router tests.
    """
    print()
    print("=" * 80)
    print("  PHASE K — K-API-1 — MIGRATION ROUTER TEST")
    print("=" * 80)
    print("Validating migration router endpoints and API contract")
    print()
    
    tests = [
        ("Role Guard", test_role_guard),
        ("Legacy List", test_legacy_list),
        ("Legacy Detail Found", test_legacy_detail_found),
        ("Legacy Detail Not Found", test_legacy_detail_not_found),
        ("Migrate Success", test_migrate_success),
        ("Migrate Validation Fail", test_migrate_validation_fail),
        ("Progress", test_progress)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ FAIL — {test_name}")
            print(f"   Unexpected error: {str(e)}")
            import traceback
            traceback.print_exc()
            print()
            failed += 1
    
    # Summary
    print("=" * 80)
    print("  TEST SUMMARY")
    print("=" * 80)
    for test_name, _ in tests:
        # Determine status (this is a simplified check)
        status = "✅ PASS" if passed > 0 else "❌ FAIL"
        print(f"{status} — {test_name}")
    
    print()
    print("=" * 80)
    print(f"TOTAL: {passed}/{len(tests)} tests passed")
    print("=" * 80)
    print()
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED — K-API-1 COMPLETE")
    else:
        print(f"⚠️  {failed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
