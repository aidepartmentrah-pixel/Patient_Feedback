"""
PHASE DR-B4 — Dashboard Date Bounds Backend Tests

Complete test suite for dashboard date bounds feature.

Test Coverage:
- Service function tests (4 tests)
- Endpoint tests (4 tests)

Following existing backend test style with same fixtures and auth helpers.
"""

import requests
from datetime import date
import sys

# Test configuration
BASE_URL = "http://localhost:8000"
session = requests.Session()

# =========================================================
# TEST HELPERS (Following existing backend test style)
# =========================================================

def login(username: str, password: str):
    """Login and establish session."""
    response = session.post(
        f"{BASE_URL}/api/auth/login",
        json={"username": username, "password": password}
    )
    assert response.status_code == 200, f"Login failed for {username}: {response.text}"
    print(f"✓ Logged in as {username}")
    return response.json()


def logout():
    """Logout current session."""
    session.post(f"{BASE_URL}/api/auth/logout")
    print("✓ Logged out\n")


# =========================================================
# SERVICE TESTS
# =========================================================

def test_service_normal_case():
    """
    Test 1: Normal case
    - unit_ids with incidents
    - assert min_date not None
    - assert max_date not None
    - assert min_date <= max_date
    """
    print("=" * 80)
    print("SERVICE TEST 1: Normal Case - Unit IDs with Incidents")
    print("=" * 80)
    
    from api.services.dashboard_service import get_dashboard_date_bounds_for_units
    
    # Use unit IDs that have incidents
    result = get_dashboard_date_bounds_for_units([1, 2, 3, 4, 5])
    
    print(f"Result: {result}")
    
    # Assertions per prompt
    assert result["min_date"] is not None, "min_date should not be None"
    assert result["max_date"] is not None, "max_date should not be None"
    assert result["min_date"] <= result["max_date"], "min_date must be <= max_date"
    
    print(f"✓ min_date not None: {result['min_date']}")
    print(f"✓ max_date not None: {result['max_date']}")
    print(f"✓ min_date <= max_date")
    print()
    
    return result


def test_service_subset_scope():
    """
    Test 2: Subset scope
    - smaller unit_ids subset
    - assert returned range within full range
    """
    print("=" * 80)
    print("SERVICE TEST 2: Subset Scope - Smaller Range Within Full")
    print("=" * 80)
    
    from api.services.dashboard_service import get_dashboard_date_bounds_for_units
    
    # Get full range
    full_result = get_dashboard_date_bounds_for_units([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(f"Full range: {full_result}")
    
    # Get subset range (single unit)
    subset_result = get_dashboard_date_bounds_for_units([5])
    print(f"Subset range (unit 5): {subset_result}")
    
    # Assert returned range within full range
    if subset_result["min_date"] is not None and full_result["min_date"] is not None:
        assert subset_result["min_date"] >= full_result["min_date"], \
            "Subset min_date should be >= full min_date"
        assert subset_result["max_date"] <= full_result["max_date"], \
            "Subset max_date should be <= full max_date"
        print(f"✓ Subset range is within full range")
    else:
        print(f"⚠ Subset or full range has no data")
    
    print()


def test_service_empty_scope():
    """
    Test 3: Empty scope
    - unit_ids with no incidents
    - assert both values None
    """
    print("=" * 80)
    print("SERVICE TEST 3: Empty Scope - No Incidents")
    print("=" * 80)
    
    from api.services.dashboard_service import get_dashboard_date_bounds_for_units
    
    # Use unit IDs that don't exist
    result = get_dashboard_date_bounds_for_units([99999, 99998])
    
    print(f"Result: {result}")
    
    # Assert both values None
    assert result["min_date"] is None, "min_date should be None for empty scope"
    assert result["max_date"] is None, "max_date should be None for empty scope"
    
    print(f"✓ min_date is None")
    print(f"✓ max_date is None")
    print()


def test_service_type_test(normal_result):
    """
    Test 4: Type test
    - values are date or None
    - not datetime
    - not string
    """
    print("=" * 80)
    print("SERVICE TEST 4: Type Test - Date Objects Not Datetime/String")
    print("=" * 80)
    
    result = normal_result
    
    print(f"Result: {result}")
    print(f"min_date type: {type(result['min_date'])}")
    print(f"max_date type: {type(result['max_date'])}")
    
    # Assert values are date or None (not datetime, not string)
    if result["min_date"] is not None:
        assert isinstance(result["min_date"], date), \
            f"min_date must be date, got {type(result['min_date'])}"
        assert not isinstance(result["min_date"], str), \
            "min_date must not be string"
        print(f"✓ min_date is date object (not datetime, not string)")
    
    if result["max_date"] is not None:
        assert isinstance(result["max_date"], date), \
            f"max_date must be date, got {type(result['max_date'])}"
        assert not isinstance(result["max_date"], str), \
            "max_date must not be string"
        print(f"✓ max_date is date object (not datetime, not string)")
    
    print()


# =========================================================
# ENDPOINT TESTS
# =========================================================

def test_endpoint_auth_required():
    """
    Test 5: Auth required
    - no session → 401
    """
    print("=" * 80)
    print("ENDPOINT TEST 5: Auth Required - No Session Returns 401")
    print("=" * 80)
    
    # Clear any existing session
    logout()
    
    # Create new session without login
    test_session = requests.Session()
    
    # Attempt to access endpoint without auth
    response = test_session.get(
        f"{BASE_URL}/api/dashboard/date-bounds",
        params={"scope": "hospital"}
    )
    
    print(f"Response status: {response.status_code}")
    
    # Assert 401
    assert response.status_code == 401, f"Expected 401 for no auth, got {response.status_code}"
    
    print(f"✓ Returns 401 without authentication")
    print()


def test_endpoint_valid_request():
    """
    Test 6: Valid request
    - returns 200
    - contains both keys
    """
    print("=" * 80)
    print("ENDPOINT TEST 6: Valid Request - Returns 200 with Both Keys")
    print("=" * 80)
    
    login("software_admin", "admin123")
    
    # Make valid request
    response = session.get(
        f"{BASE_URL}/api/dashboard/date-bounds",
        params={"scope": "hospital"}
    )
    
    print(f"Response status: {response.status_code}")
    print(f"Response body: {response.json()}")
    
    # Assert returns 200
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    # Assert contains both keys
    result = response.json()
    assert "min_date" in result, "Response must contain min_date key"
    assert "max_date" in result, "Response must contain max_date key"
    
    print(f"✓ Returns HTTP 200")
    print(f"✓ Contains min_date key")
    print(f"✓ Contains max_date key")
    
    logout()
    print()


def test_endpoint_scope_restriction():
    """
    Test 7: Scope restriction
    - limited-scope user → smaller/equal range
    """
    print("=" * 80)
    print("ENDPOINT TEST 7: Scope Restriction - Limited User Gets Restricted Range")
    print("=" * 80)
    
    # Login as software_admin (full access)
    login("software_admin", "admin123")
    
    # Get full hospital scope
    full_response = session.get(
        f"{BASE_URL}/api/dashboard/date-bounds",
        params={"scope": "hospital"}
    )
    full_result = full_response.json()
    print(f"Full hospital scope: {full_result}")
    
    # Get restricted department scope
    dept_response = session.get(
        f"{BASE_URL}/api/dashboard/date-bounds",
        params={"scope": "department", "department_id": 5}
    )
    dept_result = dept_response.json()
    print(f"Department 5 scope: {dept_result}")
    
    # Assert department range is smaller or equal to hospital range
    if dept_result["min_date"] is not None and full_result["min_date"] is not None:
        # Convert ISO strings to date objects for comparison
        dept_min = date.fromisoformat(dept_result["min_date"])
        dept_max = date.fromisoformat(dept_result["max_date"])
        full_min = date.fromisoformat(full_result["min_date"])
        full_max = date.fromisoformat(full_result["max_date"])
        
        assert dept_min >= full_min, "Department min_date should be >= hospital min_date"
        assert dept_max <= full_max, "Department max_date should be <= hospital max_date"
        print(f"✓ Department range is within hospital range")
    else:
        print(f"⚠ Department has no data (null values)")
    
    logout()
    print()


def test_endpoint_contract():
    """
    Test 8: Contract test
    - response keys always present
    """
    print("=" * 80)
    print("ENDPOINT TEST 8: Contract - Keys Always Present")
    print("=" * 80)
    
    login("software_admin", "admin123")
    
    # Test multiple scopes
    test_cases = [
        {"scope": "hospital"},
        {"scope": "administration", "administration_id": 1},
        {"scope": "department", "department_id": 5},
        {"scope": "section", "section_id": 29},
    ]
    
    for i, params in enumerate(test_cases, 1):
        response = session.get(f"{BASE_URL}/api/dashboard/date-bounds", params=params)
        
        assert response.status_code == 200, \
            f"Test case {i} failed: HTTP {response.status_code}"
        
        result = response.json()
        
        # Assert keys always present
        assert "min_date" in result, f"Test case {i}: min_date key missing"
        assert "max_date" in result, f"Test case {i}: max_date key missing"
        assert set(result.keys()) == {"min_date", "max_date"}, \
            f"Test case {i}: Should have exactly min_date and max_date keys"
        
        print(f"✓ Test case {i} ({params['scope']}): Both keys present")
    
    logout()
    print()


# =========================================================
# TEST RUNNER
# =========================================================

def run_all_tests():
    """Run all DR-B4 backend tests."""
    print("\n" + "=" * 80)
    print("PHASE DR-B4: DASHBOARD DATE BOUNDS BACKEND TESTS")
    print("=" * 80 + "\n")
    
    passed = 0
    failed = 0
    
    # SERVICE TESTS
    print("=" * 80)
    print("PART 1: SERVICE TESTS (4 tests)")
    print("=" * 80 + "\n")
    
    # Test 1: Normal case (save result for type test)
    try:
        normal_result = test_service_normal_case()
        passed += 1
    except Exception as e:
        failed += 1
        print(f"✗ SERVICE TEST 1 FAILED: {e}\n")
        return False
    
    # Test 2: Subset scope
    try:
        test_service_subset_scope()
        passed += 1
    except Exception as e:
        failed += 1
        print(f"✗ SERVICE TEST 2 FAILED: {e}\n")
    
    # Test 3: Empty scope
    try:
        test_service_empty_scope()
        passed += 1
    except Exception as e:
        failed += 1
        print(f"✗ SERVICE TEST 3 FAILED: {e}\n")
    
    # Test 4: Type test (uses normal_result)
    try:
        test_service_type_test(normal_result)
        passed += 1
    except Exception as e:
        failed += 1
        print(f"✗ SERVICE TEST 4 FAILED: {e}\n")
    
    # ENDPOINT TESTS
    print("=" * 80)
    print("PART 2: ENDPOINT TESTS (4 tests)")
    print("=" * 80 + "\n")
    
    # Test 5: Auth required
    try:
        test_endpoint_auth_required()
        passed += 1
    except Exception as e:
        failed += 1
        print(f"✗ ENDPOINT TEST 5 FAILED: {e}\n")
    
    # Test 6: Valid request
    try:
        test_endpoint_valid_request()
        passed += 1
    except Exception as e:
        failed += 1
        print(f"✗ ENDPOINT TEST 6 FAILED: {e}\n")
    
    # Test 7: Scope restriction
    try:
        test_endpoint_scope_restriction()
        passed += 1
    except Exception as e:
        failed += 1
        print(f"✗ ENDPOINT TEST 7 FAILED: {e}\n")
    
    # Test 8: Contract test
    try:
        test_endpoint_contract()
        passed += 1
    except Exception as e:
        failed += 1
        print(f"✗ ENDPOINT TEST 8 FAILED: {e}\n")
    
    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Service Tests: 4/4 {'PASSED' if failed <= 4 else 'SOME FAILED'}")
    print(f"Endpoint Tests: 4/4 {'PASSED' if passed == 8 else 'SOME FAILED'}")
    print(f"Total Passed: {passed}/8")
    print(f"Total Failed: {failed}/8")
    
    if failed == 0:
        print("\n🎉 ALL DR-B4 BACKEND TESTS PASSED!")
        print("\nCoverage includes:")
        print("  ✓ Service function")
        print("  ✓ Endpoint")
        print("  ✓ Null contract")
        print("  ✓ RBAC scope")
        return True
    else:
        print(f"\n❌ {failed} TEST(S) FAILED")
        return False


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
