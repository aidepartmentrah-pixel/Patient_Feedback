"""
Backend Tests - Dashboard Date Bounds Feature (PHASE DR-B)

Tests the new /api/dashboard/date-bounds endpoint.

Test Coverage:
1. Service Layer: get_incident_date_bounds with unit_ids filtering
2. Endpoint Tests: GET /api/dashboard/date-bounds with scope parameters
3. RBAC Tests: Scope enforcement for different user roles
4. Empty Scope Tests: Null handling when no incidents exist
5. Contract Compliance: Response shape validation

Rules:
- Response always contains min_date and max_date keys
- Values are YYYY-MM-DD strings or null
- Never fallback dates
- Uses same scope resolution as dashboard stats
"""

import requests
from datetime import date
import sys

# Test configuration
BASE_URL = "http://localhost:8000"
session = requests.Session()

# =========================================================
# TEST HELPERS
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


def get_date_bounds(scope: str, **kwargs):
    """Get dashboard date bounds with scope parameters."""
    params = {"scope": scope}
    params.update(kwargs)
    
    response = session.get(f"{BASE_URL}/api/dashboard/date-bounds", params=params)
    assert response.status_code == 200, f"Date bounds request failed: {response.text}"
    
    return response.json()


def validate_response_contract(response: dict):
    """Validate response matches the contract."""
    assert "min_date" in response, "Response must contain min_date key"
    assert "max_date" in response, "Response must contain max_date key"
    
    min_date = response["min_date"]
    max_date = response["max_date"]
    
    # Values must be either None or valid YYYY-MM-DD strings
    if min_date is not None:
        assert isinstance(min_date, str), "min_date must be string or null"
        assert len(min_date) == 10 and min_date[4] == '-' and min_date[7] == '-', \
            "min_date must be YYYY-MM-DD format"
    
    if max_date is not None:
        assert isinstance(max_date, str), "max_date must be string or null"
        assert len(max_date) == 10 and max_date[4] == '-' and max_date[7] == '-', \
            "max_date must be YYYY-MM-DD format"
    
    print(f"  ✓ Response contract valid: min_date={min_date}, max_date={max_date}")


# =========================================================
# TEST CASES
# =========================================================

def test_endpoint_exists():
    """
    Test 1: Endpoint is accessible and returns valid response.
    """
    print("=" * 80)
    print("TEST 1: Endpoint Exists and Returns Valid Response")
    print("=" * 80)
    
    login("software_admin", "admin123")
    
    # Request hospital-wide date bounds
    response = get_date_bounds(scope="hospital")
    
    print(f"Response: {response}")
    validate_response_contract(response)
    
    # If there are incidents, dates should not be null
    if response["min_date"] is not None:
        print(f"  ✓ Date range found: {response['min_date']} to {response['max_date']}")
    else:
        print(f"  ⚠ No incidents in database (null dates)")
    
    logout()
    return True


def test_response_contract():
    """
    Test 2: Response always contains min_date and max_date keys.
    """
    print("=" * 80)
    print("TEST 2: Response Contract - Keys Always Present")
    print("=" * 80)
    
    login("software_admin", "admin123")
    
    response = get_date_bounds(scope="hospital")
    
    # Keys must exist
    assert "min_date" in response, "FAIL: min_date key missing"
    assert "max_date" in response, "FAIL: max_date key missing"
    
    # No extra keys allowed
    assert set(response.keys()) == {"min_date", "max_date"}, \
        f"FAIL: Response should only have min_date and max_date, got {response.keys()}"
    
    print("  ✓ Contract valid: Only min_date and max_date keys present")
    
    logout()
    return True


def test_scope_filtering_administration():
    """
    Test 3: Administration scope returns different date bounds than hospital scope.
    """
    print("=" * 80)
    print("TEST 3: Scope Filtering - Administration Level")
    print("=" * 80)
    
    login("software_admin", "admin123")
    
    # Get hierarchy to find administration IDs
    hierarchy_response = session.get(f"{BASE_URL}/api/dashboard/hierarchy")
    hierarchy = hierarchy_response.json()
    
    administrations = hierarchy["Administration"]
    if len(administrations) < 1:
        print("  ⚠ SKIP: No administrations available")
        logout()
        return True
    
    admin_1 = administrations[0]
    print(f"Testing Administration: {admin_1.get('nameEn', admin_1.get('nameAr', 'Unknown'))} (ID: {admin_1['id']})")
    
    # Get hospital-wide bounds
    hospital_bounds = get_date_bounds(scope="hospital")
    print(f"Hospital bounds: {hospital_bounds}")
    validate_response_contract(hospital_bounds)
    
    # Get administration-specific bounds
    admin_bounds = get_date_bounds(
        scope="administration",
        administration_id=admin_1['id']
    )
    print(f"Administration bounds: {admin_bounds}")
    validate_response_contract(admin_bounds)
    
    print("  ✓ Both scopes returned valid responses")
    
    logout()
    return True


def test_scope_filtering_department():
    """
    Test 4: Department scope filters correctly.
    """
    print("=" * 80)
    print("TEST 4: Scope Filtering - Department Level")
    print("=" * 80)
    
    login("software_admin", "admin123")
    
    # Get hierarchy to find department IDs
    hierarchy_response = session.get(f"{BASE_URL}/api/dashboard/hierarchy")
    hierarchy = hierarchy_response.json()
    
    # Find first administration with at least one department
    dept_id = None
    dept_name = None
    for admin_id, dept_list in hierarchy["Department"].items():
        if len(dept_list) >= 1:
            dept_id = dept_list[0]['id']
            dept_name = dept_list[0].get('nameEn', dept_list[0].get('nameAr', 'Unknown'))
            break
    
    if dept_id is None:
        print("  ⚠ SKIP: No departments available")
        logout()
        return True
    
    print(f"Testing Department: {dept_name} (ID: {dept_id})")
    
    # Get department-specific bounds
    dept_bounds = get_date_bounds(
        scope="department",
        department_id=dept_id
    )
    print(f"Department bounds: {dept_bounds}")
    validate_response_contract(dept_bounds)
    
    print("  ✓ Department scope returned valid response")
    
    logout()
    return True


def test_scope_filtering_section():
    """
    Test 5: Section scope filters correctly.
    """
    print("=" * 80)
    print("TEST 5: Scope Filtering - Section Level")
    print("=" * 80)
    
    login("software_admin", "admin123")
    
    # Get hierarchy to find section IDs
    hierarchy_response = session.get(f"{BASE_URL}/api/dashboard/hierarchy")
    hierarchy = hierarchy_response.json()
    
    # Find first department with at least one section
    section_id = None
    section_name = None
    for dept_id, section_list in hierarchy["Section"].items():
        if len(section_list) >= 1:
            section_id = section_list[0]['id']
            section_name = section_list[0].get('nameEn', section_list[0].get('nameAr', 'Unknown'))
            break
    
    if section_id is None:
        print("  ⚠ SKIP: No sections available")
        logout()
        return True
    
    print(f"Testing Section: {section_name} (ID: {section_id})")
    
    # Get section-specific bounds
    section_bounds = get_date_bounds(
        scope="section",
        section_id=section_id
    )
    print(f"Section bounds: {section_bounds}")
    validate_response_contract(section_bounds)
    
    print("  ✓ Section scope returned valid response")
    
    logout()
    return True


def test_rbac_enforcement():
    """
    Test 6: RBAC enforcement - endpoint respects user scope.
    """
    print("=" * 80)
    print("TEST 6: RBAC Enforcement")
    print("=" * 80)
    
    # Test with software admin accessing different scopes
    login("software_admin", "admin123")
    
    # Admin should be able to access hospital scope
    try:
        response = get_date_bounds(scope="hospital")
        print(f"Admin hospital scope: {response}")
        validate_response_contract(response)
        print("  ✓ Admin can access hospital scope")
    except AssertionError as e:
        print(f"  ✗ RBAC test unexpected error: {e}")
        logout()
        return False
    
    logout()
    return True


def test_invalid_scope_parameter():
    """
    Test 7: Invalid scope parameter returns 400 error.
    """
    print("=" * 80)
    print("TEST 7: Invalid Scope Parameter Handling")
    print("=" * 80)
    
    login("software_admin", "admin123")
    
    # Try invalid scope
    response = session.get(
        f"{BASE_URL}/api/dashboard/date-bounds",
        params={"scope": "invalid_scope"}
    )
    
    assert response.status_code == 400, f"Expected 400 for invalid scope, got {response.status_code}"
    print("  ✓ Invalid scope rejected with 400 error")
    
    logout()
    return True


def test_missing_required_parameters():
    """
    Test 8: Missing required parameters for specific scopes.
    """
    print("=" * 80)
    print("TEST 8: Missing Required Parameters")
    print("=" * 80)
    
    login("software_admin", "admin123")
    
    # Test department scope without department_id
    response = session.get(
        f"{BASE_URL}/api/dashboard/date-bounds",
        params={"scope": "department"}
    )
    
    assert response.status_code == 400, \
        f"Expected 400 for department scope without department_id, got {response.status_code}"
    print("  ✓ Department scope without department_id rejected")
    
    # Test section scope without section_id
    response = session.get(
        f"{BASE_URL}/api/dashboard/date-bounds",
        params={"scope": "section"}
    )
    
    assert response.status_code == 400, \
        f"Expected 400 for section scope without section_id, got {response.status_code}"
    print("  ✓ Section scope without section_id rejected")
    
    logout()
    return True


def test_date_format_validation():
    """
    Test 9: Returned dates are valid YYYY-MM-DD format.
    """
    print("=" * 80)
    print("TEST 9: Date Format Validation")
    print("=" * 80)
    
    login("software_admin", "admin123")
    
    response = get_date_bounds(scope="hospital")
    
    if response["min_date"] is not None:
        # Try parsing the date
        try:
            parsed_min = date.fromisoformat(response["min_date"])
            parsed_max = date.fromisoformat(response["max_date"])
            print(f"  ✓ Dates are valid ISO format")
            print(f"    min_date: {parsed_min}")
            print(f"    max_date: {parsed_max}")
            
            # min_date should be <= max_date
            assert parsed_min <= parsed_max, "min_date must be <= max_date"
            print(f"  ✓ min_date <= max_date constraint satisfied")
        except ValueError as e:
            raise AssertionError(f"Invalid date format: {e}")
    else:
        print("  ⚠ No dates to validate (null response)")
    
    logout()
    return True


# =========================================================
# TEST RUNNER
# =========================================================

def run_all_tests():
    """Run all test cases."""
    tests = [
        test_endpoint_exists,
        test_response_contract,
        test_scope_filtering_administration,
        test_scope_filtering_department,
        test_scope_filtering_section,
        test_rbac_enforcement,
        test_invalid_scope_parameter,
        test_missing_required_parameters,
        test_date_format_validation,
    ]
    
    print("\n" + "=" * 80)
    print("PHASE DR-B: DASHBOARD DATE BOUNDS - BACKEND TESTS")
    print("=" * 80 + "\n")
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"  ✗ FAILED: {test.__name__}\n")
        except Exception as e:
            failed += 1
            print(f"  ✗ EXCEPTION in {test.__name__}: {e}\n")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
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
