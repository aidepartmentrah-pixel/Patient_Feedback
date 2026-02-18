"""
TEST TASK D-B12 — BACKEND SMOKE TESTS VALIDATION

Verify backend smoke tests meet requirements:
1. Test file exists: test_person_reporting_smoke.py
2. Contains at least 5 smoke tests
3. Tests call real endpoints — not service functions directly
4. Tests accept 200/403/404 but fail on 500
5. No heavy fixtures or DB seeding required
6. Matches existing backend test style
"""

import os
import re
import ast
import sys


def test_file_exists():
    """Test 1: Verify test_person_reporting_smoke.py exists"""
    test_path = "backend/tests/test_person_reporting_smoke.py"
    assert os.path.exists(test_path), f"Smoke test file not found: {test_path}"
    print("✅ Test 1: Smoke test file exists")


def test_contains_five_tests():
    """Test 2: Verify file contains at least 5 smoke tests"""
    test_path = "backend/tests/test_person_reporting_smoke.py"
    with open(test_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Count test functions (def test_...)
    test_functions = re.findall(r'^\s*def (test_\w+)\(', content, re.MULTILINE)
    
    assert len(test_functions) >= 5, \
        f"Expected at least 5 smoke tests, found {len(test_functions)}: {test_functions}"
    
    print(f"✅ Test 2: Contains {len(test_functions)} smoke tests (>= 5 required)")


def test_calls_real_endpoints():
    """Test 3: Verify tests call real endpoints, not service functions"""
    test_path = "backend/tests/test_person_reporting_smoke.py"
    with open(test_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for TestClient usage
    assert "TestClient" in content, \
        "Tests should use FastAPI TestClient for endpoint calls"
    
    # Check for HTTP method calls (client.get, client.post, etc.)
    http_methods = ["client.get(", "authenticated_client.get(", "authenticated_client.post("]
    has_http_calls = any(method in content for method in http_methods)
    assert has_http_calls, \
        "Tests should call HTTP endpoints using client.get() or client.post()"
    
    # Check tests are NOT calling service functions directly
    service_patterns = [
        "DoctorSeasonalReportingService.build",
        "WorkerSeasonalReportingService.build",
        "generate_person_seasonal_word_report("
    ]
    
    for pattern in service_patterns:
        assert pattern not in content, \
            f"Tests should not call service functions directly. Found: {pattern}"
    
    # Count endpoint calls
    endpoint_calls = re.findall(r'client\.get\(|authenticated_client\.get\(|client\.post\(', content)
    assert len(endpoint_calls) >= 5, \
        f"Expected at least 5 endpoint calls, found {len(endpoint_calls)}"
    
    print("✅ Test 3: Tests call real endpoints (not service functions directly)")


def test_accepts_valid_status_codes():
    """Test 4: Verify tests accept 200/403/404 but fail on 500"""
    test_path = "backend/tests/test_person_reporting_smoke.py"
    with open(test_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for status code assertions that accept multiple valid codes
    assert "[200, 403, 404]" in content or "[200, 404, 403]" in content, \
        "Tests should accept status codes 200, 403, or 404"
    
    assert "[401, 403]" in content, \
        "Tests should expect 401 or 403 for unauthorized requests"
    
    # Check for 500 rejection pattern
    # Should have comments or assertions mentioning 500 is a failure
    assert "500" in content, \
        "Tests should document that 500 is a failure condition"
    
    assert "server error" in content.lower() or "wiring problem" in content.lower(), \
        "Tests should document that 500 indicates wiring problems"
    
    # Verify assertions check for valid status codes
    valid_status_checks = re.findall(
        r'status_code in \[([\d, ]+)\]',
        content
    )
    
    assert len(valid_status_checks) >= 3, \
        f"Expected at least 3 status code validations, found {len(valid_status_checks)}"
    
    print("✅ Test 4: Tests accept 200/403/404 but document 500 as failure")


def test_no_heavy_fixtures():
    """Test 5: Verify no heavy fixtures or DB seeding"""
    test_path = "backend/tests/test_person_reporting_smoke.py"
    with open(test_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for heavy fixture patterns that should NOT exist
    heavy_patterns = [
        "CREATE TABLE",
        "INSERT INTO",
        "DELETE FROM",
        "TRUNCATE",
        "seed_database",
        "setup_test_data",
        "create_test_fixtures",
        "@pytest.fixture(scope=\"session\")",
        "BEGIN TRANSACTION",
    ]
    
    for pattern in heavy_patterns:
        assert pattern not in content, \
            f"Smoke tests should not contain heavy fixtures. Found: {pattern}"
    
    # Should have minimal fixtures (just authentication)
    assert "@pytest.fixture" in content, \
        "Should have minimal fixtures (authenticated_client)"
    
    # Count fixtures (should be minimal, like 1-2)
    fixtures = re.findall(r'@pytest\.fixture', content)
    assert len(fixtures) <= 2, \
        f"Should have minimal fixtures (1-2), found {len(fixtures)}"
    
    print("✅ Test 5: No heavy fixtures or DB seeding (lightweight smoke only)")


def test_matches_existing_style():
    """Test 6: Verify matches existing backend test style"""
    test_path = "backend/tests/test_person_reporting_smoke.py"
    with open(test_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for FastAPI TestClient pattern (like test_insight_endpoints.py)
    assert "from fastapi.testclient import TestClient" in content, \
        "Should import FastAPI TestClient"
    
    assert "import main" in content, \
        "Should import main app module"
    
    assert "app = main.app" in content, \
        "Should get app instance from main"
    
    assert "client = TestClient(app)" in content, \
        "Should create TestClient instance"
    
    # Check for authenticated_client fixture pattern
    assert "def authenticated_client()" in content, \
        "Should have authenticated_client fixture"
    
    assert '"/api/auth/login"' in content, \
        "Should use /api/auth/login for authentication"
    
    # Check for session-based auth (not JWT)
    assert "jwt" not in content.lower() or "not jwt" in content.lower(), \
        "Should use session-based auth, not JWT"
    
    # Check for proper docstrings
    docstring_count = content.count('"""')
    assert docstring_count >= 10, \
        f"Tests should have proper docstrings. Found {docstring_count // 2} docstrings"
    
    # Check for TEST numbering in docstrings
    assert "TEST 1" in content and "TEST 2" in content, \
        "Tests should be numbered in docstrings (TEST 1, TEST 2, etc.)"
    
    print("✅ Test 6: Matches existing backend test style (TestClient, fixtures, docstrings)")


def test_phase_d_smoke_comment():
    """Test 7: Verify Phase D smoke coverage comment exists"""
    test_path = "backend/tests/test_person_reporting_smoke.py"
    with open(test_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for required comment
    assert "Phase D smoke coverage — wiring validation only" in content, \
        "Missing required comment: 'Phase D smoke coverage — wiring validation only'"
    
    # Comment should appear multiple times (in module docstring and possibly in tests)
    comment_count = content.count("Phase D smoke coverage")
    assert comment_count >= 2, \
        f"Phase D smoke comment should appear at least 2 times, found {comment_count}"
    
    print("✅ Test 7: Phase D smoke coverage comment present")


def test_endpoint_coverage():
    """Test 8: Verify correct endpoints are tested"""
    test_path = "backend/tests/test_person_reporting_smoke.py"
    with open(test_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for required endpoint paths
    required_endpoints = [
        "/api/person-reports/doctor/",
        "/api/person-reports/worker/",
        "/api/workers/"
    ]
    
    for endpoint in required_endpoints:
        assert endpoint in content, \
            f"Missing endpoint test: {endpoint}"
    
    # Check for date query parameters
    assert "season_start" in content and "season_end" in content, \
        "Tests should include season_start and season_end query parameters"
    
    # Check for proper parameter format
    assert "params=" in content, \
        "Tests should pass query parameters using params= argument"
    
    print("✅ Test 8: Correct endpoints tested (doctor, worker, dates)")


def test_error_messages_informative():
    """Test 9: Verify error messages are informative"""
    test_path = "backend/tests/test_person_reporting_smoke.py"
    with open(test_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for informative assertion messages
    assert "f\"Expected status" in content, \
        "Assertions should have informative f-string messages"
    
    # Check assertions explain what went wrong
    assert "wiring problem" in content.lower(), \
        "Error messages should mention 'wiring problem' for 500 errors"
    
    # Check for context in error messages
    assert "response.status_code" in content, \
        "Error messages should include actual status code received"
    
    # Count assertion messages
    assertion_messages = re.findall(r'assert .+?, \\\s*f"', content)
    assert len(assertion_messages) >= 5, \
        f"Expected at least 5 assertions with messages, found {len(assertion_messages)}"
    
    print("✅ Test 9: Error messages are informative (include status codes and context)")


def test_pytest_integration():
    """Test 10: Verify proper pytest integration"""
    test_path = "backend/tests/test_person_reporting_smoke.py"
    with open(test_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check pytest import
    assert "import pytest" in content, \
        "Should import pytest"
    
    # Check pytest.skip usage for handling missing test data
    assert "pytest.skip" in content, \
        "Should use pytest.skip for graceful test skipping"
    
    # Check pytest.main for direct execution
    assert "pytest.main" in content, \
        "Should support direct execution with pytest.main"
    
    # Check if __name__ == "__main__" pattern
    assert 'if __name__ == "__main__"' in content, \
        "Should have main block for direct execution"
    
    print("✅ Test 10: Proper pytest integration (fixtures, skip, direct execution)")


def run_all_tests():
    """Run all D-B12 backend smoke test validation tests"""
    print("\n" + "="*70)
    print("PHASE D - TASK D-B12: BACKEND SMOKE TESTS VALIDATION")
    print("="*70 + "\n")
    
    tests = [
        test_file_exists,
        test_contains_five_tests,
        test_calls_real_endpoints,
        test_accepts_valid_status_codes,
        test_no_heavy_fixtures,
        test_matches_existing_style,
        test_phase_d_smoke_comment,
        test_endpoint_coverage,
        test_error_messages_informative,
        test_pytest_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: Unexpected error: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ BACKEND SMOKE TESTS OK")
    else:
        print("❌ BACKEND SMOKE TESTS FAILED")
        sys.exit(1)
    
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
