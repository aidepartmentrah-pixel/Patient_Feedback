"""
PHASE C — B-C7 — TEST ROUTER CONTRACT FREEZE

Tests to verify section creation endpoint contract is stable and documented.

Test Coverage:
1. OpenAPI schema validation
2. Example payload from docstring
3. Response field name validation
4. Contract documentation presence
5. Request model validation
6. Response model validation
7. HTTP method and path stability
8. Error response contract stability
"""

import requests
import json
import time
from typing import Dict, Any


# Configuration
BASE_URL = "http://localhost:8000"
LOGIN_URL = f"{BASE_URL}/api/auth/login"
SECTION_ENDPOINT = f"{BASE_URL}/api/admin/create-section-with-admin"
OPENAPI_URL = f"{BASE_URL}/openapi.json"


def login_as_software_admin() -> requests.Session:
    """Login as SOFTWARE_ADMIN and return authenticated session."""
    session = requests.Session()
    response = session.post(
        LOGIN_URL,
        json={"username": "software_admin", "password": "admin123"}
    )
    assert response.status_code == 200, f"Login failed: {response.text}"
    return session


def get_valid_parent_id(session: requests.Session) -> int:
    """Get a valid parent org unit ID for testing."""
    # Use test data endpoint or known valid ID
    # For now, use ID from previous tests (administration unit)
    return 1


def test_1_openapi_schema_validation():
    """
    Test 1: OpenAPI Test
    Verify /openapi.json shows correct path, method, and models.
    """
    print("\n" + "="*80)
    print("TEST 1: OpenAPI Schema Validation")
    print("="*80)
    
    # Fetch OpenAPI schema
    response = requests.get(OPENAPI_URL)
    assert response.status_code == 200, "Failed to fetch OpenAPI schema"
    
    schema = response.json()
    print("✓ OpenAPI schema retrieved")
    
    # Verify endpoint exists in paths
    endpoint_path = "/api/admin/create-section-with-admin"
    assert endpoint_path in schema["paths"], f"Endpoint {endpoint_path} not found in OpenAPI schema"
    print(f"✓ Endpoint path found: {endpoint_path}")
    
    # Verify POST method exists
    endpoint_config = schema["paths"][endpoint_path]
    assert "post" in endpoint_config, "POST method not found for endpoint"
    print("✓ POST method defined")
    
    post_config = endpoint_config["post"]
    
    # Verify request body uses SectionCreateRequest schema
    assert "requestBody" in post_config, "Request body not defined"
    request_schema_ref = post_config["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    assert "SectionCreateRequest" in request_schema_ref, "Request model is not SectionCreateRequest"
    print(f"✓ Request model: SectionCreateRequest")
    
    # Verify response uses SectionCreateResponse schema
    assert "200" in post_config["responses"], "200 response not defined"
    response_schema_ref = post_config["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    assert "SectionCreateResponse" in response_schema_ref, "Response model is not SectionCreateResponse"
    print(f"✓ Response model: SectionCreateResponse")
    
    # Verify schema definitions exist
    components = schema.get("components", {}).get("schemas", {})
    assert "SectionCreateRequest" in components, "SectionCreateRequest schema not in components"
    assert "SectionCreateResponse" in components, "SectionCreateResponse schema not in components"
    print("✓ Schema components defined in OpenAPI")
    
    # Verify SectionCreateRequest fields
    request_schema = components["SectionCreateRequest"]
    request_props = request_schema["properties"]
    assert "section_name" in request_props, "section_name missing from request schema"
    assert "parent_unit_id" in request_props, "parent_unit_id missing from request schema"
    print("✓ Request schema has required fields: section_name, parent_unit_id")
    
    # Verify SectionCreateResponse fields
    response_schema = components["SectionCreateResponse"]
    response_props = response_schema["properties"]
    expected_fields = ["section_id", "section_name", "parent_unit_id", "username", "temp_password"]
    for field in expected_fields:
        assert field in response_props, f"{field} missing from response schema"
    print(f"✓ Response schema has all required fields: {', '.join(expected_fields)}")
    
    print("\n✅ TEST 1 PASSED: OpenAPI schema validation successful")
    return True


def test_2_example_payload_from_docstring():
    """
    Test 2: Example Payload Test
    Use exact example from docstring and verify success.
    """
    print("\n" + "="*80)
    print("TEST 2: Example Payload from Docstring")
    print("="*80)
    
    # Login as software_admin
    session = login_as_software_admin()
    print("✓ Logged in as software_admin")
    
    # Get valid parent ID
    parent_id = get_valid_parent_id(session)
    print(f"✓ Using parent_unit_id: {parent_id}")
    
    # Use example payload from docstring (with unique name to avoid duplicates)
    import random
    unique_suffix = random.randint(10000, 99999)
    payload = {
        "section_name": f"Emergency Department Section A {unique_suffix}",
        "parent_unit_id": parent_id
    }
    
    print(f"✓ Payload: {json.dumps(payload, indent=2)}")
    
    # Call endpoint
    response = session.post(SECTION_ENDPOINT, json=payload)
    
    print(f"✓ Response status: {response.status_code}")
    
    # Verify success
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    
    # Verify response structure
    result = response.json()
    print(f"✓ Response: {json.dumps(result, indent=2)}")
    
    # Verify all expected fields present
    assert "section_id" in result, "section_id missing from response"
    assert "section_name" in result, "section_name missing from response"
    assert "parent_unit_id" in result, "parent_unit_id missing from response"
    assert "username" in result, "username missing from response"
    assert "temp_password" in result, "temp_password missing from response"
    print("✓ All response fields present")
    
    # Verify values match request
    assert result["section_name"] == payload["section_name"], "section_name mismatch"
    assert result["parent_unit_id"] == payload["parent_unit_id"], "parent_unit_id mismatch"
    print("✓ Response values match request")
    
    print("\n✅ TEST 2 PASSED: Example payload works successfully")
    return True


def test_3_response_field_names_exact_match():
    """
    Test 3: Field Name Test
    Verify response field names exactly match SectionCreateResponse schema.
    """
    print("\n" + "="*80)
    print("TEST 3: Response Field Names Exact Match")
    print("="*80)
    
    # Login
    session = login_as_software_admin()
    parent_id = get_valid_parent_id(session)
    
    # Create section
    import random
    payload = {
        "section_name": f"Test Section {random.randint(10000, 99999)}",
        "parent_unit_id": parent_id
    }
    
    response = session.post(SECTION_ENDPOINT, json=payload)
    assert response.status_code == 200, f"Request failed: {response.text}"
    
    result = response.json()
    print(f"✓ Response received: {json.dumps(result, indent=2)}")
    
    # Define expected field names from schema
    expected_fields = {
        "section_id",
        "section_name",
        "parent_unit_id",
        "username",
        "temp_password"
    }
    
    # Get actual field names
    actual_fields = set(result.keys())
    
    print(f"✓ Expected fields: {sorted(expected_fields)}")
    print(f"✓ Actual fields: {sorted(actual_fields)}")
    
    # Verify exact match
    missing_fields = expected_fields - actual_fields
    extra_fields = actual_fields - expected_fields
    
    assert len(missing_fields) == 0, f"Missing fields: {missing_fields}"
    assert len(extra_fields) == 0, f"Extra fields: {extra_fields}"
    assert expected_fields == actual_fields, "Field names do not match schema exactly"
    
    print("✓ Field names match schema exactly")
    
    # Verify field types
    assert isinstance(result["section_id"], int), "section_id should be int"
    assert isinstance(result["section_name"], str), "section_name should be str"
    assert isinstance(result["parent_unit_id"], int), "parent_unit_id should be int"
    assert isinstance(result["username"], str), "username should be str"
    assert isinstance(result["temp_password"], str), "temp_password should be str"
    print("✓ Field types are correct")
    
    print("\n✅ TEST 3 PASSED: Response field names match schema exactly")
    return True


def test_4_contract_documentation_present():
    """
    Test 4: Contract Documentation Check
    Verify endpoint has comprehensive contract docstring.
    """
    print("\n" + "="*80)
    print("TEST 4: Contract Documentation Present")
    print("="*80)
    
    # Read router file
    router_file = r"C:\Users\IT\Documents\GitHub Repository\Patient_Feedback\backend\api\routers\admin_section_router.py"
    
    with open(router_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Verify CONTRACT FROZEN comment present
    assert "CONTRACT FROZEN" in content, "CONTRACT FROZEN comment not found"
    print("✓ CONTRACT FROZEN comment present")
    
    # Verify contract specifications in docstring
    required_sections = [
        "CONTRACT SPECIFICATION",
        "Purpose:",
        "HTTP Method:",
        "Path:",
        "Request Model:",
        "Response Model:",
        "Request JSON Example:",
        "Response JSON Example:",
        "Field Constraints:",
        "Behavior:",
        "Authorization:",
        "Error Scenarios:"
    ]
    
    for section in required_sections:
        assert section in content, f"Missing contract section: {section}"
        print(f"✓ Contract section found: {section}")
    
    # Verify response_model in decorator
    assert 'response_model=SectionCreateResponse' in content, "response_model not set in decorator"
    print("✓ response_model=SectionCreateResponse in decorator")
    
    # Verify request model in signature
    assert 'request: SectionCreateRequest' in content, "request model not in signature"
    print("✓ request: SectionCreateRequest in signature")
    
    # Verify warning about not modifying contract
    assert "DO NOT modify" in content or "Do not change" in content, "Contract warning not present"
    print("✓ Contract modification warning present")
    
    print("\n✅ TEST 4 PASSED: Contract documentation is comprehensive")
    return True


def test_5_request_validation_enforced():
    """
    Test 5: Request Validation Test
    Verify request model validation is enforced (422 on invalid input).
    """
    print("\n" + "="*80)
    print("TEST 5: Request Validation Enforced")
    print("="*80)
    
    session = login_as_software_admin()
    parent_id = get_valid_parent_id(session)
    
    # Test 5a: Missing section_name
    print("\nTest 5a: Missing section_name field")
    response = session.post(SECTION_ENDPOINT, json={"parent_unit_id": parent_id})
    assert response.status_code == 422, f"Expected 422 for missing field, got {response.status_code}"
    print("✓ Missing section_name returns 422")
    
    # Test 5b: Empty section_name
    print("\nTest 5b: Empty section_name")
    response = session.post(SECTION_ENDPOINT, json={"section_name": "", "parent_unit_id": parent_id})
    assert response.status_code == 422, f"Expected 422 for empty name, got {response.status_code}"
    print("✓ Empty section_name returns 422")
    
    # Test 5c: section_name too short
    print("\nTest 5c: section_name too short (1 char)")
    response = session.post(SECTION_ENDPOINT, json={"section_name": "A", "parent_unit_id": parent_id})
    assert response.status_code == 422, f"Expected 422 for short name, got {response.status_code}"
    print("✓ Short section_name returns 422")
    
    # Test 5d: Missing parent_unit_id
    print("\nTest 5d: Missing parent_unit_id")
    response = session.post(SECTION_ENDPOINT, json={"section_name": "Test Section"})
    assert response.status_code == 422, f"Expected 422 for missing parent, got {response.status_code}"
    print("✓ Missing parent_unit_id returns 422")
    
    # Test 5e: Invalid parent_unit_id type
    print("\nTest 5e: Invalid parent_unit_id type (string)")
    response = session.post(SECTION_ENDPOINT, json={"section_name": "Test", "parent_unit_id": "invalid"})
    assert response.status_code == 422, f"Expected 422 for invalid type, got {response.status_code}"
    print("✓ Invalid parent_unit_id type returns 422")
    
    # Test 5f: Negative parent_unit_id
    print("\nTest 5f: Negative parent_unit_id")
    response = session.post(SECTION_ENDPOINT, json={"section_name": "Test", "parent_unit_id": -1})
    assert response.status_code == 422, f"Expected 422 for negative ID, got {response.status_code}"
    print("✓ Negative parent_unit_id returns 422")
    
    print("\n✅ TEST 5 PASSED: Request validation enforced correctly")
    return True


def test_6_http_method_and_path_stability():
    """
    Test 6: HTTP Method and Path Stability
    Verify endpoint only responds to POST and exact path.
    """
    print("\n" + "="*80)
    print("TEST 6: HTTP Method and Path Stability")
    print("="*80)
    
    session = login_as_software_admin()
    parent_id = get_valid_parent_id(session)
    
    valid_payload = {
        "section_name": "Test Section",
        "parent_unit_id": parent_id
    }
    
    # Test 6a: GET should not work
    print("\nTest 6a: GET method not allowed")
    response = session.get(SECTION_ENDPOINT)
    assert response.status_code == 405, f"GET should return 405, got {response.status_code}"
    print("✓ GET returns 405 Method Not Allowed")
    
    # Test 6b: PUT should not work
    print("\nTest 6b: PUT method not allowed")
    response = session.put(SECTION_ENDPOINT, json=valid_payload)
    assert response.status_code == 405, f"PUT should return 405, got {response.status_code}"
    print("✓ PUT returns 405 Method Not Allowed")
    
    # Test 6c: DELETE should not work
    print("\nTest 6c: DELETE method not allowed")
    response = session.delete(SECTION_ENDPOINT)
    assert response.status_code == 405, f"DELETE should return 405, got {response.status_code}"
    print("✓ DELETE returns 405 Method Not Allowed")
    
    # Test 6d: PATCH should not work
    print("\nTest 6d: PATCH method not allowed")
    response = session.patch(SECTION_ENDPOINT, json=valid_payload)
    assert response.status_code == 405, f"PATCH should return 405, got {response.status_code}"
    print("✓ PATCH returns 405 Method Not Allowed")
    
    # Test 6e: Exact path required (trailing slash redirects or works)
    print("\nTest 6e: Exact path stability (FastAPI handles trailing slash)")
    response = session.post(f"{SECTION_ENDPOINT}/", json=valid_payload)
    # FastAPI by default redirects trailing slashes, so 307 (redirect) or 200 (direct) are acceptable
    assert response.status_code in [200, 307, 308], f"Expected 200/307/308, got {response.status_code}"
    print(f"✓ Trailing slash handled correctly: {response.status_code}")
    
    # Test 6f: POST to correct path works
    print("\nTest 6f: POST to exact path succeeds")
    import random
    valid_payload["section_name"] = f"Test Section {random.randint(10000, 99999)}"
    response = session.post(SECTION_ENDPOINT, json=valid_payload)
    assert response.status_code == 200, f"POST should succeed, got {response.status_code}"
    print("✓ POST to exact path returns 200")
    
    print("\n✅ TEST 6 PASSED: Only POST method and exact path work")
    return True


def test_7_error_response_contract():
    """
    Test 7: Error Response Contract Stability
    Verify error responses follow standard FastAPI format.
    """
    print("\n" + "="*80)
    print("TEST 7: Error Response Contract Stability")
    print("="*80)
    
    # Test 7a: 401 Unauthorized format
    print("\nTest 7a: 401 Unauthorized response format")
    unauthenticated_session = requests.Session()
    response = unauthenticated_session.post(
        SECTION_ENDPOINT,
        json={"section_name": "Test", "parent_unit_id": 1}
    )
    assert response.status_code == 401, f"Expected 401, got {response.status_code}"
    error = response.json()
    assert "detail" in error, "Error response should have 'detail' field"
    print(f"✓ 401 response: {error}")
    
    # Test 7b: 422 Validation Error format
    print("\nTest 7b: 422 Validation Error response format")
    session = login_as_software_admin()
    response = session.post(SECTION_ENDPOINT, json={"section_name": ""})
    assert response.status_code == 422, f"Expected 422, got {response.status_code}"
    error = response.json()
    assert "detail" in error, "Validation error should have 'detail' field"
    print(f"✓ 422 response has detail field")
    
    # Test 7c: 403 Forbidden format (non-SOFTWARE_ADMIN user)
    print("\nTest 7c: 403 Forbidden response format")
    worker_session = requests.Session()
    login_response = worker_session.post(
        LOGIN_URL,
        json={"username": "worker", "password": "worker123"}
    )
    if login_response.status_code == 200:
        response = worker_session.post(
            SECTION_ENDPOINT,
            json={"section_name": "Test", "parent_unit_id": 1}
        )
        assert response.status_code == 403, f"Expected 403, got {response.status_code}"
        error = response.json()
        assert "detail" in error, "Forbidden error should have 'detail' field"
        print(f"✓ 403 response: {error}")
    else:
        print("⚠ Skipping 403 test (worker login failed)")
    
    print("\n✅ TEST 7 PASSED: Error responses follow standard format")
    return True


def test_8_contract_freeze_signature_stability():
    """
    Test 8: Signature Drift Check
    Verify function signature matches contract (no drift after restart).
    """
    print("\n" + "="*80)
    print("TEST 8: Contract Freeze Signature Stability")
    print("="*80)
    
    # Read router file and verify signature
    router_file = r"C:\Users\IT\Documents\GitHub Repository\Patient_Feedback\backend\api\routers\admin_section_router.py"
    
    with open(router_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Verify function signature components
    signature_checks = [
        ('def create_section_with_admin_endpoint', 'Endpoint function name'),
        ('request: SectionCreateRequest', 'Request parameter type'),
        ('admin: CurrentUser = Depends(get_current_software_admin)', 'Admin dependency'),
        ('-> Dict[str, Any]', 'Return type annotation'),
        ('@router.post("/create-section-with-admin"', 'Router decorator path'),
        ('response_model=SectionCreateResponse', 'Response model in decorator')
    ]
    
    for pattern, description in signature_checks:
        assert pattern in content, f"Signature component missing: {description}"
        print(f"✓ {description}: {pattern}")
    
    # Verify no extra parameters in signature
    # Extract function definition
    import re
    func_match = re.search(
        r'def create_section_with_admin_endpoint\((.*?)\)',
        content,
        re.DOTALL
    )
    assert func_match, "Could not find function definition"
    
    params = func_match.group(1)
    param_count = len([p for p in params.split(',') if p.strip()])
    
    assert param_count == 2, f"Expected 2 parameters (request, admin), found {param_count}"
    print(f"✓ Function has exactly 2 parameters (no drift)")
    
    # Verify imports are correct
    required_imports = [
        'from ..schemas.section_creation_schemas import',
        'SectionCreateRequest',
        'SectionCreateResponse',
        'from ..utils.guards import get_current_software_admin'
    ]
    
    for imp in required_imports:
        assert imp in content, f"Missing required import: {imp}"
        print(f"✓ Required import present: {imp}")
    
    print("\n✅ TEST 8 PASSED: Signature is stable and matches contract")
    return True


def run_all_tests():
    """Run all B-C7 contract freeze tests."""
    print("\n" + "="*80)
    print("PHASE C — B-C7 — ROUTER CONTRACT FREEZE TEST SUITE")
    print("="*80)
    print("\nStarting in 2 seconds to allow server startup...")
    time.sleep(2)
    
    tests = [
        ("OpenAPI Schema Validation", test_1_openapi_schema_validation),
        ("Example Payload from Docstring", test_2_example_payload_from_docstring),
        ("Response Field Names Exact Match", test_3_response_field_names_exact_match),
        ("Contract Documentation Present", test_4_contract_documentation_present),
        ("Request Validation Enforced", test_5_request_validation_enforced),
        ("HTTP Method and Path Stability", test_6_http_method_and_path_stability),
        ("Error Response Contract", test_7_error_response_contract),
        ("Contract Freeze Signature Stability", test_8_contract_freeze_signature_stability)
    ]
    
    results = []
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            results.append((test_name, "PASSED", None))
            passed += 1
        except AssertionError as e:
            results.append((test_name, "FAILED", str(e)))
            failed += 1
        except Exception as e:
            results.append((test_name, "ERROR", str(e)))
            failed += 1
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, status, error in results:
        status_symbol = "✅" if status == "PASSED" else "❌"
        print(f"{status_symbol} {test_name}: {status}")
        if error:
            print(f"   Error: {error}")
    
    print(f"\nTotal: {len(tests)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed/len(tests)*100):.1f}%")
    
    if failed == 0:
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED — CONTRACT FREEZE VALIDATED")
        print("="*80)
        return True
    else:
        print("\n" + "="*80)
        print(f"❌ {failed} TEST(S) FAILED")
        print("="*80)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
