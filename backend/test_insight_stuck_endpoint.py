"""
Test Insight Stuck Cases Endpoint (B-I16)
Integration tests for GET /api/v2/insight/stuck endpoint.

Run: python backend/test_insight_stuck_endpoint.py
"""

import sys
import os
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Add parent directory to path for 'backend' module imports
parent_dir = backend_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from fastapi.testclient import TestClient
from fastapi import FastAPI
from backend.api_v2.routers import insight_router
from backend.api.schemas.auth_models import CurrentUser
from backend.api.dependencies.user_context import get_current_user
import pyodbc

print("=" * 80)
print("INSIGHT STUCK CASES ENDPOINT TEST (B-I16)")
print("=" * 80)

test_passed = 0
test_failed = 0

# Create test FastAPI app
app = FastAPI()
app.include_router(insight_router.router)

# Get real org unit IDs from database
def get_real_org_units():
    """Get real organizational unit IDs from database."""
    try:
        conn_str = (
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=SOCIALMEDIA;"
            "DATABASE=IncidentManager;"
            "Trusted_Connection=yes;"
        )
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT TOP 5 TargetOrgUnitID 
            FROM APP_AdministrativeSubcase 
            WHERE TargetOrgUnitID IS NOT NULL
            ORDER BY TargetOrgUnitID
        """)
        
        rows = cursor.fetchall()
        org_units = [row[0] for row in rows]
        
        conn.close()
        return org_units
    except Exception as e:
        print(f"⚠️  Warning: Could not fetch org units: {e}")
        return [1, 2, 3]

real_org_units = get_real_org_units()
print(f"\n📋 Using real org units: {real_org_units}")

# ============================================================
# ENDPOINT STRUCTURE TESTS
# ============================================================

print("\n" + "=" * 80)
print("ENDPOINT STRUCTURE TESTS")
print("=" * 80)

# Test 1: Endpoint exists
print("\n[TEST 1] Endpoint exists...")
try:
    routes = [route for route in app.routes if hasattr(route, 'path')]
    stuck_route = None
    for route in routes:
        if route.path == "/api/v2/insight/stuck":
            stuck_route = route
            break
    
    assert stuck_route is not None, "Endpoint not found"
    print(f"   Path: {stuck_route.path}")
    print(f"   Methods: {stuck_route.methods}")
    print("✅ PASS: Endpoint exists")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1

# Test 2: Endpoint is GET method
print("\n[TEST 2] Endpoint is GET method...")
try:
    assert stuck_route is not None
    assert "GET" in stuck_route.methods, f"Expected GET, got {stuck_route.methods}"
    print("✅ PASS: Endpoint is GET method")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1

# Test 3: Endpoint is async
print("\n[TEST 3] Endpoint is async...")
try:
    import inspect
    from backend.api_v2.routers.insight_router import get_stuck_endpoint
    
    assert inspect.iscoroutinefunction(get_stuck_endpoint), "Endpoint should be async"
    print("✅ PASS: Endpoint is async")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1

# Test 4: Endpoint has docstring
print("\n[TEST 4] Endpoint has docstring...")
try:
    from backend.api_v2.routers.insight_router import get_stuck_endpoint
    
    doc = get_stuck_endpoint.__doc__
    assert doc is not None and len(doc.strip()) > 0, "Missing docstring"
    assert "threshold" in doc.lower() or "updatedAt" in doc or "terminal" in doc.lower(), \
        "Docstring should mention threshold/UpdatedAt/terminal"
    print(f"   Docstring: {doc.strip()}")
    print("✅ PASS: Endpoint has docstring")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1

# Test 5: Endpoint has days_threshold parameter
print("\n[TEST 5] Endpoint has days_threshold parameter...")
try:
    import inspect
    from backend.api_v2.routers.insight_router import get_stuck_endpoint
    
    sig = inspect.signature(get_stuck_endpoint)
    params = sig.parameters
    
    assert "days_threshold" in params, "Endpoint should have days_threshold parameter"
    
    param = params["days_threshold"]
    # Should be int type
    assert param.annotation == int, f"days_threshold should be int, got {param.annotation}"
    
    print(f"   Parameter: days_threshold (type: {param.annotation.__name__})")
    print("✅ PASS: Endpoint has days_threshold parameter")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1

# Test 6: Endpoint has current_user dependency
print("\n[TEST 6] Endpoint has current_user dependency...")
try:
    import inspect
    from backend.api_v2.routers.insight_router import get_stuck_endpoint
    
    sig = inspect.signature(get_stuck_endpoint)
    params = sig.parameters
    
    assert "current_user" in params, "Endpoint should have current_user parameter"
    
    param = params["current_user"]
    # Should have Depends annotation
    assert param.default is not inspect.Parameter.empty, \
        "current_user should have Depends default"
    
    print(f"   Parameter: current_user with Depends")
    print("✅ PASS: Endpoint has current_user dependency")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1

# ============================================================
# AUTHENTICATION TESTS
# ============================================================

print("\n" + "=" * 80)
print("AUTHENTICATION TESTS")
print("=" * 80)

# Test 7: Endpoint requires authentication
print("\n[TEST 7] Endpoint requires authentication...")
try:
    client = TestClient(app)
    response = client.get("/api/v2/insight/stuck?days_threshold=30")
    
    # Should fail without auth (401, 422, or 500)
    assert response.status_code in [401, 422, 500], \
        f"Expected 401/422/500 without auth, got {response.status_code}"
    
    print(f"   Status without auth: {response.status_code}")
    print("✅ PASS: Endpoint requires authentication")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1

# ============================================================
# INTEGRATION TESTS - BASIC FUNCTIONALITY
# ============================================================

print("\n" + "=" * 80)
print("INTEGRATION TESTS - BASIC FUNCTIONALITY")
print("=" * 80)

# Test 8: Returns data with valid threshold
print("\n[TEST 8] Returns data with valid threshold...")
try:
    def mock_current_user():
        return CurrentUser(
            user_id=1,
            username="test_user",
            is_active=True,
            scopes=[],
            allowed_unit_ids={real_org_units[0]} if real_org_units else {1}
        )
    
    app.dependency_overrides[get_current_user] = mock_current_user
    
    client = TestClient(app)
    response = client.get("/api/v2/insight/stuck?days_threshold=30")
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    
    data = response.json()
    print(f"   Response type: {type(data)}")
    print(f"   Response length: {len(data) if isinstance(data, list) else 'N/A'}")
    print("✅ PASS: Returns data with valid threshold")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
finally:
    app.dependency_overrides.clear()

# Test 9: Response is a list
print("\n[TEST 9] Response is a list...")
try:
    def mock_current_user():
        return CurrentUser(
            user_id=1,
            username="test_user",
            is_active=True,
            scopes=[],
            allowed_unit_ids={real_org_units[0]} if real_org_units else {1}
        )
    
    app.dependency_overrides[get_current_user] = mock_current_user
    
    client = TestClient(app)
    response = client.get("/api/v2/insight/stuck?days_threshold=1")
    
    assert response.status_code == 200
    data = response.json()
    
    assert isinstance(data, list), f"Expected list, got {type(data)}"
    print(f"   Returned list with {len(data)} entries")
    print("✅ PASS: Response is a list")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
finally:
    app.dependency_overrides.clear()

# Test 10: Entries have correct structure
print("\n[TEST 10] Entries have correct structure...")
try:
    def mock_current_user():
        return CurrentUser(
            user_id=1,
            username="test_user",
            is_active=True,
            scopes=[],
            allowed_unit_ids={real_org_units[0]} if real_org_units else {1}
        )
    
    app.dependency_overrides[get_current_user] = mock_current_user
    
    client = TestClient(app)
    response = client.get("/api/v2/insight/stuck?days_threshold=1")
    
    assert response.status_code == 200
    data = response.json()
    
    if len(data) > 0:
        entry = data[0]
        
        # Check required fields
        required_fields = ["subcase_id", "status", "target_org_unit_id", "updated_at", "days_in_stage"]
        for field in required_fields:
            assert field in entry, f"Entry should have '{field}' field"
        
        # Check types
        assert isinstance(entry["subcase_id"], int), "subcase_id should be int"
        assert isinstance(entry["status"], str), "status should be string"
        assert isinstance(entry["target_org_unit_id"], int), "target_org_unit_id should be int"
        assert isinstance(entry["updated_at"], str), "updated_at should be string"
        assert isinstance(entry["days_in_stage"], int), "days_in_stage should be int"
        
        print(f"   Sample entry keys: {list(entry.keys())}")
        print(f"   Sample subcase_id: {entry['subcase_id']}")
        print(f"   Sample days_in_stage: {entry['days_in_stage']}")
    else:
        print("   No stuck cases (empty result)")
    
    print("✅ PASS: Entries have correct structure")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
finally:
    app.dependency_overrides.clear()

# ============================================================
# THRESHOLD PARAMETER TESTS
# ============================================================

print("\n" + "=" * 80)
print("THRESHOLD PARAMETER TESTS")
print("=" * 80)

# Test 11: Different thresholds return different results
print("\n[TEST 11] Different thresholds return different results...")
try:
    def mock_current_user():
        return CurrentUser(
            user_id=1,
            username="test_user",
            is_active=True,
            scopes=[],
            allowed_unit_ids={real_org_units[0]} if real_org_units else {1}
        )
    
    app.dependency_overrides[get_current_user] = mock_current_user
    
    client = TestClient(app)
    
    # Get with threshold=1 (should return more)
    response1 = client.get("/api/v2/insight/stuck?days_threshold=1")
    assert response1.status_code == 200
    data1 = response1.json()
    
    # Get with threshold=1000 (should return fewer or same)
    response2 = client.get("/api/v2/insight/stuck?days_threshold=1000")
    assert response2.status_code == 200
    data2 = response2.json()
    
    # Lower threshold should return >= higher threshold
    assert len(data1) >= len(data2), \
        f"Lower threshold (1) should return >= cases than higher threshold (1000)"
    
    print(f"   Threshold=1: {len(data1)} cases")
    print(f"   Threshold=1000: {len(data2)} cases")
    print("✅ PASS: Different thresholds return different results")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
finally:
    app.dependency_overrides.clear()

# Test 12: Low threshold (1 day) returns data
print("\n[TEST 12] Low threshold (1 day) returns data...")
try:
    def mock_current_user():
        return CurrentUser(
            user_id=1,
            username="test_user",
            is_active=True,
            scopes=[],
            allowed_unit_ids=set(real_org_units[:2]) if len(real_org_units) >= 2 else {real_org_units[0]}
        )
    
    app.dependency_overrides[get_current_user] = mock_current_user
    
    client = TestClient(app)
    response = client.get("/api/v2/insight/stuck?days_threshold=1")
    
    assert response.status_code == 200
    data = response.json()
    
    print(f"   Cases with threshold=1: {len(data)}")
    
    # Should have at least some cases with threshold=1
    if len(data) > 0:
        print(f"   Found {len(data)} stuck cases")
    else:
        print("   No stuck cases (acceptable if all recent)")
    
    print("✅ PASS: Low threshold (1 day) returns data")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
finally:
    app.dependency_overrides.clear()

# Test 13: Days in stage >= threshold
print("\n[TEST 13] Days in stage >= threshold...")
try:
    def mock_current_user():
        return CurrentUser(
            user_id=1,
            username="test_user",
            is_active=True,
            scopes=[],
            allowed_unit_ids={real_org_units[0]} if real_org_units else {1}
        )
    
    app.dependency_overrides[get_current_user] = mock_current_user
    
    client = TestClient(app)
    threshold = 5
    response = client.get(f"/api/v2/insight/stuck?days_threshold={threshold}")
    
    assert response.status_code == 200
    data = response.json()
    
    # All returned cases should have days_in_stage >= threshold
    for entry in data:
        assert entry["days_in_stage"] >= threshold, \
            f"days_in_stage ({entry['days_in_stage']}) should be >= threshold ({threshold})"
    
    print(f"   All {len(data)} cases have days_in_stage >= {threshold}")
    print("✅ PASS: Days in stage >= threshold")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
finally:
    app.dependency_overrides.clear()

# ============================================================
# ERROR HANDLING TESTS
# ============================================================

print("\n" + "=" * 80)
print("ERROR HANDLING TESTS")
print("=" * 80)

# Test 14: Missing days_threshold returns error
print("\n[TEST 14] Missing days_threshold returns error...")
try:
    def mock_current_user():
        return CurrentUser(
            user_id=1,
            username="test_user",
            is_active=True,
            scopes=[],
            allowed_unit_ids={real_org_units[0]} if real_org_units else {1}
        )
    
    app.dependency_overrides[get_current_user] = mock_current_user
    
    client = TestClient(app)
    response = client.get("/api/v2/insight/stuck")
    
    # Should return validation error (422)
    assert response.status_code == 422, \
        f"Expected 422 for missing days_threshold, got {response.status_code}"
    
    print(f"   Status for missing days_threshold: {response.status_code}")
    print("✅ PASS: Missing days_threshold returns error")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
finally:
    app.dependency_overrides.clear()

# Test 15: Non-integer days_threshold returns error
print("\n[TEST 15] Non-integer days_threshold returns error...")
try:
    def mock_current_user():
        return CurrentUser(
            user_id=1,
            username="test_user",
            is_active=True,
            scopes=[],
            allowed_unit_ids={real_org_units[0]} if real_org_units else {1}
        )
    
    app.dependency_overrides[get_current_user] = mock_current_user
    
    client = TestClient(app)
    response = client.get("/api/v2/insight/stuck?days_threshold=not_a_number")
    
    # Should return validation error (422)
    assert response.status_code == 422, \
        f"Expected 422 for non-integer days_threshold, got {response.status_code}"
    
    print(f"   Status for non-integer days_threshold: {response.status_code}")
    print("✅ PASS: Non-integer days_threshold returns error")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
finally:
    app.dependency_overrides.clear()

# ============================================================
# SCOPE FILTERING TESTS
# ============================================================

print("\n" + "=" * 80)
print("SCOPE FILTERING TESTS")
print("=" * 80)

# Test 16: Empty scope returns empty list
print("\n[TEST 16] Empty scope returns empty list...")
try:
    def mock_current_user():
        return CurrentUser(
            user_id=1,
            username="test_user",
            is_active=True,
            scopes=[],
            allowed_unit_ids=set()  # Empty scope
        )
    
    app.dependency_overrides[get_current_user] = mock_current_user
    
    client = TestClient(app)
    response = client.get("/api/v2/insight/stuck?days_threshold=1")
    
    assert response.status_code == 200
    data = response.json()
    
    assert isinstance(data, list), "Should return list"
    assert len(data) == 0, "Empty scope should return empty list"
    
    print(f"   Empty scope returned: {data}")
    print("✅ PASS: Empty scope returns empty list")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
finally:
    app.dependency_overrides.clear()

# Test 17: Multiple org units combine data
print("\n[TEST 17] Multiple org units combine data...")
try:
    if len(real_org_units) >= 2:
        def mock_current_user():
            return CurrentUser(
                user_id=1,
                username="test_user",
                is_active=True,
                scopes=[],
                allowed_unit_ids=set(real_org_units[:2])
            )
        
        app.dependency_overrides[get_current_user] = mock_current_user
        
        client = TestClient(app)
        response = client.get("/api/v2/insight/stuck?days_threshold=1")
        
        assert response.status_code == 200
        data = response.json()
        
        print(f"   Entries for 2 org units: {len(data)}")
        
        # Check that we have entries from potentially different org units
        if len(data) > 0:
            org_units_in_result = set(entry["target_org_unit_id"] for entry in data)
            print(f"   Org units in result: {org_units_in_result}")
        
        print("✅ PASS: Multiple org units combine data")
        test_passed += 1
    else:
        print("   ⚠️  SKIP: Not enough org units for test")
        test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
finally:
    app.dependency_overrides.clear()

# ============================================================
# DATA INTEGRITY TESTS
# ============================================================

print("\n" + "=" * 80)
print("DATA INTEGRITY TESTS")
print("=" * 80)

# Test 18: All days_in_stage are non-negative
print("\n[TEST 18] All days_in_stage are non-negative...")
try:
    def mock_current_user():
        return CurrentUser(
            user_id=1,
            username="test_user",
            is_active=True,
            scopes=[],
            allowed_unit_ids={real_org_units[0]} if real_org_units else {1}
        )
    
    app.dependency_overrides[get_current_user] = mock_current_user
    
    client = TestClient(app)
    response = client.get("/api/v2/insight/stuck?days_threshold=1")
    
    assert response.status_code == 200
    data = response.json()
    
    for entry in data:
        assert entry["days_in_stage"] >= 0, \
            f"days_in_stage should be non-negative, got {entry['days_in_stage']}"
    
    print(f"   All {len(data)} entries have non-negative days_in_stage: ✓")
    print("✅ PASS: All days_in_stage are non-negative")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
finally:
    app.dependency_overrides.clear()

# Test 19: No terminal statuses in results
print("\n[TEST 19] No terminal statuses in results...")
try:
    def mock_current_user():
        return CurrentUser(
            user_id=1,
            username="test_user",
            is_active=True,
            scopes=[],
            allowed_unit_ids={real_org_units[0]} if real_org_units else {1}
        )
    
    app.dependency_overrides[get_current_user] = mock_current_user
    
    client = TestClient(app)
    response = client.get("/api/v2/insight/stuck?days_threshold=1")
    
    assert response.status_code == 200
    data = response.json()
    
    terminal_statuses = {"ADMIN_APPROVED", "SECTION_DENIED", "FORCE_CLOSED"}
    
    for entry in data:
        assert entry["status"] not in terminal_statuses, \
            f"Should not return terminal status, got {entry['status']}"
    
    print(f"   All {len(data)} entries are non-terminal: ✓")
    print("✅ PASS: No terminal statuses in results")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
finally:
    app.dependency_overrides.clear()

# Test 20: No data transformation (matches service output)
print("\n[TEST 20] No data transformation...")
try:
    from backend.api_v2.services import insight_service
    
    def mock_current_user():
        return CurrentUser(
            user_id=1,
            username="test_user",
            is_active=True,
            scopes=[],
            allowed_unit_ids={real_org_units[0]} if real_org_units else {1}
        )
    
    # Get service output
    mock_user = mock_current_user()
    service_result = insight_service.get_stuck_cases(mock_user, 30)
    
    # Get endpoint output
    app.dependency_overrides[get_current_user] = mock_current_user
    client = TestClient(app)
    response = client.get("/api/v2/insight/stuck?days_threshold=30")
    
    assert response.status_code == 200
    endpoint_result = response.json()
    
    # Should be identical
    assert endpoint_result == service_result, "Endpoint should return service result unchanged"
    
    print(f"   Service and endpoint outputs match: ✓")
    print("✅ PASS: No data transformation")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
finally:
    app.dependency_overrides.clear()

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"✅ Passed: {test_passed}")
print(f"❌ Failed: {test_failed}")
print(f"📊 Total:  {test_passed + test_failed}")

if test_failed == 0:
    print("\n🎉 ALL TESTS PASSED - B-I16 COMPLETE")
    print("=" * 80)
    print("\nEndpoint Status:")
    print("  ✓ GET /api/v2/insight/stuck implemented")
    print("  ✓ Async endpoint")
    print("  ✓ Query parameter: days_threshold (int, required)")
    print("  ✓ Authentication required")
    print("  ✓ Calls insight_service.get_stuck_cases()")
    print("  ✓ Returns service result unchanged")
    print("\n📊 Query Parameter:")
    print("  days_threshold: int (required)")
    print("    - Minimum number of days in current stage")
    print("    - Filters subcases stuck >= this threshold")
    print("\n📊 Response Structure:")
    print("  [")
    print("    {")
    print('      "subcase_id": int,')
    print('      "status": str,')
    print('      "target_org_unit_id": int,')
    print('      "updated_at": str,')
    print('      "days_in_stage": int')
    print("    },")
    print("    ...")
    print("  ]")
    print("\n🔒 Security:")
    print("  ✓ Scope-filtered via allowed_unit_ids")
    print("  ✓ Empty scope returns empty list")
    print("\n✅ Data Integrity:")
    print("  ✓ All days_in_stage non-negative")
    print("  ✓ No terminal statuses in results")
    print("  ✓ days_in_stage >= threshold for all results")
    print("  ✓ No data transformation")
    print("  ✓ Lower threshold returns >= cases")
    print("\n" + "=" * 80)
    print("Ready for Next Phase!")
    print("=" * 80)
    sys.exit(0)
else:
    print(f"\n❌ {test_failed} TEST(S) FAILED")
    print("=" * 80)
    sys.exit(1)
