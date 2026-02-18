"""
Test User Workload Endpoint (GET /api/v2/insight/user-workload)
Integration tests for the person-centric workload view.

Run: python backend/test_user_workload_endpoint.py
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
from core.database import get_connection

print("=" * 80)
print("USER WORKLOAD ENDPOINT TEST")
print("=" * 80)

test_passed = 0
test_failed = 0

# Create test FastAPI app
app = FastAPI()
app.include_router(insight_router.router)

# Get real org unit IDs from database
def get_real_org_units():
    """Get real org unit IDs from database."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT DISTINCT TOP 5 TargetOrgUnitID 
            FROM dbo.APP_AdministrativeSubcase 
            WHERE TargetOrgUnitID IS NOT NULL
            ORDER BY TargetOrgUnitID
        """)
        org_units = [row[0] for row in cursor.fetchall()]
        return org_units
    finally:
        cursor.close()
        conn.close()


# ============================================================
# STRUCTURE TESTS
# ============================================================

print("\n" + "=" * 80)
print("STRUCTURE TESTS")
print("=" * 80)

# Test 1: Endpoint exists
print("\n[TEST 1] Endpoint registration...")
try:
    routes = [route for route in app.routes if hasattr(route, 'path')]
    user_workload_route = None
    for route in routes:
        if route.path == "/api/v2/insight/user-workload":
            user_workload_route = route
            break
    
    assert user_workload_route is not None, "Endpoint not found"
    print(f"   Path: {user_workload_route.path}")
    print(f"   Methods: {user_workload_route.methods}")
    print("✅ PASS: Endpoint exists")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1

# Test 2: Endpoint is GET method
print("\n[TEST 2] Endpoint HTTP method...")
try:
    assert user_workload_route is not None
    assert "GET" in user_workload_route.methods, f"Expected GET, got {user_workload_route.methods}"
    print("✅ PASS: Endpoint is GET method")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1

# Test 3: Endpoint requires authentication
print("\n[TEST 3] Authentication requirement...")
try:
    from fastapi import HTTPException
    
    # Mock an unauthenticated user (raises 401)
    def mock_no_auth():
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    app.dependency_overrides[get_current_user] = mock_no_auth
    
    client = TestClient(app)
    response = client.get("/api/v2/insight/user-workload")
    
    # Should return 401 (unauthorized) without valid authentication
    assert response.status_code == 401, \
        f"Expected 401 (unauthorized), got {response.status_code}"
    
    print("✅ PASS: Endpoint requires authentication")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
finally:
    app.dependency_overrides.clear()


# ============================================================
# AUTHORIZATION TESTS
# ============================================================

print("\n" + "=" * 80)
print("AUTHORIZATION TESTS")
print("=" * 80)

# Test 4: SOFTWARE_ADMIN can access
print("\n[TEST 4] SOFTWARE_ADMIN authorization...")
try:
    org_units = get_real_org_units()
    
    def mock_current_user_admin():
        return CurrentUser(
            user_id=1,
            username="admin",
            is_active=True,
            scopes=[],
            allowed_unit_ids=set(org_units),
            roles=["SOFTWARE_ADMIN"]
        )
    
    app.dependency_overrides[get_current_user] = mock_current_user_admin
    
    client = TestClient(app)
    response = client.get("/api/v2/insight/user-workload")
    
    assert response.status_code == 200, \
        f"SOFTWARE_ADMIN should have access, got {response.status_code}"
    
    print("✅ PASS: SOFTWARE_ADMIN can access")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
finally:
    app.dependency_overrides.clear()

# Test 5: WORKER can access
print("\n[TEST 5] WORKER authorization...")
try:
    org_units = get_real_org_units()
    
    def mock_current_user_worker():
        return CurrentUser(
            user_id=2,
            username="worker",
            is_active=True,
            scopes=[],
            allowed_unit_ids=set(org_units),
            roles=["WORKER"]
        )
    
    app.dependency_overrides[get_current_user] = mock_current_user_worker
    
    client = TestClient(app)
    response = client.get("/api/v2/insight/user-workload")
    
    assert response.status_code == 200, \
        f"WORKER should have access, got {response.status_code}"
    
    print("✅ PASS: WORKER can access")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
finally:
    app.dependency_overrides.clear()

# Test 6: COMPLAINT_SUPERVISOR can access
print("\n[TEST 6] COMPLAINT_SUPERVISOR authorization...")
try:
    org_units = get_real_org_units()
    
    def mock_current_user_supervisor():
        return CurrentUser(
            user_id=3,
            username="supervisor",
            is_active=True,
            scopes=[],
            allowed_unit_ids=set(org_units),
            roles=["COMPLAINT_SUPERVISOR"]
        )
    
    app.dependency_overrides[get_current_user] = mock_current_user_supervisor
    
    client = TestClient(app)
    response = client.get("/api/v2/insight/user-workload")
    
    assert response.status_code == 200, \
        f"COMPLAINT_SUPERVISOR should have access, got {response.status_code}"
    
    print("✅ PASS: COMPLAINT_SUPERVISOR can access")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
finally:
    app.dependency_overrides.clear()

# Test 7: SECTION_ADMIN cannot access (403)
print("\n[TEST 7] SECTION_ADMIN authorization (should fail)...")
try:
    org_units = get_real_org_units()
    
    def mock_current_user_section():
        return CurrentUser(
            user_id=4,
            username="section_admin",
            is_active=True,
            scopes=[],
            allowed_unit_ids=set(org_units),
            roles=["SECTION_ADMIN"]
        )
    
    app.dependency_overrides[get_current_user] = mock_current_user_section
    
    client = TestClient(app)
    response = client.get("/api/v2/insight/user-workload")
    
    assert response.status_code == 403, \
        f"SECTION_ADMIN should be forbidden, got {response.status_code}"
    
    data = response.json()
    assert "detail" in data, "Should return error detail"
    assert "permissions" in data["detail"].lower(), "Error should mention permissions"
    
    print("✅ PASS: SECTION_ADMIN correctly forbidden")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
finally:
    app.dependency_overrides.clear()


# ============================================================
# FUNCTIONAL TESTS
# ============================================================

print("\n" + "=" * 80)
print("FUNCTIONAL TESTS")
print("=" * 80)

# Test 8: Response structure
print("\n[TEST 8] Response structure...")
try:
    org_units = get_real_org_units()
    
    def mock_current_user():
        return CurrentUser(
            user_id=1,
            username="admin",
            is_active=True,
            scopes=[],
            allowed_unit_ids=set(org_units),
            roles=["SOFTWARE_ADMIN"]
        )
    
    app.dependency_overrides[get_current_user] = mock_current_user
    
    client = TestClient(app)
    response = client.get("/api/v2/insight/user-workload")
    
    assert response.status_code == 200
    data = response.json()
    
    # Should be a list
    assert isinstance(data, list), f"Expected list, got {type(data)}"
    
    print(f"   Returned {len(data)} users")
    
    # If data is not empty, verify structure
    if len(data) > 0:
        first_user = data[0]
        required_fields = ["user_id", "user_name", "user_role", "primary_org_unit", 
                          "pending_count", "oldest_item_days"]
        
        for field in required_fields:
            assert field in first_user, f"Missing field: {field}"
        
        print(f"   Sample user: {first_user['user_name']} ({first_user['pending_count']} items)")
        print(f"   ✓ All required fields present")
    
    print("✅ PASS: Response structure correct")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
finally:
    app.dependency_overrides.clear()

# Test 9: Sorting by pending_count (default)
print("\n[TEST 9] Default sorting (pending_count desc)...")
try:
    org_units = get_real_org_units()
    
    def mock_current_user():
        return CurrentUser(
            user_id=1,
            username="admin",
            is_active=True,
            scopes=[],
            allowed_unit_ids=set(org_units),
            roles=["SOFTWARE_ADMIN"]
        )
    
    app.dependency_overrides[get_current_user] = mock_current_user
    
    client = TestClient(app)
    response = client. get("/api/v2/insight/user-workload")
    
    assert response.status_code == 200
    data = response.json()
    
    # Check if sorted by pending_count descending
    if len(data) > 1:
        for i in range(len(data) - 1):
            assert data[i]['pending_count'] >= data[i+1]['pending_count'], \
                f"Not sorted correctly: {data[i]['pending_count']} < {data[i+1]['pending_count']}"
        
        print(f"   ✓ Sorted correctly by pending_count (desc)")
        print(f"   Top user: {data[0]['user_name']} with {data[0]['pending_count']} items")
    
    print("✅ PASS: Default sorting works")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
finally:
    app.dependency_overrides.clear()

# Test 10: min_items filter
print("\n[TEST 10] min_items filter...")
try:
    org_units = get_real_org_units()
    
    def mock_current_user():
        return CurrentUser(
            user_id=1,
            username="admin",
            is_active=True,
            scopes=[],
            allowed_unit_ids=set(org_units),
            roles=["SOFTWARE_ADMIN"]
        )
    
    app.dependency_overrides[get_current_user] = mock_current_user
    
    client = TestClient(app)
    
    # Get all users
    response_all = client.get("/api/v2/insight/user-workload")
    data_all = response_all.json()
    
    # Get users with min 5 items
    response_filtered = client.get("/api/v2/insight/user-workload?min_items=5")
    data_filtered = response_filtered.json()
    
    print(f"   All users: {len(data_all)}, Filtered (≥5 items): {len(data_filtered)}")
    
    # All filtered users should have >= 5 items
    for user in data_filtered:
        assert user['pending_count'] >= 5, \
            f"User {user['user_name']} has {user['pending_count']} items, expected ≥5"
    
    print(f"   ✓ All filtered users have ≥5 pending items")
    print("✅ PASS: min_items filter works")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
finally:
    app.dependency_overrides.clear()

# Test 11: Invalid sort_by parameter
print("\n[TEST 11] Invalid sort_by parameter...")
try:
    org_units = get_real_org_units()
    
    def mock_current_user():
        return CurrentUser(
            user_id=1,
            username="admin",
            is_active=True,
            scopes=[],
            allowed_unit_ids=set(org_units),
            roles=["SOFTWARE_ADMIN"]
        )
    
    app.dependency_overrides[get_current_user] = mock_current_user
    
    client = TestClient(app)
    response = client.get("/api/v2/insight/user-workload?sort_by=invalid_field")
    
    assert response.status_code == 400, \
        f"Expected 400 for invalid sort_by, got {response.status_code}"
    
    data = response.json()
    assert "detail" in data, "Should return error detail"
    
    print(f"   ✓ Error message: {data['detail']}")
    print("✅ PASS: Invalid sort_by rejected")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
finally:
    app.dependency_overrides.clear()

# Test 12: Empty result set
print("\n[TEST 12] Empty result set (no users with many items)...")
try:
    org_units = get_real_org_units()
    
    def mock_current_user():
        return CurrentUser(
            user_id=1,
            username="admin",
            is_active=True,
            scopes=[],
            allowed_unit_ids=set(org_units),
            roles=["SOFTWARE_ADMIN"]
        )
    
    app.dependency_overrides[get_current_user] = mock_current_user
    
    client = TestClient(app)
    response = client.get("/api/v2/insight/user-workload?min_items=1000")
    
    assert response.status_code == 200, \
        f"Should return 200 with empty array, got {response.status_code}"
    
    data = response.json()
    assert isinstance(data, list), "Should return list"
    assert len(data) == 0, "Should return empty list"
    
    print(f"   ✓ Empty array returned correctly")
    print("✅ PASS: Empty result set handled")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
finally:
    app.dependency_overrides.clear()


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"Passed: {test_passed}")
print(f"Failed: {test_failed}")
print(f"Total:  {test_passed + test_failed}")

if test_failed == 0:
    print("\n✅ ALL TESTS PASSED")
    sys.exit(0)
else:
    print(f"\n❌ {test_failed} TEST(S) FAILED")
    sys.exit(1)
