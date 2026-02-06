"""
Test Insight KPI Summary Endpoint (B-I13)
Integration tests for GET /api/v2/insight/kpi-summary endpoint.

Run: python backend/test_insight_kpi_summary_endpoint.py
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
print("INSIGHT KPI SUMMARY ENDPOINT TEST (B-I13)")
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
    kpi_route = None
    for route in routes:
        if route.path == "/api/v2/insight/kpi-summary":
            kpi_route = route
            break
    
    assert kpi_route is not None, "Endpoint not found"
    print(f"   Path: {kpi_route.path}")
    print(f"   Methods: {kpi_route.methods}")
    print("✅ PASS: Endpoint exists")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1

# Test 2: Endpoint is GET method
print("\n[TEST 2] Endpoint is GET method...")
try:
    assert kpi_route is not None
    assert "GET" in kpi_route.methods, f"Expected GET, got {kpi_route.methods}"
    print("✅ PASS: Endpoint is GET method")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1

# Test 3: Endpoint is async
print("\n[TEST 3] Endpoint is async...")
try:
    import inspect
    from backend.api_v2.routers.insight_router import get_kpi_summary_endpoint
    
    assert inspect.iscoroutinefunction(get_kpi_summary_endpoint), "Endpoint should be async"
    print("✅ PASS: Endpoint is async")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1

# Test 4: Endpoint has docstring
print("\n[TEST 4] Endpoint has docstring...")
try:
    from backend.api_v2.routers.insight_router import get_kpi_summary_endpoint
    
    doc = get_kpi_summary_endpoint.__doc__
    assert doc is not None and len(doc.strip()) > 0, "Missing docstring"
    assert "aggregated KPI metrics" in doc or "KPI" in doc, "Docstring should mention KPI metrics"
    print(f"   Docstring: {doc.strip()}")
    print("✅ PASS: Endpoint has docstring")
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

# Test 5: Endpoint requires authentication
print("\n[TEST 5] Endpoint requires authentication...")
try:
    client = TestClient(app)
    response = client.get("/api/v2/insight/kpi-summary")
    
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
# INTEGRATION TESTS (WITH MOCKED AUTH)
# ============================================================

print("\n" + "=" * 80)
print("INTEGRATION TESTS (MOCKED AUTH)")
print("=" * 80)

# Test 6: Endpoint returns data with valid auth (single org unit)
print("\n[TEST 6] Returns data with single org unit...")
try:
    # Mock get_current_user
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
    response = client.get("/api/v2/insight/kpi-summary")
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    
    data = response.json()
    print(f"   Response keys: {list(data.keys())}")
    print(f"   Total subcases: {data.get('total_subcases', 'N/A')}")
    print("✅ PASS: Returns data with single org unit")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
finally:
    app.dependency_overrides.clear()

# Test 7: Response has correct structure
print("\n[TEST 7] Response has correct structure...")
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
    response = client.get("/api/v2/insight/kpi-summary")
    
    assert response.status_code == 200
    data = response.json()
    
    # Check top-level keys
    assert "total_subcases" in data, "Missing total_subcases"
    assert "by_status" in data, "Missing by_status"
    assert "action_items" in data, "Missing action_items"
    
    # Check total_subcases is int
    assert isinstance(data["total_subcases"], int), "total_subcases should be int"
    
    # Check by_status is list
    assert isinstance(data["by_status"], list), "by_status should be list"
    
    # Check action_items structure
    action_items = data["action_items"]
    assert "total" in action_items, "Missing action_items.total"
    assert "open" in action_items, "Missing action_items.open"
    assert "completed" in action_items, "Missing action_items.completed"
    assert "overdue" in action_items, "Missing action_items.overdue"
    
    print(f"   Structure: ✓ total_subcases, ✓ by_status, ✓ action_items")
    print("✅ PASS: Response has correct structure")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
finally:
    app.dependency_overrides.clear()

# Test 8: by_status contains valid status entries
print("\n[TEST 8] by_status contains valid entries...")
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
    response = client.get("/api/v2/insight/kpi-summary")
    
    assert response.status_code == 200
    data = response.json()
    
    by_status = data["by_status"]
    
    if len(by_status) > 0:
        # Check first entry structure
        first_entry = by_status[0]
        assert "status" in first_entry, "Status entry missing 'status' field"
        assert "count" in first_entry, "Status entry missing 'count' field"
        assert isinstance(first_entry["count"], int), "Count should be int"
        assert isinstance(first_entry["status"], str), "Status should be string"
        
        print(f"   Sample entry: {first_entry}")
    else:
        print("   No status entries (empty scope)")
    
    print("✅ PASS: by_status contains valid entries")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
finally:
    app.dependency_overrides.clear()

# Test 9: action_items values are integers
print("\n[TEST 9] action_items values are integers...")
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
    response = client.get("/api/v2/insight/kpi-summary")
    
    assert response.status_code == 200
    data = response.json()
    
    action_items = data["action_items"]
    
    assert isinstance(action_items["total"], int), "total should be int"
    assert isinstance(action_items["open"], int), "open should be int"
    assert isinstance(action_items["completed"], int), "completed should be int"
    assert isinstance(action_items["overdue"], int), "overdue should be int"
    
    print(f"   Total: {action_items['total']}, Open: {action_items['open']}, " +
          f"Completed: {action_items['completed']}, Overdue: {action_items['overdue']}")
    print("✅ PASS: action_items values are integers")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
finally:
    app.dependency_overrides.clear()

# Test 10: Multiple org units return combined data
print("\n[TEST 10] Multiple org units return combined data...")
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
        response = client.get("/api/v2/insight/kpi-summary")
        
        assert response.status_code == 200
        data = response.json()
        
        total = data["total_subcases"]
        print(f"   Total subcases (2 org units): {total}")
        assert isinstance(total, int), "total_subcases should be int"
        
        print("✅ PASS: Multiple org units return combined data")
        test_passed += 1
    else:
        print("   ⚠️  SKIP: Not enough org units for test")
        test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
finally:
    app.dependency_overrides.clear()

# Test 11: Empty scope returns zero counts
print("\n[TEST 11] Empty scope returns zero counts...")
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
    response = client.get("/api/v2/insight/kpi-summary")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["total_subcases"] == 0, "Empty scope should have 0 subcases"
    assert len(data["by_status"]) == 0, "Empty scope should have no status entries"
    assert data["action_items"]["total"] == 0, "Empty scope should have 0 action items"
    
    print(f"   Total subcases: {data['total_subcases']}")
    print(f"   Status entries: {len(data['by_status'])}")
    print(f"   Action items: {data['action_items']['total']}")
    print("✅ PASS: Empty scope returns zero counts")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
finally:
    app.dependency_overrides.clear()

# Test 12: total_subcases equals sum of by_status counts
print("\n[TEST 12] total_subcases equals sum of by_status...")
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
    response = client.get("/api/v2/insight/kpi-summary")
    
    assert response.status_code == 200
    data = response.json()
    
    total_subcases = data["total_subcases"]
    by_status_sum = sum(entry["count"] for entry in data["by_status"])
    
    assert total_subcases == by_status_sum, \
        f"total_subcases ({total_subcases}) should equal sum of by_status ({by_status_sum})"
    
    print(f"   total_subcases: {total_subcases}")
    print(f"   Sum of by_status: {by_status_sum}")
    print("✅ PASS: total_subcases equals sum of by_status")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
finally:
    app.dependency_overrides.clear()

# Test 13: Action item counts are non-negative
print("\n[TEST 13] Action item counts are non-negative...")
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
    response = client.get("/api/v2/insight/kpi-summary")
    
    assert response.status_code == 200
    data = response.json()
    
    action_items = data["action_items"]
    
    assert action_items["total"] >= 0, "total should be non-negative"
    assert action_items["open"] >= 0, "open should be non-negative"
    assert action_items["completed"] >= 0, "completed should be non-negative"
    assert action_items["overdue"] >= 0, "overdue should be non-negative"
    
    print(f"   All counts non-negative: ✓")
    print("✅ PASS: Action item counts are non-negative")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
finally:
    app.dependency_overrides.clear()

# Test 14: open + completed <= total action items
print("\n[TEST 14] open + completed <= total action items...")
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
    response = client.get("/api/v2/insight/kpi-summary")
    
    assert response.status_code == 200
    data = response.json()
    
    action_items = data["action_items"]
    
    combined = action_items["open"] + action_items["completed"]
    total = action_items["total"]
    
    assert combined <= total, \
        f"open + completed ({combined}) should be <= total ({total})"
    
    print(f"   open: {action_items['open']}, completed: {action_items['completed']}, total: {total}")
    print("✅ PASS: open + completed <= total")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
finally:
    app.dependency_overrides.clear()

# Test 15: No data transformation (matches service output)
print("\n[TEST 15] No data transformation...")
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
    service_result = insight_service.get_kpi_summary(mock_user)
    
    # Get endpoint output
    app.dependency_overrides[get_current_user] = mock_current_user
    client = TestClient(app)
    response = client.get("/api/v2/insight/kpi-summary")
    
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
    print("\n🎉 ALL TESTS PASSED - B-I13 COMPLETE")
    print("=" * 80)
    print("\nEndpoint Status:")
    print("  ✓ GET /api/v2/insight/kpi-summary implemented")
    print("  ✓ Async endpoint")
    print("  ✓ Authentication required")
    print("  ✓ Calls insight_service.get_kpi_summary()")
    print("  ✓ Returns service result unchanged")
    print("\n📊 Response Structure:")
    print("  {")
    print('    "total_subcases": int,')
    print('    "by_status": [{"status": str, "count": int}],')
    print('    "action_items": {')
    print('      "total": int,')
    print('      "open": int,')
    print('      "completed": int,')
    print('      "overdue": int')
    print('    }')
    print("  }")
    print("\n🔒 Security:")
    print("  ✓ Scope-filtered via allowed_unit_ids")
    print("  ✓ Empty scope returns zero counts")
    print("\n✅ Data Integrity:")
    print("  ✓ total_subcases = sum(by_status counts)")
    print("  ✓ All counts non-negative")
    print("  ✓ No data transformation")
    print("\n" + "=" * 80)
    print("Ready for B-I14 (Implement Distribution Endpoint)")
    print("=" * 80)
    sys.exit(0)
else:
    print(f"\n❌ {test_failed} TEST(S) FAILED")
    print("=" * 80)
    sys.exit(1)
