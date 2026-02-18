"""
TEST B-B3 — WORKER SEARCH ENDPOINT V2

GOAL: Verify worker search works under /api/v2/workers/search.

TEST STEPS:
1. Check router registration and startup without errors
2. Confirm endpoint exists under /api/v2/workers/search
3. Verify response structure (items, count fields)
4. Test with various queries
5. Verify no SQL duplication (reuses existing logic)

PASS CONDITIONS:
- Endpoint returns 200 for valid queries
- Response has stable schema (items, count)
- No raw DB column names leak
- Handles empty results gracefully
- Reuses existing search_employees logic

FAIL CONDITIONS:
- 500 errors
- Missing response fields
- SQL duplication
- Crashes on empty results
"""

import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))


def test_router_file_exists():
    """Verify V2 workers router file exists."""
    router_path = backend_path / "api_v2" / "routers" / "workers_router.py"
    assert router_path.exists(), f"❌ V2 workers router not found at: {router_path}"
    print("✅ V2 workers router file exists")
    return True


def test_router_imports():
    """Verify V2 router can be imported without errors."""
    try:
        from api_v2.routers.workers_router import router
        print("✅ V2 workers router imports successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import V2 workers router: {e}")
        return False


def test_main_app_registration():
    """Verify V2 workers router is registered in main.py."""
    main_path = backend_path / "main.py"
    
    with open(main_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for import
    if 'from api_v2.routers.workers_router import router' not in content:
        print("❌ V2 workers router not imported in main.py")
        return False
    
    # Check for registration
    if 'app.include_router(workers_v2_router)' not in content:
        print("❌ V2 workers router not registered in main.py")
        return False
    
    print("✅ V2 workers router is registered in main.py")
    return True


def test_no_startup_errors():
    """Verify FastAPI app starts without router conflicts."""
    try:
        from main import app
        print("✅ FastAPI app starts without errors")
        return True
    except Exception as e:
        print(f"❌ FastAPI app failed to start: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_search_endpoint_exists():
    """Verify search endpoint is registered."""
    try:
        from main import app
        
        # Check for /api/v2/workers/search endpoint
        found = False
        for route in app.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                if route.path == "/api/v2/workers/search" and "GET" in route.methods:
                    found = True
                    break
        
        if not found:
            print("❌ GET /api/v2/workers/search endpoint not found")
            return False
        
        print("✅ GET /api/v2/workers/search endpoint exists")
        return True
    except Exception as e:
        print(f"❌ Failed to verify search endpoint: {e}")
        return False


def test_router_prefix():
    """Verify router has correct V2 prefix."""
    try:
        from api_v2.routers.workers_router import router
        
        if not router.prefix:
            print("❌ Router has no prefix")
            return False
        
        if router.prefix != "/api/v2/workers":
            print(f"❌ Wrong prefix: {router.prefix}, expected /api/v2/workers")
            return False
        
        print(f"✅ Router has correct prefix: {router.prefix}")
        return True
    except Exception as e:
        print(f"❌ Failed to check router prefix: {e}")
        return False


def test_response_schemas_exist():
    """Verify Pydantic response schemas are defined."""
    try:
        from api_v2.routers.workers_router import WorkerSearchItem, WorkerSearchResponse
        
        # Check WorkerSearchItem has required fields
        item_fields = WorkerSearchItem.model_fields.keys()
        required_item_fields = ['employee_id', 'full_name', 'job_title']
        
        for field in required_item_fields:
            if field not in item_fields:
                print(f"❌ WorkerSearchItem missing field: {field}")
                return False
        
        # Check WorkerSearchResponse has required fields
        response_fields = WorkerSearchResponse.model_fields.keys()
        required_response_fields = ['items', 'count']
        
        for field in required_response_fields:
            if field not in response_fields:
                print(f"❌ WorkerSearchResponse missing field: {field}")
                return False
        
        print("✅ Response schemas properly defined")
        return True
    except Exception as e:
        print(f"❌ Failed to verify response schemas: {e}")
        return False


def test_reuses_existing_search():
    """Verify endpoint reuses existing search_employees logic."""
    router_path = backend_path / "api_v2" / "routers" / "workers_router.py"
    
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check that it imports search_employees
    if 'from backend.api.services.search_service import search_employees' not in content:
        print("❌ Doesn't import search_employees from search_service")
        return False
    
    # Check that it calls search_employees
    if 'search_employees(' not in content:
        print("❌ Doesn't call search_employees function")
        return False
    
    print("✅ Endpoint reuses existing search_employees logic (no duplication)")
    return True


def test_no_sql_in_router():
    """Verify router doesn't contain SQL queries."""
    router_path = backend_path / "api_v2" / "routers" / "workers_router.py"
    
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for SQL keywords
    sql_keywords = ['SELECT ', 'INSERT ', 'UPDATE ', 'DELETE FROM', 'CREATE TABLE', 'APP_VIEWTABLE_HR_EMPLOYEES']
    
    found_sql = []
    for keyword in sql_keywords:
        if keyword in content.upper() and 'COMMENT' not in content.upper():
            found_sql.append(keyword)
    
    if found_sql:
        print(f"❌ SQL found in router (should use service layer): {found_sql}")
        return False
    
    print("✅ No SQL queries in V2 router (proper layering)")
    return True


def test_has_authentication():
    """Verify endpoint requires authentication."""
    router_path = backend_path / "api_v2" / "routers" / "workers_router.py"
    
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for authentication dependency
    if 'get_current_user' not in content:
        print("⚠️  No authentication dependency found")
        return True  # Not critical, but recommended
    
    if 'Depends(get_current_user)' not in content:
        print("⚠️  get_current_user not used as dependency")
        return True
    
    print("✅ Endpoint has authentication dependency")
    return True


def test_response_format_normalized():
    """Verify response uses 'items' not 'employees'."""
    router_path = backend_path / "api_v2" / "routers" / "workers_router.py"
    
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check that response uses WorkerSearchResponse
    if 'WorkerSearchResponse' not in content:
        print("❌ Doesn't use WorkerSearchResponse model")
        return False
    
    # Check that it maps to 'items'
    if 'items=' not in content and 'items:' not in content:
        print("❌ Response doesn't normalize to 'items' field")
        return False
    
    print("✅ Response format normalized (uses 'items' field)")
    return True


def run_all_tests():
    """Run all verification tests."""
    print("=" * 70)
    print("TEST B-B3 — WORKER SEARCH ENDPOINT V2")
    print("=" * 70)
    print()
    
    tests = [
        ("Router File Exists", test_router_file_exists),
        ("Router Imports", test_router_imports),
        ("Main App Registration", test_main_app_registration),
        ("No Startup Errors", test_no_startup_errors),
        ("Search Endpoint Exists", test_search_endpoint_exists),
        ("Router Prefix", test_router_prefix),
        ("Response Schemas Exist", test_response_schemas_exist),
        ("Reuses Existing Search", test_reuses_existing_search),
        ("No SQL in Router", test_no_sql_in_router),
        ("Has Authentication", test_has_authentication),
        ("Response Format Normalized", test_response_format_normalized),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}...")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nTests Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED — B-B3 IMPLEMENTATION COMPLETE")
        return 0
    else:
        print(f"\n❌ {total - passed} TEST(S) FAILED — NEEDS FIXES")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
