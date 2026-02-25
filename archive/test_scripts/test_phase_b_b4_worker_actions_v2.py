"""
TEST B-B4 — WORKER ACTION LIST ENDPOINT V2
Phase B — B-B4 — Structural validation

GOAL:
Verify worker action list endpoint is properly implemented without duplicating business logic.

TEST APPROACH:
- Check router file exists
- Verify imports work
- Ensure endpoint is registered in main.py
- Validate response schemas
- Confirm Phase D DB logic is reused
- Verify no SQL in router
- Check authentication
- Verify pagination support
"""

import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

def header(msg):
    print(f"\n{'='*70}")
    print(msg)
    print('='*70)

def test_step(msg):
    print(f"\n🔍 {msg}")

def success(msg):
    print(f"✅ {msg}")

def failure(msg):
    print(f"❌ {msg}")
    return False


# ============================================================
# TEST EXECUTION
# ============================================================

header("TEST B-B4 — WORKER ACTION LIST ENDPOINT V2")
tests_passed = 0
tests_total = 0

# ------------------------------------------------------------
# TEST 1: Router File Exists
# ------------------------------------------------------------
test_step("Router File Exists...")
tests_total += 1
try:
    router_path = "backend/api_v2/routers/workers_router.py"
    assert os.path.exists(router_path), f"Router file not found: {router_path}"
    success("V2 workers router file exists")
    tests_passed += 1
except AssertionError as e:
    failure(str(e))

# ------------------------------------------------------------
# TEST 2: Router Imports Successfully
# ------------------------------------------------------------
test_step("Router Imports...")
tests_total += 1
try:
    from api_v2.routers import workers_router
    success("V2 workers router imports successfully")
    tests_passed += 1
except Exception as e:
    failure(f"Failed to import workers router: {e}")

# ------------------------------------------------------------
# TEST 3: Main App Registration
# ------------------------------------------------------------
test_step("Main App Registration...")
tests_total += 1
try:
    import inspect
    from main import app
    
    # Check if workers_v2_router is included
    router_found = False
    for route in app.routes:
        if hasattr(route, 'path') and '/api/v2/workers' in route.path:
            router_found = True
            break
    
    assert router_found, "Workers V2 router not registered in main.py (no /api/v2/workers routes found)"
    
    success("V2 workers router is registered in main.py")
    tests_passed += 1
except AssertionError as e:
    failure(str(e))
except Exception as e:
    failure(f"Failed to check main.py: {e}")

# ------------------------------------------------------------
# TEST 4: No Startup Errors
# ------------------------------------------------------------
test_step("No Startup Errors...")
tests_total += 1
try:
    from main import app
    success("FastAPI app starts without errors")
    tests_passed += 1
except Exception as e:
    failure(f"FastAPI app failed to start: {e}")

# ------------------------------------------------------------
# TEST 5: Action Endpoint Exists
# ------------------------------------------------------------
test_step("Action Endpoint Exists...")
tests_total += 1
try:
    from api_v2.routers.workers_router import router
    
    # Check if endpoint exists
    endpoint_found = False
    for route in router.routes:
        if hasattr(route, 'path') and '/actions' in route.path:
            if hasattr(route, 'methods') and 'GET' in route.methods:
                endpoint_found = True
                break
    
    assert endpoint_found, "GET /{employee_id}/actions endpoint not found"
    success("GET /api/v2/workers/{employee_id}/actions endpoint exists")
    tests_passed += 1
except AssertionError as e:
    failure(str(e))
except Exception as e:
    failure(f"Failed to check endpoint: {e}")

# ------------------------------------------------------------
# TEST 6: Router Prefix
# ------------------------------------------------------------
test_step("Router Prefix...")
tests_total += 1
try:
    from api_v2.routers.workers_router import router
    
    assert router.prefix == "/api/v2/workers", f"Expected prefix '/api/v2/workers', got '{router.prefix}'"
    success("Router has correct prefix: /api/v2/workers")
    tests_passed += 1
except AssertionError as e:
    failure(str(e))
except Exception as e:
    failure(f"Failed to check prefix: {e}")

# ------------------------------------------------------------
# TEST 7: Response Schemas Exist
# ------------------------------------------------------------
test_step("Response Schemas Exist...")
tests_total += 1
try:
    from api_v2.routers.workers_router import WorkerActionItem, WorkerActionListResponse
    
    # Check WorkerActionItem has required fields
    assert hasattr(WorkerActionItem, 'model_fields') or hasattr(WorkerActionItem, '__fields__'), \
           "WorkerActionItem is not a Pydantic model"
    
    # Check WorkerActionListResponse has required fields
    assert hasattr(WorkerActionListResponse, 'model_fields') or hasattr(WorkerActionListResponse, '__fields__'), \
           "WorkerActionListResponse is not a Pydantic model"
    
    success("Response schemas properly defined")
    tests_passed += 1
except ImportError as e:
    failure(f"Response schemas not found: {e}")
except AssertionError as e:
    failure(str(e))
except Exception as e:
    failure(f"Failed to check schemas: {e}")

# ------------------------------------------------------------
# TEST 8: Reuses Phase D DB Logic
# ------------------------------------------------------------
test_step("Reuses Phase D DB Logic...")
tests_total += 1
try:
    import inspect
    from api_v2.routers import workers_router as router_module
    
    # Get the source code
    router_source = inspect.getsource(router_module)
    
    # Check that it imports from action_item_subcase_db
    assert "action_item_subcase_db" in router_source, \
           "Router does not import from action_item_subcase_db"
    
    # Check that it calls get_worker_action_items
    assert "get_worker_action_items" in router_source, \
           "Router does not call get_worker_action_items function"
    
    success("Endpoint reuses Phase D DB logic (no duplication)")
    tests_passed += 1
except AssertionError as e:
    failure(str(e))
except Exception as e:
    failure(f"Failed to check logic reuse: {e}")

# ------------------------------------------------------------
# TEST 9: No SQL in Router
# ------------------------------------------------------------
test_step("No SQL in Router...")
tests_total += 1
try:
    import inspect
    from api_v2.routers import workers_router as router_module
    
    # Get the source code
    router_source = inspect.getsource(router_module)
    
    # Check for SQL keywords that shouldn't be in router
    sql_patterns = ["SELECT ", "INSERT ", "UPDATE ", "DELETE ", "cursor.execute"]
    
    found_sql = []
    for pattern in sql_patterns:
        if pattern in router_source:
            found_sql.append(pattern)
    
    assert len(found_sql) == 0, f"Found SQL in router: {found_sql}"
    success("No SQL queries in V2 router (proper layering)")
    tests_passed += 1
except AssertionError as e:
    failure(str(e))
except Exception as e:
    failure(f"Failed to check SQL: {e}")

# ------------------------------------------------------------
# TEST 10: Has Authentication
# ------------------------------------------------------------
test_step("Has Authentication...")
tests_total += 1
try:
    from api_v2.routers.workers_router import router
    
    # Find the actions endpoint
    endpoint_found = False
    has_auth = False
    
    for route in router.routes:
        if hasattr(route, 'path') and '/actions' in route.path:
            if hasattr(route, 'dependant') and hasattr(route.dependant, 'dependencies'):
                # Check if get_current_user is in dependencies
                for dep in route.dependant.dependencies:
                    if hasattr(dep, 'call') and 'get_current_user' in str(dep.call):
                        has_auth = True
                        break
            endpoint_found = True
            break
    
    assert endpoint_found, "Actions endpoint not found"
    assert has_auth, "Actions endpoint does not have authentication dependency"
    success("Endpoint has authentication dependency")
    tests_passed += 1
except AssertionError as e:
    failure(str(e))
except Exception as e:
    failure(f"Failed to check authentication: {e}")

# ------------------------------------------------------------
# TEST 11: Supports Pagination
# ------------------------------------------------------------
test_step("Supports Pagination...")
tests_total += 1
try:
    import inspect
    from api_v2.routers import workers_router as router_module
    from api_v2.db_layer import action_item_subcase_db
    
    # Get router source
    router_source = inspect.getsource(router_module)
    
    # Check for pagination parameters
    assert "limit" in router_source, "Endpoint does not have 'limit' parameter"
    assert "offset" in router_source, "Endpoint does not have 'offset' parameter"
    
    # Check DB function source
    db_source = inspect.getsource(action_item_subcase_db)
    
    assert "get_worker_action_items" in db_source, "DB function get_worker_action_items not found"
    assert "OFFSET" in db_source and "FETCH NEXT" in db_source, \
           "DB function does not use SQL pagination (OFFSET/FETCH)"
    
    success("Pagination supported (limit, offset)")
    tests_passed += 1
except AssertionError as e:
    failure(str(e))
except Exception as e:
    failure(f"Failed to check pagination: {e}")

# ------------------------------------------------------------
# TEST 12: Response Format Stable
# ------------------------------------------------------------
test_step("Response Format Stable...")
tests_total += 1
try:
    from api_v2.routers.workers_router import WorkerActionListResponse
    
    # Check response has required fields
    fields = WorkerActionListResponse.model_fields if hasattr(WorkerActionListResponse, 'model_fields') else WorkerActionListResponse.__fields__
    
    assert 'items' in fields, "Response missing 'items' field"
    assert 'count' in fields, "Response missing 'count' field"
    assert 'limit' in fields, "Response missing 'limit' field"
    assert 'offset' in fields, "Response missing 'offset' field"
    
    success("Response format follows pagination contract (items, count, limit, offset)")
    tests_passed += 1
except AssertionError as e:
    failure(str(e))
except Exception as e:
    failure(f"Failed to check response format: {e}")

# ============================================================
# SUMMARY
# ============================================================
header("SUMMARY")
print(f"\nTests Passed: {tests_passed}/{tests_total}")

if tests_passed == tests_total:
    print("\n✅ ALL TESTS PASSED — B-B4 IMPLEMENTATION COMPLETE")
    sys.exit(0)
else:
    print(f"\n❌ {tests_total - tests_passed} TEST(S) FAILED")
    sys.exit(1)
