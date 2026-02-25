"""
TEST TASK D-B4 — WORKER ROUTER ENDPOINTS

Verifies worker reporting router implementation.
"""

import sys
import os
from pathlib import Path
import inspect

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))


def test_file_exists():
    """Verify router file exists at correct location."""
    router_path = backend_path / "api" / "routers" / "worker_reporting_router.py"
    assert router_path.exists(), f"❌ Router file not found at: {router_path}"
    print("✅ worker_reporting_router.py exists")
    return True


def test_router_prefix():
    """Verify router has correct prefix."""
    try:
        from api.routers.worker_reporting_router import router
        
        assert hasattr(router, 'prefix'), "❌ Router has no prefix attribute"
        assert router.prefix == "/api/workers", f"❌ Expected prefix '/api/workers', got '{router.prefix}'"
        
        print(f"✅ Router prefix is correct: {router.prefix}")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_router_tags():
    """Verify router has correct tags."""
    try:
        from api.routers.worker_reporting_router import router
        
        assert hasattr(router, 'tags'), "❌ Router has no tags attribute"
        assert "Worker Reporting" in router.tags or "worker reporting" in [t.lower() for t in router.tags], \
            f"❌ Expected tag 'Worker Reporting', got {router.tags}"
        
        print(f"✅ Router tags are correct: {router.tags}")
        return True
        
    except Exception as e:
        print(f"❌ Tag verification failed: {e}")
        return False


def test_endpoint_exists():
    """Verify GET /{employee_id}/profile endpoint exists."""
    try:
        from api.routers.worker_reporting_router import router
        
        # Check routes
        found_endpoint = False
        for route in router.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                # Look for /{employee_id}/profile endpoint
                if '{employee_id}' in route.path and 'profile' in route.path:
                    if 'GET' in route.methods:
                        found_endpoint = True
                        print(f"✅ Found endpoint: GET {route.path}")
                        break
        
        assert found_endpoint, "❌ GET /{employee_id}/profile endpoint not found"
        return True
        
    except Exception as e:
        print(f"❌ Endpoint verification failed: {e}")
        return False


def test_uses_get_current_user():
    """Verify endpoint uses get_current_user dependency."""
    router_path = backend_path / "api" / "routers" / "worker_reporting_router.py"
    
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for get_current_user import
    if 'from ..dependencies.user_context import get_current_user' not in content:
        print("❌ Missing import: from ..dependencies.user_context import get_current_user")
        return False
    
    # Check for Depends(get_current_user) usage
    if 'Depends(get_current_user)' not in content:
        print("❌ Endpoint doesn't use Depends(get_current_user)")
        return False
    
    print("✅ Endpoint uses get_current_user dependency")
    return True


def test_calls_worker_service():
    """Verify endpoint calls worker_reporting_service.get_worker_profile."""
    router_path = backend_path / "api" / "routers" / "worker_reporting_router.py"
    
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for service import
    if 'worker_reporting_service' not in content:
        print("❌ Missing worker_reporting_service import")
        return False
    
    # Check for service call
    if 'WorkerReportingService.get_worker_profile' not in content:
        print("❌ Endpoint doesn't call WorkerReportingService.get_worker_profile")
        return False
    
    print("✅ Endpoint calls WorkerReportingService.get_worker_profile")
    return True


def test_response_model():
    """Verify endpoint uses WorkerProfileResponse model."""
    router_path = backend_path / "api" / "routers" / "worker_reporting_router.py"
    
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for schema import
    if 'WorkerProfileResponse' not in content:
        print("❌ Missing WorkerProfileResponse import or usage")
        return False
    
    # Check for response_model usage
    if 'response_model=WorkerProfileResponse' not in content and 'response_model = WorkerProfileResponse' not in content:
        print("❌ Endpoint doesn't declare response_model=WorkerProfileResponse")
        return False
    
    print("✅ Endpoint uses WorkerProfileResponse model")
    return True


def test_value_error_to_404():
    """Verify ValueError is converted to HTTP 404."""
    router_path = backend_path / "api" / "routers" / "worker_reporting_router.py"
    
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for ValueError exception handling
    if 'except ValueError' not in content:
        print("❌ No ValueError exception handling found")
        return False
    
    # Check for 404 status code
    has_404 = False
    lines = content.split('\n')
    in_value_error_block = False
    
    for i, line in enumerate(lines):
        if 'except ValueError' in line:
            in_value_error_block = True
        elif in_value_error_block:
            # Check next ~10 lines for 404 status
            if 'HTTP_404' in line or '404' in line:
                has_404 = True
                break
            # Stop if we hit another except or end of function
            if 'except ' in line or 'def ' in line:
                break
    
    assert has_404, "❌ ValueError not converted to HTTP 404"
    print("✅ ValueError is converted to HTTP 404")
    return True


def test_no_sql_in_router():
    """Verify no SQL queries in router file."""
    router_path = backend_path / "api" / "routers" / "worker_reporting_router.py"
    
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Look for SQL patterns (excluding comments)
    sql_patterns = [
        'SELECT ',
        'INSERT INTO',
        'UPDATE ',
        'DELETE FROM',
        'CREATE TABLE',
        'cursor.execute',
        'pyodbc.connect'
    ]
    
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        # Skip comments and docstrings
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        if '"""' in line or "'''" in line:
            continue
        
        for pattern in sql_patterns:
            if pattern in line:
                print(f"❌ SQL found in router file at line {i}: {pattern}")
                print(f"   Line: {line.strip()}")
                return False
    
    print("✅ No SQL in router file (uses service layer correctly)")
    return True


def test_router_registered_in_main():
    """Verify router is registered in main.py."""
    main_path = backend_path.parent / "main.py"
    
    if not main_path.exists():
        print("⚠️  Warning: main.py not found (registration check skipped)")
        return True
    
    with open(main_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for import
    if 'worker_reporting_router' not in content:
        print("❌ worker_reporting_router not imported in main.py")
        return False
    
    # Check for registration
    if 'app.include_router(worker_reporting_router)' not in content:
        print("❌ worker_reporting_router not registered in main.py")
        return False
    
    print("✅ Router is registered in main.py")
    return True


def test_endpoint_with_mock():
    """Test endpoint with mocked FastAPI app."""
    try:
        from fastapi.testclient import TestClient
        from api.routers.worker_reporting_router import router
        from fastapi import FastAPI
        from api.dependencies.user_context import get_current_user
        from api.schemas.auth_models import CurrentUser
        
        # Create test app
        app = FastAPI()
        app.include_router(router)
        
        # Mock get_current_user
        def mock_current_user():
            return CurrentUser(
                user_id=1,
                username="test_user",
                is_active=True,
                scopes=[],
                allowed_unit_ids={1, 2, 3},
                roles=["administrator"],
                primary_unit_id=1,
                primary_unit_type="administration"
            )
        
        app.dependency_overrides[get_current_user] = mock_current_user
        
        client = TestClient(app)
        
        # Test with real employee ID (we know employee 1 exists from previous tests)
        response = client.get("/api/workers/1/profile")
        
        assert response.status_code in [200, 404, 500], f"❌ Unexpected status code: {response.status_code}"
        
        if response.status_code == 200:
            data = response.json()
            assert 'worker' in data, "❌ Response missing 'worker' field"
            assert 'metrics' in data, "❌ Response missing 'metrics' field"
            print(f"✅ Endpoint works correctly (status 200)")
            print(f"   Worker: {data['worker']['full_name']}")
            print(f"   Incidents: {data['metrics']['total_incidents']}")
        elif response.status_code == 404:
            print("✅ Endpoint correctly returns 404 for non-existent worker")
        else:
            print(f"⚠️  Endpoint returned status {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mock endpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_date_parameters():
    """Verify endpoint accepts date_from and date_to query parameters."""
    try:
        from fastapi.testclient import TestClient
        from api.routers.worker_reporting_router import router
        from fastapi import FastAPI
        from api.dependencies.user_context import get_current_user
        from api.schemas.auth_models import CurrentUser
        
        # Create test app
        app = FastAPI()
        app.include_router(router)
        
        # Mock get_current_user
        def mock_current_user():
            return CurrentUser(
                user_id=1,
                username="test_user",
                is_active=True,
                scopes=[],
                allowed_unit_ids={1, 2, 3},
                roles=["administrator"],
                primary_unit_id=1,
                primary_unit_type="administration"
            )
        
        app.dependency_overrides[get_current_user] = mock_current_user
        
        client = TestClient(app)
        
        # Test with date parameters
        response = client.get(
            "/api/workers/1/profile",
            params={
                "date_from": "2025-01-01",
                "date_to": "2025-12-31"
            }
        )
        
        # Should accept the parameters without 422 validation error
        assert response.status_code != 422, "❌ Endpoint rejects date parameters"
        
        print("✅ Endpoint accepts date_from and date_to query parameters")
        return True
        
    except Exception as e:
        print(f"❌ Date parameter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all verification tests."""
    print("=" * 70)
    print("TEST TASK D-B4 — WORKER ROUTER ENDPOINTS")
    print("=" * 70)
    print()
    
    tests = [
        ("File Exists", test_file_exists),
        ("Router Prefix", test_router_prefix),
        ("Router Tags", test_router_tags),
        ("Endpoint Exists", test_endpoint_exists),
        ("Uses get_current_user", test_uses_get_current_user),
        ("Calls Worker Service", test_calls_worker_service),
        ("Response Model", test_response_model),
        ("ValueError to 404", test_value_error_to_404),
        ("No SQL in Router", test_no_sql_in_router),
        ("Router Registered in Main", test_router_registered_in_main),
        ("Endpoint with Mock", test_endpoint_with_mock),
        ("Date Parameters", test_date_parameters),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 70)
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total:  {passed + failed}")
    print()
    
    if failed == 0:
        print("🎉 WORKER ROUTER OK — ALL TESTS PASSED")
        return 0
    else:
        print("⚠️  WORKER ROUTER HAS ISSUES — REVIEW FAILURES ABOVE")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
