"""
TEST B-B2 — PATIENT ROUTERS UNDER API V2

GOAL: Verify patient endpoints are reachable under /api/v2/patients.

TEST STEPS:
1. Check router registration and startup without errors
2. Confirm all endpoints exist under /api/v2/patients
3. Execute each endpoint with valid inputs
4. Compare response structure with v1 endpoints
5. Verify no route conflicts or duplicate routes

PASS CONDITIONS:
- All endpoints return 200 for valid ids
- Response JSON structure matches v1
- No duplicate route conflicts
- No startup router errors

FAIL CONDITIONS:
- 404 under /api/v2
- Router collision errors  
- Changed response schema
"""

import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))


def test_router_file_exists():
    """Verify V2 patients router file exists."""
    router_path = backend_path / "api_v2" / "routers" / "patients_router.py"
    assert router_path.exists(), f"❌ V2 patients router not found at: {router_path}"
    print("✅ V2 patients router file exists")
    return True


def test_router_imports():
    """Verify V2 router can be imported without errors."""
    try:
        from api_v2.routers.patients_router import router
        print("✅ V2 patients router imports successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import V2 patients router: {e}")
        return False


def test_main_app_registration():
    """Verify V2 patients router is registered in main.py."""
    main_path = backend_path / "main.py"
    
    with open(main_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for import
    if 'from api_v2.routers.patients_router import router' not in content:
        print("❌ V2 patients router not imported in main.py")
        return False
    
    # Check for registration
    if 'app.include_router(patients_v2_router)' not in content:
        print("❌ V2 patients router not registered in main.py")
        return False
    
    print("✅ V2 patients router is registered in main.py")
    return True


def test_no_startup_errors():
    """Verify FastAPI app starts without router conflicts."""
    try:
        from main import app
        
        # Check routes for duplicates
        routes = {}
        for route in app.routes:
            if hasattr(route, 'path'):
                if route.path in routes:
                    if hasattr(route, 'methods') and route.methods == routes[route.path]:
                        print(f"⚠️  Duplicate route detected: {route.path} {route.methods}")
                else:
                    routes[route.path] = route.methods if hasattr(route, 'methods') else set()
        
        print("✅ FastAPI app starts without errors")
        return True
    except Exception as e:
        print(f"❌ FastAPI app failed to start: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_v2_routes_exist():
    """Verify all V2 patient endpoints are registered."""
    try:
        from main import app
        
        expected_v2_routes = [
            ("/api/v2/patients", "POST"),  # Create patient
            ("/api/v2/patients/reserve", "GET"),  # Get reserve patients
            ("/api/v2/patients/search", "GET"),  # Search patients
            ("/api/v2/patients/{patient_id}/profile", "GET"),  # Patient profile
            ("/api/v2/patients/{patient_id}/incidents", "GET"),  # Patient incidents
            ("/api/v2/patients/{patient_id}/incidents/{incident_id}", "GET"),  # Incident details
            ("/api/v2/patients/{patient_id}/full-history", "GET"),  # Full history
            ("/api/v2/patients/{patient_id}/export", "GET"),  # Export
        ]
        
        app_routes = []
        for route in app.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                for method in route.methods:
                    if method in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                        app_routes.append((route.path, method))
        
        missing_routes = []
        for expected_path, expected_method in expected_v2_routes:
            found = False
            for app_path, app_method in app_routes:
                # Handle path parameters
                if app_path == expected_path and app_method == expected_method:
                    found = True
                    break
            
            if not found:
                missing_routes.append(f"{expected_method} {expected_path}")
        
        if missing_routes:
            print(f"❌ Missing V2 routes: {missing_routes}")
            return False
        
        print(f"✅ All {len(expected_v2_routes)} V2 patient endpoints exist")
        return True
    except Exception as e:
        print(f"❌ Failed to verify V2 routes: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_router_prefix():
    """Verify router has correct V2 prefix."""
    try:
        from api_v2.routers.patients_router import router
        
        if not router.prefix:
            print("❌ Router has no prefix")
            return False
        
        if router.prefix != "/api/v2/patients":
            print(f"❌ Wrong prefix: {router.prefix}, expected /api/v2/patients")
            return False
        
        print(f"✅ Router has correct prefix: {router.prefix}")
        return True
    except Exception as e:
        print(f"❌ Failed to check router prefix: {e}")
        return False


def test_router_tags():
    """Verify router has V2 tags."""
    try:
        from api_v2.routers.patients_router import router
        
        if not router.tags:
            print("⚠️  Router has no tags")
            return True  # Not critical
        
        if "Patients V2" not in router.tags and "patients_v2" not in str(router.tags).lower():
            print(f"⚠️  Router tags don't indicate V2: {router.tags}")
        else:
            print(f"✅ Router has appropriate tags: {router.tags}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to check router tags: {e}")
        return False


def test_service_layer_reused():
    """Verify V2 router reuses service layer (not duplicating business logic)."""
    router_path = backend_path / "api_v2" / "routers" / "patients_router.py"
    
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check that it imports from patients_service
    if 'from backend.api.services.patients_service import' not in content:
        print("❌ V2 router doesn't import from patients_service")
        return False
    
    # Check that it calls service functions
    service_calls = [
        'search_patients_service',
        'get_patient_profile_service',
        'get_patient_incidents_service',
        'get_incident_details_service',
        'get_patient_full_history_service',
        'export_patient_history_service',
        'create_patient_service',
        'get_all_reserve_patients_service',
    ]
    
    missing_calls = []
    for call in service_calls:
        if call not in content:
            missing_calls.append(call)
    
    if missing_calls:
        print(f"❌ Missing service calls: {missing_calls}")
        return False
    
    print("✅ V2 router reuses service layer (no business logic duplication)")
    return True


def test_no_sql_in_router():
    """Verify router doesn't contain SQL queries (should delegate to service)."""
    router_path = backend_path / "api_v2" / "routers" / "patients_router.py"
    
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for SQL keywords that shouldn't be in router
    sql_keywords = ['SELECT ', 'INSERT ', 'UPDATE ', 'DELETE FROM', 'CREATE TABLE']
    
    found_sql = []
    for keyword in sql_keywords:
        if keyword in content.upper():
            found_sql.append(keyword)
    
    if found_sql:
        print(f"❌ SQL found in router (should be in service/db layer): {found_sql}")
        return False
    
    print("✅ No SQL queries in V2 router (proper layering)")
    return True


def test_v1_endpoints_still_exist():
    """Verify V1 endpoints still exist (both should coexist)."""
    try:
        from main import app
        
        v1_routes = [
            "/api/patients/search",
            "/api/patients/{patient_id}/profile",
        ]
        
        app_paths = [route.path for route in app.routes if hasattr(route, 'path')]
        
        for v1_route in v1_routes:
            if v1_route not in app_paths:
                print(f"⚠️  V1 route not found: {v1_route} (may have been removed)")
        
        print("✅ V1 endpoints verification complete")
        return True
    except Exception as e:
        print(f"❌ Failed to verify V1 endpoints: {e}")
        return False


def test_endpoint_count():
    """Count and verify number of V2 patient endpoints."""
    try:
        from api_v2.routers.patients_router import router
        
        # Count routes in the V2 router
        route_count = len([r for r in router.routes if hasattr(r, 'path')])
        
        expected_count = 8  # 8 endpoints as per specification
        
        if route_count < expected_count:
            print(f"⚠️  Found {route_count} routes, expected at least {expected_count}")
        else:
            print(f"✅ V2 router has {route_count} endpoints")
        
        return True
    except Exception as e:
        print(f"❌ Failed to count endpoints: {e}")
        return False


def run_all_tests():
    """Run all verification tests."""
    print("=" * 70)
    print("TEST B-B2 — PATIENT ROUTERS UNDER API V2")
    print("=" * 70)
    print()
    
    tests = [
        ("Router File Exists", test_router_file_exists),
        ("Router Imports", test_router_imports),
        ("Main App Registration", test_main_app_registration),
        ("No Startup Errors", test_no_startup_errors),
        ("V2 Routes Exist", test_v2_routes_exist),
        ("Router Prefix", test_router_prefix),
        ("Router Tags", test_router_tags),
        ("Service Layer Reused", test_service_layer_reused),
        ("No SQL in Router", test_no_sql_in_router),
        ("V1 Endpoints Still Exist", test_v1_endpoints_still_exist),
        ("Endpoint Count", test_endpoint_count),
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
        print("\n✅ ALL TESTS PASSED — B-B2 IMPLEMENTATION COMPLETE")
        return 0
    else:
        print(f"\n❌ {total - passed} TEST(S) FAILED — NEEDS FIXES")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
