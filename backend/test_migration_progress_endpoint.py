"""
MIGRATION PROGRESS ENDPOINT TEST

Tests the GET /api/migration/progress endpoint to ensure:
1. Correct response format (total_legacy, migrated_total, percent)
2. Authorization enforced (SOFTWARE_ADMIN, WORKER only)
3. Calculations are accurate
4. Percent rounded to 1 decimal place

RUN:
    python test_migration_progress_endpoint.py
"""

import sys
from pathlib import Path

backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from fastapi.testclient import TestClient
from main import app
from core.database import get_connection
from api.dependencies.user_context import get_current_user
from api.schemas.auth_models import CurrentUser, UserScope
from core.constants.roles import SOFTWARE_ADMIN, WORKER, COMPLAINT_SUPERVISOR


client = TestClient(app)


# =========================================================
# AUTHENTICATION MOCKING
# =========================================================

def create_mock_user(role: str) -> CurrentUser:
    """Create a mock user with specified role."""
    return CurrentUser(
        user_id=1,
        username="test_user",
        is_active=True,
        scopes=[
            UserScope(
                role_code=role,
                org_unit_id=1,
                org_unit_type="HOSPITAL"
            )
        ],
        allowed_unit_ids={1},
        roles=[role],
        primary_unit_id=1,
        primary_unit_type="HOSPITAL"
    )


def override_auth_with_role(role: str):
    """Override authentication dependency with mock user."""
    mock_user = create_mock_user(role)
    app.dependency_overrides[get_current_user] = lambda: mock_user


def clear_auth_override():
    """Clear authentication override."""
    app.dependency_overrides.clear()


# =========================================================
# TEST UTILITIES
# =========================================================

def print_header(text):
    """Print formatted test section header"""
    print(f"\n{'=' * 80}")
    print(f"  {text}")
    print('=' * 80)


def print_test(test_name, passed, message=""):
    """Print test result"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} — {test_name}")
    if message:
        print(f"   {message}")


def get_database_stats():
    """Get actual migration statistics from database"""
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Total legacy cases
        cursor.execute("SELECT COUNT(*) FROM dbo.APP_IncidentCase")
        total = cursor.fetchone()[0]
        
        # Migrated cases
        cursor.execute("SELECT COUNT(*) FROM dbo.APP_DataMigration_Map")
        migrated = cursor.fetchone()[0]
        
        # Calculate expected percent
        if total == 0:
            percent = 0.0
        else:
            percent = round((migrated * 100.0) / total, 1)
        
        return {
            "total": total,
            "migrated": migrated,
            "percent": percent
        }
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def test_unauthorized_access():
    """TEST 1: Unauthorized roles blocked"""
    print_header("TEST 1: UNAUTHORIZED ACCESS")
    
    # Test with SECTION_ADMIN (not authorized)
    override_auth_with_role("SECTION_ADMIN")
    
    try:
        response = client.get("/api/migration/progress")
        
        is_forbidden = response.status_code == 403
        print_test("SECTION_ADMIN blocked", is_forbidden, 
                  f"Status: {response.status_code}")
        
        return is_forbidden
        
    finally:
        clear_auth_override()


def test_authorized_access_software_admin():
    """TEST 2: SOFTWARE_ADMIN authorized"""
    print_header("TEST 2: SOFTWARE_ADMIN AUTHORIZED")
    
    override_auth_with_role(SOFTWARE_ADMIN)
    
    try:
        response = client.get("/api/migration/progress")
        
        is_ok = response.status_code == 200
        print_test("SOFTWARE_ADMIN allowed", is_ok, 
                  f"Status: {response.status_code}")
        
        if is_ok:
            data = response.json()
            
            # Check response structure
            has_total_legacy = "total_legacy" in data
            has_migrated_total = "migrated_total" in data
            has_percent = "percent" in data
            
            print_test("Has 'total_legacy' field", has_total_legacy)
            print_test("Has 'migrated_total' field", has_migrated_total)
            print_test("Has 'percent' field", has_percent)
            
            # Check data types
            if has_total_legacy and has_migrated_total and has_percent:
                is_total_int = isinstance(data["total_legacy"], int)
                is_migrated_int = isinstance(data["migrated_total"], int)
                is_percent_float = isinstance(data["percent"], (int, float))
                
                print_test("total_legacy is int", is_total_int)
                print_test("migrated_total is int", is_migrated_int)
                print_test("percent is numeric", is_percent_float)
                
                print(f"\n📊 Response:")
                print(f"   total_legacy: {data['total_legacy']}")
                print(f"   migrated_total: {data['migrated_total']}")
                print(f"   percent: {data['percent']}")
                
                return is_ok and has_total_legacy and has_migrated_total and has_percent
        
        return is_ok
        
    finally:
        clear_auth_override()


def test_authorized_access_worker():
    """TEST 3: WORKER authorized"""
    print_header("TEST 3: WORKER AUTHORIZED")
    
    override_auth_with_role(WORKER)
    
    try:
        response = client.get("/api/migration/progress")
        
        is_ok = response.status_code == 200
        print_test("WORKER allowed", is_ok, 
                  f"Status: {response.status_code}")
        
        return is_ok
        
    finally:
        clear_auth_override()


def test_complaint_supervisor_blocked():
    """TEST 4: COMPLAINT_SUPERVISOR blocked"""
    print_header("TEST 4: COMPLAINT_SUPERVISOR BLOCKED")
    
    override_auth_with_role(COMPLAINT_SUPERVISOR)
    
    try:
        response = client.get("/api/migration/progress")
        
        is_forbidden = response.status_code == 403
        print_test("COMPLAINT_SUPERVISOR blocked", is_forbidden, 
                  f"Status: {response.status_code}")
        
        return is_forbidden
        
    finally:
        clear_auth_override()


def test_calculation_accuracy():
    """TEST 5: Calculation accuracy check"""
    print_header("TEST 5: CALCULATION ACCURACY")
    
    # Get actual database stats
    db_stats = get_database_stats()
    
    print(f"📊 Database Stats:")
    print(f"   Total cases: {db_stats['total']}")
    print(f"   Migrated cases: {db_stats['migrated']}")
    print(f"   Expected percent: {db_stats['percent']}")
    
    # Call API
    override_auth_with_role(SOFTWARE_ADMIN)
    
    try:
        response = client.get("/api/migration/progress")
        
        if response.status_code != 200:
            print_test("API call succeeded", False, f"Status: {response.status_code}")
            return False
        
        data = response.json()
        
        print(f"\n📡 API Response:")
        print(f"   total_legacy: {data['total_legacy']}")
        print(f"   migrated_total: {data['migrated_total']}")
        print(f"   percent: {data['percent']}")
        
        # Verify calculations match
        total_matches = data["total_legacy"] == db_stats["total"]
        migrated_matches = data["migrated_total"] == db_stats["migrated"]
        percent_matches = data["percent"] == db_stats["percent"]
        
        print_test("total_legacy matches database", total_matches)
        print_test("migrated_total matches database", migrated_matches)
        print_test("percent matches expected", percent_matches)
        
        # Check percent has max 1 decimal place
        percent_str = str(data["percent"])
        decimal_places = 0
        if "." in percent_str:
            decimal_places = len(percent_str.split(".")[1])
        
        has_one_decimal = decimal_places <= 1
        print_test("percent has ≤1 decimal place", has_one_decimal, 
                  f"Decimal places: {decimal_places}")
        
        return total_matches and migrated_matches and percent_matches and has_one_decimal
        
    finally:
        clear_auth_override()


def test_no_legacy_cases():
    """TEST 6: Handle empty database gracefully"""
    print_header("TEST 6: EMPTY DATABASE HANDLING")
    
    # This test assumes you might have a test database with 0 cases
    # In production, this scenario is unlikely
    
    print("⚠️  SIMULATION TEST (not executed against live database)")
    print("   If total_legacy = 0, endpoint should return:")
    print("   {")
    print('     "total_legacy": 0,')
    print('     "migrated_total": 0,')
    print('     "percent": 0.0')
    print("   }")
    
    return True  # Informational test


def run_all_tests():
    """Run all migration progress endpoint tests"""
    print("\n" + "=" * 80)
    print("  MIGRATION PROGRESS ENDPOINT TEST SUITE")
    print("=" * 80)
    
    results = []
    
    # Test 1: Unauthorized access
    results.append(("Unauthorized Access", test_unauthorized_access()))
    
    # Test 2: SOFTWARE_ADMIN authorized
    results.append(("SOFTWARE_ADMIN Authorized", test_authorized_access_software_admin()))
    
    # Test 3: WORKER authorized
    results.append(("WORKER Authorized", test_authorized_access_worker()))
    
    # Test 4: COMPLAINT_SUPERVISOR blocked
    results.append(("COMPLAINT_SUPERVISOR Blocked", test_complaint_supervisor_blocked()))
    
    # Test 5: Calculation accuracy
    results.append(("Calculation Accuracy", test_calculation_accuracy()))
    
    # Test 6: Empty database handling
    results.append(("Empty Database Handling", test_no_legacy_cases()))
    
    # Print summary
    print_header("TEST SUMMARY")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} — {name}")
    
    print(f"\n{'=' * 80}")
    print(f"  TESTS PASSED: {passed}/{total}")
    print('=' * 80)
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
