"""
AUTHORIZATION TEST - Migration Progress Endpoint

Tests that only SOFTWARE_ADMIN and WORKER can access the endpoint.

RUN:
    python test_migration_progress_auth.py
"""

import sys
from pathlib import Path

backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from fastapi.testclient import TestClient
from main import app
from api.dependencies.user_context import get_current_user
from api.schemas.auth_models import CurrentUser, UserScope
from core.constants.roles import SOFTWARE_ADMIN, WORKER, COMPLAINT_SUPERVISOR

client = TestClient(app)


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


def test_role(role_name: str, should_pass: bool):
    """Test endpoint access with specific role"""
    mock_user = create_mock_user(role_name)
    app.dependency_overrides[get_current_user] = lambda: mock_user
    
    try:
        response = client.get("/api/migration/progress")
        
        if should_pass:
            passed = response.status_code == 200
            status_icon = "✅" if passed else "❌"
            print(f"{status_icon} {role_name:25} → Status {response.status_code} (expected 200)")
        else:
            passed = response.status_code == 403
            status_icon = "✅" if passed else "❌"
            print(f"{status_icon} {role_name:25} → Status {response.status_code} (expected 403)")
        
        return passed
        
    finally:
        app.dependency_overrides.clear()


def run_auth_tests():
    """Run authorization tests for all roles"""
    print("\n" + "=" * 80)
    print("  MIGRATION PROGRESS ENDPOINT - AUTHORIZATION TESTS")
    print("=" * 80)
    print("\nTesting access control...\n")
    
    results = []
    
    # Roles that SHOULD have access
    print("✓ Should have access:")
    results.append(test_role(SOFTWARE_ADMIN, should_pass=True))
    results.append(test_role(WORKER, should_pass=True))
    
    print("\n✗ Should be blocked:")
    # Roles that should NOT have access
    results.append(test_role(COMPLAINT_SUPERVISOR, should_pass=False))
    results.append(test_role("SECTION_ADMIN", should_pass=False))
    results.append(test_role("DEPARTMENT_VIEWER", should_pass=False))
    results.append(test_role("WORKER_VIEWER", should_pass=False))
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 80)
    if passed == total:
        print(f"  🎉 ALL AUTHORIZATION TESTS PASSED ({passed}/{total})")
    else:
        print(f"  ⚠️  SOME TESTS FAILED ({passed}/{total} passed)")
    print("=" * 80)
    
    return passed == total


if __name__ == "__main__":
    try:
        success = run_auth_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
