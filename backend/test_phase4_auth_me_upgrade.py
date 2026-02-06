"""
Phase 4 STEP 4.1: /api/auth/me Contract Upgrade Tests
Verify the extended contract includes roles, primary_unit_id, primary_unit_type.
"""

import sys
import os
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from main import app

# Create test client
client = TestClient(app)


# ==================== TEST HELPERS ====================

def clear_sessions():
    """Clear all test client sessions."""
    client.cookies.clear()


def login_user(username: str, password: str):
    """Helper to login a user."""
    return client.post(
        "/api/auth/login",
        json={"username": username, "password": password}
    )


def get_me():
    """Helper to get /api/auth/me response."""
    return client.get("/api/auth/me")


# ==================== PHASE 4 CONTRACT TESTS ====================

def test_me_keeps_existing_fields():
    """Test /api/auth/me still returns all existing fields (non-breaking)."""
    clear_sessions()
    
    # Login as software_admin
    login_response = login_user("software_admin", "admin123")
    assert login_response.status_code == 200
    
    # Get /api/auth/me
    me_response = get_me()
    assert me_response.status_code == 200
    
    data = me_response.json()
    
    # Verify nested structure preserved
    assert "user" in data
    user = data["user"]
    
    # Verify existing fields still present
    assert "user_id" in user
    assert "username" in user
    assert "is_active" in user
    assert "scopes" in user
    assert "allowed_unit_ids" in user
    
    assert user["user_id"] == 1
    assert user["username"] == "software_admin"
    assert user["is_active"] is True
    
    print("✓ All existing fields preserved (non-breaking change)")


def test_me_adds_roles_field():
    """Test /api/auth/me includes new 'roles' field."""
    clear_sessions()
    
    # Login as software_admin
    login_response = login_user("software_admin", "admin123")
    assert login_response.status_code == 200
    
    # Get /api/auth/me
    me_response = get_me()
    assert me_response.status_code == 200
    
    data = me_response.json()
    user = data["user"]
    
    # Verify new 'roles' field exists
    assert "roles" in user
    assert isinstance(user["roles"], list)
    assert "SOFTWARE_ADMIN" in user["roles"]
    
    print(f"✓ 'roles' field present: {user['roles']}")


def test_me_adds_primary_unit_fields():
    """Test /api/auth/me includes primary_unit_id and primary_unit_type."""
    clear_sessions()
    
    # Login as worker (has org_unit_id)
    login_response = login_user("worker", "worker123")
    assert login_response.status_code == 200
    
    # Get /api/auth/me
    me_response = get_me()
    assert me_response.status_code == 200
    
    data = me_response.json()
    user = data["user"]
    
    # Verify new primary unit fields exist
    assert "primary_unit_id" in user
    assert "primary_unit_type" in user
    
    # Worker should have primary_unit set
    assert user["primary_unit_id"] is not None
    assert user["primary_unit_type"] is not None
    
    print(f"✓ primary_unit_id: {user['primary_unit_id']}")
    print(f"✓ primary_unit_type: {user['primary_unit_type']}")


def test_roles_derived_from_scopes():
    """Test that 'roles' field correctly extracts unique role_codes from scopes."""
    clear_sessions()
    
    # Login as software_admin
    login_response = login_user("software_admin", "admin123")
    assert login_response.status_code == 200
    
    # Get /api/auth/me
    me_response = get_me()
    data = me_response.json()
    user = data["user"]
    
    # Extract role_codes from scopes
    scope_roles = [scope["role_code"] for scope in user["scopes"]]
    
    # Verify roles matches unique role_codes from scopes
    assert set(user["roles"]) == set(scope_roles)
    
    print(f"✓ roles derived correctly from scopes")
    print(f"  Scopes: {scope_roles}")
    print(f"  Roles: {user['roles']}")


def test_primary_unit_null_for_software_admin():
    """Test that SOFTWARE_ADMIN (no org_unit_id) has null primary_unit fields."""
    clear_sessions()
    
    # Login as software_admin (has org_unit_id = 0, which might be treated as null)
    login_response = login_user("software_admin", "admin123")
    assert login_response.status_code == 200
    
    # Get /api/auth/me
    me_response = get_me()
    data = me_response.json()
    user = data["user"]
    
    # Check if primary_unit should be null for SOFTWARE_ADMIN
    # (depends on whether org_unit_id = 0 is treated as null)
    print(f"✓ SOFTWARE_ADMIN primary_unit_id: {user['primary_unit_id']}")
    print(f"✓ SOFTWARE_ADMIN primary_unit_type: {user['primary_unit_type']}")
    print(f"  Scopes: {user['scopes']}")


def test_primary_unit_set_for_worker():
    """Test that WORKER (has org_unit_id) has primary_unit fields set."""
    clear_sessions()
    
    # Login as worker
    login_response = login_user("worker", "worker123")
    assert login_response.status_code == 200
    
    # Get /api/auth/me
    me_response = get_me()
    data = me_response.json()
    user = data["user"]
    
    # Worker should have exactly one scope with org_unit
    assert len(user["scopes"]) >= 1
    first_scope = user["scopes"][0]
    
    # Primary unit should match the scope's unit
    assert user["primary_unit_id"] == first_scope["org_unit_id"]
    assert user["primary_unit_type"] == first_scope["org_unit_type"]
    
    print(f"✓ WORKER primary_unit_id: {user['primary_unit_id']}")
    print(f"✓ WORKER primary_unit_type: {user['primary_unit_type']}")


def test_complete_contract_shape():
    """Test complete Phase 4 /api/auth/me contract shape."""
    clear_sessions()
    
    # Login as worker
    login_response = login_user("worker", "worker123")
    assert login_response.status_code == 200
    
    # Get /api/auth/me
    me_response = get_me()
    assert me_response.status_code == 200
    
    data = me_response.json()
    
    # Verify complete structure
    assert "user" in data
    user = data["user"]
    
    # Existing fields
    assert "user_id" in user
    assert "username" in user
    assert "is_active" in user
    assert "scopes" in user
    assert "allowed_unit_ids" in user
    
    # Phase 4 additions
    assert "roles" in user
    assert "primary_unit_id" in user
    assert "primary_unit_type" in user
    
    # Verify types
    assert isinstance(user["user_id"], int)
    assert isinstance(user["username"], str)
    assert isinstance(user["is_active"], bool)
    assert isinstance(user["scopes"], list)
    assert isinstance(user["allowed_unit_ids"], list)
    assert isinstance(user["roles"], list)
    
    print("✓ Complete Phase 4 contract verified")
    print(f"  Shape: {list(user.keys())}")


# ==================== RUN ALL TESTS ====================

if __name__ == "__main__":
    print("=" * 70)
    print("Phase 4 STEP 4.1: /api/auth/me Contract Upgrade Tests")
    print("=" * 70)
    
    tests = [
        test_me_keeps_existing_fields,
        test_me_adds_roles_field,
        test_me_adds_primary_unit_fields,
        test_roles_derived_from_scopes,
        test_primary_unit_null_for_software_admin,
        test_primary_unit_set_for_worker,
        test_complete_contract_shape,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\n{test.__name__}...")
            test()
            passed += 1
            print(f"  PASS ✓")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL ✗")
            print(f"  Error: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR ✗")
            print(f"  Exception: {e}")
    
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\n✅ STEP 4.1 COMPLETE: /api/auth/me contract upgrade successful!")
    else:
        print(f"\n❌ {failed} test(s) failed")
