"""
Tests for Bulk Delete Users Endpoint
Tests the POST /api/settings/users/bulk-delete endpoint.

Test Coverage:
- Successful bulk deletion
- Authorization checks (SOFTWARE_ADMIN only)
- Self-deletion prevention
- Protected user prevention (software_admin, SOFTWARE_ADMIN role)
- Non-existent user handling
- Partial success scenarios
- Empty array validation
- Array size limit validation
"""

import pytest
from fastapi.testclient import TestClient
from main import app
from core.database import get_connection

client = TestClient(app)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def software_admin_token():
    """Get authentication token for SOFTWARE_ADMIN user."""
    response = client.post(
        "/api/auth/login",
        json={"username": "software_admin", "password": "admin123"}
    )
    assert response.status_code == 200
    return response.json()["token"]


@pytest.fixture
def regular_user_token():
    """Get authentication token for a regular user (non-admin)."""
    # Assuming there's a test regular user in the database
    response = client.post(
        "/api/auth/login",
        json={"username": "test_user", "password": "test123"}
    )
    if response.status_code == 200:
        return response.json()["token"]
    return None


@pytest.fixture
def create_test_users(software_admin_token):
    """Create multiple test users for bulk deletion testing."""
    created_user_ids = []
    
    for i in range(5):
        response = client.post(
            "/api/settings/users/",
            headers={"Authorization": f"Bearer {software_admin_token}"},
            json={
                "username": f"bulk_test_user_{i}",
                "password": "Test123!",
                "display_name": f"Test User {i}",
                "department_display_name": "Test Department",
                "role_id": 6,  # Assuming role_id 6 is not SOFTWARE_ADMIN
                "org_unit_id": 1
            }
        )
        
        if response.status_code == 201:
            created_user_ids.append(response.json()["user_id"])
    
    yield created_user_ids
    
    # Cleanup: delete any remaining test users
    for user_id in created_user_ids:
        try:
            client.delete(
                f"/api/settings/users/{user_id}",
                headers={"Authorization": f"Bearer {software_admin_token}"}
            )
        except:
            pass  # Ignore errors during cleanup


# ============================================================================
# TEST 1: Successful Bulk Delete
# ============================================================================

def test_bulk_delete_success(software_admin_token, create_test_users):
    """
    Test successful bulk deletion of multiple users.
    
    Expected:
    - 200 OK
    - All users deleted successfully
    - success = true
    - deleted_count matches requested count
    - failed_count = 0
    """
    user_ids = create_test_users[:3]  # Delete first 3 users
    
    response = client.post(
        "/api/settings/users/bulk-delete",
        headers={"Authorization": f"Bearer {software_admin_token}"},
        json={"user_ids": user_ids}
    )
    
    print("\n[TEST] Bulk Delete Success")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 200
    
    data = response.json()
    assert data["success"] is True
    assert data["deleted_count"] == 3
    assert data["failed_count"] == 0
    assert len(data["deleted_users"]) == 3
    assert len(data["failed_users"]) == 0
    assert "Successfully deleted 3 user(s)" in data["message"]
    
    # Verify users are actually deleted
    for user_id in user_ids:
        response = client.get(
            "/api/settings/users/",
            headers={"Authorization": f"Bearer {software_admin_token}"}
        )
        users = response.json()
        assert not any(u["user_id"] == user_id for u in users)


# ============================================================================
# TEST 2: Authorization Check - Non-Admin User
# ============================================================================

def test_bulk_delete_authorization_failure(create_test_users):
    """
    Test that non-SOFTWARE_ADMIN users cannot bulk delete.
    
    Expected:
    - 401 Unauthorized (if no token)
    - 403 Forbidden (if regular user token)
    """
    user_ids = create_test_users[:2]
    
    # Test without token
    response = client.post(
        "/api/settings/users/bulk-delete",
        json={"user_ids": user_ids}
    )
    
    print("\n[TEST] Bulk Delete - No Authorization")
    print(f"Status Code: {response.status_code}")
    
    assert response.status_code == 401  # Unauthorized
    
    # Test with regular user token (if available)
    # This would require creating a regular user first


# ============================================================================
# TEST 3: Prevent Self-Deletion
# ============================================================================

def test_bulk_delete_prevent_self_deletion(software_admin_token, create_test_users):
    """
    Test that user cannot delete their own account.
    
    Expected:
    - 200 OK (partial success)
    - success = false
    - Current user in failed_users with appropriate reason
    - Other users deleted successfully
    """
    # Get current user ID
    response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {software_admin_token}"}
    )
    current_user_id = response.json()["user_id"]
    
    # Try to delete self + other users
    user_ids = [current_user_id] + create_test_users[:2]
    
    response = client.post(
        "/api/settings/users/bulk-delete",
        headers={"Authorization": f"Bearer {software_admin_token}"},
        json={"user_ids": user_ids}
    )
    
    print("\n[TEST] Bulk Delete - Prevent Self Deletion")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 200
    
    data = response.json()
    assert data["success"] is False
    assert data["deleted_count"] == 2
    assert data["failed_count"] == 1
    
    # Verify current user is in failed_users
    failed_user_ids = [u["user_id"] for u in data["failed_users"]]
    assert current_user_id in failed_user_ids
    
    # Find the failure reason
    failed_user = next(u for u in data["failed_users"] if u["user_id"] == current_user_id)
    assert "Cannot delete currently logged in user" in failed_user["reason"]


# ============================================================================
# TEST 4: Protected User Prevention
# ============================================================================

def test_bulk_delete_protected_users(software_admin_token, create_test_users):
    """
    Test that protected users (software_admin, SOFTWARE_ADMIN role) cannot be deleted.
    
    Expected:
    - 200 OK (partial success)
    - success = false
    - Protected users in failed_users with appropriate reasons
    """
    # Try to find software_admin user ID
    response = client.get(
        "/api/settings/users/",
        headers={"Authorization": f"Bearer {software_admin_token}"}
    )
    users = response.json()
    
    software_admin_user = next(
        (u for u in users if u["username"].lower() == "software_admin"),
        None
    )
    
    if not software_admin_user:
        pytest.skip("software_admin user not found in database")
    
    # Try to delete software_admin + regular test users
    user_ids = [software_admin_user["user_id"]] + create_test_users[:2]
    
    response = client.post(
        "/api/settings/users/bulk-delete",
        headers={"Authorization": f"Bearer {software_admin_token}"},
        json={"user_ids": user_ids}
    )
    
    print("\n[TEST] Bulk Delete - Protected Users")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 200
    
    data = response.json()
    assert data["success"] is False
    assert data["deleted_count"] == 2
    assert data["failed_count"] == 1
    
    # Verify software_admin is in failed_users
    failed_user_ids = [u["user_id"] for u in data["failed_users"]]
    assert software_admin_user["user_id"] in failed_user_ids


# ============================================================================
# TEST 5: Non-Existent User IDs
# ============================================================================

def test_bulk_delete_nonexistent_users(software_admin_token):
    """
    Test bulk delete with non-existent user IDs.
    
    Expected:
    - 200 OK
    - success = false
    - All users in failed_users with "User not found" reason
    """
    # Use very high user IDs that don't exist
    user_ids = [99999, 88888, 77777]
    
    response = client.post(
        "/api/settings/users/bulk-delete",
        headers={"Authorization": f"Bearer {software_admin_token}"},
        json={"user_ids": user_ids}
    )
    
    print("\n[TEST] Bulk Delete - Non-Existent Users")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 200
    
    data = response.json()
    assert data["success"] is False
    assert data["deleted_count"] == 0
    assert data["failed_count"] == 3
    assert len(data["failed_users"]) == 3
    
    # Verify all have "User not found" reason
    for failed_user in data["failed_users"]:
        assert "User not found" in failed_user["reason"]


# ============================================================================
# TEST 6: Empty Array Validation
# ============================================================================

def test_bulk_delete_empty_array(software_admin_token):
    """
    Test bulk delete with empty user_ids array.
    
    Expected:
    - 422 Unprocessable Entity (Pydantic validation error)
    """
    response = client.post(
        "/api/settings/users/bulk-delete",
        headers={"Authorization": f"Bearer {software_admin_token}"},
        json={"user_ids": []}
    )
    
    print("\n[TEST] Bulk Delete - Empty Array")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 422  # Validation error


# ============================================================================
# TEST 7: Array Size Limit (> 100 users)
# ============================================================================

def test_bulk_delete_array_size_limit(software_admin_token):
    """
    Test bulk delete with more than 100 user IDs.
    
    Expected:
    - 422 Unprocessable Entity (Pydantic validation error)
    """
    # Create array with 101 user IDs
    user_ids = list(range(1, 102))
    
    response = client.post(
        "/api/settings/users/bulk-delete",
        headers={"Authorization": f"Bearer {software_admin_token}"},
        json={"user_ids": user_ids}
    )
    
    print("\n[TEST] Bulk Delete - Array Size Limit")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 422  # Validation error


# ============================================================================
# TEST 8: Mixed Success and Failure
# ============================================================================

def test_bulk_delete_mixed_results(software_admin_token, create_test_users):
    """
    Test bulk delete with a mix of valid and invalid user IDs.
    
    Expected:
    - 200 OK
    - success = false
    - Some users deleted, some failed
    - Detailed breakdown in response
    """
    # Mix of valid users and non-existent users
    valid_users = create_test_users[:2]
    invalid_users = [99999, 88888]
    user_ids = valid_users + invalid_users
    
    response = client.post(
        "/api/settings/users/bulk-delete",
        headers={"Authorization": f"Bearer {software_admin_token}"},
        json={"user_ids": user_ids}
    )
    
    print("\n[TEST] Bulk Delete - Mixed Results")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 200
    
    data = response.json()
    assert data["success"] is False
    assert data["deleted_count"] == 2
    assert data["failed_count"] == 2
    assert len(data["deleted_users"]) == 2
    assert len(data["failed_users"]) == 2
    
    # Verify message contains both success and failure info
    assert "Deleted 2 out of 4 user(s)" in data["message"]


# ============================================================================
# TEST 9: Invalid User ID (negative or zero)
# ============================================================================

def test_bulk_delete_invalid_user_ids(software_admin_token):
    """
    Test bulk delete with invalid user IDs (negative, zero).
    
    Expected:
    - 422 Unprocessable Entity (Pydantic validation error)
    """
    response = client.post(
        "/api/settings/users/bulk-delete",
        headers={"Authorization": f"Bearer {software_admin_token}"},
        json={"user_ids": [0, -1, -5]}
    )
    
    print("\n[TEST] Bulk Delete - Invalid User IDs")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 422  # Validation error


# ============================================================================
# MANUAL TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    """
    Run tests manually for debugging purposes.
    Usage: python test_bulk_delete_users.py
    """
    print("=" * 80)
    print("BULK DELETE USERS - MANUAL TEST SUITE")
    print("=" * 80)
    
    # Get token
    print("\n[SETUP] Logging in as software_admin...")
    response = client.post(
        "/api/auth/login",
        json={"username": "software_admin", "password": "admin123"}
    )
    
    if response.status_code != 200:
        print(f"❌ Login failed: {response.status_code}")
        print(f"Response: {response.json()}")
        exit(1)
    
    token = response.json()["token"]
    print("✅ Login successful")
    
    # Create test users
    print("\n[SETUP] Creating test users...")
    created_user_ids = []
    
    for i in range(5):
        response = client.post(
            "/api/settings/users/",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "username": f"bulk_test_user_{i}",
                "password": "Test123!",
                "display_name": f"Test User {i}",
                "department_display_name": "Test Department",
                "role_id": 6,
                "org_unit_id": 1
            }
        )
        
        if response.status_code == 201:
            user_id = response.json()["user_id"]
            created_user_ids.append(user_id)
            print(f"✅ Created user {user_id}: bulk_test_user_{i}")
        else:
            print(f"⚠️ Failed to create user {i}: {response.status_code}")
    
    if not created_user_ids:
        print("❌ No test users created. Exiting.")
        exit(1)
    
    # Run Test 1: Successful bulk delete
    print("\n" + "=" * 80)
    print("TEST 1: Successful Bulk Delete")
    print("=" * 80)
    
    user_ids_to_delete = created_user_ids[:3]
    response = client.post(
        "/api/settings/users/bulk-delete",
        headers={"Authorization": f"Bearer {token}"},
        json={"user_ids": user_ids_to_delete}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 200:
        data = response.json()
        if data["success"] and data["deleted_count"] == 3:
            print("✅ TEST PASSED")
        else:
            print("❌ TEST FAILED - Unexpected results")
    else:
        print("❌ TEST FAILED - Wrong status code")
    
    # Cleanup remaining test users
    print("\n[CLEANUP] Deleting remaining test users...")
    for user_id in created_user_ids[3:]:
        try:
            response = client.delete(
                f"/api/settings/users/{user_id}",
                headers={"Authorization": f"Bearer {token}"}
            )
            if response.status_code == 200:
                print(f"✅ Deleted user {user_id}")
        except Exception as e:
            print(f"⚠️ Failed to delete user {user_id}: {e}")
    
    print("\n" + "=" * 80)
    print("MANUAL TEST SUITE COMPLETED")
    print("=" * 80)
