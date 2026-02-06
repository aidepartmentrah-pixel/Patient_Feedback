"""
PHASE B — B-B7 — ROUTER INTEGRATION TESTS

Test suite for Settings Users Router endpoints.
Tests CRUD operations and admin password reset.

Run from backend directory:
    python test_settings_users_router.py
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
from core.database import get_connection

# Create test client
client = TestClient(app)


def clear_sessions():
    """Clear all test client sessions."""
    client.cookies.clear()


def login_as_software_admin():
    """Helper to login as software_admin."""
    clear_sessions()
    response = client.post(
        "/api/auth/login",
        json={"username": "software_admin", "password": "admin123"}
    )
    if response.status_code != 200:
        raise Exception(f"Failed to login as software_admin: {response.status_code}")
    return response


def login_as_regular_user():
    """Helper to login as a non-SOFTWARE_ADMIN user."""
    clear_sessions()
    # Try to find a section admin or any non-SOFTWARE_ADMIN user
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Find a SECTION_ADMIN user
        cursor.execute("""
            SELECT TOP 1 u.Username
            FROM dbo.APP_Users u
            INNER JOIN dbo.APP_UserRoleScope urs ON u.UserID = urs.UserID
            INNER JOIN dbo.APP_Roles r ON urs.RoleID = r.RoleID
            WHERE r.RoleCode = 'SECTION_ADMIN'
            AND u.IsActive = 1
        """)
        
        row = cursor.fetchone()
        if row:
            username = row.Username
            # Try common test password
            response = client.post(
                "/api/auth/login",
                json={"username": username, "password": "Hospital2026!"}
            )
            return response
        
        return None
        
    finally:
        cursor.close()
        conn.close()


def cleanup_test_users(conn):
    """Clean up test users from previous runs."""
    cursor = conn.cursor()
    try:
        cursor.execute("""
            DELETE FROM dbo.APP_UserRoleScope 
            WHERE UserID IN (
                SELECT UserID FROM dbo.APP_Users 
                WHERE Username LIKE 'bb7_router_%'
            )
        """)
        cursor.execute("""
            DELETE FROM dbo.APP_Users 
            WHERE Username LIKE 'bb7_router_%'
        """)
        conn.commit()
    finally:
        cursor.close()


def test_get_users_list():
    """Test 1: GET /api/settings/users returns user list."""
    print("\n" + "="*60)
    print("TEST 1: GET /api/settings/users")
    print("="*60)
    
    try:
        # Login as admin
        login_as_software_admin()
        
        # Call endpoint
        response = client.get("/api/settings/users/")
        
        print(f"Status Code: {response.status_code}")
        
        # Assertions
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert isinstance(data, list), f"Expected list, got {type(data)}"
        
        if len(data) > 0:
            # Check first item has required keys
            first_item = data[0]
            required_keys = ["user_id", "username", "role_name", "org_unit_name", "is_active"]
            for key in required_keys:
                assert key in first_item, f"Missing key: {key}"
            
            print(f"✓ Returned {len(data)} users")
            print(f"✓ Sample user: {first_item['username']}")
        else:
            print(f"✓ Returned empty list (no users in system)")
        
        print("\n✓ TEST 1 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 1 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 1 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        clear_sessions()


def test_create_user():
    """Test 2: POST /api/settings/users creates a user."""
    print("\n" + "="*60)
    print("TEST 2: POST /api/settings/users")
    print("="*60)
    
    conn = get_connection()
    
    try:
        # Cleanup first
        cleanup_test_users(conn)
        
        # Get valid IDs
        cursor = conn.cursor()
        cursor.execute("SELECT TOP 1 UniqueID FROM dbo.AdminsrationUnit WHERE Frozen = 0")
        org_unit_id = cursor.fetchone().UniqueID
        
        cursor.execute("SELECT TOP 1 RoleID FROM dbo.APP_Roles")
        role_id = cursor.fetchone().RoleID
        cursor.close()
        
        # Login as admin
        login_as_software_admin()
        
        # Create user payload
        payload = {
            "username": "bb7_router_create_test",
            "password": "TestPass123!",
            "display_name": "Router Test User",
            "department_display_name": "Test Department",
            "role_id": role_id,
            "org_unit_id": org_unit_id
        }
        
        print(f"Creating user: {payload['username']}")
        
        # Call endpoint
        response = client.post("/api/settings/users/", json=payload)
        
        print(f"Status Code: {response.status_code}")
        
        # Print error details if not 201
        if response.status_code != 201:
            print(f"Response body: {response.text}")
        
        # Assertions
        assert response.status_code == 201, f"Expected 201, got {response.status_code}"
        
        data = response.json()
        assert "user_id" in data, "Missing user_id in response"
        assert "username" in data, "Missing username in response"
        assert data["username"] == payload["username"], f"Expected username={payload['username']}, got {data['username']}"
        
        user_id = data["user_id"]
        print(f"✓ Created user with ID: {user_id}")
        
        # Verify in database
        cursor = conn.cursor()
        cursor.execute("""
            SELECT Username, DisplayName, DepartmentDisplayName
            FROM dbo.APP_Users
            WHERE UserID = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        cursor.close()
        
        assert row is not None, "User not found in database"
        assert row.Username == payload["username"], "Username mismatch in DB"
        assert row.DisplayName == payload["display_name"], "DisplayName mismatch in DB"
        
        print(f"✓ Verified user in database")
        print(f"  Username: {row.Username}")
        print(f"  DisplayName: {row.DisplayName}")
        print(f"  DepartmentDisplayName: {row.DepartmentDisplayName}")
        
        print("\n✓ TEST 2 PASSED")
        return True, user_id
        
    except AssertionError as e:
        print(f"\n✗ TEST 2 FAILED: {str(e)}")
        return False, None
    except Exception as e:
        print(f"\n✗ TEST 2 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None
    finally:
        clear_sessions()
        conn.rollback()
        cleanup_test_users(conn)
        conn.close()


def test_update_user_identity():
    """Test 3: PATCH /api/settings/users/{user_id}/identity updates display fields."""
    print("\n" + "="*60)
    print("TEST 3: PATCH /api/settings/users/{user_id}/identity")
    print("="*60)
    
    conn = get_connection()
    
    try:
        # Cleanup and setup
        cleanup_test_users(conn)
        
        # Create a test user first
        from backend.api.db_layer.user_management_db import insert_user_record, insert_user_role_scope
        
        user_id = insert_user_record(
            conn,
            username="bb7_router_update_test",
            password_hash="HASH",
            display_name="Original Name",
            department_display_name="Original Dept"
        )
        
        cursor = conn.cursor()
        cursor.execute("SELECT TOP 1 UniqueID FROM dbo.AdminsrationUnit WHERE Frozen = 0")
        org_unit_id = cursor.fetchone().UniqueID
        cursor.execute("SELECT TOP 1 RoleID FROM dbo.APP_Roles")
        role_id = cursor.fetchone().RoleID
        cursor.close()
        
        insert_user_role_scope(conn, user_id=user_id, role_id=role_id, org_unit_id=org_unit_id)
        conn.commit()
        
        print(f"✓ Created test user with ID: {user_id}")
        
        # Login as admin
        login_as_software_admin()
        
        # Update identity
        payload = {
            "display_name": "Updated Name",
            "department_display_name": "Updated Dept"
        }
        
        print(f"Updating user identity...")
        
        response = client.patch(f"/api/settings/users/{user_id}/identity", json=payload)
        
        print(f"Status Code: {response.status_code}")
        
        # Assertions
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert data["status"] == "ok", f"Expected status=ok, got {data.get('status')}"
        
        print(f"✓ Update successful")
        
        # Verify in database
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DisplayName, DepartmentDisplayName
            FROM dbo.APP_Users
            WHERE UserID = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        cursor.close()
        
        assert row.DisplayName == payload["display_name"], f"Expected DisplayName={payload['display_name']}, got {row.DisplayName}"
        assert row.DepartmentDisplayName == payload["department_display_name"], f"Expected DepartmentDisplayName={payload['department_display_name']}, got {row.DepartmentDisplayName}"
        
        print(f"✓ Verified update in database")
        print(f"  DisplayName: {row.DisplayName}")
        print(f"  DepartmentDisplayName: {row.DepartmentDisplayName}")
        
        print("\n✓ TEST 3 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 3 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 3 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        clear_sessions()
        conn.rollback()
        cleanup_test_users(conn)
        conn.close()


def test_reset_user_password():
    """Test 4: PATCH /api/settings/users/{user_id}/password resets password."""
    print("\n" + "="*60)
    print("TEST 4: PATCH /api/settings/users/{user_id}/password")
    print("="*60)
    
    conn = get_connection()
    
    try:
        # Cleanup and setup
        cleanup_test_users(conn)
        
        # Create a test user
        from backend.api.db_layer.user_management_db import insert_user_record, insert_user_role_scope
        
        user_id = insert_user_record(
            conn,
            username="bb7_router_password_test",
            password_hash="OLD_HASH",
            display_name="Password Test",
            department_display_name="Test"
        )
        
        cursor = conn.cursor()
        cursor.execute("SELECT TOP 1 UniqueID FROM dbo.AdminsrationUnit WHERE Frozen = 0")
        org_unit_id = cursor.fetchone().UniqueID
        cursor.execute("SELECT TOP 1 RoleID FROM dbo.APP_Roles")
        role_id = cursor.fetchone().RoleID
        cursor.close()
        
        insert_user_role_scope(conn, user_id=user_id, role_id=role_id, org_unit_id=org_unit_id)
        conn.commit()
        
        print(f"✓ Created test user with ID: {user_id}")
        
        # Get original password hash
        cursor = conn.cursor()
        cursor.execute("SELECT PasswordHash FROM dbo.APP_Users WHERE UserID = ?", (user_id,))
        old_hash = cursor.fetchone().PasswordHash
        cursor.close()
        
        print(f"✓ Original hash: {old_hash[:20]}...")
        
        # Login as admin
        login_as_software_admin()
        
        # Reset password
        payload = {
            "new_password": "NewTestPass123!"
        }
        
        print(f"Resetting password...")
        
        response = client.patch(f"/api/settings/users/{user_id}/password", json=payload)
        
        print(f"Status Code: {response.status_code}")
        
        # Assertions
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert data["status"] == "password_updated", f"Expected status=password_updated, got {data.get('status')}"
        
        print(f"✓ Password reset successful")
        
        # Verify password changed in database
        cursor = conn.cursor()
        cursor.execute("SELECT PasswordHash FROM dbo.APP_Users WHERE UserID = ?", (user_id,))
        new_hash = cursor.fetchone().PasswordHash
        cursor.close()
        
        assert new_hash != old_hash, "Password hash did not change"
        assert new_hash.startswith("$2b$"), "New hash is not bcrypt format"
        
        print(f"✓ Verified password changed in database")
        print(f"  Old hash: {old_hash[:20]}...")
        print(f"  New hash: {new_hash[:20]}...")
        
        print("\n✓ TEST 4 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 4 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 4 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        clear_sessions()
        conn.rollback()
        cleanup_test_users(conn)
        conn.close()


def test_delete_user():
    """Test 5: DELETE /api/settings/users/{user_id} deletes user."""
    print("\n" + "="*60)
    print("TEST 5: DELETE /api/settings/users/{user_id}")
    print("="*60)
    
    conn = get_connection()
    
    try:
        # Cleanup and setup
        cleanup_test_users(conn)
        
        # Create a test user
        from backend.api.db_layer.user_management_db import insert_user_record, insert_user_role_scope
        
        user_id = insert_user_record(
            conn,
            username="bb7_router_delete_test",
            password_hash="HASH",
            display_name="Delete Test",
            department_display_name="Test"
        )
        
        cursor = conn.cursor()
        cursor.execute("SELECT TOP 1 UniqueID FROM dbo.AdminsrationUnit WHERE Frozen = 0")
        org_unit_id = cursor.fetchone().UniqueID
        cursor.execute("SELECT TOP 1 RoleID FROM dbo.APP_Roles WHERE RoleCode != 'SOFTWARE_ADMIN'")
        role_id = cursor.fetchone().RoleID
        cursor.close()
        
        insert_user_role_scope(conn, user_id=user_id, role_id=role_id, org_unit_id=org_unit_id)
        conn.commit()
        
        print(f"✓ Created test user with ID: {user_id}")
        
        # Login as admin
        login_as_software_admin()
        
        # Delete user
        print(f"Deleting user...")
        
        response = client.delete(f"/api/settings/users/{user_id}")
        
        print(f"Status Code: {response.status_code}")
        
        # Assertions
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert data["status"] == "deleted", f"Expected status=deleted, got {data.get('status')}"
        
        print(f"✓ Delete successful")
        
        # Verify user removed from database
        cursor = conn.cursor()
        cursor.execute("SELECT UserID FROM dbo.APP_Users WHERE UserID = ?", (user_id,))
        row = cursor.fetchone()
        cursor.close()
        
        assert row is None, "User still exists in database"
        
        print(f"✓ Verified user removed from database")
        
        print("\n✓ TEST 5 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 5 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 5 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        clear_sessions()
        conn.rollback()
        cleanup_test_users(conn)
        conn.close()


def test_guard_create_without_admin():
    """Test 6: POST without SOFTWARE_ADMIN returns 403."""
    print("\n" + "="*60)
    print("TEST 6: Guard Test - POST without SOFTWARE_ADMIN")
    print("="*60)
    
    try:
        # Try to login as regular user
        regular_user_response = login_as_regular_user()
        
        if regular_user_response is None or regular_user_response.status_code != 200:
            print("⚠ No regular user available, skipping test")
            return None  # Skip test
        
        print(f"✓ Logged in as regular user")
        
        # Try to create user
        payload = {
            "username": "should_fail",
            "password": "pass",
            "role_id": 1,
            "org_unit_id": 1
        }
        
        response = client.post("/api/settings/users/", json=payload)
        
        print(f"Status Code: {response.status_code}")
        
        # Should be 403 Forbidden
        assert response.status_code == 403, f"Expected 403, got {response.status_code}"
        
        print(f"✓ Correctly rejected with 403")
        
        print("\n✓ TEST 6 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 6 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 6 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        clear_sessions()


def test_guard_password_reset_without_admin():
    """Test 7: PATCH password without SOFTWARE_ADMIN returns 403."""
    print("\n" + "="*60)
    print("TEST 7: Guard Test - PATCH password without SOFTWARE_ADMIN")
    print("="*60)
    
    try:
        # Try to login as regular user
        regular_user_response = login_as_regular_user()
        
        if regular_user_response is None or regular_user_response.status_code != 200:
            print("⚠ No regular user available, skipping test")
            return None  # Skip test
        
        print(f"✓ Logged in as regular user")
        
        # Try to reset password
        payload = {
            "new_password": "should_fail"
        }
        
        response = client.patch("/api/settings/users/999/password", json=payload)
        
        print(f"Status Code: {response.status_code}")
        
        # Should be 403 Forbidden
        assert response.status_code == 403, f"Expected 403, got {response.status_code}"
        
        print(f"✓ Correctly rejected with 403")
        
        print("\n✓ TEST 7 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 7 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 7 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        clear_sessions()


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*60)
    print("PHASE B — B-B7 — ROUTER INTEGRATION TEST SUITE")
    print("="*60)
    
    results = []
    
    # Test 1: GET users list
    results.append(("GET /api/settings/users", test_get_users_list()))
    
    # Test 2: POST create user
    test2_result, user_id = test_create_user()
    results.append(("POST /api/settings/users", test2_result))
    
    # Test 3: PATCH update identity
    results.append(("PATCH /identity", test_update_user_identity()))
    
    # Test 4: PATCH reset password
    results.append(("PATCH /password", test_reset_user_password()))
    
    # Test 5: DELETE user
    results.append(("DELETE /user", test_delete_user()))
    
    # Test 6: Guard - POST without admin
    guard1_result = test_guard_create_without_admin()
    if guard1_result is not None:
        results.append(("Guard: POST without admin", guard1_result))
    
    # Test 7: Guard - PATCH password without admin
    guard2_result = test_guard_password_reset_without_admin()
    if guard2_result is not None:
        results.append(("Guard: PATCH password without admin", guard2_result))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    total = len(results)
    
    for test_name, result in results:
        if result is True:
            status = "✓ PASSED"
        elif result is False:
            status = "✗ FAILED"
        else:
            status = "⚠ SKIPPED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        return True
    else:
        print(f"\n⚠️ {failed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
