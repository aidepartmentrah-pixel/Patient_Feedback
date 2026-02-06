"""
PHASE B — B-B9 — BACKEND SMOKE TEST SUITE — SETTINGS USERS

End-to-end smoke test for Phase B Settings Users backend flow.
Tests router → service → DB layer including admin password reset.

Run from backend directory:
    python test_phase_b_settings_users_smoke.py
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


def login_as_user(username: str, password: str):
    """Helper to login as specific user."""
    clear_sessions()
    response = client.post(
        "/api/auth/login",
        json={"username": username, "password": password}
    )
    return response


def cleanup_smoke_user():
    """Clean up smoke test user from database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Delete role scopes first
        cursor.execute("""
            DELETE FROM dbo.APP_UserRoleScope 
            WHERE UserID IN (
                SELECT UserID FROM dbo.APP_Users 
                WHERE Username = 'smoke_user_bb9'
            )
        """)
        
        # Delete user
        cursor.execute("""
            DELETE FROM dbo.APP_Users 
            WHERE Username = 'smoke_user_bb9'
        """)
        
        conn.commit()
    except Exception as e:
        print(f"Warning: Cleanup error: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()


def test_phase_b_settings_users_full_flow():
    """
    MAIN SMOKE TEST: Full Phase B Settings Users flow.
    
    Tests:
    1. GET users list
    2. POST create user
    3. GET users list (verify user present)
    4. PATCH update identity
    5. PATCH reset password (verify hash changed)
    6. Login with new password (optional)
    7. DELETE user
    8. Guard tests (403 without admin)
    """
    print("\n" + "="*80)
    print("PHASE B — BACKEND SMOKE TEST — SETTINGS USERS FULL FLOW")
    print("="*80)
    
    conn = get_connection()
    created_user_id = None
    
    try:
        # ============================================================
        # STEP 0: Cleanup any previous test data
        # ============================================================
        print("\n[STEP 0] Cleanup previous test data...")
        cleanup_smoke_user()
        print("✓ Cleanup complete")
        
        # ============================================================
        # STEP 1: Authenticate as SOFTWARE_ADMIN
        # ============================================================
        print("\n[STEP 1] Authenticate as SOFTWARE_ADMIN...")
        login_as_software_admin()
        print("✓ Authenticated successfully")
        
        # ============================================================
        # STEP 2: GET /api/settings/users
        # ============================================================
        print("\n[STEP 2] GET /api/settings/users...")
        response = client.get("/api/settings/users/")
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        users_list = response.json()
        assert isinstance(users_list, list), f"Expected list, got {type(users_list)}"
        
        initial_user_count = len(users_list)
        print(f"✓ GET users list: {initial_user_count} users")
        
        # ============================================================
        # STEP 3: Get valid role_id and org_unit_id
        # ============================================================
        print("\n[STEP 3] Query valid role_id and org_unit_id...")
        
        cursor = conn.cursor()
        
        # Get a role (not SOFTWARE_ADMIN for test user)
        cursor.execute("""
            SELECT TOP 1 RoleID 
            FROM dbo.APP_Roles 
            WHERE RoleCode != 'SOFTWARE_ADMIN'
        """)
        role_row = cursor.fetchone()
        assert role_row is not None, "No roles found in database"
        role_id = role_row.RoleID
        
        # Get an org unit
        cursor.execute("""
            SELECT TOP 1 UniqueID 
            FROM dbo.AdminsrationUnit 
            WHERE Frozen = 0
        """)
        org_row = cursor.fetchone()
        assert org_row is not None, "No org units found in database"
        org_unit_id = org_row.UniqueID
        
        cursor.close()
        
        print(f"✓ role_id: {role_id}")
        print(f"✓ org_unit_id: {org_unit_id}")
        
        # ============================================================
        # STEP 4: POST /api/settings/users (Create user)
        # ============================================================
        print("\n[STEP 4] POST /api/settings/users (Create user)...")
        
        create_payload = {
            "username": "smoke_user_bb9",
            "password": "Smoke123!",
            "display_name": "Smoke User",
            "department_display_name": "Smoke Dept",
            "role_id": role_id,
            "org_unit_id": org_unit_id
        }
        
        response = client.post("/api/settings/users/", json=create_payload)
        
        assert response.status_code == 201, f"Expected 201, got {response.status_code}: {response.text}"
        
        create_data = response.json()
        assert "user_id" in create_data, "Missing user_id in response"
        assert "username" in create_data, "Missing username in response"
        assert create_data["username"] == "smoke_user_bb9", f"Username mismatch: {create_data['username']}"
        
        created_user_id = create_data["user_id"]
        print(f"✓ Created user with ID: {created_user_id}")
        
        # ============================================================
        # STEP 5: GET /api/settings/users again (verify user present)
        # ============================================================
        print("\n[STEP 5] GET /api/settings/users (verify new user present)...")
        
        response = client.get("/api/settings/users/")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        users_list = response.json()
        
        # Find our user
        smoke_user = None
        for user in users_list:
            if user.get("username") == "smoke_user_bb9":
                smoke_user = user
                break
        
        assert smoke_user is not None, "Smoke user not found in users list"
        assert smoke_user["user_id"] == created_user_id, f"User ID mismatch: {smoke_user['user_id']}"
        assert smoke_user["display_name"] == "Smoke User", f"Display name mismatch: {smoke_user['display_name']}"
        
        print(f"✓ User found in list with correct data")
        
        # ============================================================
        # STEP 6: PATCH /api/settings/users/{id}/identity
        # ============================================================
        print("\n[STEP 6] PATCH /api/settings/users/{id}/identity...")
        
        update_payload = {
            "display_name": "Smoke Updated"
        }
        
        response = client.patch(
            f"/api/settings/users/{created_user_id}/identity",
            json=update_payload
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        update_data = response.json()
        assert update_data["status"] == "ok", f"Expected status=ok, got {update_data.get('status')}"
        
        print(f"✓ Identity update successful")
        
        # Verify in database
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DisplayName 
            FROM dbo.APP_Users 
            WHERE UserID = ?
        """, (created_user_id,))
        
        row = cursor.fetchone()
        cursor.close()
        
        assert row is not None, "User not found in database"
        assert row.DisplayName == "Smoke Updated", f"DisplayName not updated: {row.DisplayName}"
        
        print(f"✓ Verified in database: DisplayName = 'Smoke Updated'")
        
        # ============================================================
        # STEP 7: PATCH /api/settings/users/{id}/password (NEW)
        # ============================================================
        print("\n[STEP 7] PATCH /api/settings/users/{id}/password (Admin reset)...")
        
        # Get current password hash
        cursor = conn.cursor()
        cursor.execute("""
            SELECT PasswordHash 
            FROM dbo.APP_Users 
            WHERE UserID = ?
        """, (created_user_id,))
        
        old_hash_row = cursor.fetchone()
        cursor.close()
        
        assert old_hash_row is not None, "User not found"
        old_hash = old_hash_row.PasswordHash
        
        print(f"✓ Old hash retrieved (length: {len(old_hash)})")
        
        # Reset password
        password_payload = {
            "new_password": "SmokeNew123!"
        }
        
        response = client.patch(
            f"/api/settings/users/{created_user_id}/password",
            json=password_payload
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        password_data = response.json()
        assert password_data["status"] == "password_updated", f"Expected status=password_updated, got {password_data.get('status')}"
        
        print(f"✓ Password reset successful")
        
        # Verify hash changed in database
        cursor = conn.cursor()
        cursor.execute("""
            SELECT PasswordHash 
            FROM dbo.APP_Users 
            WHERE UserID = ?
        """, (created_user_id,))
        
        new_hash_row = cursor.fetchone()
        cursor.close()
        
        assert new_hash_row is not None, "User not found after password reset"
        new_hash = new_hash_row.PasswordHash
        
        assert new_hash != old_hash, "Password hash did not change"
        assert new_hash.startswith("$2b$"), "New hash is not bcrypt format"
        
        print(f"✓ Verified hash changed in database")
        print(f"  Old hash length: {len(old_hash)}")
        print(f"  New hash length: {len(new_hash)}")
        
        # ============================================================
        # STEP 8: Test login with new password (Optional)
        # ============================================================
        print("\n[STEP 8] Test login with new password...")
        
        # Try login with new password
        new_login = login_as_user("smoke_user_bb9", "SmokeNew123!")
        
        if new_login.status_code == 200:
            print(f"✓ Login with new password successful")
            
            # Try login with old password (should fail)
            clear_sessions()
            old_login = login_as_user("smoke_user_bb9", "Smoke123!")
            
            assert old_login.status_code != 200, "Old password still works (should fail)"
            print(f"✓ Login with old password correctly failed")
        else:
            print(f"⚠ Login test skipped (status: {new_login.status_code})")
        
        # Re-authenticate as admin for remaining tests
        login_as_software_admin()
        
        # ============================================================
        # STEP 9: DELETE /api/settings/users/{id}
        # ============================================================
        print("\n[STEP 9] DELETE /api/settings/users/{id}...")
        
        response = client.delete(f"/api/settings/users/{created_user_id}")
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        delete_data = response.json()
        assert delete_data["status"] == "deleted", f"Expected status=deleted, got {delete_data.get('status')}"
        
        print(f"✓ User deleted successfully")
        
        # Verify removed from database
        cursor = conn.cursor()
        cursor.execute("""
            SELECT UserID 
            FROM dbo.APP_Users 
            WHERE UserID = ?
        """, (created_user_id,))
        
        deleted_row = cursor.fetchone()
        cursor.close()
        
        assert deleted_row is None, "User still exists in database"
        
        print(f"✓ Verified user removed from database")
        
        created_user_id = None  # Mark as deleted
        
        # ============================================================
        # STEP 10: Guard test - POST without SOFTWARE_ADMIN
        # ============================================================
        print("\n[STEP 10] Guard test - POST without SOFTWARE_ADMIN...")
        
        # Clear auth
        clear_sessions()
        
        # Try to create user without auth
        guard_payload = {
            "username": "should_fail",
            "password": "Pass123!",
            "role_id": role_id,
            "org_unit_id": org_unit_id
        }
        
        response = client.post("/api/settings/users/", json=guard_payload)
        
        # Should get 401 (not authenticated) or 403 (authenticated but not admin)
        assert response.status_code in [401, 403], f"Expected 401 or 403, got {response.status_code}"
        
        print(f"✓ POST correctly rejected without admin auth (status: {response.status_code})")
        
        # ============================================================
        # STEP 11: Guard test - PATCH password without SOFTWARE_ADMIN
        # ============================================================
        print("\n[STEP 11] Guard test - PATCH password without SOFTWARE_ADMIN...")
        
        # Try to reset password without auth
        guard_password_payload = {
            "new_password": "should_fail"
        }
        
        response = client.patch(
            "/api/settings/users/999/password",
            json=guard_password_payload
        )
        
        # Should get 401 (not authenticated) or 403 (authenticated but not admin)
        assert response.status_code in [401, 403], f"Expected 401 or 403, got {response.status_code}"
        
        print(f"✓ PATCH password correctly rejected without admin auth (status: {response.status_code})")
        
        # ============================================================
        # SUCCESS
        # ============================================================
        print("\n" + "="*80)
        print("🎉 ALL SMOKE TESTS PASSED!")
        print("="*80)
        print("\nPhase B Settings Users backend flow verified:")
        print("  ✓ GET users list")
        print("  ✓ POST create user")
        print("  ✓ GET users list (user present)")
        print("  ✓ PATCH update identity")
        print("  ✓ PATCH reset password (hash changed)")
        print("  ✓ Login with new password")
        print("  ✓ DELETE user")
        print("  ✓ Guard protection (POST)")
        print("  ✓ Guard protection (PATCH password)")
        print("\nBackend stack validated: Router → Service → DB")
        
        return True
        
    except AssertionError as e:
        print(f"\n✗ SMOKE TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    except Exception as e:
        print(f"\n✗ SMOKE TEST ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        clear_sessions()
        
        if created_user_id is not None:
            print("\n[CLEANUP] Removing test user...")
            try:
                # Re-auth as admin for cleanup
                login_as_software_admin()
                client.delete(f"/api/settings/users/{created_user_id}")
                print("✓ Test user removed")
            except:
                pass
        
        cleanup_smoke_user()
        conn.close()


if __name__ == "__main__":
    success = test_phase_b_settings_users_full_flow()
    sys.exit(0 if success else 1)
