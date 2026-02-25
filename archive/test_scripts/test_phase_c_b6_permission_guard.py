"""
Phase C - B-C6 Test Suite: Permission Guard Hardening

Tests the get_current_software_admin dependency guard for section creation endpoint.

Test Coverage:
1. Authorized access - SOFTWARE_ADMIN can create sections
2. Unauthorized access - WORKER role gets 403 Forbidden
3. Unauthorized access - SECTION_ADMIN role gets 403 Forbidden
4. Missing authentication - No token gets 401 Unauthorized
5. Invalid token - Bad token gets 401 Unauthorized
6. Department admin access - DEPARTMENT_ADMIN gets 403 (if role exists)
7. Guard is applied at dependency level (not in function body)
8. Error messages are clear and informative

Expected: 100% pass rate
"""

import sys
import os
import requests
from typing import Dict, Optional

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))

from core.database import get_connection
from api.constants.org_unit_types import ORG_TYPE_DEPARTMENT

# Test configuration
BASE_URL = "http://localhost:8000"
SECTION_CREATE_ENDPOINT = f"{BASE_URL}/api/admin/create-section-with-admin"


def cleanup_test_section(section_name: str):
    """Remove test section and associated users"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get section ID
        cursor.execute(
            "SELECT UniqueID FROM AdminsrationUnit WHERE Name = ?",
            (section_name,)
        )
        row = cursor.fetchone()
        if row:
            section_id = row[0]
            
            # Delete APP_UserRoleScope entries
            cursor.execute(
                "DELETE FROM APP_UserRoleScope WHERE OrgUnitID = ?",
                (section_id,)
            )
            
            # Delete APP_Users
            cursor.execute(
                """
                DELETE FROM APP_Users 
                WHERE username LIKE ?
                """,
                (f"%{section_name[:10]}%",)
            )
            
            # Delete section
            cursor.execute(
                "DELETE FROM AdminsrationUnit WHERE UniqueID = ?",
                (section_id,)
            )
            
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Cleanup warning: {e}")


def get_valid_parent_department():
    """Get a valid department ID for testing"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get first department (type 325)
    cursor.execute(
        "SELECT TOP 1 UniqueID FROM AdminsrationUnit WHERE Type = ?",
        (ORG_TYPE_DEPARTMENT,)
    )
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise Exception("No department found for testing")
    
    return row[0]


def get_user_by_role(role_code: str) -> Optional[Dict]:
    """Get a test user with specific role"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get user with specific role
        cursor.execute(
            """
            SELECT TOP 1 u.UserID, u.Username
            FROM APP_Users u
            INNER JOIN APP_UserRoleScope urs ON u.UserID = urs.UserID
            INNER JOIN APP_Roles r ON urs.RoleID = r.RoleID
            WHERE r.RoleCode = ? AND u.IsActive = 1
            """,
            (role_code,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "user_id": row[0],
                "username": row[1],
                "password": "Hospital2026!"  # Known test password
            }
        return None
        
    except Exception as e:
        print(f"Error getting user: {e}")
        return None


def login_user(username: str, password: str) -> Optional[requests.Session]:
    """Login and return authenticated session"""
    try:
        print(f"  Attempting login: {username} / {password}")
        session = requests.Session()
        response = session.post(
            f"{BASE_URL}/api/auth/login",
            json={
                "username": username,
                "password": password
            }
        )
        
        print(f"  Login response: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print(f"  Login successful - session established")
                print(f"  Cookies: {session.cookies}")
                return session
            else:
                print(f"  Login failed: {data}")
                return None
        else:
            print(f"  Login failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"  Login error: {e}")
        return None


def test_1_authorized_software_admin_success():
    """Test 1: SOFTWARE_ADMIN can create sections (200 OK)"""
    print("\n=== Test 1: Authorized SOFTWARE_ADMIN ===")
    
    section_name = "Test_B6_AuthorizedAdmin"
    cleanup_test_section(section_name)
    
    try:
        # Use known SOFTWARE_ADMIN credentials
        username = "software_admin"
        password = "admin123"
        
        # Login
        session = login_user(username, password)
        if not session:
            print("FAIL: Test 1 - Login failed")
            return False
        
        # Call endpoint
        parent_id = get_valid_parent_department()
        print(f"  Making request with cookies: {session.cookies}")
        response = session.post(
            SECTION_CREATE_ENDPOINT,
            json={
                "section_name": section_name,
                "parent_unit_id": parent_id
            }
        )
        print(f"  Response: {response.status_code}")
        
        # Verify success
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "section_id" in data, "Missing section_id"
        assert "username" in data, "Missing username"
        assert data["section_name"] == section_name, "Section name mismatch"
        
        print(f"✓ SOFTWARE_ADMIN authorized successfully")
        print(f"✓ Section created: ID={data['section_id']}, Admin={data['username']}")
        print("PASS: Test 1")
        return True
        
    except AssertionError as e:
        print(f"FAIL: Test 1 - {e}")
        return False
    except Exception as e:
        print(f"FAIL: Test 1 - {e}")
        return False
    finally:
        cleanup_test_section(section_name)


def test_2_unauthorized_worker_forbidden():
    """Test 2: WORKER role gets 403 Forbidden"""
    print("\n=== Test 2: Unauthorized WORKER ===")
    
    section_name = "Test_B6_WorkerAttempt"
    
    try:
        # Use known WORKER credentials
        username = "worker"
        password = "worker123"
        
        # Login
        session = login_user(username, password)
        if not session:
            print("FAIL: Test 2 - Login failed")
            return False
        
        # Call endpoint
        parent_id = get_valid_parent_department()
        print(f"  Making request with cookies: {session.cookies}")
        response = session.post(
            SECTION_CREATE_ENDPOINT,
            json={
                "section_name": section_name,
                "parent_unit_id": parent_id
            }
        )
        print(f"  Response: {response.status_code}")
        
        # Verify forbidden
        assert response.status_code == 403, f"Expected 403, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "detail" in data, "Missing detail field"
        detail = data["detail"]
        
        # Verify error structure
        if isinstance(detail, dict):
            assert "error" in detail, "Missing error field"
            assert detail["error"] == "FORBIDDEN", f"Wrong error code: {detail.get('error')}"
            assert "required_roles" in detail, "Missing required_roles field"
            assert "SOFTWARE_ADMIN" in detail["required_roles"], "SOFTWARE_ADMIN not in required roles"
        
        print(f"✓ WORKER correctly denied (403 Forbidden)")
        print(f"✓ Error message: {detail}")
        print("PASS: Test 2")
        return True
        
    except AssertionError as e:
        print(f"FAIL: Test 2 - {e}")
        return False
    except Exception as e:
        print(f"FAIL: Test 2 - {e}")
        return False


def test_3_unauthorized_section_admin_forbidden():
    """Test 3: SECTION_ADMIN role gets 403 Forbidden"""
    print("\n=== Test 3: Unauthorized SECTION_ADMIN ===")
    
    section_name = "Test_B6_SectionAdminAttempt"
    
    try:
        # Use known SECTION_ADMIN credentials
        username = "section_admin"
        password = "section123"
        session = login_user(username, password)
        if not session:
            print("FAIL: Test 3 - Login failed")
            return False
        
        # Call endpoint
        parent_id = get_valid_parent_department()
        response = session.post(
            SECTION_CREATE_ENDPOINT,
            json={
                "section_name": section_name,
                "parent_unit_id": parent_id
            }
        )    "section_name": section_name,
                "parent_unit_id": parent_id
            }
        )
        
        # Verify forbidden
        assert response.status_code == 403, f"Expected 403, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "detail" in data, "Missing detail field"
        
        print(f"✓ SECTION_ADMIN correctly denied (403 Forbidden)")
        print(f"✓ Error: {data['detail']}")
        print("PASS: Test 3")
        return True
        
    except AssertionError as e:
        print(f"FAIL: Test 3 - {e}")
        return False
    except Exception as e:
        print(f"FAIL: Test 3 - {e}")
        return False


def test_4_missing_auth_unauthorized():
    """Test 4: No authentication token gets 401 Unauthorized"""
    print("\n=== Test 4: Missing Authentication ===")
    
    section_name = "Test_B6_NoAuth"
    
    try:
        # Call endpoint without token
        parent_id = get_valid_parent_department()
        response = requests.post(
            SECTION_CREATE_ENDPOINT,
            json={
                "section_name": section_name,
                "parent_unit_id": parent_id
            }
            # No Authorization header
        )
        
        # Verify unauthorized
        assert response.status_code == 401, f"Expected 401, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "detail" in data, "Missing detail field"
        
        print(f"✓ Missing auth correctly rejected (401 Unauthorized)")
        print(f"✓ Error: {data['detail']}")
        print("PASS: Test 4")
        return True
        
    except AssertionError as e:
        print(f"FAIL: Test 4 - {e}")
        return False
    except Exception as e:
        print(f"FAIL: Test 4 - {e}")
        return False


def test_5_invalid_token_unauthorized():
    """Test 5: Invalid authentication token gets 401 Unauthorized"""
    print("\n=== Test 5: Invalid Token ===")
    
    section_name = "Test_B6_BadToken"
    
    try:
        # Call endpoint with invalid token
        parent_id = get_valid_parent_department()
        response = requests.post(
            SECTION_CREATE_ENDPOINT,
            json={
                "section_name": section_name,
                "parent_unit_id": parent_id
            },
            headers={"Authorization": "Bearer invalid_token_12345"}
        )
        
        # Verify unauthorized
        assert response.status_code == 401, f"Expected 401, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "detail" in data, "Missing detail field"
        
        print(f"✓ Invalid token correctly rejected (401 Unauthorized)")
        print(f"✓ Error: {data['detail']}")
        print("PASS: Test 5")
        return True
        
    except AssertionError as e:
        print(f"FAIL: Test 5 - {e}")
        return False
    except Exception as e:
        print(f"FAIL: Test 5 - {e}")
        return False


def test_6_department_admin_forbidden():
    """Test 6: DEPARTMENT_ADMIN role gets 403 Forbidden (if role exists)"""
    print("\n=== Test 6: Unauthorized DEPARTMENT_ADMIN ===")
    
    section_name = "Test_B6_DeptAdminAttempt"
    
    try:
        # Use known DEPARTMENT_ADMIN credentials
        username = "department_admin"
        password = "dept123"
        
        # Login
        session = login_user(username, password)
        if not session:
            print("FAIL: Test 6 - Login failed")
            return False
        
        # Call endpoint
        parent_id = get_valid_parent_department()
        response = session.post(
            SECTION_CREATE_ENDPOINT,
            json={
                "section_name": section_name,
                "parent_unit_id": parent_id
            }
        )
        
        # Verify forbidden
        assert response.status_code == 403, f"Expected 403, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "detail" in data, "Missing detail field"
        
        print(f"✓ DEPARTMENT_ADMIN correctly denied (403 Forbidden)")
        print(f"✓ Error: {data['detail']}")
        print("PASS: Test 6")
        return True
        
    except AssertionError as e:
        print(f"FAIL: Test 6 - {e}")
        return False
    except Exception as e:
        print(f"FAIL: Test 6 - {e}")
        return False


def test_7_guard_dependency_level():
    """Test 7: Guard is applied at FastAPI dependency level"""
    print("\n=== Test 7: Guard at Dependency Level ===")
    
    try:
        # Read router file
        router_path = os.path.join(
            os.path.dirname(__file__),
            "backend", "api", "routers", "admin_section_router.py"
        )
        
        with open(router_path, 'r', encoding='utf-8') as f:
            router_content = f.read()
        
        # Verify guard import
        assert "from ..utils.guards import get_current_software_admin" in router_content, \
            "Missing get_current_software_admin import"
        
        # Verify Depends usage in route
        assert "Depends(get_current_software_admin)" in router_content, \
            "Guard not applied as FastAPI dependency"
        
        # Verify NOT using old pattern (manual check in body)
        assert "require_software_admin(current_user)" not in router_content, \
            "Old pattern found - should use dependency guard"
        
        # Verify function parameter
        lines = router_content.split('\n')
        depends_line = None
        for line in lines:
            if "Depends(get_current_software_admin)" in line:
                depends_line = line
                break
        
        assert depends_line is not None, "Could not find Depends line"
        assert "CurrentUser" in depends_line, "Dependency should return CurrentUser"
        
        print(f"✓ Guard correctly applied as FastAPI dependency")
        print(f"✓ Using: Depends(get_current_software_admin)")
        print(f"✓ Not using old pattern (manual body check)")
        print("PASS: Test 7")
        return True
        
    except AssertionError as e:
        print(f"FAIL: Test 7 - {e}")
        return False
    except Exception as e:
        print(f"FAIL: Test 7 - {e}")
        return False


def test_8_error_messages_informative():
    """Test 8: Error messages are clear and informative"""
    print("\n=== Test 8: Error Message Quality ===")
    
    try:
        # Test 401 error message
        response_401 = requests.post(
            SECTION_CREATE_ENDPOINT,
            json={
                "section_name": "test",
                "parent_unit_id": 1
            }
        )
        
        assert response_401.status_code == 401
        data_401 = response_401.json()
        assert "detail" in data_401, "Missing detail in 401 response"
        
        # Test 403 error message (use WORKER if available)
        username = "worker"
        password = "worker123"
        token = login_user(username, password)
        session = login_user(username, password)
        if session:
            response_403 = session.post(
                SECTION_CREATE_ENDPOINT,
                json={
                    "section_name": "test",
                    "parent_unit_id": 1
                
            
            assert response_403.status_code == 403
            data_403 = response_403.json()
            assert "detail" in data_403, "Missing detail in 403 response"
            
            detail = data_403["detail"]
            if isinstance(detail, dict):
                # Check for structured error
                assert "error" in detail, "Missing error field in 403"
                assert "message" in detail or "required_roles" in detail, \
                    "403 should have helpful message or required_roles"
                
                print(f"✓ 403 error structure: {detail}")
        
        print(f"✓ 401 error message present")
        print(f"✓ Error messages are informative")
        print("PASS: Test 8")
        return True
        
    except AssertionError as e:
        print(f"FAIL: Test 8 - {e}")
        return False
    except Exception as e:
        print(f"FAIL: Test 8 - {e}")
        return False


def run_all_tests():
    """Run all B-C6 tests"""
    print("="*70)
    print("Phase C - B-C6 Test Suite: Permission Guard Hardening")
    print("="*70)
    print("\nNOTE: Tests require running FastAPI server at http://localhost:8000")
    print("      Start server: uvicorn main:app --reload")
    print()
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=2)
    except Exception:
        print("ERROR: FastAPI server not running at http://localhost:8000")
        print("       Start server first: uvicorn main:app --reload")
        return 1
    
    tests = [
        test_1_authorized_software_admin_success,
        test_2_unauthorized_worker_forbidden,
        test_3_unauthorized_section_admin_forbidden,
        test_4_missing_auth_unauthorized,
        test_5_invalid_token_unauthorized,
        test_6_department_admin_forbidden,
        test_7_guard_dependency_level,
        test_8_error_messages_informative
    ]
    
    results = []
    for test_func in tests:
        result = test_func()
        results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    # Count only True/False (excluding None for skipped tests)
    valid_results = [r for r in results if r is not None]
    passed = sum(1 for r in valid_results if r)
    skipped = sum(1 for r in results if r is None)
    total = len(valid_results)
    
    print(f"Passed: {passed}/{total}")
    if skipped > 0:
        print(f"Skipped: {skipped} (missing test users)")
    
    if passed == total and total > 0:
        print("\n✓ ALL TESTS PASSED - B-C6 COMPLETE")
        return 0
    else:
        print(f"\n✗ {total - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    exit(run_all_tests())
