"""
Phase C - B-C5 Test Suite: Optional Section-Only Creation Path

Tests the create_admin parameter in create_section_with_admin service.

Test Coverage:
1. Default behavior (create_admin=True) - section + admin created
2. Section-only path (create_admin=False) - section created, no admin
3. Response structure validation for both paths
4. Database verification for both scenarios
5. Rollback handling with create_admin=False
6. Router behavior unchanged (always uses default)

Expected: 100% pass rate
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))

from api.services.section_admin_creator_service import (
    create_section_with_admin,
    create_section_core,
    attach_section_admin_user
)
from core.database import get_connection
from api.constants.org_unit_types import ORG_TYPE_SECTION


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
        "SELECT TOP 1 UniqueID FROM AdminsrationUnit WHERE Type = 325"
    )
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise Exception("No department found for testing")
    
    return row[0]


def test_1_default_behavior_creates_both():
    """Test 1: Default behavior (create_admin=True) creates section + admin"""
    print("\n=== Test 1: Default behavior (create_admin=True) ===")
    
    section_name = "Test_B5_DefaultBehavior"
    cleanup_test_section(section_name)
    
    try:
        parent_id = get_valid_parent_department()
        
        # Call with explicit create_admin=True
        result = create_section_with_admin(
            section_name=section_name,
            parent_department_id=parent_id,
            create_admin=True
        )
        
        # Verify response structure
        assert "section_id" in result, "Missing section_id"
        assert "section_name" in result, "Missing section_name"
        assert "parent_unit_id" in result, "Missing parent_unit_id"
        assert "username" in result, "Missing username"
        assert "temp_password" in result, "Missing temp_password"
        
        # Verify values
        assert result["section_id"] > 0, "Invalid section_id"
        assert result["section_name"] == section_name, "Section name mismatch"
        assert result["parent_unit_id"] == parent_id, "Parent ID mismatch"
        assert result["username"] is not None, "Username should not be None"
        assert result["temp_password"] is not None, "Temp password should not be None"
        
        # Verify database: section created
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT UniqueID, Name, ParentID, Type FROM AdminsrationUnit WHERE UniqueID = ?",
            (result["section_id"],)
        )
        section_row = cursor.fetchone()
        assert section_row is not None, "Section not found in DB"
        assert section_row[1] == section_name, "DB section name mismatch"
        assert section_row[2] == parent_id, "DB parent ID mismatch"
        assert section_row[3] == ORG_TYPE_SECTION, "DB type not SECTION"
        
        # Verify database: admin user created
        cursor.execute(
            "SELECT UserID, username FROM APP_Users WHERE username = ?",
            (result["username"],)
        )
        user_row = cursor.fetchone()
        assert user_row is not None, "Admin user not found in DB"
        
        # Verify database: role scope entry
        cursor.execute(
            "SELECT UserID FROM APP_UserRoleScope WHERE UserID = ? AND OrgUnitID = ?",
            (user_row[0], result["section_id"])
        )
        scope_row = cursor.fetchone()
        assert scope_row is not None, "Role scope entry not found"
        
        conn.close()
        
        print(f"✓ Section created: ID={result['section_id']}")
        print(f"✓ Admin created: {result['username']}")
        print(f"✓ Database verification passed")
        print("PASS: Test 1")
        return True
        
    except Exception as e:
        print(f"FAIL: Test 1 - {e}")
        return False
    finally:
        cleanup_test_section(section_name)


def test_2_section_only_no_admin():
    """Test 2: Section-only path (create_admin=False) creates no admin"""
    print("\n=== Test 2: Section-only path (create_admin=False) ===")
    
    section_name = "Test_B5_SectionOnly"
    cleanup_test_section(section_name)
    
    try:
        parent_id = get_valid_parent_department()
        
        # Call with create_admin=False
        result = create_section_with_admin(
            section_name=section_name,
            parent_department_id=parent_id,
            create_admin=False
        )
        
        # Verify response structure (same keys)
        assert "section_id" in result, "Missing section_id"
        assert "section_name" in result, "Missing section_name"
        assert "parent_unit_id" in result, "Missing parent_unit_id"
        assert "username" in result, "Missing username"
        assert "temp_password" in result, "Missing temp_password"
        
        # Verify values (credentials should be None)
        assert result["section_id"] > 0, "Invalid section_id"
        assert result["section_name"] == section_name, "Section name mismatch"
        assert result["parent_unit_id"] == parent_id, "Parent ID mismatch"
        assert result["username"] is None, "Username should be None when create_admin=False"
        assert result["temp_password"] is None, "Temp password should be None when create_admin=False"
        
        # Verify database: section created
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT UniqueID, Name, ParentID, Type FROM AdminsrationUnit WHERE UniqueID = ?",
            (result["section_id"],)
        )
        section_row = cursor.fetchone()
        assert section_row is not None, "Section not found in DB"
        assert section_row[1] == section_name, "DB section name mismatch"
        assert section_row[2] == parent_id, "DB parent ID mismatch"
        assert section_row[3] == ORG_TYPE_SECTION, "DB type not SECTION"
        
        # Verify database: NO admin user created
        cursor.execute(
            "SELECT COUNT(*) FROM APP_Users WHERE username LIKE ?",
            (f"%{section_name[:10]}%",)
        )
        user_count = cursor.fetchone()[0]
        assert user_count == 0, f"Admin user should not exist (found {user_count})"
        
        # Verify database: NO role scope entry
        cursor.execute(
            "SELECT COUNT(*) FROM APP_UserRoleScope WHERE OrgUnitID = ?",
            (result["section_id"],)
        )
        scope_count = cursor.fetchone()[0]
        assert scope_count == 0, f"Role scope entry should not exist (found {scope_count})"
        
        conn.close()
        
        print(f"✓ Section created: ID={result['section_id']}")
        print(f"✓ No admin user created (username=None, temp_password=None)")
        print(f"✓ Database verification passed")
        print("PASS: Test 2")
        return True
        
    except Exception as e:
        print(f"FAIL: Test 2 - {e}")
        return False
    finally:
        cleanup_test_section(section_name)


def test_3_default_parameter_value():
    """Test 3: Default parameter value (omit create_admin) creates both"""
    print("\n=== Test 3: Default parameter value (omit create_admin) ===")
    
    section_name = "Test_B5_DefaultParam"
    cleanup_test_section(section_name)
    
    try:
        parent_id = get_valid_parent_department()
        
        # Call WITHOUT create_admin parameter (should default to True)
        result = create_section_with_admin(
            section_name=section_name,
            parent_department_id=parent_id
            # create_admin parameter omitted - defaults to True
        )
        
        # Verify credentials exist (admin was created)
        assert result["username"] is not None, "Username should not be None (default create_admin=True)"
        assert result["temp_password"] is not None, "Temp password should not be None (default create_admin=True)"
        
        # Verify database: admin user exists
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT UserID FROM APP_Users WHERE username = ?",
            (result["username"],)
        )
        user_row = cursor.fetchone()
        assert user_row is not None, "Admin user should exist (default create_admin=True)"
        conn.close()
        
        print(f"✓ Default parameter correctly creates admin")
        print(f"✓ Admin username: {result['username']}")
        print("PASS: Test 3")
        return True
        
    except Exception as e:
        print(f"FAIL: Test 3 - {e}")
        return False
    finally:
        cleanup_test_section(section_name)


def test_4_response_consistency():
    """Test 4: Response structure consistent for both paths"""
    print("\n=== Test 4: Response structure consistency ===")
    
    section_name_with_admin = "Test_B5_WithAdmin"
    section_name_without_admin = "Test_B5_WithoutAdmin"
    
    cleanup_test_section(section_name_with_admin)
    cleanup_test_section(section_name_without_admin)
    
    try:
        parent_id = get_valid_parent_department()
        
        # Create with admin
        result_with = create_section_with_admin(
            section_name=section_name_with_admin,
            parent_department_id=parent_id,
            create_admin=True
        )
        
        # Create without admin
        result_without = create_section_with_admin(
            section_name=section_name_without_admin,
            parent_department_id=parent_id,
            create_admin=False
        )
        
        # Verify both have same keys
        assert set(result_with.keys()) == set(result_without.keys()), "Response keys mismatch"
        
        expected_keys = {"section_id", "section_name", "parent_unit_id", "username", "temp_password"}
        assert set(result_with.keys()) == expected_keys, "Response keys incorrect"
        
        # Verify differences
        assert result_with["username"] is not None, "With admin: username should not be None"
        assert result_without["username"] is None, "Without admin: username should be None"
        assert result_with["temp_password"] is not None, "With admin: temp_password should not be None"
        assert result_without["temp_password"] is None, "Without admin: temp_password should be None"
        
        print(f"✓ Response structure consistent across both paths")
        print(f"✓ Keys: {expected_keys}")
        print("PASS: Test 4")
        return True
        
    except Exception as e:
        print(f"FAIL: Test 4 - {e}")
        return False
    finally:
        cleanup_test_section(section_name_with_admin)
        cleanup_test_section(section_name_without_admin)


def test_5_transaction_rollback_section_only():
    """Test 5: Transaction rollback works for section-only path"""
    print("\n=== Test 5: Transaction rollback for section-only ===")
    
    # This test verifies that if something fails during section-only creation,
    # the transaction rolls back properly (no orphaned sections)
    
    # Since we don't have an easy way to force a failure in the middle,
    # we'll verify the service handles exceptions properly
    
    try:
        # Test with invalid parent_id (should trigger exception)
        try:
            result = create_section_with_admin(
                section_name="Test_B5_Rollback",
                parent_department_id=-9999,  # Invalid ID
                create_admin=False
            )
            print("FAIL: Test 5 - Exception expected but not raised")
            return False
        except Exception as e:
            # Expected exception
            assert "Failed to create section with admin" in str(e), "Wrong exception message"
            
            # Verify no section was created
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM AdminsrationUnit WHERE Name = 'Test_B5_Rollback'"
            )
            count = cursor.fetchone()[0]
            conn.close()
            
            assert count == 0, "Section should not exist after rollback"
            
        print(f"✓ Exception raised correctly")
        print(f"✓ Transaction rolled back (no orphaned section)")
        print("PASS: Test 5")
        return True
        
    except Exception as e:
        print(f"FAIL: Test 5 - {e}")
        return False


def test_6_backward_compatibility():
    """Test 6: Backward compatibility - existing calls still work"""
    print("\n=== Test 6: Backward compatibility ===")
    
    section_name = "Test_B5_BackCompat"
    cleanup_test_section(section_name)
    
    try:
        parent_id = get_valid_parent_department()
        
        # Call exactly like before B-C5 (no create_admin parameter)
        result = create_section_with_admin(
            section_name=section_name,
            parent_department_id=parent_id
        )
        
        # Should behave exactly like before B-C5: create both section + admin
        assert result["section_id"] > 0, "Section ID invalid"
        assert result["username"] is not None, "Username should exist (backward compat)"
        assert result["temp_password"] is not None, "Password should exist (backward compat)"
        
        # Verify database
        conn = get_connection()
        cursor = conn.cursor()
        
        # Section exists
        cursor.execute(
            "SELECT UniqueID FROM AdminsrationUnit WHERE UniqueID = ?",
            (result["section_id"],)
        )
        assert cursor.fetchone() is not None, "Section should exist"
        
        # Admin exists
        cursor.execute(
            "SELECT UserID FROM APP_Users WHERE username = ?",
            (result["username"],)
        )
        assert cursor.fetchone() is not None, "Admin should exist"
        
        conn.close()
        
        print(f"✓ Backward compatible call works")
        print(f"✓ Section + admin created as before B-C5")
        print("PASS: Test 6")
        return True
        
    except Exception as e:
        print(f"FAIL: Test 6 - {e}")
        return False
    finally:
        cleanup_test_section(section_name)


def test_7_router_behavior_unchanged():
    """Test 7: Router still calls with default behavior"""
    print("\n=== Test 7: Router behavior unchanged ===")
    
    # This test verifies that the router code hasn't changed
    # Router should NOT pass create_admin parameter (uses default=True)
    
    try:
        # Read router file
        router_path = os.path.join(
            os.path.dirname(__file__),
            "backend", "api", "routers", "admin_section_router.py"
        )
        
        with open(router_path, 'r', encoding='utf-8') as f:
            router_content = f.read()
        
        # Verify router calls create_section_with_admin
        assert "create_section_with_admin(" in router_content, "Router should call create_section_with_admin"
        
        # Verify router does NOT pass create_admin parameter
        assert "create_admin=" not in router_content, "Router should NOT pass create_admin parameter"
        
        # Verify router passes section_name and parent_department_id
        assert "section_name=request.section_name" in router_content, "Router should pass section_name"
        assert "parent_department_id=request.parent_unit_id" in router_content, "Router should pass parent_department_id"
        
        print(f"✓ Router unchanged: calls with default parameters")
        print(f"✓ No create_admin parameter in router")
        print("PASS: Test 7")
        return True
        
    except Exception as e:
        print(f"FAIL: Test 7 - {e}")
        return False


def test_8_section_only_can_attach_admin_later():
    """Test 8: Section created without admin can have admin attached later"""
    print("\n=== Test 8: Section-only can attach admin later ===")
    
    section_name = "Test_B5_AttachLater"
    cleanup_test_section(section_name)
    
    try:
        parent_id = get_valid_parent_department()
        
        # Step 1: Create section WITHOUT admin
        result = create_section_with_admin(
            section_name=section_name,
            parent_department_id=parent_id,
            create_admin=False
        )
        
        section_id = result["section_id"]
        assert result["username"] is None, "Initial: username should be None"
        
        # Step 2: Later, attach admin to existing section
        conn = get_connection()
        username, temp_password = attach_section_admin_user(
            conn=conn,
            section_id=section_id,
            created_by_user_id=None
        )
        conn.commit()
        conn.close()
        
        # Verify admin attached
        assert username is not None, "Admin username should exist after attach"
        assert temp_password is not None, "Admin password should exist after attach"
        
        # Verify database
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT UserID FROM APP_Users WHERE username = ?",
            (username,)
        )
        user_row = cursor.fetchone()
        assert user_row is not None, "Admin user should exist in DB"
        
        cursor.execute(
            "SELECT UserID FROM APP_UserRoleScope WHERE UserID = ? AND OrgUnitID = ?",
            (user_row[0], section_id)
        )
        scope_row = cursor.fetchone()
        assert scope_row is not None, "Role scope should exist after attach"
        conn.close()
        
        print(f"✓ Section created without admin")
        print(f"✓ Admin attached later: {username}")
        print(f"✓ Database verification passed")
        print("PASS: Test 8")
        return True
        
    except Exception as e:
        print(f"FAIL: Test 8 - {e}")
        return False
    finally:
        cleanup_test_section(section_name)


def run_all_tests():
    """Run all B-C5 tests"""
    print("="*70)
    print("Phase C - B-C5 Test Suite: Optional Section-Only Creation Path")
    print("="*70)
    
    tests = [
        test_1_default_behavior_creates_both,
        test_2_section_only_no_admin,
        test_3_default_parameter_value,
        test_4_response_consistency,
        test_5_transaction_rollback_section_only,
        test_6_backward_compatibility,
        test_7_router_behavior_unchanged,
        test_8_section_only_can_attach_admin_later
    ]
    
    results = []
    for test_func in tests:
        results.append(test_func())
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED - B-C5 COMPLETE")
        return 0
    else:
        print(f"\n✗ {total - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    exit(run_all_tests())
