"""
PHASE B — B-B2 — DB TEST — USER ROLE SCOPE ASSIGNMENT

Test suite for insert_user_role_scope function.
Tests role+scope assignment at the database layer.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.core.database import get_connection
from backend.api.db_layer.user_management_db import (
    insert_user_record,
    insert_user_role_scope
)


def cleanup_test_data(conn):
    """Clean up test users and their scopes from previous runs."""
    cursor = conn.cursor()
    try:
        # Delete scopes first (foreign key constraint)
        cursor.execute("""
            DELETE FROM dbo.APP_UserRoleScope 
            WHERE UserID IN (
                SELECT UserID FROM dbo.APP_Users 
                WHERE Username LIKE 'test_user_bb2%'
            )
        """)
        
        # Then delete users
        cursor.execute("""
            DELETE FROM dbo.APP_Users 
            WHERE Username LIKE 'test_user_bb2%'
        """)
    finally:
        cursor.close()


def get_test_role_id(conn):
    """Get a valid role ID for testing."""
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT TOP 1 RoleID, RoleCode, RoleNameEn
            FROM dbo.APP_Roles
            ORDER BY RoleID
        """)
        
        row = cursor.fetchone()
        if not row:
            raise Exception("No roles found in APP_Roles table")
        
        return row.RoleID, row.RoleCode, row.RoleNameEn
    finally:
        cursor.close()


def get_test_org_unit_id(conn):
    """Get a valid org unit ID for testing."""
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT TOP 1 UniqueID, Name
            FROM dbo.AdminsrationUnit
            WHERE Type IN (323, 324, 325)
            ORDER BY UniqueID
        """)
        
        row = cursor.fetchone()
        if not row:
            raise Exception("No org units found in AdminsrationUnit table")
        
        return row.UniqueID, row.Name
    finally:
        cursor.close()


def test_basic_role_scope_assignment():
    """Test 1: Basic role+scope assignment."""
    print("\n" + "="*60)
    print("TEST 1: Basic Role+Scope Assignment")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Clean up first
        cleanup_test_data(conn)
        
        # Create test user
        user_id = insert_user_record(
            conn,
            username="test_user_bb2_basic",
            password_hash="TEST_HASH",
            display_name="Test User BB2",
            department_display_name="Test Dept"
        )
        
        print(f"✓ Created test user with ID: {user_id}")
        
        # Get test role and org unit
        role_id, role_code, role_name = get_test_role_id(conn)
        org_unit_id, org_unit_name = get_test_org_unit_id(conn)
        
        print(f"✓ Using RoleID: {role_id} ({role_code} - {role_name})")
        print(f"✓ Using OrgUnitID: {org_unit_id} ({org_unit_name})")
        
        # Assign role+scope
        print(f"Assigning role+scope...")
        insert_user_role_scope(
            conn,
            user_id=user_id,
            role_id=role_id,
            org_unit_id=org_unit_id
        )
        
        print(f"✓ Role+scope assigned")
        
        # Verify assignment exists
        cursor.execute("""
            SELECT COUNT(*) AS cnt
            FROM dbo.APP_UserRoleScope
            WHERE UserID = ?
              AND RoleID = ?
              AND OrgUnitID = ?
        """, (user_id, role_id, org_unit_id))
        
        result = cursor.fetchone()
        
        # Assertions
        assert result.cnt == 1, f"Expected 1 scope assignment, found {result.cnt}"
        
        print(f"✓ Verified: 1 scope assignment exists in database")
        
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
        # Rollback to clean up
        conn.rollback()
        cursor.close()
        conn.close()


def test_duplicate_assignment_prevention():
    """Test 2: Duplicate assignment prevention (idempotent)."""
    print("\n" + "="*60)
    print("TEST 2: Duplicate Assignment Prevention")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Clean up first
        cleanup_test_data(conn)
        
        # Create test user
        user_id = insert_user_record(
            conn,
            username="test_user_bb2_duplicate",
            password_hash="TEST_HASH",
            display_name="Test User Duplicate",
            department_display_name="Test Dept"
        )
        
        print(f"✓ Created test user with ID: {user_id}")
        
        # Get test role and org unit
        role_id, role_code, role_name = get_test_role_id(conn)
        org_unit_id, org_unit_name = get_test_org_unit_id(conn)
        
        print(f"✓ Using RoleID: {role_id} ({role_code})")
        print(f"✓ Using OrgUnitID: {org_unit_id}")
        
        # First assignment
        print(f"First assignment...")
        insert_user_role_scope(
            conn,
            user_id=user_id,
            role_id=role_id,
            org_unit_id=org_unit_id
        )
        
        print(f"✓ First assignment completed")
        
        # Verify 1 assignment
        cursor.execute("""
            SELECT COUNT(*) AS cnt
            FROM dbo.APP_UserRoleScope
            WHERE UserID = ?
              AND RoleID = ?
              AND OrgUnitID = ?
        """, (user_id, role_id, org_unit_id))
        
        result = cursor.fetchone()
        assert result.cnt == 1, f"Expected 1 assignment after first call, found {result.cnt}"
        print(f"✓ Verified: 1 assignment exists")
        
        # Second assignment (should be idempotent)
        print(f"Second assignment (should be no-op)...")
        insert_user_role_scope(
            conn,
            user_id=user_id,
            role_id=role_id,
            org_unit_id=org_unit_id
        )
        
        print(f"✓ Second assignment completed")
        
        # Verify STILL 1 assignment (no duplicate)
        cursor.execute("""
            SELECT COUNT(*) AS cnt
            FROM dbo.APP_UserRoleScope
            WHERE UserID = ?
              AND RoleID = ?
              AND OrgUnitID = ?
        """, (user_id, role_id, org_unit_id))
        
        result = cursor.fetchone()
        assert result.cnt == 1, f"Expected 1 assignment after second call (no duplicate), found {result.cnt}"
        print(f"✓ Verified: Still only 1 assignment (no duplicate created)")
        
        print("\n✓ TEST 2 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 2 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 2 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Rollback to clean up
        conn.rollback()
        cursor.close()
        conn.close()


def test_multiple_roles_for_same_user():
    """Test 3: User can have multiple roles for different org units."""
    print("\n" + "="*60)
    print("TEST 3: Multiple Roles for Same User")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Clean up first
        cleanup_test_data(conn)
        
        # Create test user
        user_id = insert_user_record(
            conn,
            username="test_user_bb2_multi",
            password_hash="TEST_HASH",
            display_name="Test User Multi",
            department_display_name="Test Dept"
        )
        
        print(f"✓ Created test user with ID: {user_id}")
        
        # Get multiple roles
        cursor.execute("""
            SELECT TOP 2 RoleID, RoleCode
            FROM dbo.APP_Roles
            ORDER BY RoleID
        """)
        roles = cursor.fetchall()
        
        if len(roles) < 2:
            print("⚠️ WARNING: Less than 2 roles available, using same role for both assignments")
            roles = [roles[0], roles[0]]
        
        # Get multiple org units
        cursor.execute("""
            SELECT TOP 2 UniqueID, Name
            FROM dbo.AdminsrationUnit
            WHERE Type IN (323, 324, 325)
            ORDER BY UniqueID
        """)
        org_units = cursor.fetchall()
        
        if len(org_units) < 2:
            raise Exception("Need at least 2 org units for this test")
        
        role1_id, role1_code = roles[0].RoleID, roles[0].RoleCode
        role2_id, role2_code = roles[1].RoleID, roles[1].RoleCode
        org1_id, org1_name = org_units[0].UniqueID, org_units[0].Name
        org2_id, org2_name = org_units[1].UniqueID, org_units[1].Name
        
        print(f"✓ Using Role 1: {role1_id} ({role1_code})")
        print(f"✓ Using Role 2: {role2_id} ({role2_code})")
        print(f"✓ Using OrgUnit 1: {org1_id} ({org1_name})")
        print(f"✓ Using OrgUnit 2: {org2_id} ({org2_name})")
        
        # First assignment
        print(f"Assigning role 1 + org unit 1...")
        insert_user_role_scope(
            conn,
            user_id=user_id,
            role_id=role1_id,
            org_unit_id=org1_id
        )
        
        # Second assignment (different combo)
        print(f"Assigning role 2 + org unit 2...")
        insert_user_role_scope(
            conn,
            user_id=user_id,
            role_id=role2_id,
            org_unit_id=org2_id
        )
        
        print(f"✓ Both assignments completed")
        
        # Verify 2 total assignments
        cursor.execute("""
            SELECT COUNT(*) AS cnt
            FROM dbo.APP_UserRoleScope
            WHERE UserID = ?
        """, (user_id,))
        
        result = cursor.fetchone()
        assert result.cnt == 2, f"Expected 2 total assignments, found {result.cnt}"
        print(f"✓ Verified: User has 2 role+scope assignments")
        
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
        # Rollback to clean up
        conn.rollback()
        cursor.close()
        conn.close()


def test_invalid_user_id_raises_error():
    """Test 4: Invalid user_id raises ValueError."""
    print("\n" + "="*60)
    print("TEST 4: Invalid user_id Validation")
    print("="*60)
    
    conn = get_connection()
    
    try:
        # Get valid role and org unit
        role_id, _, _ = get_test_role_id(conn)
        org_unit_id, _ = get_test_org_unit_id(conn)
        
        # Try with invalid user_id
        print("Attempting to assign with user_id=0...")
        
        insert_user_role_scope(
            conn,
            user_id=0,
            role_id=role_id,
            org_unit_id=org_unit_id
        )
        
        # Should not reach here
        print(f"\n✗ TEST 4 FAILED: Invalid user_id was accepted")
        return False
        
    except ValueError as e:
        # Expected behavior
        print(f"✓ ValueError raised as expected: {str(e)}")
        print("\n✓ TEST 4 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 4 ERROR: Unexpected exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        conn.rollback()
        conn.close()


def test_invalid_role_id_raises_error():
    """Test 5: Invalid role_id raises ValueError."""
    print("\n" + "="*60)
    print("TEST 5: Invalid role_id Validation")
    print("="*60)
    
    conn = get_connection()
    
    try:
        # Get valid org unit
        org_unit_id, _ = get_test_org_unit_id(conn)
        
        # Try with invalid role_id
        print("Attempting to assign with role_id=-1...")
        
        insert_user_role_scope(
            conn,
            user_id=999,
            role_id=-1,
            org_unit_id=org_unit_id
        )
        
        # Should not reach here
        print(f"\n✗ TEST 5 FAILED: Invalid role_id was accepted")
        return False
        
    except ValueError as e:
        # Expected behavior
        print(f"✓ ValueError raised as expected: {str(e)}")
        print("\n✓ TEST 5 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 5 ERROR: Unexpected exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        conn.rollback()
        conn.close()


def test_invalid_org_unit_id_raises_error():
    """Test 6: Invalid org_unit_id raises ValueError."""
    print("\n" + "="*60)
    print("TEST 6: Invalid org_unit_id Validation")
    print("="*60)
    
    conn = get_connection()
    
    try:
        # Get valid role
        role_id, _, _ = get_test_role_id(conn)
        
        # Try with invalid org_unit_id
        print("Attempting to assign with org_unit_id=0...")
        
        insert_user_role_scope(
            conn,
            user_id=999,
            role_id=role_id,
            org_unit_id=0
        )
        
        # Should not reach here
        print(f"\n✗ TEST 6 FAILED: Invalid org_unit_id was accepted")
        return False
        
    except ValueError as e:
        # Expected behavior
        print(f"✓ ValueError raised as expected: {str(e)}")
        print("\n✓ TEST 6 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 6 ERROR: Unexpected exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        conn.rollback()
        conn.close()


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*60)
    print("PHASE B — B-B2 — DB TEST SUITE — USER ROLE SCOPE")
    print("="*60)
    
    tests = [
        ("Basic Role+Scope Assignment", test_basic_role_scope_assignment),
        ("Duplicate Assignment Prevention", test_duplicate_assignment_prevention),
        ("Multiple Roles for Same User", test_multiple_roles_for_same_user),
        ("Invalid user_id Validation", test_invalid_user_id_raises_error),
        ("Invalid role_id Validation", test_invalid_role_id_raises_error),
        ("Invalid org_unit_id Validation", test_invalid_org_unit_id_raises_error),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return True
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
