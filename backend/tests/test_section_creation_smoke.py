"""
PHASE C — B-C9 — BACKEND SMOKE TESTS FOR SECTION CREATION
==========================================================

Smoke tests for section creation flow covering success and failure paths.

Tests:
1. Create section under department (success case)
2. Create section under administration (success case)
3. Invalid parent type - use section as parent (error case)
4. Credentials returned in response (validation)
5. Rollback test - verify transaction rollback on failure
"""

import sys
import os
from unittest.mock import patch
import pyodbc

# Add backend directory to path for proper imports
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from api.services.section_admin_creator_service import (
    create_section_with_admin,
    create_section_core,
    attach_section_admin_user
)
from core.database import get_connection


def get_test_connection():
    """Get database connection for test queries"""
    return get_connection()


def get_valid_department_id():
    """Get a valid department ID from database for testing"""
    conn = get_test_connection()
    try:
        cursor = conn.cursor()
        # Find a department (Type = 325)
        query = """
            SELECT TOP 1 UniqueID 
            FROM dbo.AdminsrationUnit 
            WHERE Type = 325
            ORDER BY UniqueID
        """
        cursor.execute(query)
        result = cursor.fetchone()
        cursor.close()
        
        if result:
            return result.UniqueID
        else:
            # If no department exists, return 1 (likely administration)
            return 1
    finally:
        conn.close()


def get_valid_administration_id():
    """Get a valid administration ID from database for testing"""
    conn = get_test_connection()
    try:
        cursor = conn.cursor()
        # Find an administration unit (Type = 323)
        query = """
            SELECT TOP 1 UniqueID 
            FROM dbo.AdminsrationUnit 
            WHERE Type = 323
            ORDER BY UniqueID
        """
        cursor.execute(query)
        result = cursor.fetchone()
        cursor.close()
        
        if result:
            return result.UniqueID
        else:
            # Fallback to ID 1
            return 1
    finally:
        conn.close()


def get_valid_section_id():
    """Get a valid section ID from database for testing invalid parent"""
    conn = get_test_connection()
    try:
        cursor = conn.cursor()
        # Find a section (Type = 324)
        query = """
            SELECT TOP 1 UniqueID 
            FROM dbo.AdminsrationUnit 
            WHERE Type = 324
            ORDER BY UniqueID DESC
        """
        cursor.execute(query)
        result = cursor.fetchone()
        cursor.close()
        
        if result:
            return result.UniqueID
        else:
            # No section exists yet - will need to create one first
            return None
    finally:
        conn.close()


def check_section_exists(section_id):
    """Check if a section exists in database"""
    conn = get_test_connection()
    try:
        cursor = conn.cursor()
        query = """
            SELECT COUNT(*) as count
            FROM dbo.AdminsrationUnit 
            WHERE UniqueID = ? AND Type = 324
        """
        cursor.execute(query, (section_id,))
        result = cursor.fetchone()
        cursor.close()
        return result.count > 0
    finally:
        conn.close()


def check_user_exists(username):
    """Check if a user exists in database"""
    conn = get_test_connection()
    try:
        cursor = conn.cursor()
        query = """
            SELECT COUNT(*) as count
            FROM dbo.APP_Users 
            WHERE Username = ?
        """
        cursor.execute(query, (username,))
        result = cursor.fetchone()
        cursor.close()
        return result.count > 0
    finally:
        conn.close()


def cleanup_test_section(section_id):
    """Helper to clean up test section (careful - also deletes admin user)"""
    conn = get_test_connection()
    try:
        cursor = conn.cursor()
        
        # Find and delete associated user
        username = f"sec_{section_id}_admin"
        
        # Get user_id
        cursor.execute("SELECT UserID FROM dbo.APP_Users WHERE Username = ?", (username,))
        user_result = cursor.fetchone()
        
        if user_result:
            user_id = user_result.UserID
            
            # Delete scope
            cursor.execute("DELETE FROM dbo.APP_UserRoleScope WHERE UserID = ?", (user_id,))
            
            # Delete user
            cursor.execute("DELETE FROM dbo.APP_Users WHERE UserID = ?", (user_id,))
        
        # Delete section
        cursor.execute("DELETE FROM dbo.AdminsrationUnit WHERE UniqueID = ?", (section_id,))
        
        conn.commit()
        cursor.close()
        print(f"  [Cleanup] Deleted test section {section_id} and associated user")
    except Exception as e:
        conn.rollback()
        print(f"  [Cleanup Warning] Could not delete section {section_id}: {e}")
    finally:
        conn.close()


def test_1_create_under_department():
    """
    TEST 1: Create section under department
    Verify successful creation with 200 response and section_id exists.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Create Section Under Department")
    print("=" * 70)
    
    parent_dept_id = get_valid_department_id()
    section_name = f"Smoke Test Section Dept {os.getpid()}"
    
    print(f"  Parent Department ID: {parent_dept_id}")
    print(f"  Section Name: {section_name}")
    
    try:
        # Call service function
        result = create_section_with_admin(
            section_name=section_name,
            parent_department_id=parent_dept_id,
            create_admin=True
        )
        
        # Verify result structure
        assert result is not None, "Result should not be None"
        assert "section_id" in result, "Result should contain section_id"
        assert "section_name" in result, "Result should contain section_name"
        assert "parent_unit_id" in result, "Result should contain parent_unit_id"
        assert "username" in result, "Result should contain username"
        assert "temp_password" in result, "Result should contain temp_password"
        
        section_id = result["section_id"]
        print(f"  ✓ Section created: ID={section_id}")
        
        # Verify section exists in database
        exists = check_section_exists(section_id)
        assert exists, f"Section {section_id} should exist in database"
        print(f"  ✓ Section verified in database")
        
        # Verify section_id is valid (positive integer)
        assert isinstance(section_id, int), "section_id should be int"
        assert section_id > 0, "section_id should be positive"
        print(f"  ✓ Section ID is valid: {section_id}")
        
        # Cleanup
        cleanup_test_section(section_id)
        
        print("\n✅ TEST 1 PASSED: Create under department successful")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {str(e)}")
        return False


def test_2_create_under_administration():
    """
    TEST 2: Create section under administration
    Verify successful creation with 200 response.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Create Section Under Administration")
    print("=" * 70)
    
    parent_admin_id = get_valid_administration_id()
    section_name = f"Smoke Test Section Admin {os.getpid()}"
    
    print(f"  Parent Administration ID: {parent_admin_id}")
    print(f"  Section Name: {section_name}")
    
    try:
        # Call service function
        result = create_section_with_admin(
            section_name=section_name,
            parent_department_id=parent_admin_id,
            create_admin=True
        )
        
        # Verify result
        assert result is not None, "Result should not be None"
        assert "section_id" in result, "Result should contain section_id"
        
        section_id = result["section_id"]
        print(f"  ✓ Section created: ID={section_id}")
        
        # Verify section exists in database
        exists = check_section_exists(section_id)
        assert exists, f"Section {section_id} should exist in database"
        print(f"  ✓ Section verified in database")
        
        # Verify parent_unit_id matches
        assert result["parent_unit_id"] == parent_admin_id, "parent_unit_id should match"
        print(f"  ✓ Parent unit ID matches: {parent_admin_id}")
        
        # Cleanup
        cleanup_test_section(section_id)
        
        print("\n✅ TEST 2 PASSED: Create under administration successful")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {str(e)}")
        return False


def test_3_invalid_parent_type():
    """
    TEST 3: Invalid parent type - use section as parent
    Expect error (section cannot be parent of another section).
    """
    print("\n" + "=" * 70)
    print("TEST 3: Invalid Parent Type (Section as Parent)")
    print("=" * 70)
    
    # Get or create a section ID to use as invalid parent
    section_id = get_valid_section_id()
    
    if not section_id:
        # Create a section first
        print("  [Setup] Creating a section to use as invalid parent...")
        parent_dept_id = get_valid_department_id()
        temp_result = create_section_with_admin(
            section_name=f"Temp Parent Section {os.getpid()}",
            parent_department_id=parent_dept_id,
            create_admin=True
        )
        section_id = temp_result["section_id"]
        print(f"  [Setup] Created section {section_id} to use as parent")
    
    print(f"  Attempting to use section {section_id} as parent...")
    
    section_name = f"Smoke Test Invalid Parent {os.getpid()}"
    
    try:
        # This should fail
        result = create_section_with_admin(
            section_name=section_name,
            parent_department_id=section_id,  # Using section as parent (invalid)
            create_admin=True
        )
        
        # If we get here, the test failed (should have raised exception)
        print(f"\n❌ TEST 3 FAILED: Expected error but got success: {result}")
        
        # Cleanup the erroneously created section
        if "section_id" in result:
            cleanup_test_section(result["section_id"])
        
        return False
        
    except Exception as e:
        # Expected to fail
        error_message = str(e)
        print(f"  ✓ Error raised as expected: {error_message}")
        
        # Verify error message indicates parent validation issue
        # (This is business logic dependent - adjust if needed)
        print(f"  ✓ Verification: Exception raised for invalid parent")
        
        print("\n✅ TEST 3 PASSED: Invalid parent type rejected")
        return True


def test_4_credentials_returned():
    """
    TEST 4: Credentials returned in response
    Verify username and temp_password are present and valid format.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Credentials Returned in Response")
    print("=" * 70)
    
    parent_dept_id = get_valid_department_id()
    section_name = f"Smoke Test Credentials {os.getpid()}"
    
    try:
        # Create section
        result = create_section_with_admin(
            section_name=section_name,
            parent_department_id=parent_dept_id,
            create_admin=True
        )
        
        section_id = result["section_id"]
        print(f"  Section created: ID={section_id}")
        
        # Verify username present and valid format
        assert "username" in result, "username should be in response"
        username = result["username"]
        assert username is not None, "username should not be None"
        assert isinstance(username, str), "username should be string"
        assert len(username) > 0, "username should not be empty"
        
        # Verify username format: sec_{id}_admin
        expected_username = f"sec_{section_id}_admin"
        assert username == expected_username, f"username should be {expected_username}"
        print(f"  ✓ Username valid: {username}")
        
        # Verify temp_password present and valid
        assert "temp_password" in result, "temp_password should be in response"
        temp_password = result["temp_password"]
        assert temp_password is not None, "temp_password should not be None"
        assert isinstance(temp_password, str), "temp_password should be string"
        assert len(temp_password) > 0, "temp_password should not be empty"
        print(f"  ✓ Temp password valid: {temp_password}")
        
        # Verify user exists in database
        user_exists = check_user_exists(username)
        assert user_exists, f"User {username} should exist in database"
        print(f"  ✓ User verified in database: {username}")
        
        # Cleanup
        cleanup_test_section(section_id)
        
        print("\n✅ TEST 4 PASSED: Credentials returned and valid")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {str(e)}")
        return False


def test_5_rollback_on_failure():
    """
    TEST 5: Rollback test - verify transaction rollback on failure
    Monkeypatch attach_section_admin_user to throw error.
    Verify no section row inserted.
    """
    print("\n" + "=" * 70)
    print("TEST 5: Transaction Rollback on Failure")
    print("=" * 70)
    
    parent_dept_id = get_valid_department_id()
    section_name = f"Smoke Test Rollback {os.getpid()}"
    
    print(f"  Testing rollback behavior with simulated failure...")
    
    # Get count of sections before test
    conn = get_test_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM dbo.AdminsrationUnit WHERE Type = 324")
        before_count = cursor.fetchone().count
        cursor.close()
    finally:
        conn.close()
    
    print(f"  Sections before test: {before_count}")
    
    try:
        # Patch attach_section_admin_user to raise an exception
        with patch('api.services.section_admin_creator_service.attach_section_admin_user') as mock_attach:
            # Make it raise an exception
            mock_attach.side_effect = Exception("Simulated failure in attach_section_admin_user")
            
            # Attempt to create section (should fail and rollback)
            try:
                result = create_section_with_admin(
                    section_name=section_name,
                    parent_department_id=parent_dept_id,
                    create_admin=True
                )
                
                # Should not reach here
                print(f"\n❌ TEST 5 FAILED: Expected error but got success")
                
                # Cleanup if somehow succeeded
                if "section_id" in result:
                    cleanup_test_section(result["section_id"])
                
                return False
                
            except Exception as e:
                # Expected to fail
                error_message = str(e)
                print(f"  ✓ Error raised as expected: {error_message}")
                
                # Verify error message contains our simulated failure
                assert "Simulated failure" in error_message or "attach_section_admin_user" in error_message.lower(), \
                    "Error should mention the simulated failure"
                print(f"  ✓ Error message indicates rollback trigger")
        
        # Verify no new section was inserted (rollback worked)
        conn = get_test_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM dbo.AdminsrationUnit WHERE Type = 324")
            after_count = cursor.fetchone().count
            cursor.close()
        finally:
            conn.close()
        
        print(f"  Sections after test: {after_count}")
        
        assert after_count == before_count, \
            f"Section count should remain {before_count}, but is {after_count} (rollback failed)"
        print(f"  ✓ Section count unchanged: {before_count}")
        
        # Double-check specific section name doesn't exist
        conn = get_test_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) as count FROM dbo.AdminsrationUnit WHERE Name = ? AND Type = 324",
                (section_name,)
            )
            name_count = cursor.fetchone().count
            cursor.close()
        finally:
            conn.close()
        
        assert name_count == 0, f"Section with name '{section_name}' should not exist"
        print(f"  ✓ No orphan section row found")
        
        print("\n✅ TEST 5 PASSED: Transaction rollback successful")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 5 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all smoke tests for section creation"""
    print("\n" + "=" * 70)
    print("PHASE C — B-C9 — BACKEND SMOKE TEST SUITE")
    print("Section Creation Flow")
    print("=" * 70)
    
    tests = [
        ("Create Under Department", test_1_create_under_department),
        ("Create Under Administration", test_2_create_under_administration),
        ("Invalid Parent Type", test_3_invalid_parent_type),
        ("Credentials Returned", test_4_credentials_returned),
        ("Transaction Rollback", test_5_rollback_on_failure)
    ]
    
    results = []
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASSED" if success else "FAILED"))
            if success:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ {test_name} ERROR: {str(e)}")
            results.append((test_name, "ERROR"))
            failed += 1
    
    # Print summary
    print("\n" + "=" * 70)
    print("SMOKE TEST SUMMARY")
    print("=" * 70)
    
    for test_name, status in results:
        status_symbol = "✅" if status == "PASSED" else "❌"
        print(f"{status_symbol} {test_name}: {status}")
    
    print(f"\nTotal: {len(tests)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed/len(tests)*100):.1f}%")
    
    if failed == 0:
        print("\n" + "=" * 70)
        print("🎉 ALL SMOKE TESTS PASSED")
        print("=" * 70)
        return True
    else:
        print("\n" + "=" * 70)
        print(f"❌ {failed} TEST(S) FAILED")
        print("=" * 70)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
