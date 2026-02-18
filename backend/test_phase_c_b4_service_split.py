"""
PHASE C — B-C4 — TEST SERVICE SPLIT
Tests for refactored section creation service with core + attach split.

Test Coverage:
1. Import Test - Verify new functions can be imported
2. Core Function Test - Test create_section_core independently
3. Attach Function Test - Test attach_section_admin_user independently
4. Orchestration Test - Test full create_section_with_admin
5. Transaction Rollback Test - Verify rollback works correctly
6. Endpoint Contract Test - Verify endpoint behavior unchanged
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent
backend_path = Path(__file__).parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from core.database import get_connection


class TestResults:
    """Track test results"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def add_pass(self, test_name: str):
        self.passed += 1
        print(f"✅ PASS: {test_name}")
    
    def add_fail(self, test_name: str, reason: str):
        self.failed += 1
        error_msg = f"❌ FAIL: {test_name} - {reason}"
        self.errors.append(error_msg)
        print(error_msg)
    
    def summary(self):
        total = self.passed + self.failed
        print("\n" + "=" * 80)
        print(f"TEST SUMMARY: {self.passed}/{total} passed")
        if self.errors:
            print("\nFailed Tests:")
            for error in self.errors:
                print(f"  {error}")
        print("=" * 80)
        return self.failed == 0


def test_1_import_split_functions():
    """Test 1: Verify new split functions can be imported"""
    results = TestResults()
    
    print("\n" + "=" * 80)
    print("TEST 1: IMPORT SPLIT FUNCTIONS")
    print("=" * 80)
    
    try:
        from api.services.section_admin_creator_service import (
            create_section_core,
            attach_section_admin_user,
            create_section_with_admin
        )
        
        results.add_pass("Import create_section_core")
        results.add_pass("Import attach_section_admin_user")
        results.add_pass("Import create_section_with_admin")
        
        # Verify they are callable
        if callable(create_section_core):
            results.add_pass("create_section_core is callable")
        else:
            results.add_fail("create_section_core type", "Not callable")
        
        if callable(attach_section_admin_user):
            results.add_pass("attach_section_admin_user is callable")
        else:
            results.add_fail("attach_section_admin_user type", "Not callable")
        
        if callable(create_section_with_admin):
            results.add_pass("create_section_with_admin is callable")
        else:
            results.add_fail("create_section_with_admin type", "Not callable")
        
    except ImportError as e:
        results.add_fail("Import functions", str(e))
    except Exception as e:
        results.add_fail("Import functions", f"Unexpected error: {str(e)}")
    
    return results.summary()


def test_2_core_function_creates_section_only():
    """Test 2: Test create_section_core creates section without user"""
    results = TestResults()
    
    print("\n" + "=" * 80)
    print("TEST 2: CORE FUNCTION - SECTION ONLY (NO USER)")
    print("=" * 80)
    
    conn = None
    created_section_id = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Find a parent department
        cursor.execute("SELECT TOP 1 UniqueID, Name FROM dbo.AdminsrationUnit WHERE Type = 325 ORDER BY UniqueID")
        dept = cursor.fetchone()
        
        if not dept:
            results.add_fail("Find parent department", "No department found")
            return False
        
        parent_dept_id = dept.UniqueID
        parent_dept_name = dept.Name
        print(f"Using parent department: {parent_dept_name} (ID: {parent_dept_id})")
        
        # Import core function
        from api.services.section_admin_creator_service import create_section_core
        
        test_section_name = f"TEST_CORE_C4_{os.urandom(4).hex()}"
        
        try:
            # Call core function (within same transaction)
            section_id = create_section_core(
                conn=conn,
                section_name=test_section_name,
                parent_unit_id=parent_dept_id,
                created_by_user_id=None
            )
            
            created_section_id = section_id
            
            results.add_pass("Core function executed")
            print(f"  Created section ID: {section_id}")
            
            # Verify section exists in database
            cursor.execute("""
                SELECT UniqueID, Name, Type, ParentID
                FROM dbo.AdminsrationUnit
                WHERE UniqueID = ?
            """, (section_id,))
            
            section_row = cursor.fetchone()
            
            if section_row:
                results.add_pass("Section exists in database")
                
                if section_row.Name == test_section_name:
                    results.add_pass("Section name matches input")
                else:
                    results.add_fail("Section name", f"Expected '{test_section_name}', got '{section_row.Name}'")
                
                if section_row.Type == 324:
                    results.add_pass("Section has correct Type = 324")
                else:
                    results.add_fail("Section Type", f"Expected 324, got {section_row.Type}")
                
                if section_row.ParentID == parent_dept_id:
                    results.add_pass("Section has correct ParentID")
                else:
                    results.add_fail("Section ParentID", f"Expected {parent_dept_id}, got {section_row.ParentID}")
            else:
                results.add_fail("Section exists", "Section not found in database")
            
            # CRITICAL: Verify NO USER was created
            expected_username = f"sec_{section_id}_admin"
            cursor.execute("SELECT UserID FROM dbo.APP_Users WHERE Username = ?", (expected_username,))
            user_row = cursor.fetchone()
            
            if user_row is None:
                results.add_pass("NO user created (core function isolated correctly)")
            else:
                results.add_fail("User isolation", f"User '{expected_username}' was created (should not be)")
            
            # Commit to save section for cleanup
            conn.commit()
        
        except Exception as e:
            results.add_fail("Core function execution", str(e))
            conn.rollback()
    
    except Exception as e:
        results.add_fail("Core function test setup", str(e))
    
    finally:
        # Cleanup
        if conn:
            cursor = conn.cursor()
            try:
                if created_section_id:
                    cursor.execute("DELETE FROM dbo.AdminsrationUnit WHERE UniqueID = ?", (created_section_id,))
                    conn.commit()
                    print(f"\n🧹 Cleaned up section ID {created_section_id}")
            except Exception as e:
                print(f"⚠️  Cleanup error: {e}")
                conn.rollback()
            
            conn.close()
    
    return results.summary()


def test_3_attach_function_creates_user_only():
    """Test 3: Test attach_section_admin_user creates user for existing section"""
    results = TestResults()
    
    print("\n" + "=" * 80)
    print("TEST 3: ATTACH FUNCTION - USER ONLY (FOR EXISTING SECTION)")
    print("=" * 80)
    
    conn = None
    created_section_id = None
    created_user_id = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Find a parent department
        cursor.execute("SELECT TOP 1 UniqueID FROM dbo.AdminsrationUnit WHERE Type = 325 ORDER BY UniqueID")
        dept = cursor.fetchone()
        
        if not dept:
            results.add_fail("Find parent department", "No department found")
            return False
        
        parent_dept_id = dept.UniqueID
        
        # First, create a section using core function
        from api.services.section_admin_creator_service import (
            create_section_core,
            attach_section_admin_user
        )
        
        test_section_name = f"TEST_ATTACH_C4_{os.urandom(4).hex()}"
        
        try:
            # Step 1: Create section without user
            section_id = create_section_core(
                conn=conn,
                section_name=test_section_name,
                parent_unit_id=parent_dept_id,
                created_by_user_id=None
            )
            
            created_section_id = section_id
            results.add_pass("Pre-created section for attach test")
            print(f"  Section ID: {section_id}")
            
            # Step 2: Attach admin user to existing section
            username, temp_password = attach_section_admin_user(
                conn=conn,
                section_id=section_id,
                created_by_user_id=None
            )
            
            results.add_pass("Attach function executed")
            print(f"  Username: {username}")
            print(f"  Temp Password: {temp_password}")
            
            # Verify username format
            expected_username = f"sec_{section_id}_admin"
            if username == expected_username:
                results.add_pass("Username follows correct format")
            else:
                results.add_fail("Username format", f"Expected '{expected_username}', got '{username}'")
            
            # Verify password
            if temp_password == "Hospital2026!":
                results.add_pass("Temp password correct")
            else:
                results.add_fail("Temp password", f"Expected 'Hospital2026!', got '{temp_password}'")
            
            # Verify user was created in database
            cursor.execute("SELECT UserID, Username FROM dbo.APP_Users WHERE Username = ?", (username,))
            user_row = cursor.fetchone()
            
            if user_row:
                results.add_pass("User created in database")
                created_user_id = user_row.UserID
            else:
                results.add_fail("User creation", "User not found in database")
                conn.rollback()
                return results.summary()
            
            # Verify scope was assigned
            cursor.execute("""
                SELECT s.UserID, s.OrgUnitID, s.OrgUnitType, r.RoleCode
                FROM dbo.APP_UserRoleScope s
                JOIN dbo.APP_Roles r ON s.RoleID = r.RoleID
                WHERE s.UserID = ?
            """, (created_user_id,))
            
            scope_row = cursor.fetchone()
            
            if scope_row:
                results.add_pass("User scope assigned")
                
                if scope_row.OrgUnitID == section_id:
                    results.add_pass("Scope linked to correct section")
                else:
                    results.add_fail("Scope OrgUnitID", f"Expected {section_id}, got {scope_row.OrgUnitID}")
                
                if scope_row.RoleCode == "SECTION_ADMIN":
                    results.add_pass("User has SECTION_ADMIN role")
                else:
                    results.add_fail("User role", f"Expected SECTION_ADMIN, got {scope_row.RoleCode}")
                
                if scope_row.OrgUnitType == "SECTION":
                    results.add_pass("Scope has correct OrgUnitType")
                else:
                    results.add_fail("Scope OrgUnitType", f"Expected SECTION, got {scope_row.OrgUnitType}")
            else:
                results.add_fail("User scope", "No scope found for user")
            
            # Commit
            conn.commit()
        
        except Exception as e:
            results.add_fail("Attach function execution", str(e))
            conn.rollback()
    
    except Exception as e:
        results.add_fail("Attach function test setup", str(e))
    
    finally:
        # Cleanup
        if conn:
            cursor = conn.cursor()
            try:
                if created_user_id:
                    cursor.execute("DELETE FROM dbo.APP_UserRoleScope WHERE UserID = ?", (created_user_id,))
                    cursor.execute("DELETE FROM dbo.APP_Users WHERE UserID = ?", (created_user_id,))
                    print(f"\n🧹 Cleaned up user ID {created_user_id}")
                
                if created_section_id:
                    cursor.execute("DELETE FROM dbo.AdminsrationUnit WHERE UniqueID = ?", (created_section_id,))
                    print(f"🧹 Cleaned up section ID {created_section_id}")
                
                conn.commit()
            except Exception as e:
                print(f"⚠️  Cleanup error: {e}")
                conn.rollback()
            
            conn.close()
    
    return results.summary()


def test_4_orchestration_function_complete():
    """Test 4: Test full create_section_with_admin orchestration"""
    results = TestResults()
    
    print("\n" + "=" * 80)
    print("TEST 4: ORCHESTRATION FUNCTION - COMPLETE FLOW")
    print("=" * 80)
    
    conn = None
    created_section_id = None
    created_user_id = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Find a parent department
        cursor.execute("SELECT TOP 1 UniqueID, Name FROM dbo.AdminsrationUnit WHERE Type = 325 ORDER BY UniqueID")
        dept = cursor.fetchone()
        
        if not dept:
            results.add_fail("Find parent department", "No department found")
            return False
        
        parent_dept_id = dept.UniqueID
        parent_dept_name = dept.Name
        print(f"Using parent department: {parent_dept_name} (ID: {parent_dept_id})")
        
        # Import orchestration function
        from api.services.section_admin_creator_service import create_section_with_admin
        
        test_section_name = f"TEST_ORCH_C4_{os.urandom(4).hex()}"
        
        try:
            # Call orchestration function
            result = create_section_with_admin(test_section_name, parent_dept_id)
            
            results.add_pass("Orchestration function executed")
            
            # Verify return shape (all 5 fields)
            required_fields = ['section_id', 'section_name', 'parent_unit_id', 'username', 'temp_password']
            
            for field_name in required_fields:
                if field_name in result:
                    results.add_pass(f"Response has '{field_name}'")
                else:
                    results.add_fail("Response fields", f"Missing '{field_name}'")
            
            created_section_id = result.get("section_id")
            username = result.get("username")
            
            print(f"  Section ID: {created_section_id}")
            print(f"  Section Name: {result.get('section_name')}")
            print(f"  Parent Unit ID: {result.get('parent_unit_id')}")
            print(f"  Username: {username}")
            print(f"  Temp Password: {result.get('temp_password')}")
            
            # Verify section exists
            cursor.execute("SELECT UniqueID, Name, Type FROM dbo.AdminsrationUnit WHERE UniqueID = ?", (created_section_id,))
            section_row = cursor.fetchone()
            
            if section_row:
                results.add_pass("Section created")
            else:
                results.add_fail("Section creation", "Section not found")
            
            # Verify user exists
            cursor.execute("SELECT UserID FROM dbo.APP_Users WHERE Username = ?", (username,))
            user_row = cursor.fetchone()
            
            if user_row:
                results.add_pass("User created")
                created_user_id = user_row.UserID
            else:
                results.add_fail("User creation", "User not found")
            
            # Verify scope exists
            if created_user_id:
                cursor.execute("SELECT UserID FROM dbo.APP_UserRoleScope WHERE UserID = ?", (created_user_id,))
                scope_row = cursor.fetchone()
                
                if scope_row:
                    results.add_pass("User scope assigned")
                else:
                    results.add_fail("Scope assignment", "No scope found")
        
        except Exception as e:
            results.add_fail("Orchestration function execution", str(e))
    
    except Exception as e:
        results.add_fail("Orchestration function test setup", str(e))
    
    finally:
        # Cleanup
        if conn:
            cursor = conn.cursor()
            try:
                if created_user_id:
                    cursor.execute("DELETE FROM dbo.APP_UserRoleScope WHERE UserID = ?", (created_user_id,))
                    cursor.execute("DELETE FROM dbo.APP_Users WHERE UserID = ?", (created_user_id,))
                    print(f"\n🧹 Cleaned up user ID {created_user_id}")
                
                if created_section_id:
                    cursor.execute("DELETE FROM dbo.AdminsrationUnit WHERE UniqueID = ?", (created_section_id,))
                    print(f"🧹 Cleaned up section ID {created_section_id}")
                
                conn.commit()
            except Exception as e:
                print(f"⚠️  Cleanup error: {e}")
                conn.rollback()
            
            conn.close()
    
    return results.summary()


def test_5_transaction_rollback():
    """Test 5: Verify transaction rollback works correctly"""
    results = TestResults()
    
    print("\n" + "=" * 80)
    print("TEST 5: TRANSACTION ROLLBACK TEST")
    print("=" * 80)
    
    conn = None
    test_section_id = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Find a parent department
        cursor.execute("SELECT TOP 1 UniqueID FROM dbo.AdminsrationUnit WHERE Type = 325 ORDER BY UniqueID")
        dept = cursor.fetchone()
        
        if not dept:
            results.add_fail("Find parent department", "No department found")
            return False
        
        parent_dept_id = dept.UniqueID
        
        from api.services.section_admin_creator_service import (
            create_section_core,
            attach_section_admin_user
        )
        
        test_section_name = f"TEST_ROLLBACK_C4_{os.urandom(4).hex()}"
        
        try:
            # Create section
            section_id = create_section_core(
                conn=conn,
                section_name=test_section_name,
                parent_unit_id=parent_dept_id,
                created_by_user_id=None
            )
            
            test_section_id = section_id
            results.add_pass("Section created for rollback test")
            print(f"  Section ID: {section_id}")
            
            # Force an error by using invalid section_id in attach
            # (or we could test with duplicate username)
            try:
                # This should fail - non-existent section
                attach_section_admin_user(
                    conn=conn,
                    section_id=999999,  # Invalid ID
                    created_by_user_id=None
                )
                results.add_fail("Rollback trigger", "Attach should have failed with invalid section_id")
            except Exception as e:
                results.add_pass("Attach failed as expected (for rollback test)")
                print(f"  Error message: {str(e)}")
            
            # Rollback transaction
            conn.rollback()
            results.add_pass("Transaction rolled back")
            
            # Verify section was NOT persisted (due to rollback)
            cursor.execute("SELECT UniqueID FROM dbo.AdminsrationUnit WHERE UniqueID = ?", (section_id,))
            section_row = cursor.fetchone()
            
            if section_row is None:
                results.add_pass("Section correctly rolled back (not in database)")
                test_section_id = None  # Don't try to clean up
            else:
                results.add_fail("Rollback verification", "Section still exists after rollback")
        
        except Exception as e:
            results.add_fail("Rollback test execution", str(e))
            conn.rollback()
    
    except Exception as e:
        results.add_fail("Rollback test setup", str(e))
    
    finally:
        # Cleanup (in case rollback didn't work)
        if conn:
            if test_section_id:
                cursor = conn.cursor()
                try:
                    cursor.execute("DELETE FROM dbo.AdminsrationUnit WHERE UniqueID = ?", (test_section_id,))
                    conn.commit()
                    print(f"\n🧹 Cleaned up section ID {test_section_id} (rollback didn't work)")
                except Exception as e:
                    print(f"⚠️  Cleanup error: {e}")
            
            conn.close()
    
    return results.summary()


def test_6_endpoint_contract_unchanged():
    """Test 6: Verify endpoint contract remains unchanged after refactor"""
    results = TestResults()
    
    print("\n" + "=" * 80)
    print("TEST 6: ENDPOINT CONTRACT UNCHANGED")
    print("=" * 80)
    
    try:
        # Verify function signature
        from api.services.section_admin_creator_service import create_section_with_admin
        import inspect
        
        sig = inspect.signature(create_section_with_admin)
        params = list(sig.parameters.keys())
        
        # Should have exactly 2 parameters
        if len(params) == 2:
            results.add_pass("Function has 2 parameters")
        else:
            results.add_fail("Function signature", f"Expected 2 params, got {len(params)}")
        
        # Check parameter names
        if 'section_name' in params:
            results.add_pass("Has 'section_name' parameter")
        else:
            results.add_fail("Function parameters", "Missing 'section_name'")
        
        if 'parent_department_id' in params:
            results.add_pass("Has 'parent_department_id' parameter")
        else:
            results.add_fail("Function parameters", "Missing 'parent_department_id'")
        
        # Verify router still imports correctly
        try:
            from api.routers.admin_section_router import router
            results.add_pass("Router imports successfully")
        except ImportError as e:
            results.add_fail("Router import", str(e))
        
    except Exception as e:
        results.add_fail("Contract verification", str(e))
    
    return results.summary()


def run_all_tests():
    """Run all test suites"""
    print("\n" + "=" * 80)
    print("PHASE C — B-C4 — SERVICE SPLIT TEST SUITE")
    print("=" * 80)
    
    all_passed = True
    
    # Test 1: Import Split Functions
    if not test_1_import_split_functions():
        all_passed = False
    
    # Test 2: Core Function - Section Only
    if not test_2_core_function_creates_section_only():
        all_passed = False
    
    # Test 3: Attach Function - User Only
    if not test_3_attach_function_creates_user_only():
        all_passed = False
    
    # Test 4: Orchestration Function - Complete
    if not test_4_orchestration_function_complete():
        all_passed = False
    
    # Test 5: Transaction Rollback
    if not test_5_transaction_rollback():
        all_passed = False
    
    # Test 6: Endpoint Contract Unchanged
    if not test_6_endpoint_contract_unchanged():
        all_passed = False
    
    # Final Summary
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED - SERVICE SPLIT COMPLETE")
        print("\n📝 Key Benefits:")
        print("   ✅ create_section_core - Can create sections without users")
        print("   ✅ attach_section_admin_user - Can add users to existing sections")
        print("   ✅ create_section_with_admin - Orchestrates both (unchanged contract)")
        print("   ✅ Transaction safety maintained")
        print("   ✅ Independent function testing enabled")
    else:
        print("❌ SOME TESTS FAILED - REVIEW ERRORS ABOVE")
    print("=" * 80 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
