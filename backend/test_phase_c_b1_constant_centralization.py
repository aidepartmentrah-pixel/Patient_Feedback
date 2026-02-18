"""
PHASE C — B-C1 — TEST CONSTANT CENTRALIZATION
Tests for org unit type constants centralization.

Test Coverage:
1. Import Test - Verify constants module loads correctly
2. Static Scan Test - Verify no magic numbers remain
3. Runtime Section Creation Test - Verify section creation still works
4. DB Verification - Verify correct Type values in database
5. Other Services - Verify export services use constants correctly
"""

import sys
import os
import re
import pyodbc
from pathlib import Path

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


def test_1_import_constants():
    """Test 1: Verify constants module can be imported"""
    results = TestResults()
    
    print("\n" + "=" * 80)
    print("TEST 1: IMPORT CONSTANTS MODULE")
    print("=" * 80)
    
    try:
        from api.constants.org_unit_types import (
            ORG_TYPE_ADMINISTRATION,
            ORG_TYPE_DEPARTMENT,
            ORG_TYPE_SECTION,
            ORG_TYPE_NAME_MAP,
            ORG_TYPE_NAME_TO_CODE_MAP
        )
        
        # Verify values are correct
        if ORG_TYPE_ADMINISTRATION == 323:
            results.add_pass("ORG_TYPE_ADMINISTRATION = 323")
        else:
            results.add_fail("ORG_TYPE_ADMINISTRATION", f"Expected 323, got {ORG_TYPE_ADMINISTRATION}")
        
        if ORG_TYPE_SECTION == 324:
            results.add_pass("ORG_TYPE_SECTION = 324")
        else:
            results.add_fail("ORG_TYPE_SECTION", f"Expected 324, got {ORG_TYPE_SECTION}")
        
        if ORG_TYPE_DEPARTMENT == 325:
            results.add_pass("ORG_TYPE_DEPARTMENT = 325")
        else:
            results.add_fail("ORG_TYPE_DEPARTMENT", f"Expected 325, got {ORG_TYPE_DEPARTMENT}")
        
        # Verify mappings
        expected_name_map = {
            323: "ADMINISTRATION",
            325: "DEPARTMENT",
            324: "SECTION"
        }
        
        if ORG_TYPE_NAME_MAP == expected_name_map:
            results.add_pass("ORG_TYPE_NAME_MAP correct")
        else:
            results.add_fail("ORG_TYPE_NAME_MAP", f"Mapping mismatch: {ORG_TYPE_NAME_MAP}")
        
        expected_reverse_map = {
            "administration": 323,
            "department": 325,
            "section": 324
        }
        
        if ORG_TYPE_NAME_TO_CODE_MAP == expected_reverse_map:
            results.add_pass("ORG_TYPE_NAME_TO_CODE_MAP correct")
        else:
            results.add_fail("ORG_TYPE_NAME_TO_CODE_MAP", f"Mapping mismatch: {ORG_TYPE_NAME_TO_CODE_MAP}")
        
    except ImportError as e:
        results.add_fail("Import constants module", str(e))
    except Exception as e:
        results.add_fail("Import constants module", f"Unexpected error: {str(e)}")
    
    return results.summary()


def test_2_static_scan_no_magic_numbers():
    """Test 2: Scan for remaining magic numbers in section creation files"""
    results = TestResults()
    
    print("\n" + "=" * 80)
    print("TEST 2: STATIC SCAN FOR MAGIC NUMBERS")
    print("=" * 80)
    
    # Files that should use constants
    files_to_check = [
        "api/db_layer/section_admin_creator_db.py",
        "api/services/section_admin_recreate_service.py",
        "api/services/report_export_service.py",
        "api/services/multi_report_export_service.py",
        "api/services/multi_seasonal_export_service.py"
    ]
    
    # Patterns to search for hardcoded values
    # Look for numeric literals 323, 324, 325 NOT in comments or strings
    magic_number_pattern = re.compile(r'(?<!["\'])(?<![0-9])32[345](?![0-9])(?!["\'])')
    
    for file_path in files_to_check:
        full_path = Path(backend_path) / file_path
        
        if not full_path.exists():
            results.add_fail(f"File exists: {file_path}", "File not found")
            continue
        
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove comments and docstrings to avoid false positives
        lines = content.split('\n')
        code_lines = []
        in_docstring = False
        
        for line in lines:
            stripped = line.strip()
            
            # Toggle docstring state
            if '"""' in stripped or "'''" in stripped:
                in_docstring = not in_docstring
                continue
            
            # Skip comment lines and lines inside docstrings
            if in_docstring or stripped.startswith('#'):
                continue
            
            code_lines.append(line)
        
        code_content = '\n'.join(code_lines)
        
        # Search for magic numbers
        matches = magic_number_pattern.findall(code_content)
        
        if matches:
            results.add_fail(f"No magic numbers in {file_path}", f"Found hardcoded values: {matches}")
        else:
            results.add_pass(f"No magic numbers in {file_path}")
    
    return results.summary()


def test_3_verify_imports_used():
    """Test 3: Verify files import and use the constants"""
    results = TestResults()
    
    print("\n" + "=" * 80)
    print("TEST 3: VERIFY CONSTANTS IMPORTED AND USED")
    print("=" * 80)
    
    # Files that should import constants
    import_checks = {
        "api/db_layer/section_admin_creator_db.py": "ORG_TYPE_SECTION",
        "api/services/section_admin_recreate_service.py": "ORG_TYPE_SECTION",
        "api/services/report_export_service.py": "ORG_TYPE_ADMINISTRATION",
        "api/services/multi_report_export_service.py": "ORG_TYPE_ADMINISTRATION",
        "api/services/multi_seasonal_export_service.py": "ORG_TYPE_ADMINISTRATION"
    }
    
    for file_path, expected_constant in import_checks.items():
        full_path = Path(backend_path) / file_path
        
        if not full_path.exists():
            results.add_fail(f"File exists: {file_path}", "File not found")
            continue
        
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for import statement
        if "from ..constants.org_unit_types import" in content or "from api.constants.org_unit_types import" in content:
            results.add_pass(f"Imports constants: {file_path}")
        else:
            results.add_fail(f"Imports constants: {file_path}", "No import statement found")
            continue
        
        # Check if constant is used in code
        if expected_constant in content:
            results.add_pass(f"Uses {expected_constant}: {file_path}")
        else:
            results.add_fail(f"Uses {expected_constant}: {file_path}", "Constant not used in code")
    
    return results.summary()


def test_4_runtime_section_creation():
    """Test 4: Test section creation still works with constants"""
    results = TestResults()
    
    print("\n" + "=" * 80)
    print("TEST 4: RUNTIME SECTION CREATION TEST")
    print("=" * 80)
    
    conn = None
    created_section_id = None
    created_user_id = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Find a department to use as parent
        cursor.execute("SELECT TOP 1 UniqueID, Name FROM dbo.AdminsrationUnit WHERE Type = 325 ORDER BY UniqueID")
        dept = cursor.fetchone()
        
        if not dept:
            results.add_fail("Find parent department", "No department found in database")
            return False
        
        parent_dept_id = dept.UniqueID
        parent_dept_name = dept.Name
        print(f"Using parent department: {parent_dept_name} (ID: {parent_dept_id})")
        
        # Import and test section creation service
        from api.services.section_admin_creator_service import create_section_with_admin
        
        test_section_name = f"TEST_SECTION_C1_{os.urandom(4).hex()}"
        
        try:
            result = create_section_with_admin(test_section_name, parent_dept_id)
            
            created_section_id = result["section_id"]
            username = result["username"]
            password = result["temp_password"]
            
            results.add_pass(f"Section creation service executed")
            print(f"  Created section ID: {created_section_id}")
            print(f"  Username: {username}")
            print(f"  Password: {password}")
            
            # Verify section in database
            cursor.execute("""
                SELECT UniqueID, Name, Type, ParentID
                FROM dbo.AdminsrationUnit
                WHERE UniqueID = ?
            """, (created_section_id,))
            
            section_row = cursor.fetchone()
            
            if section_row:
                results.add_pass("Section exists in database")
                
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
            
            # Verify user was created
            cursor.execute("""
                SELECT UserID, Username
                FROM dbo.APP_Users
                WHERE Username = ?
            """, (username,))
            
            user_row = cursor.fetchone()
            
            if user_row:
                results.add_pass("Section admin user created")
                created_user_id = user_row.UserID
                
                # Verify scope assignment
                cursor.execute("""
                    SELECT s.UserID, s.OrgUnitID, s.OrgUnitType, r.RoleCode
                    FROM dbo.APP_UserRoleScope s
                    JOIN dbo.APP_Roles r ON s.RoleID = r.RoleID
                    WHERE s.UserID = ?
                """, (created_user_id,))
                
                scope_row = cursor.fetchone()
                
                if scope_row:
                    results.add_pass("User scope assigned")
                    
                    if scope_row.OrgUnitID == created_section_id:
                        results.add_pass("Scope linked to correct section")
                    else:
                        results.add_fail("Scope OrgUnitID", f"Expected {created_section_id}, got {scope_row.OrgUnitID}")
                    
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
            else:
                results.add_fail("User creation", "User not found in database")
        
        except Exception as e:
            results.add_fail("Section creation service", str(e))
        
    except Exception as e:
        results.add_fail("Runtime test setup", str(e))
    
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


def test_5_recreate_service_uses_constants():
    """Test 5: Verify recreate service validates Type using constants"""
    results = TestResults()
    
    print("\n" + "=" * 80)
    print("TEST 5: RECREATE SERVICE VALIDATION TEST")
    print("=" * 80)
    
    conn = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Find an administration unit (Type = 323) to test validation
        cursor.execute("SELECT TOP 1 UniqueID, Name, Type FROM dbo.AdminsrationUnit WHERE Type = 323 ORDER BY UniqueID")
        admin_unit = cursor.fetchone()
        
        if not admin_unit:
            print("⚠️  SKIP: No administration unit found for validation test")
            return True
        
        admin_id = admin_unit.UniqueID
        admin_name = admin_unit.Name
        print(f"Testing with administration unit: {admin_name} (ID: {admin_id}, Type: 323)")
        
        # Import recreate service
        from api.services.section_admin_recreate_service import recreate_section_admin_service
        
        try:
            # This should fail because Type != 324
            result = recreate_section_admin_service(admin_id)
            results.add_fail("Recreate service validation", "Should have rejected non-section unit")
        except Exception as e:
            error_msg = str(e)
            
            # Check if error message references the constant or correct value
            if "324" in error_msg and "not a section" in error_msg.lower():
                results.add_pass("Recreate service rejects non-section with Type validation")
                print(f"  Error message: {error_msg}")
            else:
                results.add_fail("Recreate service error message", f"Unexpected error: {error_msg}")
    
    except Exception as e:
        results.add_fail("Recreate service test setup", str(e))
    
    finally:
        if conn:
            conn.close()
    
    return results.summary()


def run_all_tests():
    """Run all test suites"""
    print("\n" + "=" * 80)
    print("PHASE C — B-C1 — CONSTANT CENTRALIZATION TEST SUITE")
    print("=" * 80)
    
    all_passed = True
    
    # Test 1: Import Test
    if not test_1_import_constants():
        all_passed = False
    
    # Test 2: Static Scan
    if not test_2_static_scan_no_magic_numbers():
        all_passed = False
    
    # Test 3: Verify Imports
    if not test_3_verify_imports_used():
        all_passed = False
    
    # Test 4: Runtime Section Creation
    if not test_4_runtime_section_creation():
        all_passed = False
    
    # Test 5: Recreate Service Validation
    if not test_5_recreate_service_uses_constants():
        all_passed = False
    
    # Final Summary
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED - CONSTANT CENTRALIZATION COMPLETE")
    else:
        print("❌ SOME TESTS FAILED - REVIEW ERRORS ABOVE")
    print("=" * 80 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
