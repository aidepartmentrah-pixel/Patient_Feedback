"""
PHASE C — B-C3 — TEST RESPONSE SCHEMA
Tests for enhanced section creation response schema.

Test Coverage:
1. Schema Import Test - Verify enhanced response schema
2. Schema Structure Test - Verify all 5 required fields
3. Service Returns Test - Verify service returns all fields
4. Router Import Test - Verify router uses enhanced schema
5. Runtime Creation Test - Create section and verify response shape
6. Field Type Test - Verify field types are correct
7. Field Name Match Test - Verify exact field name matches
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


def test_1_import_enhanced_schema():
    """Test 1: Verify enhanced response schema can be imported"""
    results = TestResults()
    
    print("\n" + "=" * 80)
    print("TEST 1: IMPORT ENHANCED RESPONSE SCHEMA")
    print("=" * 80)
    
    try:
        from api.schemas.section_creation_schemas import SectionCreateResponse
        
        results.add_pass("Import SectionCreateResponse")
        
        # Verify it's a Pydantic model
        from pydantic import BaseModel
        
        if issubclass(SectionCreateResponse, BaseModel):
            results.add_pass("SectionCreateResponse is BaseModel")
        else:
            results.add_fail("SectionCreateResponse type", "Not a BaseModel")
        
    except ImportError as e:
        results.add_fail("Import schema", str(e))
    except Exception as e:
        results.add_fail("Import schema", f"Unexpected error: {str(e)}")
    
    return results.summary()


def test_2_schema_has_all_fields():
    """Test 2: Verify response schema has all 5 required fields"""
    results = TestResults()
    
    print("\n" + "=" * 80)
    print("TEST 2: SCHEMA STRUCTURE - ALL 5 FIELDS")
    print("=" * 80)
    
    try:
        from api.schemas.section_creation_schemas import SectionCreateResponse
        
        # Check for all 5 required fields
        required_fields = [
            'section_id',
            'section_name',
            'parent_unit_id',
            'username',
            'temp_password'
        ]
        
        schema_fields = SectionCreateResponse.model_fields
        
        for field_name in required_fields:
            if field_name in schema_fields:
                results.add_pass(f"Schema has '{field_name}' field")
            else:
                results.add_fail("Schema fields", f"Missing '{field_name}'")
        
        # Check that we don't have the old 'password' field
        if 'password' in schema_fields:
            results.add_fail("Old field removal", "'password' field still exists (should be 'temp_password')")
        else:
            results.add_pass("Old 'password' field removed")
        
    except Exception as e:
        results.add_fail("Schema structure test", str(e))
    
    return results.summary()


def test_3_schema_field_types():
    """Test 3: Verify schema field types are correct"""
    results = TestResults()
    
    print("\n" + "=" * 80)
    print("TEST 3: SCHEMA FIELD TYPES")
    print("=" * 80)
    
    try:
        from api.schemas.section_creation_schemas import SectionCreateResponse
        
        schema_fields = SectionCreateResponse.model_fields
        
        # Check integer fields
        int_fields = ['section_id', 'parent_unit_id']
        for field_name in int_fields:
            field_info = schema_fields[field_name]
            if field_info.annotation == int:
                results.add_pass(f"'{field_name}' is int type")
            else:
                results.add_fail(f"Field type: {field_name}", f"Expected int, got {field_info.annotation}")
        
        # Check string fields
        str_fields = ['section_name', 'username', 'temp_password']
        for field_name in str_fields:
            field_info = schema_fields[field_name]
            if field_info.annotation == str:
                results.add_pass(f"'{field_name}' is str type")
            else:
                results.add_fail(f"Field type: {field_name}", f"Expected str, got {field_info.annotation}")
        
    except Exception as e:
        results.add_fail("Field type test", str(e))
    
    return results.summary()


def test_4_schema_validation():
    """Test 4: Verify schema validation works"""
    results = TestResults()
    
    print("\n" + "=" * 80)
    print("TEST 4: SCHEMA VALIDATION")
    print("=" * 80)
    
    try:
        from api.schemas.section_creation_schemas import SectionCreateResponse
        from pydantic import ValidationError
        
        # Test valid response
        try:
            response = SectionCreateResponse(
                section_id=101,
                section_name="Emergency Department",
                parent_unit_id=5,
                username="sec_101_admin",
                temp_password="Hospital2026!"
            )
            results.add_pass("Valid response accepted")
            print(f"  Section ID: {response.section_id}")
            print(f"  Section Name: {response.section_name}")
            print(f"  Parent Unit ID: {response.parent_unit_id}")
            print(f"  Username: {response.username}")
            print(f"  Temp Password: {response.temp_password}")
        except Exception as e:
            results.add_fail("Valid response", str(e))
        
        # Test missing field
        try:
            response = SectionCreateResponse(
                section_id=101,
                section_name="Test",
                username="sec_101_admin",
                temp_password="Hospital2026!"
                # Missing parent_unit_id
            )
            results.add_fail("Missing field validation", "Missing parent_unit_id was accepted")
        except ValidationError as e:
            results.add_pass("Missing field rejected")
            print(f"  Error: {e.errors()[0]['msg']}")
        
        # Test wrong type (string for int field)
        try:
            response = SectionCreateResponse(
                section_id="not_an_int",
                section_name="Test",
                parent_unit_id=5,
                username="sec_101_admin",
                temp_password="Hospital2026!"
            )
            results.add_fail("Type validation", "String for section_id was accepted")
        except ValidationError as e:
            results.add_pass("Wrong type rejected")
            print(f"  Error: {e.errors()[0]['msg']}")
        
    except Exception as e:
        results.add_fail("Schema validation test", str(e))
    
    return results.summary()


def test_5_service_returns_all_fields():
    """Test 5: Verify service returns all 5 fields"""
    results = TestResults()
    
    print("\n" + "=" * 80)
    print("TEST 5: SERVICE RETURNS ALL FIELDS")
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
            results.add_fail("Find parent department", "No department found")
            return False
        
        parent_dept_id = dept.UniqueID
        parent_dept_name = dept.Name
        print(f"Using parent department: {parent_dept_name} (ID: {parent_dept_id})")
        
        # Import and test service
        from api.services.section_admin_creator_service import create_section_with_admin
        
        test_section_name = f"TEST_SECTION_C3_{os.urandom(4).hex()}"
        
        try:
            result = create_section_with_admin(test_section_name, parent_dept_id)
            
            # Check all 5 required fields are present
            required_fields = ['section_id', 'section_name', 'parent_unit_id', 'username', 'temp_password']
            
            for field_name in required_fields:
                if field_name in result:
                    results.add_pass(f"Service returns '{field_name}'")
                else:
                    results.add_fail("Service return fields", f"Missing '{field_name}'")
            
            # Verify values match input
            created_section_id = result.get("section_id")
            
            if result.get("section_name") == test_section_name:
                results.add_pass("section_name matches input")
            else:
                results.add_fail("section_name value", f"Expected '{test_section_name}', got '{result.get('section_name')}'")
            
            if result.get("parent_unit_id") == parent_dept_id:
                results.add_pass("parent_unit_id matches input")
            else:
                results.add_fail("parent_unit_id value", f"Expected {parent_dept_id}, got {result.get('parent_unit_id')}")
            
            # Verify username format
            expected_username = f"sec_{created_section_id}_admin"
            if result.get("username") == expected_username:
                results.add_pass("username follows correct format")
            else:
                results.add_fail("username format", f"Expected '{expected_username}', got '{result.get('username')}'")
            
            # Get user ID for cleanup
            cursor.execute("SELECT UserID FROM dbo.APP_Users WHERE Username = ?", (result.get("username"),))
            user_row = cursor.fetchone()
            if user_row:
                created_user_id = user_row.UserID
        
        except Exception as e:
            results.add_fail("Service execution", str(e))
    
    except Exception as e:
        results.add_fail("Service test setup", str(e))
    
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


def test_6_router_uses_enhanced_schema():
    """Test 6: Verify router uses enhanced response schema"""
    results = TestResults()
    
    print("\n" + "=" * 80)
    print("TEST 6: ROUTER USES ENHANCED SCHEMA")
    print("=" * 80)
    
    try:
        # Check router file
        router_file = Path(backend_path) / "api" / "routers" / "admin_section_router.py"
        
        if not router_file.exists():
            results.add_fail("Router file exists", "admin_section_router.py not found")
            return False
        
        with open(router_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check router returns all 5 fields
        required_return_fields = ['section_id', 'section_name', 'parent_unit_id', 'username', 'temp_password']
        
        for field_name in required_return_fields:
            if f'"{field_name}"' in content or f"'{field_name}'" in content:
                results.add_pass(f"Router returns '{field_name}'")
            else:
                results.add_fail("Router return fields", f"Missing '{field_name}' in return statement")
        
        # Check that old 'password' field is replaced
        # Look for exact patterns to avoid false positives
        if '"password": result' in content or "'password': result" in content:
            results.add_fail("Old field in router", "'password' field still used (should be 'temp_password')")
        else:
            results.add_pass("Old 'password' field replaced with 'temp_password'")
        
        # Verify router imports without errors
        try:
            from api.routers.admin_section_router import router
            results.add_pass("Router imports without errors")
        except ImportError as e:
            results.add_fail("Router import", str(e))
        
    except Exception as e:
        results.add_fail("Router test", str(e))
    
    return results.summary()


def test_7_response_model_in_decorator():
    """Test 7: Verify response_model is set in router decorator"""
    results = TestResults()
    
    print("\n" + "=" * 80)
    print("TEST 7: RESPONSE MODEL IN DECORATOR")
    print("=" * 80)
    
    try:
        router_file = Path(backend_path) / "api" / "routers" / "admin_section_router.py"
        
        with open(router_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for response_model=SectionCreateResponse
        if "response_model=SectionCreateResponse" in content:
            results.add_pass("response_model=SectionCreateResponse in decorator")
        else:
            results.add_fail("Decorator response_model", "response_model not set or incorrect")
        
        # Verify no leftover old response model names
        if "CreateSectionResponse" in content and "SectionCreateResponse" not in content:
            results.add_fail("Old response model", "Old CreateSectionResponse model name still used")
        else:
            results.add_pass("Using correct response model name")
        
    except Exception as e:
        results.add_fail("Decorator test", str(e))
    
    return results.summary()


def test_8_recreate_admin_response_consistency():
    """Test 8: Verify recreate admin response uses temp_password consistently"""
    results = TestResults()
    
    print("\n" + "=" * 80)
    print("TEST 8: RECREATE ADMIN RESPONSE CONSISTENCY")
    print("=" * 80)
    
    try:
        from api.schemas.section_creation_schemas import SectionRecreateAdminResponse
        
        schema_fields = SectionRecreateAdminResponse.model_fields
        
        # Check temp_password exists
        if 'temp_password' in schema_fields:
            results.add_pass("SectionRecreateAdminResponse has 'temp_password'")
        else:
            results.add_fail("Recreate response field", "Missing 'temp_password'")
        
        # Check old 'password' field is removed
        if 'password' in schema_fields:
            results.add_fail("Old field in recreate response", "'password' field still exists")
        else:
            results.add_pass("Old 'password' field removed from recreate response")
        
        # Check recreate router
        router_file = Path(backend_path) / "api" / "routers" / "admin_section_admin_recreate_router.py"
        
        if router_file.exists():
            with open(router_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if '"temp_password"' in content or "'temp_password'" in content:
                results.add_pass("Recreate router returns 'temp_password'")
            else:
                results.add_fail("Recreate router return", "Not returning 'temp_password'")
        
    except Exception as e:
        results.add_fail("Recreate admin consistency test", str(e))
    
    return results.summary()


def run_all_tests():
    """Run all test suites"""
    print("\n" + "=" * 80)
    print("PHASE C — B-C3 — RESPONSE SCHEMA TEST SUITE")
    print("=" * 80)
    
    all_passed = True
    
    # Test 1: Import Enhanced Schema
    if not test_1_import_enhanced_schema():
        all_passed = False
    
    # Test 2: Schema Has All Fields
    if not test_2_schema_has_all_fields():
        all_passed = False
    
    # Test 3: Schema Field Types
    if not test_3_schema_field_types():
        all_passed = False
    
    # Test 4: Schema Validation
    if not test_4_schema_validation():
        all_passed = False
    
    # Test 5: Service Returns All Fields
    if not test_5_service_returns_all_fields():
        all_passed = False
    
    # Test 6: Router Uses Enhanced Schema
    if not test_6_router_uses_enhanced_schema():
        all_passed = False
    
    # Test 7: Response Model in Decorator
    if not test_7_response_model_in_decorator():
        all_passed = False
    
    # Test 8: Recreate Admin Response Consistency
    if not test_8_recreate_admin_response_consistency():
        all_passed = False
    
    # Final Summary
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED - RESPONSE SCHEMA COMPLETE")
        print("\n📝 NOTE: To verify in OpenAPI docs, start the server and:")
        print("   1. Visit http://localhost:8000/docs")
        print("   2. Check /api/admin/create-section-with-admin endpoint")
        print("   3. Verify response schema shows all 5 fields")
        print("   4. Test actual creation and verify response shape")
    else:
        print("❌ SOME TESTS FAILED - REVIEW ERRORS ABOVE")
    print("=" * 80 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
