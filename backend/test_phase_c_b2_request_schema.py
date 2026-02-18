"""
PHASE C — B-C2 — TEST REQUEST SCHEMA
Tests for section creation request schema validation.

Test Coverage:
1. Schema Import Test - Verify schemas can be imported
2. Schema Structure Test - Verify fields and constraints
3. Validation Tests - Test various invalid inputs
4. Router Import Test - Verify routers use new schemas
5. API Test - Test actual endpoint validation (requires running server)
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


def test_1_import_schemas():
    """Test 1: Verify schemas can be imported"""
    results = TestResults()
    
    print("\n" + "=" * 80)
    print("TEST 1: IMPORT SCHEMAS")
    print("=" * 80)
    
    try:
        from api.schemas.section_creation_schemas import (
            SectionCreateRequest,
            SectionCreateResponse,
            SectionRecreateAdminResponse
        )
        
        results.add_pass("Import SectionCreateRequest")
        results.add_pass("Import SectionCreateResponse")
        results.add_pass("Import SectionRecreateAdminResponse")
        
        # Verify they are Pydantic models
        from pydantic import BaseModel
        
        if issubclass(SectionCreateRequest, BaseModel):
            results.add_pass("SectionCreateRequest is BaseModel")
        else:
            results.add_fail("SectionCreateRequest type", "Not a BaseModel")
        
        if issubclass(SectionCreateResponse, BaseModel):
            results.add_pass("SectionCreateResponse is BaseModel")
        else:
            results.add_fail("SectionCreateResponse type", "Not a BaseModel")
        
        if issubclass(SectionRecreateAdminResponse, BaseModel):
            results.add_pass("SectionRecreateAdminResponse is BaseModel")
        else:
            results.add_fail("SectionRecreateAdminResponse type", "Not a BaseModel")
        
    except ImportError as e:
        results.add_fail("Import schemas", str(e))
    except Exception as e:
        results.add_fail("Import schemas", f"Unexpected error: {str(e)}")
    
    return results.summary()


def test_2_schema_structure():
    """Test 2: Verify schema structure and fields"""
    results = TestResults()
    
    print("\n" + "=" * 80)
    print("TEST 2: SCHEMA STRUCTURE")
    print("=" * 80)
    
    try:
        from api.schemas.section_creation_schemas import (
            SectionCreateRequest,
            SectionCreateResponse
        )
        
        # Check SectionCreateRequest fields
        request_fields = SectionCreateRequest.model_fields
        
        if 'section_name' in request_fields:
            results.add_pass("SectionCreateRequest has section_name field")
        else:
            results.add_fail("SectionCreateRequest fields", "Missing section_name")
        
        if 'parent_unit_id' in request_fields:
            results.add_pass("SectionCreateRequest has parent_unit_id field")
        else:
            results.add_fail("SectionCreateRequest fields", "Missing parent_unit_id")
        
        # Check SectionCreateResponse fields
        response_fields = SectionCreateResponse.model_fields
        
        expected_response_fields = ['section_id', 'username', 'password']
        for field_name in expected_response_fields:
            if field_name in response_fields:
                results.add_pass(f"SectionCreateResponse has {field_name} field")
            else:
                results.add_fail("SectionCreateResponse fields", f"Missing {field_name}")
        
    except Exception as e:
        results.add_fail("Schema structure test", str(e))
    
    return results.summary()


def test_3_validation_empty_name():
    """Test 3: Validate empty section name is rejected"""
    results = TestResults()
    
    print("\n" + "=" * 80)
    print("TEST 3: VALIDATION - EMPTY NAME")
    print("=" * 80)
    
    try:
        from api.schemas.section_creation_schemas import SectionCreateRequest
        from pydantic import ValidationError
        
        # Test empty string
        try:
            request = SectionCreateRequest(
                section_name="",
                parent_unit_id=5
            )
            results.add_fail("Empty name validation", "Empty string was accepted")
        except ValidationError as e:
            results.add_pass("Empty string rejected")
            print(f"  Error: {e.errors()[0]['msg']}")
        
        # Test whitespace only
        try:
            request = SectionCreateRequest(
                section_name="   ",
                parent_unit_id=5
            )
            results.add_fail("Whitespace validation", "Whitespace-only string was accepted")
        except ValidationError as e:
            results.add_pass("Whitespace-only string rejected")
            print(f"  Error: {e.errors()[0]['msg']}")
        
        # Test single character (min_length=2)
        try:
            request = SectionCreateRequest(
                section_name="A",
                parent_unit_id=5
            )
            results.add_fail("Min length validation", "Single character was accepted")
        except ValidationError as e:
            results.add_pass("Single character rejected (min_length=2)")
            print(f"  Error: {e.errors()[0]['msg']}")
        
    except Exception as e:
        results.add_fail("Validation test setup", str(e))
    
    return results.summary()


def test_4_validation_bad_parent_id():
    """Test 4: Validate invalid parent_unit_id is rejected"""
    results = TestResults()
    
    print("\n" + "=" * 80)
    print("TEST 4: VALIDATION - BAD PARENT ID")
    print("=" * 80)
    
    try:
        from api.schemas.section_creation_schemas import SectionCreateRequest
        from pydantic import ValidationError
        
        # Test zero
        try:
            request = SectionCreateRequest(
                section_name="Test Section",
                parent_unit_id=0
            )
            results.add_fail("Zero parent_id validation", "Zero was accepted")
        except ValidationError as e:
            results.add_pass("Zero parent_id rejected")
            print(f"  Error: {e.errors()[0]['msg']}")
        
        # Test negative number
        try:
            request = SectionCreateRequest(
                section_name="Test Section",
                parent_unit_id=-1
            )
            results.add_fail("Negative parent_id validation", "Negative number was accepted")
        except ValidationError as e:
            results.add_pass("Negative parent_id rejected")
            print(f"  Error: {e.errors()[0]['msg']}")
        
    except Exception as e:
        results.add_fail("Validation test setup", str(e))
    
    return results.summary()


def test_5_validation_valid_input():
    """Test 5: Validate valid input is accepted"""
    results = TestResults()
    
    print("\n" + "=" * 80)
    print("TEST 5: VALIDATION - VALID INPUT")
    print("=" * 80)
    
    try:
        from api.schemas.section_creation_schemas import SectionCreateRequest
        
        # Test valid input
        try:
            request = SectionCreateRequest(
                section_name="Emergency Department Section A",
                parent_unit_id=5
            )
            results.add_pass("Valid input accepted")
            print(f"  Section Name: {request.section_name}")
            print(f"  Parent Unit ID: {request.parent_unit_id}")
        except Exception as e:
            results.add_fail("Valid input", f"Valid input was rejected: {str(e)}")
        
        # Test with whitespace (should be stripped)
        try:
            request = SectionCreateRequest(
                section_name="  Test Section  ",
                parent_unit_id=10
            )
            
            if request.section_name == "Test Section":
                results.add_pass("Whitespace stripped correctly")
            else:
                results.add_fail("Whitespace stripping", f"Expected 'Test Section', got '{request.section_name}'")
        except Exception as e:
            results.add_fail("Whitespace stripping", str(e))
        
        # Test max length boundary (100 characters)
        try:
            max_name = "A" * 100
            request = SectionCreateRequest(
                section_name=max_name,
                parent_unit_id=5
            )
            results.add_pass("Max length (100 chars) accepted")
        except Exception as e:
            results.add_fail("Max length validation", str(e))
        
        # Test over max length (101 characters)
        try:
            from pydantic import ValidationError
            over_max_name = "A" * 101
            request = SectionCreateRequest(
                section_name=over_max_name,
                parent_unit_id=5
            )
            results.add_fail("Over max length validation", "101 characters was accepted")
        except ValidationError as e:
            results.add_pass("Over max length (101 chars) rejected")
            print(f"  Error: {e.errors()[0]['msg']}")
        
    except Exception as e:
        results.add_fail("Valid input test", str(e))
    
    return results.summary()


def test_6_router_imports_schemas():
    """Test 6: Verify routers import and use the new schemas"""
    results = TestResults()
    
    print("\n" + "=" * 80)
    print("TEST 6: ROUTER SCHEMA USAGE")
    print("=" * 80)
    
    try:
        # Check admin_section_router imports schema
        router_file = Path(backend_path) / "api" / "routers" / "admin_section_router.py"
        
        if not router_file.exists():
            results.add_fail("Router file exists", "admin_section_router.py not found")
            return False
        
        with open(router_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for schema imports
        if "from ..schemas.section_creation_schemas import" in content:
            results.add_pass("Router imports section_creation_schemas")
        else:
            results.add_fail("Router imports", "No import from section_creation_schemas")
        
        # Check for SectionCreateRequest usage
        if "SectionCreateRequest" in content:
            results.add_pass("Router uses SectionCreateRequest")
        else:
            results.add_fail("Router usage", "SectionCreateRequest not found in router")
        
        # Check for SectionCreateResponse usage
        if "SectionCreateResponse" in content:
            results.add_pass("Router uses SectionCreateResponse")
        else:
            results.add_fail("Router usage", "SectionCreateResponse not found in router")
        
        # Check that old inline models are removed
        if "class CreateSectionRequest(BaseModel):" in content:
            results.add_fail("Old inline model", "CreateSectionRequest inline model still exists")
        else:
            results.add_pass("Old inline models removed")
        
        # Verify router can be imported
        try:
            from api.routers.admin_section_router import router
            results.add_pass("Router imports without errors")
        except ImportError as e:
            results.add_fail("Router import", str(e))
        
    except Exception as e:
        results.add_fail("Router schema usage test", str(e))
    
    return results.summary()


def test_7_response_schema_validation():
    """Test 7: Verify response schema validation"""
    results = TestResults()
    
    print("\n" + "=" * 80)
    print("TEST 7: RESPONSE SCHEMA VALIDATION")
    print("=" * 80)
    
    try:
        from api.schemas.section_creation_schemas import SectionCreateResponse
        
        # Test valid response
        try:
            response = SectionCreateResponse(
                section_id=101,
                username="sec_101_admin",
                password="Hospital2026!"
            )
            results.add_pass("Valid response accepted")
            print(f"  Section ID: {response.section_id}")
            print(f"  Username: {response.username}")
            print(f"  Password: {response.password}")
        except Exception as e:
            results.add_fail("Valid response", str(e))
        
        # Test missing fields
        try:
            from pydantic import ValidationError
            response = SectionCreateResponse(
                section_id=101
            )
            results.add_fail("Missing fields validation", "Incomplete response was accepted")
        except ValidationError as e:
            results.add_pass("Missing fields rejected")
            print(f"  Errors: {len(e.errors())} fields missing")
        
    except Exception as e:
        results.add_fail("Response schema test", str(e))
    
    return results.summary()


def test_8_recreate_admin_response_schema():
    """Test 8: Verify recreate admin response schema"""
    results = TestResults()
    
    print("\n" + "=" * 80)
    print("TEST 8: RECREATE ADMIN RESPONSE SCHEMA")
    print("=" * 80)
    
    try:
        from api.schemas.section_creation_schemas import SectionRecreateAdminResponse
        
        # Test valid recreate response
        try:
            response = SectionRecreateAdminResponse(
                section_id=101,
                username="sec_101_admin_v2",
                password="Hospital2026!"
            )
            results.add_pass("Valid recreate response accepted")
            print(f"  Section ID: {response.section_id}")
            print(f"  Username: {response.username}")
            print(f"  Password: {response.password}")
        except Exception as e:
            results.add_fail("Valid recreate response", str(e))
        
        # Check recreate admin router imports schema
        router_file = Path(backend_path) / "api" / "routers" / "admin_section_admin_recreate_router.py"
        
        if router_file.exists():
            with open(router_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if "SectionRecreateAdminResponse" in content:
                results.add_pass("Recreate router uses SectionRecreateAdminResponse")
            else:
                results.add_fail("Recreate router usage", "SectionRecreateAdminResponse not found")
        else:
            results.add_fail("Recreate router file", "File not found")
        
    except Exception as e:
        results.add_fail("Recreate admin response test", str(e))
    
    return results.summary()


def run_all_tests():
    """Run all test suites"""
    print("\n" + "=" * 80)
    print("PHASE C — B-C2 — REQUEST SCHEMA TEST SUITE")
    print("=" * 80)
    
    all_passed = True
    
    # Test 1: Import Schemas
    if not test_1_import_schemas():
        all_passed = False
    
    # Test 2: Schema Structure
    if not test_2_schema_structure():
        all_passed = False
    
    # Test 3: Empty Name Validation
    if not test_3_validation_empty_name():
        all_passed = False
    
    # Test 4: Bad Parent ID Validation
    if not test_4_validation_bad_parent_id():
        all_passed = False
    
    # Test 5: Valid Input
    if not test_5_validation_valid_input():
        all_passed = False
    
    # Test 6: Router Schema Usage
    if not test_6_router_imports_schemas():
        all_passed = False
    
    # Test 7: Response Schema
    if not test_7_response_schema_validation():
        all_passed = False
    
    # Test 8: Recreate Admin Response Schema
    if not test_8_recreate_admin_response_schema():
        all_passed = False
    
    # Final Summary
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED - REQUEST SCHEMA VALIDATION COMPLETE")
        print("\n📝 NOTE: To test endpoint validation (422 errors), start the server and:")
        print("   1. Visit http://localhost:8000/docs")
        print("   2. Try the /api/admin/create-section-with-admin endpoint")
        print("   3. Test with invalid payloads (empty name, parent_id=0)")
        print("   4. Verify you get 422 validation errors")
    else:
        print("❌ SOME TESTS FAILED - REVIEW ERRORS ABOVE")
    print("=" * 80 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
