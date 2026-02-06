"""
PHASE B — B-B8 — SCHEMA VALIDATION TESTS

Test suite for Settings Users Pydantic schemas.
Tests validation rules and model behavior.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pydantic import ValidationError
from backend.api.schemas.settings_users_models import (
    CreateUserRequest,
    UpdateUserIdentityRequest,
    UpdateUserPasswordRequest,
    SettingsUserListItemResponse,
    CreateUserResponse
)


def test_create_user_request_valid():
    """Test 1: CreateUserRequest with valid data."""
    print("\n" + "="*60)
    print("TEST 1: CreateUserRequest Valid Case")
    print("="*60)
    
    try:
        # Create valid request
        data = {
            "username": "test_user",
            "password": "Pass123!",
            "display_name": "Test User",
            "department_display_name": "Test Dept",
            "role_id": 5,
            "org_unit_id": 10
        }
        
        request = CreateUserRequest(**data)
        
        print(f"✓ Model parsed successfully")
        print(f"  username: {request.username}")
        print(f"  password: {'*' * len(request.password)}")
        print(f"  display_name: {request.display_name}")
        print(f"  role_id: {request.role_id}")
        print(f"  org_unit_id: {request.org_unit_id}")
        
        assert request.username == data["username"]
        assert request.password == data["password"]
        assert request.role_id == data["role_id"]
        assert request.org_unit_id == data["org_unit_id"]
        
        print("\n✓ TEST 1 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 1 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_create_user_request_invalid_role_id():
    """Test 2: CreateUserRequest with invalid role_id (0)."""
    print("\n" + "="*60)
    print("TEST 2: CreateUserRequest Invalid role_id")
    print("="*60)
    
    try:
        # Try to create with role_id = 0
        data = {
            "username": "test_user",
            "password": "Pass123!",
            "role_id": 0,  # Invalid
            "org_unit_id": 10
        }
        
        print(f"Attempting to create with role_id=0...")
        
        try:
            request = CreateUserRequest(**data)
            print(f"\n✗ TEST 2 FAILED: role_id=0 was accepted")
            return False
        except ValidationError as e:
            print(f"✓ ValidationError raised as expected")
            print(f"  Error: {e.errors()[0]['msg']}")
            
        print("\n✓ TEST 2 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 2 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_create_user_request_short_username():
    """Test 3: CreateUserRequest with username too short (< 3 chars)."""
    print("\n" + "="*60)
    print("TEST 3: CreateUserRequest Short Username")
    print("="*60)
    
    try:
        # Try to create with username < 3 chars
        data = {
            "username": "ab",  # Only 2 chars
            "password": "Pass123!",
            "role_id": 5,
            "org_unit_id": 10
        }
        
        print(f"Attempting to create with username='ab' (2 chars)...")
        
        try:
            request = CreateUserRequest(**data)
            print(f"\n✗ TEST 3 FAILED: Short username was accepted")
            return False
        except ValidationError as e:
            print(f"✓ ValidationError raised as expected")
            print(f"  Error: {e.errors()[0]['msg']}")
            
        print("\n✓ TEST 3 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 3 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_create_user_request_short_password():
    """Test 4: CreateUserRequest with password too short (< 6 chars)."""
    print("\n" + "="*60)
    print("TEST 4: CreateUserRequest Short Password")
    print("="*60)
    
    try:
        # Try to create with password < 6 chars
        data = {
            "username": "test_user",
            "password": "12345",  # Only 5 chars
            "role_id": 5,
            "org_unit_id": 10
        }
        
        print(f"Attempting to create with password='12345' (5 chars)...")
        
        try:
            request = CreateUserRequest(**data)
            print(f"\n✗ TEST 4 FAILED: Short password was accepted")
            return False
        except ValidationError as e:
            print(f"✓ ValidationError raised as expected")
            print(f"  Error: {e.errors()[0]['msg']}")
            
        print("\n✓ TEST 4 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 4 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_update_identity_request_both_none():
    """Test 5: UpdateUserIdentityRequest with both fields None."""
    print("\n" + "="*60)
    print("TEST 5: UpdateUserIdentityRequest Both Fields None")
    print("="*60)
    
    try:
        # Try to create with both fields None
        data = {
            "display_name": None,
            "department_display_name": None
        }
        
        print(f"Attempting to create with both fields=None...")
        
        try:
            request = UpdateUserIdentityRequest(**data)
            print(f"\n✗ TEST 5 FAILED: Both fields None was accepted")
            return False
        except ValidationError as e:
            print(f"✓ ValidationError raised as expected")
            error_msg = str(e.errors()[0]['msg']) if e.errors() else str(e)
            print(f"  Error: {error_msg}")
            
        print("\n✓ TEST 5 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 5 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_update_identity_request_partial_display_name():
    """Test 6: UpdateUserIdentityRequest with only display_name."""
    print("\n" + "="*60)
    print("TEST 6: UpdateUserIdentityRequest Partial - display_name Only")
    print("="*60)
    
    try:
        # Create with only display_name
        data = {
            "display_name": "Updated Name",
            "department_display_name": None
        }
        
        request = UpdateUserIdentityRequest(**data)
        
        print(f"✓ Model parsed successfully with partial update")
        print(f"  display_name: {request.display_name}")
        print(f"  department_display_name: {request.department_display_name}")
        
        assert request.display_name == "Updated Name"
        assert request.department_display_name is None
        
        print("\n✓ TEST 6 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 6 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_update_identity_request_partial_department():
    """Test 7: UpdateUserIdentityRequest with only department_display_name."""
    print("\n" + "="*60)
    print("TEST 7: UpdateUserIdentityRequest Partial - department_display_name Only")
    print("="*60)
    
    try:
        # Create with only department_display_name
        data = {
            "display_name": None,
            "department_display_name": "Updated Dept"
        }
        
        request = UpdateUserIdentityRequest(**data)
        
        print(f"✓ Model parsed successfully with partial update")
        print(f"  display_name: {request.display_name}")
        print(f"  department_display_name: {request.department_display_name}")
        
        assert request.display_name is None
        assert request.department_display_name == "Updated Dept"
        
        print("\n✓ TEST 7 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 7 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_update_password_request_valid():
    """Test 8: UpdateUserPasswordRequest with valid password."""
    print("\n" + "="*60)
    print("TEST 8: UpdateUserPasswordRequest Valid")
    print("="*60)
    
    try:
        # Create with valid password
        data = {
            "new_password": "NewPass123!"
        }
        
        request = UpdateUserPasswordRequest(**data)
        
        print(f"✓ Model parsed successfully")
        print(f"  new_password: {'*' * len(request.new_password)}")
        
        assert request.new_password == "NewPass123!"
        
        print("\n✓ TEST 8 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 8 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_update_password_request_empty():
    """Test 9: UpdateUserPasswordRequest with empty password."""
    print("\n" + "="*60)
    print("TEST 9: UpdateUserPasswordRequest Empty Password")
    print("="*60)
    
    try:
        # Try to create with empty password
        data = {
            "new_password": ""
        }
        
        print(f"Attempting to create with empty password...")
        
        try:
            request = UpdateUserPasswordRequest(**data)
            print(f"\n✗ TEST 9 FAILED: Empty password was accepted")
            return False
        except ValidationError as e:
            print(f"✓ ValidationError raised as expected")
            # Check for either min_length or custom validation error
            errors = e.errors()
            print(f"  Error count: {len(errors)}")
            for err in errors:
                print(f"  Error: {err['msg']}")
            
        print("\n✓ TEST 9 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 9 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_update_password_request_whitespace():
    """Test 10: UpdateUserPasswordRequest with whitespace only."""
    print("\n" + "="*60)
    print("TEST 10: UpdateUserPasswordRequest Whitespace Only")
    print("="*60)
    
    try:
        # Try to create with whitespace only
        data = {
            "new_password": "      "
        }
        
        print(f"Attempting to create with whitespace-only password...")
        
        try:
            request = UpdateUserPasswordRequest(**data)
            print(f"\n✗ TEST 10 FAILED: Whitespace password was accepted")
            return False
        except ValidationError as e:
            print(f"✓ ValidationError raised as expected")
            error_msg = str(e.errors()[0]['msg']) if e.errors() else str(e)
            print(f"  Error: {error_msg}")
            
        print("\n✓ TEST 10 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 10 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_update_password_request_short():
    """Test 11: UpdateUserPasswordRequest with password too short (< 6 chars)."""
    print("\n" + "="*60)
    print("TEST 11: UpdateUserPasswordRequest Short Password")
    print("="*60)
    
    try:
        # Try to create with password < 6 chars
        data = {
            "new_password": "12345"  # Only 5 chars
        }
        
        print(f"Attempting to create with password='12345' (5 chars)...")
        
        try:
            request = UpdateUserPasswordRequest(**data)
            print(f"\n✗ TEST 11 FAILED: Short password was accepted")
            return False
        except ValidationError as e:
            print(f"✓ ValidationError raised as expected")
            print(f"  Error: {e.errors()[0]['msg']}")
            
        print("\n✓ TEST 11 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 11 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_settings_user_list_item_response_serialization():
    """Test 12: SettingsUserListItemResponse serialization."""
    print("\n" + "="*60)
    print("TEST 12: SettingsUserListItemResponse Serialization")
    print("="*60)
    
    try:
        # Create response model
        data = {
            "user_id": 123,
            "username": "test_user",
            "display_name": "Test User",
            "department_display_name": "Test Dept",
            "role_name": "SECTION_ADMIN",
            "org_unit_name": "Test Org Unit",
            "is_active": True
        }
        
        response = SettingsUserListItemResponse(**data)
        
        print(f"✓ Model created successfully")
        
        # Serialize to dict
        dumped = response.model_dump()
        
        print(f"✓ Model serialized successfully")
        
        # Check keys present
        required_keys = ["user_id", "username", "display_name", "department_display_name", 
                        "role_name", "org_unit_name", "is_active"]
        
        for key in required_keys:
            assert key in dumped, f"Missing key: {key}"
            print(f"  ✓ {key}: {dumped[key]}")
        
        print("\n✓ TEST 12 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 12 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_create_user_response():
    """Test 13: CreateUserResponse model."""
    print("\n" + "="*60)
    print("TEST 13: CreateUserResponse Model")
    print("="*60)
    
    try:
        # Create response model
        data = {
            "user_id": 456,
            "username": "new_user"
        }
        
        response = CreateUserResponse(**data)
        
        print(f"✓ Model created successfully")
        print(f"  user_id: {response.user_id}")
        print(f"  username: {response.username}")
        
        assert response.user_id == 456
        assert response.username == "new_user"
        
        # Serialize
        dumped = response.model_dump()
        assert "user_id" in dumped
        assert "username" in dumped
        
        print(f"✓ Model serialized successfully")
        
        print("\n✓ TEST 13 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 13 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*60)
    print("PHASE B — B-B8 — SCHEMA VALIDATION TEST SUITE")
    print("="*60)
    
    tests = [
        ("CreateUserRequest Valid Case", test_create_user_request_valid),
        ("CreateUserRequest Invalid role_id", test_create_user_request_invalid_role_id),
        ("CreateUserRequest Short Username", test_create_user_request_short_username),
        ("CreateUserRequest Short Password", test_create_user_request_short_password),
        ("UpdateUserIdentityRequest Both Fields None", test_update_identity_request_both_none),
        ("UpdateUserIdentityRequest Partial - display_name", test_update_identity_request_partial_display_name),
        ("UpdateUserIdentityRequest Partial - department", test_update_identity_request_partial_department),
        ("UpdateUserPasswordRequest Valid", test_update_password_request_valid),
        ("UpdateUserPasswordRequest Empty Password", test_update_password_request_empty),
        ("UpdateUserPasswordRequest Whitespace Only", test_update_password_request_whitespace),
        ("UpdateUserPasswordRequest Short Password", test_update_password_request_short),
        ("SettingsUserListItemResponse Serialization", test_settings_user_list_item_response_serialization),
        ("CreateUserResponse Model", test_create_user_response),
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
