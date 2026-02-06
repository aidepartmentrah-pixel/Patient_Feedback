"""
Test Phase A Step 3: Extend CurrentUser Model
===============================================

Tests that CurrentUser model has new display identity fields:
- display_name: Optional[str]
- department_display_name: Optional[str]

Tests:
- Fields are optional (can be None)
- Fields accept string values
- Backward compatibility (old code without these fields works)
- Pydantic validation
- Model serialization
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.api.schemas.auth_models import CurrentUser, UserScope


def test_fields_exist():
    """Test 1: Verify display fields exist on CurrentUser model."""
    print("\n" + "="*60)
    print("TEST 1: Display Fields Exist")
    print("="*60)
    
    try:
        # Check if fields exist in model
        assert hasattr(CurrentUser, 'model_fields'), "CurrentUser should be a Pydantic model"
        fields = CurrentUser.model_fields
        
        assert 'display_name' in fields, "display_name field missing"
        assert 'department_display_name' in fields, "department_display_name field missing"
        
        print("✓ display_name field exists")
        print("✓ department_display_name field exists")
        print("\n✓ TEST 1 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 1 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 1 ERROR: {str(e)}")
        return False


def test_fields_are_optional():
    """Test 2: Verify display fields are optional (can be None)."""
    print("\n" + "="*60)
    print("TEST 2: Fields Are Optional")
    print("="*60)
    
    try:
        # Create CurrentUser without display fields
        user = CurrentUser(
            user_id=1,
            username="test_user",
            is_active=True,
            scopes=[]
        )
        
        assert user.display_name is None, "display_name should default to None"
        assert user.department_display_name is None, "department_display_name should default to None"
        
        print(f"✓ Created user without display fields")
        print(f"  display_name: {user.display_name}")
        print(f"  department_display_name: {user.department_display_name}")
        print("\n✓ TEST 2 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 2 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 2 ERROR: {str(e)}")
        return False


def test_fields_accept_string_values():
    """Test 3: Verify display fields accept string values."""
    print("\n" + "="*60)
    print("TEST 3: Fields Accept String Values")
    print("="*60)
    
    try:
        # Create CurrentUser with display fields
        user = CurrentUser(
            user_id=2,
            username="john_doe",
            display_name="Dr. John Doe",
            department_display_name="Cardiology Department",
            is_active=True,
            scopes=[]
        )
        
        assert user.display_name == "Dr. John Doe", f"Expected 'Dr. John Doe', got '{user.display_name}'"
        assert user.department_display_name == "Cardiology Department", f"Expected 'Cardiology Department', got '{user.department_display_name}'"
        
        print(f"✓ Created user with display fields")
        print(f"  username: {user.username}")
        print(f"  display_name: {user.display_name}")
        print(f"  department_display_name: {user.department_display_name}")
        print("\n✓ TEST 3 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 3 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 3 ERROR: {str(e)}")
        return False


def test_fields_accept_none_explicitly():
    """Test 4: Verify display fields accept explicit None values."""
    print("\n" + "="*60)
    print("TEST 4: Fields Accept Explicit None")
    print("="*60)
    
    try:
        # Create CurrentUser with explicit None
        user = CurrentUser(
            user_id=3,
            username="jane_smith",
            display_name=None,
            department_display_name=None,
            is_active=True,
            scopes=[]
        )
        
        assert user.display_name is None, "display_name should be None"
        assert user.department_display_name is None, "department_display_name should be None"
        
        print(f"✓ Created user with explicit None values")
        print(f"  display_name: {user.display_name}")
        print(f"  department_display_name: {user.department_display_name}")
        print("\n✓ TEST 4 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 4 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 4 ERROR: {str(e)}")
        return False


def test_partial_display_fields():
    """Test 5: Verify can set one display field without the other."""
    print("\n" + "="*60)
    print("TEST 5: Partial Display Fields")
    print("="*60)
    
    try:
        # Create with only display_name
        user1 = CurrentUser(
            user_id=4,
            username="user_one",
            display_name="Display Name Only",
            is_active=True,
            scopes=[]
        )
        
        assert user1.display_name == "Display Name Only"
        assert user1.department_display_name is None
        print(f"✓ User with display_name only:")
        print(f"  display_name: {user1.display_name}")
        print(f"  department_display_name: {user1.department_display_name}")
        
        # Create with only department_display_name
        user2 = CurrentUser(
            user_id=5,
            username="user_two",
            department_display_name="Department Only",
            is_active=True,
            scopes=[]
        )
        
        assert user2.display_name is None
        assert user2.department_display_name == "Department Only"
        print(f"\n✓ User with department_display_name only:")
        print(f"  display_name: {user2.display_name}")
        print(f"  department_display_name: {user2.department_display_name}")
        
        print("\n✓ TEST 5 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 5 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 5 ERROR: {str(e)}")
        return False


def test_model_serialization():
    """Test 6: Verify model serialization includes display fields."""
    print("\n" + "="*60)
    print("TEST 6: Model Serialization")
    print("="*60)
    
    try:
        # Create user with display fields
        user = CurrentUser(
            user_id=6,
            username="serialize_test",
            display_name="Dr. Test",
            department_display_name="Test Department",
            is_active=True,
            scopes=[
                UserScope(role_code="ADMIN", org_unit_id=1, org_unit_type="ADMIN")
            ]
        )
        
        # Serialize to dict
        user_dict = user.model_dump()
        
        assert 'display_name' in user_dict, "display_name missing from serialization"
        assert 'department_display_name' in user_dict, "department_display_name missing from serialization"
        assert user_dict['display_name'] == "Dr. Test"
        assert user_dict['department_display_name'] == "Test Department"
        
        print("✓ Model serialization includes display fields:")
        print(f"  display_name: {user_dict['display_name']}")
        print(f"  department_display_name: {user_dict['department_display_name']}")
        
        # Serialize to JSON
        user_json = user.model_dump_json()
        assert '"display_name":"Dr. Test"' in user_json
        assert '"department_display_name":"Test Department"' in user_json
        
        print("✓ JSON serialization includes display fields")
        print("\n✓ TEST 6 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 6 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 6 ERROR: {str(e)}")
        return False


def test_backward_compatibility():
    """Test 7: Verify backward compatibility with existing code."""
    print("\n" + "="*60)
    print("TEST 7: Backward Compatibility")
    print("="*60)
    
    try:
        # Simulate old code that doesn't know about display fields
        old_style_user_data = {
            "user_id": 7,
            "username": "legacy_user",
            "is_active": True,
            "scopes": [],
            "allowed_unit_ids": set(),
            "roles": [],
            "primary_unit_id": None,
            "primary_unit_type": None
        }
        
        # Should create successfully
        user = CurrentUser(**old_style_user_data)
        
        assert user.user_id == 7
        assert user.username == "legacy_user"
        assert user.display_name is None
        assert user.department_display_name is None
        
        print("✓ Old-style user data works without display fields")
        print(f"  user_id: {user.user_id}")
        print(f"  username: {user.username}")
        print(f"  display_name: {user.display_name} (defaults to None)")
        print(f"  department_display_name: {user.department_display_name} (defaults to None)")
        print("\n✓ TEST 7 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 7 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 7 ERROR: {str(e)}")
        return False


def test_complete_user_with_scopes():
    """Test 8: Verify display fields work with complete user including scopes."""
    print("\n" + "="*60)
    print("TEST 8: Complete User with Scopes")
    print("="*60)
    
    try:
        # Create complete user with scopes and display fields
        user = CurrentUser(
            user_id=8,
            username="section_admin",
            display_name="Dr. Sarah Johnson",
            department_display_name="Emergency Medicine",
            is_active=True,
            scopes=[
                UserScope(role_code="SECTION_ADMIN", org_unit_id=10, org_unit_type="SECTION"),
                UserScope(role_code="VIEWER", org_unit_id=20, org_unit_type="DEPARTMENT")
            ],
            allowed_unit_ids={10, 20},
            roles=["SECTION_ADMIN", "VIEWER"],
            primary_unit_id=10,
            primary_unit_type="SECTION"
        )
        
        # Verify all fields
        assert user.user_id == 8
        assert user.username == "section_admin"
        assert user.display_name == "Dr. Sarah Johnson"
        assert user.department_display_name == "Emergency Medicine"
        assert user.is_active == True
        assert len(user.scopes) == 2
        assert user.allowed_unit_ids == {10, 20}
        assert user.roles == ["SECTION_ADMIN", "VIEWER"]
        assert user.primary_unit_id == 10
        assert user.primary_unit_type == "SECTION"
        
        print("✓ Complete user created successfully:")
        print(f"  user_id: {user.user_id}")
        print(f"  username: {user.username}")
        print(f"  display_name: {user.display_name}")
        print(f"  department_display_name: {user.department_display_name}")
        print(f"  scopes: {len(user.scopes)} scopes")
        print(f"  roles: {user.roles}")
        print("\n✓ TEST 8 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 8 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 8 ERROR: {str(e)}")
        return False


def test_empty_string_values():
    """Test 9: Verify display fields handle empty strings."""
    print("\n" + "="*60)
    print("TEST 9: Empty String Values")
    print("="*60)
    
    try:
        # Create user with empty strings
        user = CurrentUser(
            user_id=9,
            username="empty_test",
            display_name="",
            department_display_name="",
            is_active=True,
            scopes=[]
        )
        
        assert user.display_name == "", "Empty string should be preserved"
        assert user.department_display_name == "", "Empty string should be preserved"
        
        print("✓ Empty strings handled correctly:")
        print(f"  display_name: '{user.display_name}'")
        print(f"  department_display_name: '{user.department_display_name}'")
        print("\n✓ TEST 9 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 9 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 9 ERROR: {str(e)}")
        return False


def test_field_types():
    """Test 10: Verify field types are correct."""
    print("\n" + "="*60)
    print("TEST 10: Field Types")
    print("="*60)
    
    try:
        fields = CurrentUser.model_fields
        
        # Check display_name field
        display_name_field = fields['display_name']
        print(f"✓ display_name field info:")
        print(f"  Required: {display_name_field.is_required()}")
        print(f"  Default: {display_name_field.default}")
        
        # Check department_display_name field
        dept_field = fields['department_display_name']
        print(f"\n✓ department_display_name field info:")
        print(f"  Required: {dept_field.is_required()}")
        print(f"  Default: {dept_field.default}")
        
        # Both should not be required
        assert not display_name_field.is_required(), "display_name should be optional"
        assert not dept_field.is_required(), "department_display_name should be optional"
        
        print("\n✓ Both fields are optional (not required)")
        print("\n✓ TEST 10 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 10 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 10 ERROR: {str(e)}")
        return False


def run_all_tests():
    """Run all Phase A Step 3 tests."""
    print("\n" + "="*60)
    print("PHASE A STEP 3: EXTEND CURRENTUSER MODEL")
    print("TEST SUITE")
    print("="*60)
    
    tests = [
        ("Test 1: Display Fields Exist", test_fields_exist),
        ("Test 2: Fields Are Optional", test_fields_are_optional),
        ("Test 3: Fields Accept String Values", test_fields_accept_string_values),
        ("Test 4: Fields Accept Explicit None", test_fields_accept_none_explicitly),
        ("Test 5: Partial Display Fields", test_partial_display_fields),
        ("Test 6: Model Serialization", test_model_serialization),
        ("Test 7: Backward Compatibility", test_backward_compatibility),
        ("Test 8: Complete User with Scopes", test_complete_user_with_scopes),
        ("Test 9: Empty String Values", test_empty_string_values),
        ("Test 10: Field Types", test_field_types),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ {name} EXCEPTION: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {len(tests)}")
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print(f"Pass Rate: {(passed/len(tests)*100):.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED (100%)")
        print("✅ CurrentUser model extended with display identity fields")
        print("\nKey Features Verified:")
        print("  ✓ display_name and department_display_name fields exist")
        print("  ✓ Both fields are Optional[str]")
        print("  ✓ Fields default to None when not provided")
        print("  ✓ Fields accept string values")
        print("  ✓ Can set fields independently")
        print("  ✓ Model serialization includes display fields")
        print("  ✓ Backward compatibility maintained")
        print("  ✓ Works with complete user context (scopes, roles)")
        print("  ✓ Empty strings handled correctly")
        print("  ✓ Field types correctly configured")
    else:
        print(f"\n❌ {failed} TEST(S) FAILED")
    
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
