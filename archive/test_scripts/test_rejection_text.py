"""Quick test to verify rejection text is saved"""
import sys
sys.path.insert(0, r'c:\Users\IT\Documents\GitHub Repository\Patient_Feedback')

from backend.api_v2.services import case_response_service
from backend.api_v2.db_layer import administrative_subcase_db
from test_workflow_comprehensive import create_test_subcase, create_test_user, cleanup_test_subcase

# Test 1: Section rejection
print("=" * 60)
print("TEST: Section Rejection Text")
print("=" * 60)

subcase_id = create_test_subcase()
user = create_test_user('SECTION', 2, 'SECTION')
print(f"Created subcase {subcase_id}")

try:
    case_response_service.reject_responsibility(subcase_id, 'My section rejection text here', user)
    print("✓ reject_responsibility() executed without error")
    
    subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
    print(f"Status: {subcase['status']}")
    print(f"Rejection text: '{subcase['section_rejection_text']}'")
    
    if subcase['section_rejection_text'] == 'My section rejection text here':
        print("✓ SUCCESS - Text saved correctly")
    else:
        print("✗ FAILED - Text not saved or incorrect")
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
finally:
    cleanup_test_subcase(subcase_id)

print()

# Test 2: Department rejection  
print("=" * 60)
print("TEST: Department Rejection Text")
print("=" * 60)

subcase_id = create_test_subcase('SECTION_ACCEPTED_PENDING_DEPT')
dept_user = create_test_user('DEPARTMENT', 1, 'DEPARTMENT')
print(f"Created subcase {subcase_id}")

try:
    case_response_service.reject_department(subcase_id, 'My department rejection text here', dept_user)
    print("✓ reject_department() executed without error")
    
    subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
    print(f"Status: {subcase['status']}")
    print(f"Rejection text: '{subcase['department_rejection_text']}'")
    
    if subcase['department_rejection_text'] == 'My department rejection text here':
        print("✓ SUCCESS - Text saved correctly")
    else:
        print("✗ FAILED - Text not saved or incorrect")
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
finally:
    cleanup_test_subcase(subcase_id)

print()

# Test 3: Force close
print("=" * 60)
print("TEST: Force Close Reason Text")
print("=" * 60)

subcase_id = create_test_subcase('SUBMITTED_TO_SECTION')
admin_user = create_test_user('ADMINISTRATION', 0, 'ADMINISTRATION')
print(f"Created subcase {subcase_id}")

try:
    case_response_service.force_close_subcase(subcase_id, 'My force close reason here', admin_user)
    print("✓ force_close_subcase() executed without error")
    
    subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
    print(f"Status: {subcase['status']}")
    
    # Check if there's a force_close_reason field
    if 'force_close_reason' in subcase:
        print(f"Force close reason: '{subcase['force_close_reason']}'")
        if subcase['force_close_reason'] == 'My force close reason here':
            print("✓ SUCCESS - Reason saved correctly")
        else:
            print("✗ FAILED - Reason not saved or incorrect")
    else:
        print("✗ ISSUE: No force_close_reason field in subcase dict")
        print(f"Available fields: {list(subcase.keys())}")
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
finally:
    cleanup_test_subcase(subcase_id)
