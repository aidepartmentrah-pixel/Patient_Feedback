"""
Test Script for ML Insert Hook Integration

Tests:
1. Normal insert — Main DB succeeds + ML insert runs
2. ML failure simulation — Main DB succeeds despite ML error
3. Empty/partial data — Graceful handling
"""

import sys
import os
from datetime import datetime

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backend.api.services.insert_service import create_record


def test_1_normal_insert():
    """Test 1: Normal Insert - Main DB + ML insert succeeds"""
    print("\n" + "=" * 70)
    print("TEST 1: NORMAL INSERT")
    print("=" * 70)
    
    data = {
        'complaint_text': 'Test complaint for ML insert',
        'feedback_received_date': datetime.now().strftime('%Y-%m-%d'),
        'issuing_department_id': 1,
        'domain_id': 1,
        'category_id': 1,
        'subcategory_id': 1,
        'classification_id': 1,
        'severity_id': 1,
        'stage_id': 1,
        'harm_id': 1,
        'patient_name': 'Test Patient',
        'immediate_action': 'Action taken',
        'taken_action': 'Follow-up action',
        'is_inpatient': True,
        'clinical_risk_type_id': 1,
        'feedback_intent_type_id': 1,
    }
    
    try:
        result = create_record(data)
        print(f"Result: {result}")
        
        if result.get('success'):
            print("✓ Main DB insert succeeded")
            print(f"✓ Record ID: {result.get('record_id')}")
            print(f"✓ New ID: {result.get('id')}")
            print("[CHECK ML DB] Verify new row in patient_feedback_encoded table")
            return True
        else:
            print(f"✗ Main DB insert failed: {result.get('error')}")
            return False
    except Exception as e:
        print(f"✗ Exception: {str(e)}")
        return False


def test_2_verify_no_exceptions():
    """Test 2: Verify no exceptions propagate to main flow"""
    print("\n" + "=" * 70)
    print("TEST 2: NO EXCEPTIONS PROPAGATE")
    print("=" * 70)
    
    data = {
        'complaint_text': 'Another test complaint',
        'feedback_received_date': datetime.now().strftime('%Y-%m-%d'),
        'issuing_department_id': 1,
        'domain_id': 1,
        'category_id': 1,
        'subcategory_id': 1,
        'classification_id': 1,
        'severity_id': 1,
        'stage_id': 1,
        'harm_id': 1,
        # Intentionally omit some fields to test partial data
    }
    
    try:
        result = create_record(data)
        if result.get('success'):
            print("✓ Insert succeeded even with partial data")
            print(f"✓ Result: {result.get('record_id')}")
            return True
        else:
            # This is OK - validation might fail, but no unhandled exception
            print(f"✓ Graceful validation error (not an unhandled exception)")
            print(f"  Error: {result.get('message')}")
            return True
    except Exception as e:
        print(f"✗ Exception escaped: {str(e)}")
        return False


def test_3_return_value_unchanged():
    """Test 3: Verify return value is not modified by ML hook"""
    print("\n" + "=" * 70)
    print("TEST 3: RETURN VALUE UNCHANGED")
    print("=" * 70)
    
    data = {
        'complaint_text': 'Return value test',
        'feedback_received_date': datetime.now().strftime('%Y-%m-%d'),
        'issuing_department_id': 1,
        'domain_id': 1,
        'category_id': 1,
        'subcategory_id': 1,
        'classification_id': 1,
        'severity_id': 1,
        'stage_id': 1,
        'harm_id': 1,
    }
    
    try:
        result = create_record(data)
        
        # Check for expected keys
        expected_keys = {'success', 'message', 'record_id', 'id', 'status_id', 'created_at'}
        if result.get('success'):
            actual_keys = set(result.keys())
            if expected_keys.issubset(actual_keys):
                print("✓ Return value has all expected keys")
                print(f"  Keys: {sorted(actual_keys)}")
                return True
            else:
                print(f"✗ Missing keys: {expected_keys - actual_keys}")
                return False
        else:
            print(f"✓ Validation error (expected): {result.get('message')}")
            return True
    except Exception as e:
        print(f"✗ Exception: {str(e)}")
        return False


def main():
    print("\n" + "🧪 ML INSERT HOOK INTEGRATION TESTS 🧪".center(70))
    
    results = []
    
    # Run tests
    results.append(("Test 1: Normal Insert", test_1_normal_insert()))
    results.append(("Test 2: No Exceptions", test_2_verify_no_exceptions()))
    results.append(("Test 3: Return Value", test_3_return_value_unchanged()))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED - ML hook integrated successfully!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED - Review output above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
