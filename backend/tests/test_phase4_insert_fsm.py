"""
PHASE 4 TEST: Insert Service with Explanation FSM Logic
=======================================================
Tests the three-path FSM logic for explanation workflow during case creation.

Three Paths:
1. Red Flag (ClinicalRiskTypeID=2) -> Open + Waiting
2. Never Event (ClinicalRiskTypeID=3) -> Open + Waiting  
3. Ordinary with RequiresExplanation=True -> Open + Waiting
4. Ordinary with RequiresExplanation=False -> Closed + No Explanation Needed
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.api.services.insert_service import create_record
from backend.api.db_layer.incident_case import get_incident_case_by_id
from datetime import datetime


def cleanup_test_case(case_id):
    """Helper to clean up test data"""
    try:
        from backend.api.db_layer.incident_case import hard_delete_incident_case
        hard_delete_incident_case(case_id)
        print(f"  [Cleanup] Deleted test case {case_id}")
    except Exception as e:
        print(f"  [Cleanup Warning] Could not delete case {case_id}: {e}")


def test_red_flag_path():
    """Test 1: Red Flag -> Open + Waiting"""
    print("=" * 70)
    print("TEST 1: Red Flag Path (ClinicalRiskTypeID=2)")
    print("=" * 70)
    
    case_id = None
    try:
        data = {
            "complaint_text": "Test Red Flag Case - Requires Explanation",
            "feedback_received_date": "2026-01-19",
            "issuing_department_id": 43,
            "domain_id": 2,
            "category_id": 5,
            "subcategory_id": 13,
            "classification_id": 106,
            "severity_id": 2,
            "stage_id": 2,
            "harm_id": 2,
            "building_id": 2,
            "source_id": 4,
            "clinical_risk_type_id": 2,  # Red Flag
            "requires_explanation": False,  # Shouldn't matter - Red Flag overrides
        }
        
        result = create_record(data)
        
        if not result['success']:
            print(f"✗ Failed to create case: {result.get('error')}")
            print(f"  Full error message: {result.get('message')}")
            return False
        
        case_id = result['incident_id']
        print(f"✓ Created case {case_id}")
        
        # Verify FSM state
        case = get_incident_case_by_id(case_id)
        
        if not case:
            print(f"✗ Could not retrieve created case")
            return False
        
        expected_case_status = 1  # Open
        expected_explanation_status = 1  # Waiting
        expected_requires_explanation = 0  # False, because only Red Flag matters
        
        if case['CaseStatusID'] == expected_case_status:
            print(f"✓ CaseStatusID = {case['CaseStatusID']} (Open)")
        else:
            print(f"✗ CaseStatusID = {case['CaseStatusID']}, expected {expected_case_status}")
            return False
        
        if case['ExplanationStatusID'] == expected_explanation_status:
            print(f"✓ ExplanationStatusID = {case['ExplanationStatusID']} (Waiting)")
        else:
            print(f"✗ ExplanationStatusID = {case['ExplanationStatusID']}, expected {expected_explanation_status}")
            return False
        
        if case['RequiresExplanation'] == expected_requires_explanation:
            print(f"✓ RequiresExplanation = {case['RequiresExplanation']} (False - Red Flag determines status)")
        else:
            print(f"✗ RequiresExplanation = {case['RequiresExplanation']}, expected {expected_requires_explanation}")
            return False
        
        print(f"✓ Red Flag path validated: Open + Waiting")
        
        cleanup_test_case(case_id)
        return True
        
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        if case_id:
            cleanup_test_case(case_id)
        return False


def test_never_event_path():
    """Test 2: Never Event -> Open + Waiting"""
    print("\n" + "=" * 70)
    print("TEST 2: Never Event Path (ClinicalRiskTypeID=3)")
    print("=" * 70)
    
    case_id = None
    try:
        data = {
            "complaint_text": "Test Never Event Case - Requires Explanation",
            "feedback_received_date": "2026-01-19",
            "issuing_department_id": 43,
            "domain_id": 2,
            "category_id": 5,
            "subcategory_id": 13,
            "classification_id": 106,
            "severity_id": 2,
            "stage_id": 2,
            "harm_id": 2,
            "building_id": 2,
            "source_id": 4,
            "clinical_risk_type_id": 3,  # Never Event
            "requires_explanation": False,  # Shouldn't matter - Never Event overrides
        }
        
        result = create_record(data)
        
        if not result['success']:
            print(f"✗ Failed to create case: {result.get('error')}")
            return False
        
        case_id = result['incident_id']
        print(f"✓ Created case {case_id}")
        
        # Verify FSM state
        case = get_incident_case_by_id(case_id)
        
        expected_case_status = 1  # Open
        expected_explanation_status = 1  # Waiting
        
        if case['CaseStatusID'] == expected_case_status:
            print(f"✓ CaseStatusID = {case['CaseStatusID']} (Open)")
        else:
            print(f"✗ CaseStatusID = {case['CaseStatusID']}, expected {expected_case_status}")
            return False
        
        if case['ExplanationStatusID'] == expected_explanation_status:
            print(f"✓ ExplanationStatusID = {case['ExplanationStatusID']} (Waiting)")
        else:
            print(f"✗ ExplanationStatusID = {case['ExplanationStatusID']}, expected {expected_explanation_status}")
            return False
        
        print(f"✓ Never Event path validated: Open + Waiting")
        
        cleanup_test_case(case_id)
        return True
        
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        if case_id:
            cleanup_test_case(case_id)
        return False


def test_ordinary_requires_explanation_true():
    """Test 3: Ordinary + RequiresExplanation=True -> Open + Waiting"""
    print("\n" + "=" * 70)
    print("TEST 3: Ordinary Complaint with RequiresExplanation=True")
    print("=" * 70)
    
    case_id = None
    try:
        data = {
            "complaint_text": "Test Ordinary Case that Requires Explanation",
            "feedback_received_date": "2026-01-19",
            "issuing_department_id": 43,
            "domain_id": 2,
            "category_id": 5,
            "subcategory_id": 13,
            "classification_id": 106,
            "severity_id": 2,
            "stage_id": 2,
            "harm_id": 2,
            "building_id": 2,
            "source_id": 4,
            "clinical_risk_type_id": 1,  # Ordinary
            "requires_explanation": True,  # Explicitly requires explanation
        }
        
        result = create_record(data)
        
        if not result['success']:
            print(f"✗ Failed to create case: {result.get('error')}")
            return False
        
        case_id = result['incident_id']
        print(f"✓ Created case {case_id}")
        
        # Verify FSM state
        case = get_incident_case_by_id(case_id)
        
        expected_case_status = 1  # Open
        expected_explanation_status = 1  # Waiting
        expected_requires_explanation = 1  # True
        
        if case['CaseStatusID'] == expected_case_status:
            print(f"✓ CaseStatusID = {case['CaseStatusID']} (Open)")
        else:
            print(f"✗ CaseStatusID = {case['CaseStatusID']}, expected {expected_case_status}")
            return False
        
        if case['ExplanationStatusID'] == expected_explanation_status:
            print(f"✓ ExplanationStatusID = {case['ExplanationStatusID']} (Waiting)")
        else:
            print(f"✗ ExplanationStatusID = {case['ExplanationStatusID']}, expected {expected_explanation_status}")
            return False
        
        if case['RequiresExplanation'] == expected_requires_explanation:
            print(f"✓ RequiresExplanation = {case['RequiresExplanation']} (True)")
        else:
            print(f"✗ RequiresExplanation = {case['RequiresExplanation']}, expected {expected_requires_explanation}")
            return False
        
        print(f"✓ Ordinary + RequiresExplanation path validated: Open + Waiting")
        
        cleanup_test_case(case_id)
        return True
        
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        if case_id:
            cleanup_test_case(case_id)
        return False


def test_ordinary_no_explanation_needed():
    """Test 4: Ordinary + RequiresExplanation=False -> Closed + No Explanation Needed"""
    print("\n" + "=" * 70)
    print("TEST 4: Ordinary Complaint with RequiresExplanation=False")
    print("=" * 70)
    
    case_id = None
    try:
        data = {
            "complaint_text": "Test Ordinary Case that Does Not Require Explanation",
            "feedback_received_date": "2026-01-19",
            "issuing_department_id": 43,
            "domain_id": 2,
            "category_id": 5,
            "subcategory_id": 13,
            "classification_id": 106,
            "severity_id": 2,
            "stage_id": 2,
            "harm_id": 2,
            "building_id": 2,
            "source_id": 4,
            "clinical_risk_type_id": 1,  # Ordinary
            "requires_explanation": False,  # Explicitly does NOT require explanation
        }
        
        result = create_record(data)
        
        if not result['success']:
            print(f"✗ Failed to create case: {result.get('error')}")
            return False
        
        case_id = result['incident_id']
        print(f"✓ Created case {case_id}")
        
        # Verify FSM state
        case = get_incident_case_by_id(case_id)
        
        expected_case_status = 3  # Closed
        expected_explanation_status = 4  # No Explanation Needed
        expected_requires_explanation = 0  # False
        
        if case['CaseStatusID'] == expected_case_status:
            print(f"✓ CaseStatusID = {case['CaseStatusID']} (Closed)")
        else:
            print(f"✗ CaseStatusID = {case['CaseStatusID']}, expected {expected_case_status}")
            return False
        
        if case['ExplanationStatusID'] == expected_explanation_status:
            print(f"✓ ExplanationStatusID = {case['ExplanationStatusID']} (No Explanation Needed)")
        else:
            print(f"✗ ExplanationStatusID = {case['ExplanationStatusID']}, expected {expected_explanation_status}")
            return False
        
        if case['RequiresExplanation'] == expected_requires_explanation:
            print(f"✓ RequiresExplanation = {case['RequiresExplanation']} (False)")
        else:
            print(f"✗ RequiresExplanation = {case['RequiresExplanation']}, expected {expected_requires_explanation}")
            return False
        
        print(f"✓ Ordinary without explanation path validated: Closed + No Explanation Needed")
        
        cleanup_test_case(case_id)
        return True
        
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        if case_id:
            cleanup_test_case(case_id)
        return False


def test_ordinary_default_false():
    """Test 5: Ordinary with no RequiresExplanation field (defaults to False)"""
    print("\n" + "=" * 70)
    print("TEST 5: Ordinary Complaint with Default RequiresExplanation")
    print("=" * 70)
    
    case_id = None
    try:
        data = {
            "complaint_text": "Test Ordinary Case with Default Behavior",
            "feedback_received_date": "2026-01-19",
            "issuing_department_id": 43,
            "domain_id": 2,
            "category_id": 5,
            "subcategory_id": 13,
            "classification_id": 106,
            "severity_id": 2,
            "stage_id": 2,
            "harm_id": 2,
            "building_id": 2,
            "source_id": 4,
            "clinical_risk_type_id": 1,  # Ordinary
            # No requires_explanation field - should default to False
        }
        
        result = create_record(data)
        
        if not result['success']:
            print(f"✗ Failed to create case: {result.get('error')}")
            return False
        
        case_id = result['incident_id']
        print(f"✓ Created case {case_id}")
        
        # Verify FSM state
        case = get_incident_case_by_id(case_id)
        
        expected_case_status = 3  # Closed
        expected_explanation_status = 4  # No Explanation Needed
        expected_requires_explanation = 0  # False (default)
        
        if case['CaseStatusID'] == expected_case_status:
            print(f"✓ CaseStatusID = {case['CaseStatusID']} (Closed)")
        else:
            print(f"✗ CaseStatusID = {case['CaseStatusID']}, expected {expected_case_status}")
            return False
        
        if case['ExplanationStatusID'] == expected_explanation_status:
            print(f"✓ ExplanationStatusID = {case['ExplanationStatusID']} (No Explanation Needed)")
        else:
            print(f"✗ ExplanationStatusID = {case['ExplanationStatusID']}, expected {expected_explanation_status}")
            return False
        
        if case['RequiresExplanation'] == expected_requires_explanation:
            print(f"✓ RequiresExplanation = {case['RequiresExplanation']} (Default False)")
        else:
            print(f"✗ RequiresExplanation = {case['RequiresExplanation']}, expected {expected_requires_explanation}")
            return False
        
        print(f"✓ Default behavior validated: Closed + No Explanation Needed")
        
        cleanup_test_case(case_id)
        return True
        
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        if case_id:
            cleanup_test_case(case_id)
        return False


def test_requires_explanation_string_conversion():
    """Test 6: Test string value conversion for requires_explanation"""
    print("\n" + "=" * 70)
    print("TEST 6: RequiresExplanation String Conversion")
    print("=" * 70)
    
    case_id = None
    try:
        data = {
            "complaint_text": "Test String Conversion for RequiresExplanation",
            "feedback_received_date": "2026-01-19",
            "issuing_department_id": 43,
            "domain_id": 2,
            "category_id": 5,
            "subcategory_id": 13,
            "classification_id": 106,
            "severity_id": 2,
            "stage_id": 2,
            "harm_id": 2,
            "building_id": 2,
            "source_id": 4,
            "clinical_risk_type_id": 1,  # Ordinary
            "requires_explanation": "true",  # String value
        }
        
        result = create_record(data)
        
        if not result['success']:
            print(f"✗ Failed to create case: {result.get('error')}")
            return False
        
        case_id = result['incident_id']
        print(f"✓ Created case {case_id} with requires_explanation='true' (string)")
        
        # Verify FSM state
        case = get_incident_case_by_id(case_id)
        
        expected_case_status = 1  # Open
        expected_explanation_status = 1  # Waiting
        expected_requires_explanation = 1  # True (converted from string)
        
        if case['RequiresExplanation'] == expected_requires_explanation:
            print(f"✓ String 'true' correctly converted to RequiresExplanation = 1")
        else:
            print(f"✗ RequiresExplanation = {case['RequiresExplanation']}, expected {expected_requires_explanation}")
            return False
        
        if case['CaseStatusID'] == expected_case_status and case['ExplanationStatusID'] == expected_explanation_status:
            print(f"✓ FSM state correct: Open + Waiting")
        else:
            print(f"✗ FSM state incorrect")
            return False
        
        cleanup_test_case(case_id)
        return True
        
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        if case_id:
            cleanup_test_case(case_id)
        return False


def run_all_tests():
    """Run all Phase 4 tests"""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  PHASE 4: INSERT SERVICE FSM LOGIC TESTS".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")
    
    tests = [
        ("Red Flag Path", test_red_flag_path),
        ("Never Event Path", test_never_event_path),
        ("Ordinary + RequiresExplanation=True", test_ordinary_requires_explanation_true),
        ("Ordinary + RequiresExplanation=False", test_ordinary_no_explanation_needed),
        ("Ordinary Default (No Field)", test_ordinary_default_false),
        ("String Conversion Test", test_requires_explanation_string_conversion),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ TEST FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n")
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test_name:<45} {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print("=" * 70)
    print(f"  Total: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Phase 4 Complete!")
        print("\nFSM Logic Validated:")
        print("  ✓ Red Flag -> Open + Waiting")
        print("  ✓ Never Event -> Open + Waiting")
        print("  ✓ Ordinary + RequiresExplanation=True -> Open + Waiting")
        print("  ✓ Ordinary + RequiresExplanation=False -> Closed + No Explanation Needed")
    else:
        print(f"\n⚠ {total - passed} test(s) failed - Please review")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
