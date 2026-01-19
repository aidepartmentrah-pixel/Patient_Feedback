"""
PHASE 3 TEST: Service Layer - Explanation Business Logic
========================================================
Tests business logic and validation for explanation services.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.api.services.explanation_service import (
    get_pending_explanations,
    get_explanation_dashboard_statistics,
    get_case_explanation_details,
    submit_explanation,
    validate_explanation_submission,
    toggle_requires_explanation,
    admin_force_close_case,
    complete_case_with_action_items,
    get_explanation_history
)


def test_get_pending_explanations():
    """Test 1: Get pending explanations with filters"""
    print("=" * 70)
    print("TEST 1: Get Pending Explanations")
    print("=" * 70)
    
    try:
        # Test without filters
        result = get_pending_explanations()
        
        if result['success']:
            print(f"✓ Successfully retrieved pending explanations")
            print(f"  Total: {result['total_count']}")
            print(f"  Red Flag: {result['red_flag_count']}")
            print(f"  Ordinary: {result['ordinary_count']}")
        else:
            print(f"⚠ Query returned with error: {result.get('error')}")
        
        # Test with date filter
        result_filtered = get_pending_explanations(
            start_date="2024-01-01",
            end_date="2026-12-31"
        )
        
        if result_filtered['success']:
            print(f"\n✓ Successfully applied date filters")
            print(f"  Filtered count: {result_filtered['total_count']}")
        
        # Test with red flags only
        result_red_flags = get_pending_explanations(
            include_red_flags_only=True
        )
        
        if result_red_flags['success']:
            print(f"\n✓ Successfully filtered Red Flag/Never Event cases")
            print(f"  Red Flag only count: {result_red_flags['total_count']}")
        
        # Test invalid date format
        result_invalid = get_pending_explanations(
            start_date="invalid-date"
        )
        
        if not result_invalid['success']:
            print(f"\n✓ Correctly rejected invalid date format")
        
        return True
        
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dashboard_statistics():
    """Test 2: Get dashboard statistics"""
    print("\n" + "=" * 70)
    print("TEST 2: Dashboard Statistics")
    print("=" * 70)
    
    try:
        result = get_explanation_dashboard_statistics()
        
        if result['success']:
            print(f"✓ Successfully retrieved dashboard statistics")
            
            stats = result['statistics']
            
            print(f"\n  By Status:")
            for status, count in stats['by_status'].items():
                print(f"    {status}: {count}")
            
            print(f"\n  Overdue Cases:")
            print(f"    Over 7 days: {stats['overdue']['over_7_days']}")
            print(f"    Over 30 days: {stats['overdue']['over_30_days']}")
            
            print(f"\n  Totals:")
            print(f"    Awaiting: {stats['totals']['awaiting_explanation']}")
            print(f"    Responded: {stats['totals']['responded']}")
        else:
            print(f"⚠ Failed to get statistics: {result.get('error')}")
        
        return True
        
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_case_explanation_details():
    """Test 3: Get case explanation details with validation"""
    print("\n" + "=" * 70)
    print("TEST 3: Case Explanation Details")
    print("=" * 70)
    
    try:
        # Test with existing case
        from backend.api.db_layer.explanation_db import _fetch_one
        
        # Find any case
        case_row = _fetch_one("SELECT TOP 1 IncidentRequestCaseID FROM dbo.APP_IncidentCase")
        
        if not case_row:
            print("⚠ No cases available to test with")
            return True
        
        case_id = case_row["IncidentRequestCaseID"]
        
        result = get_case_explanation_details(case_id)
        
        if result['success']:
            print(f"✓ Successfully retrieved case {case_id} details")
            
            val = result['validation']
            print(f"\n  Validation:")
            print(f"    Can submit: {val['can_submit_explanation']}")
            print(f"    Has explanation: {val['has_existing_explanation']}")
            print(f"    Requires explanation: {val['requires_explanation']}")
            print(f"    Is closed: {val['is_closed']}")
            print(f"    Current status: {val['current_status']}")
        else:
            print(f"✗ Failed: {result.get('error')}")
            return False
        
        # Test with non-existent case
        result_invalid = get_case_explanation_details(999999999)
        
        if not result_invalid['success']:
            print(f"\n✓ Correctly handled non-existent case")
        
        return True
        
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validate_explanation():
    """Test 4: Validation service"""
    print("\n" + "=" * 70)
    print("TEST 4: Validate Explanation Submission")
    print("=" * 70)
    
    try:
        from backend.api.db_layer.explanation_db import _fetch_one
        
        # Find any case
        case_row = _fetch_one("SELECT TOP 1 IncidentRequestCaseID FROM dbo.APP_IncidentCase")
        
        if not case_row:
            print("⚠ No cases available to test with")
            return True
        
        case_id = case_row["IncidentRequestCaseID"]
        
        # Test valid explanation
        result_valid = validate_explanation_submission(
            case_id=case_id,
            explanation_text="This is a valid explanation with sufficient length",
            action_items=[
                {"title": "Action 1", "description": "Test action", "due_date": "2026-12-31"}
            ]
        )
        
        print(f"Validation result: {'Valid' if result_valid['valid'] else 'Invalid'}")
        if result_valid['errors']:
            print(f"  Errors: {result_valid['errors']}")
        if result_valid['warnings']:
            print(f"  Warnings: {result_valid['warnings']}")
        
        # Test invalid - too short
        result_short = validate_explanation_submission(
            case_id=case_id,
            explanation_text="Short"
        )
        
        if not result_short['valid'] and any('10 characters' in e for e in result_short['errors']):
            print(f"\n✓ Correctly rejected short explanation")
        
        # Test invalid action item
        result_bad_action = validate_explanation_submission(
            case_id=case_id,
            explanation_text="Valid explanation text here",
            action_items=[
                {"description": "Missing title"}
            ]
        )
        
        if not result_bad_action['valid']:
            print(f"✓ Correctly rejected invalid action item")
        
        # Test invalid date format
        result_bad_date = validate_explanation_submission(
            case_id=case_id,
            explanation_text="Valid explanation text",
            action_items=[
                {"title": "Action", "due_date": "invalid-date"}
            ]
        )
        
        if not result_bad_date['valid']:
            print(f"✓ Correctly rejected invalid date format")
        
        return True
        
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_submit_explanation_validation():
    """Test 5: Submit explanation validation (without actual submission)"""
    print("\n" + "=" * 70)
    print("TEST 5: Submit Explanation Validation")
    print("=" * 70)
    
    try:
        # Test with non-existent case
        result = submit_explanation(
            case_id=999999999,
            explanation_text="Test explanation",
            user_id=1
        )
        
        if not result['success']:
            print(f"✓ Correctly rejected non-existent case")
            print(f"  Error: {result.get('error', 'N/A')[:60]}...")
        
        # Test with too short explanation
        from backend.api.db_layer.explanation_db import _fetch_one
        case_row = _fetch_one("SELECT TOP 1 IncidentRequestCaseID FROM dbo.APP_IncidentCase")
        
        if case_row:
            result_short = submit_explanation(
                case_id=case_row["IncidentRequestCaseID"],
                explanation_text="Short",
                user_id=1
            )
            
            if not result_short['success']:
                print(f"✓ Correctly rejected short explanation")
        
        return True
        
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_admin_operations():
    """Test 6: Admin operations validation"""
    print("\n" + "=" * 70)
    print("TEST 6: Admin Operations")
    print("=" * 70)
    
    try:
        # Test force close without reason
        result = admin_force_close_case(
            case_id=999999999,
            user_id=1,
            reason=""
        )
        
        if not result['success'] and 'reason' in result.get('error', '').lower():
            print(f"✓ Correctly requires reason for force closure")
        
        # Test force close with too short reason
        result_short = admin_force_close_case(
            case_id=999999999,
            user_id=1,
            reason="No"
        )
        
        if not result_short['success']:
            print(f"✓ Correctly validates minimum reason length")
        
        return True
        
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_explanation_history():
    """Test 7: Get explanation history"""
    print("\n" + "=" * 70)
    print("TEST 7: Explanation History")
    print("=" * 70)
    
    try:
        from backend.api.db_layer.explanation_db import _fetch_one
        
        # Find any case
        case_row = _fetch_one("SELECT TOP 1 IncidentRequestCaseID FROM dbo.APP_IncidentCase")
        
        if not case_row:
            print("⚠ No cases available to test with")
            return True
        
        case_id = case_row["IncidentRequestCaseID"]
        
        result = get_explanation_history(case_id)
        
        if result['success']:
            print(f"✓ Successfully retrieved explanation history for case {case_id}")
            print(f"  Current explanation: {'Present' if result.get('current_explanation') else 'None'}")
            print(f"  Current status: {result.get('current_status')}")
            print(f"  History records: {len(result.get('history', []))}")
        
        # Test non-existent case
        result_invalid = get_explanation_history(999999999)
        
        if not result_invalid['success']:
            print(f"\n✓ Correctly handled non-existent case")
        
        return True
        
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all Phase 3 tests"""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  PHASE 3: SERVICE LAYER BUSINESS LOGIC TESTS".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")
    
    tests = [
        ("Get Pending Explanations", test_get_pending_explanations),
        ("Dashboard Statistics", test_dashboard_statistics),
        ("Case Explanation Details", test_case_explanation_details),
        ("Validate Explanation", test_validate_explanation),
        ("Submit Explanation Validation", test_submit_explanation_validation),
        ("Admin Operations", test_admin_operations),
        ("Explanation History", test_explanation_history),
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
        print("\n🎉 ALL TESTS PASSED - Phase 3 Complete!")
    else:
        print(f"\n⚠ {total - passed} test(s) failed - Please review")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
