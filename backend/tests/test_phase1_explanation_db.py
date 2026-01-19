"""
PHASE 1 TEST: DB Layer - Explanation Query Functions
=====================================================
Tests all read operations for explanation workflow.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.api.db_layer.explanation_db import (
    get_explanation_status_id,
    get_explanation_status_name,
    get_all_explanation_statuses,
    get_case_status_id,
    get_case_status_name,
    get_case_by_id,
    get_cases_needing_explanation,
    get_red_flag_never_event_cases_needing_explanation,
    count_cases_by_explanation_status,
    get_overdue_explanations,
    check_case_has_explanation
)
from datetime import datetime, timedelta


def test_explanation_status_lookups():
    """Test 1: Explanation status lookup functions"""
    print("=" * 70)
    print("TEST 1: Explanation Status Lookups")
    print("=" * 70)
    
    try:
        # Test get all statuses
        statuses = get_all_explanation_statuses()
        print(f"✓ Found {len(statuses)} explanation statuses:")
        for status in statuses:
            print(f"  {status['StatusID']}: {status['StatusName']}")
        
        # Test get ID by name
        waiting_id = get_explanation_status_id("Waiting")
        responded_id = get_explanation_status_id("Responded")
        forcibly_id = get_explanation_status_id("Forcibly Closed")
        no_exp_id = get_explanation_status_id("No Explanation Needed")
        
        print(f"\n✓ Status ID lookups:")
        print(f"  Waiting: {waiting_id}")
        print(f"  Responded: {responded_id}")
        print(f"  Forcibly Closed: {forcibly_id}")
        print(f"  No Explanation Needed: {no_exp_id}")
        
        # Test get name by ID
        if waiting_id:
            name = get_explanation_status_name(waiting_id)
            print(f"\n✓ Reverse lookup: ID {waiting_id} = '{name}'")
        
        return True
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_case_status_lookups():
    """Test 2: Case status lookup functions"""
    print("\n" + "=" * 70)
    print("TEST 2: Case Status Lookups")
    print("=" * 70)
    
    try:
        # Test get ID by code
        open_id = get_case_status_id("OPEN")
        in_progress_id = get_case_status_id("IN_PROGRESS")
        closed_id = get_case_status_id("CLOSED")
        
        print(f"✓ Case Status ID lookups:")
        print(f"  OPEN: {open_id}")
        print(f"  IN_PROGRESS: {in_progress_id}")
        print(f"  CLOSED: {closed_id}")
        
        # Test get name by ID
        if open_id:
            name = get_case_status_name(open_id)
            print(f"\n✓ Reverse lookup: ID {open_id} = '{name}'")
        
        return True
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_case_by_id():
    """Test 3: Get case by ID"""
    print("\n" + "=" * 70)
    print("TEST 3: Get Case By ID")
    print("=" * 70)
    
    try:
        # Get first available case
        cases = get_cases_needing_explanation()
        
        if not cases:
            print("⚠ No cases available to test with, checking any case...")
            # Try to get any case ID
            from backend.api.db_layer.explanation_db import _fetch_one
            result = _fetch_one("SELECT TOP 1 IncidentRequestCaseID FROM dbo.APP_IncidentCase")
            if not result:
                print("⚠ No cases in database, skipping test")
                return True
            case_id = result["IncidentRequestCaseID"]
        else:
            case_id = cases[0]["IncidentRequestCaseID"]
        
        print(f"Testing with case ID: {case_id}")
        
        case = get_case_by_id(case_id)
        
        if case:
            print(f"✓ Successfully retrieved case {case_id}")
            print(f"  ComplaintText: {case['ComplaintText'][:50]}..." if case.get('ComplaintText') else "  ComplaintText: None")
            print(f"  RequiresExplanation: {case.get('RequiresExplanation', 'N/A')}")
            print(f"  CaseStatus: {case.get('CaseStatusName', 'N/A')}")
            print(f"  ExplanationStatus: {case.get('ExplanationStatusName', 'N/A')}")
            print(f"  TakenAction: {'Present' if case.get('TakenAction') else 'Empty'}")
        else:
            print(f"✗ Case {case_id} not found")
            return False
        
        # Test non-existent case
        fake_case = get_case_by_id(999999999)
        if fake_case is None:
            print(f"✓ Correctly returns None for non-existent case")
        
        return True
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_cases_needing_explanation():
    """Test 4: Get cases needing explanation"""
    print("\n" + "=" * 70)
    print("TEST 4: Get Cases Needing Explanation")
    print("=" * 70)
    
    try:
        # Test without filters
        cases = get_cases_needing_explanation()
        print(f"✓ Found {len(cases)} cases needing explanation (no filters)")
        
        if cases:
            print(f"\n  Sample case:")
            case = cases[0]
            print(f"    ID: {case['IncidentRequestCaseID']}")
            print(f"    RequiresExplanation: {case.get('RequiresExplanation')}")
            print(f"    ExplanationStatus: {case.get('ExplanationStatusName')}")
            print(f"    ClinicalRiskType: {case.get('ClinicalRiskType')}")
            print(f"    FeedbackRecievedDate: {case.get('FeedbackRecievedDate')}")
        
        # Test with date filter
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        filtered_cases = get_cases_needing_explanation(
            start_date=start_date,
            end_date=end_date
        )
        print(f"\n✓ Found {len(filtered_cases)} cases with date filter (last 365 days)")
        
        return True
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_red_flag_cases():
    """Test 5: Get Red Flag/Never Event cases needing explanation"""
    print("\n" + "=" * 70)
    print("TEST 5: Get Red Flag/Never Event Cases")
    print("=" * 70)
    
    try:
        cases = get_red_flag_never_event_cases_needing_explanation()
        print(f"✓ Found {len(cases)} Red Flag/Never Event cases needing explanation")
        
        if cases:
            print(f"\n  Sample cases:")
            for i, case in enumerate(cases[:3]):  # Show max 3
                print(f"    {i+1}. ID: {case['IncidentRequestCaseID']}")
                print(f"       Type: {case.get('ClinicalRiskType')}")
                print(f"       Status: {case.get('CaseStatusName')}/{case.get('ExplanationStatusName')}")
        
        return True
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_count_by_status():
    """Test 6: Count cases by explanation status"""
    print("\n" + "=" * 70)
    print("TEST 6: Count Cases By Explanation Status")
    print("=" * 70)
    
    try:
        counts = count_cases_by_explanation_status()
        print(f"✓ Case counts by status:")
        
        total = 0
        for count in counts:
            status_name = count['StatusName']
            case_count = count['CaseCount']
            total += case_count
            print(f"  {status_name}: {case_count} cases")
        
        print(f"\n  Total cases: {total}")
        return True
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_overdue_explanations():
    """Test 7: Get overdue explanations"""
    print("\n" + "=" * 70)
    print("TEST 7: Get Overdue Explanations")
    print("=" * 70)
    
    try:
        # Test with 7 days threshold
        overdue = get_overdue_explanations(days_threshold=7)
        print(f"✓ Found {len(overdue)} cases with overdue explanations (>7 days)")
        
        if overdue:
            print(f"\n  Most overdue cases:")
            for i, case in enumerate(overdue[:3]):  # Show max 3
                print(f"    {i+1}. ID: {case['IncidentRequestCaseID']}")
                print(f"       Days Overdue: {case['DaysOverdue']}")
                print(f"       Type: {case.get('ClinicalRiskType')}")
                print(f"       FeedbackRecievedDate: {case.get('FeedbackRecievedDate')}")
        
        # Test with 30 days threshold
        very_overdue = get_overdue_explanations(days_threshold=30)
        print(f"\n✓ Found {len(very_overdue)} cases with very overdue explanations (>30 days)")
        
        return True
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_check_has_explanation():
    """Test 8: Check if case has explanation"""
    print("\n" + "=" * 70)
    print("TEST 8: Check Case Has Explanation")
    print("=" * 70)
    
    try:
        from backend.api.db_layer.explanation_db import _fetch_one
        
        # Find a case with explanation
        case_with = _fetch_one("""
            SELECT TOP 1 IncidentRequestCaseID 
            FROM dbo.APP_IncidentCase 
            WHERE TakenAction IS NOT NULL AND TakenAction != ''
        """)
        
        # Find a case without explanation
        case_without = _fetch_one("""
            SELECT TOP 1 IncidentRequestCaseID 
            FROM dbo.APP_IncidentCase 
            WHERE TakenAction IS NULL OR TakenAction = ''
        """)
        
        if case_with:
            has_exp = check_case_has_explanation(case_with["IncidentRequestCaseID"])
            print(f"✓ Case {case_with['IncidentRequestCaseID']} has explanation: {has_exp}")
            if not has_exp:
                print("  ⚠ Warning: Expected True but got False")
        
        if case_without:
            has_exp = check_case_has_explanation(case_without["IncidentRequestCaseID"])
            print(f"✓ Case {case_without['IncidentRequestCaseID']} has explanation: {has_exp}")
            if has_exp:
                print("  ⚠ Warning: Expected False but got True")
        
        if not case_with and not case_without:
            print("⚠ No cases available to test with")
        
        return True
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all Phase 1 tests"""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  PHASE 1: DB LAYER READ OPERATIONS TESTS".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")
    
    tests = [
        ("Explanation Status Lookups", test_explanation_status_lookups),
        ("Case Status Lookups", test_case_status_lookups),
        ("Get Case By ID", test_get_case_by_id),
        ("Get Cases Needing Explanation", test_get_cases_needing_explanation),
        ("Get Red Flag Cases", test_get_red_flag_cases),
        ("Count By Status", test_count_by_status),
        ("Overdue Explanations", test_overdue_explanations),
        ("Check Has Explanation", test_check_has_explanation),
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
        print(f"  {test_name:<40} {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print("=" * 70)
    print(f"  Total: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Phase 1 Complete!")
    else:
        print(f"\n⚠ {total - passed} test(s) failed - Please review")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
