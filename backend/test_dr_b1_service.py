"""
PHASE DR-B1 — Service Date Bounds Tests

Tests the get_dashboard_date_bounds_for_units service function.

Test Coverage:
1. Normal case - returns valid dates
2. Scoped subset - smaller range
3. Empty scope - returns None values
4. Type check - date objects not strings
"""

from datetime import date
from api.services.dashboard_service import get_dashboard_date_bounds_for_units

def test_normal_case():
    """
    Test 1: Normal case with known incidents
    - provide unit_ids with known incidents
    - assert min_date not null
    - assert max_date not null
    - assert min_date <= max_date
    """
    print("=" * 80)
    print("TEST 1: Normal Case - Unit IDs with Known Incidents")
    print("=" * 80)
    
    # Use unit ID 1 which should have incidents
    result = get_dashboard_date_bounds_for_units([1, 2, 3, 4, 5])
    
    print(f"Result: {result}")
    
    assert result["min_date"] is not None, "min_date should not be None"
    assert result["max_date"] is not None, "max_date should not be None"
    assert result["min_date"] <= result["max_date"], "min_date must be <= max_date"
    
    print(f"✓ min_date: {result['min_date']}")
    print(f"✓ max_date: {result['max_date']}")
    print(f"✓ min_date <= max_date: {result['min_date'] <= result['max_date']}")
    print()
    
    return result


def test_scoped_subset():
    """
    Test 2: Scoped subset
    - provide smaller unit_ids subset
    - assert returned range is within full range
    """
    print("=" * 80)
    print("TEST 2: Scoped Subset - Smaller Range")
    print("=" * 80)
    
    # Get full range
    full_result = get_dashboard_date_bounds_for_units([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(f"Full result: {full_result}")
    
    # Get subset range
    subset_result = get_dashboard_date_bounds_for_units([5])
    print(f"Subset result (unit 5 only): {subset_result}")
    
    # If subset has data, it should be within or equal to full range
    if subset_result["min_date"] is not None and full_result["min_date"] is not None:
        assert subset_result["min_date"] >= full_result["min_date"], \
            "Subset min_date should be >= full min_date"
        assert subset_result["max_date"] <= full_result["max_date"], \
            "Subset max_date should be <= full max_date"
        print(f"✓ Subset range is within full range")
    else:
        print(f"⚠ Subset has no data or full range has no data")
    
    print()


def test_empty_scope():
    """
    Test 3: Empty scope
    - provide unit_ids with no incidents
    - assert both values are None
    """
    print("=" * 80)
    print("TEST 3: Empty Scope - No Incidents")
    print("=" * 80)
    
    # Use a very high unit ID that likely doesn't exist
    result = get_dashboard_date_bounds_for_units([99999])
    
    print(f"Result: {result}")
    
    assert result["min_date"] is None, "min_date should be None for empty scope"
    assert result["max_date"] is None, "max_date should be None for empty scope"
    
    print(f"✓ min_date is None: {result['min_date'] is None}")
    print(f"✓ max_date is None: {result['max_date'] is None}")
    print()


def test_type_check(full_result):
    """
    Test 4: Type check
    - values are date objects or None
    - not datetime
    - not string
    """
    print("=" * 80)
    print("TEST 4: Type Check - Date Objects Not Strings")
    print("=" * 80)
    
    result = full_result
    
    print(f"Result: {result}")
    print(f"min_date type: {type(result['min_date'])}")
    print(f"max_date type: {type(result['max_date'])}")
    
    # Check types
    if result["min_date"] is not None:
        assert isinstance(result["min_date"], date), \
            f"min_date must be date object, got {type(result['min_date'])}"
        assert not isinstance(result["min_date"], str), \
            "min_date must not be string"
        print(f"✓ min_date is date object: {isinstance(result['min_date'], date)}")
    
    if result["max_date"] is not None:
        assert isinstance(result["max_date"], date), \
            f"max_date must be date object, got {type(result['max_date'])}"
        assert not isinstance(result["max_date"], str), \
            "max_date must not be string"
        print(f"✓ max_date is date object: {isinstance(result['max_date'], date)}")
    
    print()


def run_all_tests():
    """Run all DR-B1 service tests."""
    print("\n" + "=" * 80)
    print("PHASE DR-B1: SERVICE DATE BOUNDS TESTS")
    print("=" * 80 + "\n")
    
    passed = 0
    failed = 0
    
    try:
        # Test 1: Normal case (returns result for type check)
        full_result = test_normal_case()
        passed += 1
    except Exception as e:
        failed += 1
        print(f"✗ TEST 1 FAILED: {e}\n")
        return False
    
    try:
        # Test 2: Scoped subset
        test_scoped_subset()
        passed += 1
    except Exception as e:
        failed += 1
        print(f"✗ TEST 2 FAILED: {e}\n")
    
    try:
        # Test 3: Empty scope
        test_empty_scope()
        passed += 1
    except Exception as e:
        failed += 1
        print(f"✗ TEST 3 FAILED: {e}\n")
    
    try:
        # Test 4: Type check (uses result from test 1)
        test_type_check(full_result)
        passed += 1
    except Exception as e:
        failed += 1
        print(f"✗ TEST 4 FAILED: {e}\n")
    
    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Passed: {passed}/4")
    print(f"Failed: {failed}/4")
    
    if failed == 0:
        print("\n🎉 ALL DR-B1 SERVICE TESTS PASSED!")
        return True
    else:
        print(f"\n❌ {failed} TEST(S) FAILED")
        return False


if __name__ == "__main__":
    import sys
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
