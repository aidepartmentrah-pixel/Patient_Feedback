"""
Test script for Red Flags and Never Events API endpoints
Tests all the fixed backend endpoints to ensure they work correctly.
"""

import requests
import json
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8001"

def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def test_endpoint(name, url, params=None):
    """Test a GET endpoint and print results"""
    print(f"\n🔍 Testing: {name}")
    print(f"   URL: {url}")
    if params:
        print(f"   Params: {params}")
    
    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ SUCCESS")
            
            # Print summary of data
            if isinstance(data, dict):
                if 'total' in data:
                    print(f"   Total records: {data['total']}")
                if 'red_flags' in data:
                    print(f"   Red flags count: {len(data['red_flags'])}")
                if 'never_events' in data:
                    print(f"   Never events count: {len(data['never_events'])}")
                if 'data' in data and isinstance(data['data'], list):
                    print(f"   Trend data points: {len(data['data'])}")
            
            return True
        else:
            print(f"   ❌ FAILED: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)}")
        return False

def main():
    print_section("RED FLAGS API TESTS")
    
    # Test Red Flags endpoints
    results = {}
    
    results['rf_list'] = test_endpoint(
        "Red Flags List",
        f"{BASE_URL}/api/red-flags",
        params={"limit": 10}
    )
    
    results['rf_stats'] = test_endpoint(
        "Red Flags Statistics",
        f"{BASE_URL}/api/red-flags/statistics"
    )
    
    results['rf_trends'] = test_endpoint(
        "Red Flags Trends",
        f"{BASE_URL}/api/red-flags/trends",
        params={"granularity": "monthly"}
    )
    
    print_section("NEVER EVENTS API TESTS")
    
    # Test Never Events endpoints
    results['ne_list'] = test_endpoint(
        "Never Events List",
        f"{BASE_URL}/api/never-events",
        params={"limit": 10}
    )
    
    results['ne_stats'] = test_endpoint(
        "Never Events Statistics",
        f"{BASE_URL}/api/never-events/statistics"
    )
    
    results['ne_trends'] = test_endpoint(
        "Never Events Trends",
        f"{BASE_URL}/api/never-events/trends",
        params={"granularity": "monthly"}
    )
    
    results['ne_category'] = test_endpoint(
        "Never Events Category Breakdown",
        f"{BASE_URL}/api/never-events/category-breakdown"
    )
    
    results['ne_timeline'] = test_endpoint(
        "Never Events Timeline Comparison",
        f"{BASE_URL}/api/never-events/timeline-comparison",
        params={"period": "month"}
    )
    
    print_section("TEST SUMMARY")
    
    total_tests = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total_tests - passed
    
    print(f"\nTotal Tests: {total_tests}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Backend is working correctly.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Check the errors above.")
        print("\nFailed endpoints:")
        for name, result in results.items():
            if not result:
                print(f"  - {name}")

if __name__ == "__main__":
    main()
