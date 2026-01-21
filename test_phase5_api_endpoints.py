"""
PHASE 5: API Endpoints Test Script
Tests all seasonal comparison API endpoints.
"""

import sys
import os
import requests
import json
from datetime import datetime

# Base URL for API (adjust if needed)
BASE_URL = "http://localhost:8000"

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(title)
    print("="*80 + "\n")


def test_available_quarters():
    """Test GET /api/seasonal-comparison/available-quarters"""
    print_section("TEST 1: Get Available Quarters")
    
    try:
        response = requests.get(
            f"{BASE_URL}/api/seasonal-comparison/available-quarters",
            params={
                "orgunit_id": 1,
                "orgunit_type": 0
            }
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success: Found {data['total_count']} available seasons")
            print("\nAvailable Seasons:")
            for season in data['available_seasons'][:5]:  # Show first 5
                print(f"   - Season {season['season_id']}: {season['name']}")
                print(f"     {season['start_date']} to {season['end_date']}")
            
            if data['total_count'] > 5:
                print(f"   ... and {data['total_count'] - 5} more")
            
            return data['available_seasons']
        else:
            print(f"❌ Failed: {response.text}")
            return []
    
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return []


def test_two_quarter_comparison_json():
    """Test POST /api/seasonal-comparison/2-quarters (JSON format)"""
    print_section("TEST 2: 2-Quarter Comparison (JSON)")
    
    try:
        payload = {
            "season_ids": [4, 5],  # Q4-2025, Q1-2026
            "orgunit_id": 1,
            "orgunit_type": 0,
            "user_id": 1,
            "format": "json"
        }
        
        print(f"Request Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(
            f"{BASE_URL}/api/seasonal-comparison/2-quarters",
            json=payload
        )
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success: Generated 2-quarter comparison")
            print(f"\nComparison Details:")
            print(f"   Type: {data['comparison_type']}")
            print(f"   Periods: {data['periods']}")
            print(f"   Organization: {data['orgunit_name']}")
            
            # Show summary
            summary = data['data']['summary']
            print(f"\n📊 Summary:")
            print(f"   Previous ({summary['previous']['period']}): {summary['previous']['total_cases']} cases")
            print(f"   Current ({summary['current']['period']}): {summary['current']['total_cases']} cases")
            
            # Show percentage changes
            changes = data['data']['percentage_changes']
            print(f"\n📈 Percentage Changes:")
            for metric, change in list(changes.items())[:5]:
                direction = "↑" if change > 0 else "↓" if change < 0 else "→"
                print(f"   - {metric}: {change:+.2f}% {direction}")
            
            return True
        else:
            print(f"❌ Failed: {response.text}")
            return False
    
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_two_quarter_comparison_docx():
    """Test POST /api/seasonal-comparison/2-quarters (DOCX format)"""
    print_section("TEST 3: 2-Quarter Comparison (DOCX Download)")
    
    try:
        payload = {
            "season_ids": [4, 5],
            "orgunit_id": 1,
            "orgunit_type": 0,
            "user_id": 1,
            "format": "docx"
        }
        
        print(f"Request Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(
            f"{BASE_URL}/api/seasonal-comparison/2-quarters",
            json=payload
        )
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            # Check content type
            content_type = response.headers.get('Content-Type', '')
            print(f"Content-Type: {content_type}")
            
            if 'wordprocessingml' in content_type:
                # Save file
                filename = f"test_2quarter_api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
                with open(filename, 'wb') as f:
                    f.write(response.content)
                
                file_size = os.path.getsize(filename) / 1024  # KB
                print(f"✅ Success: Downloaded Word document")
                print(f"   Filename: {filename}")
                print(f"   Size: {file_size:.2f} KB")
                return True
            else:
                print(f"❌ Failed: Expected Word document, got {content_type}")
                return False
        else:
            print(f"❌ Failed: {response.text}")
            return False
    
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_three_quarter_comparison_json():
    """Test POST /api/seasonal-comparison/3-quarters (JSON format)"""
    print_section("TEST 4: 3-Quarter Comparison (JSON)")
    
    try:
        payload = {
            "season_ids": [4, 5, 6],  # Q4-2025, Q1-2026, Q2-2026
            "orgunit_id": 1,
            "orgunit_type": 0,
            "user_id": 1,
            "format": "json"
        }
        
        print(f"Request Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(
            f"{BASE_URL}/api/seasonal-comparison/3-quarters",
            json=payload
        )
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success: Generated 3-quarter comparison")
            print(f"\nComparison Details:")
            print(f"   Type: {data['comparison_type']}")
            print(f"   Periods: {data['periods']}")
            print(f"   Organization: {data['orgunit_name']}")
            
            # Show trends
            trends = data['data']['trends']
            print(f"\n📊 Trend Indicators:")
            for metric, trend in list(trends.items())[:7]:
                print(f"   - {metric}: {trend}")
            
            return True
        else:
            print(f"❌ Failed: {response.text}")
            return False
    
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_three_quarter_comparison_docx():
    """Test POST /api/seasonal-comparison/3-quarters (DOCX format)"""
    print_section("TEST 5: 3-Quarter Comparison (DOCX Download)")
    
    try:
        payload = {
            "season_ids": [4, 5, 6],
            "orgunit_id": 1,
            "orgunit_type": 0,
            "user_id": 1,
            "format": "docx"
        }
        
        response = requests.post(
            f"{BASE_URL}/api/seasonal-comparison/3-quarters",
            json=payload
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')
            
            if 'wordprocessingml' in content_type:
                filename = f"test_3quarter_api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
                with open(filename, 'wb') as f:
                    f.write(response.content)
                
                file_size = os.path.getsize(filename) / 1024
                print(f"✅ Success: Downloaded Word document")
                print(f"   Filename: {filename}")
                print(f"   Size: {file_size:.2f} KB")
                return True
            else:
                print(f"❌ Failed: Expected Word document")
                return False
        else:
            print(f"❌ Failed: {response.text}")
            return False
    
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False


def test_four_quarter_comparison_json():
    """Test POST /api/seasonal-comparison/4-quarters (JSON format)"""
    print_section("TEST 6: 4-Quarter Comparison (JSON)")
    
    try:
        payload = {
            "season_ids": [4, 5, 6, 7],  # Full year
            "orgunit_id": 1,
            "orgunit_type": 0,
            "user_id": 1,
            "format": "json"
        }
        
        print(f"Request Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(
            f"{BASE_URL}/api/seasonal-comparison/4-quarters",
            json=payload
        )
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success: Generated 4-quarter comparison")
            print(f"\nComparison Details:")
            print(f"   Type: {data['comparison_type']}")
            print(f"   Periods: {data['periods']}")
            print(f"   Organization: {data['orgunit_name']}")
            
            # Show yearly totals
            yearly_totals = data['data'].get('yearly_totals', {})
            if yearly_totals:
                print(f"\n📅 Yearly Totals:")
                for metric, total in list(yearly_totals.items())[:5]:
                    print(f"   - {metric}: {total}")
            
            return True
        else:
            print(f"❌ Failed: {response.text}")
            return False
    
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_four_quarter_comparison_docx():
    """Test POST /api/seasonal-comparison/4-quarters (DOCX format)"""
    print_section("TEST 7: 4-Quarter Comparison (DOCX Download)")
    
    try:
        payload = {
            "season_ids": [4, 5, 6, 7],
            "orgunit_id": 1,
            "orgunit_type": 0,
            "user_id": 1,
            "format": "docx"
        }
        
        response = requests.post(
            f"{BASE_URL}/api/seasonal-comparison/4-quarters",
            json=payload
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')
            
            if 'wordprocessingml' in content_type:
                filename = f"test_4quarter_api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
                with open(filename, 'wb') as f:
                    f.write(response.content)
                
                file_size = os.path.getsize(filename) / 1024
                print(f"✅ Success: Downloaded Word document")
                print(f"   Filename: {filename}")
                print(f"   Size: {file_size:.2f} KB")
                return True
            else:
                print(f"❌ Failed: Expected Word document")
                return False
        else:
            print(f"❌ Failed: {response.text}")
            return False
    
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False


def test_error_handling():
    """Test API error handling"""
    print_section("TEST 8: Error Handling")
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Invalid season_ids count
    print("8A. Testing invalid season count (2-quarters with 3 IDs)...")
    try:
        payload = {
            "season_ids": [4, 5, 6],  # 3 IDs instead of 2
            "orgunit_id": 1,
            "orgunit_type": 0,
            "format": "json"
        }
        
        response = requests.post(
            f"{BASE_URL}/api/seasonal-comparison/2-quarters",
            json=payload
        )
        
        if response.status_code == 422:  # Validation error
            print(f"✅ Correctly rejected invalid request (422)")
            tests_passed += 1
        else:
            print(f"⚠️  Expected 422, got {response.status_code}")
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
    
    # Test 2: Invalid format
    print("\n8B. Testing invalid format parameter...")
    try:
        payload = {
            "season_ids": [4, 5],
            "orgunit_id": 1,
            "orgunit_type": 0,
            "format": "pdf"  # Invalid format
        }
        
        response = requests.post(
            f"{BASE_URL}/api/seasonal-comparison/2-quarters",
            json=payload
        )
        
        if response.status_code == 422:
            print(f"✅ Correctly rejected invalid format (422)")
            tests_passed += 1
        else:
            print(f"⚠️  Expected 422, got {response.status_code}")
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
    
    # Test 3: Invalid orgunit_id
    print("\n8C. Testing invalid orgunit_id...")
    try:
        payload = {
            "season_ids": [4, 5],
            "orgunit_id": 0,  # Invalid (must be >= 1)
            "orgunit_type": 0,
            "format": "json"
        }
        
        response = requests.post(
            f"{BASE_URL}/api/seasonal-comparison/2-quarters",
            json=payload
        )
        
        if response.status_code == 422:
            print(f"✅ Correctly rejected invalid orgunit_id (422)")
            tests_passed += 1
        else:
            print(f"⚠️  Expected 422, got {response.status_code}")
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
    
    print(f"\n✅ Error handling tests: {tests_passed}/{total_tests} passed")
    return tests_passed == total_tests


def run_all_tests():
    """Run all API endpoint tests"""
    print("\n" + "="*80)
    print("PHASE 5: API ENDPOINTS - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code != 200:
            print("\n❌ ERROR: API server not responding at {BASE_URL}")
            print("Please start the server with: uvicorn backend.main:app --reload")
            return False
    except Exception as e:
        print(f"\n❌ ERROR: Cannot connect to API server at {BASE_URL}")
        print(f"   {str(e)}")
        print("\nPlease start the server with: uvicorn backend.main:app --reload")
        return False
    
    print(f"\n✅ API server is running at {BASE_URL}")
    
    # Run tests
    results = {
        "available_quarters": test_available_quarters() is not None,
        "2q_json": test_two_quarter_comparison_json(),
        "2q_docx": test_two_quarter_comparison_docx(),
        "3q_json": test_three_quarter_comparison_json(),
        "3q_docx": test_three_quarter_comparison_docx(),
        "4q_json": test_four_quarter_comparison_json(),
        "4q_docx": test_four_quarter_comparison_docx(),
        "error_handling": test_error_handling()
    }
    
    # Summary
    print_section("TEST SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    print("\nDetailed Results:")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status} - {test_name}")
    
    if passed == total:
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED - PHASE 5 API ENDPOINTS COMPLETE!")
        print("="*80)
        return True
    else:
        print("\n⚠️  Some tests failed. Review output above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
