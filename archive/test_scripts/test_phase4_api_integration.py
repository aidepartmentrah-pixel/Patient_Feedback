"""
Phase 4 API Integration Test
Tests the chart API endpoints: /api/settings/training/charts/*
"""

import requests
import json
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"
API_PREFIX = "/api/settings/training"

print("=" * 80)
print("PHASE 4: API INTEGRATION TEST - Visual Charts")
print("=" * 80)
print()

def test_chart_endpoint(endpoint_path, chart_name, query_params=None):
    """Generic function to test a chart endpoint."""
    print(f"[TEST] GET {API_PREFIX}/{endpoint_path}")
    print("-" * 80)
    
    try:
        url = f"{BASE_URL}{API_PREFIX}/{endpoint_path}"
        response = requests.get(url, params=query_params, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ FAILED: Expected status 200, got {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        data = response.json()
        
        # Validate structure
        required_fields = ["labels", "datasets", "metadata"]
        for field in required_fields:
            if field not in data:
                print(f"❌ FAILED: Missing field '{field}' in response")
                return False
        
        print(f"✓ Response has all required fields")
        
        # Check datasets
        datasets = data["datasets"]
        print(f"✓ Found {len(datasets)} dataset(s)")
        
        if len(datasets) > 0:
            first_dataset = datasets[0]
            print(f"  - First dataset label: {first_dataset.get('label', 'N/A')}")
            print(f"  - Data points: {len(first_dataset.get('data', []))}")
            
            # Check bilingual support
            if "label_ar" in first_dataset:
                print(f"  ✓ Bilingual support present")
        
        # Check metadata
        metadata = data["metadata"]
        print(f"✓ Metadata:")
        for key, value in metadata.items():
            if isinstance(value, dict):
                print(f"  - {key}: {json.dumps(value)}")
            else:
                print(f"  - {key}: {value}")
        
        # Check labels
        print(f"✓ Labels: {len(data['labels'])} items")
        if len(data['labels']) > 0:
            print(f"  - First label: {data['labels'][0]}")
            if len(data['labels']) > 1:
                print(f"  - Last label: {data['labels'][-1]}")
        
        print()
        print(f"✅ PASSED: {chart_name}")
        print()
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ FAILED: Could not connect to server")
        print("   Make sure the FastAPI server is running on http://localhost:8000")
        return False
    except Exception as e:
        print(f"❌ FAILED: Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_db_growth_chart():
    """Test DB growth chart endpoint."""
    return test_chart_endpoint("charts/db-growth", "DB Growth Chart", {"days": 30})


def test_performance_trends_chart():
    """Test performance trends chart endpoint."""
    return test_chart_endpoint("charts/performance-trends", "Performance Trends Chart")


def test_training_timeline_chart():
    """Test training timeline chart endpoint."""
    return test_chart_endpoint("charts/training-timeline", "Training Timeline Chart", {"limit": 20})


def test_family_comparison_chart():
    """Test family comparison chart endpoint."""
    return test_chart_endpoint("charts/family-comparison", "Family Comparison Chart")


def test_chart_with_different_params():
    """Test that query parameters work correctly."""
    print(f"[TEST] Testing query parameters")
    print("-" * 80)
    
    try:
        # Test with different day ranges
        for days in [7, 30, 90]:
            url = f"{BASE_URL}{API_PREFIX}/charts/db-growth?days={days}"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                print(f"❌ FAILED: days={days} returned {response.status_code}")
                return False
            
            data = response.json()
            print(f"✓ days={days}: {len(data['labels'])} data points")
        
        # Test with different limits
        for limit in [5, 10, 20]:
            url = f"{BASE_URL}{API_PREFIX}/charts/training-timeline?limit={limit}"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                print(f"❌ FAILED: limit={limit} returned {response.status_code}")
                return False
            
            data = response.json()
            actual_count = min(limit, data['metadata']['total_runs'])
            print(f"✓ limit={limit}: {actual_count} runs shown")
        
        print()
        print(f"✅ PASSED: Query parameters work correctly")
        print()
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def test_json_serialization():
    """Test that all responses are valid JSON."""
    print(f"[TEST] JSON serialization")
    print("-" * 80)
    
    try:
        endpoints = [
            "charts/db-growth",
            "charts/performance-trends",
            "charts/training-timeline",
            "charts/family-comparison"
        ]
        
        for endpoint in endpoints:
            url = f"{BASE_URL}{API_PREFIX}/{endpoint}"
            response = requests.get(url, timeout=10)
            
            # Try to parse JSON
            try:
                data = response.json()
                # Try to re-serialize
                json_str = json.dumps(data, ensure_ascii=False)
                print(f"✓ {endpoint}: Valid JSON ({len(json_str)} bytes)")
            except (ValueError, TypeError) as e:
                print(f"❌ FAILED: {endpoint} - {str(e)}")
                return False
        
        print()
        print(f"✅ PASSED: All endpoints return valid JSON")
        print()
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


if __name__ == "__main__":
    print("Prerequisites:")
    print("  1. FastAPI server must be running on port 8000")
    print("  2. Training must have been run at least once")
    print()
    
    # Check server availability
    try:
        response = requests.get(f"{BASE_URL}{API_PREFIX}/status", timeout=5)
        print("✅ Backend server is running")
        print()
    except Exception as e:
        print("❌ Backend server is not reachable")
        print(f"   Error: {e}")
        print()
        print("Please start the backend server:")
        print("  cd backend")
        print("  uvicorn main:app --reload")
        exit(1)
    
    # Run tests
    results = []
    
    results.append(test_db_growth_chart())
    results.append(test_performance_trends_chart())
    results.append(test_training_timeline_chart())
    results.append(test_family_comparison_chart())
    results.append(test_chart_with_different_params())
    results.append(test_json_serialization())
    
    # Summary
    print()
    print("=" * 80)
    passed = sum(results)
    total = len(results)
    
    print(f"RESULTS: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 80)
    
    if passed == total:
        print("🎉 ALL API INTEGRATION TESTS PASSED!")
    else:
        print("⚠️ Some tests failed. Please review the output above.")
        exit(1)
