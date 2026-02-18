"""
Test Patient Search Endpoint in API v2
Tests if /api/v2/patients/search is working properly
"""

import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_patient_search_v2():
    """Test the v2 patient search endpoint"""
    print("\n" + "="*60)
    print("Testing Patient Search - API v2")
    print("="*60)
    
    # Test 1: Search with query parameter (name search)
    print("\n[TEST 1] Search by name query...")
    try:
        response = requests.get(
            f"{BASE_URL}/api/v2/patients/search",
            params={"query": "محمد", "limit": 5}
        )
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ SUCCESS")
            print(f"  - Count: {data.get('count', 0)}")
            print(f"  - Patients returned: {len(data.get('patients', []))}")
            if data.get('patients'):
                print(f"\nFirst patient:")
                first = data['patients'][0]
                print(f"  - ID: {first.get('patient_id')}")
                print(f"  - Name: {first.get('full_name')}")
                print(f"  - MRN: {first.get('mrn')}")
                print(f"  - Source: {first.get('source')}")
        else:
            print(f"✗ FAILED")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
    
    # Test 2: Search without parameters
    print("\n[TEST 2] Search without parameters (should return empty)...")
    try:
        response = requests.get(f"{BASE_URL}/api/v2/patients/search")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ SUCCESS")
            print(f"  - Message: {data.get('message')}")
            print(f"  - Count: {data.get('count', 0)}")
        else:
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
    
    # Test 3: Compare v1 vs v2
    print("\n[TEST 3] Comparing v1 vs v2 results...")
    try:
        # Call v1
        v1_response = requests.get(
            f"{BASE_URL}/api/patients/search",
            params={"limit": 5}
        )
        
        # Call v2
        v2_response = requests.get(
            f"{BASE_URL}/api/v2/patients/search",
            params={"limit": 5}
        )
        
        print(f"v1 Status: {v1_response.status_code}")
        print(f"v2 Status: {v2_response.status_code}")
        
        if v1_response.status_code == 200:
            v1_data = v1_response.json()
            print(f"v1 returned {v1_data.get('count', 0)} patients")
        
        if v2_response.status_code == 200:
            v2_data = v2_response.json()
            print(f"v2 returned {v2_data.get('count', 0)} patients")
            
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
    
    print("\n" + "="*60)
    print("Test Complete")
    print("="*60)

if __name__ == "__main__":
    test_patient_search_v2()
