"""
Test script for search endpoints
Run this after starting the server with: python -m uvicorn main:app --reload
"""
import requests
import json

base_url = "http://127.0.0.1:8000"

def test_search_patients():
    """Test patient search endpoint"""
    print("\n" + "="*60)
    print("Testing Patient Search")
    print("="*60)
    
    # Test with a common Arabic letter
    response = requests.get(f"{base_url}/api/records/search/patients", params={"q": "ا", "limit": 5})
    
    print(f"Status Code: {response.status_code}")
    print(f"Response:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    
    return response.status_code == 200

def test_search_doctors():
    """Test doctor search endpoint"""
    print("\n" + "="*60)
    print("Testing Doctor Search")
    print("="*60)
    
    # Test with a common Arabic letter
    response = requests.get(f"{base_url}/api/records/search/doctors", params={"q": "د", "limit": 5})
    
    print(f"Status Code: {response.status_code}")
    print(f"Response:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    
    return response.status_code == 200

def test_search_employees():
    """Test employee search endpoint"""
    print("\n" + "="*60)
    print("Testing Employee Search")
    print("="*60)
    
    # Test with a common Arabic letter
    response = requests.get(f"{base_url}/api/records/search/employees", params={"q": "م", "limit": 5})
    
    print(f"Status Code: {response.status_code}")
    print(f"Response:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    
    return response.status_code == 200

def test_health():
    """Test if server is running"""
    print("\n" + "="*60)
    print("Testing Server Health")
    print("="*60)
    
    try:
        response = requests.get(f"{base_url}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the server is running: python -m uvicorn main:app --reload")
        return False

if __name__ == "__main__":
    print("\n🚀 Starting Search API Tests")
    print(f"Base URL: {base_url}")
    
    # Test server health first
    if not test_health():
        print("\n❌ Server is not running. Please start it first.")
        exit(1)
    
    # Run search tests
    results = {
        "patients": test_search_patients(),
        "doctors": test_search_doctors(),
        "employees": test_search_employees()
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.capitalize()}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
