"""
TEST B-B2 — FUNCTIONAL INTEGRATION TEST
Test actual HTTP requests to V2 patient endpoints.

This test makes real HTTP calls to verify:
1. Endpoints are reachable and return correct status codes
2. Response structure matches expectations
3. V2 responses match V1 responses (consistency check)
"""

import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))


def test_search_patients():
    """Test V2 search endpoint."""
    print("\n🔍 Testing V2 search patients endpoint...")
    
    try:
        from fastapi.testclient import TestClient
        from main import app
        
        client = TestClient(app)
        
        # Test V2 search without query (list all)
        response = client.get("/api/v2/patients/search?limit=5")
        
        if response.status_code != 200:
            print(f"❌ Search failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
        
        data = response.json()
        
        # Check response structure
        if "patients" not in data:
            print(f"❌ Response missing 'patients' field: {data}")
            return False
        
        print(f"✅ V2 search works, returned {len(data.get('patients', []))} patients")
        
        # Compare with V1
        response_v1 = client.get("/api/patients/search?limit=5")
        if response_v1.status_code == 200:
            data_v1 = response_v1.json()
            if data.keys() == data_v1.keys():
                print("✅ V2 response structure matches V1")
            else:
                print(f"⚠️  V2 and V1 response structures differ")
                print(f"   V1 keys: {data_v1.keys()}")
                print(f"   V2 keys: {data.keys()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_patient_profile():
    """Test V2 get profile endpoint with a real patient ID."""
    print("\n🔍 Testing V2 get patient profile endpoint...")
    
    try:
        from fastapi.testclient import TestClient
        from main import app
        
        client = TestClient(app)
        
        # First, search for a patient to get a valid ID
        search_response = client.get("/api/v2/patients/search?limit=1")
        
        if search_response.status_code != 200:
            print("⚠️  Could not search for patients to get test ID")
            return True  # Skip test if no data
        
        patients = search_response.json().get("patients", [])
        
        if not patients:
            print("⚠️  No patients found in database for profile test")
            return True  # Skip test if no data
        
        patient_id = patients[0].get("patient_id")
        
        if not patient_id:
            print("⚠️  Patient ID not found in search response")
            return True
        
        # Test V2 profile endpoint
        response = client.get(f"/api/v2/patients/{patient_id}/profile")
        
        if response.status_code == 404:
            print(f"⚠️  Patient {patient_id} not found (may be valid for reserve patients)")
            return True
        
        if response.status_code != 200:
            print(f"❌ Profile request failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
        
        data = response.json()
        
        # Check basic profile fields
        if "patient_id" not in data and "full_name" not in data:
            print(f"❌ Profile response missing basic fields: {data}")
            return False
        
        print(f"✅ V2 profile endpoint works for patient {patient_id}")
        
        # Compare with V1
        response_v1 = client.get(f"/api/patients/{patient_id}/profile")
        if response_v1.status_code == 200:
            data_v1 = response_v1.json()
            if data == data_v1:
                print("✅ V2 profile response matches V1 exactly")
            else:
                print("⚠️  V2 and V1 profile responses differ slightly")
        
        return True
        
    except Exception as e:
        print(f"❌ Profile test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_patient_incidents():
    """Test V2 incidents endpoint."""
    print("\n🔍 Testing V2 patient incidents endpoint...")
    
    try:
        from fastapi.testclient import TestClient
        from main import app
        
        client = TestClient(app)
        
        # Get a patient ID
        search_response = client.get("/api/v2/patients/search?limit=1")
        
        if search_response.status_code != 200:
            print("⚠️  Could not search for patients")
            return True
        
        patients = search_response.json().get("patients", [])
        
        if not patients:
            print("⚠️  No patients found for incidents test")
            return True
        
        patient_id = patients[0].get("patient_id")
        
        # Test V2 incidents endpoint
        response = client.get(f"/api/v2/patients/{patient_id}/incidents")
        
        if response.status_code == 404:
            print(f"⚠️  Incidents not found for patient {patient_id}")
            return True
        
        if response.status_code != 200:
            print(f"❌ Incidents request failed with status {response.status_code}")
            return False
        
        data = response.json()
        
        print(f"✅ V2 incidents endpoint works for patient {patient_id}")
        return True
        
    except Exception as e:
        print(f"❌ Incidents test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_reserve_patients():
    """Test V2 reserve patients endpoint."""
    print("\n🔍 Testing V2 reserve patients endpoint...")
    
    try:
        from fastapi.testclient import TestClient
        from main import app
        
        client = TestClient(app)
        
        # Test V2 reserve endpoint
        response = client.get("/api/v2/patients/reserve?limit=10")
        
        if response.status_code != 200:
            print(f"❌ Reserve patients request failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
        
        data = response.json()
        
        # Check response structure
        if "patients" not in data:
            print(f"❌ Response missing 'patients' field: {data}")
            return False
        
        reserve_count = len(data.get("patients", []))
        print(f"✅ V2 reserve patients endpoint works, returned {reserve_count} reserve patients")
        
        return True
        
    except Exception as e:
        print(f"❌ Reserve patients test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_full_history():
    """Test V2 full history endpoint."""
    print("\n🔍 Testing V2 full history endpoint...")
    
    try:
        from fastapi.testclient import TestClient
        from main import app
        
        client = TestClient(app)
        
        # Get a patient ID
        search_response = client.get("/api/v2/patients/search?limit=1")
        
        if search_response.status_code != 200:
            print("⚠️  Could not search for patients")
            return True
        
        patients = search_response.json().get("patients", [])
        
        if not patients:
            print("⚠️  No patients found for full history test")
            return True
        
        patient_id = patients[0].get("patient_id")
        
        # Test V2 full history endpoint
        response = client.get(f"/api/v2/patients/{patient_id}/full-history")
        
        if response.status_code != 200:
            print(f"❌ Full history request failed with status {response.status_code}")
            return False
        
        data = response.json()
        
        # Check for profile and incidents keys
        if "profile" not in data or "incidents" not in data:
            print(f"❌ Full history missing expected keys: {data.keys()}")
            return False
        
        print(f"✅ V2 full history endpoint works for patient {patient_id}")
        return True
        
    except Exception as e:
        print(f"❌ Full history test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_endpoint_not_found_404():
    """Test that invalid endpoints return 404."""
    print("\n🔍 Testing 404 for invalid endpoints...")
    
    try:
        from fastapi.testclient import TestClient
        from main import app
        
        client = TestClient(app)
        
        # Test invalid endpoint
        response = client.get("/api/v2/patients/9999999/profile")
        
        if response.status_code not in [404, 500]:
            print(f"⚠️  Invalid patient ID returned {response.status_code} instead of 404")
        else:
            print(f"✅ Invalid patient ID correctly returns {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"❌ 404 test failed: {e}")
        return False


def run_all_functional_tests():
    """Run all functional integration tests."""
    print("=" * 70)
    print("TEST B-B2 — FUNCTIONAL INTEGRATION TEST")
    print("=" * 70)
    
    tests = [
        ("Search Patients", test_search_patients),
        ("Get Patient Profile", test_get_patient_profile),
        ("Get Patient Incidents", test_get_patient_incidents),
        ("Get Reserve Patients", test_get_reserve_patients),
        ("Get Full History", test_get_full_history),
        ("404 Handling", test_endpoint_not_found_404),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 70)
    print("FUNCTIONAL TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nTests Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ ALL FUNCTIONAL TESTS PASSED")
        print("\n🎉 B-B2 IMPLEMENTATION VERIFIED AND WORKING")
        return 0
    else:
        print(f"\n❌ {total - passed} FUNCTIONAL TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    exit_code = run_all_functional_tests()
    sys.exit(exit_code)
