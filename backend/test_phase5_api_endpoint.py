"""
Phase 5 Test: API Router - POST Endpoint
Test the POST /api/doctors endpoint
"""

import requests
import sys
from datetime import datetime


BASE_URL = "http://localhost:8000"


def test_create_doctor_success():
    """Test 1: Create doctor via API - Success (201)"""
    print("\n" + "="*60)
    print("TEST 1: Create doctor via API - Success")
    print("="*60)
    
    try:
        unique_name = f"Dr. API Test {datetime.now().strftime('%H%M%S')}"
        
        payload = {
            "doctor_name": unique_name,
            "specialty": "Cardiology",
            "is_active": True,
            "source_system": "API_TEST"
        }
        
        print(f"\nSending POST request to {BASE_URL}/api/doctors")
        print(f"Payload: {payload}")
        
        response = requests.post(f"{BASE_URL}/api/doctors", json=payload)
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 201:
            data = response.json()
            print(f"✅ Success: {data.get('message')}")
            print(f"   Message (AR): {data.get('message_ar')}")
            print(f"   Doctor ID: {data.get('doctor', {}).get('id')}")
            print(f"   Doctor Name: {data.get('doctor', {}).get('name_en')}")
            print(f"   Source: {data.get('doctor', {}).get('source')}")
            print("\n✅ PASS: Doctor created successfully with 201")
            return True, data.get('doctor', {}).get('id')
        else:
            print(f"❌ FAIL: Expected 201, got {response.status_code}")
            print(f"Response: {response.text}")
            return False, None
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False, None


def test_create_doctor_minimal():
    """Test 2: Create doctor with minimal fields"""
    print("\n" + "="*60)
    print("TEST 2: Create doctor with minimal required fields")
    print("="*60)
    
    try:
        unique_name = f"Dr. Minimal API {datetime.now().strftime('%H%M%S')}"
        
        payload = {
            "doctor_name": unique_name
            # Only required field
        }
        
        print(f"\nPayload: {payload}")
        response = requests.post(f"{BASE_URL}/api/doctors", json=payload)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 201:
            data = response.json()
            print(f"✅ Doctor created with minimal fields")
            print(f"   ID: {data.get('doctor', {}).get('id')}")
            print("\n✅ PASS: Minimal fields accepted")
            return True
        else:
            print(f"❌ FAIL: Expected 201, got {response.status_code}")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


def test_validation_empty_name():
    """Test 3: Reject empty name - 400 Bad Request"""
    print("\n" + "="*60)
    print("TEST 3: Reject empty name (400)")
    print("="*60)
    
    try:
        payload = {
            "doctor_name": ""  # Empty
        }
        
        response = requests.post(f"{BASE_URL}/api/doctors", json=payload)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 422 or response.status_code == 400:
            print(f"✅ Correctly rejected with {response.status_code}")
            print("\n✅ PASS: Empty name validation working")
            return True
        else:
            print(f"❌ FAIL: Expected 400/422, got {response.status_code}")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


def test_validation_short_name():
    """Test 4: Reject short name - 422 Unprocessable Entity"""
    print("\n" + "="*60)
    print("TEST 4: Reject name < 3 characters (422)")
    print("="*60)
    
    try:
        payload = {
            "doctor_name": "Dr"  # Only 2 chars
        }
        
        response = requests.post(f"{BASE_URL}/api/doctors", json=payload)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 422:
            print(f"✅ Correctly rejected with 422")
            data = response.json()
            print(f"   Detail: {data.get('detail', 'N/A')}")
            print("\n✅ PASS: Short name validation working")
            return True
        else:
            print(f"❌ FAIL: Expected 422, got {response.status_code}")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


def test_duplicate_doctor():
    """Test 5: Reject duplicate - 409 Conflict"""
    print("\n" + "="*60)
    print("TEST 5: Reject duplicate doctor (409)")
    print("="*60)
    
    try:
        unique_name = f"Dr. Duplicate API {datetime.now().strftime('%H%M%S%f')}"
        
        payload = {
            "doctor_name": unique_name,
            "specialty": "Test"
        }
        
        # Create first
        response1 = requests.post(f"{BASE_URL}/api/doctors", json=payload)
        
        if response1.status_code != 201:
            print(f"⚠️  First creation failed with {response1.status_code}")
            return False
        
        print(f"✅ First doctor created: {unique_name}")
        
        # Try duplicate
        response2 = requests.post(f"{BASE_URL}/api/doctors", json=payload)
        
        print(f"Status Code: {response2.status_code}")
        
        if response2.status_code == 409:
            data = response2.json()
            print(f"✅ Correctly rejected with 409 Conflict")
            print(f"   Error: {data.get('detail', {}).get('error')}")
            print(f"   Message: {data.get('detail', {}).get('message')}")
            print("\n✅ PASS: Duplicate detection working")
            return True
        else:
            print(f"❌ FAIL: Expected 409, got {response2.status_code}")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


def test_response_structure():
    """Test 6: Verify response structure"""
    print("\n" + "="*60)
    print("TEST 6: Verify response structure")
    print("="*60)
    
    try:
        unique_name = f"Dr. Structure {datetime.now().strftime('%H%M%S')}"
        
        payload = {
            "doctor_name": unique_name,
            "specialty": "Test Specialty"
        }
        
        response = requests.post(f"{BASE_URL}/api/doctors", json=payload)
        
        if response.status_code != 201:
            print(f"❌ FAIL: Creation failed with {response.status_code}")
            return False
        
        data = response.json()
        
        required_keys = ['success', 'message', 'message_ar', 'doctor']
        required_doctor_keys = ['id', 'name_en', 'name_ar', 'specialty', 'status', 'source']
        
        print("\nChecking response structure:")
        
        all_present = True
        for key in required_keys:
            if key in data:
                print(f"  ✅ {key}")
            else:
                print(f"  ❌ {key} - MISSING")
                all_present = False
        
        print("\nChecking doctor object:")
        doctor = data.get('doctor', {})
        for key in required_doctor_keys:
            if key in doctor:
                print(f"  ✅ {key}")
            else:
                print(f"  ❌ {key} - MISSING")
                all_present = False
        
        if all_present:
            print("\n✅ PASS: Response structure correct")
            return True
        else:
            print("\n❌ FAIL: Missing required fields")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


def test_created_doctor_searchable():
    """Test 7: Verify created doctor appears in search"""
    print("\n" + "="*60)
    print("TEST 7: Created doctor appears in search")
    print("="*60)
    
    try:
        unique_term = f"ZZZSearchAPI{datetime.now().strftime('%H%M%S')}"
        unique_name = f"Dr. {unique_term}"
        
        # Create doctor
        payload = {
            "doctor_name": unique_name,
            "specialty": "Searchable"
        }
        
        response = requests.post(f"{BASE_URL}/api/doctors", json=payload)
        
        if response.status_code != 201:
            print(f"❌ FAIL: Creation failed")
            return False
        
        doctor_id = response.json().get('doctor', {}).get('id')
        print(f"✅ Created doctor ID: {doctor_id}")
        
        # Search for it
        search_response = requests.get(
            f"{BASE_URL}/api/doctors",
            params={"query": unique_term, "limit": 10}
        )
        
        if search_response.status_code != 200:
            print(f"❌ FAIL: Search failed with {search_response.status_code}")
            return False
        
        search_data = search_response.json()
        doctors = search_data.get('doctors', [])
        
        found = any(d.get('id') == doctor_id for d in doctors)
        
        if found:
            print(f"✅ Found doctor in search results")
            print("\n✅ PASS: Created doctor is searchable")
            return True
        else:
            print(f"❌ FAIL: Doctor not found in search")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


def run_all_tests():
    """Run all Phase 5 tests"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║  PHASE 5: API ROUTER - POST ENDPOINT                    ║")
    print("║  Testing POST /api/doctors                              ║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    print("\n⚠️  NOTE: Make sure the API server is running!")
    print("   Run: cd backend; uvicorn main:app --reload")
    print()
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=2)
        print("✅ Server is running\n")
    except:
        print("❌ ERROR: Server is not running!")
        print("   Please start the server first:")
        print("   cd backend; uvicorn main:app --reload\n")
        return False
    
    tests = [
        ("Create Doctor Success (201)", test_create_doctor_success),
        ("Create with Minimal Fields", test_create_doctor_minimal),
        ("Reject Empty Name (400)", test_validation_empty_name),
        ("Reject Short Name (422)", test_validation_short_name),
        ("Reject Duplicate (409)", test_duplicate_doctor),
        ("Response Structure", test_response_structure),
        ("Created Doctor Searchable", test_created_doctor_searchable),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if isinstance(result, tuple):
                result = result[0]
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║  TEST SUMMARY                                            ║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print("\n" + "-"*60)
    print(f"  Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("-"*60)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Phase 5 is 100% complete!")
        print("✅ POST endpoint working correctly")
        print("✅ Proper HTTP status codes")
        print("✅ Error handling implemented")
        print("✅ Ready to proceed to Phase 6 (Integration Testing)")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix issues before proceeding.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
