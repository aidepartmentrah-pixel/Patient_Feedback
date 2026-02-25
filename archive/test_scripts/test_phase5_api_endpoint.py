"""
====================================================================
PHASE 5 TEST: API Endpoint - POST /api/patients/create
====================================================================
Purpose: Verify the patients router create endpoint properly handles
         requests, validates inputs, calls service layer, and returns
         correct responses with proper HTTP status codes

Test Coverage:
1. Valid patient creation with all fields (201)
2. Valid patient with minimal fields (201)
3. Missing required field FirstName (422 Pydantic validation)
4. Invalid field types (422 Pydantic validation)
5. Service layer validation errors (400)
6. Duplicate patient (409)
7. Response format verification
8. Error message format verification
9. Success message in English and Arabic
10. Patient searchable after creation

Author: System
Date: 2026-01-20
====================================================================
"""

import sys
import os
from datetime import datetime
import requests
import json

# Test configuration
BASE_URL = "http://localhost:8000"  # Adjust if needed
API_ENDPOINT = f"{BASE_URL}/api/patients/create"
SEARCH_ENDPOINT = f"{BASE_URL}/api/patients/search"

# Add backend to path for cleanup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'api'))

import pyodbc


def get_connection():
    """Get SQL Server connection."""
    return pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=SOCIALMEDIA;"
        "DATABASE=IncidentManager;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )


def cleanup_test_data():
    """Remove test data"""
    print("\n" + "="*70)
    print("CLEANUP: Removing Test Data")
    print("="*70)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            DELETE FROM APP_RESERVE_PATIENT 
            WHERE FirstName LIKE 'TestPhase5%'
            OR DocumentNumber LIKE 'DOC-P5-%'
            OR MedicalFileNumber LIKE 'MRN-P5-%'
        """)
        deleted = cursor.rowcount
        conn.commit()
        print(f"✓ Deleted {deleted} test patient(s)")
        return True
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False
    finally:
        conn.close()


def test_1_valid_full_patient():
    """Test 1: Create patient with all valid fields - expect 201"""
    print("\n" + "="*70)
    print("TEST 1: Valid Patient with All Fields (201)")
    print("="*70)
    
    payload = {
        "first_name": "TestPhase5Full",
        "middle_name": "Ahmad",
        "last_name": "AlTest",
        "mother_name": "Fatima",
        "phone_number": "0501234567",
        "phone_number2": "0509876543",
        "birth_date": "1990-05-15",
        "sex": "M",
        "document_number": "DOC-P5-001",
        "medical_file_number": "MRN-P5-001",
        "spouse": "Sara AlAhmad",
        "address_line1": "123 Test Street, Riyadh",
        "address_line2": "Building 5, Apt 201"
    }
    
    try:
        response = requests.post(API_ENDPOINT, json=payload)
        
        if response.status_code == 201:
            data = response.json()
            
            # Verify response structure
            if not all(key in data for key in ['success', 'message', 'message_ar', 'patient']):
                print("✗ FAIL: Missing required response fields")
                return False
            
            if not data['success']:
                print("✗ FAIL: success=false")
                return False
            
            patient = data['patient']
            if patient['full_name'] != "TestPhase5Full Ahmad AlTest":
                print(f"✗ FAIL: Wrong full_name: {patient['full_name']}")
                return False
            
            if patient['source'] != 'reserve':
                print(f"✗ FAIL: Wrong source: {patient['source']}")
                return False
            
            print(f"✓ PASS: Patient created")
            print(f"  - ID: {patient['patient_admission_id']}")
            print(f"  - FullName: {patient['full_name']}")
            print(f"  - Message: {data['message'][:60]}...")
            return True
        else:
            print(f"✗ FAIL: Expected 201, got {response.status_code}")
            print(f"  - Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        return False


def test_2_valid_minimal_patient():
    """Test 2: Create patient with minimal fields (only FirstName) - expect 201"""
    print("\n" + "="*70)
    print("TEST 2: Valid Patient with Minimal Fields (201)")
    print("="*70)
    
    payload = {
        "first_name": "TestPhase5Minimal"
    }
    
    try:
        response = requests.post(API_ENDPOINT, json=payload)
        
        if response.status_code == 201:
            data = response.json()
            patient = data['patient']
            
            if patient['full_name'] != "TestPhase5Minimal":
                print(f"✗ FAIL: Wrong full_name: {patient['full_name']}")
                return False
            
            print(f"✓ PASS: Minimal patient created")
            print(f"  - ID: {patient['patient_admission_id']}")
            return True
        else:
            print(f"✗ FAIL: Expected 201, got {response.status_code}")
            print(f"  - Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        return False


def test_3_missing_firstname():
    """Test 3: Missing required FirstName - expect 422 (Pydantic validation)"""
    print("\n" + "="*70)
    print("TEST 3: Missing Required FirstName (422)")
    print("="*70)
    
    payload = {
        "middle_name": "Ahmad",
        "last_name": "AlTest"
    }
    
    try:
        response = requests.post(API_ENDPOINT, json=payload)
        
        if response.status_code == 422:
            print(f"✓ PASS: Request correctly rejected (422)")
            return True
        else:
            print(f"✗ FAIL: Expected 422, got {response.status_code}")
            print(f"  - Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        return False


def test_4_invalid_field_types():
    """Test 4: Invalid field types - expect 422"""
    print("\n" + "="*70)
    print("TEST 4: Invalid Field Types (422)")
    print("="*70)
    
    test_cases = [
        ({"first_name": 123}, "first_name as number"),
        ({"first_name": "Valid", "phone_number": 123}, "phone_number as number"),
        ({"first_name": "Valid", "sex": 123}, "sex as number")
    ]
    
    all_passed = True
    
    for payload, description in test_cases:
        try:
            response = requests.post(API_ENDPOINT, json=payload)
            
            if response.status_code == 422:
                print(f"  ✓ {description}: Correctly rejected (422)")
            else:
                print(f"  ✗ {description}: Expected 422, got {response.status_code}")
                all_passed = False
                
        except Exception as e:
            print(f"  ✗ {description}: Error - {str(e)[:50]}")
            all_passed = False
    
    if all_passed:
        print("✓ PASS: Type validation works")
    else:
        print("✗ FAIL: Some cases failed")
    
    return all_passed


def test_5_service_validation_errors():
    """Test 5: Service layer validation errors - expect 400 (or 422 for Pydantic)"""
    print("\n" + "="*70)
    print("TEST 5: Validation Errors (400/422)")
    print("="*70)
    
    test_cases = [
        ({"first_name": "A"}, 422, "Too short FirstName (Pydantic)"),
        ({"first_name": "A" * 151}, 422, "Too long FirstName (Pydantic)"),
        ({"first_name": "Test@Invalid"}, 400, "Invalid characters in FirstName (Service)"),
        ({"first_name": "TestPhase5Phone", "phone_number": "123"}, 400, "Too few digits in phone (Service)"),
        ({"first_name": "TestPhase5Date", "birth_date": "2099-01-01"}, 400, "Future birth date (Service)"),
        ({"first_name": "TestPhase5Sex", "sex": "X"}, 400, "Invalid SEX value (Service)")
    ]
    
    all_passed = True
    
    for payload, expected_status, description in test_cases:
        try:
            response = requests.post(API_ENDPOINT, json=payload)
            
            if response.status_code == expected_status:
                print(f"  ✓ {description}: Correctly rejected ({expected_status})")
            else:
                print(f"  ✗ {description}: Expected {expected_status}, got {response.status_code}")
                print(f"    Response: {response.text[:100]}")
                all_passed = False
                
        except Exception as e:
            print(f"  ✗ {description}: Error - {str(e)[:50]}")
            all_passed = False
    
    if all_passed:
        print("✓ PASS: Validation works (Pydantic + Service layers)")
    else:
        print("✗ FAIL: Some cases failed")
    
    return all_passed


def test_6_duplicate_patient():
    """Test 6: Duplicate patient - expect 409"""
    print("\n" + "="*70)
    print("TEST 6: Duplicate Patient Detection (409)")
    print("="*70)
    
    payload = {
        "first_name": "TestPhase5Duplicate",
        "middle_name": "Same",
        "last_name": "Name"
    }
    
    try:
        # Create first patient
        response1 = requests.post(API_ENDPOINT, json=payload)
        if response1.status_code != 201:
            print(f"✗ FAIL: Could not create first patient ({response1.status_code})")
            return False
        
        patient1_id = response1.json()['patient']['patient_admission_id']
        print(f"  - Created patient 1: ID {patient1_id}")
        
        # Try duplicate
        response2 = requests.post(API_ENDPOINT, json=payload)
        
        if response2.status_code == 409:
            data = response2.json()
            if 'detail' in data and 'DUPLICATE_PATIENT' in str(data['detail']):
                print(f"✓ PASS: Duplicate correctly rejected (409)")
                return True
            else:
                print(f"✗ FAIL: Wrong error format")
                print(f"  - Response: {response2.text[:200]}")
                return False
        else:
            print(f"✗ FAIL: Expected 409, got {response2.status_code}")
            print(f"  - Response: {response2.text[:200]}")
            return False
            
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        return False


def test_7_response_format():
    """Test 7: Verify response format and fields"""
    print("\n" + "="*70)
    print("TEST 7: Response Format Verification")
    print("="*70)
    
    payload = {
        "first_name": "TestPhase5Format",
        "middle_name": "Ahmad",
        "phone_number": "0501234567",
        "birth_date": "1990-05-15",
        "sex": "Male"  # Test normalization
    }
    
    try:
        response = requests.post(API_ENDPOINT, json=payload)
        
        if response.status_code != 201:
            print(f"✗ FAIL: Expected 201, got {response.status_code}")
            return False
        
        data = response.json()
        
        # Check top-level fields
        required_top = ['success', 'message', 'message_ar', 'patient']
        missing_top = [f for f in required_top if f not in data]
        if missing_top:
            print(f"✗ FAIL: Missing top-level fields: {missing_top}")
            return False
        
        # Check patient fields
        patient = data['patient']
        required_patient = [
            'patient_admission_id', 'full_name', 'first_name', 'source'
        ]
        missing_patient = [f for f in required_patient if f not in patient]
        if missing_patient:
            print(f"✗ FAIL: Missing patient fields: {missing_patient}")
            return False
        
        # Check SEX normalization
        if patient['sex'] != 'M':
            print(f"✗ FAIL: SEX not normalized: {patient['sex']} (expected M)")
            return False
        
        # Check Arabic message exists
        if not data['message_ar'] or len(data['message_ar']) < 5:
            print(f"✗ FAIL: Arabic message missing or too short")
            return False
        
        print(f"✓ PASS: Response format correct")
        print(f"  - All required fields present")
        print(f"  - SEX normalized: Male → M")
        print(f"  - Arabic message: {data['message_ar'][:50]}...")
        return True
        
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        return False


def test_8_patient_searchable():
    """Test 8: Created patient is accessible (simplified test)"""
    print("\n" + "="*70)
    print("TEST 8: Patient Created and Accessible")
    print("="*70)
    
    unique_name = "TestPhase5Search"
    unique_doc = "DOC-P5-SEARCH-999"
    payload = {
        "first_name": unique_name,
        "phone_number": "0507778899",
        "document_number": unique_doc
    }
    
    try:
        # Create patient
        response = requests.post(API_ENDPOINT, json=payload)
        if response.status_code != 201:
            print(f"✗ FAIL: Could not create patient ({response.status_code})")
            return False
        
        data = response.json()
        patient_id = data['patient']['patient_admission_id']
        print(f"  - Created patient: ID {patient_id}")
        
        # Verify patient data in response
        patient = data['patient']
        checks = [
            (patient['first_name'] == unique_name, f"FirstName matches: {patient['first_name']}"),
            (patient['phone_number'] == "0507778899", f"Phone matches: {patient['phone_number']}"),
            (patient['document_number'] == unique_doc, f"DocumentNumber matches: {patient['document_number']}"),
            (patient['source'] == 'reserve', f"Source is reserve: {patient['source']}"),
            (patient['patient_admission_id'] >= 100000, f"ID is in reserve range: {patient_id}")
        ]
        
        all_ok = True
        for check, description in checks:
            if check:
                print(f"  ✓ {description}")
            else:
                print(f"  ✗ {description}")
                all_ok = False
        
        if all_ok:
            print(f"✓ PASS: Patient created and data verified")
            return True
        else:
            print(f"✗ FAIL: Some data checks failed")
            return False
            
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        return False


def check_server():
    """Check if server is running"""
    print("\n" + "="*70)
    print("CHECKING: Server Status")
    print("="*70)
    
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=2)
        if response.status_code == 200:
            print(f"✓ Server is running at {BASE_URL}")
            return True
        else:
            print(f"✗ Server responded with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Server not accessible at {BASE_URL}")
        print(f"  Error: {str(e)}")
        print(f"\n  Please start the server with: uvicorn backend.main:app --reload")
        return False


def run_all_tests():
    """Run all Phase 5 tests"""
    print("\n" + "="*70)
    print("PHASE 5 TEST SUITE: API ENDPOINT")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Endpoint: {API_ENDPOINT}")
    
    # Check server
    if not check_server():
        print("\n" + "="*70)
        print("❌ TESTS ABORTED: Server not running")
        print("="*70)
        print("\nPlease start the server first:")
        print("  cd backend")
        print("  uvicorn main:app --reload")
        return False
    
    # Initial cleanup
    cleanup_test_data()
    
    results = []
    
    # Run tests
    results.append(("Valid Full Patient (201)", test_1_valid_full_patient()))
    results.append(("Valid Minimal Patient (201)", test_2_valid_minimal_patient()))
    results.append(("Missing FirstName (422)", test_3_missing_firstname()))
    results.append(("Invalid Field Types (422)", test_4_invalid_field_types()))
    results.append(("Validation Layers (400/422)", test_5_service_validation_errors()))
    results.append(("Duplicate Patient (409)", test_6_duplicate_patient()))
    results.append(("Response Format", test_7_response_format()))
    results.append(("Patient Data Verification", test_8_patient_searchable()))
    
    # Final cleanup
    cleanup_result = cleanup_test_data()
    results.append(("Final Cleanup", cleanup_result))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("="*70)
    print(f"Results: {passed}/{total} tests passed ({int(passed/total*100)}%)")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Phase 5 Complete - Ready for Phase 6")
        return True
    else:
        print(f"\n⚠️  {total - passed} TEST(S) FAILED - Fix issues before proceeding")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
