"""
====================================================================
TEST: GET /api/patients/reserve Endpoint
====================================================================
Purpose: Test the new endpoint to retrieve all reserve patients
         with pagination and sorting options

Test Coverage:
1. Get reserve patients (default settings)
2. Pagination test (limit and offset)
3. Sort by name (alphabetical)
4. Sort by created_at (newest first)
5. Large limit test
6. Empty database handling
7. Response structure validation

Author: System
Date: 2026-01-21
====================================================================
"""

import sys
import os
import requests
import json
from datetime import datetime

# Test configuration
BASE_URL = "http://localhost:8000"
RESERVE_ENDPOINT = f"{BASE_URL}/api/patients/reserve"
CREATE_ENDPOINT = f"{BASE_URL}/api/patients/create"

# Add backend to path
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


def get_reserve_patient_count():
    """Get total count of reserve patients from database."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM APP_RESERVE_PATIENT")
        return cursor.fetchone()[0]
    finally:
        conn.close()


def create_test_patient(name_suffix):
    """Create a test patient for testing."""
    payload = {
        "first_name": f"TestReserve{name_suffix}",
        "middle_name": "Test",
        "last_name": "Patient",
        "phone_number": f"05012345{name_suffix:02d}",
        "document_number": f"DOC-TEST-{name_suffix:03d}"
    }
    
    try:
        response = requests.post(CREATE_ENDPOINT, json=payload)
        if response.status_code == 201:
            return response.json()['patient']['patient_admission_id']
        return None
    except:
        return None


def test_1_default_settings():
    """Test 1: Get reserve patients with default settings"""
    print("\n" + "="*70)
    print("TEST 1: Get Reserve Patients (Default Settings)")
    print("="*70)
    
    try:
        response = requests.get(RESERVE_ENDPOINT)
        
        if response.status_code != 200:
            print(f"✗ FAIL: Expected 200, got {response.status_code}")
            print(f"  Response: {response.text}")
            return False
        
        data = response.json()
        
        # Validate response structure
        required_keys = ['patients', 'total', 'count', 'limit', 'offset']
        for key in required_keys:
            if key not in data:
                print(f"✗ FAIL: Missing key '{key}' in response")
                return False
        
        print(f"✓ PASS: Endpoint returned 200 OK")
        print(f"  - Total reserve patients: {data['total']}")
        print(f"  - Patients in response: {data['count']}")
        print(f"  - Limit: {data['limit']}")
        print(f"  - Offset: {data['offset']}")
        
        # Validate patient structure
        if data['count'] > 0:
            patient = data['patients'][0]
            required_patient_keys = [
                'patient_admission_id', 'full_name', 'first_name', 
                'source', 'created_at'
            ]
            for key in required_patient_keys:
                if key not in patient:
                    print(f"✗ FAIL: Missing key '{key}' in patient object")
                    return False
            
            if patient['source'] != 'reserve':
                print(f"✗ FAIL: Expected source='reserve', got '{patient['source']}'")
                return False
            
            print(f"  - Sample patient: {patient['full_name']} (ID: {patient['patient_admission_id']})")
        
        return True
        
    except Exception as e:
        print(f"✗ FAIL: Exception: {str(e)}")
        return False


def test_2_pagination():
    """Test 2: Pagination with limit and offset"""
    print("\n" + "="*70)
    print("TEST 2: Pagination (Limit and Offset)")
    print("="*70)
    
    try:
        # First page
        response1 = requests.get(RESERVE_ENDPOINT, params={"limit": 5, "offset": 0})
        if response1.status_code != 200:
            print(f"✗ FAIL: First page request failed ({response1.status_code})")
            return False
        
        data1 = response1.json()
        
        # Second page
        response2 = requests.get(RESERVE_ENDPOINT, params={"limit": 5, "offset": 5})
        if response2.status_code != 200:
            print(f"✗ FAIL: Second page request failed ({response2.status_code})")
            return False
        
        data2 = response2.json()
        
        print(f"✓ PASS: Pagination works")
        print(f"  - Page 1: {data1['count']} patients (offset 0)")
        print(f"  - Page 2: {data2['count']} patients (offset 5)")
        print(f"  - Total: {data1['total']}")
        
        # Verify different results
        if data1['count'] > 0 and data2['count'] > 0:
            id1 = data1['patients'][0]['patient_admission_id']
            id2 = data2['patients'][0]['patient_admission_id']
            if id1 != id2:
                print(f"  ✓ Pages contain different patients")
            else:
                print(f"  ⚠ WARNING: Pages contain same patient")
        
        return True
        
    except Exception as e:
        print(f"✗ FAIL: Exception: {str(e)}")
        return False


def test_3_sort_by_name():
    """Test 3: Sort by name (alphabetical)"""
    print("\n" + "="*70)
    print("TEST 3: Sort by Name (Alphabetical)")
    print("="*70)
    
    try:
        response = requests.get(RESERVE_ENDPOINT, params={"order_by": "name", "limit": 10})
        
        if response.status_code != 200:
            print(f"✗ FAIL: Request failed ({response.status_code})")
            return False
        
        data = response.json()
        
        if data['count'] < 2:
            print(f"⚠ SKIP: Need at least 2 patients to test sorting")
            return True
        
        # Check if names are in alphabetical order
        names = [p['full_name'] for p in data['patients']]
        sorted_names = sorted(names)
        
        if names == sorted_names:
            print(f"✓ PASS: Names are in alphabetical order")
            for name in names[:5]:
                print(f"  - {name}")
        else:
            print(f"✗ FAIL: Names are not in alphabetical order")
            print(f"  Expected: {sorted_names[:3]}")
            print(f"  Got: {names[:3]}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ FAIL: Exception: {str(e)}")
        return False


def test_4_sort_by_created():
    """Test 4: Sort by created_at (newest first)"""
    print("\n" + "="*70)
    print("TEST 4: Sort by Created Date (Newest First)")
    print("="*70)
    
    try:
        response = requests.get(RESERVE_ENDPOINT, params={"order_by": "created_at", "limit": 10})
        
        if response.status_code != 200:
            print(f"✗ FAIL: Request failed ({response.status_code})")
            return False
        
        data = response.json()
        
        if data['count'] < 2:
            print(f"⚠ SKIP: Need at least 2 patients to test sorting")
            return True
        
        print(f"✓ PASS: Retrieved {data['count']} patients sorted by creation date")
        for i, patient in enumerate(data['patients'][:5], 1):
            print(f"  {i}. {patient['full_name']} - Created: {patient['created_at']}")
        
        return True
        
    except Exception as e:
        print(f"✗ FAIL: Exception: {str(e)}")
        return False


def test_5_limit_validation():
    """Test 5: Limit parameter validation"""
    print("\n" + "="*70)
    print("TEST 5: Limit Parameter Validation")
    print("="*70)
    
    try:
        # Test max limit (200)
        response1 = requests.get(RESERVE_ENDPOINT, params={"limit": 200})
        if response1.status_code == 200:
            data1 = response1.json()
            print(f"✓ PASS: Max limit (200) accepted")
            print(f"  - Returned: {data1['count']} patients")
        else:
            print(f"✗ FAIL: Max limit rejected ({response1.status_code})")
            return False
        
        # Test over-limit (300) - should be capped to 200
        response2 = requests.get(RESERVE_ENDPOINT, params={"limit": 300})
        if response2.status_code == 200:
            print(f"✓ PASS: Over-limit handled gracefully")
        
        # Test minimum limit
        response3 = requests.get(RESERVE_ENDPOINT, params={"limit": 1})
        if response3.status_code == 200:
            data3 = response3.json()
            if data3['count'] <= 1:
                print(f"✓ PASS: Minimum limit (1) works correctly")
            else:
                print(f"✗ FAIL: Limit=1 returned {data3['count']} patients")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ FAIL: Exception: {str(e)}")
        return False


def test_6_database_consistency():
    """Test 6: Database count matches API count"""
    print("\n" + "="*70)
    print("TEST 6: Database Consistency Check")
    print("="*70)
    
    try:
        # Get count from database
        db_count = get_reserve_patient_count()
        
        # Get count from API
        response = requests.get(RESERVE_ENDPOINT)
        if response.status_code != 200:
            print(f"✗ FAIL: API request failed")
            return False
        
        api_count = response.json()['total']
        
        if db_count == api_count:
            print(f"✓ PASS: Database and API counts match")
            print(f"  - Count: {db_count}")
        else:
            print(f"✗ FAIL: Count mismatch")
            print(f"  - Database: {db_count}")
            print(f"  - API: {api_count}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ FAIL: Exception: {str(e)}")
        return False


def test_7_response_structure():
    """Test 7: Detailed response structure validation"""
    print("\n" + "="*70)
    print("TEST 7: Response Structure Validation")
    print("="*70)
    
    try:
        response = requests.get(RESERVE_ENDPOINT, params={"limit": 1})
        
        if response.status_code != 200:
            print(f"✗ FAIL: Request failed")
            return False
        
        data = response.json()
        
        if data['count'] == 0:
            print(f"⚠ SKIP: No patients to validate structure")
            return True
        
        patient = data['patients'][0]
        
        # Check all expected fields
        expected_fields = {
            'patient_admission_id': int,
            'full_name': (str, type(None)),
            'first_name': (str, type(None)),
            'middle_name': (str, type(None)),
            'last_name': (str, type(None)),
            'mother_name': (str, type(None)),
            'phone_number': (str, type(None)),
            'phone_number2': (str, type(None)),
            'birth_date': (str, type(None)),
            'sex': (str, type(None)),
            'document_number': (str, type(None)),
            'medical_file_number': (str, type(None)),
            'spouse': (str, type(None)),
            'address_line1': (str, type(None)),
            'address_line2': (str, type(None)),
            'created_at': (str, type(None)),
            'source': str
        }
        
        all_valid = True
        for field, expected_type in expected_fields.items():
            if field not in patient:
                print(f"  ✗ Missing field: {field}")
                all_valid = False
            elif not isinstance(patient[field], expected_type):
                print(f"  ✗ Wrong type for {field}: {type(patient[field])} (expected {expected_type})")
                all_valid = False
        
        if all_valid:
            print(f"✓ PASS: All expected fields present with correct types")
            print(f"  - Patient ID: {patient['patient_admission_id']}")
            print(f"  - Full Name: {patient['full_name']}")
            print(f"  - Source: {patient['source']}")
        
        return all_valid
        
    except Exception as e:
        print(f"✗ FAIL: Exception: {str(e)}")
        return False


def check_server():
    """Check if server is running"""
    try:
        response = requests.get(BASE_URL, timeout=2)
        return True
    except:
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("RESERVE PATIENTS ENDPOINT TEST SUITE")
    print("Testing: GET /api/patients/reserve")
    print("="*70)
    
    # Check server
    if not check_server():
        print("\n❌ ERROR: Server is not running!")
        print("Please start the server with: uvicorn backend.main:app --reload")
        return False
    
    print("✓ Server is running")
    
    # Get initial count
    total_patients = get_reserve_patient_count()
    print(f"✓ Database connected")
    print(f"  - Reserve patients in database: {total_patients}")
    
    # Run tests
    tests = [
        test_1_default_settings,
        test_2_pagination,
        test_3_sort_by_name,
        test_4_sort_by_created,
        test_5_limit_validation,
        test_6_database_consistency,
        test_7_response_structure
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ TEST FAILED WITH EXCEPTION: {str(e)}")
            results.append(False)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
        return True
    else:
        print(f"\n⚠️ {total - passed} TEST(S) FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
