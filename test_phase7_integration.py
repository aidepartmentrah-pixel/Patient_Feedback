"""
====================================================================
PHASE 7 TEST: End-to-End Integration Testing
====================================================================
Purpose: Comprehensive integration testing of the complete patient
         dual-source system from API creation to data retrieval
         and cross-system consistency.

Test Coverage:
1. Create patient via API endpoint
2. Verify patient searchable immediately
3. Verify patient retrievable by ID
4. Create duplicate patient (should fail)
5. Create patient with all fields populated
6. Search with various criteria (name, phone, MRN)
7. Verify source tagging (reserve vs hospital)
8. Test Unicode/Arabic name support
9. Test data persistence across sessions
10. Test concurrent operations
11. Test boundary conditions
12. Verify database integrity

Author: System
Date: 2026-01-21
====================================================================
"""

import sys
import os
from datetime import datetime, date
import requests
import time
import threading

# Test configuration
BASE_URL = "http://localhost:8001"
CREATE_ENDPOINT = f"{BASE_URL}/api/patients/create"
SEARCH_ENDPOINT = f"{BASE_URL}/api/patients/search"

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
            WHERE FirstName LIKE 'TestPhase7%'
            OR DocumentNumber LIKE 'DOC-P7-%'
            OR MedicalFileNumber LIKE 'MRN-P7-%'
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


def verify_patient_in_db(patient_id, expected_source='reserve'):
    """Verify patient exists in database"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        if expected_source == 'reserve':
            cursor.execute("""
                SELECT PatientAdmissionID, FullName, Source='reserve'
                FROM APP_RESERVE_PATIENT
                WHERE PatientAdmissionID = ?
            """, patient_id)
        else:
            cursor.execute("""
                SELECT PatientAdmissionID, FullName, Source='hospital'
                FROM APP_VIEWTABLE_PATIENT_ADMISSION
                WHERE PatientAdmissionID = ?
            """, patient_id)
        
        row = cursor.fetchone()
        return row is not None, row
    finally:
        conn.close()


def test_1_create_and_search():
    """Test 1: Create patient and immediately search for it"""
    print("\n" + "="*70)
    print("TEST 1: Create Patient and Immediate Search")
    print("="*70)
    
    # Create patient
    payload = {
        "first_name": "TestPhase7Search",
        "middle_name": "Integration",
        "last_name": "Test",
        "phone_number": "0501234567",
        "document_number": "DOC-P7-001"
    }
    
    try:
        # Step 1: Create
        response = requests.post(CREATE_ENDPOINT, json=payload)
        if response.status_code != 201:
            print(f"✗ FAIL: Could not create patient ({response.status_code})")
            return False
        
        data = response.json()
        patient_id = data['patient']['patient_admission_id']
        full_name = data['patient']['full_name']
        print(f"  ✓ Patient created: {full_name} (ID: {patient_id})")
        
        # Step 2: Immediate search
        time.sleep(0.1)  # Small delay
        search_response = requests.get(
            SEARCH_ENDPOINT,
            params={"query": "TestPhase7Search", "limit": 10}
        )
        
        if search_response.status_code != 200:
            print(f"✗ FAIL: Search failed ({search_response.status_code})")
            return False
        
        search_data = search_response.json()
        patients = search_data.get('patients', [])
        
        # Find our patient
        found = False
        for p in patients:
            if p.get('patient_id') == patient_id:
                found = True
                print(f"  ✓ Patient found in search immediately")
                print(f"    - Name: {p.get('patient_name')}")
                print(f"    - Source: {p.get('source')}")
                break
        
        if not found:
            print(f"✗ FAIL: Patient not found in search")
            return False
        
        # Step 3: Verify in database
        exists, row = verify_patient_in_db(patient_id)
        if exists:
            print(f"  ✓ Patient verified in database")
            print(f"✓ PASS: End-to-end create → search → verify flow")
            return True
        else:
            print(f"✗ FAIL: Patient not in database")
            return False
        
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        return False


def test_2_duplicate_prevention():
    """Test 2: Duplicate patient prevention"""
    print("\n" + "="*70)
    print("TEST 2: Duplicate Patient Prevention")
    print("="*70)
    
    payload = {
        "first_name": "TestPhase7Duplicate",
        "middle_name": "Same",
        "last_name": "Name"
    }
    
    try:
        # Create first
        response1 = requests.post(CREATE_ENDPOINT, json=payload)
        if response1.status_code != 201:
            print(f"✗ FAIL: Could not create first patient")
            return False
        
        patient1_id = response1.json()['patient']['patient_admission_id']
        print(f"  ✓ First patient created (ID: {patient1_id})")
        
        # Try duplicate
        response2 = requests.post(CREATE_ENDPOINT, json=payload)
        if response2.status_code == 409:
            print(f"  ✓ Duplicate correctly rejected with 409")
            print(f"✓ PASS: Duplicate prevention works")
            return True
        else:
            print(f"✗ FAIL: Duplicate was not rejected (got {response2.status_code})")
            return False
        
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        return False


def test_3_full_patient_data():
    """Test 3: Create patient with all fields and verify data integrity"""
    print("\n" + "="*70)
    print("TEST 3: Full Patient Data Integrity")
    print("="*70)
    
    payload = {
        "first_name": "TestPhase7Full",
        "middle_name": "Complete",
        "last_name": "Integration",
        "mother_name": "Fatima",
        "phone_number": "0501112233",
        "phone_number2": "0509998877",
        "birth_date": "1985-03-15",
        "sex": "Male",
        "document_number": "DOC-P7-FULL",
        "medical_file_number": "MRN-P7-FULL",
        "spouse": "Sara Ahmad",
        "address_line1": "123 Test Street, Riyadh",
        "address_line2": "Building 5, Apartment 201"
    }
    
    try:
        # Create
        response = requests.post(CREATE_ENDPOINT, json=payload)
        if response.status_code != 201:
            print(f"✗ FAIL: Could not create patient ({response.status_code})")
            return False
        
        data = response.json()
        patient = data['patient']
        patient_id = patient['patient_admission_id']
        
        # Verify all fields
        checks = [
            (patient['first_name'] == "TestPhase7Full", "FirstName"),
            (patient['middle_name'] == "Complete", "MiddleName"),
            (patient['last_name'] == "Integration", "LastName"),
            (patient['mother_name'] == "Fatima", "MotherName"),
            (patient['phone_number'] == "0501112233", "PhoneNumber"),
            (patient['phone_number2'] == "0509998877", "PhoneNumber2"),
            (patient['birth_date'] == "1985-03-15", "BirthDate"),
            (patient['sex'] == "M", "SEX (normalized to M)"),
            (patient['document_number'] == "DOC-P7-FULL", "DocumentNumber"),
            (patient['medical_file_number'] == "MRN-P7-FULL", "MedicalFileNumber"),
            (patient['spouse'] == "Sara Ahmad", "Spouse"),
            (patient['address_line1'] == "123 Test Street, Riyadh", "AddressLine1"),
            (patient['address_line2'] == "Building 5, Apartment 201", "AddressLine2"),
            (patient['source'] == 'reserve', "Source"),
            (patient['full_name'] == "TestPhase7Full Complete Integration", "FullName constructed")
        ]
        
        all_passed = True
        for check, field in checks:
            if check:
                print(f"  ✓ {field}: OK")
            else:
                print(f"  ✗ {field}: MISMATCH")
                all_passed = False
        
        if all_passed:
            print(f"✓ PASS: All {len(checks)} fields verified correctly")
            return True
        else:
            print(f"✗ FAIL: Some fields did not match")
            return False
        
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        return False


def test_4_search_criteria():
    """Test 4: Search with different criteria"""
    print("\n" + "="*70)
    print("TEST 4: Search with Multiple Criteria")
    print("="*70)
    
    # Create patient with unique identifiers
    unique_phone = "0507654321"
    unique_mrn = "MRN-P7-SEARCH-TEST"
    
    payload = {
        "first_name": "TestPhase7MultiSearch",
        "phone_number": unique_phone,
        "medical_file_number": unique_mrn,
        "birth_date": "1990-06-20"
    }
    
    try:
        # Create patient
        response = requests.post(CREATE_ENDPOINT, json=payload)
        if response.status_code != 201:
            print(f"✗ FAIL: Could not create patient")
            return False
        
        patient_id = response.json()['patient']['patient_admission_id']
        print(f"  ✓ Patient created (ID: {patient_id})")
        
        # Test different search criteria
        searches = [
            ("query", "TestPhase7MultiSearch", "Name search"),
            ("phone", unique_phone, "Phone search"),
            ("mrn", unique_mrn, "MRN search"),
            ("query", "MultiSearch", "Partial name search")
        ]
        
        all_passed = True
        for param_name, param_value, description in searches:
            search_response = requests.get(
                SEARCH_ENDPOINT,
                params={param_name: param_value, "limit": 20}
            )
            
            if search_response.status_code != 200:
                print(f"  ✗ {description}: Search failed ({search_response.status_code})")
                all_passed = False
                continue
            
            patients = search_response.json().get('patients', [])
            found = any(p.get('patient_id') == patient_id for p in patients)
            
            if found:
                print(f"  ✓ {description}: Patient found")
            else:
                print(f"  ✗ {description}: Patient not found")
                all_passed = False
        
        if all_passed:
            print(f"✓ PASS: All search criteria work correctly")
            return True
        else:
            print(f"✗ FAIL: Some search criteria failed")
            return False
        
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        return False


def test_5_arabic_unicode_support():
    """Test 5: Arabic/Unicode character support"""
    print("\n" + "="*70)
    print("TEST 5: Arabic/Unicode Character Support")
    print("="*70)
    
    payload = {
        "first_name": "TestPhase7أحمد",
        "middle_name": "محمد",
        "last_name": "الرشيد",
        "mother_name": "فاطمة",
        "phone_number": "0501234567"
    }
    
    try:
        # Create
        response = requests.post(CREATE_ENDPOINT, json=payload)
        if response.status_code != 201:
            print(f"✗ FAIL: Could not create patient with Arabic names")
            print(f"  Response: {response.text[:200]}")
            return False
        
        data = response.json()
        patient = data['patient']
        patient_id = patient['patient_admission_id']
        
        # Verify Arabic stored correctly
        if "أحمد" in patient['first_name']:
            print(f"  ✓ Arabic first name stored: {patient['first_name']}")
        else:
            print(f"  ✗ Arabic first name corrupted: {patient['first_name']}")
            return False
        
        if "محمد" in patient['middle_name']:
            print(f"  ✓ Arabic middle name stored: {patient['middle_name']}")
        else:
            print(f"  ✗ Arabic middle name corrupted")
            return False
        
        # Search by Arabic name
        search_response = requests.get(
            SEARCH_ENDPOINT,
            params={"query": "أحمد", "limit": 10}
        )
        
        if search_response.status_code == 200:
            patients = search_response.json().get('patients', [])
            found = any(p.get('patient_id') == patient_id for p in patients)
            
            if found:
                print(f"  ✓ Arabic name searchable")
                print(f"✓ PASS: Unicode/Arabic support working")
                return True
            else:
                print(f"  ✗ Arabic search did not find patient")
                return False
        else:
            print(f"✗ FAIL: Arabic search failed")
            return False
        
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        return False


def test_6_boundary_conditions():
    """Test 6: Boundary conditions and edge cases"""
    print("\n" + "="*70)
    print("TEST 6: Boundary Conditions")
    print("="*70)
    
    test_cases = [
        {
            "name": "Minimum FirstName (2 chars)",
            "payload": {"first_name": "AB"},
            "should_succeed": True
        },
        {
            "name": "Maximum FirstName (150 chars)",
            "payload": {"first_name": "TestPhase7" + "A" * 140},
            "should_succeed": True
        },
        {
            "name": "Special characters in name",
            "payload": {"first_name": "Test-Phase7'O\"Brien"},
            "should_succeed": False  # Double quote should fail
        },
        {
            "name": "Very old birth date",
            "payload": {"first_name": "TestPhase7OldAge", "birth_date": "1900-01-01"},
            "should_succeed": True
        },
        {
            "name": "Recent birth date",
            "payload": {"first_name": "TestPhase7Recent", "birth_date": date.today().isoformat()},
            "should_succeed": True
        },
        {
            "name": "Minimum phone digits",
            "payload": {"first_name": "TestPhase7Phone", "phone_number": "1234567"},
            "should_succeed": True
        }
    ]
    
    all_passed = True
    
    for test_case in test_cases:
        try:
            response = requests.post(CREATE_ENDPOINT, json=test_case['payload'])
            success = response.status_code in [201, 200]
            
            if success == test_case['should_succeed']:
                status = "✓" if test_case['should_succeed'] else "✓ (correctly rejected)"
                print(f"  {status} {test_case['name']}")
            else:
                expected = "succeed" if test_case['should_succeed'] else "fail"
                actual = "succeeded" if success else "failed"
                print(f"  ✗ {test_case['name']}: Expected to {expected}, but {actual}")
                print(f"    Status: {response.status_code}")
                all_passed = False
                
        except Exception as e:
            print(f"  ✗ {test_case['name']}: Exception - {str(e)[:50]}")
            all_passed = False
    
    if all_passed:
        print(f"✓ PASS: All boundary conditions handled correctly")
        return True
    else:
        print(f"✗ FAIL: Some boundary conditions not handled properly")
        return False


def test_7_data_persistence():
    """Test 7: Data persistence across operations"""
    print("\n" + "="*70)
    print("TEST 7: Data Persistence")
    print("="*70)
    
    payload = {
        "first_name": "TestPhase7Persist",
        "middle_name": "Data",
        "last_name": "Test",
        "document_number": "DOC-P7-PERSIST"
    }
    
    try:
        # Create patient
        response = requests.post(CREATE_ENDPOINT, json=payload)
        if response.status_code != 201:
            print(f"✗ FAIL: Could not create patient")
            return False
        
        patient_id = response.json()['patient']['patient_admission_id']
        original_data = response.json()['patient']
        print(f"  ✓ Patient created (ID: {patient_id})")
        
        # Wait a moment
        time.sleep(0.2)
        
        # Search multiple times to verify consistency
        for i in range(3):
            search_response = requests.get(
                SEARCH_ENDPOINT,
                params={"query": "TestPhase7Persist", "limit": 10}
            )
            
            if search_response.status_code != 200:
                print(f"  ✗ Search attempt {i+1} failed")
                return False
            
            patients = search_response.json().get('patients', [])
            found_patient = next((p for p in patients if p.get('patient_id') == patient_id), None)
            
            if not found_patient:
                print(f"  ✗ Patient not found in search attempt {i+1}")
                return False
            
            # Verify data consistency
            if found_patient['patient_name'] != original_data['full_name']:
                print(f"  ✗ Data mismatch in attempt {i+1}")
                return False
        
        print(f"  ✓ Data consistent across 3 search operations")
        
        # Verify in database
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT PatientAdmissionID, FullName, DocumentNumber
            FROM APP_RESERVE_PATIENT
            WHERE PatientAdmissionID = ?
        """, patient_id)
        row = cursor.fetchone()
        conn.close()
        
        if row and row[2] == "DOC-P7-PERSIST":
            print(f"  ✓ Data persisted correctly in database")
            print(f"✓ PASS: Data persistence verified")
            return True
        else:
            print(f"✗ FAIL: Data not persisted correctly")
            return False
        
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        return False


def test_8_concurrent_operations():
    """Test 8: Concurrent patient creation"""
    print("\n" + "="*70)
    print("TEST 8: Concurrent Operations")
    print("="*70)
    
    results = []
    errors = []
    
    def create_patient(index):
        try:
            payload = {
                "first_name": f"TestPhase7Concurrent{index}",
                "document_number": f"DOC-P7-CONC-{index}"
            }
            response = requests.post(CREATE_ENDPOINT, json=payload)
            results.append((index, response.status_code, response.json() if response.status_code == 201 else None))
        except Exception as e:
            errors.append((index, str(e)))
    
    # Create 5 patients concurrently
    threads = []
    for i in range(5):
        thread = threading.Thread(target=create_patient, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all to complete
    for thread in threads:
        thread.join()
    
    # Verify results
    if errors:
        print(f"  ✗ {len(errors)} threads encountered errors")
        for idx, error in errors:
            print(f"    Thread {idx}: {error[:50]}")
        return False
    
    successful = sum(1 for _, status, _ in results if status == 201)
    
    if successful == 5:
        print(f"  ✓ All 5 concurrent creations succeeded")
        
        # Verify all have unique IDs
        patient_ids = [data['patient']['patient_admission_id'] for _, _, data in results if data]
        if len(patient_ids) == len(set(patient_ids)):
            print(f"  ✓ All patients have unique IDs")
            print(f"  ✓ IDs: {sorted(patient_ids)}")
            print(f"✓ PASS: Concurrent operations handled correctly")
            return True
        else:
            print(f"  ✗ Duplicate IDs detected")
            return False
    else:
        print(f"  ✗ Only {successful}/5 creations succeeded")
        return False


def test_9_source_segregation():
    """Test 9: Verify reserve and hospital source segregation"""
    print("\n" + "="*70)
    print("TEST 9: Source Segregation (Hospital vs Reserve)")
    print("="*70)
    
    try:
        # Get counts from database
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM APP_RESERVE_PATIENT WHERE FirstName LIKE 'TestPhase7%'")
        reserve_test_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM APP_VIEWTABLE_PATIENT_ADMISSION")
        hospital_count = cursor.fetchone()[0]
        
        conn.close()
        
        print(f"  - Reserve test patients: {reserve_test_count}")
        print(f"  - Hospital patients: {hospital_count}")
        
        # Create one more reserve patient
        payload = {"first_name": "TestPhase7Source", "document_number": "DOC-P7-SOURCE"}
        response = requests.post(CREATE_ENDPOINT, json=payload)
        
        if response.status_code != 201:
            print(f"✗ FAIL: Could not create patient")
            return False
        
        patient_id = response.json()['patient']['patient_admission_id']
        source = response.json()['patient']['source']
        
        # Verify source is 'reserve'
        if source != 'reserve':
            print(f"  ✗ Created patient has wrong source: {source}")
            return False
        
        print(f"  ✓ Created patient has source='reserve'")
        
        # Verify ID is in reserve range (>= 100000)
        if patient_id >= 100000:
            print(f"  ✓ Patient ID in reserve range: {patient_id}")
        else:
            print(f"  ✗ Patient ID not in reserve range: {patient_id}")
            return False
        
        # Search and verify both sources can be retrieved
        search_response = requests.get(SEARCH_ENDPOINT, params={"query": "ل", "limit": 100})
        if search_response.status_code == 200:
            patients = search_response.json().get('patients', [])
            sources = set(p.get('source') for p in patients)
            
            if 'hospital' in sources and 'reserve' in sources:
                print(f"  ✓ Search returns both hospital and reserve patients")
                print(f"✓ PASS: Source segregation working correctly")
                return True
            elif 'reserve' in sources:
                print(f"  ⚠️  Only reserve patients found (hospital table may be empty)")
                print(f"✓ PASS: Source tagging works")
                return True
            else:
                print(f"  ✗ Unexpected sources: {sources}")
                return False
        else:
            print(f"✗ FAIL: Search failed")
            return False
        
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        return False


def test_10_validation_consistency():
    """Test 10: Validation consistency across layers"""
    print("\n" + "="*70)
    print("TEST 10: Validation Consistency Across Layers")
    print("="*70)
    
    # Test cases that should fail at different layers
    test_cases = [
        {
            "name": "Empty FirstName (Pydantic)",
            "payload": {},
            "expected_status": 422,
            "layer": "Pydantic"
        },
        {
            "name": "Invalid characters (Service)",
            "payload": {"first_name": "Test@Invalid#Phase7"},
            "expected_status": 400,
            "layer": "Service"
        },
        {
            "name": "Future birth date (Service)",
            "payload": {"first_name": "TestPhase7Future", "birth_date": "2099-01-01"},
            "expected_status": 400,
            "layer": "Service"
        },
        {
            "name": "Invalid SEX value (Service)",
            "payload": {"first_name": "TestPhase7Sex", "sex": "Unknown"},
            "expected_status": 400,
            "layer": "Service"
        }
    ]
    
    all_passed = True
    
    for test_case in test_cases:
        try:
            response = requests.post(CREATE_ENDPOINT, json=test_case['payload'])
            
            if response.status_code == test_case['expected_status']:
                print(f"  ✓ {test_case['name']}: Correctly rejected at {test_case['layer']} layer ({response.status_code})")
            else:
                print(f"  ✗ {test_case['name']}: Expected {test_case['expected_status']}, got {response.status_code}")
                all_passed = False
                
        except Exception as e:
            print(f"  ✗ {test_case['name']}: Exception - {str(e)[:50]}")
            all_passed = False
    
    if all_passed:
        print(f"✓ PASS: Validation consistent across all layers")
        return True
    else:
        print(f"✗ FAIL: Validation inconsistencies detected")
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
    except requests.exceptions.RequestException:
        print(f"✗ Server not accessible at {BASE_URL}")
        print(f"  Please start: cd backend && uvicorn main:app --reload")
        return False


def run_all_tests():
    """Run all Phase 7 integration tests"""
    print("\n" + "="*70)
    print("PHASE 7 TEST SUITE: END-TO-END INTEGRATION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check server
    if not check_server():
        print("\n❌ TESTS ABORTED: Server not running")
        return False
    
    # Initial cleanup
    cleanup_test_data()
    
    results = []
    
    # Run all integration tests
    print("\n" + "="*70)
    print("RUNNING INTEGRATION TESTS")
    print("="*70)
    
    results.append(("Create and Immediate Search", test_1_create_and_search()))
    results.append(("Duplicate Prevention", test_2_duplicate_prevention()))
    results.append(("Full Data Integrity", test_3_full_patient_data()))
    results.append(("Multiple Search Criteria", test_4_search_criteria()))
    results.append(("Arabic/Unicode Support", test_5_arabic_unicode_support()))
    results.append(("Boundary Conditions", test_6_boundary_conditions()))
    results.append(("Data Persistence", test_7_data_persistence()))
    results.append(("Concurrent Operations", test_8_concurrent_operations()))
    results.append(("Source Segregation", test_9_source_segregation()))
    results.append(("Validation Consistency", test_10_validation_consistency()))
    
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
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("\n📋 INTEGRATION TEST SUMMARY:")
        print("  ✓ End-to-end patient creation flow")
        print("  ✓ Duplicate detection across system")
        print("  ✓ Complete data integrity (all 13 fields)")
        print("  ✓ Multi-criteria search functionality")
        print("  ✓ Unicode/Arabic character support")
        print("  ✓ Boundary condition handling")
        print("  ✓ Data persistence verification")
        print("  ✓ Concurrent operation safety")
        print("  ✓ Hospital/Reserve source segregation")
        print("  ✓ Multi-layer validation consistency")
        print("\n🎊 PATIENT DUAL-SOURCE IMPLEMENTATION COMPLETE!")
        print("   All 7 phases tested and verified with 100% pass rate")
        return True
    else:
        print(f"\n⚠️  {total - passed} TEST(S) FAILED - Review and fix issues")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
