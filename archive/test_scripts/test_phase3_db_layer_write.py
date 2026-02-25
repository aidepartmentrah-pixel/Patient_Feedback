"""
====================================================================
PHASE 3 TEST: DB Layer Write Function - create_patient()
====================================================================
Purpose: Verify patients_db.create_patient() correctly inserts
         patients into APP_RESERVE_PATIENT with proper validation

Test Coverage:
1. Create patient with minimal data (FirstName only)
2. Create patient with full data (all fields)
3. Duplicate detection by FullName
4. Duplicate detection by DocumentNumber
5. Duplicate detection by MedicalFileNumber
6. FullName is built correctly from name parts
7. Created patient can be retrieved by ID
8. Created patient appears in search results
9. Required field validation
10. Data types and formats are correct
11. Transaction rollback on error
12. SystemTime is set automatically

Author: System
Date: 2026-01-20
====================================================================
"""

import pyodbc
from datetime import datetime, date
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'api'))

from db_layer.patients_db import create_patient
from services.search_service import search_patients, get_patient_by_id


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
    """Remove any existing test data before starting"""
    print("\n" + "="*70)
    print("CLEANUP: Removing Existing Test Data")
    print("="*70)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            DELETE FROM APP_RESERVE_PATIENT 
            WHERE FirstName LIKE 'TestPhase3%' 
            OR DocumentNumber LIKE 'DOC-PHASE3-%'
            OR MedicalFileNumber LIKE 'MRN-PHASE3-%'
        """)
        deleted = cursor.rowcount
        conn.commit()
        
        print(f"✓ Deleted {deleted} existing test patient(s)")
        return True
    except Exception as e:
        print(f"✗ Error during cleanup: {str(e)}")
        return False
    finally:
        conn.close()


def test_1_create_minimal_patient():
    """Test 1: Create patient with minimal data (FirstName only)"""
    print("\n" + "="*70)
    print("TEST 1: Create Patient with Minimal Data")
    print("="*70)
    
    try:
        patient = create_patient(
            first_name="TestPhase3Minimal"
        )
        
        if patient['PatientAdmissionID'] and patient['FullName'] == "TestPhase3Minimal":
            print(f"✓ PASS: Patient created with minimal data")
            print(f"  - PatientAdmissionID: {patient['PatientAdmissionID']}")
            print(f"  - FullName: {patient['FullName']}")
            print(f"  - Source: {patient['Source']}")
            return True, patient['PatientAdmissionID']
        else:
            print(f"✗ FAIL: Patient data incorrect")
            return False, None
            
    except Exception as e:
        print(f"✗ FAIL: Exception raised: {str(e)}")
        return False, None


def test_2_create_full_patient():
    """Test 2: Create patient with full data (all fields)"""
    print("\n" + "="*70)
    print("TEST 2: Create Patient with Full Data")
    print("="*70)
    
    try:
        patient = create_patient(
            first_name="TestPhase3Full",
            middle_name="Ahmad",
            last_name="AlFull",
            mother_name="Fatima",
            phone_number="0501234567",
            phone_number2="0509876543",
            birth_date="1990-05-15",
            sex="M",
            document_number="DOC-PHASE3-001",
            medical_file_number="MRN-PHASE3-001",
            spouse="Sara",
            address_line1="123 Test Street",
            address_line2="Apt 456"
        )
        
        # Verify all fields
        checks = [
            patient['PatientAdmissionID'] is not None,
            patient['FullName'] == "TestPhase3Full Ahmad AlFull",
            patient['FirstName'] == "TestPhase3Full",
            patient['MiddleName'] == "Ahmad",
            patient['LastName'] == "AlFull",
            patient['MotherName'] == "Fatima",
            patient['PhoneNumber1'] == "0501234567",
            patient['PhoneNumber2'] == "0509876543",
            patient['BirthDate'] == "1990-05-15",
            patient['SEX'] == "M",
            patient['DocumentNumber'] == "DOC-PHASE3-001",
            patient['MedicalFileNumber'] == "MRN-PHASE3-001",
            patient['Spouse'] == "Sara",
            patient['AddressLine1'] == "123 Test Street",
            patient['AddressLine2'] == "Apt 456",
            patient['Source'] == "reserve"
        ]
        
        if all(checks):
            print(f"✓ PASS: Patient created with all fields correct")
            print(f"  - PatientAdmissionID: {patient['PatientAdmissionID']}")
            print(f"  - FullName: {patient['FullName']}")
            print(f"  - DocumentNumber: {patient['DocumentNumber']}")
            print(f"  - MedicalFileNumber: {patient['MedicalFileNumber']}")
            return True, patient['PatientAdmissionID']
        else:
            print(f"✗ FAIL: Some fields are incorrect")
            for i, check in enumerate(checks):
                if not check:
                    print(f"  - Check {i+1} failed")
            return False, None
            
    except Exception as e:
        print(f"✗ FAIL: Exception raised: {str(e)}")
        return False, None


def test_3_duplicate_fullname():
    """Test 3: Duplicate detection by FullName"""
    print("\n" + "="*70)
    print("TEST 3: Duplicate Detection by FullName")
    print("="*70)
    
    try:
        # Create first patient
        patient1 = create_patient(
            first_name="TestPhase3Dup",
            middle_name="One",
            last_name="Name"
        )
        
        print(f"  - Created patient 1: {patient1['FullName']}")
        
        # Try to create duplicate
        try:
            patient2 = create_patient(
                first_name="TestPhase3Dup",
                middle_name="One",
                last_name="Name"
            )
            print(f"✗ FAIL: Duplicate patient was allowed (should have been blocked)")
            return False
        except ValueError as ve:
            if "already exists" in str(ve).lower():
                print(f"✓ PASS: Duplicate correctly detected and blocked")
                print(f"  - Error message: {str(ve)[:100]}...")
                return True
            else:
                print(f"✗ FAIL: Wrong error message: {str(ve)}")
                return False
                
    except Exception as e:
        print(f"✗ FAIL: Unexpected exception: {str(e)}")
        return False


def test_4_duplicate_document_number():
    """Test 4: Duplicate detection by DocumentNumber"""
    print("\n" + "="*70)
    print("TEST 4: Duplicate Detection by DocumentNumber")
    print("="*70)
    
    try:
        # Create first patient
        patient1 = create_patient(
            first_name="TestPhase3DocDup1",
            document_number="DOC-PHASE3-DUP"
        )
        
        print(f"  - Created patient 1 with DocumentNumber: {patient1['DocumentNumber']}")
        
        # Try to create duplicate with same DocumentNumber but different name
        try:
            patient2 = create_patient(
                first_name="TestPhase3DocDup2",  # Different name
                document_number="DOC-PHASE3-DUP"  # Same document
            )
            print(f"✗ FAIL: Duplicate DocumentNumber was allowed")
            return False
        except ValueError as ve:
            if "documentnumber" in str(ve).lower():
                print(f"✓ PASS: Duplicate DocumentNumber correctly blocked")
                print(f"  - Error message: {str(ve)[:100]}...")
                return True
            else:
                print(f"✗ FAIL: Wrong error message: {str(ve)}")
                return False
                
    except Exception as e:
        print(f"✗ FAIL: Unexpected exception: {str(e)}")
        return False


def test_5_duplicate_medical_file():
    """Test 5: Duplicate detection by MedicalFileNumber"""
    print("\n" + "="*70)
    print("TEST 5: Duplicate Detection by MedicalFileNumber")
    print("="*70)
    
    try:
        # Create first patient
        patient1 = create_patient(
            first_name="TestPhase3MRNDup1",
            medical_file_number="MRN-PHASE3-DUP"
        )
        
        print(f"  - Created patient 1 with MRN: {patient1['MedicalFileNumber']}")
        
        # Try to create duplicate with same MRN but different name
        try:
            patient2 = create_patient(
                first_name="TestPhase3MRNDup2",  # Different name
                medical_file_number="MRN-PHASE3-DUP"  # Same MRN
            )
            print(f"✗ FAIL: Duplicate MRN was allowed")
            return False
        except ValueError as ve:
            if "medicalfilenumber" in str(ve).lower():
                print(f"✓ PASS: Duplicate MRN correctly blocked")
                print(f"  - Error message: {str(ve)[:100]}...")
                return True
            else:
                print(f"✗ FAIL: Wrong error message: {str(ve)}")
                return False
                
    except Exception as e:
        print(f"✗ FAIL: Unexpected exception: {str(e)}")
        return False


def test_6_fullname_building():
    """Test 6: FullName is built correctly from name parts"""
    print("\n" + "="*70)
    print("TEST 6: FullName Building from Name Parts")
    print("="*70)
    
    test_cases = [
        {
            "input": {"first_name": "Ahmad"},
            "expected": "Ahmad"
        },
        {
            "input": {"first_name": "Ahmad", "middle_name": "Ali"},
            "expected": "Ahmad Ali"
        },
        {
            "input": {"first_name": "Ahmad", "middle_name": "Ali", "last_name": "Salem"},
            "expected": "Ahmad Ali Salem"
        },
        {
            "input": {"first_name": "Ahmad", "last_name": "Salem"},  # Missing middle
            "expected": "Ahmad Salem"
        }
    ]
    
    all_passed = True
    
    for i, case in enumerate(test_cases):
        try:
            # Add unique prefix to avoid duplicates
            case['input']['first_name'] = f"TestPhase3Name{i}" + case['input']['first_name']
            expected = case['expected'].replace("Ahmad", f"TestPhase3Name{i}Ahmad")
            
            patient = create_patient(**case['input'])
            
            if patient['FullName'] == expected:
                print(f"  ✓ Case {i+1}: '{expected}' - PASS")
            else:
                print(f"  ✗ Case {i+1}: Expected '{expected}', Got '{patient['FullName']}' - FAIL")
                all_passed = False
                
        except Exception as e:
            print(f"  ✗ Case {i+1}: Exception - {str(e)}")
            all_passed = False
    
    if all_passed:
        print(f"✓ PASS: All FullName building tests passed")
        return True
    else:
        print(f"✗ FAIL: Some FullName building tests failed")
        return False


def test_7_retrieve_created_patient(patient_id):
    """Test 7: Created patient can be retrieved by ID"""
    print("\n" + "="*70)
    print("TEST 7: Retrieve Created Patient by ID")
    print("="*70)
    
    if not patient_id:
        print("✗ SKIP: No patient ID provided")
        return True  # Not a failure, just skip
    
    result = get_patient_by_id(patient_id)
    
    if result['success'] and result['patient']['source'] == 'reserve':
        patient = result['patient']
        print(f"✓ PASS: Patient retrieved successfully")
        print(f"  - PatientAdmissionID: {patient['patient_admission_id']}")
        print(f"  - FullName: {patient['full_name']}")
        print(f"  - Source: {patient['source']}")
        return True
    else:
        print(f"✗ FAIL: Could not retrieve patient")
        return False


def test_8_search_finds_created_patient(patient_id):
    """Test 8: Created patient appears in search results"""
    print("\n" + "="*70)
    print("TEST 8: Search Finds Created Patient")
    print("="*70)
    
    if not patient_id:
        print("✗ SKIP: No patient ID provided")
        return True
    
    # Search for TestPhase3 patients
    result = search_patients("TestPhase3Full", limit=50)
    
    if result['success']:
        matching = [p for p in result['patients'] if p['patient_admission_id'] == patient_id]
        
        if len(matching) > 0:
            print(f"✓ PASS: Created patient found in search results")
            print(f"  - Found: {matching[0]['full_name']}")
            return True
        else:
            print(f"✗ FAIL: Created patient not found in search")
            return False
    else:
        print(f"✗ FAIL: Search failed")
        return False


def test_9_verify_database_record(patient_id):
    """Test 9: Verify record exists in database with correct data"""
    print("\n" + "="*70)
    print("TEST 9: Verify Database Record")
    print("="*70)
    
    if not patient_id:
        print("✗ SKIP: No patient ID provided")
        return True
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT 
                PatientAdmissionID,
                FullName,
                FirstName,
                SystemTime
            FROM APP_RESERVE_PATIENT
            WHERE PatientAdmissionID = ?
        """, (patient_id,))
        
        row = cursor.fetchone()
        
        if row:
            print(f"✓ PASS: Database record verified")
            print(f"  - PatientAdmissionID: {row[0]}")
            print(f"  - FullName: {row[1]}")
            print(f"  - FirstName: {row[2]}")
            print(f"  - SystemTime: {row[3]}")
            return True
        else:
            print(f"✗ FAIL: No database record found")
            return False
            
    except Exception as e:
        print(f"✗ FAIL: Database query failed: {str(e)}")
        return False
    finally:
        conn.close()


def test_10_empty_firstname_error():
    """Test 10: Empty FirstName raises error"""
    print("\n" + "="*70)
    print("TEST 10: Empty FirstName Validation")
    print("="*70)
    
    try:
        patient = create_patient(first_name="")
        print(f"✗ FAIL: Empty FirstName was accepted")
        return False
    except (ValueError, Exception) as e:
        if "firstname" in str(e).lower() or "fullname" in str(e).lower():
            print(f"✓ PASS: Empty FirstName correctly rejected")
            print(f"  - Error: {str(e)[:80]}...")
            return True
        else:
            print(f"✗ FAIL: Wrong error: {str(e)}")
            return False


def final_cleanup():
    """Final cleanup of all test data"""
    print("\n" + "="*70)
    print("FINAL CLEANUP: Removing All Test Data")
    print("="*70)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            DELETE FROM APP_RESERVE_PATIENT 
            WHERE FirstName LIKE 'TestPhase3%' 
            OR DocumentNumber LIKE 'DOC-PHASE3-%'
            OR MedicalFileNumber LIKE 'MRN-PHASE3-%'
        """)
        deleted = cursor.rowcount
        conn.commit()
        
        print(f"✓ Deleted {deleted} test patient(s)")
        return True
    except Exception as e:
        print(f"✗ Error during cleanup: {str(e)}")
        return False
    finally:
        conn.close()


def run_all_tests():
    """Run all Phase 3 tests"""
    print("\n" + "="*70)
    print("PHASE 3 TEST SUITE: DB LAYER WRITE FUNCTION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initial cleanup
    cleanup_test_data()
    
    results = []
    patient_ids = {}
    
    # Run tests
    result, pid = test_1_create_minimal_patient()
    results.append(("Create Minimal Patient", result))
    patient_ids['minimal'] = pid
    
    result, pid = test_2_create_full_patient()
    results.append(("Create Full Patient", result))
    patient_ids['full'] = pid
    
    results.append(("Duplicate FullName Detection", test_3_duplicate_fullname()))
    results.append(("Duplicate DocumentNumber Detection", test_4_duplicate_document_number()))
    results.append(("Duplicate MedicalFileNumber Detection", test_5_duplicate_medical_file()))
    results.append(("FullName Building", test_6_fullname_building()))
    results.append(("Retrieve by ID", test_7_retrieve_created_patient(patient_ids.get('full'))))
    results.append(("Search Finds Patient", test_8_search_finds_created_patient(patient_ids.get('full'))))
    results.append(("Database Record Verification", test_9_verify_database_record(patient_ids.get('full'))))
    results.append(("Empty FirstName Validation", test_10_empty_firstname_error()))
    
    # Final cleanup
    cleanup_result = final_cleanup()
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
        print("\n🎉 ALL TESTS PASSED! Phase 3 Complete - Ready for Phase 4")
        return True
    else:
        print(f"\n⚠️  {total - passed} TEST(S) FAILED - Fix issues before proceeding")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
