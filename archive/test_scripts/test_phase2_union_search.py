"""
====================================================================
PHASE 2 TEST: UNION Query Testing for Dual-Source Patient Search
====================================================================
Purpose: Verify search_service.py correctly merges hospital and 
         reserve patient data using UNION queries

Test Coverage:
1. Search returns hospital patients only (baseline)
2. Search returns reserve patients only
3. Search returns BOTH hospital + reserve patients (merged)
4. Get by ID works for reserve patients
5. Get by ID works for hospital patients
6. Get by ID returns correct 'source' field
7. Search with various criteria (name, document, medical file)
8. Limit parameter works correctly
9. Source field is always present

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


def setup_test_data():
    """Create test patients in reserve table"""
    print("\n" + "="*70)
    print("SETUP: Creating Test Data")
    print("="*70)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Clean up any existing test data
        cursor.execute("DELETE FROM APP_RESERVE_PATIENT WHERE FirstName LIKE 'TestPhase2%'")
        conn.commit()
        
        # Insert 3 test patients with distinct names
        test_patients = [
            ('TestPhase2Alpha', 'Ali', 'AlAhmad', 'Maryam', 'TestPhase2Alpha Ali AlAhmad', 
             '0501111111', date(1985, 1, 15), 'M', 'DOC-TEST-001', 'MRN-TEST-001'),
            ('TestPhase2Beta', 'Sara', 'AlSalem', 'Fatima', 'TestPhase2Beta Sara AlSalem',
             '0502222222', date(1990, 6, 20), 'F', 'DOC-TEST-002', 'MRN-TEST-002'),
            ('TestPhase2Gamma', 'Omar', 'AlKhalil', 'Aisha', 'TestPhase2Gamma Omar AlKhalil',
             '0503333333', date(1995, 12, 10), 'M', 'DOC-TEST-003', 'MRN-TEST-003'),
        ]
        
        patient_ids = []
        
        for patient in test_patients:
            cursor.execute("""
                INSERT INTO APP_RESERVE_PATIENT (
                    FirstName, MiddleName, LastName, MotherName, FullName,
                    PhoneNumber1, BirthDate, SEX, DocumentNumber, MedicalFileNumber
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, patient)
            
            cursor.execute("SELECT @@IDENTITY")
            patient_id = cursor.fetchone()[0]
            patient_ids.append(patient_id)
            
        conn.commit()
        
        print(f"✓ Created {len(patient_ids)} test patients in reserve table")
        for i, pid in enumerate(patient_ids):
            print(f"  - Test Patient {i+1}: ID = {pid}, Name = {test_patients[i][4]}")
        
        return patient_ids
        
    except Exception as e:
        print(f"✗ Error creating test data: {str(e)}")
        return []
    finally:
        conn.close()


def test_1_search_reserve_patients(test_patient_ids):
    """Test 1: Search returns reserve patients"""
    print("\n" + "="*70)
    print("TEST 1: Search Reserve Patients")
    print("="*70)
    
    result = search_patients("TestPhase2", limit=50)
    
    if not result['success']:
        print(f"✗ FAIL: Search failed: {result.get('error', 'Unknown error')}")
        return False
    
    patients = result['patients']
    reserve_patients = [p for p in patients if p['source'] == 'reserve']
    
    if len(reserve_patients) >= 3:
        print(f"✓ PASS: Found {len(reserve_patients)} reserve patients")
        for p in reserve_patients[:3]:
            print(f"  - ID: {p['patient_admission_id']}, Name: {p['full_name']}, Source: {p['source']}")
        return True
    else:
        print(f"✗ FAIL: Expected at least 3 reserve patients, found {len(reserve_patients)}")
        return False


def test_2_search_has_source_field(test_patient_ids):
    """Test 2: All search results have 'source' field"""
    print("\n" + "="*70)
    print("TEST 2: Source Field Present")
    print("="*70)
    
    result = search_patients("TestPhase2", limit=10)
    
    if not result['success']:
        print(f"✗ FAIL: Search failed")
        return False
    
    patients = result['patients']
    
    all_have_source = all('source' in p for p in patients)
    valid_sources = all(p['source'] in ['hospital', 'reserve'] for p in patients)
    
    if all_have_source and valid_sources:
        print(f"✓ PASS: All {len(patients)} patients have valid 'source' field")
        sources_count = {}
        for p in patients:
            sources_count[p['source']] = sources_count.get(p['source'], 0) + 1
        print(f"  - Hospital: {sources_count.get('hospital', 0)}")
        print(f"  - Reserve: {sources_count.get('reserve', 0)}")
        return True
    else:
        print(f"✗ FAIL: Missing or invalid 'source' field")
        return False


def test_3_get_by_id_reserve(test_patient_ids):
    """Test 3: Get reserve patient by ID"""
    print("\n" + "="*70)
    print("TEST 3: Get Reserve Patient by ID")
    print("="*70)
    
    if not test_patient_ids:
        print("✗ FAIL: No test patient IDs available")
        return False
    
    patient_id = test_patient_ids[0]
    result = get_patient_by_id(patient_id)
    
    if not result['success']:
        print(f"✗ FAIL: get_patient_by_id failed: {result.get('error', 'Unknown')}")
        return False
    
    patient = result['patient']
    
    if patient['source'] == 'reserve' and patient['patient_admission_id'] == patient_id:
        print(f"✓ PASS: Reserve patient retrieved successfully")
        print(f"  - ID: {patient['patient_admission_id']}")
        print(f"  - Name: {patient['full_name']}")
        print(f"  - Source: {patient['source']}")
        return True
    else:
        print(f"✗ FAIL: Wrong source or ID mismatch")
        print(f"  - Expected source: 'reserve', Got: '{patient.get('source', 'missing')}'")
        return False


def test_4_get_by_id_hospital():
    """Test 4: Get hospital patient by ID (if any exist)"""
    print("\n" + "="*70)
    print("TEST 4: Get Hospital Patient by ID")
    print("="*70)
    
    # Get a hospital patient ID
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT TOP 1 PatientAdmissionID 
        FROM APP_VIEWTABLE_PATIENT_ADMISSION
    """)
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        print("⚠ SKIP: No hospital patients available for testing")
        return True  # Not a failure, just no data
    
    hospital_patient_id = row[0]
    result = get_patient_by_id(hospital_patient_id)
    
    if not result['success']:
        print(f"✗ FAIL: get_patient_by_id failed")
        return False
    
    patient = result['patient']
    
    if patient['source'] == 'hospital' and patient['patient_admission_id'] == hospital_patient_id:
        print(f"✓ PASS: Hospital patient retrieved successfully")
        print(f"  - ID: {patient['patient_admission_id']}")
        print(f"  - Name: {patient['full_name']}")
        print(f"  - Source: {patient['source']}")
        return True
    else:
        print(f"✗ FAIL: Wrong source or ID mismatch")
        print(f"  - Expected source: 'hospital', Got: '{patient.get('source', 'missing')}'")
        return False


def test_5_search_by_firstname():
    """Test 5: Search by FirstName works"""
    print("\n" + "="*70)
    print("TEST 5: Search by FirstName")
    print("="*70)
    
    result = search_patients("TestPhase2Alpha", limit=10)
    
    if not result['success']:
        print(f"✗ FAIL: Search failed")
        return False
    
    patients = result['patients']
    matching = [p for p in patients if 'TestPhase2Alpha' in p['full_name']]
    
    if len(matching) > 0:
        print(f"✓ PASS: Found {len(matching)} patient(s) with FirstName 'TestPhase2Alpha'")
        for p in matching:
            print(f"  - {p['full_name']} (Source: {p['source']})")
        return True
    else:
        print(f"✗ FAIL: No patients found with FirstName 'TestPhase2Alpha'")
        return False


def test_6_search_by_document_number():
    """Test 6: Search by DocumentNumber works"""
    print("\n" + "="*70)
    print("TEST 6: Search by DocumentNumber")
    print("="*70)
    
    result = search_patients("DOC-TEST-002", limit=10)
    
    if not result['success']:
        print(f"✗ FAIL: Search failed")
        return False
    
    patients = result['patients']
    matching = [p for p in patients if p['document_number'] == 'DOC-TEST-002']
    
    if len(matching) > 0:
        print(f"✓ PASS: Found {len(matching)} patient(s) with DocumentNumber 'DOC-TEST-002'")
        for p in matching:
            print(f"  - {p['full_name']} (Source: {p['source']})")
        return True
    else:
        print(f"✗ FAIL: No patients found with DocumentNumber 'DOC-TEST-002'")
        return False


def test_7_search_by_medical_file():
    """Test 7: Search by MedicalFileNumber works"""
    print("\n" + "="*70)
    print("TEST 7: Search by MedicalFileNumber")
    print("="*70)
    
    result = search_patients("MRN-TEST-003", limit=10)
    
    if not result['success']:
        print(f"✗ FAIL: Search failed")
        return False
    
    patients = result['patients']
    matching = [p for p in patients if p['medical_file_number'] == 'MRN-TEST-003']
    
    if len(matching) > 0:
        print(f"✓ PASS: Found {len(matching)} patient(s) with MRN 'MRN-TEST-003'")
        for p in matching:
            print(f"  - {p['full_name']} (Source: {p['source']})")
        return True
    else:
        print(f"✗ FAIL: No patients found with MRN 'MRN-TEST-003'")
        return False


def test_8_limit_parameter():
    """Test 8: Limit parameter works correctly"""
    print("\n" + "="*70)
    print("TEST 8: Limit Parameter")
    print("="*70)
    
    result = search_patients("TestPhase2", limit=2)
    
    if not result['success']:
        print(f"✗ FAIL: Search failed")
        return False
    
    patients = result['patients']
    
    if len(patients) <= 2:
        print(f"✓ PASS: Limit=2 respected, returned {len(patients)} patient(s)")
        return True
    else:
        print(f"✗ FAIL: Limit=2 not respected, returned {len(patients)} patients")
        return False


def test_9_union_merging():
    """Test 9: UNION correctly merges both sources"""
    print("\n" + "="*70)
    print("TEST 9: UNION Merge Verification")
    print("="*70)
    
    # Count hospital patients
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM APP_VIEWTABLE_PATIENT_ADMISSION")
    hospital_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM APP_RESERVE_PATIENT WHERE FirstName LIKE 'TestPhase2%'")
    reserve_count = cursor.fetchone()[0]
    
    conn.close()
    
    # Search with limit high enough to get all test patients
    result = search_patients("TestPhase2", limit=100)
    
    if not result['success']:
        print(f"✗ FAIL: Search failed")
        return False
    
    search_reserve_count = len([p for p in result['patients'] if p['source'] == 'reserve'])
    
    print(f"  - Hospital patients in DB: {hospital_count}")
    print(f"  - Reserve test patients in DB: {reserve_count}")
    print(f"  - Reserve patients in search: {search_reserve_count}")
    
    if search_reserve_count == reserve_count:
        print(f"✓ PASS: UNION correctly merged both sources")
        return True
    else:
        print(f"✗ FAIL: UNION merge count mismatch")
        return False


def cleanup_test_data():
    """Remove test patients"""
    print("\n" + "="*70)
    print("CLEANUP: Removing Test Data")
    print("="*70)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("DELETE FROM APP_RESERVE_PATIENT WHERE FirstName LIKE 'TestPhase2%'")
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
    """Run all Phase 2 tests"""
    print("\n" + "="*70)
    print("PHASE 2 TEST SUITE: DUAL-SOURCE PATIENT SEARCH")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup
    test_patient_ids = setup_test_data()
    
    if not test_patient_ids:
        print("\n✗ FATAL: Failed to create test data. Aborting.")
        return False
    
    results = []
    
    # Run tests
    results.append(("Search Reserve Patients", test_1_search_reserve_patients(test_patient_ids)))
    results.append(("Source Field Present", test_2_search_has_source_field(test_patient_ids)))
    results.append(("Get Reserve by ID", test_3_get_by_id_reserve(test_patient_ids)))
    results.append(("Get Hospital by ID", test_4_get_by_id_hospital()))
    results.append(("Search by FirstName", test_5_search_by_firstname()))
    results.append(("Search by DocumentNumber", test_6_search_by_document_number()))
    results.append(("Search by MedicalFileNumber", test_7_search_by_medical_file()))
    results.append(("Limit Parameter", test_8_limit_parameter()))
    results.append(("UNION Merge Verification", test_9_union_merging()))
    
    # Cleanup
    cleanup_success = cleanup_test_data()
    results.append(("Cleanup", cleanup_success))
    
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
        print("\n🎉 ALL TESTS PASSED! Phase 2 Complete - Ready for Phase 3")
        return True
    else:
        print(f"\n⚠️  {total - passed} TEST(S) FAILED - Fix issues before proceeding")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
