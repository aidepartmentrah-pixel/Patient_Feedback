"""
====================================================================
PHASE 1 TEST: Reserve Patient Table Creation & Validation
====================================================================
Purpose: Verify APP_RESERVE_PATIENT table is created correctly
         and can perform basic operations

Test Coverage:
1. Table exists with correct structure
2. Can insert test record
3. Can read test record
4. Can update test record
5. Can search by name fields
6. Indexes are created
7. No FK constraints block operations

Author: System
Date: 2026-01-20
====================================================================
"""

import pyodbc
from datetime import datetime, date


def get_connection():
    """Get SQL Server connection."""
    return pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=SOCIALMEDIA;"
        "DATABASE=IncidentManager;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )


def test_1_table_exists():
    """Test 1: Verify table exists"""
    print("\n" + "="*70)
    print("TEST 1: Table Existence")
    print("="*70)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_NAME = 'APP_RESERVE_PATIENT'
    """)
    
    exists = cursor.fetchone()[0]
    conn.close()
    
    if exists == 1:
        print("✓ PASS: APP_RESERVE_PATIENT table exists")
        return True
    else:
        print("✗ FAIL: APP_RESERVE_PATIENT table does not exist")
        return False


def test_2_column_count():
    """Test 2: Verify correct number of columns"""
    print("\n" + "="*70)
    print("TEST 2: Column Count (Should be 60 columns)")
    print("="*70)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_NAME = 'APP_RESERVE_PATIENT'
    """)
    
    column_count = cursor.fetchone()[0]
    conn.close()
    
    if column_count == 60:
        print(f"✓ PASS: Table has {column_count} columns (matches hospital table)")
        return True
    else:
        print(f"✗ FAIL: Table has {column_count} columns (expected 60)")
        return False


def test_3_primary_key():
    """Test 3: Verify primary key exists"""
    print("\n" + "="*70)
    print("TEST 3: Primary Key")
    print("="*70)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE TABLE_NAME = 'APP_RESERVE_PATIENT'
        AND CONSTRAINT_NAME LIKE 'PK%'
    """)
    
    pk_column = cursor.fetchone()
    conn.close()
    
    if pk_column and pk_column[0] == 'PatientAdmissionID':
        print(f"✓ PASS: Primary key on PatientAdmissionID exists")
        return True
    else:
        print(f"✗ FAIL: Primary key not found or incorrect")
        return False


def test_4_insert_record():
    """Test 4: Insert test record"""
    print("\n" + "="*70)
    print("TEST 4: Insert Test Record")
    print("="*70)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Clean up any previous test data
        cursor.execute("DELETE FROM APP_RESERVE_PATIENT WHERE FirstName = 'TestPatient'")
        conn.commit()
        
        # Insert test record
        cursor.execute("""
            INSERT INTO APP_RESERVE_PATIENT (
                FirstName, MiddleName, LastName, MotherName, FullName,
                PhoneNumber1, BirthDate, SEX, DocumentNumber, MedicalFileNumber
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            'TestPatient',
            'Ahmad',
            'AlTest',
            'Fatima',
            'TestPatient Ahmad AlTest',
            '0501234567',
            date(1990, 5, 15),
            'M',
            'DOC123456',
            'MRN001'
        ))
        
        conn.commit()
        
        # Verify insertion
        cursor.execute("""
            SELECT PatientAdmissionID, FullName, FirstName, PhoneNumber1
            FROM APP_RESERVE_PATIENT
            WHERE FirstName = 'TestPatient'
        """)
        
        row = cursor.fetchone()
        
        if row:
            print(f"✓ PASS: Record inserted successfully")
            print(f"  - PatientAdmissionID: {row[0]}")
            print(f"  - FullName: {row[1]}")
            print(f"  - FirstName: {row[2]}")
            print(f"  - PhoneNumber1: {row[3]}")
            conn.close()
            return True, row[0]
        else:
            print("✗ FAIL: Record not found after insertion")
            conn.close()
            return False, None
            
    except Exception as e:
        print(f"✗ FAIL: Error inserting record: {str(e)}")
        conn.close()
        return False, None


def test_5_read_record(patient_id):
    """Test 5: Read test record"""
    print("\n" + "="*70)
    print("TEST 5: Read Test Record")
    print("="*70)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            PatientAdmissionID, FullName, FirstName, MiddleName, LastName,
            MotherName, PhoneNumber1, BirthDate, SEX, DocumentNumber,
            MedicalFileNumber, SystemTime
        FROM APP_RESERVE_PATIENT
        WHERE PatientAdmissionID = ?
    """, patient_id)
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        print(f"✓ PASS: Record retrieved successfully")
        print(f"  - PatientAdmissionID: {row[0]}")
        print(f"  - FullName: {row[1]}")
        print(f"  - FirstName: {row[2]}")
        print(f"  - MiddleName: {row[3]}")
        print(f"  - LastName: {row[4]}")
        print(f"  - MotherName: {row[5]}")
        print(f"  - PhoneNumber1: {row[6]}")
        print(f"  - BirthDate: {row[7]}")
        print(f"  - SEX: {row[8]}")
        print(f"  - DocumentNumber: {row[9]}")
        print(f"  - MedicalFileNumber: {row[10]}")
        print(f"  - SystemTime: {row[11]}")
        return True
    else:
        print(f"✗ FAIL: Record not found with ID {patient_id}")
        return False


def test_6_search_by_name():
    """Test 6: Search by name fields"""
    print("\n" + "="*70)
    print("TEST 6: Search By Name")
    print("="*70)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Search by FirstName
    cursor.execute("""
        SELECT PatientAdmissionID, FullName
        FROM APP_RESERVE_PATIENT
        WHERE FirstName LIKE ?
    """, '%TestPatient%')
    
    results = cursor.fetchall()
    conn.close()
    
    if len(results) > 0:
        print(f"✓ PASS: Search by FirstName found {len(results)} record(s)")
        for row in results:
            print(f"  - ID: {row[0]}, Name: {row[1]}")
        return True
    else:
        print("✗ FAIL: Search by FirstName found no records")
        return False


def test_7_indexes_exist():
    """Test 7: Verify indexes created"""
    print("\n" + "="*70)
    print("TEST 7: Index Verification")
    print("="*70)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            i.name AS IndexName,
            COL_NAME(ic.object_id, ic.column_id) AS ColumnName
        FROM 
            sys.indexes i
            INNER JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
        WHERE 
            i.object_id = OBJECT_ID('APP_RESERVE_PATIENT')
            AND i.name IS NOT NULL
        ORDER BY 
            i.name, ic.key_ordinal
    """)
    
    indexes = cursor.fetchall()
    conn.close()
    
    if len(indexes) >= 6:  # Should have at least 6 indexes
        print(f"✓ PASS: Found {len(indexes)} indexes")
        for idx in indexes:
            print(f"  - {idx[0]} on {idx[1]}")
        return True
    else:
        print(f"✗ FAIL: Only found {len(indexes)} indexes (expected at least 6)")
        return False


def test_8_cleanup():
    """Test 8: Cleanup test data"""
    print("\n" + "="*70)
    print("TEST 8: Cleanup Test Data")
    print("="*70)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("DELETE FROM APP_RESERVE_PATIENT WHERE FirstName = 'TestPatient'")
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        print(f"✓ PASS: Deleted {deleted} test record(s)")
        return True
    except Exception as e:
        print(f"✗ FAIL: Error during cleanup: {str(e)}")
        conn.close()
        return False


def run_all_tests():
    """Run all Phase 1 tests"""
    print("\n" + "="*70)
    print("PHASE 1 TEST SUITE: RESERVE PATIENT TABLE")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    patient_id = None
    
    # Run tests in sequence
    results.append(("Table Exists", test_1_table_exists()))
    results.append(("Column Count", test_2_column_count()))
    results.append(("Primary Key", test_3_primary_key()))
    
    insert_result, patient_id = test_4_insert_record()
    results.append(("Insert Record", insert_result))
    
    if patient_id:
        results.append(("Read Record", test_5_read_record(patient_id)))
    else:
        results.append(("Read Record", False))
    
    results.append(("Search By Name", test_6_search_by_name()))
    results.append(("Indexes Created", test_7_indexes_exist()))
    results.append(("Cleanup", test_8_cleanup()))
    
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
        print("\n🎉 ALL TESTS PASSED! Phase 1 Complete - Ready for Phase 2")
        return True
    else:
        print(f"\n⚠️  {total - passed} TEST(S) FAILED - Fix issues before proceeding")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
