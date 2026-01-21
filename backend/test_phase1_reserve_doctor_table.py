"""
Phase 1 Test: Verify APP_RESERVE_DOCTOR table creation and basic operations
"""

import pyodbc
from datetime import datetime


def get_connection():
    """Get database connection."""
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=SOCIALMEDIA;"
        "DATABASE=IncidentManager;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )
    return conn


def test_table_exists():
    """Test 1: Verify table exists in database"""
    print("\n" + "="*60)
    print("TEST 1: Verify APP_RESERVE_DOCTOR table exists")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = 'dbo' 
            AND TABLE_NAME = 'APP_RESERVE_DOCTOR'
        """)
        count = cursor.fetchone()[0]
        
        if count == 1:
            print("✅ PASS: APP_RESERVE_DOCTOR table exists")
            return True
        else:
            print("❌ FAIL: APP_RESERVE_DOCTOR table does NOT exist")
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    finally:
        conn.close()


def test_table_structure():
    """Test 2: Verify table has correct columns"""
    print("\n" + "="*60)
    print("TEST 2: Verify table structure")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    expected_columns = [
        'DoctorID', 'EmployeeID', 'DoctorName', 'Department', 
        'Specialty', 'Email', 'Phone', 'LicenseNumber', 
        'HireDate', 'IsActive', 'CreatedAt', 'UpdatedAt'
    ]
    
    try:
        cursor.execute("""
            SELECT COLUMN_NAME 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = 'dbo' 
            AND TABLE_NAME = 'APP_RESERVE_DOCTOR'
            ORDER BY ORDINAL_POSITION
        """)
        
        actual_columns = [row[0] for row in cursor.fetchall()]
        
        print(f"\nExpected columns: {len(expected_columns)}")
        print(f"Actual columns: {len(actual_columns)}")
        print(f"\nColumn list:")
        
        all_match = True
        for col in expected_columns:
            if col in actual_columns:
                print(f"  ✅ {col}")
            else:
                print(f"  ❌ {col} - MISSING")
                all_match = False
        
        if all_match:
            print("\n✅ PASS: All expected columns exist")
            return True
        else:
            print("\n❌ FAIL: Some columns are missing")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    finally:
        conn.close()


def test_select_data():
    """Test 3: Verify we can SELECT from table"""
    print("\n" + "="*60)
    print("TEST 3: Verify SELECT operation")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT * FROM dbo.APP_RESERVE_DOCTOR")
        rows = cursor.fetchall()
        
        print(f"\nFound {len(rows)} record(s) in table")
        
        if len(rows) > 0:
            print("\nSample records:")
            for row in rows[:5]:  # Show first 5
                print(f"  • ID: {row.DoctorID}, EmpID: {row.EmployeeID}, Name: {row.DoctorName}, Dept: {row.Department}")
            
            print("\n✅ PASS: Can SELECT from table")
            return True
        else:
            print("⚠️  WARNING: Table is empty (no test data)")
            print("✅ PASS: Can SELECT from table (but empty)")
            return True
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    finally:
        conn.close()


def test_insert_operation():
    """Test 4: Verify we can INSERT into table"""
    print("\n" + "="*60)
    print("TEST 4: Verify INSERT operation")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    test_employee_id = f"TEST-VERIFY-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    try:
        # Insert test record
        cursor.execute("""
            INSERT INTO dbo.APP_RESERVE_DOCTOR 
                (EmployeeID, DoctorName, Department, Specialty, Email, Phone)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            test_employee_id,
            'Dr. Test Verification',
            'Test Department',
            'Test Specialty',
            'test@hospital.com',
            '+966500000000'
        ))
        conn.commit()
        
        # Verify insertion
        cursor.execute("""
            SELECT DoctorID, EmployeeID, DoctorName, Department
            FROM dbo.APP_RESERVE_DOCTOR
            WHERE EmployeeID = ?
        """, (test_employee_id,))
        
        row = cursor.fetchone()
        
        if row:
            print(f"\n✅ Inserted record:")
            print(f"   DoctorID: {row.DoctorID}")
            print(f"   EmployeeID: {row.EmployeeID}")
            print(f"   DoctorName: {row.DoctorName}")
            print(f"   Department: {row.Department}")
            print("\n✅ PASS: Can INSERT into table")
            return True
        else:
            print("❌ FAIL: Insert succeeded but cannot find record")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def test_unique_constraint():
    """Test 5: Verify EmployeeID unique constraint"""
    print("\n" + "="*60)
    print("TEST 5: Verify EmployeeID unique constraint")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    test_employee_id = f"TEST-UNIQUE-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    try:
        # Insert first record
        cursor.execute("""
            INSERT INTO dbo.APP_RESERVE_DOCTOR 
                (EmployeeID, DoctorName)
            VALUES (?, ?)
        """, (test_employee_id, 'Dr. First Insert'))
        conn.commit()
        print(f"✅ First insert successful with EmployeeID: {test_employee_id}")
        
        # Try to insert duplicate
        try:
            cursor.execute("""
                INSERT INTO dbo.APP_RESERVE_DOCTOR 
                    (EmployeeID, DoctorName)
                VALUES (?, ?)
            """, (test_employee_id, 'Dr. Duplicate Attempt'))
            conn.commit()
            print("❌ FAIL: Duplicate EmployeeID was allowed (should have been rejected)")
            return False
        except pyodbc.IntegrityError as ie:
            print(f"✅ Duplicate correctly rejected: {str(ie)[:100]}...")
            print("\n✅ PASS: Unique constraint is working")
            return True
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def test_auto_increment():
    """Test 6: Verify DoctorID auto-increment"""
    print("\n" + "="*60)
    print("TEST 6: Verify DoctorID auto-increment")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Insert two records and check IDs
        test_id_1 = f"TEST-AUTO-1-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        test_id_2 = f"TEST-AUTO-2-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Insert first record
        cursor.execute("""
            INSERT INTO dbo.APP_RESERVE_DOCTOR (EmployeeID, DoctorName)
            OUTPUT INSERTED.DoctorID
            VALUES (?, ?)
        """, (test_id_1, 'Dr. Auto Test 1'))
        
        id1 = cursor.fetchone()[0]
        conn.commit()
        
        # Insert second record
        cursor.execute("""
            INSERT INTO dbo.APP_RESERVE_DOCTOR (EmployeeID, DoctorName)
            OUTPUT INSERTED.DoctorID
            VALUES (?, ?)
        """, (test_id_2, 'Dr. Auto Test 2'))
        
        id2 = cursor.fetchone()[0]
        conn.commit()
        
        print(f"First insert DoctorID: {id1}")
        print(f"Second insert DoctorID: {id2}")
        
        if id2 > id1:
            print(f"✅ IDs are incrementing correctly ({id2} > {id1})")
            print("\n✅ PASS: Auto-increment is working")
            return True
        else:
            print(f"❌ FAIL: IDs not incrementing ({id2} should be > {id1})")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def run_all_tests():
    """Run all Phase 1 tests"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║  PHASE 1: DATABASE FOUNDATION - VERIFICATION TESTS      ║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    tests = [
        ("Table Exists", test_table_exists),
        ("Table Structure", test_table_structure),
        ("SELECT Operation", test_select_data),
        ("INSERT Operation", test_insert_operation),
        ("Unique Constraint", test_unique_constraint),
        ("Auto-Increment", test_auto_increment),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR in {test_name}: {e}")
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
        print("\n🎉 ALL TESTS PASSED! Phase 1 is 100% complete!")
        print("✅ Ready to proceed to Phase 2")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix issues before proceeding.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
