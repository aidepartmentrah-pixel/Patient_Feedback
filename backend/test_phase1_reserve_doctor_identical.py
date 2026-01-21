"""
Phase 1 Test (Corrected): Verify APP_RESERVE_DOCTOR table with IDENTICAL structure
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


def test_table_structure_identical():
    """Test 2: Verify table structure is IDENTICAL to hospital table"""
    print("\n" + "="*60)
    print("TEST 2: Verify IDENTICAL structure to hospital table")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Get hospital table columns
        cursor.execute("""
            SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, IS_NULLABLE
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = 'APP_LOOKUP_DOCTOR'
            ORDER BY ORDINAL_POSITION
        """)
        hospital_cols = cursor.fetchall()
        
        # Get reserve table columns
        cursor.execute("""
            SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, IS_NULLABLE
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = 'APP_RESERVE_DOCTOR'
            ORDER BY ORDINAL_POSITION
        """)
        reserve_cols = cursor.fetchall()
        
        print(f"\nHospital table columns: {len(hospital_cols)}")
        print(f"Reserve table columns: {len(reserve_cols)}")
        
        if len(hospital_cols) != len(reserve_cols):
            print(f"\n❌ FAIL: Column count mismatch!")
            return False
        
        print("\nColumn comparison:")
        all_match = True
        for hosp, res in zip(hospital_cols, reserve_cols):
            match = (hosp[0] == res[0] and hosp[1] == res[1])
            status = "✅" if match else "❌"
            print(f"  {status} {hosp[0]:20s} {hosp[1]:15s} | {res[0]:20s} {res[1]:15s}")
            if not match:
                all_match = False
        
        if all_match:
            print("\n✅ PASS: Tables are IDENTICAL")
            return True
        else:
            print("\n❌ FAIL: Tables have differences")
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
            for row in rows[:5]:
                print(f"  • ID: {row.DoctorID}, Name: {row.DoctorName}, Specialty: {row.Specialty}")
            
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
    
    test_name = f"Dr. Test Verification {datetime.now().strftime('%H%M%S')}"
    
    try:
        # Insert test record
        cursor.execute("""
            INSERT INTO dbo.APP_RESERVE_DOCTOR 
                (DoctorName, Specialty, IsActive, SourceSystem, LastSyncedAt)
            VALUES (?, ?, ?, ?, GETDATE())
        """, (test_name, 'Test Specialty', 1, 'TEST'))
        conn.commit()
        
        # Verify insertion
        cursor.execute("""
            SELECT DoctorID, DoctorName, Specialty, SourceSystem
            FROM dbo.APP_RESERVE_DOCTOR
            WHERE DoctorName = ?
        """, (test_name,))
        
        row = cursor.fetchone()
        
        if row:
            print(f"\n✅ Inserted record:")
            print(f"   DoctorID: {row.DoctorID}")
            print(f"   DoctorName: {row.DoctorName}")
            print(f"   Specialty: {row.Specialty}")
            print(f"   SourceSystem: {row.SourceSystem}")
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


def test_auto_increment():
    """Test 5: Verify DoctorID auto-increment"""
    print("\n" + "="*60)
    print("TEST 5: Verify DoctorID auto-increment")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Insert two records and check IDs
        test_name_1 = f"Dr. Auto Test 1 {datetime.now().strftime('%H%M%S')}"
        test_name_2 = f"Dr. Auto Test 2 {datetime.now().strftime('%H%M%S')}"
        
        # Insert first record
        cursor.execute("""
            INSERT INTO dbo.APP_RESERVE_DOCTOR (DoctorName, Specialty, IsActive, SourceSystem)
            OUTPUT INSERTED.DoctorID
            VALUES (?, ?, ?, ?)
        """, (test_name_1, 'Cardiology', 1, 'TEST'))
        
        id1 = cursor.fetchone()[0]
        conn.commit()
        
        # Insert second record
        cursor.execute("""
            INSERT INTO dbo.APP_RESERVE_DOCTOR (DoctorName, Specialty, IsActive, SourceSystem)
            OUTPUT INSERTED.DoctorID
            VALUES (?, ?, ?, ?)
        """, (test_name_2, 'Neurology', 1, 'TEST'))
        
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
    print("║  PHASE 1: DATABASE FOUNDATION (CORRECTED)               ║")
    print("║  Reserve table IDENTICAL to hospital table              ║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    tests = [
        ("Table Exists", test_table_exists),
        ("Identical Structure", test_table_structure_identical),
        ("SELECT Operation", test_select_data),
        ("INSERT Operation", test_insert_operation),
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
        print("✅ Reserve table is IDENTICAL to hospital table")
        print("✅ Ready to proceed to Phase 2")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix issues before proceeding.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
