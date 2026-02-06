"""
Test Phase A Step 1: Extend User Table Schema
==============================================

Tests that the DisplayName and DepartmentDisplayName columns
have been added to the APP_Users table correctly.

Run after executing: phase_a_step1_extend_user_table.sql
"""

import sys
import os
import pyodbc

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def get_connection():
    """Get SQL Server database connection."""
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=SOCIALMEDIA;"
        "DATABASE=IncidentManager;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )
    return conn


def test_displayname_column_exists():
    """Test that DisplayName column exists in APP_Users table."""
    print("\n" + "="*60)
    print("TEST 1: DisplayName Column Exists")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT 
                c.name AS ColumnName,
                t.name AS DataType,
                c.max_length AS MaxLength,
                c.is_nullable AS IsNullable
            FROM sys.columns c
            INNER JOIN sys.types t ON c.user_type_id = t.user_type_id
            WHERE c.object_id = OBJECT_ID('dbo.APP_Users')
            AND c.name = 'DisplayName'
        """)
        
        row = cursor.fetchone()
        
        assert row is not None, "DisplayName column does not exist"
        
        print(f"✓ Column exists: {row.ColumnName}")
        print(f"✓ Data type: {row.DataType}")
        print(f"✓ Max length: {row.MaxLength}")
        print(f"✓ Nullable: {bool(row.IsNullable)}")
        
        # Verify specifications
        assert row.DataType == 'nvarchar', f"Expected nvarchar, got {row.DataType}"
        assert row.MaxLength == 300, f"Expected 300 bytes (150 chars), got {row.MaxLength}"  # nvarchar uses 2 bytes per char
        assert row.IsNullable == 1, f"Expected nullable, got {row.IsNullable}"
        
        print("\n✓ TEST 1 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 1 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 1 ERROR: {str(e)}")
        return False
    finally:
        cursor.close()
        conn.close()


def test_departmentdisplayname_column_exists():
    """Test that DepartmentDisplayName column exists in APP_Users table."""
    print("\n" + "="*60)
    print("TEST 2: DepartmentDisplayName Column Exists")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT 
                c.name AS ColumnName,
                t.name AS DataType,
                c.max_length AS MaxLength,
                c.is_nullable AS IsNullable
            FROM sys.columns c
            INNER JOIN sys.types t ON c.user_type_id = t.user_type_id
            WHERE c.object_id = OBJECT_ID('dbo.APP_Users')
            AND c.name = 'DepartmentDisplayName'
        """)
        
        row = cursor.fetchone()
        
        assert row is not None, "DepartmentDisplayName column does not exist"
        
        print(f"✓ Column exists: {row.ColumnName}")
        print(f"✓ Data type: {row.DataType}")
        print(f"✓ Max length: {row.MaxLength}")
        print(f"✓ Nullable: {bool(row.IsNullable)}")
        
        # Verify specifications
        assert row.DataType == 'nvarchar', f"Expected nvarchar, got {row.DataType}"
        assert row.MaxLength == 300, f"Expected 300 bytes (150 chars), got {row.MaxLength}"  # nvarchar uses 2 bytes per char
        assert row.IsNullable == 1, f"Expected nullable, got {row.IsNullable}"
        
        print("\n✓ TEST 2 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 2 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 2 ERROR: {str(e)}")
        return False
    finally:
        cursor.close()
        conn.close()


def test_no_constraints_on_new_columns():
    """Test that new columns have no constraints, indexes, or defaults."""
    print("\n" + "="*60)
    print("TEST 3: No Constraints on New Columns")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Check for constraints
        cursor.execute("""
            SELECT COUNT(*) AS constraint_count
            FROM sys.default_constraints
            WHERE parent_object_id = OBJECT_ID('dbo.APP_Users')
            AND parent_column_id IN (
                SELECT column_id 
                FROM sys.columns 
                WHERE object_id = OBJECT_ID('dbo.APP_Users')
                AND name IN ('DisplayName', 'DepartmentDisplayName')
            )
        """)
        
        row = cursor.fetchone()
        constraint_count = row.constraint_count
        
        print(f"✓ Default constraints found: {constraint_count}")
        assert constraint_count == 0, f"Expected 0 default constraints, found {constraint_count}"
        
        # Check for indexes specifically on these columns
        cursor.execute("""
            SELECT COUNT(*) AS index_count
            FROM sys.indexes i
            INNER JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
            INNER JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
            WHERE i.object_id = OBJECT_ID('dbo.APP_Users')
            AND c.name IN ('DisplayName', 'DepartmentDisplayName')
            AND i.is_primary_key = 0
            AND i.is_unique_constraint = 0
        """)
        
        row = cursor.fetchone()
        index_count = row.index_count
        
        print(f"✓ Indexes found: {index_count}")
        assert index_count == 0, f"Expected 0 indexes, found {index_count}"
        
        print("\n✓ TEST 3 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 3 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 3 ERROR: {str(e)}")
        return False
    finally:
        cursor.close()
        conn.close()


def test_existing_users_still_work():
    """Test that existing users in the table are not affected."""
    print("\n" + "="*60)
    print("TEST 4: Existing Users Still Work")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Query existing users
        cursor.execute("""
            SELECT 
                UserID,
                Username,
                IsActive,
                DisplayName,
                DepartmentDisplayName
            FROM dbo.APP_Users
            ORDER BY UserID
        """)
        
        users = cursor.fetchall()
        user_count = len(users)
        
        print(f"✓ Found {user_count} existing users")
        
        assert user_count > 0, "No users found in APP_Users table"
        
        # Verify each user has NULL values for new columns (since they're new)
        null_display_name_count = sum(1 for u in users if u.DisplayName is None)
        null_dept_count = sum(1 for u in users if u.DepartmentDisplayName is None)
        
        print(f"✓ Users with NULL DisplayName: {null_display_name_count}/{user_count}")
        print(f"✓ Users with NULL DepartmentDisplayName: {null_dept_count}/{user_count}")
        
        # All existing users should have NULL for new columns (unless previously populated)
        # This is expected behavior for backward compatibility
        
        # Verify we can still query all columns
        for user in users:
            assert hasattr(user, 'UserID'), "Missing UserID attribute"
            assert hasattr(user, 'Username'), "Missing Username attribute"
            assert hasattr(user, 'DisplayName'), "Missing DisplayName attribute"
            assert hasattr(user, 'DepartmentDisplayName'), "Missing DepartmentDisplayName attribute"
        
        print(f"✓ All users queryable with new columns")
        
        print("\n✓ TEST 4 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 4 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 4 ERROR: {str(e)}")
        return False
    finally:
        cursor.close()
        conn.close()


def test_can_insert_with_new_columns():
    """Test that we can insert data into the new columns."""
    print("\n" + "="*60)
    print("TEST 5: Can Insert with New Columns")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Get a test user
        cursor.execute("""
            SELECT TOP 1 UserID, Username, DisplayName, DepartmentDisplayName
            FROM dbo.APP_Users
            ORDER BY UserID
        """)
        
        user = cursor.fetchone()
        assert user is not None, "No user found to test with"
        
        test_user_id = user.UserID
        original_display_name = user.DisplayName
        original_dept_name = user.DepartmentDisplayName
        
        print(f"✓ Testing with UserID: {test_user_id}")
        print(f"  Original DisplayName: {original_display_name}")
        print(f"  Original DepartmentDisplayName: {original_dept_name}")
        
        # Update with test values
        test_display_name = "Test User"
        test_dept_name = "Test Department"
        
        cursor.execute("""
            UPDATE dbo.APP_Users
            SET DisplayName = ?,
                DepartmentDisplayName = ?
            WHERE UserID = ?
        """, (test_display_name, test_dept_name, test_user_id))
        
        conn.commit()
        print(f"✓ Updated user {test_user_id} with test values")
        
        # Verify update
        cursor.execute("""
            SELECT DisplayName, DepartmentDisplayName
            FROM dbo.APP_Users
            WHERE UserID = ?
        """, (test_user_id,))
        
        updated_user = cursor.fetchone()
        
        assert updated_user.DisplayName == test_display_name, \
            f"Expected DisplayName '{test_display_name}', got '{updated_user.DisplayName}'"
        assert updated_user.DepartmentDisplayName == test_dept_name, \
            f"Expected DepartmentDisplayName '{test_dept_name}', got '{updated_user.DepartmentDisplayName}'"
        
        print(f"✓ Verified DisplayName: {updated_user.DisplayName}")
        print(f"✓ Verified DepartmentDisplayName: {updated_user.DepartmentDisplayName}")
        
        # Restore original values
        cursor.execute("""
            UPDATE dbo.APP_Users
            SET DisplayName = ?,
                DepartmentDisplayName = ?
            WHERE UserID = ?
        """, (original_display_name, original_dept_name, test_user_id))
        
        conn.commit()
        print(f"✓ Restored original values for user {test_user_id}")
        
        print("\n✓ TEST 5 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 5 FAILED: {str(e)}")
        # Try to rollback
        try:
            conn.rollback()
        except:
            pass
        return False
    except Exception as e:
        print(f"\n✗ TEST 5 ERROR: {str(e)}")
        try:
            conn.rollback()
        except:
            pass
        return False
    finally:
        cursor.close()
        conn.close()


def run_all_tests():
    """Run all Phase A Step 1 tests."""
    print("\n" + "="*60)
    print("PHASE A - STEP 1: EXTEND USER TABLE SCHEMA")
    print("TEST SUITE")
    print("="*60)
    
    tests = [
        ("Test 1: DisplayName Column Exists", test_displayname_column_exists),
        ("Test 2: DepartmentDisplayName Column Exists", test_departmentdisplayname_column_exists),
        ("Test 3: No Constraints on New Columns", test_no_constraints_on_new_columns),
        ("Test 4: Existing Users Still Work", test_existing_users_still_work),
        ("Test 5: Can Insert with New Columns", test_can_insert_with_new_columns),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ {name} EXCEPTION: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {len(tests)}")
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print(f"Pass Rate: {(passed/len(tests)*100):.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED (100%)")
        print("✅ Phase A Step 1 schema changes verified successfully")
    else:
        print(f"\n❌ {failed} TEST(S) FAILED")
        print("Please run the migration script first:")
        print("  sqlcmd -S SOCIALMEDIA -d IncidentManager -E -i backend\\database_migrations\\phase_a_step1_extend_user_table.sql")
    
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
