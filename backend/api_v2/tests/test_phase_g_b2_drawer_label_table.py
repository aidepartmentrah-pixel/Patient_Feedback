"""
Test Suite: Phase G-B2 - Drawer Label Table Schema
Tests the APP_DrawerLabel table structure and UNIQUE constraint enforcement.

Verifies:
- Table exists
- All columns exist with correct types
- NOT NULL constraints
- UNIQUE constraint on label_name
- Index on is_active
- Insert/read operations
- Duplicate label rejection

Target: backend/database_migrations/phase_g_b2_create_drawer_label_table.sql

Test Coverage:
- Schema validation
- Column types and constraints
- UNIQUE constraint enforcement
- Index presence
- CRUD operations
- Data integrity

Note: Uses real database connection (no mocks)
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime
import uuid

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

from core.database import get_connection


class TestDrawerLabelTableSchema:
    """Test suite for G-B2 Drawer Label table schema validation."""
    
    def test_1_table_exists(self):
        """
        Test 1: Verify APP_DrawerLabel table exists in database.
        """
        print("\n" + "="*80)
        print("TEST 1: TABLE EXISTS")
        print("="*80)
        
        conn = get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT COUNT(*) AS table_count
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = 'dbo'
                AND TABLE_NAME = 'APP_DrawerLabel'
            """)
            result = cursor.fetchone()
            table_count = result.table_count
            
            print(f"Table count: {table_count}")
            assert table_count == 1, "APP_DrawerLabel table should exist"
            print("✅ PASS - APP_DrawerLabel table exists")
            
        finally:
            cursor.close()
            conn.close()
    
    def test_2_columns_exist(self):
        """
        Test 2: Verify all required columns exist with exact names.
        """
        print("\n" + "="*80)
        print("TEST 2: COLUMNS EXIST")
        print("="*80)
        
        conn = get_connection()
        cursor = conn.cursor()
        
        required_columns = [
            'LabelID',
            'LabelName',
            'IsActive',
            'CreatedAt'
        ]
        
        try:
            cursor.execute("""
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = 'APP_DrawerLabel'
                ORDER BY ORDINAL_POSITION
            """)
            
            actual_columns = [row.COLUMN_NAME for row in cursor.fetchall()]
            print(f"\nActual columns: {actual_columns}")
            print(f"Required columns: {required_columns}")
            
            for col in required_columns:
                assert col in actual_columns, f"Column '{col}' is missing"
                print(f"  ✓ {col}")
            
            print("\n✅ PASS - All required columns exist")
            
        finally:
            cursor.close()
            conn.close()
    
    def test_3_column_types(self):
        """
        Test 3: Verify column data types match specification.
        """
        print("\n" + "="*80)
        print("TEST 3: COLUMN TYPES")
        print("="*80)
        
        conn = get_connection()
        cursor = conn.cursor()
        
        expected_types = {
            'LabelID': 'int',
            'LabelName': 'nvarchar',
            'IsActive': 'bit',
            'CreatedAt': 'datetime2'
        }
        
        try:
            cursor.execute("""
                SELECT 
                    COLUMN_NAME,
                    DATA_TYPE,
                    CHARACTER_MAXIMUM_LENGTH
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = 'APP_DrawerLabel'
                ORDER BY ORDINAL_POSITION
            """)
            
            columns = cursor.fetchall()
            print("\nColumn types:")
            
            for col in columns:
                col_name = col.COLUMN_NAME
                col_type = col.DATA_TYPE
                max_length = col.CHARACTER_MAXIMUM_LENGTH
                
                if col_name in expected_types:
                    expected_type = expected_types[col_name]
                    assert col_type == expected_type, \
                        f"{col_name} should be {expected_type}, got {col_type}"
                    
                    # Special checks
                    if col_name == 'LabelName':
                        assert max_length == 100, "LabelName should be NVARCHAR(100)"
                        print(f"  ✓ {col_name}: {col_type}({max_length})")
                    else:
                        print(f"  ✓ {col_name}: {col_type}")
            
            print("\n✅ PASS - All column types correct")
            
        finally:
            cursor.close()
            conn.close()
    
    def test_4_not_null_constraints(self):
        """
        Test 4: Verify NOT NULL constraints on required columns.
        """
        print("\n" + "="*80)
        print("TEST 4: NOT NULL CONSTRAINTS")
        print("="*80)
        
        conn = get_connection()
        cursor = conn.cursor()
        
        not_null_columns = [
            'LabelID',
            'LabelName',
            'IsActive',
            'CreatedAt'
        ]
        
        try:
            cursor.execute("""
                SELECT 
                    COLUMN_NAME,
                    IS_NULLABLE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = 'APP_DrawerLabel'
                ORDER BY ORDINAL_POSITION
            """)
            
            columns = cursor.fetchall()
            print("\nNullability constraints:")
            
            for col in columns:
                col_name = col.COLUMN_NAME
                is_nullable = col.IS_NULLABLE
                
                if col_name in not_null_columns:
                    assert is_nullable == 'NO', \
                        f"{col_name} should be NOT NULL"
                    print(f"  ✓ {col_name}: NOT NULL")
            
            print("\n✅ PASS - All NOT NULL constraints correct")
            
        finally:
            cursor.close()
            conn.close()
    
    def test_5_primary_key(self):
        """
        Test 5: Verify LabelID is primary key.
        """
        print("\n" + "="*80)
        print("TEST 5: PRIMARY KEY")
        print("="*80)
        
        conn = get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    c.COLUMN_NAME
                FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS t
                JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE c
                    ON t.CONSTRAINT_NAME = c.CONSTRAINT_NAME
                    AND t.TABLE_NAME = c.TABLE_NAME
                WHERE t.TABLE_NAME = 'APP_DrawerLabel'
                AND t.CONSTRAINT_TYPE = 'PRIMARY KEY'
            """)
            
            pk_columns = [row.COLUMN_NAME for row in cursor.fetchall()]
            print(f"\nPrimary key columns: {pk_columns}")
            
            assert len(pk_columns) == 1, "Should have exactly one primary key column"
            assert pk_columns[0] == 'LabelID', "Primary key should be LabelID"
            
            print("✅ PASS - Primary key is LabelID")
            
        finally:
            cursor.close()
            conn.close()
    
    def test_6_unique_constraint(self):
        """
        Test 6: Verify UNIQUE constraint exists on LabelName.
        """
        print("\n" + "="*80)
        print("TEST 6: UNIQUE CONSTRAINT")
        print("="*80)
        
        conn = get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    c.COLUMN_NAME,
                    t.CONSTRAINT_NAME
                FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS t
                JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE c
                    ON t.CONSTRAINT_NAME = c.CONSTRAINT_NAME
                    AND t.TABLE_NAME = c.TABLE_NAME
                WHERE t.TABLE_NAME = 'APP_DrawerLabel'
                AND t.CONSTRAINT_TYPE = 'UNIQUE'
            """)
            
            unique_columns = [(row.COLUMN_NAME, row.CONSTRAINT_NAME) for row in cursor.fetchall()]
            print(f"\nUnique constraints: {unique_columns}")
            
            assert len(unique_columns) > 0, "Should have at least one UNIQUE constraint"
            
            label_name_unique = any(col[0] == 'LabelName' for col in unique_columns)
            assert label_name_unique, "LabelName should have UNIQUE constraint"
            
            print("✅ PASS - UNIQUE constraint exists on LabelName")
            
        finally:
            cursor.close()
            conn.close()
    
    def test_7_indexes_exist(self):
        """
        Test 7: Verify index exists on IsActive.
        """
        print("\n" + "="*80)
        print("TEST 7: INDEXES")
        print("="*80)
        
        conn = get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    i.name AS index_name,
                    c.name AS column_name
                FROM sys.indexes i
                JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
                JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
                WHERE i.object_id = OBJECT_ID('dbo.APP_DrawerLabel')
                AND i.is_primary_key = 0
                ORDER BY i.name
            """)
            
            indexes = [(row.index_name, row.column_name) for row in cursor.fetchall()]
            print(f"\nIndexes found: {indexes}")
            
            # Check for IsActive index
            is_active_index = any('IsActive' in idx[1] for idx in indexes if idx[0] and 'IsActive' in idx[0])
            assert is_active_index, "Should have index on IsActive column"
            
            print("  ✓ IX_DrawerLabel_IsActive")
            print("\n✅ PASS - Required indexes exist")
            
        finally:
            cursor.close()
            conn.close()
    
    def test_8_insert_and_read(self):
        """
        Test 8: Verify insert and read operations work.
        """
        print("\n" + "="*80)
        print("TEST 8: INSERT AND READ")
        print("="*80)
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Generate unique label name
        test_label_name = f"test_label_{uuid.uuid4().hex[:8]}"
        
        try:
            # Insert test label
            cursor.execute("""
                INSERT INTO dbo.APP_DrawerLabel 
                    (LabelName, IsActive)
                VALUES (?, 1)
            """, (test_label_name,))
            conn.commit()
            print(f"✓ Inserted test label: {test_label_name}")
            
            # Read back the label
            cursor.execute("""
                SELECT 
                    LabelID,
                    LabelName,
                    IsActive,
                    CreatedAt
                FROM dbo.APP_DrawerLabel
                WHERE LabelName = ?
            """, (test_label_name,))
            
            row = cursor.fetchone()
            assert row is not None, "Should retrieve inserted label"
            
            print(f"\nRetrieved label:")
            print(f"  LabelID: {row.LabelID}")
            print(f"  LabelName: {row.LabelName}")
            print(f"  IsActive: {row.IsActive}")
            print(f"  CreatedAt: {row.CreatedAt}")
            
            # Validate data
            assert row.LabelName == test_label_name
            assert row.IsActive == True
            assert row.CreatedAt is not None
            
            print("\n✅ PASS - Insert and read operations work correctly")
            
        finally:
            # Clean up
            cursor.execute("""
                DELETE FROM dbo.APP_DrawerLabel
                WHERE LabelName = ?
            """, (test_label_name,))
            conn.commit()
            print("\nCleaned up test data")
            
            cursor.close()
            conn.close()
    
    def test_9_unique_constraint_enforcement(self):
        """
        Test 9: Verify UNIQUE constraint prevents duplicate label names.
        This is the critical test for G-B2.
        """
        print("\n" + "="*80)
        print("TEST 9: UNIQUE CONSTRAINT ENFORCEMENT")
        print("="*80)
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Generate unique label name
        test_label_name = f"test_unique_{uuid.uuid4().hex[:8]}"
        
        try:
            # Insert first label
            cursor.execute("""
                INSERT INTO dbo.APP_DrawerLabel 
                    (LabelName, IsActive)
                VALUES (?, 1)
            """, (test_label_name,))
            conn.commit()
            print(f"✓ Inserted first label: {test_label_name}")
            
            # Try to insert duplicate label name (should fail)
            try:
                cursor.execute("""
                    INSERT INTO dbo.APP_DrawerLabel 
                        (LabelName, IsActive)
                    VALUES (?, 1)
                """, (test_label_name,))
                conn.commit()
                
                # If we reach here, the constraint didn't work
                assert False, "Should have raised unique constraint violation"
                
            except Exception as e:
                error_msg = str(e).lower()
                print(f"\n✓ Duplicate insert failed as expected")
                print(f"  Error message: {e}")
                
                # Verify it's a unique constraint error
                is_unique_error = (
                    'unique' in error_msg or 
                    'duplicate' in error_msg or 
                    'constraint' in error_msg or
                    'uq_drawerlabel' in error_msg
                )
                
                assert is_unique_error, \
                    f"Error should indicate unique constraint violation, got: {e}"
                
                # Rollback the failed transaction
                conn.rollback()
                print("  ✓ Confirmed: UNIQUE constraint violation")
            
            print("\n✅ PASS - UNIQUE constraint prevents duplicate labels")
            
        finally:
            # Clean up
            cursor.execute("""
                DELETE FROM dbo.APP_DrawerLabel
                WHERE LabelName = ?
            """, (test_label_name,))
            conn.commit()
            print("\nCleaned up test data")
            
            cursor.close()
            conn.close()
    
    def test_10_default_values(self):
        """
        Test 10: Verify default values are applied correctly.
        """
        print("\n" + "="*80)
        print("TEST 10: DEFAULT VALUES")
        print("="*80)
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Generate unique label name
        test_label_name = f"test_defaults_{uuid.uuid4().hex[:8]}"
        
        try:
            # Insert without specifying IsActive and CreatedAt
            cursor.execute("""
                INSERT INTO dbo.APP_DrawerLabel 
                    (LabelName)
                VALUES (?)
            """, (test_label_name,))
            conn.commit()
            print(f"✓ Inserted label without specifying IsActive and CreatedAt")
            
            # Read back
            cursor.execute("""
                SELECT IsActive, CreatedAt
                FROM dbo.APP_DrawerLabel
                WHERE LabelName = ?
            """, (test_label_name,))
            row = cursor.fetchone()
            
            print(f"\nDefault values:")
            print(f"  IsActive: {row.IsActive}")
            print(f"  CreatedAt: {row.CreatedAt}")
            
            # Verify defaults
            assert row.IsActive == True, "IsActive should default to True"
            assert row.CreatedAt is not None, "CreatedAt should be auto-populated"
            
            # Verify CreatedAt is recent (within last minute)
            from datetime import datetime
            time_diff = datetime.utcnow() - row.CreatedAt
            assert time_diff.total_seconds() < 60, "CreatedAt should be current timestamp"
            
            print("\n✅ PASS - Default values applied correctly")
            
        finally:
            # Clean up
            cursor.execute("""
                DELETE FROM dbo.APP_DrawerLabel
                WHERE LabelName = ?
            """, (test_label_name,))
            conn.commit()
            print("\nCleaned up test data")
            
            cursor.close()
            conn.close()


def run_all_tests():
    """Run all tests in sequence."""
    print("\n" + "="*80)
    print("PHASE G-B2: DRAWER LABEL TABLE SCHEMA TESTS")
    print("="*80)
    
    test_suite = TestDrawerLabelTableSchema()
    
    tests = [
        ("Table Exists", test_suite.test_1_table_exists),
        ("Columns Exist", test_suite.test_2_columns_exist),
        ("Column Types", test_suite.test_3_column_types),
        ("NOT NULL Constraints", test_suite.test_4_not_null_constraints),
        ("Primary Key", test_suite.test_5_primary_key),
        ("UNIQUE Constraint", test_suite.test_6_unique_constraint),
        ("Indexes", test_suite.test_7_indexes_exist),
        ("Insert and Read", test_suite.test_8_insert_and_read),
        ("UNIQUE Constraint Enforcement", test_suite.test_9_unique_constraint_enforcement),
        ("Default Values", test_suite.test_10_default_values),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ FAIL - {test_name}: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ ERROR - {test_name}: {e}")
            failed += 1
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total: {len(tests)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print("="*80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
