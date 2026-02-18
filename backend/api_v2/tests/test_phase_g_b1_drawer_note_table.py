"""
Test Suite: Phase G-B1 - Drawer Note Table Schema
Tests the APP_DrawerNote table structure and basic operations.

Verifies:
- Table exists
- All columns exist with correct types
- NOT NULL constraints
- Indexes exist
- Insert/read operations work
- Soft delete functionality

Target: backend/database_migrations/phase_g_b1_create_drawer_note_table.sql

Test Coverage:
- Schema validation
- Column types and constraints
- Index presence
- CRUD operations
- Data integrity

Note: Uses real database connection (no mocks)
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

from core.database import get_connection


class TestDrawerNoteTableSchema:
    """Test suite for G-B1 Drawer Note table schema validation."""
    
    def test_1_table_exists(self):
        """
        Test 1: Verify APP_DrawerNote table exists in database.
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
                AND TABLE_NAME = 'APP_DrawerNote'
            """)
            result = cursor.fetchone()
            table_count = result.table_count
            
            print(f"Table count: {table_count}")
            assert table_count == 1, "APP_DrawerNote table should exist"
            print("✅ PASS - APP_DrawerNote table exists")
            
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
            'NoteID',
            'NoteText',
            'CreatedAt',
            'CreatedByUserID',
            'CreatedByName',
            'IsDeleted'
        ]
        
        try:
            cursor.execute("""
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = 'APP_DrawerNote'
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
            'NoteID': 'int',
            'NoteText': 'nvarchar',
            'CreatedAt': 'datetime2',
            'CreatedByUserID': 'int',
            'CreatedByName': 'nvarchar',
            'IsDeleted': 'bit'
        }
        
        try:
            cursor.execute("""
                SELECT 
                    COLUMN_NAME,
                    DATA_TYPE,
                    CHARACTER_MAXIMUM_LENGTH
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = 'APP_DrawerNote'
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
                    if col_name == 'NoteText':
                        assert max_length == -1, "NoteText should be NVARCHAR(MAX)"
                        print(f"  ✓ {col_name}: {col_type}(MAX)")
                    elif col_name == 'CreatedByName':
                        assert max_length == 200, "CreatedByName should be NVARCHAR(200)"
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
            'NoteID',
            'NoteText',
            'CreatedAt',
            'CreatedByUserID',
            'CreatedByName',
            'IsDeleted'
        ]
        
        try:
            cursor.execute("""
                SELECT 
                    COLUMN_NAME,
                    IS_NULLABLE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = 'APP_DrawerNote'
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
        Test 5: Verify NoteID is primary key.
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
                WHERE t.TABLE_NAME = 'APP_DrawerNote'
                AND t.CONSTRAINT_TYPE = 'PRIMARY KEY'
            """)
            
            pk_columns = [row.COLUMN_NAME for row in cursor.fetchall()]
            print(f"\nPrimary key columns: {pk_columns}")
            
            assert len(pk_columns) == 1, "Should have exactly one primary key column"
            assert pk_columns[0] == 'NoteID', "Primary key should be NoteID"
            
            print("✅ PASS - Primary key is NoteID")
            
        finally:
            cursor.close()
            conn.close()
    
    def test_6_indexes_exist(self):
        """
        Test 6: Verify indexes exist for performance.
        """
        print("\n" + "="*80)
        print("TEST 6: INDEXES")
        print("="*80)
        
        conn = get_connection()
        cursor = conn.cursor()
        
        expected_indexes = [
            'IX_DrawerNote_CreatedAt',
            'IX_DrawerNote_CreatedByUserID',
            'IX_DrawerNote_IsDeleted'
        ]
        
        try:
            cursor.execute("""
                SELECT 
                    i.name AS index_name,
                    c.name AS column_name
                FROM sys.indexes i
                JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
                JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
                WHERE i.object_id = OBJECT_ID('dbo.APP_DrawerNote')
                AND i.is_primary_key = 0
                ORDER BY i.name
            """)
            
            actual_indexes = [row.index_name for row in cursor.fetchall()]
            print(f"\nActual indexes: {set(actual_indexes)}")
            print(f"Expected indexes: {expected_indexes}")
            
            for idx in expected_indexes:
                assert idx in actual_indexes, f"Index '{idx}' is missing"
                print(f"  ✓ {idx}")
            
            print("\n✅ PASS - All required indexes exist")
            
        finally:
            cursor.close()
            conn.close()
    
    def test_7_insert_and_read(self):
        """
        Test 7: Verify insert and read operations work.
        """
        print("\n" + "="*80)
        print("TEST 7: INSERT AND READ")
        print("="*80)
        
        conn = get_connection()
        cursor = conn.cursor()
        
        test_note_text = "Test drawer note for G-B1 validation"
        test_user_id = 1
        test_user_name = "test_user"
        
        try:
            # Clean up any previous test notes
            cursor.execute("""
                DELETE FROM dbo.APP_DrawerNote
                WHERE CreatedByName = ?
            """, (test_user_name,))
            conn.commit()
            print("Cleaned up previous test notes")
            
            # Insert test note
            cursor.execute("""
                INSERT INTO dbo.APP_DrawerNote 
                    (NoteText, CreatedByUserID, CreatedByName, IsDeleted)
                VALUES (?, ?, ?, 0)
            """, (test_note_text, test_user_id, test_user_name))
            conn.commit()
            print(f"✓ Inserted test note")
            
            # Read back the note
            cursor.execute("""
                SELECT 
                    NoteID,
                    NoteText,
                    CreatedAt,
                    CreatedByUserID,
                    CreatedByName,
                    IsDeleted
                FROM dbo.APP_DrawerNote
                WHERE CreatedByName = ?
            """, (test_user_name,))
            
            row = cursor.fetchone()
            assert row is not None, "Should retrieve inserted note"
            
            print(f"\nRetrieved note:")
            print(f"  NoteID: {row.NoteID}")
            print(f"  NoteText: {row.NoteText}")
            print(f"  CreatedAt: {row.CreatedAt}")
            print(f"  CreatedByUserID: {row.CreatedByUserID}")
            print(f"  CreatedByName: {row.CreatedByName}")
            print(f"  IsDeleted: {row.IsDeleted}")
            
            # Validate data
            assert row.NoteText == test_note_text
            assert row.CreatedByUserID == test_user_id
            assert row.CreatedByName == test_user_name
            assert row.IsDeleted == False
            assert row.CreatedAt is not None
            
            print("\n✅ PASS - Insert and read operations work correctly")
            
        finally:
            # Clean up
            cursor.execute("""
                DELETE FROM dbo.APP_DrawerNote
                WHERE CreatedByName = ?
            """, (test_user_name,))
            conn.commit()
            print("\nCleaned up test data")
            
            cursor.close()
            conn.close()
    
    def test_8_soft_delete(self):
        """
        Test 8: Verify soft delete functionality.
        """
        print("\n" + "="*80)
        print("TEST 8: SOFT DELETE")
        print("="*80)
        
        conn = get_connection()
        cursor = conn.cursor()
        
        test_note_text = "Test note for soft delete"
        test_user_id = 1
        test_user_name = "test_soft_delete"
        
        try:
            # Insert test note
            cursor.execute("""
                INSERT INTO dbo.APP_DrawerNote 
                    (NoteText, CreatedByUserID, CreatedByName, IsDeleted)
                VALUES (?, ?, ?, 0)
            """, (test_note_text, test_user_id, test_user_name))
            conn.commit()
            
            # Get note ID
            cursor.execute("""
                SELECT NoteID, IsDeleted
                FROM dbo.APP_DrawerNote
                WHERE CreatedByName = ?
            """, (test_user_name,))
            row = cursor.fetchone()
            note_id = row.NoteID
            
            print(f"Created note with ID: {note_id}")
            print(f"Initial IsDeleted: {row.IsDeleted}")
            assert row.IsDeleted == False, "New note should not be deleted"
            
            # Soft delete the note
            cursor.execute("""
                UPDATE dbo.APP_DrawerNote
                SET IsDeleted = 1
                WHERE NoteID = ?
            """, (note_id,))
            conn.commit()
            print("✓ Updated IsDeleted to 1")
            
            # Verify soft delete
            cursor.execute("""
                SELECT IsDeleted
                FROM dbo.APP_DrawerNote
                WHERE NoteID = ?
            """, (note_id,))
            row = cursor.fetchone()
            
            print(f"After soft delete IsDeleted: {row.IsDeleted}")
            assert row.IsDeleted == True, "Note should be marked as deleted"
            
            print("\n✅ PASS - Soft delete functionality works correctly")
            
        finally:
            # Clean up (hard delete for test cleanup)
            cursor.execute("""
                DELETE FROM dbo.APP_DrawerNote
                WHERE CreatedByName = ?
            """, (test_user_name,))
            conn.commit()
            print("\nCleaned up test data")
            
            cursor.close()
            conn.close()
    
    def test_9_default_values(self):
        """
        Test 9: Verify default values are applied correctly.
        """
        print("\n" + "="*80)
        print("TEST 9: DEFAULT VALUES")
        print("="*80)
        
        conn = get_connection()
        cursor = conn.cursor()
        
        test_note_text = "Test note for default values"
        test_user_id = 1
        test_user_name = "test_defaults"
        
        try:
            # Insert without specifying IsDeleted and CreatedAt
            cursor.execute("""
                INSERT INTO dbo.APP_DrawerNote 
                    (NoteText, CreatedByUserID, CreatedByName)
                VALUES (?, ?, ?)
            """, (test_note_text, test_user_id, test_user_name))
            conn.commit()
            print("✓ Inserted note without specifying IsDeleted and CreatedAt")
            
            # Read back
            cursor.execute("""
                SELECT IsDeleted, CreatedAt
                FROM dbo.APP_DrawerNote
                WHERE CreatedByName = ?
            """, (test_user_name,))
            row = cursor.fetchone()
            
            print(f"\nDefault values:")
            print(f"  IsDeleted: {row.IsDeleted}")
            print(f"  CreatedAt: {row.CreatedAt}")
            
            # Verify defaults
            assert row.IsDeleted == False, "IsDeleted should default to False"
            assert row.CreatedAt is not None, "CreatedAt should be auto-populated"
            
            # Verify CreatedAt is recent (within last minute)
            time_diff = datetime.utcnow() - row.CreatedAt
            assert time_diff.total_seconds() < 60, "CreatedAt should be current timestamp"
            
            print("\n✅ PASS - Default values applied correctly")
            
        finally:
            # Clean up
            cursor.execute("""
                DELETE FROM dbo.APP_DrawerNote
                WHERE CreatedByName = ?
            """, (test_user_name,))
            conn.commit()
            print("\nCleaned up test data")
            
            cursor.close()
            conn.close()


def run_all_tests():
    """Run all tests in sequence."""
    print("\n" + "="*80)
    print("PHASE G-B1: DRAWER NOTE TABLE SCHEMA TESTS")
    print("="*80)
    
    test_suite = TestDrawerNoteTableSchema()
    
    tests = [
        ("Table Exists", test_suite.test_1_table_exists),
        ("Columns Exist", test_suite.test_2_columns_exist),
        ("Column Types", test_suite.test_3_column_types),
        ("NOT NULL Constraints", test_suite.test_4_not_null_constraints),
        ("Primary Key", test_suite.test_5_primary_key),
        ("Indexes", test_suite.test_6_indexes_exist),
        ("Insert and Read", test_suite.test_7_insert_and_read),
        ("Soft Delete", test_suite.test_8_soft_delete),
        ("Default Values", test_suite.test_9_default_values),
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
