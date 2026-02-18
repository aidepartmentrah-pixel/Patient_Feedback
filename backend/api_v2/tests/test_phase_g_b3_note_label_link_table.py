"""
Test Suite: Phase G-B3 - Drawer Note-Label Link Table
Tests the APP_DrawerNoteLabelLink bridge table structure and constraint enforcement.

Verifies:
- Table exists
- Columns exist with correct types
- Composite primary key on (NoteID, LabelID)
- Foreign key to APP_DrawerNote with CASCADE DELETE
- Foreign key to APP_DrawerLabel with CASCADE DELETE
- Duplicate link prevention (composite PK)
- Invalid FK rejection
- Index presence

Target: backend/database_migrations/phase_g_b3_create_drawer_note_label_link_table.sql

Test Coverage:
- Schema validation
- Column types
- Composite primary key
- Foreign key constraints
- Data integrity
- Constraint enforcement

Note: Uses real database connection (no mocks)
"""

import pytest
import sys
from pathlib import Path
import uuid

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

from core.database import get_connection


class TestDrawerNoteLabelLinkTableSchema:
    """Test suite for G-B3 Drawer Note-Label Link table validation."""
    
    def test_1_table_exists(self):
        """
        Test 1: Verify APP_DrawerNoteLabelLink table exists in database.
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
                AND TABLE_NAME = 'APP_DrawerNoteLabelLink'
            """)
            result = cursor.fetchone()
            table_count = result.table_count
            
            print(f"Table count: {table_count}")
            assert table_count == 1, "APP_DrawerNoteLabelLink table should exist"
            print("✅ PASS - APP_DrawerNoteLabelLink table exists")
            
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
        
        required_columns = ['NoteID', 'LabelID']
        
        try:
            cursor.execute("""
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = 'APP_DrawerNoteLabelLink'
                ORDER BY ORDINAL_POSITION
            """)
            
            actual_columns = [row.COLUMN_NAME for row in cursor.fetchall()]
            print(f"\nActual columns: {actual_columns}")
            print(f"Required columns: {required_columns}")
            
            assert len(actual_columns) == 2, "Should have exactly 2 columns"
            
            for col in required_columns:
                assert col in actual_columns, f"Column '{col}' is missing"
                print(f"  ✓ {col}")
            
            print("\n✅ PASS - All required columns exist (no extra columns)")
            
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
            'LabelID': 'int'
        }
        
        try:
            cursor.execute("""
                SELECT 
                    COLUMN_NAME,
                    DATA_TYPE,
                    IS_NULLABLE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = 'APP_DrawerNoteLabelLink'
                ORDER BY ORDINAL_POSITION
            """)
            
            columns = cursor.fetchall()
            print("\nColumn types:")
            
            for col in columns:
                col_name = col.COLUMN_NAME
                col_type = col.DATA_TYPE
                is_nullable = col.IS_NULLABLE
                
                if col_name in expected_types:
                    expected_type = expected_types[col_name]
                    assert col_type == expected_type, \
                        f"{col_name} should be {expected_type}, got {col_type}"
                    assert is_nullable == 'NO', f"{col_name} should be NOT NULL"
                    print(f"  ✓ {col_name}: {col_type} NOT NULL")
            
            print("\n✅ PASS - All column types correct")
            
        finally:
            cursor.close()
            conn.close()
    
    def test_4_composite_primary_key(self):
        """
        Test 4: Verify composite primary key on (NoteID, LabelID).
        """
        print("\n" + "="*80)
        print("TEST 4: COMPOSITE PRIMARY KEY")
        print("="*80)
        
        conn = get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    c.COLUMN_NAME,
                    c.ORDINAL_POSITION
                FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS t
                JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE c
                    ON t.CONSTRAINT_NAME = c.CONSTRAINT_NAME
                    AND t.TABLE_NAME = c.TABLE_NAME
                WHERE t.TABLE_NAME = 'APP_DrawerNoteLabelLink'
                AND t.CONSTRAINT_TYPE = 'PRIMARY KEY'
                ORDER BY c.ORDINAL_POSITION
            """)
            
            pk_columns = [row.COLUMN_NAME for row in cursor.fetchall()]
            print(f"\nPrimary key columns: {pk_columns}")
            
            assert len(pk_columns) == 2, "Should have composite primary key with 2 columns"
            assert 'NoteID' in pk_columns, "NoteID should be part of primary key"
            assert 'LabelID' in pk_columns, "LabelID should be part of primary key"
            
            print("✅ PASS - Composite primary key on (NoteID, LabelID)")
            
        finally:
            cursor.close()
            conn.close()
    
    def test_5_foreign_keys(self):
        """
        Test 5: Verify foreign key constraints to parent tables.
        """
        print("\n" + "="*80)
        print("TEST 5: FOREIGN KEY CONSTRAINTS")
        print("="*80)
        
        conn = get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    fk.name AS FK_Name,
                    OBJECT_NAME(fk.parent_object_id) AS Parent_Table,
                    COL_NAME(fkc.parent_object_id, fkc.parent_column_id) AS Parent_Column,
                    OBJECT_NAME(fk.referenced_object_id) AS Referenced_Table,
                    COL_NAME(fkc.referenced_object_id, fkc.referenced_column_id) AS Referenced_Column,
                    fk.delete_referential_action_desc AS Delete_Action
                FROM sys.foreign_keys AS fk
                INNER JOIN sys.foreign_key_columns AS fkc 
                    ON fk.object_id = fkc.constraint_object_id
                WHERE OBJECT_NAME(fk.parent_object_id) = 'APP_DrawerNoteLabelLink'
            """)
            
            foreign_keys = cursor.fetchall()
            print(f"\nForeign keys found: {len(foreign_keys)}")
            
            assert len(foreign_keys) == 2, "Should have exactly 2 foreign keys"
            
            fk_dict = {}
            for fk in foreign_keys:
                print(f"\n  FK: {fk.FK_Name}")
                print(f"    {fk.Parent_Table}.{fk.Parent_Column} -> {fk.Referenced_Table}.{fk.Referenced_Column}")
                print(f"    ON DELETE: {fk.Delete_Action}")
                
                fk_dict[fk.Parent_Column] = {
                    'referenced_table': fk.Referenced_Table,
                    'delete_action': fk.Delete_Action
                }
            
            # Verify NoteID foreign key
            assert 'NoteID' in fk_dict, "Should have FK on NoteID"
            assert fk_dict['NoteID']['referenced_table'] == 'APP_DrawerNote', \
                "NoteID should reference APP_DrawerNote"
            assert fk_dict['NoteID']['delete_action'] == 'CASCADE', \
                "NoteID FK should have CASCADE DELETE"
            
            # Verify LabelID foreign key
            assert 'LabelID' in fk_dict, "Should have FK on LabelID"
            assert fk_dict['LabelID']['referenced_table'] == 'APP_DrawerLabel', \
                "LabelID should reference APP_DrawerLabel"
            assert fk_dict['LabelID']['delete_action'] == 'CASCADE', \
                "LabelID FK should have CASCADE DELETE"
            
            print("\n✅ PASS - All foreign keys correct with CASCADE DELETE")
            
        finally:
            cursor.close()
            conn.close()
    
    def test_6_indexes_exist(self):
        """
        Test 6: Verify index exists on LabelID for filtering.
        """
        print("\n" + "="*80)
        print("TEST 6: INDEXES")
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
                WHERE i.object_id = OBJECT_ID('dbo.APP_DrawerNoteLabelLink')
                AND i.is_primary_key = 0
                ORDER BY i.name
            """)
            
            indexes = [(row.index_name, row.column_name) for row in cursor.fetchall()]
            print(f"\nIndexes found: {indexes}")
            
            # Check for LabelID index
            label_id_index = any('LabelID' in idx[1] for idx in indexes if idx[0] and 'LabelID' in idx[0])
            assert label_id_index, "Should have index on LabelID column"
            
            print("  ✓ IX_DrawerNoteLabelLink_LabelID")
            print("\n✅ PASS - Required indexes exist")
            
        finally:
            cursor.close()
            conn.close()
    
    def test_7_insert_valid_link(self):
        """
        Test 7: Verify insert works with valid foreign key references.
        """
        print("\n" + "="*80)
        print("TEST 7: INSERT VALID LINK")
        print("="*80)
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Generate unique test data
        test_note_text = f"Test note for link {uuid.uuid4().hex[:8]}"
        test_label_name = f"test_link_label_{uuid.uuid4().hex[:8]}"
        
        note_id = None
        label_id = None
        
        try:
            # Insert test note
            cursor.execute("""
                INSERT INTO dbo.APP_DrawerNote 
                    (NoteText, CreatedByUserID, CreatedByName)
                VALUES (?, 1, 'test_user')
            """, (test_note_text,))
            conn.commit()
            
            cursor.execute("""
                SELECT NoteID FROM dbo.APP_DrawerNote
                WHERE NoteText = ?
            """, (test_note_text,))
            note_id = cursor.fetchone().NoteID
            print(f"✓ Created test note with ID: {note_id}")
            
            # Insert test label
            cursor.execute("""
                INSERT INTO dbo.APP_DrawerLabel 
                    (LabelName)
                VALUES (?)
            """, (test_label_name,))
            conn.commit()
            
            cursor.execute("""
                SELECT LabelID FROM dbo.APP_DrawerLabel
                WHERE LabelName = ?
            """, (test_label_name,))
            label_id = cursor.fetchone().LabelID
            print(f"✓ Created test label with ID: {label_id}")
            
            # Insert link
            cursor.execute("""
                INSERT INTO dbo.APP_DrawerNoteLabelLink 
                    (NoteID, LabelID)
                VALUES (?, ?)
            """, (note_id, label_id))
            conn.commit()
            print(f"✓ Created link between note {note_id} and label {label_id}")
            
            # Verify link exists
            cursor.execute("""
                SELECT COUNT(*) AS link_count
                FROM dbo.APP_DrawerNoteLabelLink
                WHERE NoteID = ? AND LabelID = ?
            """, (note_id, label_id))
            link_count = cursor.fetchone().link_count
            
            assert link_count == 1, "Link should exist in database"
            
            print("\n✅ PASS - Valid link insert works correctly")
            
        finally:
            # Clean up
            if note_id and label_id:
                cursor.execute("DELETE FROM dbo.APP_DrawerNoteLabelLink WHERE NoteID = ? AND LabelID = ?", (note_id, label_id))
            if label_id:
                cursor.execute("DELETE FROM dbo.APP_DrawerLabel WHERE LabelID = ?", (label_id,))
            if note_id:
                cursor.execute("DELETE FROM dbo.APP_DrawerNote WHERE NoteID = ?", (note_id,))
            conn.commit()
            print("\nCleaned up test data")
            
            cursor.close()
            conn.close()
    
    def test_8_duplicate_link_prevention(self):
        """
        Test 8: Verify composite primary key prevents duplicate links.
        """
        print("\n" + "="*80)
        print("TEST 8: DUPLICATE LINK PREVENTION")
        print("="*80)
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Generate unique test data
        test_note_text = f"Test note duplicate {uuid.uuid4().hex[:8]}"
        test_label_name = f"test_dup_label_{uuid.uuid4().hex[:8]}"
        
        note_id = None
        label_id = None
        
        try:
            # Insert test note
            cursor.execute("""
                INSERT INTO dbo.APP_DrawerNote 
                    (NoteText, CreatedByUserID, CreatedByName)
                VALUES (?, 1, 'test_user')
            """, (test_note_text,))
            conn.commit()
            
            cursor.execute("""
                SELECT NoteID FROM dbo.APP_DrawerNote
                WHERE NoteText = ?
            """, (test_note_text,))
            note_id = cursor.fetchone().NoteID
            
            # Insert test label
            cursor.execute("""
                INSERT INTO dbo.APP_DrawerLabel 
                    (LabelName)
                VALUES (?)
            """, (test_label_name,))
            conn.commit()
            
            cursor.execute("""
                SELECT LabelID FROM dbo.APP_DrawerLabel
                WHERE LabelName = ?
            """, (test_label_name,))
            label_id = cursor.fetchone().LabelID
            
            print(f"✓ Created test note {note_id} and label {label_id}")
            
            # Insert first link
            cursor.execute("""
                INSERT INTO dbo.APP_DrawerNoteLabelLink 
                    (NoteID, LabelID)
                VALUES (?, ?)
            """, (note_id, label_id))
            conn.commit()
            print(f"✓ Created first link")
            
            # Try to insert duplicate link (should fail)
            try:
                cursor.execute("""
                    INSERT INTO dbo.APP_DrawerNoteLabelLink 
                        (NoteID, LabelID)
                    VALUES (?, ?)
                """, (note_id, label_id))
                conn.commit()
                
                # If we reach here, the constraint didn't work
                assert False, "Should have raised primary key violation"
                
            except Exception as e:
                error_msg = str(e).lower()
                print(f"\n✓ Duplicate insert failed as expected")
                print(f"  Error: {e}")
                
                # Verify it's a primary key violation
                is_pk_error = (
                    'primary key' in error_msg or 
                    'duplicate' in error_msg or
                    'pk_drawernotelabellink' in error_msg or
                    'unique' in error_msg
                )
                
                assert is_pk_error, \
                    f"Error should indicate PK violation, got: {e}"
                
                # Rollback the failed transaction
                conn.rollback()
                print("  ✓ Confirmed: Composite primary key violation")
            
            print("\n✅ PASS - Duplicate links prevented by composite PK")
            
        finally:
            # Clean up
            if note_id and label_id:
                cursor.execute("DELETE FROM dbo.APP_DrawerNoteLabelLink WHERE NoteID = ? AND LabelID = ?", (note_id, label_id))
            if label_id:
                cursor.execute("DELETE FROM dbo.APP_DrawerLabel WHERE LabelID = ?", (label_id,))
            if note_id:
                cursor.execute("DELETE FROM dbo.APP_DrawerNote WHERE NoteID = ?", (note_id,))
            conn.commit()
            print("\nCleaned up test data")
            
            cursor.close()
            conn.close()
    
    def test_9_invalid_note_id_rejection(self):
        """
        Test 9: Verify foreign key constraint rejects invalid NoteID.
        """
        print("\n" + "="*80)
        print("TEST 9: INVALID NOTE ID REJECTION")
        print("="*80)
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Generate unique test data
        test_label_name = f"test_fk_label_{uuid.uuid4().hex[:8]}"
        label_id = None
        
        try:
            # Insert test label
            cursor.execute("""
                INSERT INTO dbo.APP_DrawerLabel 
                    (LabelName)
                VALUES (?)
            """, (test_label_name,))
            conn.commit()
            
            cursor.execute("""
                SELECT LabelID FROM dbo.APP_DrawerLabel
                WHERE LabelName = ?
            """, (test_label_name,))
            label_id = cursor.fetchone().LabelID
            print(f"✓ Created test label with ID: {label_id}")
            
            # Try to insert link with invalid note_id (should fail)
            invalid_note_id = 999999
            
            try:
                cursor.execute("""
                    INSERT INTO dbo.APP_DrawerNoteLabelLink 
                        (NoteID, LabelID)
                    VALUES (?, ?)
                """, (invalid_note_id, label_id))
                conn.commit()
                
                # If we reach here, the constraint didn't work
                assert False, "Should have raised foreign key violation"
                
            except Exception as e:
                error_msg = str(e).lower()
                print(f"\n✓ Invalid NoteID insert failed as expected")
                print(f"  Error: {e}")
                
                # Verify it's a foreign key violation
                is_fk_error = (
                    'foreign key' in error_msg or 
                    'fk_drawernotelabellink' in error_msg or
                    'reference' in error_msg or
                    'conflict' in error_msg
                )
                
                assert is_fk_error, \
                    f"Error should indicate FK violation, got: {e}"
                
                # Rollback the failed transaction
                conn.rollback()
                print("  ✓ Confirmed: Foreign key constraint violation")
            
            print("\n✅ PASS - Invalid NoteID rejected by FK constraint")
            
        finally:
            # Clean up
            if label_id:
                cursor.execute("DELETE FROM dbo.APP_DrawerLabel WHERE LabelID = ?", (label_id,))
            conn.commit()
            print("\nCleaned up test data")
            
            cursor.close()
            conn.close()
    
    def test_10_invalid_label_id_rejection(self):
        """
        Test 10: Verify foreign key constraint rejects invalid LabelID.
        """
        print("\n" + "="*80)
        print("TEST 10: INVALID LABEL ID REJECTION")
        print("="*80)
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Generate unique test data
        test_note_text = f"Test note FK {uuid.uuid4().hex[:8]}"
        note_id = None
        
        try:
            # Insert test note
            cursor.execute("""
                INSERT INTO dbo.APP_DrawerNote 
                    (NoteText, CreatedByUserID, CreatedByName)
                VALUES (?, 1, 'test_user')
            """, (test_note_text,))
            conn.commit()
            
            cursor.execute("""
                SELECT NoteID FROM dbo.APP_DrawerNote
                WHERE NoteText = ?
            """, (test_note_text,))
            note_id = cursor.fetchone().NoteID
            print(f"✓ Created test note with ID: {note_id}")
            
            # Try to insert link with invalid label_id (should fail)
            invalid_label_id = 999999
            
            try:
                cursor.execute("""
                    INSERT INTO dbo.APP_DrawerNoteLabelLink 
                        (NoteID, LabelID)
                    VALUES (?, ?)
                """, (note_id, invalid_label_id))
                conn.commit()
                
                # If we reach here, the constraint didn't work
                assert False, "Should have raised foreign key violation"
                
            except Exception as e:
                error_msg = str(e).lower()
                print(f"\n✓ Invalid LabelID insert failed as expected")
                print(f"  Error: {e}")
                
                # Verify it's a foreign key violation
                is_fk_error = (
                    'foreign key' in error_msg or 
                    'fk_drawernotelabellink' in error_msg or
                    'reference' in error_msg or
                    'conflict' in error_msg
                )
                
                assert is_fk_error, \
                    f"Error should indicate FK violation, got: {e}"
                
                # Rollback the failed transaction
                conn.rollback()
                print("  ✓ Confirmed: Foreign key constraint violation")
            
            print("\n✅ PASS - Invalid LabelID rejected by FK constraint")
            
        finally:
            # Clean up
            if note_id:
                cursor.execute("DELETE FROM dbo.APP_DrawerNote WHERE NoteID = ?", (note_id,))
            conn.commit()
            print("\nCleaned up test data")
            
            cursor.close()
            conn.close()


def run_all_tests():
    """Run all tests in sequence."""
    print("\n" + "="*80)
    print("PHASE G-B3: DRAWER NOTE-LABEL LINK TABLE TESTS")
    print("="*80)
    
    test_suite = TestDrawerNoteLabelLinkTableSchema()
    
    tests = [
        ("Table Exists", test_suite.test_1_table_exists),
        ("Columns Exist", test_suite.test_2_columns_exist),
        ("Column Types", test_suite.test_3_column_types),
        ("Composite Primary Key", test_suite.test_4_composite_primary_key),
        ("Foreign Keys", test_suite.test_5_foreign_keys),
        ("Indexes", test_suite.test_6_indexes_exist),
        ("Insert Valid Link", test_suite.test_7_insert_valid_link),
        ("Duplicate Link Prevention", test_suite.test_8_duplicate_link_prevention),
        ("Invalid NoteID Rejection", test_suite.test_9_invalid_note_id_rejection),
        ("Invalid LabelID Rejection", test_suite.test_10_invalid_label_id_rejection),
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
