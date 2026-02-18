"""
PHASE K — KDB1 — MAPPING TABLE VERIFICATION TEST

This script:
1. Creates the APP_DataMigration_Map table
2. Runs comprehensive verification tests
3. Reports results

WARNING: This will drop and recreate the table if it exists.
"""

import sys
import os
from pathlib import Path
import pyodbc

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from core.database import get_connection


def print_header(text):
    """Print formatted test section header"""
    print(f"\n{'=' * 80}")
    print(f"  {text}")
    print('=' * 80)


def print_test(test_name, passed, message=""):
    """Print test result"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} — {test_name}")
    if message:
        print(f"   {message}")


def execute_migration_script():
    """Execute the table creation script"""
    print_header("STEP 1: CREATE MIGRATION TABLE")
    
    script_path = backend_path / "database_migrations" / "phase_k_db1_create_migration_map_table.sql"
    
    if not script_path.exists():
        print(f"❌ ERROR: Migration script not found at {script_path}")
        return False
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Read the SQL script
        with open(script_path, 'r', encoding='utf-8') as f:
            sql_script = f.read()
        
        # Drop table if exists (for clean test)
        print("Dropping existing table if present...")
        try:
            cursor.execute("DROP TABLE IF EXISTS dbo.APP_DataMigration_Map")
            conn.commit()
            print("✅ Existing table dropped")
        except Exception as e:
            print(f"⚠️  No existing table to drop: {e}")
        
        # Execute the creation script
        print("Executing migration script...")
        
        # Remove comments and PRINT statements
        lines = []
        for line in sql_script.split('\n'):
            stripped = line.strip()
            # Skip comments and PRINT statements
            if (not stripped.startswith('--') and 
                not stripped.startswith('PRINT') and 
                stripped):
                lines.append(line)
        
        clean_sql = '\n'.join(lines)
        
        # Execute the entire script at once
        cursor.execute(clean_sql)
        conn.commit()
        
        print("✅ Migration script executed successfully")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ ERROR executing migration script: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_schema_check():
    """TEST 1: Verify table schema"""
    print_header("TEST 1: SCHEMA CHECK")
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        query = """
        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = 'APP_DataMigration_Map'
        ORDER BY ORDINAL_POSITION
        """
        
        cursor.execute(query)
        columns = cursor.fetchall()
        
        if not columns:
            print_test("Table exists", False, "Table not found")
            cursor.close()
            conn.close()
            return False
        
        print("\nColumns found:")
        for col in columns:
            print(f"  - {col[0]}: {col[1]} (Nullable: {col[2]})")
        
        # Check required columns
        expected_columns = {
            'MapID': 'int',
            'legacy_case_id': 'int',
            'new_case_id': 'int',
            'migrated_by_user_id': 'int',
            'migrated_at': 'datetime2'
        }
        
        found_columns = {col[0]: col[1] for col in columns}
        
        all_passed = True
        for col_name, expected_type in expected_columns.items():
            if col_name not in found_columns:
                print_test(f"Column {col_name}", False, "Missing")
                all_passed = False
            elif found_columns[col_name] != expected_type:
                print_test(f"Column {col_name}", False, f"Wrong type: {found_columns[col_name]} (expected {expected_type})")
                all_passed = False
            else:
                print_test(f"Column {col_name}", True)
        
        cursor.close()
        conn.close()
        return all_passed
        
    except Exception as e:
        print_test("Schema check", False, str(e))
        return False


def test_primary_key():
    """TEST 2: Verify primary key"""
    print_header("TEST 2: PRIMARY KEY CHECK")
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        query = """
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE TABLE_NAME = 'APP_DataMigration_Map'
        AND CONSTRAINT_NAME LIKE 'PK_%'
        """
        
        cursor.execute(query)
        pk_columns = cursor.fetchall()
        
        if not pk_columns:
            print_test("Primary key exists", False)
            cursor.close()
            conn.close()
            return False
        
        pk_col = pk_columns[0][0]
        print_test("Primary key exists", True, f"PK on {pk_col}")
        
        if pk_col == 'MapID':
            print_test("Primary key on MapID", True)
            cursor.close()
            conn.close()
            return True
        else:
            print_test("Primary key on MapID", False, f"PK is on {pk_col}")
            cursor.close()
            conn.close()
            return False
        
    except Exception as e:
        print_test("Primary key check", False, str(e))
        return False


def test_unique_constraint():
    """TEST 3: Verify unique constraint on legacy_case_id"""
    print_header("TEST 3: UNIQUE CONSTRAINT CHECK")
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # First check if constraint exists
        query = """
        SELECT CONSTRAINT_NAME
        FROM INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE
        WHERE TABLE_NAME = 'APP_DataMigration_Map'
        AND COLUMN_NAME = 'legacy_case_id'
        AND CONSTRAINT_NAME LIKE 'UQ_%'
        """
        
        cursor.execute(query)
        constraints = cursor.fetchall()
        
        if not constraints:
            print_test("Unique constraint exists", False)
            cursor.close()
            conn.close()
            return False
        
        print_test("Unique constraint exists", True, f"Constraint: {constraints[0][0]}")
        
        # Test duplicate prevention
        print("\nTesting duplicate prevention...")
        
        # Get a valid case ID and user ID first
        cursor.execute("SELECT TOP 1 IncidentRequestCaseID FROM APP_IncidentCase")
        case_row = cursor.fetchone()
        
        cursor.execute("SELECT TOP 1 UserID FROM APP_Users")
        user_row = cursor.fetchone()
        
        if not case_row or not user_row:
            print_test("Test data available", False, "Missing required test data (case or user)")
            cursor.close()
            conn.close()
            return False
        
        case_id = case_row[0]
        user_id = user_row[0]
        test_legacy_id = 999999
        
        # Clean up any existing test data
        cursor.execute("DELETE FROM APP_DataMigration_Map WHERE legacy_case_id = ?", test_legacy_id)
        conn.commit()
        
        # Insert first record
        cursor.execute("""
            INSERT INTO APP_DataMigration_Map 
            (legacy_case_id, new_case_id, migrated_by_user_id, migrated_at)
            VALUES (?, ?, ?, GETDATE())
        """, test_legacy_id, case_id, user_id)
        conn.commit()
        
        print_test("First insert", True)
        
        # Try to insert duplicate
        duplicate_failed = False
        try:
            cursor.execute("""
                INSERT INTO APP_DataMigration_Map 
                (legacy_case_id, new_case_id, migrated_by_user_id, migrated_at)
                VALUES (?, ?, ?, GETDATE())
            """, test_legacy_id, case_id, user_id)
            conn.commit()
            print_test("Duplicate insert blocked", False, "Duplicate was allowed!")
        except pyodbc.IntegrityError as e:
            duplicate_failed = True
            conn.rollback()
            print_test("Duplicate insert blocked", True, "Unique constraint enforced")
        
        # Clean up
        cursor.execute("DELETE FROM APP_DataMigration_Map WHERE legacy_case_id = ?", test_legacy_id)
        conn.commit()
        
        cursor.close()
        conn.close()
        
        return duplicate_failed
        
    except Exception as e:
        print_test("Unique constraint check", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_foreign_key_case():
    """TEST 4: Verify FK to APP_IncidentCase"""
    print_header("TEST 4: FOREIGN KEY CHECK — CASE")
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Check FK exists
        query = """
        SELECT fk.name
        FROM sys.foreign_keys fk
        INNER JOIN sys.tables t ON fk.parent_object_id = t.object_id
        WHERE t.name = 'APP_DataMigration_Map'
        AND fk.name LIKE '%NewCase%'
        """
        
        cursor.execute(query)
        fk = cursor.fetchone()
        
        if not fk:
            print_test("FK to APP_IncidentCase exists", False)
            cursor.close()
            conn.close()
            return False
        
        print_test("FK to APP_IncidentCase exists", True, f"Constraint: {fk[0]}")
        
        # Test FK violation
        print("\nTesting FK enforcement...")
        
        cursor.execute("SELECT TOP 1 UserID FROM APP_Users")
        user_row = cursor.fetchone()
        
        if not user_row:
            print_test("Test data available", False)
            cursor.close()
            conn.close()
            return False
        
        user_id = user_row[0]
        invalid_case_id = 999999999
        test_legacy_id = 999998
        
        # Clean up
        cursor.execute("DELETE FROM APP_DataMigration_Map WHERE legacy_case_id = ?", test_legacy_id)
        conn.commit()
        
        # Try to insert with invalid case ID
        fk_blocked = False
        try:
            cursor.execute("""
                INSERT INTO APP_DataMigration_Map 
                (legacy_case_id, new_case_id, migrated_by_user_id, migrated_at)
                VALUES (?, ?, ?, GETDATE())
            """, test_legacy_id, invalid_case_id, user_id)
            conn.commit()
            print_test("Invalid case ID blocked", False, "FK violation not enforced")
        except pyodbc.IntegrityError:
            fk_blocked = True
            conn.rollback()
            print_test("Invalid case ID blocked", True, "FK enforced correctly")
        
        cursor.close()
        conn.close()
        
        return fk_blocked
        
    except Exception as e:
        print_test("FK case check", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_foreign_key_user():
    """TEST 5: Verify FK to APP_Users"""
    print_header("TEST 5: FOREIGN KEY CHECK — USER")
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Check FK exists
        query = """
        SELECT fk.name
        FROM sys.foreign_keys fk
        INNER JOIN sys.tables t ON fk.parent_object_id = t.object_id
        WHERE t.name = 'APP_DataMigration_Map'
        AND fk.name LIKE '%User%'
        """
        
        cursor.execute(query)
        fk = cursor.fetchone()
        
        if not fk:
            print_test("FK to APP_Users exists", False)
            cursor.close()
            conn.close()
            return False
        
        print_test("FK to APP_Users exists", True, f"Constraint: {fk[0]}")
        
        # Test FK violation
        print("\nTesting FK enforcement...")
        
        cursor.execute("SELECT TOP 1 IncidentRequestCaseID FROM APP_IncidentCase")
        case_row = cursor.fetchone()
        
        if not case_row:
            print_test("Test data available", False)
            cursor.close()
            conn.close()
            return False
        
        case_id = case_row[0]
        invalid_user_id = 999999999
        test_legacy_id = 999997
        
        # Clean up
        cursor.execute("DELETE FROM APP_DataMigration_Map WHERE legacy_case_id = ?", test_legacy_id)
        conn.commit()
        
        # Try to insert with invalid user ID
        fk_blocked = False
        try:
            cursor.execute("""
                INSERT INTO APP_DataMigration_Map 
                (legacy_case_id, new_case_id, migrated_by_user_id, migrated_at)
                VALUES (?, ?, ?, GETDATE())
            """, test_legacy_id, case_id, invalid_user_id)
            conn.commit()
            print_test("Invalid user ID blocked", False, "FK violation not enforced")
        except pyodbc.IntegrityError:
            fk_blocked = True
            conn.rollback()
            print_test("Invalid user ID blocked", True, "FK enforced correctly")
        
        cursor.close()
        conn.close()
        
        return fk_blocked
        
    except Exception as e:
        print_test("FK user check", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_cascade_safety():
    """TEST 6: Verify no cascade delete"""
    print_header("TEST 6: CASCADE SAFETY CHECK")
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Check FK definitions for NO ACTION or NO CASCADE
        query = """
        SELECT 
            fk.name,
            delete_referential_action_desc
        FROM sys.foreign_keys fk
        INNER JOIN sys.tables t ON fk.parent_object_id = t.object_id
        WHERE t.name = 'APP_DataMigration_Map'
        """
        
        cursor.execute(query)
        fks = cursor.fetchall()
        
        if not fks:
            print_test("Foreign keys found", False)
            cursor.close()
            conn.close()
            return False
        
        all_safe = True
        for fk_name, delete_action in fks:
            is_safe = delete_action in ('NO_ACTION', 'RESTRICT')
            print_test(f"FK {fk_name} cascade safety", is_safe, f"Delete action: {delete_action}")
            if not is_safe:
                all_safe = False
        
        cursor.close()
        conn.close()
        
        return all_safe
        
    except Exception as e:
        print_test("Cascade safety check", False, str(e))
        return False


def test_index_exists():
    """TEST 7: Verify index on legacy_case_id"""
    print_header("TEST 7: INDEX CHECK")
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        query = """
        SELECT name, type_desc
        FROM sys.indexes
        WHERE object_id = OBJECT_ID('dbo.APP_DataMigration_Map')
        AND name = 'IX_APP_DataMigration_Map_Legacy'
        """
        
        cursor.execute(query)
        index = cursor.fetchone()
        
        if not index:
            print_test("Index IX_APP_DataMigration_Map_Legacy exists", False)
            cursor.close()
            conn.close()
            return False
        
        print_test("Index exists", True, f"{index[0]} ({index[1]})")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print_test("Index check", False, str(e))
        return False


def main():
    """Run all tests"""
    print_header("PHASE K — KDB1 — MAPPING TABLE VERIFICATION")
    print("This test will create/recreate the APP_DataMigration_Map table")
    print("and run comprehensive verification tests.")
    
    # Step 1: Create table
    if not execute_migration_script():
        print("\n❌ FAILED: Could not create migration table")
        return False
    
    # Step 2: Run tests
    results = []
    
    results.append(("Schema Check", test_schema_check()))
    results.append(("Primary Key", test_primary_key()))
    results.append(("Unique Constraint", test_unique_constraint()))
    results.append(("FK to Case", test_foreign_key_case()))
    results.append(("FK to User", test_foreign_key_user()))
    results.append(("Cascade Safety", test_cascade_safety()))
    results.append(("Index Exists", test_index_exists()))
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} — {test_name}")
    
    print(f"\n{'=' * 80}")
    print(f"TOTAL: {passed}/{total} tests passed")
    print('=' * 80)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED — K-DB-1 COMPLETE")
        return True
    else:
        print(f"\n❌ {total - passed} TEST(S) FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
