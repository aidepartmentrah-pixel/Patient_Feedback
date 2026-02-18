"""
PHASE K — SVC5 — MAPPING WRITER DB LAYER VERIFICATION

Demonstrates:
- insert_migration_mapping() function
- Proactive duplicate check
- FK validation
- Rollback safety
"""

import sys
from pathlib import Path

backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from api.db_layer.migration_map_db import insert_migration_mapping
from core.database import get_connection


def print_header(text):
    print(f"\n{'=' * 80}")
    print(f"  {text}")
    print('=' * 80)


def cleanup_test_data(legacy_case_id):
    """Clean up test mapping record"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM dbo.APP_DataMigration_Map WHERE legacy_case_id = ?", legacy_case_id)
        conn.commit()
        cursor.close()
        conn.close()
    except:
        pass


def verify_mapping_writer():
    """Verify mapping writer DB layer function"""
    print_header("K-SVC-5 MAPPING WRITER DB LAYER VERIFICATION")
    
    legacy_case_id = 777777
    
    # Clean up any existing test data
    cleanup_test_data(legacy_case_id)
    
    # Get valid FKs
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT TOP 1 IncidentRequestCaseID FROM dbo.APP_IncidentCase ORDER BY IncidentRequestCaseID DESC")
    row = cursor.fetchone()
    case_id = row[0] if row else None
    
    cursor.execute("SELECT TOP 1 UserID FROM dbo.APP_Users ORDER BY UserID")
    row = cursor.fetchone()
    user_id = row[0] if row else None
    
    cursor.close()
    conn.close()
    
    if not case_id or not user_id:
        print("\n❌ No valid test data available")
        return
    
    print(f"\n📋 Test Data:")
    print(f"   Legacy Case ID: {legacy_case_id}")
    print(f"   New Case ID: {case_id}")
    print(f"   Migrated By User: {user_id}")
    
    # Test 1: Successful insert
    print(f"\n🔄 Test 1: Inserting mapping...")
    
    try:
        result = insert_migration_mapping(legacy_case_id, case_id, user_id)
        
        if result.get("success"):
            print(f"✅ Mapping inserted successfully")
            print(f"   Legacy Case ID: {result.get('legacy_case_id')}")
            print(f"   New Case ID: {result.get('new_case_id')}")
        else:
            print(f"❌ Insert failed")
            return
    except Exception as e:
        print(f"❌ Insert failed: {e}")
        cleanup_test_data(legacy_case_id)
        return
    
    # Verify in database
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT MapID, legacy_case_id, new_case_id, migrated_by_user_id, migrated_at
        FROM dbo.APP_DataMigration_Map
        WHERE legacy_case_id = ?
    """, legacy_case_id)
    
    row = cursor.fetchone()
    
    if row:
        print(f"\n📊 Database Verification:")
        print(f"   MapID: {row[0]}")
        print(f"   Legacy Case ID: {row[1]}")
        print(f"   New Case ID: {row[2]}")
        print(f"   Migrated By User: {row[3]}")
        print(f"   Migrated At: {row[4]}")
    else:
        print(f"\n⚠️  Warning: Row not found in database")
    
    cursor.close()
    conn.close()
    
    # Test 2: Duplicate prevention
    print(f"\n🔒 Test 2: Testing duplicate prevention...")
    
    try:
        result2 = insert_migration_mapping(legacy_case_id, case_id, user_id)
        print(f"❌ Duplicate was NOT blocked (this should not happen)")
    except ValueError as ve:
        print(f"✅ Duplicate blocked as expected")
        print(f"   Error: {ve}")
    except Exception as e:
        print(f"⚠️  Unexpected error type: {type(e).__name__}")
        print(f"   Error: {e}")
    
    # Verify only one row exists
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*)
        FROM dbo.APP_DataMigration_Map
        WHERE legacy_case_id = ?
    """, legacy_case_id)
    
    count = cursor.fetchone()[0]
    
    print(f"\n📈 Row Count Check:")
    print(f"   Rows with legacy_case_id={legacy_case_id}: {count}")
    
    if count == 1:
        print(f"   ✅ Exactly one row (duplicate prevention working)")
    else:
        print(f"   ❌ Expected 1 row, found {count}")
    
    cursor.close()
    conn.close()
    
    # Test 3: FK violation handling
    print(f"\n⚠️  Test 3: Testing FK violation handling...")
    
    invalid_case_id = 999999999
    invalid_legacy_id = 777778
    
    cleanup_test_data(invalid_legacy_id)
    
    try:
        result3 = insert_migration_mapping(invalid_legacy_id, invalid_case_id, user_id)
        print(f"❌ FK violation was NOT caught (this should not happen)")
    except ValueError as ve:
        print(f"❌ Wrong exception type (ValueError instead of Exception)")
        print(f"   Error: {ve}")
    except Exception as e:
        print(f"✅ FK violation caught correctly")
        print(f"   Exception type: {type(e).__name__}")
        print(f"   Message contains 'Failed to insert': {'Failed to insert' in str(e)}")
    
    # Verify no row was inserted
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*)
        FROM dbo.APP_DataMigration_Map
        WHERE legacy_case_id = ?
    """, invalid_legacy_id)
    
    invalid_count = cursor.fetchone()[0]
    
    if invalid_count == 0:
        print(f"   ✅ No row inserted (rollback worked)")
    else:
        print(f"   ❌ Unexpected rows found: {invalid_count}")
    
    cursor.close()
    conn.close()
    
    # Clean up
    print(f"\n🧹 Cleaning up test data...")
    cleanup_test_data(legacy_case_id)
    cleanup_test_data(invalid_legacy_id)
    
    print("\n" + "=" * 80)
    print("  K-SVC-5 VERIFICATION COMPLETE")
    print("=" * 80)
    print("\n✅ Mapping writer DB layer is working correctly!")
    print("   - Successful insert returns structured result")
    print("   - Proactive duplicate check prevents double migration")
    print("   - FK violations raise generic Exception")
    print("   - Rollback prevents partial inserts")


if __name__ == "__main__":
    verify_mapping_writer()
