"""
Quick validation test for Force Close feature
Uses existing data to minimize setup complexity
"""
import sys
sys.path.insert(0, 'backend')

from datetime import datetime
from core.database import get_connection

def test_migration_columns():
    """Verify migration schema changes"""
    print("=" * 70)
    print(" TEST: Migration Schema Validation")
    print("=" * 70)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Check subcase columns
        cursor.execute("""
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = 'APP_AdministrativeSubcase'
            AND COLUMN_NAME IN ('ForceClosedAt', 'ForceClosedByUserID', 'ForceCloseReason')
        """)
        subcase_cols = [row[0] for row in cursor.fetchall()]
        
        print(f"Subcase columns: {len(subcase_cols)}/3 found")
        if len(subcase_cols) == 3:
            print("[PASS] All subcase columns exist")
        else:
            print(f"[FAIL] Missing columns: {set(['ForceClosedAt', 'ForceClosedByUserID', 'ForceCloseReason']) - set(subcase_cols)}")
            return False
        
        # Check incident columns
        cursor.execute("""
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = 'APP_IncidentCase'
            AND COLUMN_NAME IN ('ForceClosedAt', 'ForceClosedByUserID', 'ForceCloseReason')
        """)
        incident_cols = [row[0] for row in cursor.fetchall()]
        
        print(f"Incident columns: {len(incident_cols)}/3 found")
        if len(incident_cols) == 3:
            print("[PASS] All incident columns exist")
        else:
            print(f"[FAIL] Missing columns: {set(['ForceClosedAt', 'ForceClosedByUserID', 'ForceCloseReason']) - set(incident_cols)}")
            return False
        
        # Check FK constraints
        cursor.execute("""
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS
            WHERE CONSTRAINT_NAME LIKE 'FK_%ForceClosedBy%'
        """)
        fk_count = cursor.fetchone()[0]
        print(f"Foreign key constraints: {fk_count}")
        if fk_count >= 2:
            print("[PASS] FK constraints exist")
        else:
            print(f"[FAIL] Expected at least 2 FK constraints, found {fk_count}")
            return False
        
        # Check indexes
        cursor.execute("""
            SELECT COUNT(*)
            FROM sys.indexes
            WHERE name LIKE 'IX_%ForceClosedAt'
        """)
        idx_count = cursor.fetchone()[0]
        print(f"Filtered indexes: {idx_count}")
        if idx_count >= 2:
            print("[PASS] Indexes exist")
        else:
            print(f"[FAIL] Expected at least 2 indexes, found {idx_count}")
            return False
        
        return True
        
    finally:
        cursor.close()
        conn.close()

def test_api_endpoint():
    """Check if API endpoint exists"""
    print("\n" + "=" * 70)
    print(" TEST: API Endpoint Validation")
    print("=" * 70)
    
    try:
        # Read the file directly to check
        import os
        router_path = os.path.join('backend', 'api_v2', 'routers', 'workflow_router.py')
        with open(router_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'def force_close_case_and_subcases' in content:
                print("[PASS] Endpoint function exists in router file")
                return True
            else:
                print("[FAIL] Endpoint function not found in router file")
                return False
    except Exception as e:
        print(f"[ERROR] Could not check router filet found. Available functions: {functions[:10]}")
            return False
    except Exception as e:
        print(f"[ERROR] Could not import router: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_service_layer():
    """Check if service function exists"""
    print("\n" + "=" * 70)
    print(" TEST: Service Layer Validation")
    print("=" * 70)
    
    try:
        from api_v2.services.case_response_service import force_close_incident
        
        # Check function signature
        import inspect
        sig = inspect.signature(force_close_incident)
        params = list(sig.parameters.keys())
        
        print(f"Function parameters: {params}")
        
        required_params = ['incident_id', 'reason_text', 'current_user']
        has_all = all(p in params for p in required_params)
        
        if has_all:
            print("[PASS] Service function has correct signature")
            return True
        else:
            print(f"[FAIL] Missing parameters: {set(required_params) - set(params)}")
            return False
    except Exception as e:
        print(f"[ERROR] Could not import service: {e}")
        return False

def test_database_functions():
    """Check if database functions exist"""
    print("\n" + "=" * 70)
    print(" TEST: Database Layer Validation")
    print("=" * 70)
    
    try:
        from api_v2.db_layer import administrative_subcase_db
        from api.db_layer import incident_case
        
        # Check subcase functions
        if hasattr(administrative_subcase_db, 'update_force_close_tracking'):
            print("[PASS] Subcase force_close function exists")
        else:
            print("[FAIL] Subcase force_close function missing")
            return False
        
        # Check incident functions
        if hasattr(incident_case, 'update_force_close_tracking'):
            print("[PASS] Incident force_close function exists")
        else:
            print("[FAIL] Incident force_close function missing")
            return False
        
        return True
    except Exception as e:
        print(f"[ERROR] Could not import DB functions: {e}")
        return False

def main():
    print("\n")
    print("*" * 70)
    print(" FORCE CLOSE FEATURE - QUICK VALIDATION TEST")
    print(" Migration + Code Integration Check")
    print("*" * 70)
    print(f" Test Date: {datetime.now()}")
    print("*" * 70)
    
    results = []
    results.append(("Migration Schema", test_migration_columns()))
    results.append(("Service Layer", test_service_layer()))
    results.append(("Database Functions", test_database_functions()))
    results.append(("API Endpoint", test_api_endpoint()))
    
    print("\n" + "=" * 70)
    print(" TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {test_name}")
    
    print("\n" + "=" * 70)
    print(f" RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        print(" [SUCCESS] ALL VALIDATION TESTS PASSED")
        print(" Migration complete, code integrated successfully")
    else:
        print(" [WARNING] SOME TESTS FAILED - Review required")
    
    print("=" * 70)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
