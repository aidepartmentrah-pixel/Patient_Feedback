"""
Phase 1 Comprehensive Test: Many-to-Many Employee-Incident Linkage

Tests:
1. Schema verification (ID column, PK, UNIQUE constraint, FK)
2. Same employee can link to MULTIPLE incidents
3. Same incident can have MULTIPLE employees
4. Duplicate (employee, incident) pair is prevented (upsert, not error)
5. get_employees_for_incident returns correct data
6. get_incidents_for_employee returns ALL linked incidents (not just 1)
7. Existing data preserved after migration
8. IsPrimary flag works correctly per incident
"""
import sys
sys.path.insert(0, '.')
from backend.core.database import get_connection
from backend.api.db_layer.incident_case_employee import (
    add_employee_to_case,
    get_employees_for_incident,
    get_incidents_for_employee
)

PASS = 0
FAIL = 0
TESTS = []

def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        status = "PASS"
    else:
        FAIL += 1
        status = "FAIL"
    TESTS.append((name, status, detail))
    icon = "✅" if condition else "❌"
    print(f"  {icon} {name}" + (f" — {detail}" if detail else ""))


def run_all_tests():
    global PASS, FAIL
    
    print("=" * 70)
    print("PHASE 1 COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # ============================================================
    # TEST GROUP 1: SCHEMA VERIFICATION
    # ============================================================
    print("\n--- GROUP 1: SCHEMA VERIFICATION ---")
    
    # Test 1.1: ID column exists and is IDENTITY
    cursor.execute("""
        SELECT c.name, c.is_identity 
        FROM sys.columns c
        WHERE c.object_id = OBJECT_ID('dbo.APP_IncidentCaseEmployee') AND c.name = 'ID'
    """)
    row = cursor.fetchone()
    test("1.1 ID column exists", row is not None)
    test("1.2 ID column is IDENTITY", row.is_identity == True if row else False)
    
    # Test 1.3: PK is on ID (not EmployeeID)
    cursor.execute("""
        SELECT c.name AS pk_column
        FROM sys.key_constraints kc
        JOIN sys.index_columns ic ON kc.unique_index_id = ic.index_id AND kc.parent_object_id = ic.object_id
        JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
        WHERE kc.parent_object_id = OBJECT_ID('dbo.APP_IncidentCaseEmployee') AND kc.type = 'PK'
    """)
    pk_row = cursor.fetchone()
    test("1.3 PK is on ID column", pk_row and pk_row.pk_column == 'ID', 
         f"PK column: {pk_row.pk_column if pk_row else 'NONE'}")
    
    # Test 1.4: UNIQUE constraint on (EmployeeID, IncidentRequestCaseID)
    cursor.execute("""
        SELECT i.name, STRING_AGG(c.name, ', ') WITHIN GROUP (ORDER BY ic.key_ordinal) AS columns
        FROM sys.indexes i
        JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
        JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
        WHERE i.object_id = OBJECT_ID('dbo.APP_IncidentCaseEmployee')
            AND i.is_unique = 1
            AND i.is_primary_key = 0
        GROUP BY i.name
    """)
    uq_row = cursor.fetchone()
    test("1.4 UNIQUE constraint exists on (EmployeeID, IncidentRequestCaseID)", 
         uq_row and 'EmployeeID' in uq_row.columns and 'IncidentRequestCaseID' in uq_row.columns,
         f"Unique cols: {uq_row.columns if uq_row else 'NONE'}")
    
    # Test 1.5: FK still intact
    cursor.execute("""
        SELECT fk.name
        FROM sys.foreign_keys fk
        WHERE fk.parent_object_id = OBJECT_ID('dbo.APP_IncidentCaseEmployee')
    """)
    fk_row = cursor.fetchone()
    test("1.5 FK to APP_IncidentCase preserved", fk_row is not None,
         f"FK: {fk_row.name if fk_row else 'NONE'}")
    
    # ============================================================
    # TEST GROUP 2: EXISTING DATA PRESERVED
    # ============================================================
    print("\n--- GROUP 2: EXISTING DATA PRESERVED ---")
    
    cursor.execute("SELECT COUNT(*) FROM dbo.APP_IncidentCaseEmployee")
    count = cursor.fetchone()[0]
    test("2.1 Existing data preserved (4 rows)", count >= 4, f"Count: {count}")
    
    # Check incident 491 employees still there
    cursor.execute("""
        SELECT EmployeeID FROM dbo.APP_IncidentCaseEmployee 
        WHERE IncidentRequestCaseID = 491 ORDER BY EmployeeID
    """)
    emp_ids_491 = [r.EmployeeID for r in cursor.fetchall()]
    test("2.2 Incident 491 still has employees 1,2", emp_ids_491 == [1, 2],
         f"Employee IDs: {emp_ids_491}")
    
    # Check incident 490 employees still there
    cursor.execute("""
        SELECT EmployeeID FROM dbo.APP_IncidentCaseEmployee 
        WHERE IncidentRequestCaseID = 490 ORDER BY EmployeeID
    """)
    emp_ids_490 = [r.EmployeeID for r in cursor.fetchall()]
    test("2.3 Incident 490 still has employees 101,102", emp_ids_490 == [101, 102],
         f"Employee IDs: {emp_ids_490}")
    
    cursor.close()
    conn.close()
    
    # ============================================================
    # TEST GROUP 3: MANY-TO-MANY LINKAGE VIA DB LAYER
    # ============================================================
    print("\n--- GROUP 3: MANY-TO-MANY LINKAGE (using db_layer functions) ---")
    
    # Test 3.1: Link employee 1 to ANOTHER incident (491 already linked, now also 490)
    # Employee 1 should now appear in BOTH incident 491 AND 490
    try:
        result = add_employee_to_case(
            incident_id=490,
            employee_id=1,
            assigned_by_user_id=1,
            full_name="دينا كمال رقم 1",
            is_primary=False
        )
        test("3.1 Employee 1 linked to incident 490 (was already in 491)", result > 0,
             f"Returned ID: {result}")
    except Exception as e:
        test("3.1 Employee 1 linked to incident 490", False, f"Error: {e}")
    
    # Test 3.2: Verify employee 1 is now in BOTH incidents
    try:
        incidents_for_emp1 = get_incidents_for_employee(1)
        test("3.2 Employee 1 linked to multiple incidents", len(incidents_for_emp1) >= 2,
             f"Incidents: {incidents_for_emp1}")
    except Exception as e:
        test("3.2 Employee 1 linked to multiple incidents", False, f"Error: {e}")
    
    # Test 3.3: Verify get_employees_for_incident still works for 490
    try:
        emps_490 = get_employees_for_incident(490)
        emp_ids = [e['employee_id'] for e in emps_490]
        test("3.3 Incident 490 now has employee 1 too", 1 in emp_ids,
             f"Employee IDs in 490: {emp_ids}")
    except Exception as e:
        test("3.3 Incident 490 employees", False, f"Error: {e}")
    
    # Test 3.4: Duplicate (employee, incident) pair should upsert, not error
    try:
        result2 = add_employee_to_case(
            incident_id=490,
            employee_id=1,
            assigned_by_user_id=1,
            full_name="دينا كمال رقم 1",
            is_primary=True  # Changed to primary
        )
        test("3.4 Duplicate pair upserts (no error)", result2 > 0,
             f"Returned ID: {result2}")
    except Exception as e:
        test("3.4 Duplicate pair upserts", False, f"Error: {e}")
    
    # Test 3.5: After upsert, check IsPrimary was updated
    try:
        emps_490 = get_employees_for_incident(490)
        emp1_in_490 = next((e for e in emps_490 if e['employee_id'] == 1), None)
        test("3.5 IsPrimary updated on upsert", 
             emp1_in_490 and emp1_in_490['is_primary'] == True,
             f"IsPrimary: {emp1_in_490['is_primary'] if emp1_in_490 else 'NOT FOUND'}")
    except Exception as e:
        test("3.5 IsPrimary updated", False, f"Error: {e}")
    
    # Test 3.6: Incident 491 still has its own employees (not affected)
    try:
        emps_491 = get_employees_for_incident(491)
        emp_ids_491 = [e['employee_id'] for e in emps_491]
        test("3.6 Incident 491 employees unaffected", 1 in emp_ids_491 and 2 in emp_ids_491,
             f"Employee IDs in 491: {emp_ids_491}")
    except Exception as e:
        test("3.6 Incident 491 unaffected", False, f"Error: {e}")
    
    # Test 3.7: Total row count increased (was 4, now should be 5)
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM dbo.APP_IncidentCaseEmployee")
    new_count = cursor.fetchone()[0]
    test("3.7 Row count increased (many-to-many working)", new_count == 5,
         f"Count: {new_count} (expected 5: 4 original + 1 new linkage)")
    
    # ============================================================
    # TEST GROUP 4: EDGE CASES
    # ============================================================
    print("\n--- GROUP 4: EDGE CASES ---")
    
    # Test 4.1: Employee 2 can also be linked to 490 (3rd employee for incident 490)
    try:
        result3 = add_employee_to_case(
            incident_id=490,
            employee_id=2,
            assigned_by_user_id=1,
            full_name="دينا كمال رقم 2",
            is_primary=False
        )
        test("4.1 Employee 2 also linked to incident 490", result3 > 0,
             f"Returned ID: {result3}")
    except Exception as e:
        test("4.1 Employee 2 linked to 490", False, f"Error: {e}")
    
    # Test 4.2: Incident 490 now has 4 employees (101, 102, 1, 2)
    try:
        emps_490_final = get_employees_for_incident(490)
        emp_ids_final = sorted([e['employee_id'] for e in emps_490_final])
        test("4.2 Incident 490 has 4 employees", len(emp_ids_final) == 4,
             f"Employee IDs: {emp_ids_final}")
    except Exception as e:
        test("4.2 Incident 490 employees", False, f"Error: {e}")
    
    # Test 4.3: Employee 1 is in exactly 2 incidents (490, 491)
    try:
        incidents_emp1 = get_incidents_for_employee(1)
        test("4.3 Employee 1 in exactly 2 incidents", 
             sorted(incidents_emp1) == [490, 491],
             f"Incidents: {sorted(incidents_emp1)}")
    except Exception as e:
        test("4.3 Employee 1 incidents", False, f"Error: {e}")
    
    # Test 4.4: Employee 2 is in exactly 2 incidents (490, 491)
    try:
        incidents_emp2 = get_incidents_for_employee(2)
        test("4.4 Employee 2 in exactly 2 incidents", 
             sorted(incidents_emp2) == [490, 491],
             f"Incidents: {sorted(incidents_emp2)}")
    except Exception as e:
        test("4.4 Employee 2 incidents", False, f"Error: {e}")
    
    # ============================================================
    # CLEANUP: Remove test linkages (restore to original 4 rows)
    # ============================================================
    print("\n--- CLEANUP ---")
    cursor.execute("""
        DELETE FROM dbo.APP_IncidentCaseEmployee 
        WHERE EmployeeID IN (1, 2) AND IncidentRequestCaseID = 490
    """)
    conn.commit()
    
    cursor.execute("SELECT COUNT(*) FROM dbo.APP_IncidentCaseEmployee")
    cleanup_count = cursor.fetchone()[0]
    test("CLEANUP: Restored to original data", cleanup_count == 4,
         f"Final count: {cleanup_count}")
    
    cursor.close()
    conn.close()
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print(f"PHASE 1 TEST RESULTS: {PASS} PASSED, {FAIL} FAILED out of {PASS+FAIL} tests")
    print("=" * 70)
    
    if FAIL == 0:
        print("🎉 ALL TESTS PASSED! Phase 1 is complete and verified.")
    else:
        print("⚠️  SOME TESTS FAILED — review output above.")
        for name, status, detail in TESTS:
            if status == "FAIL":
                print(f"   ❌ {name}: {detail}")
    
    return FAIL == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
