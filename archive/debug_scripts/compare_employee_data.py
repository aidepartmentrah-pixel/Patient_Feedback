"""
Compare Old vs New Employee Data
Shows the difference before and after the validation fix
"""
from backend.core.database import get_connection


def compare_incidents():
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        print("=" * 80)
        print("COMPARISON: Before Fix vs After Fix")
        print("=" * 80)
        
        # Incident 490 - BEFORE FIX (wrong data)
        print("\n🔴 INCIDENT 490 - BEFORE FIX (Made-up names)")
        print("-" * 80)
        
        cursor.execute("""
            SELECT 
                e.EmployeeID,
                e.FullName AS StoredName,
                hr.FullName AS ActualHRName,
                CASE 
                    WHEN e.FullName = hr.FullName THEN 'MATCH'
                    ELSE 'MISMATCH'
                END AS Status
            FROM dbo.APP_IncidentCaseEmployee e
            LEFT JOIN dbo.APP_VIEWTABLE_HR_EMPLOYEES hr ON e.EmployeeID = hr.EmployeeID
            WHERE e.IncidentRequestCaseID = 490
            ORDER BY e.EmployeeID
        """)
        
        old_employees = cursor.fetchall()
        
        for emp in old_employees:
            print(f"\nEmployee {emp.EmployeeID}:")
            print(f"  ❌ Stored (WRONG):  {emp.StoredName}")
            print(f"  ✅ Should be (HR):  {emp.ActualHRName}")
            print(f"  Status: {emp.Status}")
        
        # Incident 491 - AFTER FIX (correct data)
        print("\n" + "=" * 80)
        print("✅ INCIDENT 491 - AFTER FIX (HR system names)")
        print("-" * 80)
        
        cursor.execute("""
            SELECT 
                e.EmployeeID,
                e.FullName AS StoredName,
                hr.FullName AS ActualHRName,
                CASE 
                    WHEN e.FullName = hr.FullName THEN 'MATCH'
                    ELSE 'MISMATCH'
                END AS Status
            FROM dbo.APP_IncidentCaseEmployee e
            LEFT JOIN dbo.APP_VIEWTABLE_HR_EMPLOYEES hr ON e.EmployeeID = hr.EmployeeID
            WHERE e.IncidentRequestCaseID = 491
            ORDER BY e.EmployeeID
        """)
        
        new_employees = cursor.fetchall()
        
        for emp in new_employees:
            print(f"\nEmployee {emp.EmployeeID}:")
            print(f"  ✅ Stored (CORRECT): {emp.StoredName}")
            print(f"  ✅ HR System:        {emp.ActualHRName}")
            print(f"  Status: {emp.Status}")
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        old_mismatches = sum(1 for emp in old_employees if emp.StoredName != emp.ActualHRName)
        new_mismatches = sum(1 for emp in new_employees if emp.StoredName != emp.ActualHRName)
        
        print(f"\nIncident 490 (Before Fix): {old_mismatches} mismatches")
        print(f"Incident 491 (After Fix):  {new_mismatches} mismatches")
        
        if new_mismatches == 0:
            print("\n✅ FIX SUCCESSFUL - All employee names now match HR system!")
        else:
            print("\n❌ Fix incomplete - Still have mismatches")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


if __name__ == "__main__":
    compare_incidents()
