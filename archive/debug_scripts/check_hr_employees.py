"""
Check APP_VIEWTABLE_HR_EMPLOYEES Table
"""
from backend.core.database import get_connection


def check_hr_employees():
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        print("=" * 60)
        print("APP_VIEWTABLE_HR_EMPLOYEES Table")
        print("=" * 60)
        
        # Check if table exists
        cursor.execute("""
            SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_NAME = 'APP_VIEWTABLE_HR_EMPLOYEES'
        """)
        
        if cursor.fetchone()[0] == 0:
            print("\n❌ Table APP_VIEWTABLE_HR_EMPLOYEES does not exist!")
            return
        
        # Get sample employees
        cursor.execute("""
            SELECT TOP 10
                EmployeeID,
                FullName,
                JobTitle,
                DepartmentID,
                IsActive
            FROM APP_VIEWTABLE_HR_EMPLOYEES
            WHERE IsActive = 1
            ORDER BY EmployeeID
        """)
        
        employees = cursor.fetchall()
        
        if not employees:
            print("\n⚠️  Table exists but has no active employees!")
            return
        
        print(f"\n✅ Found {len(employees)} sample active employees:")
        print("-" * 60)
        
        for emp in employees:
            print(f"ID: {emp.EmployeeID:6} | Name: {emp.FullName:30} | Job: {emp.JobTitle or 'N/A'}")
        
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM APP_VIEWTABLE_HR_EMPLOYEES WHERE IsActive = 1")
        total = cursor.fetchone()[0]
        print("-" * 60)
        print(f"Total Active Employees: {total}")
        
        # Check if test employees exist
        print("\n" + "=" * 60)
        print("Checking Test Employee IDs:")
        print("=" * 60)
        
        for test_id in [101, 102]:
            cursor.execute("""
                SELECT EmployeeID, FullName, IsActive
                FROM APP_VIEWTABLE_HR_EMPLOYEES
                WHERE EmployeeID = ?
            """, (test_id,))
            
            emp = cursor.fetchone()
            if emp:
                active_status = "✅ ACTIVE" if emp.IsActive else "❌ INACTIVE"
                print(f"Employee {test_id}: {active_status} - {emp.FullName}")
            else:
                print(f"Employee {test_id}: ❌ NOT FOUND")
        
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
    check_hr_employees()
