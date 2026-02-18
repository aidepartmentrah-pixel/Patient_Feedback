"""
Verify Employee Linkage in Database
"""
from backend.core.database import get_connection


def verify_employee_linkage(incident_id: int):
    """Check if employees were linked to the incident"""
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        print("=" * 60)
        print(f"Verifying Employee Linkage for Incident {incident_id}")
        print("=" * 60)
        
        # Check employees linked to this incident
        cursor.execute("""
            SELECT 
                EmployeeID,
                FullName,
                IncidentRequestCaseID,
                IsPrimary,
                AssignedAt,
                AssignedByUserID
            FROM dbo.APP_IncidentCaseEmployee
            WHERE IncidentRequestCaseID = ?
            ORDER BY IsPrimary DESC, EmployeeID
        """, (incident_id,))
        
        employees = cursor.fetchall()
        
        if not employees:
            print(f"\n❌ NO employees linked to incident {incident_id}")
            return False
        
        print(f"\n✅ Found {len(employees)} employees linked to incident {incident_id}:")
        print("-" * 60)
        
        for emp in employees:
            primary_marker = "🔵 PRIMARY" if emp.IsPrimary else "  Secondary"
            print(f"{primary_marker}")
            print(f"  Employee ID: {emp.EmployeeID}")
            print(f"  Full Name: {emp.FullName}")
            print(f"  Incident ID: {emp.IncidentRequestCaseID}")
            print(f"  Assigned At: {emp.AssignedAt}")
            print(f"  Assigned By: {emp.AssignedByUserID}")
            print("-" * 60)
        
        # Verify the incident exists
        cursor.execute("""
            SELECT 
                IncidentRequestCaseID,
                ComplaintText,
                PatientName,
                CaseStatusID,
                ExplanationStatusID
            FROM dbo.APP_IncidentCase
            WHERE IncidentRequestCaseID = ?
        """, (incident_id,))
        
        incident = cursor.fetchone()
        
        if incident:
            print(f"\n✅ Incident Details:")
            print(f"  ID: {incident.IncidentRequestCaseID}")
            print(f"  Patient: {incident.PatientName}")
            print(f"  Complaint: {incident.ComplaintText[:50]}...")
            print(f"  Case Status: {incident.CaseStatusID}")
            print(f"  Explanation Status: {incident.ExplanationStatusID}")
        
        print("\n" + "=" * 60)
        print("✅ VERIFICATION PASSED - Employees are linked!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        incident_id = int(sys.argv[1])
    else:
        incident_id = 490  # From the test
    
    verify_employee_linkage(incident_id)
