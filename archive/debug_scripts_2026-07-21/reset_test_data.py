"""Reset test data for closed-loop testing.

This script:
1. Deletes incident cases 502, 503 and their related data
2. Deletes all SEASONAL_REPORT_RESPONSE subcases
3. Resets all INCIDENT_RESPONSE subcases to SUBMITTED_TO_SECTION status
4. Clears explanation/rejection fields
"""
from core.database import get_connection

def reset_test_data():
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        print("=" * 60)
        print("RESETTING TEST DATA FOR CLOSED-LOOP TESTING")
        print("=" * 60)
        
        # ============================================================
        # STEP 1: Delete cases 502 and 503 and their related data
        # ============================================================
        cases_to_delete = [502, 503]
        print(f"\n[STEP 1] Deleting cases {cases_to_delete} and related data...")
        
        # First, get subcases for these cases
        cursor.execute("""
            SELECT SubcaseID FROM dbo.APP_AdministrativeSubcase
            WHERE IncidentRequestCaseID IN (502, 503)
        """)
        subcases_to_delete = [r[0] for r in cursor.fetchall()]
        print(f"   Found {len(subcases_to_delete)} subcases to delete: {subcases_to_delete}")
        
        # Delete action items for subcases first
        if subcases_to_delete:
            cursor.execute("""
                DELETE FROM dbo.APP_SubcaseActionItem
                WHERE SubcaseID IN (
                    SELECT SubcaseID FROM dbo.APP_AdministrativeSubcase
                    WHERE IncidentRequestCaseID IN (502, 503)
                )
            """)
            print(f"   Deleted {cursor.rowcount} action items for subcases")
        
        # Delete subcases for cases 502, 503
        cursor.execute("""
            DELETE FROM dbo.APP_AdministrativeSubcase
            WHERE IncidentRequestCaseID IN (502, 503)
        """)
        print(f"   Deleted {cursor.rowcount} subcases for cases 502, 503")
        
        # Delete target departments for cases 502, 503
        cursor.execute("""
            DELETE FROM dbo.APP_IncidentCaseTargetDepartment
            WHERE IncidentRequestCaseID IN (502, 503)
        """)
        print(f"   Deleted {cursor.rowcount} target department records")
        
        # Delete doctors linked to cases 502, 503
        cursor.execute("""
            DELETE FROM dbo.APP_IncidentCaseDoctor
            WHERE IncidentRequestCaseID IN (502, 503)
        """)
        print(f"   Deleted {cursor.rowcount} doctor records")
        
        # Delete employees linked to cases 502, 503  
        cursor.execute("""
            DELETE FROM dbo.APP_IncidentCaseEmployee
            WHERE IncidentRequestCaseID IN (502, 503)
        """)
        print(f"   Deleted {cursor.rowcount} employee records")
        
        # Delete the incident cases themselves
        cursor.execute("""
            DELETE FROM dbo.APP_IncidentCase
            WHERE IncidentRequestCaseID IN (502, 503)
        """)
        print(f"   Deleted {cursor.rowcount} incident cases")
        
        # ============================================================
        # STEP 2: Delete ALL SEASONAL_REPORT_RESPONSE subcases
        # ============================================================
        print(f"\n[STEP 2] Deleting all SEASONAL_REPORT_RESPONSE subcases...")
        
        cursor.execute("""
            SELECT SubcaseID FROM dbo.APP_AdministrativeSubcase
            WHERE CaseType = 'SEASONAL_REPORT_RESPONSE'
        """)
        seasonal_subcases = [r[0] for r in cursor.fetchall()]
        print(f"   Found {len(seasonal_subcases)} seasonal report subcases to delete")
        
        # Delete action items for seasonal subcases first
        if seasonal_subcases:
            cursor.execute("""
                DELETE FROM dbo.APP_SubcaseActionItem
                WHERE SubcaseID IN (
                    SELECT SubcaseID FROM dbo.APP_AdministrativeSubcase
                    WHERE CaseType = 'SEASONAL_REPORT_RESPONSE'
                )
            """)
            print(f"   Deleted {cursor.rowcount} action items for seasonal subcases")
        
        cursor.execute("""
            DELETE FROM dbo.APP_AdministrativeSubcase
            WHERE CaseType = 'SEASONAL_REPORT_RESPONSE'
        """)
        print(f"   Deleted {cursor.rowcount} SEASONAL_REPORT_RESPONSE subcases")
        
        # ============================================================
        # STEP 3: Reset all INCIDENT_RESPONSE subcases
        # ============================================================
        print(f"\n[STEP 3] Resetting INCIDENT_RESPONSE subcases to SUBMITTED_TO_SECTION...")
        
        # First show current state
        cursor.execute("""
            SELECT SubcaseID, Status, IncidentRequestCaseID
            FROM dbo.APP_AdministrativeSubcase
            WHERE CaseType = 'INCIDENT_RESPONSE'
            ORDER BY IncidentRequestCaseID, SubcaseID
        """)
        current_subcases = cursor.fetchall()
        print(f"   Found {len(current_subcases)} INCIDENT_RESPONSE subcases")
        
        # Delete action items for these subcases (to start fresh)
        cursor.execute("""
            DELETE FROM dbo.APP_SubcaseActionItem
            WHERE SubcaseID IN (
                SELECT SubcaseID FROM dbo.APP_AdministrativeSubcase
                WHERE CaseType = 'INCIDENT_RESPONSE'
            )
        """)
        print(f"   Deleted {cursor.rowcount} action items for INCIDENT_RESPONSE subcases")
        
        # Reset status and clear all explanation/rejection fields
        cursor.execute("""
            UPDATE dbo.APP_AdministrativeSubcase
            SET 
                Status = 'SUBMITTED_TO_SECTION',
                SectionExplanationText = NULL,
                SectionRejectionText = NULL,
                DepartmentExplanationText = NULL,
                DepartmentRejectionText = NULL,
                AdministrationExplanationText = NULL,
                AdministrationRejectionText = NULL,
                ForceClosedAt = NULL,
                ForceClosedByUserID = NULL,
                ForceCloseReason = NULL,
                UpdatedAt = GETDATE()
            WHERE CaseType = 'INCIDENT_RESPONSE'
        """)
        print(f"   Updated {cursor.rowcount} subcases to SUBMITTED_TO_SECTION")
        
        # ============================================================
        # STEP 4: Verify final state
        # ============================================================
        print(f"\n[STEP 4] Verifying final state...")
        
        # Count remaining cases
        cursor.execute("""
            SELECT COUNT(DISTINCT IncidentRequestCaseID) 
            FROM dbo.APP_AdministrativeSubcase
            WHERE CaseType = 'INCIDENT_RESPONSE'
        """)
        case_count = cursor.fetchone()[0]
        
        # Count remaining subcases
        cursor.execute("""
            SELECT COUNT(*) FROM dbo.APP_AdministrativeSubcase
            WHERE CaseType = 'INCIDENT_RESPONSE'
        """)
        subcase_count = cursor.fetchone()[0]
        
        # Verify all are SUBMITTED_TO_SECTION
        cursor.execute("""
            SELECT Status, COUNT(*) as cnt
            FROM dbo.APP_AdministrativeSubcase
            WHERE CaseType = 'INCIDENT_RESPONSE'
            GROUP BY Status
        """)
        status_breakdown = cursor.fetchall()
        
        print(f"\n   FINAL STATE:")
        print(f"   - Incident cases with subcases: {case_count}")
        print(f"   - Total INCIDENT_RESPONSE subcases: {subcase_count}")
        print(f"   - Status breakdown:")
        for status, cnt in status_breakdown:
            print(f"       {status}: {cnt}")
        
        # Show cases and their subcase counts
        cursor.execute("""
            SELECT IncidentRequestCaseID, COUNT(*) as subcase_count
            FROM dbo.APP_AdministrativeSubcase
            WHERE CaseType = 'INCIDENT_RESPONSE'
            GROUP BY IncidentRequestCaseID
            ORDER BY IncidentRequestCaseID
        """)
        print(f"\n   Subcases per case:")
        for case_id, cnt in cursor.fetchall():
            print(f"       Case {case_id}: {cnt} subcases")
        
        # Commit all changes
        conn.commit()
        print("\n" + "=" * 60)
        print("SUCCESS! All changes committed.")
        print("=" * 60)
        
    except Exception as e:
        conn.rollback()
        print(f"\nERROR: {e}")
        print("All changes rolled back.")
        raise
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    reset_test_data()
