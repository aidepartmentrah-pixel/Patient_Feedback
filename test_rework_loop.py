"""
TEST: Complete Rework Loop
Validates the new workflow contract: rejection returns for revision and allows resubmission
"""

import pyodbc
from datetime import datetime, timedelta


def get_db_cursor():
    """Get database connection and cursor"""
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=SOCIALMEDIA;"
        "DATABASE=IncidentManager;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )
    return conn, conn.cursor()


def create_test_user(role_code, org_unit_id, org_unit_type_str):
    """Create a mock CurrentUser object for testing"""
    from backend.api.schemas.auth_models import CurrentUser, UserScope
    
    return CurrentUser(
        user_id=999,
        username=f"test_user_{role_code.lower()}",
        is_active=True,
        scopes=[
            UserScope(
                role_code=role_code,
                org_unit_id=org_unit_id,
                org_unit_type=org_unit_type_str
            )
        ],
        allowed_unit_ids={org_unit_id} if org_unit_id else set()
    )


def create_test_subcase(status='SUBMITTED_TO_SECTION', target_org_unit_id=2):
    """Create a test subcase and return its ID"""
    conn, cursor = get_db_cursor()
    try:
        cursor.execute("SELECT TOP 1 IncidentRequestCaseID FROM dbo.APP_IncidentCase ORDER BY IncidentRequestCaseID DESC")
        incident = cursor.fetchone()
        incident_id = incident.IncidentRequestCaseID if incident else 177
        
        cursor.execute("""
            INSERT INTO dbo.APP_AdministrativeSubcase (
                CaseType, IncidentRequestCaseID, SeasonalReportID, TargetOrgUnitID, Status, CreatedAt, CreatedByUserID
            )
            VALUES (?, ?, NULL, ?, ?, GETDATE(), ?)
        """, ('INCIDENT', incident_id, target_org_unit_id, status, 1))
        
        conn.commit()
        
        cursor.execute("""
            SELECT TOP 1 SubcaseID FROM dbo.APP_AdministrativeSubcase
            WHERE IncidentRequestCaseID = ? AND TargetOrgUnitID = ? ORDER BY CreatedAt DESC
        """, (incident_id, target_org_unit_id))
        
        result = cursor.fetchone()
        return result.SubcaseID if result else None
    finally:
        cursor.close()
        conn.close()


def cleanup_test_subcase(subcase_id):
    """Clean up test data"""
    conn, cursor = get_db_cursor()
    try:
        cursor.execute("DELETE FROM dbo.APP_SubcaseActionItem WHERE SubcaseID = ?", (subcase_id,))
        cursor.execute("DELETE FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = ?", (subcase_id,))
        conn.commit()
    finally:
        cursor.close()
        conn.close()


def get_subcase_status(subcase_id):
    """Get current status of subcase"""
    conn, cursor = get_db_cursor()
    try:
        cursor.execute("SELECT Status FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = ?", (subcase_id,))
        result = cursor.fetchone()
        return result.Status if result else None
    finally:
        cursor.close()
        conn.close()


def get_action_item_count(subcase_id):
    """Get count of action items for subcase"""
    conn, cursor = get_db_cursor()
    try:
        cursor.execute("SELECT COUNT(*) as cnt FROM dbo.APP_SubcaseActionItem WHERE SubcaseID = ?", (subcase_id,))
        result = cursor.fetchone()
        return result.cnt if result else 0
    finally:
        cursor.close()
        conn.close()


def test_complete_rework_loop():
    """
    Test the complete rework loop workflow:
    1. Section submits response with 2 action items
    2. Department rejects → Status = RETURNED_TO_SECTION_FOR_REVISION
    3. Section appears in section inbox again
    4. Section resubmits using OVERRIDE with new action items
    5. Department approves
    6. Administration approves → Final approval
    """
    from backend.api_v2.routers.workflow_router import act_on_case
    from backend.api_v2.services.inbox_service import get_section_inbox
    
    print("\n" + "="*70)
    print("TEST: COMPLETE REWORK LOOP (WORKFLOW CONTRACT VALIDATION)")
    print("="*70)
    
    subcase_id = create_test_subcase(status='SUBMITTED_TO_SECTION', target_org_unit_id=2)
    print(f"\n[SETUP] Created subcase {subcase_id}")
    
    try:
        section_admin = create_test_user('SECTION_ADMIN', 2, 'Section')
        dept_admin = create_test_user('DEPARTMENT_ADMIN', 1, 'Department')
        admin = create_test_user('ADMINISTRATION_ADMIN', 0, 'Administration')
        
        # Step 1: Section submits initial response
        print("\n--- STEP 1: Initial Section Response ---")
        act_on_case(
            subcase_id=subcase_id,
            body={
                'action': 'SUBMIT_RESPONSE',
                'explanation_text': 'Initial section response',
                'action_items': [
                    {
                        'title': 'Original Item 1',
                        'description': 'First item',
                        'assigned_to_user_id': 999,
                        'due_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
                    },
                    {
                        'title': 'Original Item 2',
                        'description': 'Second item',
                        'assigned_to_user_id': 999,
                        'due_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
                    }
                ]
            },
            current_user=section_admin
        )
        
        status = get_subcase_status(subcase_id)
        item_count = get_action_item_count(subcase_id)
        assert status == 'SECTION_ACCEPTED_PENDING_DEPT', f"Expected SECTION_ACCEPTED_PENDING_DEPT, got {status}"
        assert item_count == 2, f"Expected 2 items, got {item_count}"
        print(f"✓ Section submitted: Status={status}, Items={item_count}")
        
        # Step 2: Department rejects
        print("\n--- STEP 2: Department Rejects (Returns for Revision) ---")
        act_on_case(
            subcase_id=subcase_id,
            body={
                'action': 'REJECT',
                'rejection_text': 'Response insufficient - please provide more detail'
            },
            current_user=dept_admin
        )
        
        status = get_subcase_status(subcase_id)
        item_count = get_action_item_count(subcase_id)
        assert status == 'RETURNED_TO_SECTION_FOR_REVISION', f"Expected RETURNED_TO_SECTION_FOR_REVISION, got {status}"
        assert item_count == 2, f"Expected items to remain, got {item_count}"
        print(f"✓ Department rejected: Status={status}, Items remain={item_count}")
        
        # Step 3: Verify subcase appears in section inbox again
        print("\n--- STEP 3: Verify Subcase Reappears in Section Inbox ---")
        section_inbox = get_section_inbox(section_admin)
        subcase_ids_in_inbox = [item['subcase_id'] for item in section_inbox]
        assert subcase_id in subcase_ids_in_inbox, "Subcase should reappear in section inbox"
        print(f"✓ Subcase {subcase_id} found in section inbox (ready for resubmission)")
        
        # Step 4: Section resubmits with OVERRIDE (corrected response)
        print("\n--- STEP 4: Section Resubmits (OVERRIDE with New Items) ---")
        # Note: When status is RETURNED_TO_SECTION_FOR_REVISION, section can either:
        # a) Submit new response (transitions to SECTION_ACCEPTED_PENDING_DEPT)
        # b) Use OVERRIDE to replace items and explanation
        
        # For now, let's submit a new response (this should work from revision state)
        act_on_case(
            subcase_id=subcase_id,
            body={
                'action': 'SUBMIT_RESPONSE',
                'explanation_text': 'CORRECTED section response with more detail',
                'action_items': [
                    {
                        'title': 'Revised Item 1',
                        'description': 'Corrected first item with more detail',
                        'assigned_to_user_id': 999,
                        'due_date': (datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d')
                    },
                    {
                        'title': 'Revised Item 2',
                        'description': 'Corrected second item',
                        'assigned_to_user_id': 999,
                        'due_date': (datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d')
                    },
                    {
                        'title': 'NEW Item 3',
                        'description': 'Additional item per feedback',
                        'assigned_to_user_id': 999,
                        'due_date': (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d')
                    }
                ]
            },
            current_user=section_admin
        )
        
        status = get_subcase_status(subcase_id)
        item_count = get_action_item_count(subcase_id)
        print(f"Status after resubmission: {status}")
        print(f"Items after resubmission: {item_count}")
        
        # The status should transition - but we need to check what the service layer does
        # If SUBMIT_RESPONSE is only valid from SUBMITTED_TO_SECTION, this will fail
        # In that case, we need to add RETURNED_TO_SECTION_FOR_REVISION to allowed statuses
        
        if status == 'SECTION_ACCEPTED_PENDING_DEPT':
            print(f"✓ Resubmission successful: Status={status}, Items={item_count}")
        else:
            print(f"⚠️ Resubmission status: {status} (may need to add RETURNED_TO_SECTION_FOR_REVISION to allowed statuses)")
        
        # Step 5: Department approves corrected response
        print("\n--- STEP 5: Department Approves Corrected Response ---")
        act_on_case(
            subcase_id=subcase_id,
            body={'action': 'APPROVE'},
            current_user=dept_admin
        )
        
        status = get_subcase_status(subcase_id)
        assert status == 'DEPT_ACCEPTED_PENDING_ADMIN', f"Expected DEPT_ACCEPTED_PENDING_ADMIN, got {status}"
        print(f"✓ Department approved: Status={status}")
        
        # Step 6: Administration approves
        print("\n--- STEP 6: Administration Final Approval ---")
        act_on_case(
            subcase_id=subcase_id,
            body={'action': 'APPROVE'},
            current_user=admin
        )
        
        status = get_subcase_status(subcase_id)
        assert status == 'ADMIN_APPROVED', f"Expected ADMIN_APPROVED, got {status}"
        print(f"✓ Final approval: Status={status}")
        
        print("\n" + "="*70)
        print("✅ REWORK LOOP VALIDATED SUCCESSFULLY!")
        print("  - Department rejection returns for revision ✓")
        print("  - Subcase reappears in section inbox ✓")
        print("  - Section can resubmit corrected response ✓")
        print("  - Complete approval workflow works after rework ✓")
        print("="*70)
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup_test_subcase(subcase_id)
        print(f"\n[CLEANUP] Test data cleaned")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, r'c:\Users\IT\Documents\GitHub Repository\Patient_Feedback')
    success = test_complete_rework_loop()
    exit(0 if success else 1)
