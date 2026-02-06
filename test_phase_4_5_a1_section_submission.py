"""
PHASE 4.5-A1 — SECTION PLAN SUBMISSION FLOW CERTIFICATION

Validates that the Section Administrator "accept responsibility + submit response"
workflow behaves correctly end-to-end.

This is a DATAFLOW CERTIFICATION TEST. We verify:
- Status transitions are correct
- Data persistence is correct
- Routing to correct inboxes works
- Follow-up items are accessible

Test Flow:
1. Create test subcase (SUBMITTED_TO_SECTION)
2. Section Admin submits response with explanation + 2 action items
3. Verify status → SECTION_ACCEPTED_PENDING_DEPT
4. Verify explanation saved
5. Verify action items created (count, fields, status)
6. Verify Department inbox contains the subcase
7. Verify Section inbox does NOT contain the subcase
8. Verify Follow-up endpoint returns the action items

Run: python test_phase_4_5_a1_section_submission.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pyodbc
from datetime import datetime, timedelta
from typing import Dict, Any, List

# =============================================================================
# TEST INFRASTRUCTURE
# =============================================================================

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
        # Get an existing incident
        cursor.execute("SELECT TOP 1 IncidentRequestCaseID FROM dbo.APP_IncidentCase ORDER BY IncidentRequestCaseID DESC")
        incident = cursor.fetchone()
        incident_id = incident.IncidentRequestCaseID if incident else 177
        
        # Create subcase
        cursor.execute("""
            INSERT INTO dbo.APP_AdministrativeSubcase (
                CaseType,
                IncidentRequestCaseID,
                SeasonalReportID,
                TargetOrgUnitID,
                Status,
                CreatedAt,
                CreatedByUserID
            )
            VALUES (?, ?, NULL, ?, ?, GETDATE(), ?)
        """, ('INCIDENT', incident_id, target_org_unit_id, status, 1))
        
        conn.commit()
        
        # Get the created subcase ID
        cursor.execute("""
            SELECT TOP 1 SubcaseID
            FROM dbo.APP_AdministrativeSubcase
            WHERE IncidentRequestCaseID = ? AND TargetOrgUnitID = ?
            ORDER BY CreatedAt DESC
        """, (incident_id, target_org_unit_id))
        
        subcase = cursor.fetchone()
        return subcase.SubcaseID
        
    finally:
        cursor.close()
        conn.close()


def delete_test_subcase(subcase_id):
    """Delete test subcase and all related data (cascade)"""
    conn, cursor = get_db_cursor()
    try:
        # Delete action items first
        cursor.execute("DELETE FROM dbo.APP_SubcaseActionItem WHERE SubcaseID = ?", (subcase_id,))
        
        # Delete subcase
        cursor.execute("DELETE FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = ?", (subcase_id,))
        
        conn.commit()
        print(f"[CLEANUP] Deleted subcase {subcase_id} and related data")
        
    finally:
        cursor.close()
        conn.close()


def get_subcase_status(subcase_id):
    """Get current status of a subcase"""
    conn, cursor = get_db_cursor()
    try:
        cursor.execute("SELECT Status FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = ?", (subcase_id,))
        row = cursor.fetchone()
        return row.Status if row else None
    finally:
        cursor.close()
        conn.close()


def get_section_explanation(subcase_id):
    """Get section explanation text"""
    conn, cursor = get_db_cursor()
    try:
        cursor.execute("SELECT SectionExplanationText FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = ?", (subcase_id,))
        row = cursor.fetchone()
        return row.SectionExplanationText if row else None
    finally:
        cursor.close()
        conn.close()


def get_action_items_by_subcase(subcase_id):
    """Get all action items for a subcase"""
    conn, cursor = get_db_cursor()
    try:
        cursor.execute("""
            SELECT ActionItemID, SubcaseID, Status, Title, Description, 
                   DueDate, AssignedToUserID, CreatedByUserID, CreatedAt
            FROM dbo.APP_SubcaseActionItem
            WHERE SubcaseID = ?
            ORDER BY ActionItemID
        """, (subcase_id,))
        
        items = []
        for row in cursor.fetchall():
            items.append({
                'action_item_id': row.ActionItemID,
                'subcase_id': row.SubcaseID,
                'status': row.Status,
                'title': row.Title,
                'description': row.Description,
                'due_date': row.DueDate,
                'assigned_to_user_id': row.AssignedToUserID,
                'created_by_user_id': row.CreatedByUserID,
                'created_at': row.CreatedAt
            })
        return items
        
    finally:
        cursor.close()
        conn.close()


# =============================================================================
# TEST EXECUTION
# =============================================================================

def run_test():
    """Execute the complete test flow"""
    print("\n" + "="*80)
    print("PHASE 4.5-A1 — SECTION PLAN SUBMISSION FLOW CERTIFICATION")
    print("="*80)
    
    subcase_id = None
    
    try:
        # =========================================================================
        # PHASE 1: SETUP
        # =========================================================================
        print("\n[PHASE 1: SETUP]")
        print("-" * 80)
        
        # Step 1.1: Create test subcase
        subcase_id = create_test_subcase(status='SUBMITTED_TO_SECTION', target_org_unit_id=2)
        print(f"✓ Created test subcase: SubcaseID={subcase_id}")
        print(f"  Initial status: SUBMITTED_TO_SECTION")
        print(f"  Target org unit: 2")
        
        # Step 1.2: Create test users
        section_admin = create_test_user('SECTION_ADMIN', 2, 'Section')
        dept_admin = create_test_user('DEPARTMENT_ADMIN', 2, 'Department')
        print(f"✓ Created Section Admin user (org_unit_id=2)")
        print(f"✓ Created Department Admin user (org_unit_id=2)")
        
        # =========================================================================
        # PHASE 2: EXECUTION
        # =========================================================================
        print("\n[PHASE 2: EXECUTION]")
        print("-" * 80)
        
        # Step 2.1: Section Admin submits response with action items
        from backend.api_v2.routers.workflow_router import act_on_case
        
        request_body = {
            'action': 'SUBMIT_RESPONSE',
            'explanation_text': 'Section accepts responsibility. Root cause identified as training gap. Corrective measures proposed.',
            'action_items': [
                {
                    'title': 'AI-1: Conduct Staff Retraining',
                    'description': 'Organize mandatory training session for all staff on proper incident reporting procedures.',
                    'due_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
                    'assigned_to_user_id': 1
                },
                {
                    'title': 'AI-2: Update Documentation',
                    'description': 'Revise standard operating procedures manual to include clearer guidelines.',
                    'due_date': (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d'),
                    'assigned_to_user_id': 1
                }
            ]
        }
        
        print(f"→ Section Admin calling: POST /api/v2/workflow/case/{subcase_id}/act")
        print(f"  Action: SUBMIT_RESPONSE")
        print(f"  Explanation: {request_body['explanation_text'][:60]}...")
        print(f"  Action items: {len(request_body['action_items'])}")
        
        response = act_on_case(
            subcase_id=subcase_id,
            body=request_body,
            current_user=section_admin
        )
        
        assert response.get('success') == True, "API call should return success=True"
        print(f"✓ API returned: {response}")
        
        # =========================================================================
        # PHASE 3: VERIFICATION — DATABASE LAYER
        # =========================================================================
        print("\n[PHASE 3: VERIFICATION — DATABASE LAYER]")
        print("-" * 80)
        
        # Assertion 3.1: Status transition
        print(f"\n[3.1] Verify status transition")
        current_status = get_subcase_status(subcase_id)
        print(f"  Current status: {current_status}")
        assert current_status == 'SECTION_ACCEPTED_PENDING_DEPT', \
            f"Expected SECTION_ACCEPTED_PENDING_DEPT, got {current_status}"
        print(f"✓ Status correctly transitioned to SECTION_ACCEPTED_PENDING_DEPT")
        
        # Assertion 3.2: Explanation saved
        print(f"\n[3.2] Verify explanation text saved")
        explanation = get_section_explanation(subcase_id)
        print(f"  Saved explanation: {explanation[:60] if explanation else None}...")
        assert explanation is not None, "Explanation should not be None"
        assert 'Section accepts responsibility' in explanation, "Explanation content mismatch"
        print(f"✓ Explanation text saved correctly")
        
        # Assertion 3.3: Action items created
        print(f"\n[3.3] Verify action items created")
        action_items = get_action_items_by_subcase(subcase_id)
        print(f"  Action items count: {len(action_items)}")
        
        assert len(action_items) == 2, f"Expected 2 action items, got {len(action_items)}"
        print(f"✓ Correct number of action items created")
        
        # Assertion 3.4: Action item fields
        print(f"\n[3.4] Verify action item fields")
        for idx, item in enumerate(action_items, 1):
            print(f"  Action Item {idx}:")
            print(f"    - ID: {item['action_item_id']}")
            print(f"    - SubcaseID: {item['subcase_id']}")
            print(f"    - Status: {item['status']}")
            print(f"    - Title: {item['title']}")
            print(f"    - Description: {item['description'][:50]}...")
            print(f"    - DueDate: {item['due_date']}")
            print(f"    - AssignedTo: {item['assigned_to_user_id']}")
            print(f"    - CreatedBy: {item['created_by_user_id']}")
            
            # Validate fields
            assert item['subcase_id'] == subcase_id, \
                f"Item {idx}: SubcaseID mismatch"
            assert item['status'] == 'DRAFT', \
                f"Item {idx}: Expected status DRAFT, got {item['status']}"
            assert item['title'] is not None and len(item['title']) > 0, \
                f"Item {idx}: Title is empty"
            assert item['created_by_user_id'] == section_admin.user_id, \
                f"Item {idx}: CreatedByUserID should be {section_admin.user_id}"
            assert item['assigned_to_user_id'] == 1, \
                f"Item {idx}: AssignedToUserID should be 1"
        
        print(f"✓ All action item fields are correct")
        
        # =========================================================================
        # PHASE 4: VERIFICATION — INBOX ROUTING
        # =========================================================================
        print("\n[PHASE 4: VERIFICATION — INBOX ROUTING]")
        print("-" * 80)
        
        # Assertion 4.1: Department inbox contains the subcase
        print(f"\n[4.1] Verify Department Admin inbox contains subcase")
        from backend.api_v2.routers.workflow_router import get_inbox
        
        dept_inbox_response = get_inbox(current_user=dept_admin)
        dept_inbox = dept_inbox_response.get('items', [])
        print(f"  Department inbox item count: {len(dept_inbox)}")
        
        subcase_in_dept_inbox = any(item['subcase_id'] == subcase_id for item in dept_inbox)
        assert subcase_in_dept_inbox, \
            f"Subcase {subcase_id} should be in Department Admin inbox"
        
        # Find our subcase
        our_subcase = next((item for item in dept_inbox if item['subcase_id'] == subcase_id), None)
        if our_subcase:
            print(f"  Found in inbox:")
            print(f"    - SubcaseID: {our_subcase['subcase_id']}")
            print(f"    - Status: {our_subcase['status']}")
            print(f"    - TargetOrgUnit: {our_subcase['target_org_unit_id']}")
            print(f"    - AllowedActions: {our_subcase['allowed_actions']}")
        
        print(f"✓ Subcase correctly routed to Department inbox")
        
        # Assertion 4.2: Section inbox does NOT contain the subcase
        print(f"\n[4.2] Verify Section Admin inbox does NOT contain subcase")
        section_inbox_response = get_inbox(current_user=section_admin)
        section_inbox = section_inbox_response.get('items', [])
        print(f"  Section inbox item count: {len(section_inbox)}")
        
        subcase_in_section_inbox = any(item['subcase_id'] == subcase_id for item in section_inbox)
        assert not subcase_in_section_inbox, \
            f"Subcase {subcase_id} should NOT be in Section Admin inbox after submission"
        
        print(f"✓ Subcase correctly removed from Section inbox")
        
        # =========================================================================
        # PHASE 5: VERIFICATION — FOLLOW-UP ENDPOINT
        # =========================================================================
        print("\n[PHASE 5: VERIFICATION — FOLLOW-UP ENDPOINT]")
        print("-" * 80)
        
        # Assertion 5.1: Follow-up endpoint returns action items
        print(f"\n[5.1] Verify follow-up endpoint returns action items")
        from backend.api_v2.routers.workflow_router import get_follow_up_items
        
        # Create a user with assignment to action items
        worker_user = create_test_user('SECTION_ADMIN', 2, 'Section')
        worker_user.user_id = 1  # Match the assigned_to_user_id
        
        follow_up_response = get_follow_up_items(current_user=worker_user)
        follow_up_items = follow_up_response.get('items', [])
        print(f"  Total follow-up items: {len(follow_up_items)}")
        
        # Filter to our subcase's action items
        our_action_items = [item for item in follow_up_items if item.get('subcase_id') == subcase_id]
        print(f"  Our subcase's action items: {len(our_action_items)}")
        
        assert len(our_action_items) >= 2, \
            f"Expected at least 2 action items in follow-up, got {len(our_action_items)}"
        
        for item in our_action_items:
            print(f"  Follow-up item:")
            print(f"    - ActionItemID: {item.get('action_item_id')}")
            print(f"    - SubcaseID: {item.get('subcase_id')}")
            print(f"    - Title: {item.get('title')}")
            print(f"    - Status: {item.get('status')}")
            print(f"    - AssignedTo: {item.get('assigned_to_user_id')}")
        
        print(f"✓ Action items correctly appear in follow-up endpoint")
        
        # =========================================================================
        # FINAL SUMMARY
        # =========================================================================
        print("\n" + "="*80)
        print("TEST RESULT: ✅ ALL ASSERTIONS PASSED")
        print("="*80)
        print("\n✓ Status transition: SUBMITTED_TO_SECTION → SECTION_ACCEPTED_PENDING_DEPT")
        print("✓ Explanation text: Saved correctly")
        print("✓ Action items: Created (count=2, status=DRAFT, fields correct)")
        print("✓ Department inbox: Contains subcase")
        print("✓ Section inbox: Does NOT contain subcase")
        print("✓ Follow-up endpoint: Returns action items")
        print("\n🎉 DATAFLOW CERTIFICATION: PASSED")
        print("="*80 + "\n")
        
    except AssertionError as e:
        print("\n" + "="*80)
        print("TEST RESULT: ❌ ASSERTION FAILED")
        print("="*80)
        print(f"\nError: {str(e)}")
        print("\n⚠️ DATAFLOW CERTIFICATION: FAILED")
        print("="*80 + "\n")
        raise
        
    except Exception as e:
        print("\n" + "="*80)
        print("TEST RESULT: ❌ EXCEPTION OCCURRED")
        print("="*80)
        print(f"\nError: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n⚠️ DATAFLOW CERTIFICATION: ERROR")
        print("="*80 + "\n")
        raise
        
    finally:
        # Cleanup
        if subcase_id:
            try:
                delete_test_subcase(subcase_id)
            except Exception as e:
                print(f"[WARNING] Cleanup failed: {e}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_test()
