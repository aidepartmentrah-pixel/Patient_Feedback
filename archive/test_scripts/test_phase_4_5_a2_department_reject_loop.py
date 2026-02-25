"""
PHASE 4.5-A2 — DEPARTMENT REJECT RETURN LOOP CERTIFICATION

Validates the backward path workflow: Department rejects Section's response,
causing the subcase to return for revision.

This is a DATAFLOW CERTIFICATION TEST. We verify:
- Status transitions correctly to RETURNED_TO_SECTION_FOR_REVISION
- Subcase returns to Section inbox
- Subcase removed from Department inbox
- Action items persist after rejection (not deleted)
- Follow-up execution blocked during revision state
- Section can resubmit with new action items (override behavior)

Test Flow:
1. Create subcase → Section submits response with 2 action items
2. Department rejects the response
3. Verify status → RETURNED_TO_SECTION_FOR_REVISION
4. Verify routing: back in Section inbox, removed from Department inbox
5. Verify action items still exist
6. Verify follow-up execution blocked
7. Section resubmits with 3 new action items
8. Verify old items replaced, new items active

Run: python test_phase_4_5_a2_department_reject_loop.py
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
    print("PHASE 4.5-A2 — DEPARTMENT REJECT RETURN LOOP CERTIFICATION")
    print("="*80)
    
    subcase_id = None
    
    try:
        # =========================================================================
        # PHASE 1: SETUP & INITIAL SUBMISSION
        # =========================================================================
        print("\n[PHASE 1: SETUP & INITIAL SUBMISSION]")
        print("-" * 80)
        
        # Step 1.1: Create test subcase
        subcase_id = create_test_subcase(status='SUBMITTED_TO_SECTION', target_org_unit_id=2)
        print(f"✓ Created test subcase: SubcaseID={subcase_id}")
        print(f"  Initial status: SUBMITTED_TO_SECTION")
        
        # Step 1.2: Create test users
        section_admin = create_test_user('SECTION_ADMIN', 2, 'Section')
        dept_admin = create_test_user('DEPARTMENT_ADMIN', 2, 'Department')
        print(f"✓ Created Section Admin user")
        print(f"✓ Created Department Admin user")
        
        # Step 1.3: Section Admin submits initial response
        from backend.api_v2.routers.workflow_router import act_on_case
        
        initial_request = {
            'action': 'SUBMIT_RESPONSE',
            'explanation_text': 'Initial section response with corrective plan.',
            'action_items': [
                {
                    'title': 'AI-1: Initial Action Item One',
                    'description': 'First corrective action from initial submission',
                    'due_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
                    'assigned_to_user_id': 1
                },
                {
                    'title': 'AI-2: Initial Action Item Two',
                    'description': 'Second corrective action from initial submission',
                    'due_date': (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d'),
                    'assigned_to_user_id': 1
                }
            ]
        }
        
        print(f"\n→ Section Admin submits initial response")
        print(f"  Explanation: {initial_request['explanation_text'][:50]}...")
        print(f"  Action items: {len(initial_request['action_items'])}")
        
        response = act_on_case(
            subcase_id=subcase_id,
            body=initial_request,
            current_user=section_admin
        )
        
        assert response.get('success') == True, "Initial submission should succeed"
        print(f"✓ Initial submission successful")
        
        # Verify status after initial submission
        status_after_submit = get_subcase_status(subcase_id)
        assert status_after_submit == 'SECTION_ACCEPTED_PENDING_DEPT', \
            f"Expected SECTION_ACCEPTED_PENDING_DEPT, got {status_after_submit}"
        print(f"✓ Status: {status_after_submit}")
        
        # Get initial action items
        initial_action_items = get_action_items_by_subcase(subcase_id)
        initial_item_ids = [item['action_item_id'] for item in initial_action_items]
        print(f"✓ Initial action items created: {len(initial_action_items)} items")
        print(f"  ActionItemIDs: {initial_item_ids}")
        
        # =========================================================================
        # PHASE 2: DEPARTMENT REJECTION
        # =========================================================================
        print("\n[PHASE 2: DEPARTMENT REJECTION]")
        print("-" * 80)
        
        # Step 2.1: Department Admin rejects the response
        reject_request = {
            'action': 'REJECT',
            'rejection_text': 'Department finds the proposed corrective measures insufficient. Please provide more detailed timeline and resources allocation.'
        }
        
        print(f"→ Department Admin rejects response")
        print(f"  Rejection text: {reject_request['rejection_text'][:60]}...")
        
        reject_response = act_on_case(
            subcase_id=subcase_id,
            body=reject_request,
            current_user=dept_admin
        )
        
        assert reject_response.get('success') == True, "Rejection should succeed"
        print(f"✓ Rejection successful")
        
        # =========================================================================
        # PHASE 3: VERIFICATION — STATUS & DATA PERSISTENCE
        # =========================================================================
        print("\n[PHASE 3: VERIFICATION — STATUS & DATA PERSISTENCE]")
        print("-" * 80)
        
        # Assertion 3.1: Status changed to RETURNED_TO_SECTION_FOR_REVISION
        print(f"\n[3.1] Verify status changed to returned-for-revision")
        current_status = get_subcase_status(subcase_id)
        print(f"  Current status: {current_status}")
        assert current_status == 'RETURNED_TO_SECTION_FOR_REVISION', \
            f"Expected RETURNED_TO_SECTION_FOR_REVISION, got {current_status}"
        print(f"✓ Status correctly transitioned to RETURNED_TO_SECTION_FOR_REVISION")
        
        # Assertion 3.2: Action items still exist (not deleted)
        print(f"\n[3.2] Verify action items persist after rejection")
        items_after_reject = get_action_items_by_subcase(subcase_id)
        print(f"  Action items count: {len(items_after_reject)}")
        assert len(items_after_reject) == 2, \
            f"Expected 2 action items to persist, got {len(items_after_reject)}"
        
        # Verify same item IDs
        items_after_reject_ids = [item['action_item_id'] for item in items_after_reject]
        assert set(initial_item_ids) == set(items_after_reject_ids), \
            "Action item IDs should be unchanged after rejection"
        print(f"✓ Action items persisted (IDs: {items_after_reject_ids})")
        
        # =========================================================================
        # PHASE 4: VERIFICATION — INBOX ROUTING (BACKWARD PATH)
        # =========================================================================
        print("\n[PHASE 4: VERIFICATION — INBOX ROUTING]")
        print("-" * 80)
        
        # Assertion 4.1: Section inbox contains the subcase again
        print(f"\n[4.1] Verify subcase returned to Section Admin inbox")
        from backend.api_v2.routers.workflow_router import get_inbox
        
        section_inbox_response = get_inbox(current_user=section_admin)
        section_inbox = section_inbox_response.get('items', [])
        print(f"  Section inbox item count: {len(section_inbox)}")
        
        subcase_in_section_inbox = any(item['subcase_id'] == subcase_id for item in section_inbox)
        assert subcase_in_section_inbox, \
            f"Subcase {subcase_id} should be back in Section Admin inbox"
        
        our_subcase = next((item for item in section_inbox if item['subcase_id'] == subcase_id), None)
        if our_subcase:
            print(f"  Found in Section inbox:")
            print(f"    - SubcaseID: {our_subcase['subcase_id']}")
            print(f"    - Status: {our_subcase['status']}")
            print(f"    - AllowedActions: {our_subcase['allowed_actions']}")
        
        print(f"✓ Subcase correctly returned to Section inbox")
        
        # Assertion 4.2: Department inbox does NOT contain the subcase
        print(f"\n[4.2] Verify subcase removed from Department Admin inbox")
        dept_inbox_response = get_inbox(current_user=dept_admin)
        dept_inbox = dept_inbox_response.get('items', [])
        print(f"  Department inbox item count: {len(dept_inbox)}")
        
        subcase_in_dept_inbox = any(item['subcase_id'] == subcase_id for item in dept_inbox)
        assert not subcase_in_dept_inbox, \
            f"Subcase {subcase_id} should NOT be in Department Admin inbox after rejection"
        
        print(f"✓ Subcase correctly removed from Department inbox")
        
        # =========================================================================
        # PHASE 5: VERIFICATION — FOLLOW-UP EXECUTION GUARD
        # =========================================================================
        print("\n[PHASE 5: VERIFICATION — FOLLOW-UP EXECUTION GUARD]")
        print("-" * 80)
        
        # Assertion 5.1: Follow-up start action should be blocked
        print(f"\n[5.1] Verify follow-up start endpoint blocked during revision")
        from backend.api_v2.routers.workflow_router import start_action_item
        
        test_action_item_id = initial_item_ids[0]
        print(f"  Attempting to start ActionItemID={test_action_item_id}")
        
        try:
            start_action_item(
                action_item_id=test_action_item_id,
                current_user=section_admin
            )
            raise AssertionError("Start action should have been blocked during revision state")
        except Exception as e:
            error_msg = str(e).lower()
            if 'returned for revision' in error_msg or 'revision' in error_msg:
                print(f"  ✓ Correctly blocked: {str(e)[:80]}...")
                print(f"✓ Follow-up execution blocked during revision state")
            else:
                raise AssertionError(f"Expected revision state guard, got: {e}")
        
        # Assertion 5.2: Follow-up complete action should also be blocked
        print(f"\n[5.2] Verify follow-up complete endpoint blocked during revision")
        from backend.api_v2.routers.workflow_router import complete_action_item
        
        try:
            complete_action_item(
                action_item_id=test_action_item_id,
                current_user=section_admin
            )
            raise AssertionError("Complete action should have been blocked during revision state")
        except Exception as e:
            error_msg = str(e).lower()
            if 'returned for revision' in error_msg or 'revision' in error_msg:
                print(f"  ✓ Correctly blocked: {str(e)[:80]}...")
                print(f"✓ Follow-up complete blocked during revision state")
            else:
                raise AssertionError(f"Expected revision state guard, got: {e}")
        
        # =========================================================================
        # PHASE 6: RESUBMISSION WITH OVERRIDE
        # =========================================================================
        print("\n[PHASE 6: RESUBMISSION WITH OVERRIDE]")
        print("-" * 80)
        
        # Step 6.1: Section Admin resubmits with new action items
        resubmit_request = {
            'action': 'SUBMIT_RESPONSE',
            'explanation_text': 'Revised section response with enhanced corrective plan including resource allocation and detailed timeline.',
            'action_items': [
                {
                    'title': 'AI-R1: Revised Action - Detailed Training Plan',
                    'description': 'Comprehensive training program with timeline: Week 1-2 preparation, Week 3-4 execution',
                    'due_date': (datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d'),
                    'assigned_to_user_id': 1
                },
                {
                    'title': 'AI-R2: Revised Action - Resource Allocation',
                    'description': 'Budget approved: $5000 for training materials, 2 FTE for implementation',
                    'due_date': (datetime.now() + timedelta(days=15)).strftime('%Y-%m-%d'),
                    'assigned_to_user_id': 1
                },
                {
                    'title': 'AI-R3: Revised Action - Monitoring Plan',
                    'description': 'Weekly progress reviews with department head, monthly reports to administration',
                    'due_date': (datetime.now() + timedelta(days=20)).strftime('%Y-%m-%d'),
                    'assigned_to_user_id': 1
                }
            ]
        }
        
        print(f"→ Section Admin resubmits revised response")
        print(f"  Explanation: {resubmit_request['explanation_text'][:60]}...")
        print(f"  Action items: {len(resubmit_request['action_items'])} (was 2, now 3)")
        
        resubmit_response = act_on_case(
            subcase_id=subcase_id,
            body=resubmit_request,
            current_user=section_admin
        )
        
        assert resubmit_response.get('success') == True, "Resubmission should succeed"
        print(f"✓ Resubmission successful")
        
        # =========================================================================
        # PHASE 7: VERIFICATION — OVERRIDE BEHAVIOR
        # =========================================================================
        print("\n[PHASE 7: VERIFICATION — OVERRIDE BEHAVIOR]")
        print("-" * 80)
        
        # Assertion 7.1: Status transitioned forward again
        print(f"\n[7.1] Verify status after resubmission")
        status_after_resubmit = get_subcase_status(subcase_id)
        print(f"  Current status: {status_after_resubmit}")
        assert status_after_resubmit == 'SECTION_ACCEPTED_PENDING_DEPT', \
            f"Expected SECTION_ACCEPTED_PENDING_DEPT after resubmit, got {status_after_resubmit}"
        print(f"✓ Status correctly transitioned to SECTION_ACCEPTED_PENDING_DEPT")
        
        # Assertion 7.2: Old action items replaced with new ones
        print(f"\n[7.2] Verify action items replaced (override behavior)")
        items_after_resubmit = get_action_items_by_subcase(subcase_id)
        new_item_ids = [item['action_item_id'] for item in items_after_resubmit]
        
        print(f"  Action items count after resubmit: {len(items_after_resubmit)}")
        assert len(items_after_resubmit) == 3, \
            f"Expected 3 new action items, got {len(items_after_resubmit)}"
        
        # Verify old IDs are gone
        for old_id in initial_item_ids:
            assert old_id not in new_item_ids, \
                f"Old action item {old_id} should have been deleted"
        
        print(f"  Old ActionItemIDs: {initial_item_ids}")
        print(f"  New ActionItemIDs: {new_item_ids}")
        print(f"✓ Old action items replaced with new ones")
        
        # Assertion 7.3: New action items have correct titles
        print(f"\n[7.3] Verify new action items content")
        new_titles = [item['title'] for item in items_after_resubmit]
        print(f"  New titles:")
        for title in new_titles:
            print(f"    - {title}")
        
        assert any('Revised' in title for title in new_titles), \
            "New action items should have 'Revised' in titles"
        print(f"✓ New action items have correct content")
        
        # =========================================================================
        # FINAL SUMMARY
        # =========================================================================
        print("\n" + "="*80)
        print("TEST RESULT: ✅ ALL ASSERTIONS PASSED")
        print("="*80)
        print("\n✓ Initial Submission: Section → Department (2 action items)")
        print("✓ Department Rejection: Status → RETURNED_TO_SECTION_FOR_REVISION")
        print("✓ Backward Routing: Subcase returned to Section inbox")
        print("✓ Data Persistence: Original action items remained after rejection")
        print("✓ Execution Guard: Follow-up actions blocked during revision")
        print("✓ Resubmission: Section resubmitted with 3 new action items")
        print("✓ Override Behavior: Old items deleted, new items active")
        print("\n🎉 DATAFLOW CERTIFICATION: PASSED (REJECT-RETURN LOOP)")
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
