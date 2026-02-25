"""
PHASE 4.5-A3 — DEPARTMENT OVERRIDE FLOW CERTIFICATION

Validates that Department Administrator override action completely replaces
Section's action items and moves workflow forward to Administration level.

This is a DATAFLOW CERTIFICATION TEST. We verify:
- Override replaces (not appends) action items
- Old section items are deleted, only new department items remain
- Status transitions forward to DEPT_ACCEPTED_PENDING_ADMIN
- Subcase routed to Administration inbox
- Follow-up shows only override items

Test Flow:
1. Create subcase → Section submits response with 2 action items
2. Department overrides with 3 new action items
3. Verify old 2 items deleted, new 3 items active
4. Verify status → DEPT_ACCEPTED_PENDING_ADMIN
5. Verify routing: Admin inbox has it, Dept inbox doesn't
6. Verify follow-up returns only 3 override items

Run: python test_phase_4_5_a3_department_override.py
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
    print("PHASE 4.5-A3 — DEPARTMENT OVERRIDE FLOW CERTIFICATION")
    print("="*80)
    
    subcase_id = None
    
    try:
        # =========================================================================
        # PHASE 1: SETUP & SECTION SUBMISSION
        # =========================================================================
        print("\n[PHASE 1: SETUP & SECTION SUBMISSION]")
        print("-" * 80)
        
        # Step 1.1: Create test subcase
        subcase_id = create_test_subcase(status='SUBMITTED_TO_SECTION', target_org_unit_id=2)
        print(f"✓ Created test subcase: SubcaseID={subcase_id}")
        print(f"  Initial status: SUBMITTED_TO_SECTION")
        
        # Step 1.2: Create test users
        section_admin = create_test_user('SECTION_ADMIN', 2, 'Section')
        dept_admin = create_test_user('DEPARTMENT_ADMIN', 2, 'Department')
        # Admin sees org unit 2 (target of this subcase)
        admin_admin = create_test_user('ADMINISTRATION_ADMIN', 2, 'Administration')
        print(f"✓ Created Section Admin user")
        print(f"✓ Created Department Admin user")
        print(f"✓ Created Administration Admin user")
        
        # Step 1.3: Section Admin submits response with 2 action items
        from backend.api_v2.routers.workflow_router import act_on_case
        
        section_request = {
            'action': 'SUBMIT_RESPONSE',
            'explanation_text': 'Section response: We have identified the root cause and propose corrective measures.',
            'action_items': [
                {
                    'title': 'Section AI-1: Immediate Action',
                    'description': 'Section proposes immediate staff training on incident reporting',
                    'due_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
                    'assigned_to_user_id': 1
                },
                {
                    'title': 'Section AI-2: Documentation Update',
                    'description': 'Section proposes updating standard operating procedures',
                    'due_date': (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d'),
                    'assigned_to_user_id': 1
                }
            ]
        }
        
        print(f"\n→ Section Admin submits response")
        print(f"  Explanation: {section_request['explanation_text'][:50]}...")
        print(f"  Action items: {len(section_request['action_items'])}")
        
        response = act_on_case(
            subcase_id=subcase_id,
            body=section_request,
            current_user=section_admin
        )
        
        assert response.get('success') == True, "Section submission should succeed"
        print(f"✓ Section submission successful")
        
        # Verify status after section submission
        status_after_section = get_subcase_status(subcase_id)
        assert status_after_section == 'SECTION_ACCEPTED_PENDING_DEPT', \
            f"Expected SECTION_ACCEPTED_PENDING_DEPT, got {status_after_section}"
        print(f"✓ Status: {status_after_section}")
        
        # =========================================================================
        # PHASE 2: VERIFICATION — INITIAL ACTION ITEMS
        # =========================================================================
        print("\n[PHASE 2: VERIFICATION — INITIAL ACTION ITEMS]")
        print("-" * 80)
        
        # Assertion 2.1: Verify 2 action items created by section
        print(f"\n[2.1] Verify section action items count")
        section_action_items = get_action_items_by_subcase(subcase_id)
        section_item_ids = [item['action_item_id'] for item in section_action_items]
        
        print(f"  Section action items count: {len(section_action_items)}")
        print(f"  ActionItemIDs: {section_item_ids}")
        
        assert len(section_action_items) == 2, \
            f"Expected 2 section action items, got {len(section_action_items)}"
        
        for item in section_action_items:
            print(f"    - {item['action_item_id']}: {item['title']}")
        
        print(f"✓ Section created 2 action items as expected")
        
        # =========================================================================
        # PHASE 3: DEPARTMENT OVERRIDE
        # =========================================================================
        print("\n[PHASE 3: DEPARTMENT OVERRIDE]")
        print("-" * 80)
        
        # Step 3.1: Department Admin overrides with 3 new action items
        override_request = {
            'action': 'OVERRIDE',
            'explanation_text': 'Department overrides: Section plan is acceptable but we need additional measures for organizational-level compliance.',
            'action_items': [
                {
                    'title': 'Department AI-1: Policy Revision',
                    'description': 'Department mandates review and update of hospital-wide incident reporting policy',
                    'due_date': (datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d'),
                    'assigned_to_user_id': 1
                },
                {
                    'title': 'Department AI-2: Cross-Section Training',
                    'description': 'Department requires coordinated training across all sections in department',
                    'due_date': (datetime.now() + timedelta(days=15)).strftime('%Y-%m-%d'),
                    'assigned_to_user_id': 1
                },
                {
                    'title': 'Department AI-3: Compliance Audit',
                    'description': 'Department schedules quarterly compliance audit for next 12 months',
                    'due_date': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
                    'assigned_to_user_id': 1
                }
            ]
        }
        
        print(f"→ Department Admin overrides section response")
        print(f"  Explanation: {override_request['explanation_text'][:60]}...")
        print(f"  New action items: {len(override_request['action_items'])} (replacing 2 section items)")
        
        override_response = act_on_case(
            subcase_id=subcase_id,
            body=override_request,
            current_user=dept_admin
        )
        
        assert override_response.get('success') == True, "Override should succeed"
        print(f"✓ Department override successful")
        
        # =========================================================================
        # PHASE 4: VERIFICATION — OVERRIDE REPLACEMENT BEHAVIOR
        # =========================================================================
        print("\n[PHASE 4: VERIFICATION — OVERRIDE REPLACEMENT BEHAVIOR]")
        print("-" * 80)
        
        # Assertion 4.1: Status transitioned forward to admin level
        print(f"\n[4.1] Verify status after override")
        status_after_override = get_subcase_status(subcase_id)
        print(f"  Current status: {status_after_override}")
        assert status_after_override == 'DEPT_ACCEPTED_PENDING_ADMIN', \
            f"Expected DEPT_ACCEPTED_PENDING_ADMIN, got {status_after_override}"
        print(f"✓ Status correctly transitioned to DEPT_ACCEPTED_PENDING_ADMIN")
        
        # Assertion 4.2: Action item count is exactly 3 (override count)
        print(f"\n[4.2] Verify action items replaced (not appended)")
        override_action_items = get_action_items_by_subcase(subcase_id)
        override_item_ids = [item['action_item_id'] for item in override_action_items]
        
        print(f"  Action items count after override: {len(override_action_items)}")
        print(f"  New ActionItemIDs: {override_item_ids}")
        
        assert len(override_action_items) == 3, \
            f"Expected exactly 3 action items (override count), got {len(override_action_items)}"
        print(f"✓ Action item count = 3 (correct)")
        
        # Assertion 4.3: Old section items are gone
        print(f"\n[4.3] Verify old section items deleted")
        print(f"  Old section ActionItemIDs: {section_item_ids}")
        print(f"  New override ActionItemIDs: {override_item_ids}")
        
        for old_id in section_item_ids:
            assert old_id not in override_item_ids, \
                f"Old section action item {old_id} should have been deleted by override"
        
        print(f"✓ All old section items deleted (replacement confirmed)")
        
        # Assertion 4.4: All active items are from override batch
        print(f"\n[4.4] Verify all active items are department-created")
        for item in override_action_items:
            print(f"  - {item['action_item_id']}: {item['title']}")
            assert 'Department' in item['title'], \
                f"Action item {item['action_item_id']} should be department-created"
        
        print(f"✓ All active items belong to override batch")
        
        # =========================================================================
        # PHASE 5: VERIFICATION — INBOX ROUTING
        # =========================================================================
        print("\n[PHASE 5: VERIFICATION — INBOX ROUTING]")
        print("-" * 80)
        
        # Assertion 5.1: Administration inbox contains the subcase
        print(f"\n[5.1] Verify Administration Admin inbox contains subcase")
        from backend.api_v2.routers.workflow_router import get_inbox
        
        admin_inbox_response = get_inbox(current_user=admin_admin)
        admin_inbox = admin_inbox_response.get('items', [])
        print(f"  Administration inbox item count: {len(admin_inbox)}")
        
        subcase_in_admin_inbox = any(item['subcase_id'] == subcase_id for item in admin_inbox)
        assert subcase_in_admin_inbox, \
            f"Subcase {subcase_id} should be in Administration Admin inbox"
        
        our_subcase = next((item for item in admin_inbox if item['subcase_id'] == subcase_id), None)
        if our_subcase:
            print(f"  Found in Admin inbox:")
            print(f"    - SubcaseID: {our_subcase['subcase_id']}")
            print(f"    - Status: {our_subcase['status']}")
            print(f"    - AllowedActions: {our_subcase['allowed_actions']}")
        
        print(f"✓ Subcase correctly routed to Administration inbox")
        
        # Assertion 5.2: Department inbox does NOT contain the subcase
        print(f"\n[5.2] Verify Department Admin inbox does NOT contain subcase")
        dept_inbox_response = get_inbox(current_user=dept_admin)
        dept_inbox = dept_inbox_response.get('items', [])
        print(f"  Department inbox item count: {len(dept_inbox)}")
        
        subcase_in_dept_inbox = any(item['subcase_id'] == subcase_id for item in dept_inbox)
        assert not subcase_in_dept_inbox, \
            f"Subcase {subcase_id} should NOT be in Department Admin inbox after override"
        
        print(f"✓ Subcase correctly removed from Department inbox")
        
        # =========================================================================
        # PHASE 6: VERIFICATION — FOLLOW-UP ENDPOINT
        # =========================================================================
        print("\n[PHASE 6: VERIFICATION — FOLLOW-UP ENDPOINT]")
        print("-" * 80)
        
        # Assertion 6.1: Follow-up returns only override items (3 items)
        print(f"\n[6.1] Verify follow-up endpoint returns override items only")
        from backend.api_v2.routers.workflow_router import get_follow_up_items
        
        # Create worker user matching assigned_to_user_id
        worker_user = create_test_user('SECTION_ADMIN', 2, 'Section')
        worker_user.user_id = 1
        
        follow_up_response = get_follow_up_items(current_user=worker_user)
        follow_up_items = follow_up_response.get('items', [])
        print(f"  Total follow-up items: {len(follow_up_items)}")
        
        # Filter to our subcase's items
        our_follow_up_items = [item for item in follow_up_items if item.get('subcase_id') == subcase_id]
        print(f"  Our subcase's follow-up items: {len(our_follow_up_items)}")
        
        assert len(our_follow_up_items) == 3, \
            f"Expected exactly 3 follow-up items (override count), got {len(our_follow_up_items)}"
        
        for item in our_follow_up_items:
            print(f"  - ActionItemID={item.get('action_item_id')}: {item.get('title')}")
            assert 'Department' in item.get('title', ''), \
                "Follow-up items should be department-created (override batch)"
        
        print(f"✓ Follow-up endpoint returns only override items (3 items)")
        
        # =========================================================================
        # FINAL SUMMARY
        # =========================================================================
        print("\n" + "="*80)
        print("TEST RESULT: ✅ ALL ASSERTIONS PASSED")
        print("="*80)
        print("\n✓ Section Submission: 2 action items created")
        print("✓ Department Override: Replaced with 3 new action items")
        print("✓ Replacement Verified: Old items deleted, only new items active")
        print("✓ Status Transition: SECTION_ACCEPTED_PENDING_DEPT → DEPT_ACCEPTED_PENDING_ADMIN")
        print("✓ Inbox Routing: Moved to Administration level")
        print("✓ Follow-Up: Returns only override items (3 items)")
        print("\n🎉 DATAFLOW CERTIFICATION: PASSED (OVERRIDE FLOW)")
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
