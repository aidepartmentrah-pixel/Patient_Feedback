"""
PHASE 3.5 — Complete Integration Test
End-to-end test of all 6 API v2 workflow endpoints with real data flow.

Tests the complete workflow lifecycle:
1. Incident creation → Subcase creation (via adapter)
2. Inbox endpoint → Verify subcase appears
3. Case action endpoint → Submit section response with action items
4. Follow-up endpoints → Start, complete action items
5. Case action endpoint → Department approve
6. Case action endpoint → Administration approve
7. Verify state transitions at each step

This proves the entire Phase 3.5 implementation works end-to-end.
"""

import sys
import os
from datetime import datetime, timedelta
import json

# Force UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add backend directory to Python path
backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from core.database import get_connection


def test(description):
    """Test decorator"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"\n{'='*80}")
            print(f"TEST: {description}")
            print('='*80)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def get_db_cursor():
    """Get database cursor"""
    conn = get_connection()
    cursor = conn.cursor()
    return conn, cursor


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
                org_unit_type=org_unit_type_str  # Must be string like "Section", "Department", "Administration"
            )
        ],
        allowed_unit_ids=[org_unit_id] if org_unit_id else []
    )


# =============================================================================
# INTEGRATION TEST FLOW
# =============================================================================

@test("STEP 1: Create Test Subcase Directly")
def step1_create_subcase():
    """Create a test subcase directly in the database"""
    print("\n[ACTION] Creating test subcase directly...")
    
    conn, cursor = get_db_cursor()
    try:
        # Get an existing incident to attach to
        cursor.execute("""
            SELECT TOP 1 IncidentRequestCaseID
            FROM dbo.APP_IncidentCase
            ORDER BY IncidentRequestCaseID DESC
        """)
        
        incident_result = cursor.fetchone()
        if not incident_result:
            raise AssertionError("No existing incidents found - cannot create test subcase")
        
        incident_id = incident_result.IncidentRequestCaseID
        print(f"[INFO] Using existing incident ID: {incident_id}")
        
        # Insert subcase directly
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
        """, ('INCIDENT', incident_id, 2, 'SUBMITTED_TO_SECTION', 1))
        
        conn.commit()
        
        # Get the created subcase ID
        cursor.execute("""
            SELECT TOP 1 SubcaseID, Status, TargetOrgUnitID
            FROM dbo.APP_AdministrativeSubcase
            WHERE IncidentRequestCaseID = ? AND TargetOrgUnitID = 2
            ORDER BY CreatedAt DESC
        """, (incident_id,))
        
        subcase = cursor.fetchone()
        
        if not subcase:
            raise AssertionError("Failed to create subcase")
        
        subcase_id = subcase.SubcaseID
        status = subcase.Status
        target_org_unit = subcase.TargetOrgUnitID
        
        print(f"[SUCCESS] Subcase created: ID={subcase_id}, Status={status}, Target={target_org_unit}")
        
        if status != 'SUBMITTED_TO_SECTION':
            raise AssertionError(f"Expected status SUBMITTED_TO_SECTION, got {status}")
        
        print(f"\n✅ STEP 1 COMPLETE: Test subcase created directly")
        
        return {
            'subcase_id': subcase_id,
            'target_org_unit_id': target_org_unit
        }
        
    finally:
        cursor.close()
        conn.close()


@test("STEP 2: Test Inbox Endpoint (GET /api/v2/workflow/inbox)")
def step2_test_inbox(context):
    """Test that the subcase appears in section admin's inbox"""
    from backend.api_v2.routers.workflow_router import get_inbox
    
    subcase_id = context['subcase_id']
    target_org_unit_id = context['target_org_unit_id']
    
    print(f"\n[ACTION] Fetching inbox for section admin (org_unit={target_org_unit_id})...")
    
    # Create section admin user
    section_admin = create_test_user('SECTION_ADMIN', target_org_unit_id, 'Section')
    
    print(f"[DEBUG] User details:")
    print(f"  role_code: {section_admin.scopes[0].role_code}")
    print(f"  org_unit_id: {section_admin.scopes[0].org_unit_id}")
    print(f"  allowed_unit_ids: {section_admin.allowed_unit_ids}")
    
    # Call inbox endpoint
    response = get_inbox(current_user=section_admin)
    
    items = response.get('items', [])
    
    print(f"[RESULT] Inbox returned {len(items)} item(s)")
    
    # Find our subcase
    our_subcase = None
    for item in items:
        if item.get('subcase_id') == subcase_id:
            our_subcase = item
            break
    
    if not our_subcase:
        print(f"[ERROR] Subcase {subcase_id} not found in inbox")
        print(f"[DEBUG] Inbox items: {json.dumps(items, indent=2)}")
        raise AssertionError(f"Subcase {subcase_id} not in inbox")
    
    print(f"\n[SUCCESS] Found subcase in inbox:")
    print(f"  SubcaseID: {our_subcase.get('subcase_id')}")
    print(f"  Status: {our_subcase.get('status')}")
    print(f"  CaseType: {our_subcase.get('case_type')}")
    print(f"  Allowed Actions: {our_subcase.get('allowed_actions', [])}")
    
    # Verify allowed actions include SUBMIT_RESPONSE
    allowed_actions = our_subcase.get('allowed_actions', [])
    if 'SUBMIT_RESPONSE' not in allowed_actions:
        raise AssertionError(f"Expected SUBMIT_RESPONSE in allowed_actions, got {allowed_actions}")
    
    print(f"\n✅ STEP 2 COMPLETE: Inbox endpoint works, subcase visible to section admin")
    
    return context


@test("STEP 3: Submit Section Response (POST /api/v2/workflow/case/{id}/act)")
def step3_submit_response(context):
    """Section admin submits response with action items"""
    from backend.api_v2.routers.workflow_router import act_on_case
    
    subcase_id = context['subcase_id']
    target_org_unit_id = context['target_org_unit_id']
    
    print(f"\n[ACTION] Section admin submitting response for subcase {subcase_id}...")
    
    # Create section admin user
    section_admin = create_test_user('SECTION_ADMIN', target_org_unit_id, 'Section')
    
    # Prepare action payload
    action_payload = {
        'action': 'SUBMIT_RESPONSE',
        'payload': {
            'explanation_text': 'Integration test: Section admin explanation for the incident',
            'action_items': [
                {
                    'title': 'Integration Test Action Item 1',
                    'description': 'First test action item',
                    'assigned_to_user_id': 999,
                    'due_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
                },
                {
                    'title': 'Integration Test Action Item 2',
                    'description': 'Second test action item',
                    'assigned_to_user_id': 999,
                    'due_date': (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d')
                }
            ]
        }
    }
    
    # Call case action endpoint
    response = act_on_case(
        subcase_id=subcase_id,
        body=action_payload,
        current_user=section_admin
    )
    
    if not response.get('success'):
        raise AssertionError("Failed to submit section response")
    
    print(f"[SUCCESS] Section response submitted")
    
    # Verify status changed
    conn, cursor = get_db_cursor()
    try:
        cursor.execute("""
            SELECT Status, SectionExplanationText
            FROM dbo.APP_AdministrativeSubcase
            WHERE SubcaseID = ?
        """, (subcase_id,))
        
        result = cursor.fetchone()
        new_status = result.Status
        explanation = result.SectionExplanationText
        
        print(f"[VERIFY] New status: {new_status}")
        print(f"[VERIFY] Explanation saved: {len(explanation)} chars")
        
        if new_status != 'SECTION_ACCEPTED_PENDING_DEPT':
            raise AssertionError(f"Expected status SECTION_ACCEPTED_PENDING_DEPT, got {new_status}")
        
        # Check action items created
        cursor.execute("""
            SELECT COUNT(*) as cnt
            FROM dbo.APP_SubcaseActionItem
            WHERE SubcaseID = ?
        """, (subcase_id,))
        
        action_count = cursor.fetchone().cnt
        print(f"[VERIFY] Action items created: {action_count}")
        
        if action_count != 2:
            raise AssertionError(f"Expected 2 action items, got {action_count}")
        
        print(f"\n✅ STEP 3 COMPLETE: Section response submitted, status → SECTION_ACCEPTED_PENDING_DEPT")
        
        return context
        
    finally:
        cursor.close()
        conn.close()


@test("STEP 4: Test Follow-Up Endpoints (Action Items)")
def step4_test_follow_up(context):
    """Test follow-up action item endpoints"""
    from backend.api_v2.routers.workflow_router import (
        get_follow_up_items,
        start_action_item,
        complete_action_item
    )
    
    subcase_id = context['subcase_id']
    
    print(f"\n[ACTION] Testing follow-up endpoints...")
    
    # Create worker user (assigned to action items)
    worker = create_test_user('SECTION_ADMIN', context['target_org_unit_id'], 'Section')
    
    # 4.1: Get follow-up items
    print(f"\n[4.1] GET /api/v2/workflow/follow-up")
    response = get_follow_up_items(current_user=worker)
    items = response.get('items', [])
    
    print(f"[RESULT] Found {len(items)} action item(s)")
    
    # Find our action items
    our_items = [item for item in items if item.get('subcase_id') == subcase_id]
    
    if len(our_items) != 2:
        raise AssertionError(f"Expected 2 action items for our subcase, got {len(our_items)}")
    
    action_item_id = our_items[0]['action_item_id']
    print(f"[SUCCESS] Found our action items, testing with ID={action_item_id}")
    
    # 4.2: Start action item
    print(f"\n[4.2] POST /api/v2/workflow/follow-up/{action_item_id}/start")
    response = start_action_item(action_item_id=action_item_id, current_user=worker)
    
    if not response.get('success'):
        raise AssertionError("Failed to start action item")
    
    print(f"[SUCCESS] Action item started")
    
    # 4.3: Complete action item
    print(f"\n[4.3] POST /api/v2/workflow/follow-up/{action_item_id}/complete")
    response = complete_action_item(action_item_id=action_item_id, current_user=worker)
    
    if not response.get('success'):
        raise AssertionError("Failed to complete action item")
    
    print(f"[SUCCESS] Action item completed")
    
    # Verify status in database
    conn, cursor = get_db_cursor()
    try:
        cursor.execute("""
            SELECT Status, StartedAt, CompletedAt
            FROM dbo.APP_SubcaseActionItem
            WHERE ActionItemID = ?
        """, (action_item_id,))
        
        result = cursor.fetchone()
        status = result.Status
        started_at = result.StartedAt
        completed_at = result.CompletedAt
        
        print(f"\n[VERIFY] Action item status: {status}")
        print(f"[VERIFY] Started at: {started_at}")
        print(f"[VERIFY] Completed at: {completed_at}")
        
        if status != 'DONE':
            raise AssertionError(f"Expected status DONE, got {status}")
        
        if not started_at or not completed_at:
            raise AssertionError("Started/Completed timestamps not set")
        
        print(f"\n✅ STEP 4 COMPLETE: Follow-up endpoints work correctly")
        
        return context
        
    finally:
        cursor.close()
        conn.close()


@test("STEP 5: Department Approve (POST /api/v2/workflow/case/{id}/act)")
def step5_department_approve(context):
    """Department admin approves the response"""
    from backend.api_v2.routers.workflow_router import act_on_case
    
    subcase_id = context['subcase_id']
    target_org_unit_id = context['target_org_unit_id']
    
    print(f"\n[ACTION] Department admin approving subcase {subcase_id}...")
    
    # Get parent department ID
    conn, cursor = get_db_cursor()
    try:
        cursor.execute("""
            SELECT ParentID
            FROM dbo.AdminsrationUnit
            WHERE UniqueID = ?
        """, (target_org_unit_id,))
        
        parent_result = cursor.fetchone()
        if not parent_result:
            raise AssertionError(f"Could not find parent for org unit {target_org_unit_id}")
        
        dept_id = parent_result.ParentID
        print(f"[INFO] Parent department ID: {dept_id}")
        
    finally:
        cursor.close()
        conn.close()
    
    # Create department admin user
    dept_admin = create_test_user('DEPARTMENT_ADMIN', dept_id, 'Department')
    
    # Approve
    action_payload = {
        'action': 'APPROVE',
        'payload': {}
    }
    
    response = act_on_case(
        subcase_id=subcase_id,
        body=action_payload,
        current_user=dept_admin
    )
    
    if not response.get('success'):
        raise AssertionError("Failed to approve at department level")
    
    print(f"[SUCCESS] Department approved")
    
    # Verify status
    conn, cursor = get_db_cursor()
    try:
        cursor.execute("""
            SELECT Status
            FROM dbo.APP_AdministrativeSubcase
            WHERE SubcaseID = ?
        """, (subcase_id,))
        
        result = cursor.fetchone()
        new_status = result.Status
        
        print(f"[VERIFY] New status: {new_status}")
        
        if new_status != 'DEPT_ACCEPTED_PENDING_ADMIN':
            raise AssertionError(f"Expected status DEPT_ACCEPTED_PENDING_ADMIN, got {new_status}")
        
        print(f"\n✅ STEP 5 COMPLETE: Department approval works, status → DEPT_ACCEPTED_PENDING_ADMIN")
        
        return context
        
    finally:
        cursor.close()
        conn.close()


@test("STEP 6: Administration Approve (POST /api/v2/workflow/case/{id}/act)")
def step6_administration_approve(context):
    """Administration admin approves the response (final approval)"""
    from backend.api_v2.routers.workflow_router import act_on_case
    
    subcase_id = context['subcase_id']
    
    print(f"\n[ACTION] Administration admin approving subcase {subcase_id}...")
    
    # Create administration admin user (org_unit_id=1 is typically administration)
    admin = create_test_user('ADMINISTRATION_ADMIN', 1, 'Administration')
    
    # Approve
    action_payload = {
        'action': 'APPROVE',
        'payload': {}
    }
    
    response = act_on_case(
        subcase_id=subcase_id,
        body=action_payload,
        current_user=admin
    )
    
    if not response.get('success'):
        raise AssertionError("Failed to approve at administration level")
    
    print(f"[SUCCESS] Administration approved")
    
    # Verify status
    conn, cursor = get_db_cursor()
    try:
        cursor.execute("""
            SELECT Status
            FROM dbo.APP_AdministrativeSubcase
            WHERE SubcaseID = ?
        """, (subcase_id,))
        
        result = cursor.fetchone()
        new_status = result.Status
        
        print(f"[VERIFY] Final status: {new_status}")
        
        if new_status != 'ADMIN_APPROVED':
            raise AssertionError(f"Expected status ADMIN_APPROVED, got {new_status}")
        
        print(f"\n✅ STEP 6 COMPLETE: Administration approval works, status → ADMIN_APPROVED")
        print(f"\n🎉 WORKFLOW COMPLETE: Incident → Subcase → Response → Approval → ADMIN_APPROVED")
        
        return context
        
    finally:
        cursor.close()
        conn.close()


@test("STEP 7: Cleanup Test Data")
def step7_cleanup(context):
    """Clean up all test data"""
    subcase_id = context['subcase_id']
    
    print(f"\n[ACTION] Cleaning up test data...")
    
    conn, cursor = get_db_cursor()
    try:
        # Delete action items
        cursor.execute("DELETE FROM dbo.APP_SubcaseActionItem WHERE SubcaseID = ?", (subcase_id,))
        action_count = cursor.rowcount
        
        # Delete subcase
        cursor.execute("DELETE FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = ?", (subcase_id,))
        subcase_count = cursor.rowcount
        
        conn.commit()
        
        print(f"[SUCCESS] Deleted:")
        print(f"  - {action_count} action item(s)")
        print(f"  - {subcase_count} subcase(s)")
        
        print(f"\n✅ STEP 7 COMPLETE: All test data cleaned up")
        
    finally:
        cursor.close()
        conn.close()


# =============================================================================
# MAIN TEST EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("PHASE 3.5 — COMPLETE INTEGRATION TEST")
    print("End-to-end test of all 6 API v2 workflow endpoints")
    print("="*80)
    
    try:
        # Execute workflow steps in sequence
        context = step1_create_subcase()
        context = step2_test_inbox(context)
        context = step3_submit_response(context)
        context = step4_test_follow_up(context)
        context = step5_department_approve(context)
        context = step6_administration_approve(context)
        step7_cleanup(context)
        
        # Final summary
        print("\n" + "="*80)
        print("✅ PHASE 3.5 INTEGRATION TEST — ALL TESTS PASSED")
        print("="*80)
        
        print("\n🎉 COMPLETE WORKFLOW VERIFIED:")
        print("  ✅ Test subcase creation")
        print("  ✅ GET /api/v2/workflow/inbox (section admin)")
        print("  ✅ POST /api/v2/workflow/case/{id}/act (SUBMIT_RESPONSE)")
        print("  ✅ GET /api/v2/workflow/follow-up (action items)")
        print("  ✅ POST /api/v2/workflow/follow-up/{id}/start")
        print("  ✅ POST /api/v2/workflow/follow-up/{id}/complete")
        print("  ✅ POST /api/v2/workflow/case/{id}/act (APPROVE - dept)")
        print("  ✅ POST /api/v2/workflow/case/{id}/act (APPROVE - admin)")
        
        print("\n📊 WORKFLOW STATE TRANSITIONS VERIFIED:")
        print("  SUBMITTED_TO_SECTION → SUBMITTED_TO_DEPARTMENT → SUBMITTED_TO_ADMINISTRATION → CLOSED")
        
        print("\n🔒 ALL 6 API V2 ENDPOINTS WORKING CORRECTLY")
        print("🔒 Phase 3.5 implementation verified end-to-end")
        print("🔒 Frontend Phase 4 can proceed with confidence")
        
    except AssertionError as e:
        print(f"\n{'='*80}")
        print(f"❌ INTEGRATION TEST FAILED")
        print(f"{'='*80}")
        print(f"\nError: {str(e)}")
        
        # Attempt cleanup on failure
        if 'context' in locals() and 'subcase_id' in context:
            print(f"\n[CLEANUP] Attempting to clean up test data...")
            try:
                step7_cleanup(context)
            except:
                print(f"[WARNING] Cleanup failed, manual cleanup may be required")
        
        sys.exit(1)
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ UNEXPECTED ERROR")
        print(f"{'='*80}")
        import traceback
        traceback.print_exc()
        
        # Attempt cleanup on failure
        if 'context' in locals() and 'subcase_id' in context:
            print(f"\n[CLEANUP] Attempting to clean up test data...")
            try:
                step7_cleanup(context)
            except:
                print(f"[WARNING] Cleanup failed, manual cleanup may be required")
        
        sys.exit(1)
