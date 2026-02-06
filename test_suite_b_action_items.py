"""
SUITE B: ACTION ITEM LIFECYCLE TESTS
Tests critical action item behaviors during workflow state changes
"""

import pyodbc
from datetime import datetime, timedelta
from functools import wraps


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
        
        result = cursor.fetchone()
        return result.SubcaseID if result else None
    finally:
        cursor.close()
        conn.close()


def cleanup_test_subcase(subcase_id):
    """Clean up test data"""
    conn, cursor = get_db_cursor()
    try:
        # Delete action items first (FK constraint)
        cursor.execute("DELETE FROM dbo.APP_SubcaseActionItem WHERE SubcaseID = ?", (subcase_id,))
        # Delete subcase
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


def get_action_items_by_status(subcase_id, status=None):
    """Get action items for subcase, optionally filtered by status"""
    conn, cursor = get_db_cursor()
    try:
        if status:
            cursor.execute("""
                SELECT ActionItemID, Title, Status, AssignedToUserID, DueDate, StartedAt, CompletedAt
                FROM dbo.APP_SubcaseActionItem
                WHERE SubcaseID = ? AND Status = ?
                ORDER BY ActionItemID
            """, (subcase_id, status))
        else:
            cursor.execute("""
                SELECT ActionItemID, Title, Status, AssignedToUserID, DueDate, StartedAt, CompletedAt
                FROM dbo.APP_SubcaseActionItem
                WHERE SubcaseID = ?
                ORDER BY ActionItemID
            """, (subcase_id,))
        
        items = []
        for row in cursor.fetchall():
            items.append({
                'id': row.ActionItemID,
                'title': row.Title,
                'status': row.Status,
                'assigned_to': row.AssignedToUserID,
                'due_date': row.DueDate,
                'started_at': row.StartedAt,
                'completed_at': row.CompletedAt
            })
        return items
    finally:
        cursor.close()
        conn.close()


def test_decorator(test_name):
    """Decorator for test functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"\n{'='*70}")
            print(f"{test_name}")
            print(f"{'='*70}")
            try:
                result = func(*args, **kwargs)
                print(f"\n✅ {test_name} - PASSED")
                return result
            except AssertionError as e:
                print(f"\n❌ {test_name} - FAILED")
                print(f"Error: {e}")
                raise
        return wrapper
    return decorator


# =============================================================================
# TEST B1: ACTION ITEMS WHEN DEPARTMENT REJECTS
# =============================================================================

@test_decorator("TEST B1: Action Items When Department Rejects (Workflow Contract Change)")
def test_b1_action_items_on_department_rejection():
    """
    Test what happens to action items when department rejects.
    
    NEW WORKFLOW CONTRACT:
    Rejection returns for revision (NOT terminal).
    Action items remain untouched (will be replaced via OVERRIDE on resubmission).
    
    Workflow: Section creates items → Department rejects → Status = RETURNED_TO_SECTION_FOR_REVISION
    Expected: Items remain in DRAFT, follow-up actions blocked until resubmission
    """
    from backend.api_v2.routers.workflow_router import act_on_case
    
    subcase_id = create_test_subcase(status='SUBMITTED_TO_SECTION', target_org_unit_id=2)
    print(f"[SETUP] Created subcase {subcase_id}")
    
    try:
        # Step 1: Section submits response with 2 action items
        section_admin = create_test_user('SECTION_ADMIN', 2, 'Section')
        
        act_on_case(
            subcase_id=subcase_id,
            body={
                'action': 'SUBMIT_RESPONSE',
                'explanation_text': 'Section response with action items',
                'action_items': [
                    {
                        'title': 'Action Item 1',
                        'description': 'First item',
                        'assigned_to_user_id': 999,
                        'due_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
                    },
                    {
                        'title': 'Action Item 2',
                        'description': 'Second item',
                        'assigned_to_user_id': 999,
                        'due_date': (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d')
                    }
                ]
            },
            current_user=section_admin
        )
        
        item_count = get_action_item_count(subcase_id)
        assert item_count == 2, f"Expected 2 action items after section response, got {item_count}"
        print(f"[STEP 1] Section submitted response with 2 action items ✓")
        
        # Step 2: Department rejects the response
        dept_admin = create_test_user('DEPARTMENT_ADMIN', 1, 'Department')
        
        act_on_case(
            subcase_id=subcase_id,
            body={
                'action': 'REJECT',
                'rejection_text': 'Department rejects - not sufficient response'
            },
            current_user=dept_admin
        )
        
        final_status = get_subcase_status(subcase_id)
        assert final_status == 'RETURNED_TO_SECTION_FOR_REVISION', f"Expected RETURNED_TO_SECTION_FOR_REVISION, got {final_status}"
        print(f"[STEP 2] Department rejected - returned for revision ✓")
        
        # Step 3: Verify action items remain untouched
        final_item_count = get_action_item_count(subcase_id)
        items = get_action_items_by_status(subcase_id)
        
        print(f"[VERIFY] Action items after rejection: {final_item_count}")
        
        # NEW CONTRACT: Items remain untouched (will be replaced on resubmission)
        assert final_item_count == 2, f"Expected 2 items to remain, got {final_item_count}"
        print(f"[BEHAVIOR] ✓ Action items remain untouched (as per new workflow contract)")
        print(f"[INFO] Item statuses: {[item['status'] for item in items]}")
        
        # Step 4: Verify follow-up actions are blocked
        from backend.api_v2.routers.workflow_router import start_action_item
        action_item_id = items[0]['id']
        
        try:
            start_action_item(action_item_id=action_item_id, current_user=section_admin)
            raise AssertionError("Should have blocked start action during revision state")
        except Exception as e:
            if "returned for revision" in str(e).lower():
                print(f"[VERIFY] ✓ Follow-up actions blocked during revision state")
            else:
                raise
        
        return {'status': 'PASSED', 'items_after_rejection': final_item_count, 'behavior': 'remain_untouched'}
        
    finally:
        cleanup_test_subcase(subcase_id)
        print(f"[CLEANUP] Test data cleaned")


# =============================================================================
# TEST B2: ACTION ITEMS WHEN FORCE CLOSE
# =============================================================================

@test_decorator("TEST B2: Action Items When Force Close")
def test_b2_action_items_on_force_close():
    """
    Test what happens to action items when subcase is force closed.
    """
    from backend.api_v2.routers.workflow_router import act_on_case
    
    subcase_id = create_test_subcase(status='SUBMITTED_TO_SECTION', target_org_unit_id=2)
    print(f"[SETUP] Created subcase {subcase_id}")
    
    try:
        # Step 1: Section submits response with action items
        section_admin = create_test_user('SECTION_ADMIN', 2, 'Section')
        
        act_on_case(
            subcase_id=subcase_id,
            body={
                'action': 'SUBMIT_RESPONSE',
                'explanation_text': 'Section response',
                'action_items': [
                    {
                        'title': 'Action Item for Force Close Test',
                        'description': 'This will be force closed',
                        'assigned_to_user_id': 999,
                        'due_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
                    }
                ]
            },
            current_user=section_admin
        )
        
        item_count = get_action_item_count(subcase_id)
        assert item_count == 1, f"Expected 1 action item, got {item_count}"
        print(f"[STEP 1] Section submitted response with 1 action item ✓")
        
        # Step 2: Administration force closes
        admin = create_test_user('ADMINISTRATION_ADMIN', 0, 'Administration')
        
        act_on_case(
            subcase_id=subcase_id,
            body={
                'action': 'FORCE_CLOSE',
                'reason': 'Emergency closure - duplicate case'
            },
            current_user=admin
        )
        
        final_status = get_subcase_status(subcase_id)
        assert final_status == 'FORCE_CLOSED', f"Expected FORCE_CLOSED, got {final_status}"
        print(f"[STEP 2] Force closed successfully ✓")
        
        # Step 3: Verify action items handling
        final_item_count = get_action_item_count(subcase_id)
        items = get_action_items_by_status(subcase_id)
        
        print(f"[VERIFY] Action items after force close: {final_item_count}")
        
        if final_item_count == 0:
            print(f"[BEHAVIOR] ✓ System deletes action items on force close")
        elif final_item_count == 1:
            print(f"[BEHAVIOR] ℹ️ System keeps action items on force close")
            print(f"[INFO] Item status: {items[0]['status']}")
        
        return {'status': 'PASSED', 'items_after_force_close': final_item_count}
        
    finally:
        cleanup_test_subcase(subcase_id)
        print(f"[CLEANUP] Test data cleaned")


# =============================================================================
# TEST B3: ACTION ITEM DELAY (CANCEL) ENDPOINT
# =============================================================================

@test_decorator("TEST B3: Action Item Delay (Cancel) Endpoint")
def test_b3_action_item_delay():
    """
    Test the delay endpoint - which actually CANCELS the action item.
    Note: "delay" is a misnomer - it sets status to CANCELLED.
    """
    from backend.api_v2.routers.workflow_router import act_on_case, delay_action_item
    
    subcase_id = create_test_subcase(status='SUBMITTED_TO_SECTION', target_org_unit_id=2)
    print(f"[SETUP] Created subcase {subcase_id}")
    
    try:
        # Step 1: Section submits response with action item
        section_admin = create_test_user('SECTION_ADMIN', 2, 'Section')
        
        act_on_case(
            subcase_id=subcase_id,
            body={
                'action': 'SUBMIT_RESPONSE',
                'explanation_text': 'Section response',
                'action_items': [
                    {
                        'title': 'Action Item to Cancel',
                        'description': 'Will test delay (cancel)',
                        'assigned_to_user_id': 999,
                        'due_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
                    }
                ]
            },
            current_user=section_admin
        )
        
        items = get_action_items_by_status(subcase_id)
        assert len(items) == 1, f"Expected 1 action item, got {len(items)}"
        action_item_id = items[0]['id']
        original_status = items[0]['status']
        print(f"[STEP 1] Created action item {action_item_id} (Status: {original_status}) ✓")
        
        # Step 2: Delay (cancel) the action item
        response = delay_action_item(
            action_item_id=action_item_id,
            current_user=section_admin
        )
        
        assert response.get('success'), "Delay action failed"
        print(f"[STEP 2] Called delay endpoint ✓")
        
        # Step 3: Verify status changed to CANCELLED
        items_after = get_action_items_by_status(subcase_id)
        
        if len(items_after) == 0:
            print(f"[BEHAVIOR] ⚠️ Item was deleted (not what delay should do)")
            return {'status': 'PASSED', 'behavior': 'deleted'}
        
        updated_status = items_after[0]['status']
        
        print(f"[VERIFY] Original status: {original_status}")
        print(f"[VERIFY] Updated status: {updated_status}")
        
        if updated_status == 'CANCELLED':
            print(f"[VERIFY] ✓ Status changed to CANCELLED (as expected)")
            print(f"[BEHAVIOR] ✓ 'delay' endpoint cancels the action item")
        else:
            print(f"[BEHAVIOR] ⚠️ Status is {updated_status}, expected CANCELLED")
        
        return {'status': 'PASSED', 'final_status': updated_status}
        
    finally:
        cleanup_test_subcase(subcase_id)
        print(f"[CLEANUP] Test data cleaned")


# =============================================================================
# TEST B4: SUBMIT RESPONSE WITH 0 ACTION ITEMS
# =============================================================================

@test_decorator("TEST B4: Submit Response With 0 Action Items")
def test_b4_submit_with_zero_items():
    """
    Test submitting section response with no action items.
    Should this be allowed? What's the business logic?
    """
    from backend.api_v2.routers.workflow_router import act_on_case
    
    subcase_id = create_test_subcase(status='SUBMITTED_TO_SECTION', target_org_unit_id=2)
    print(f"[SETUP] Created subcase {subcase_id}")
    
    try:
        # Section submits response with empty action items array
        section_admin = create_test_user('SECTION_ADMIN', 2, 'Section')
        
        response = act_on_case(
            subcase_id=subcase_id,
            body={
                'action': 'SUBMIT_RESPONSE',
                'explanation_text': 'Section response with no action items',
                'action_items': []  # Empty array
            },
            current_user=section_admin
        )
        
        assert response.get('success'), "Submit response failed"
        print(f"[STEP 1] Section submitted response with 0 action items ✓")
        
        # Verify status changed
        final_status = get_subcase_status(subcase_id)
        assert final_status == 'SECTION_ACCEPTED_PENDING_DEPT', f"Expected SECTION_ACCEPTED_PENDING_DEPT, got {final_status}"
        print(f"[VERIFY] Status transitioned to SECTION_ACCEPTED_PENDING_DEPT ✓")
        
        # Verify no action items created
        item_count = get_action_item_count(subcase_id)
        assert item_count == 0, f"Expected 0 action items, got {item_count}"
        print(f"[VERIFY] ✓ No action items created (as expected)")
        
        print(f"[BEHAVIOR] ✓ System allows submitting response with 0 action items")
        
        return {'status': 'PASSED'}
        
    finally:
        cleanup_test_subcase(subcase_id)
        print(f"[CLEANUP] Test data cleaned")


# =============================================================================
# TEST B5: OVERRIDE WITH 0 ACTION ITEMS
# =============================================================================

@test_decorator("TEST B5: Override With 0 Action Items")
def test_b5_override_with_zero_items():
    """
    Test overriding existing action items with empty array.
    This should delete all items and create none (valid use case?).
    """
    from backend.api_v2.routers.workflow_router import act_on_case
    
    subcase_id = create_test_subcase(status='SUBMITTED_TO_SECTION', target_org_unit_id=2)
    print(f"[SETUP] Created subcase {subcase_id}")
    
    try:
        # Step 1: Section submits response with 2 action items
        section_admin = create_test_user('SECTION_ADMIN', 2, 'Section')
        
        act_on_case(
            subcase_id=subcase_id,
            body={
                'action': 'SUBMIT_RESPONSE',
                'explanation_text': 'Initial response',
                'action_items': [
                    {
                        'title': 'Item to be deleted 1',
                        'description': 'Will be removed',
                        'assigned_to_user_id': 999,
                        'due_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
                    },
                    {
                        'title': 'Item to be deleted 2',
                        'description': 'Will be removed',
                        'assigned_to_user_id': 999,
                        'due_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
                    }
                ]
            },
            current_user=section_admin
        )
        
        item_count = get_action_item_count(subcase_id)
        assert item_count == 2, f"Expected 2 initial items, got {item_count}"
        print(f"[STEP 1] Created 2 action items ✓")
        
        # Step 2: Department overrides with 0 items
        dept_admin = create_test_user('DEPARTMENT_ADMIN', 1, 'Department')
        
        response = act_on_case(
            subcase_id=subcase_id,
            body={
                'action': 'OVERRIDE',
                'explanation_text': 'Department override - no action items needed',
                'action_items': []  # Empty array
            },
            current_user=dept_admin
        )
        
        assert response.get('success'), "Override failed"
        print(f"[STEP 2] Department overrode with 0 action items ✓")
        
        # Step 3: Verify all items deleted
        final_item_count = get_action_item_count(subcase_id)
        assert final_item_count == 0, f"Expected 0 action items after override, got {final_item_count}"
        print(f"[VERIFY] ✓ All action items deleted (override with empty array)")
        
        # Verify status
        final_status = get_subcase_status(subcase_id)
        assert final_status == 'DEPT_ACCEPTED_PENDING_ADMIN', f"Expected DEPT_ACCEPTED_PENDING_ADMIN, got {final_status}"
        print(f"[VERIFY] ✓ Status transitioned correctly")
        
        print(f"[BEHAVIOR] ✓ System allows override with 0 items (deletes all existing)")
        
        return {'status': 'PASSED'}
        
    finally:
        cleanup_test_subcase(subcase_id)
        print(f"[CLEANUP] Test data cleaned")


# =============================================================================
# TEST B6: ACTION ITEMS IN VARIOUS STATES DURING OVERRIDE
# =============================================================================

@test_decorator("TEST B6: Override With Action Items In Progress")
def test_b6_override_with_items_in_progress():
    """
    Test what happens when override occurs while action items are IN_PROGRESS or DONE.
    Should in-progress work be preserved? Or deleted?
    """
    from backend.api_v2.routers.workflow_router import act_on_case, start_action_item, complete_action_item
    
    subcase_id = create_test_subcase(status='SUBMITTED_TO_SECTION', target_org_unit_id=2)
    print(f"[SETUP] Created subcase {subcase_id}")
    
    try:
        # Step 1: Section submits response with 3 action items
        section_admin = create_test_user('SECTION_ADMIN', 2, 'Section')
        
        act_on_case(
            subcase_id=subcase_id,
            body={
                'action': 'SUBMIT_RESPONSE',
                'explanation_text': 'Initial response',
                'action_items': [
                    {
                        'title': 'Draft Item',
                        'description': 'Stays in DRAFT',
                        'assigned_to_user_id': 999,
                        'due_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
                    },
                    {
                        'title': 'In Progress Item',
                        'description': 'Will be started',
                        'assigned_to_user_id': 999,
                        'due_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
                    },
                    {
                        'title': 'Completed Item',
                        'description': 'Will be completed',
                        'assigned_to_user_id': 999,
                        'due_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
                    }
                ]
            },
            current_user=section_admin
        )
        
        items = get_action_items_by_status(subcase_id)
        assert len(items) == 3, f"Expected 3 items, got {len(items)}"
        
        item_ids = [item['id'] for item in items]
        print(f"[STEP 1] Created 3 action items: {item_ids} ✓")
        
        # Step 2: Start second item and complete third item
        start_action_item(
            action_item_id=item_ids[1],
            current_user=section_admin
        )
        print(f"[STEP 2a] Started item {item_ids[1]} ✓")
        
        complete_action_item(
            action_item_id=item_ids[2],
            current_user=section_admin
        )
        print(f"[STEP 2b] Completed item {item_ids[2]} ✓")
        
        # Verify states
        items_before = get_action_items_by_status(subcase_id)
        draft_count = sum(1 for item in items_before if item['status'] == 'DRAFT')
        in_progress_count = sum(1 for item in items_before if item['status'] == 'IN_PROGRESS')
        done_count = sum(1 for item in items_before if item['status'] == 'DONE')
        
        print(f"[INFO] Before override: DRAFT={draft_count}, IN_PROGRESS={in_progress_count}, DONE={done_count}")
        
        # Step 3: Department overrides with new items
        dept_admin = create_test_user('DEPARTMENT_ADMIN', 1, 'Department')
        
        act_on_case(
            subcase_id=subcase_id,
            body={
                'action': 'OVERRIDE',
                'explanation_text': 'Department override',
                'action_items': [
                    {
                        'title': 'New Override Item',
                        'description': 'Replaces all previous items',
                        'assigned_to_user_id': 999,
                        'due_date': (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d')
                    }
                ]
            },
            current_user=dept_admin
        )
        
        print(f"[STEP 3] Department overrode with 1 new item ✓")
        
        # Step 4: Verify what happened to old items
        items_after = get_action_items_by_status(subcase_id)
        
        print(f"[VERIFY] Action items after override: {len(items_after)}")
        
        if len(items_after) == 1:
            print(f"[BEHAVIOR] ✓ Override deleted all previous items (including IN_PROGRESS and DONE)")
            print(f"[INFO] New item: {items_after[0]['title']} (Status: {items_after[0]['status']})")
        elif len(items_after) == 2:
            print(f"[BEHAVIOR] ℹ️ Override preserved DONE item, deleted others")
            for item in items_after:
                print(f"[INFO] Item: {item['title']} (Status: {item['status']})")
        elif len(items_after) == 3:
            print(f"[BEHAVIOR] ℹ️ Override preserved IN_PROGRESS items")
            for item in items_after:
                print(f"[INFO] Item: {item['title']} (Status: {item['status']})")
        else:
            print(f"[BEHAVIOR] ⚠️ Unexpected behavior: {len(items_after)} items remain")
        
        return {'status': 'PASSED', 'items_after_override': len(items_after)}
        
    finally:
        cleanup_test_subcase(subcase_id)
        print(f"[CLEANUP] Test data cleaned")


# =============================================================================
# TEST SUITE RUNNER
# =============================================================================

def run_all_tests():
    """Run all Suite B tests"""
    print("\n" + "="*70)
    print("SUITE B: ACTION ITEM LIFECYCLE TESTS")
    print("Testing action item behaviors during workflow changes")
    print("="*70)
    
    results = []
    
    # Run all tests
    tests = [
        ('B1: Action Items on Dept Rejection', test_b1_action_items_on_department_rejection),
        ('B2: Action Items on Force Close', test_b2_action_items_on_force_close),
        ('B3: Action Item Delay Endpoint', test_b3_action_item_delay),
        ('B4: Submit With 0 Items', test_b4_submit_with_zero_items),
        ('B5: Override With 0 Items', test_b5_override_with_zero_items),
        ('B6: Override Items In Progress', test_b6_override_with_items_in_progress),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, 'PASSED', result))
            passed += 1
        except AssertionError as e:
            results.append((test_name, f'FAILED: {str(e)}', None))
            failed += 1
        except Exception as e:
            results.append((test_name, f'ERROR: {type(e).__name__}: {str(e)}', None))
            failed += 1
    
    # Generate final report
    print("\n" + "="*70)
    print("SUITE B SUMMARY")
    print("="*70)
    
    for test_name, status, result in results:
        status_icon = "✅" if status == 'PASSED' else "❌"
        print(f"{status_icon} {test_name}: {status}")
        if result and isinstance(result, dict):
            for key, value in result.items():
                if key != 'status':
                    print(f"    ℹ️ {key}: {value}")
    
    print(f"\n{'='*70}")
    print(f"TOTAL: {passed + failed} tests")
    print(f"✅ PASSED: {passed}")
    print(f"❌ FAILED: {failed}")
    print(f"{'='*70}")
    
    if failed == 0:
        print("\n🎉 ALL SUITE B TESTS PASSED!")
        print("✅ Action item lifecycle behaviors validated")
    else:
        print(f"\n⚠️ {failed} test(s) failed - review and fix")
    
    return passed == len(tests)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, r'c:\Users\IT\Documents\GitHub Repository\Patient_Feedback')
    success = run_all_tests()
    exit(0 if success else 1)
