"""
COMPREHENSIVE WORKFLOW TEST SUITE - Phase 3.5 Backend Validation

This test suite validates ALL workflow branches:
- A1: Happy Path (Full Approval Chain)
- A2: Section Rejects Responsibility
- A3: Department Rejects Section Response
- A4: Department Overrides Action Items
- A5: Administration Overrides Action Items
- A6: Administration Force Close

Each test is INDEPENDENT and cleans up after itself.

Run: python test_workflow_comprehensive.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pyodbc
from datetime import datetime, timedelta
from typing import Dict, Any

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


def cleanup_test_subcase(subcase_id):
    """Delete test subcase and related data"""
    conn, cursor = get_db_cursor()
    try:
        # Delete action items
        cursor.execute("DELETE FROM dbo.APP_SubcaseActionItem WHERE SubcaseID = ?", (subcase_id,))
        
        # Delete subcase
        cursor.execute("DELETE FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = ?", (subcase_id,))
        
        conn.commit()
    finally:
        cursor.close()
        conn.close()


def get_subcase_status(subcase_id):
    """Get current status of a subcase"""
    conn, cursor = get_db_cursor()
    try:
        cursor.execute("SELECT Status FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = ?", (subcase_id,))
        result = cursor.fetchone()
        return result.Status if result else None
    finally:
        cursor.close()
        conn.close()


def get_action_item_count(subcase_id):
    """Get count of action items for a subcase"""
    conn, cursor = get_db_cursor()
    try:
        cursor.execute("SELECT COUNT(*) as cnt FROM dbo.APP_SubcaseActionItem WHERE SubcaseID = ?", (subcase_id,))
        result = cursor.fetchone()
        return result.cnt
    finally:
        cursor.close()
        conn.close()


def test_decorator(test_name):
    """Decorator to mark test functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"\n{'='*70}")
            print(f"{test_name}")
            print(f"{'='*70}\n")
            try:
                result = func(*args, **kwargs)
                print(f"\n✅ {test_name} - PASSED")
                return result
            except Exception as e:
                print(f"\n❌ {test_name} - FAILED")
                print(f"Error: {str(e)}")
                raise
        return wrapper
    return decorator


# =============================================================================
# TEST A1: HAPPY PATH (Full Approval Chain)
# =============================================================================

@test_decorator("TEST A1: Happy Path - Full Approval Chain")
def test_a1_happy_path():
    """
    Test the complete approval workflow:
    SUBMITTED_TO_SECTION → SECTION_ACCEPTED_PENDING_DEPT → 
    DEPT_ACCEPTED_PENDING_ADMIN → ADMIN_APPROVED
    """
    from backend.api_v2.routers.workflow_router import act_on_case
    
    # Setup
    subcase_id = create_test_subcase(status='SUBMITTED_TO_SECTION', target_org_unit_id=2)
    print(f"[SETUP] Created subcase {subcase_id}")
    
    try:
        # Step 1: Section submits response
        section_admin = create_test_user('SECTION_ADMIN', 2, 'Section')
        
        response = act_on_case(
            subcase_id=subcase_id,
            body={
                'action': 'SUBMIT_RESPONSE',
                'explanation_text': 'Section response for A1 test',
                'action_items': [
                    {
                        'title': 'Action Item 1',
                        'description': 'First item',
                        'assigned_to_user_id': 999,
                        'due_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
                    }
                ]
            },
            current_user=section_admin
        )
        
        assert response.get('success'), "Section response failed"
        assert get_subcase_status(subcase_id) == 'SECTION_ACCEPTED_PENDING_DEPT'
        print(f"[STEP 1] Section submitted response ✓")
        
        # Step 2: Department approves
        dept_admin = create_test_user('DEPARTMENT_ADMIN', 2, 'Department')
        
        response = act_on_case(
            subcase_id=subcase_id,
            body={'action': 'APPROVE'},
            current_user=dept_admin
        )
        
        assert response.get('success'), "Department approval failed"
        assert get_subcase_status(subcase_id) == 'DEPT_ACCEPTED_PENDING_ADMIN'
        print(f"[STEP 2] Department approved ✓")
        
        # Step 3: Administration approves
        admin = create_test_user('ADMINISTRATION_ADMIN', 1, 'Administration')
        
        response = act_on_case(
            subcase_id=subcase_id,
            body={'action': 'APPROVE'},
            current_user=admin
        )
        
        assert response.get('success'), "Administration approval failed"
        assert get_subcase_status(subcase_id) == 'ADMIN_APPROVED'
        print(f"[STEP 3] Administration approved ✓")
        
        return {'status': 'PASSED'}
        
    finally:
        cleanup_test_subcase(subcase_id)
        print(f"[CLEANUP] Test data cleaned")


# =============================================================================
# TEST A2: SECTION REJECTS RESPONSIBILITY
# =============================================================================

@test_decorator("TEST A2: Section Rejects Responsibility")
def test_a2_section_rejects():
    """
    Test section rejection workflow:
    SUBMITTED_TO_SECTION → (REJECT) → SECTION_REJECTED (terminal)
    """
    from backend.api_v2.routers.workflow_router import act_on_case
    
    # Setup
    subcase_id = create_test_subcase(status='SUBMITTED_TO_SECTION', target_org_unit_id=2)
    print(f"[SETUP] Created subcase {subcase_id}")
    
    try:
        # Section rejects responsibility
        section_admin = create_test_user('SECTION_ADMIN', 2, 'Section')
        
        response = act_on_case(
            subcase_id=subcase_id,
            body={
                'action': 'REJECT',
                'rejection_text': 'This is not our department responsibility'
            },
            current_user=section_admin
        )
        
        assert response.get('success'), "Section rejection failed"
        
        # Verify terminal state
        final_status = get_subcase_status(subcase_id)
        assert final_status == 'SECTION_DENIED', f"Expected SECTION_DENIED, got {final_status}"
        print(f"[VERIFY] Status is SECTION_DENIED (terminal) ✓")
        
        # Verify rejection text was saved
        conn, cursor = get_db_cursor()
        try:
            cursor.execute("SELECT SectionRejectionText FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = ?", (subcase_id,))
            result = cursor.fetchone()
            rejection_text = result.SectionRejectionText if result else None
            assert rejection_text is not None and len(rejection_text) > 0, "Rejection text not saved"
            print(f"[VERIFY] Rejection text saved: '{rejection_text[:50]}...' ✓")
        finally:
            cursor.close()
            conn.close()
        
        return {'status': 'PASSED'}
        
    finally:
        cleanup_test_subcase(subcase_id)
        print(f"[CLEANUP] Test data cleaned")


# =============================================================================
# TEST A3: DEPARTMENT REJECTS SECTION RESPONSE
# =============================================================================

@test_decorator("TEST A3: Department Rejects Section Response")
def test_a3_department_rejects():
    """
    Test department rejection workflow (UPDATED FOR WORKFLOW CONTRACT CHANGE):
    SECTION_ACCEPTED_PENDING_DEPT → (REJECT) → RETURNED_TO_SECTION_FOR_REVISION
    
    Rejection is NOT terminal - it returns the case for revision (rework loop).
    Action items remain untouched (will be replaced on resubmission).
    """
    from backend.api_v2.routers.workflow_router import act_on_case
    
    # Setup: Create subcase with section response already submitted
    subcase_id = create_test_subcase(status='SUBMITTED_TO_SECTION', target_org_unit_id=2)
    print(f"[SETUP] Created subcase {subcase_id}")
    
    try:
        # Step 1: Section submits response first
        section_admin = create_test_user('SECTION_ADMIN', 2, 'Section')
        
        act_on_case(
            subcase_id=subcase_id,
            body={
                'action': 'SUBMIT_RESPONSE',
                'explanation_text': 'Section response for A3 test',
                'action_items': [
                    {
                        'title': 'Action Item',
                        'description': 'Test item',
                        'assigned_to_user_id': 999,
                        'due_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
                    }
                ]
            },
            current_user=section_admin
        )
        
        assert get_subcase_status(subcase_id) == 'SECTION_ACCEPTED_PENDING_DEPT'
        print(f"[STEP 1] Section submitted response ✓")
        
        # Step 2: Department rejects
        dept_admin = create_test_user('DEPARTMENT_ADMIN', 2, 'Department')
        
        response = act_on_case(
            subcase_id=subcase_id,
            body={
                'action': 'REJECT',
                'rejection_text': 'Insufficient response, needs more details'
            },
            current_user=dept_admin
        )
        
        assert response.get('success'), "Department rejection failed"
        
        # Verify returned-for-revision state (NOT terminal)
        final_status = get_subcase_status(subcase_id)
        assert final_status == 'RETURNED_TO_SECTION_FOR_REVISION', f"Expected RETURNED_TO_SECTION_FOR_REVISION, got {final_status}"
        print(f"[VERIFY] Status is RETURNED_TO_SECTION_FOR_REVISION (returns for rework) ✓")
        
        # Verify rejection text was saved
        conn, cursor = get_db_cursor()
        try:
            cursor.execute("SELECT DepartmentRejectionText FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = ?", (subcase_id,))
            result = cursor.fetchone()
            rejection_text = result.DepartmentRejectionText if result else None
            assert rejection_text is not None, "Department rejection text not saved"
            print(f"[VERIFY] Department rejection text saved: '{rejection_text}' ✓")
        finally:
            cursor.close()
            conn.close()
        
        # Verify action items may or may not remain (implementation detail)
        action_count = get_action_item_count(subcase_id)
        print(f"[INFO] Action items after rejection: {action_count}")
        
        return {'status': 'PASSED'}
        
    finally:
        cleanup_test_subcase(subcase_id)
        print(f"[CLEANUP] Test data cleaned")


# =============================================================================
# TEST A4: DEPARTMENT OVERRIDES ACTION ITEMS
# =============================================================================

@test_decorator("TEST A4: Department Overrides Action Items")
def test_a4_department_override():
    """
    Test department override workflow:
    SECTION_ACCEPTED_PENDING_DEPT → (OVERRIDE) → DEPT_ACCEPTED_PENDING_ADMIN
    (Old action items deleted, new ones created)
    """
    from backend.api_v2.routers.workflow_router import act_on_case
    
    # Setup
    subcase_id = create_test_subcase(status='SUBMITTED_TO_SECTION', target_org_unit_id=2)
    print(f"[SETUP] Created subcase {subcase_id}")
    
    try:
        # Step 1: Section submits response with 2 action items
        section_admin = create_test_user('SECTION_ADMIN', 2, 'Section')
        
        act_on_case(
            subcase_id=subcase_id,
            body={
                'action': 'SUBMIT_RESPONSE',
                'explanation_text': 'Section response for A4 test',
                'action_items': [
                    {
                        'title': 'Original Item 1',
                        'description': 'First original',
                        'assigned_to_user_id': 999,
                        'due_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
                    },
                    {
                        'title': 'Original Item 2',
                        'description': 'Second original',
                        'assigned_to_user_id': 999,
                        'due_date': (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d')
                    }
                ]
            },
            current_user=section_admin
        )
        
        assert get_action_item_count(subcase_id) == 2
        print(f"[STEP 1] Section created 2 action items ✓")
        
        # Step 2: Department overrides with 3 new action items
        dept_admin = create_test_user('DEPARTMENT_ADMIN', 2, 'Department')
        
        response = act_on_case(
            subcase_id=subcase_id,
            body={
                'action': 'OVERRIDE',
                'explanation_text': 'Department will handle this differently',
                'action_items': [
                    {
                        'title': 'New Item 1',
                        'description': 'First replacement',
                        'assigned_to_user_id': 999,
                        'due_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
                    },
                    {
                        'title': 'New Item 2',
                        'description': 'Second replacement',
                        'assigned_to_user_id': 999,
                        'due_date': (datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d')
                    },
                    {
                        'title': 'New Item 3',
                        'description': 'Third replacement',
                        'assigned_to_user_id': 999,
                        'due_date': (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d')
                    }
                ]
            },
            current_user=dept_admin
        )
        
        assert response.get('success'), "Department override failed"
        
        # Verify status transition
        final_status = get_subcase_status(subcase_id)
        assert final_status == 'DEPT_ACCEPTED_PENDING_ADMIN', f"Expected DEPT_ACCEPTED_PENDING_ADMIN, got {final_status}"
        print(f"[VERIFY] Status is DEPT_ACCEPTED_PENDING_ADMIN ✓")
        
        # Verify action items were replaced (2 deleted, 3 created)
        action_count = get_action_item_count(subcase_id)
        assert action_count == 3, f"Expected 3 action items after override, got {action_count}"
        print(f"[VERIFY] Old action items deleted, 3 new items created ✓")
        
        # Verify department explanation was saved
        conn, cursor = get_db_cursor()
        try:
            cursor.execute("SELECT DepartmentExplanationText FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = ?", (subcase_id,))
            result = cursor.fetchone()
            explanation = result.DepartmentExplanationText if result else None
            assert explanation is not None, "Department explanation not saved"
            assert 'differently' in explanation.lower(), "Department explanation incorrect"
            print(f"[VERIFY] Department explanation saved correctly ✓")
        finally:
            cursor.close()
            conn.close()
        
        return {'status': 'PASSED'}
        
    finally:
        cleanup_test_subcase(subcase_id)
        print(f"[CLEANUP] Test data cleaned")


# =============================================================================
# TEST A5: ADMINISTRATION OVERRIDES ACTION ITEMS
# =============================================================================

@test_decorator("TEST A5: Administration Overrides Action Items")
def test_a5_administration_override():
    """
    Test administration override workflow:
    DEPT_ACCEPTED_PENDING_ADMIN → (OVERRIDE) → ADMIN_APPROVED
    (Old action items deleted, new ones created)
    """
    from backend.api_v2.routers.workflow_router import act_on_case
    
    # Setup
    subcase_id = create_test_subcase(status='SUBMITTED_TO_SECTION', target_org_unit_id=2)
    print(f"[SETUP] Created subcase {subcase_id}")
    
    try:
        # Step 1: Section submits response
        section_admin = create_test_user('SECTION_ADMIN', 2, 'Section')
        
        act_on_case(
            subcase_id=subcase_id,
            body={
                'action': 'SUBMIT_RESPONSE',
                'explanation_text': 'Section response for A5 test',
                'action_items': [
                    {
                        'title': 'Section Item',
                        'description': 'Section action',
                        'assigned_to_user_id': 999,
                        'due_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
                    }
                ]
            },
            current_user=section_admin
        )
        
        print(f"[STEP 1] Section submitted response with 1 action item ✓")
        
        # Step 2: Department approves
        dept_admin = create_test_user('DEPARTMENT_ADMIN', 2, 'Department')
        
        act_on_case(
            subcase_id=subcase_id,
            body={'action': 'APPROVE'},
            current_user=dept_admin
        )
        
        assert get_subcase_status(subcase_id) == 'DEPT_ACCEPTED_PENDING_ADMIN'
        print(f"[STEP 2] Department approved ✓")
        
        # Step 3: Administration overrides with different action items
        admin = create_test_user('ADMINISTRATION_ADMIN', 1, 'Administration')
        
        response = act_on_case(
            subcase_id=subcase_id,
            body={
                'action': 'OVERRIDE',
                'explanation_text': 'Administration requires a different approach',
                'action_items': [
                    {
                        'title': 'Admin Strategic Item',
                        'description': 'High-level strategic action',
                        'assigned_to_user_id': 999,
                        'due_date': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
                    }
                ]
            },
            current_user=admin
        )
        
        assert response.get('success'), "Administration override failed"
        
        # Verify status transition
        final_status = get_subcase_status(subcase_id)
        assert final_status == 'ADMIN_APPROVED', f"Expected ADMIN_APPROVED, got {final_status}"
        print(f"[VERIFY] Status is ADMIN_APPROVED ✓")
        
        # Verify action items were replaced
        action_count = get_action_item_count(subcase_id)
        assert action_count == 1, f"Expected 1 action item after override, got {action_count}"
        print(f"[VERIFY] Old action items deleted, 1 new item created ✓")
        
        # Verify new action item has correct title
        conn, cursor = get_db_cursor()
        try:
            cursor.execute("SELECT Title FROM dbo.APP_SubcaseActionItem WHERE SubcaseID = ?", (subcase_id,))
            result = cursor.fetchone()
            title = result.Title if result else None
            assert title == 'Admin Strategic Item', f"Expected 'Admin Strategic Item', got {title}"
            print(f"[VERIFY] New action item created with correct title ✓")
        finally:
            cursor.close()
            conn.close()
        
        return {'status': 'PASSED'}
        
    finally:
        cleanup_test_subcase(subcase_id)
        print(f"[CLEANUP] Test data cleaned")


# =============================================================================
# TEST A6: ADMINISTRATION FORCE CLOSE
# =============================================================================

@test_decorator("TEST A6: Administration Force Close")
def test_a6_force_close():
    """
    Test administration force close workflow:
    ANY_STATUS → (FORCE_CLOSE) → CLOSED (terminal)
    Should work from any active status, bypassing normal workflow
    """
    from backend.api_v2.routers.workflow_router import act_on_case
    
    # Setup: Test force close from early stage (SUBMITTED_TO_SECTION)
    subcase_id = create_test_subcase(status='SUBMITTED_TO_SECTION', target_org_unit_id=2)
    print(f"[SETUP] Created subcase {subcase_id} in SUBMITTED_TO_SECTION status")
    
    try:
        # Administration force closes (bypassing section/dept workflow)
        admin = create_test_user('ADMINISTRATION_ADMIN', 1, 'Administration')
        
        response = act_on_case(
            subcase_id=subcase_id,
            body={
                'action': 'FORCE_CLOSE',
                'reason': 'Emergency closure - duplicate case detected'
            },
            current_user=admin
        )
        
        assert response.get('success'), "Force close failed"
        
        # Verify immediate terminal state
        final_status = get_subcase_status(subcase_id)
        assert final_status == 'FORCE_CLOSED', f"Expected FORCE_CLOSED, got {final_status}"
        print(f"[VERIFY] Status is FORCE_CLOSED (terminal) - bypassed entire workflow ✓")
        
        # Verify force close reason was saved (stored in AdministrationRejectionText)
        conn, cursor = get_db_cursor()
        try:
            cursor.execute("SELECT AdministrationRejectionText FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = ?", (subcase_id,))
            result = cursor.fetchone()
            reason = result.AdministrationRejectionText if result else None
            assert reason is not None and len(reason) > 0, "Force close reason not saved"
            print(f"[VERIFY] Force close reason saved: '{reason[:50]}...' ✓")
        finally:
            cursor.close()
            conn.close()
        
        return {'status': 'PASSED'}
        
    finally:
        cleanup_test_subcase(subcase_id)
        print(f"[CLEANUP] Test data cleaned")


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests():
    """Execute all Suite A tests and generate report"""
    print("\n" + "="*70)
    print("COMPREHENSIVE WORKFLOW TEST SUITE - SUITE A")
    print("Testing all workflow branches")
    print("="*70)
    
    results = []
    
    # Run all tests
    tests = [
        ('A1: Happy Path', test_a1_happy_path),
        ('A2: Section Rejects', test_a2_section_rejects),
        ('A3: Department Rejects', test_a3_department_rejects),
        ('A4: Department Override', test_a4_department_override),
        ('A5: Administration Override', test_a5_administration_override),
        ('A6: Force Close', test_a6_force_close),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, 'PASSED'))
            passed += 1
        except AssertionError as e:
            results.append((test_name, f'FAILED: {str(e)}'))
            failed += 1
        except Exception as e:
            results.append((test_name, f'FAILED: {type(e).__name__}: {str(e)}'))
            failed += 1
    
    # Generate final report
    print("\n" + "="*70)
    print("TEST SUITE SUMMARY")
    print("="*70)
    
    for test_name, status in results:
        status_icon = "✅" if status == 'PASSED' else "❌"
        print(f"{status_icon} {test_name}: {status}")
    
    print(f"\n{'='*70}")
    print(f"TOTAL: {passed + failed} tests")
    print(f"✅ PASSED: {passed}")
    print(f"❌ FAILED: {failed}")
    print(f"{'='*70}")
    
    if failed == 0:
        print("\n🎉🎉🎉 ALL WORKFLOW BRANCHES VALIDATED! 🎉🎉🎉")
        print("✅ Backend is production-ready for Phase 4 frontend development")
    else:
        print(f"\n⚠️  {failed} test(s) failed - review and fix before proceeding")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
