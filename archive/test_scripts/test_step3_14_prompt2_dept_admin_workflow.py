"""
STEP 3.14 TEST — Prompt 2: Department & Administration Workflow Actions

Tests department and administration workflow functions in case_response_service.py:
Department:
- approve_department()
- reject_department()
- override_department()

Administration:
- approve_administration()
- reject_administration()
- override_administration()

Pure state machine testing - NO permissions, NO scopes, NO router logic.
"""

import sys
import os
from datetime import datetime

# Force UTF-8 encoding for emoji support
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
        def wrapper():
            print(f"\n{'='*80}")
            print(f"TEST: {description}")
            print('='*80)
            try:
                func()
                print("\n✅ TEST PASSED")
            except AssertionError as e:
                print(f"\n❌ TEST FAILED: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
            except Exception as e:
                print(f"\n❌ TEST ERROR: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
        return wrapper
    return decorator


# Mock user class
class MockUser:
    def __init__(self, user_id):
        self.user_id = user_id


def create_test_incident():
    """Create a minimal real incident."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO dbo.APP_IncidentCase (
                ComplaintText, FeedbackRecievedDate, IssuingOrgUnitID,
                SourceID, ExplanationStatusID, ImmediateAction, TakenAction,
                CreatedByUserID, PatientName, isINPatient, ClinicalRiskTypeID,
                FeedbackIntentTypeID, BuildingID, DomainID, CategoryID,
                SubCategoryID, ClassificationID, SeverityID, StageID,
                HarmLevelID, CaseStatusID, RequiresExplanation
            )
            OUTPUT INSERTED.IncidentRequestCaseID
            VALUES (
                'Test incident for STEP 3.14 Prompt 2', GETDATE(), 1,
                1, 1, 'Immediate action', 'Taken action',
                1, 'Test Patient', 1, 1,
                1, 1, 1, 1,
                1, 78, 1, 1,
                1, 1, 0
            )
        """)
        
        row = cursor.fetchone()
        incident_id = row[0]
        conn.commit()
        return incident_id
        
    finally:
        cursor.close()
        conn.close()


def create_test_subcase(incident_id, status='SUBMITTED_TO_SECTION'):
    """Create a test subcase."""
    from api_v2.db_layer import administrative_subcase_db
    
    subcase_id = administrative_subcase_db.create_subcase(
        case_type='INCIDENT_RESPONSE',
        incident_id=incident_id,
        seasonal_report_id=None,
        target_org_unit_id=1,
        created_by_user_id=1,
        initial_status=status
    )
    return subcase_id


def cleanup_test_incident(incident_id):
    """Clean up test incident and all related data."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            DELETE FROM dbo.APP_SubcaseActionItem 
            WHERE SubcaseID IN (
                SELECT SubcaseID FROM dbo.APP_AdministrativeSubcase 
                WHERE IncidentRequestCaseID = ?
            )
        """, (incident_id,))
        
        cursor.execute("""
            DELETE FROM dbo.APP_AdministrativeSubcase 
            WHERE IncidentRequestCaseID = ?
        """, (incident_id,))
        
        cursor.execute("""
            DELETE FROM dbo.APP_IncidentCase 
            WHERE IncidentRequestCaseID = ?
        """, (incident_id,))
        
        conn.commit()
        
    finally:
        cursor.close()
        conn.close()


# =============================================================================
# DEPARTMENT TESTS
# =============================================================================

@test("1. Department Approve - Happy Path")
def test_department_approve_happy_path():
    """Test successful department approval."""
    from api_v2.services import case_response_service
    from api_v2.db_layer import administrative_subcase_db
    
    incident_id = create_test_incident()
    subcase_id = create_test_subcase(incident_id, 'SECTION_ACCEPTED_PENDING_DEPT')
    user = MockUser(user_id=2)
    
    try:
        print("\n[EXECUTE] Approving at department level...")
        case_response_service.approve_department(
            subcase_id=subcase_id,
            current_user=user
        )
        
        print("[VERIFY] Checking status transition...")
        subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
        
        assert subcase['status'] == 'DEPT_ACCEPTED_PENDING_ADMIN', \
            f"Expected 'DEPT_ACCEPTED_PENDING_ADMIN', got '{subcase['status']}'"
        
        print(f"  ✅ Status: {subcase['status']}")
        
    finally:
        cleanup_test_incident(incident_id)


@test("2. Department Reject - Happy Path")
def test_department_reject_happy_path():
    """Test successful department rejection."""
    from api_v2.services import case_response_service
    from api_v2.db_layer import administrative_subcase_db
    
    incident_id = create_test_incident()
    subcase_id = create_test_subcase(incident_id, 'SECTION_ACCEPTED_PENDING_DEPT')
    user = MockUser(user_id=2)
    
    try:
        print("\n[EXECUTE] Rejecting at department level...")
        case_response_service.reject_department(
            subcase_id=subcase_id,
            rejection_text='Department disagrees with section assessment. Insufficient evidence.',
            current_user=user
        )
        
        print("[VERIFY] Checking status and rejection text...")
        subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
        
        assert subcase['status'] == 'DEPT_REJECTED', \
            f"Expected 'DEPT_REJECTED', got '{subcase['status']}'"
        assert subcase['department_rejection_text'] == 'Department disagrees with section assessment. Insufficient evidence.', \
            "Department rejection text not saved"
        
        print(f"  ✅ Status: {subcase['status']} (terminal)")
        print(f"  ✅ Rejection text saved")
        
    finally:
        cleanup_test_incident(incident_id)


@test("3. Department Override - Action Item Replacement")
def test_department_override_action_items():
    """
    Test department override replaces ALL existing action items.
    CRITICAL: This tests the _replace_action_items() helper function.
    """
    from api_v2.services import case_response_service
    from api_v2.db_layer import administrative_subcase_db, action_item_subcase_db
    
    incident_id = create_test_incident()
    subcase_id = create_test_subcase(incident_id, 'SUBMITTED_TO_SECTION')
    user = MockUser(user_id=1)
    
    try:
        # Step 1: Section creates 3 action items
        print("\n[STEP 1] Section creates 3 action items...")
        section_items = [
            {'title': 'Section Item 1', 'description': 'Desc 1', 'due_date': None},
            {'title': 'Section Item 2', 'description': 'Desc 2', 'due_date': '2026-02-10'},
            {'title': 'Section Item 3', 'description': 'Desc 3', 'due_date': None},
        ]
        
        case_response_service.submit_section_response(
            subcase_id=subcase_id,
            explanation_text='Section explanation',
            action_items=section_items,
            current_user=user
        )
        
        items_after_section = action_item_subcase_db.get_action_items_by_subcase(subcase_id)
        print(f"  ✅ Section created {len(items_after_section)} items")
        assert len(items_after_section) == 3, "Expected 3 items after section"
        
        # Step 2: Department overrides with 2 different items
        print("\n[STEP 2] Department overrides with 2 new items...")
        dept_user = MockUser(user_id=2)
        dept_items = [
            {'title': 'Dept Override 1', 'description': 'New desc 1', 'due_date': '2026-03-01'},
            {'title': 'Dept Override 2', 'description': 'New desc 2', 'due_date': None},
        ]
        
        case_response_service.override_department(
            subcase_id=subcase_id,
            explanation_text='Department overrides section plan with revised approach.',
            action_items=dept_items,
            current_user=dept_user
        )
        
        # Step 3: Verify replacement (CRITICAL INVARIANT)
        print("\n[VERIFY] Checking action item replacement...")
        items_after_dept = action_item_subcase_db.get_action_items_by_subcase(subcase_id)
        
        assert len(items_after_dept) == 2, \
            f"Expected exactly 2 items after override, got {len(items_after_dept)}"
        
        # Verify old items are GONE
        titles_after = {item['title'] for item in items_after_dept}
        assert 'Section Item 1' not in titles_after, "Old section item 1 still exists!"
        assert 'Section Item 2' not in titles_after, "Old section item 2 still exists!"
        assert 'Section Item 3' not in titles_after, "Old section item 3 still exists!"
        
        # Verify new items exist with correct details
        assert 'Dept Override 1' in titles_after, "New dept item 1 missing"
        assert 'Dept Override 2' in titles_after, "New dept item 2 missing"
        
        # Deep validation of new items
        for item in items_after_dept:
            assert item['status'] == 'DRAFT', f"Item {item['title']} not in DRAFT status"
            if item['title'] == 'Dept Override 1':
                assert item['description'] == 'New desc 1', "Description mismatch"
                assert item['due_date'] is not None, "Due date should be set"
            elif item['title'] == 'Dept Override 2':
                assert item['description'] == 'New desc 2', "Description mismatch"
                assert item['due_date'] is None, "Due date should be None"
        
        print(f"  ✅ Old items deleted: 3")
        print(f"  ✅ New items created: 2")
        print(f"  ✅ All new items in DRAFT status")
        
        # Verify department explanation saved
        subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
        assert subcase['department_explanation_text'] == 'Department overrides section plan with revised approach.', \
            "Department explanation not saved"
        assert subcase['status'] == 'DEPT_ACCEPTED_PENDING_ADMIN', \
            f"Expected 'DEPT_ACCEPTED_PENDING_ADMIN', got '{subcase['status']}'"
        
        print(f"  ✅ Department explanation saved")
        print(f"  ✅ Status: {subcase['status']}")
        
    finally:
        cleanup_test_incident(incident_id)


@test("4. Department - Wrong Status Errors")
def test_department_wrong_status():
    """Test department functions fail on wrong status."""
    from api_v2.services import case_response_service
    
    incident_id = create_test_incident()
    subcase_id = create_test_subcase(incident_id, 'SUBMITTED_TO_SECTION')  # Wrong status
    user = MockUser(user_id=2)
    
    try:
        print("\n[TEST] Department approve on SUBMITTED_TO_SECTION...")
        try:
            case_response_service.approve_department(subcase_id=subcase_id, current_user=user)
            raise AssertionError("Should have raised Exception")
        except Exception as e:
            assert "SECTION_ACCEPTED_PENDING_DEPT" in str(e), \
                f"Error should mention required status, got: {str(e)}"
            print(f"  ✅ Correct error: {str(e)[:80]}...")
        
        print("\n[TEST] Department reject on SUBMITTED_TO_SECTION...")
        try:
            case_response_service.reject_department(subcase_id=subcase_id, rejection_text='test', current_user=user)
            raise AssertionError("Should have raised Exception")
        except Exception as e:
            assert "SECTION_ACCEPTED_PENDING_DEPT" in str(e)
            print(f"  ✅ Correct error")
        
        print("\n[TEST] Department override on SUBMITTED_TO_SECTION...")
        try:
            case_response_service.override_department(
                subcase_id=subcase_id, explanation_text='test', action_items=[], current_user=user
            )
            raise AssertionError("Should have raised Exception")
        except Exception as e:
            assert "SECTION_ACCEPTED_PENDING_DEPT" in str(e)
            print(f"  ✅ Correct error")
        
    finally:
        cleanup_test_incident(incident_id)


# =============================================================================
# ADMINISTRATION TESTS
# =============================================================================

@test("5. Administration Approve - Happy Path")
def test_administration_approve_happy_path():
    """Test successful administration approval."""
    from api_v2.services import case_response_service
    from api_v2.db_layer import administrative_subcase_db
    
    incident_id = create_test_incident()
    subcase_id = create_test_subcase(incident_id, 'DEPT_ACCEPTED_PENDING_ADMIN')
    user = MockUser(user_id=3)
    
    try:
        print("\n[EXECUTE] Approving at administration level...")
        case_response_service.approve_administration(
            subcase_id=subcase_id,
            current_user=user
        )
        
        print("[VERIFY] Checking final status...")
        subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
        
        assert subcase['status'] == 'ADMIN_APPROVED', \
            f"Expected 'ADMIN_APPROVED', got '{subcase['status']}'"
        
        print(f"  ✅ Status: {subcase['status']} (final approval)")
        
    finally:
        cleanup_test_incident(incident_id)


@test("6. Administration Reject - Happy Path")
def test_administration_reject_happy_path():
    """Test successful administration rejection."""
    from api_v2.services import case_response_service
    from api_v2.db_layer import administrative_subcase_db
    
    incident_id = create_test_incident()
    subcase_id = create_test_subcase(incident_id, 'DEPT_ACCEPTED_PENDING_ADMIN')
    user = MockUser(user_id=3)
    
    try:
        print("\n[EXECUTE] Rejecting at administration level...")
        case_response_service.reject_administration(
            subcase_id=subcase_id,
            rejection_text='Administration requires additional evidence before approval.',
            current_user=user
        )
        
        print("[VERIFY] Checking status and rejection text...")
        subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
        
        assert subcase['status'] == 'ADMIN_REJECTED', \
            f"Expected 'ADMIN_REJECTED', got '{subcase['status']}'"
        assert subcase['administration_rejection_text'] == 'Administration requires additional evidence before approval.', \
            "Administration rejection text not saved"
        
        print(f"  ✅ Status: {subcase['status']} (terminal)")
        print(f"  ✅ Rejection text saved")
        
    finally:
        cleanup_test_incident(incident_id)


@test("7. Administration Override - Multi-Stage Replacement")
def test_administration_override_after_department():
    """
    Test administration override after department already overrode section.
    Tests multi-stage action item replacement chain.
    """
    from api_v2.services import case_response_service
    from api_v2.db_layer import administrative_subcase_db, action_item_subcase_db
    
    incident_id = create_test_incident()
    subcase_id = create_test_subcase(incident_id, 'SUBMITTED_TO_SECTION')
    
    try:
        # Stage 1: Section creates 3 items
        print("\n[STAGE 1] Section creates 3 items...")
        user1 = MockUser(user_id=1)
        case_response_service.submit_section_response(
            subcase_id=subcase_id,
            explanation_text='Section plan',
            action_items=[
                {'title': 'S1', 'description': 'Section 1', 'due_date': None},
                {'title': 'S2', 'description': 'Section 2', 'due_date': None},
                {'title': 'S3', 'description': 'Section 3', 'due_date': None},
            ],
            current_user=user1
        )
        
        items_stage1 = action_item_subcase_db.get_action_items_by_subcase(subcase_id)
        print(f"  ✅ Stage 1: {len(items_stage1)} items")
        
        # Stage 2: Department overrides with 2 items
        print("\n[STAGE 2] Department overrides with 2 items...")
        user2 = MockUser(user_id=2)
        case_response_service.override_department(
            subcase_id=subcase_id,
            explanation_text='Dept revision',
            action_items=[
                {'title': 'D1', 'description': 'Dept 1', 'due_date': '2026-03-15'},
                {'title': 'D2', 'description': 'Dept 2', 'due_date': None},
            ],
            current_user=user2
        )
        
        items_stage2 = action_item_subcase_db.get_action_items_by_subcase(subcase_id)
        print(f"  ✅ Stage 2: {len(items_stage2)} items")
        assert len(items_stage2) == 2, "Department override failed"
        
        # Stage 3: Administration overrides with 4 items
        print("\n[STAGE 3] Administration overrides with 4 items...")
        user3 = MockUser(user_id=3)
        case_response_service.override_administration(
            subcase_id=subcase_id,
            explanation_text='Final admin plan with comprehensive actions.',
            action_items=[
                {'title': 'A1', 'description': 'Admin 1', 'due_date': '2026-04-01'},
                {'title': 'A2', 'description': 'Admin 2', 'due_date': '2026-04-15'},
                {'title': 'A3', 'description': 'Admin 3', 'due_date': None},
                {'title': 'A4', 'description': 'Admin 4', 'due_date': '2026-05-01'},
            ],
            current_user=user3
        )
        
        # Final verification
        print("\n[VERIFY] Checking final action items...")
        items_final = action_item_subcase_db.get_action_items_by_subcase(subcase_id)
        
        assert len(items_final) == 4, \
            f"Expected exactly 4 items after admin override, got {len(items_final)}"
        
        # Verify ALL previous items gone
        titles_final = {item['title'] for item in items_final}
        assert 'S1' not in titles_final and 'S2' not in titles_final and 'S3' not in titles_final, \
            "Section items should be deleted"
        assert 'D1' not in titles_final and 'D2' not in titles_final, \
            "Department items should be deleted"
        
        # Verify admin items present
        assert titles_final == {'A1', 'A2', 'A3', 'A4'}, \
            f"Expected admin items only, got {titles_final}"
        
        print(f"  ✅ Final count: {len(items_final)} items")
        print(f"  ✅ All admin items present")
        print(f"  ✅ All previous items deleted")
        
        # Verify status and explanation
        subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
        assert subcase['status'] == 'ADMIN_APPROVED', "Wrong final status"
        assert subcase['administration_explanation_text'] == 'Final admin plan with comprehensive actions.', \
            "Admin explanation not saved"
        
        print(f"  ✅ Status: {subcase['status']}")
        print(f"  ✅ Admin explanation saved")
        
    finally:
        cleanup_test_incident(incident_id)


@test("8. Administration - Wrong Status Errors")
def test_administration_wrong_status():
    """Test administration functions fail on wrong status."""
    from api_v2.services import case_response_service
    
    incident_id = create_test_incident()
    subcase_id = create_test_subcase(incident_id, 'SECTION_ACCEPTED_PENDING_DEPT')  # Wrong
    user = MockUser(user_id=3)
    
    try:
        print("\n[TEST] Admin approve on wrong status...")
        try:
            case_response_service.approve_administration(subcase_id=subcase_id, current_user=user)
            raise AssertionError("Should have raised Exception")
        except Exception as e:
            assert "DEPT_ACCEPTED_PENDING_ADMIN" in str(e), \
                f"Error should mention required status, got: {str(e)}"
            print(f"  ✅ Correct error")
        
        print("\n[TEST] Admin reject on wrong status...")
        try:
            case_response_service.reject_administration(subcase_id=subcase_id, rejection_text='test', current_user=user)
            raise AssertionError("Should have raised Exception")
        except Exception as e:
            assert "DEPT_ACCEPTED_PENDING_ADMIN" in str(e)
            print(f"  ✅ Correct error")
        
        print("\n[TEST] Admin override on wrong status...")
        try:
            case_response_service.override_administration(
                subcase_id=subcase_id, explanation_text='test', action_items=[], current_user=user
            )
            raise AssertionError("Should have raised Exception")
        except Exception as e:
            assert "DEPT_ACCEPTED_PENDING_ADMIN" in str(e)
            print(f"  ✅ Correct error")
        
    finally:
        cleanup_test_incident(incident_id)


# =============================================================================
# MAIN TEST EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("STEP 3.14 — PROMPT 2: DEPARTMENT & ADMINISTRATION WORKFLOW TESTS")
    print("Testing dept/admin actions in case_response_service.py")
    print("="*80)
    
    test_count = 0
    passed = 0
    failed = 0
    
    tests = [
        test_department_approve_happy_path,
        test_department_reject_happy_path,
        test_department_override_action_items,
        test_department_wrong_status,
        test_administration_approve_happy_path,
        test_administration_reject_happy_path,
        test_administration_override_after_department,
        test_administration_wrong_status,
    ]
    
    for test_func in tests:
        test_count += 1
        try:
            test_func()
            passed += 1
        except:
            failed += 1
    
    print("\n" + "="*80)
    print("TEST SUITE COMPLETE")
    print("="*80)
    print(f"\nTotal Tests: {test_count}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Prompt 2 implementation is complete.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the errors above.")
        sys.exit(1)





