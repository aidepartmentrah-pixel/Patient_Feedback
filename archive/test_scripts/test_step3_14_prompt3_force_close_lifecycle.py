"""
STEP 3.14 TEST — Prompt 3: Force Close & Full Workflow Lifecycle

Tests force close and complete workflow lifecycle:
- force_close_subcase()
- Full approval chain (Section → Dept → Admin)
- Early rejection paths
- Terminal state validation

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
                'Test incident for STEP 3.14 Prompt 3', GETDATE(), 1,
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
# FORCE CLOSE TESTS
# =============================================================================

@test("1. Force Close from SUBMITTED_TO_SECTION")
def test_force_close_from_initial():
    """Test force close from initial workflow state."""
    from api_v2.services import case_response_service
    from api_v2.db_layer import administrative_subcase_db
    
    incident_id = create_test_incident()
    subcase_id = create_test_subcase(incident_id, 'SUBMITTED_TO_SECTION')
    user = MockUser(user_id=99)  # Admin user
    
    try:
        print("\n[EXECUTE] Force closing from SUBMITTED_TO_SECTION...")
        case_response_service.force_close_subcase(
            subcase_id=subcase_id,
            reason_text='Case closed due to duplicate entry in system.',
            current_user=user
        )
        
        print("[VERIFY] Checking force close result...")
        subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
        
        assert subcase['status'] == 'FORCE_CLOSED', \
            f"Expected 'FORCE_CLOSED', got '{subcase['status']}'"
        assert subcase['administration_rejection_text'] == 'Case closed due to duplicate entry in system.', \
            "Force close reason not saved correctly"
        
        print(f"  ✅ Status: {subcase['status']} (terminal)")
        print(f"  ✅ Reason saved in administration_rejection_text")
        
    finally:
        cleanup_test_incident(incident_id)


@test("2. Force Close from DEPT_ACCEPTED_PENDING_ADMIN")
def test_force_close_from_mid_workflow():
    """Test force close from middle of workflow."""
    from api_v2.services import case_response_service
    from api_v2.db_layer import administrative_subcase_db
    
    incident_id = create_test_incident()
    subcase_id = create_test_subcase(incident_id, 'DEPT_ACCEPTED_PENDING_ADMIN')
    user = MockUser(user_id=99)
    
    try:
        print("\n[EXECUTE] Force closing from DEPT_ACCEPTED_PENDING_ADMIN...")
        case_response_service.force_close_subcase(
            subcase_id=subcase_id,
            reason_text='Incident no longer relevant due to policy change.',
            current_user=user
        )
        
        print("[VERIFY] Checking force close result...")
        subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
        
        assert subcase['status'] == 'FORCE_CLOSED', \
            f"Expected 'FORCE_CLOSED', got '{subcase['status']}'"
        assert 'policy change' in subcase['administration_rejection_text'], \
            "Reason text not saved"
        
        print(f"  ✅ Status: {subcase['status']}")
        
    finally:
        cleanup_test_incident(incident_id)


@test("3. Force Close - Already CLOSED Error")
def test_force_close_already_closed():
    """Test that force close fails if already in CLOSED status."""
    from api_v2.services import case_response_service
    from api_v2.db_layer import administrative_subcase_db
    
    incident_id = create_test_incident()
    subcase_id = create_test_subcase(incident_id, 'CLOSED')
    user = MockUser(user_id=99)
    
    try:
        print("\n[EXECUTE] Attempting force close on CLOSED case...")
        
        try:
            case_response_service.force_close_subcase(
                subcase_id=subcase_id,
                reason_text='This should fail',
                current_user=user
            )
            raise AssertionError("Should have raised Exception")
            
        except Exception as e:
            error_msg = str(e)
            print(f"[VERIFY] Exception raised: {error_msg}")
            
            assert "CLOSED" in error_msg, \
                f"Error should mention CLOSED status, got: {error_msg}"
            assert "terminal" in error_msg.lower(), \
                f"Error should mention terminal state, got: {error_msg}"
            
            print(f"  ✅ Correct exception raised")
        
    finally:
        cleanup_test_incident(incident_id)


@test("4. Force Close - Already FORCE_CLOSED Error")
def test_force_close_already_force_closed():
    """Test that force close fails if already force closed."""
    from api_v2.services import case_response_service
    
    incident_id = create_test_incident()
    subcase_id = create_test_subcase(incident_id, 'FORCE_CLOSED')
    user = MockUser(user_id=99)
    
    try:
        print("\n[EXECUTE] Attempting force close on already FORCE_CLOSED case...")
        
        try:
            case_response_service.force_close_subcase(
                subcase_id=subcase_id,
                reason_text='This should fail',
                current_user=user
            )
            raise AssertionError("Should have raised Exception")
            
        except Exception as e:
            error_msg = str(e)
            assert "FORCE_CLOSED" in error_msg, \
                f"Error should mention status, got: {error_msg}"
            print(f"  ✅ Correct exception raised")
        
    finally:
        cleanup_test_incident(incident_id)


@test("5. Force Close - Works from Terminal Rejection States")
def test_force_close_from_rejection_states():
    """
    Test that force close WORKS from rejection terminal states.
    These are terminal but not 'closed' states, so force close should work.
    """
    from api_v2.services import case_response_service
    from api_v2.db_layer import administrative_subcase_db
    
    # Test from each rejection state
    rejection_states = ['SECTION_DENIED', 'DEPT_REJECTED', 'ADMIN_REJECTED']
    
    for state in rejection_states:
        incident_id = create_test_incident()
        subcase_id = create_test_subcase(incident_id, state)
        user = MockUser(user_id=99)
        
        try:
            print(f"\n[TEST] Force close from {state}...")
            
            # This should SUCCEED - rejection states are terminal but not "closed"
            case_response_service.force_close_subcase(
                subcase_id=subcase_id,
                reason_text=f'Force closing from {state}',
                current_user=user
            )
            
            subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
            assert subcase['status'] == 'FORCE_CLOSED', \
                f"Force close should work from {state}"
            
            print(f"  ✅ Force close worked from {state}")
            
        finally:
            cleanup_test_incident(incident_id)


# =============================================================================
# FULL LIFECYCLE TESTS
# =============================================================================

@test("6. Happy Path - Full Approval Chain")
def test_full_approval_chain():
    """
    Test complete approval workflow:
    SUBMITTED_TO_SECTION → SECTION_ACCEPTED_PENDING_DEPT → 
    DEPT_ACCEPTED_PENDING_ADMIN → ADMIN_APPROVED
    """
    from api_v2.services import case_response_service
    from api_v2.db_layer import administrative_subcase_db
    
    incident_id = create_test_incident()
    subcase_id = create_test_subcase(incident_id, 'SUBMITTED_TO_SECTION')
    
    try:
        print("\n[STAGE 1] Section accepts...")
        user1 = MockUser(user_id=1)
        case_response_service.submit_section_response(
            subcase_id=subcase_id,
            explanation_text='Section accepts',
            action_items=[{'title': 'Item 1', 'description': 'Desc', 'due_date': None}],
            current_user=user1
        )
        
        subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
        assert subcase['status'] == 'SECTION_ACCEPTED_PENDING_DEPT'
        print(f"  ✅ Stage 1: {subcase['status']}")
        
        print("\n[STAGE 2] Department approves...")
        user2 = MockUser(user_id=2)
        case_response_service.approve_department(
            subcase_id=subcase_id,
            current_user=user2
        )
        
        subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
        assert subcase['status'] == 'DEPT_ACCEPTED_PENDING_ADMIN'
        print(f"  ✅ Stage 2: {subcase['status']}")
        
        print("\n[STAGE 3] Administration approves...")
        user3 = MockUser(user_id=3)
        case_response_service.approve_administration(
            subcase_id=subcase_id,
            current_user=user3
        )
        
        subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
        assert subcase['status'] == 'ADMIN_APPROVED'
        print(f"  ✅ Stage 3: {subcase['status']} (FINAL)")
        
        print("\n[VERIFY] Complete workflow succeeded!")
        print(f"  ✅ Workflow: SUBMITTED → SECTION_ACCEPTED → DEPT_ACCEPTED → ADMIN_APPROVED")
        
    finally:
        cleanup_test_incident(incident_id)


@test("7. Early Rejection - Section Level")
def test_early_rejection_section():
    """Test workflow termination at section level."""
    from api_v2.services import case_response_service
    from api_v2.db_layer import administrative_subcase_db
    
    incident_id = create_test_incident()
    subcase_id = create_test_subcase(incident_id, 'SUBMITTED_TO_SECTION')
    user = MockUser(user_id=1)
    
    try:
        print("\n[EXECUTE] Section rejects immediately...")
        case_response_service.reject_responsibility(
            subcase_id=subcase_id,
            rejection_text='Not our responsibility',
            current_user=user
        )
        
        subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
        assert subcase['status'] == 'SECTION_DENIED'
        print(f"  ✅ Workflow terminated at section: {subcase['status']}")
        
    finally:
        cleanup_test_incident(incident_id)


@test("8. Early Rejection - Department Level")
def test_early_rejection_department():
    """Test workflow termination at department level."""
    from api_v2.services import case_response_service
    from api_v2.db_layer import administrative_subcase_db
    
    incident_id = create_test_incident()
    subcase_id = create_test_subcase(incident_id, 'SUBMITTED_TO_SECTION')
    
    try:
        # Section accepts
        print("\n[STAGE 1] Section accepts...")
        user1 = MockUser(user_id=1)
        case_response_service.submit_section_response(
            subcase_id=subcase_id,
            explanation_text='Section accepts',
            action_items=[],
            current_user=user1
        )
        
        # Department rejects
        print("\n[STAGE 2] Department rejects...")
        user2 = MockUser(user_id=2)
        case_response_service.reject_department(
            subcase_id=subcase_id,
            rejection_text='Department disagrees',
            current_user=user2
        )
        
        subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
        assert subcase['status'] == 'DEPT_REJECTED'
        print(f"  ✅ Workflow terminated at department: {subcase['status']}")
        
    finally:
        cleanup_test_incident(incident_id)


@test("9. Early Rejection - Administration Level")
def test_early_rejection_administration():
    """Test workflow termination at administration level."""
    from api_v2.services import case_response_service
    from api_v2.db_layer import administrative_subcase_db
    
    incident_id = create_test_incident()
    subcase_id = create_test_subcase(incident_id, 'SUBMITTED_TO_SECTION')
    
    try:
        # Section accepts
        print("\n[STAGE 1] Section accepts...")
        user1 = MockUser(user_id=1)
        case_response_service.submit_section_response(
            subcase_id=subcase_id,
            explanation_text='Section accepts',
            action_items=[],
            current_user=user1
        )
        
        # Department approves
        print("\n[STAGE 2] Department approves...")
        user2 = MockUser(user_id=2)
        case_response_service.approve_department(
            subcase_id=subcase_id,
            current_user=user2
        )
        
        # Administration rejects
        print("\n[STAGE 3] Administration rejects...")
        user3 = MockUser(user_id=3)
        case_response_service.reject_administration(
            subcase_id=subcase_id,
            rejection_text='Admin requires more evidence',
            current_user=user3
        )
        
        subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
        assert subcase['status'] == 'ADMIN_REJECTED'
        print(f"  ✅ Workflow terminated at administration: {subcase['status']}")
        
    finally:
        cleanup_test_incident(incident_id)


@test("10. Override Chain - Complete Replacement Lifecycle")
def test_complete_override_chain():
    """
    Test action item lifecycle through complete override chain.
    Section creates → Dept overrides → Admin overrides.
    Verifies final state has only admin items.
    """
    from api_v2.services import case_response_service
    from api_v2.db_layer import administrative_subcase_db, action_item_subcase_db
    
    incident_id = create_test_incident()
    subcase_id = create_test_subcase(incident_id, 'SUBMITTED_TO_SECTION')
    
    try:
        # Stage 1: Section
        print("\n[STAGE 1] Section creates 5 items...")
        user1 = MockUser(user_id=1)
        case_response_service.submit_section_response(
            subcase_id=subcase_id,
            explanation_text='Section plan',
            action_items=[
                {'title': f'S{i}', 'description': f'Section {i}', 'due_date': None}
                for i in range(1, 6)
            ],
            current_user=user1
        )
        count1 = len(action_item_subcase_db.get_action_items_by_subcase(subcase_id))
        assert count1 == 5
        print(f"  ✅ {count1} items after section")
        
        # Stage 2: Department
        print("\n[STAGE 2] Department overrides with 3 items...")
        user2 = MockUser(user_id=2)
        case_response_service.override_department(
            subcase_id=subcase_id,
            explanation_text='Dept plan',
            action_items=[
                {'title': f'D{i}', 'description': f'Dept {i}', 'due_date': None}
                for i in range(1, 4)
            ],
            current_user=user2
        )
        count2 = len(action_item_subcase_db.get_action_items_by_subcase(subcase_id))
        assert count2 == 3
        print(f"  ✅ {count2} items after department (5 deleted, 3 created)")
        
        # Stage 3: Administration
        print("\n[STAGE 3] Administration overrides with 1 item...")
        user3 = MockUser(user_id=3)
        case_response_service.override_administration(
            subcase_id=subcase_id,
            explanation_text='Final admin plan',
            action_items=[
                {'title': 'FINAL', 'description': 'Single final action', 'due_date': '2026-06-01'}
            ],
            current_user=user3
        )
        
        final_items = action_item_subcase_db.get_action_items_by_subcase(subcase_id)
        assert len(final_items) == 1
        assert final_items[0]['title'] == 'FINAL'
        print(f"  ✅ {len(final_items)} item after admin (3 deleted, 1 created)")
        
        # Verify final state
        subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
        assert subcase['status'] == 'ADMIN_APPROVED'
        
        print("\n[VERIFY] Complete override chain succeeded!")
        print(f"  ✅ Item count progression: 5 → 3 → 1")
        print(f"  ✅ Final status: {subcase['status']}")
        
    finally:
        cleanup_test_incident(incident_id)


# =============================================================================
# MAIN TEST EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("STEP 3.14 — PROMPT 3: FORCE CLOSE & LIFECYCLE TESTS")
    print("Testing force close and complete workflow lifecycle")
    print("="*80)
    
    test_count = 0
    passed = 0
    failed = 0
    
    tests = [
        test_force_close_from_initial,
        test_force_close_from_mid_workflow,
        test_force_close_already_closed,
        test_force_close_already_force_closed,
        test_force_close_from_rejection_states,
        test_full_approval_chain,
        test_early_rejection_section,
        test_early_rejection_department,
        test_early_rejection_administration,
        test_complete_override_chain,
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
        print("\n🎉 ALL TESTS PASSED! Prompt 3 & STEP 3.14 are COMPLETE!")
        print("\n✨ case_response_service.py is fully validated and ready for production!")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the errors above.")
        sys.exit(1)


