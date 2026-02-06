"""
STEP 3.14 TEST — Prompt 1: Section-Level Workflow Actions

Tests section-level workflow functions in case_response_service.py:
- submit_section_response()
- reject_responsibility()

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


# Mock user class for testing
class MockUser:
    def __init__(self, user_id):
        self.user_id = user_id


def create_test_incident():
    """Create a minimal real incident for full integration testing."""
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
                'Test incident for STEP 3.14', GETDATE(), 1,
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


def create_test_subcase(incident_id, status='SUBMITTED_TO_SECTION', target_org_unit_id=1):
    """Create a test subcase linked to a real incident."""
    from api_v2.db_layer import administrative_subcase_db
    
    subcase_id = administrative_subcase_db.create_subcase(
        case_type='INCIDENT_RESPONSE',
        incident_id=incident_id,
        seasonal_report_id=None,
        target_org_unit_id=target_org_unit_id,
        created_by_user_id=1,
        initial_status=status
    )
    return subcase_id


def cleanup_test_incident(incident_id):
    """Clean up test incident and all related data."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Delete subcases first (FK constraint)
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
            DELETE FROM dbo.APP_IncidentCaseTargetDepartment 
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
# HAPPY PATH TESTS
# =============================================================================

@test("1. Section Accept - Happy Path")
def test_section_accept_happy_path():
    """
    Test successful section response submission.
    Verifies:
    - Status transition: SUBMITTED_TO_SECTION -> SECTION_ACCEPTED_PENDING_DEPT
    - Explanation text saved
    - Action items created with correct properties
    """
    from api_v2.services import case_response_service
    from api_v2.db_layer import administrative_subcase_db, action_item_subcase_db
    
    # Setup
    incident_id = create_test_incident()
    subcase_id = create_test_subcase(incident_id, 'SUBMITTED_TO_SECTION')
    user = MockUser(user_id=1)
    
    action_items = [
        {
            'title': 'Review incident protocol',
            'description': 'Review and update the incident response protocol',
            'due_date': '2026-02-15'
        },
        {
            'title': 'Staff training',
            'description': 'Conduct staff training on new procedures',
            'due_date': None
        }
    ]
    
    try:
        print("\n[EXECUTE] Submitting section response...")
        case_response_service.submit_section_response(
            subcase_id=subcase_id,
            explanation_text='Section has reviewed and accepts responsibility. Action plan created.',
            action_items=action_items,
            current_user=user
        )
        
        print("[VERIFY] Checking subcase status and explanation...")
        subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
        
        assert subcase is not None, "Subcase not found after update"
        assert subcase['status'] == 'SECTION_ACCEPTED_PENDING_DEPT', \
            f"Expected status 'SECTION_ACCEPTED_PENDING_DEPT', got '{subcase['status']}'"
        assert subcase['section_explanation_text'] == 'Section has reviewed and accepts responsibility. Action plan created.', \
            "Section explanation text not saved correctly"
        
        print(f"  ✅ Status: {subcase['status']}")
        print(f"   Explanation saved: {len(subcase['section_explanation_text'])} chars")
        
        print("\n[VERIFY] Checking action items...")
        items = action_item_subcase_db.get_action_items_by_subcase(subcase_id)
        
        assert len(items) == 2, f"Expected 2 action items, got {len(items)}"
        
        # Verify all items are in DRAFT status
        for item in items:
            assert item['status'] == 'DRAFT', \
                f"Expected status 'DRAFT', got '{item['status']}' for item {item['ActionItemID']}"
        
        # Verify item details (deep validation)
        titles = {item['title'] for item in items}
        assert 'Review incident protocol' in titles, "First action item title missing"
        assert 'Staff training' in titles, "Second action item title missing"
        
        # Verify specific item properties
        for item in items:
            if item['title'] == 'Review incident protocol':
                assert item['description'] == 'Review and update the incident response protocol', \
                    "Description mismatch for first item"
                assert item['due_date'] is not None, "Due date should be set for first item"
            elif item['title'] == 'Staff training':
                assert item['description'] == 'Conduct staff training on new procedures', \
                    "Description mismatch for second item"
                assert item['due_date'] is None, "Due date should be None for second item"
        
        print(f"  ✅ Created {len(items)} action items (all DRAFT)")
        print(f"  ✅ Titles: {sorted(titles)}")
        print(f"  ✅ Descriptions and due dates validated")
        
    finally:
        cleanup_test_incident(incident_id)


@test("2. Section Reject - Happy Path")
def test_section_reject_happy_path():
    """
    Test successful section responsibility rejection.
    Verifies:
    - Status transition: SUBMITTED_TO_SECTION -> SECTION_DENIED (terminal)
    - Rejection text saved
    """
    from api_v2.services import case_response_service
    from api_v2.db_layer import administrative_subcase_db
    
    # Setup
    incident_id = create_test_incident()
    subcase_id = create_test_subcase(incident_id, 'SUBMITTED_TO_SECTION')
    user = MockUser(user_id=1)
    
    try:
        print("\n[EXECUTE] Rejecting section responsibility...")
        case_response_service.reject_responsibility(
            subcase_id=subcase_id,
            rejection_text='This incident does not fall under section jurisdiction. Recommend reassignment.',
            current_user=user
        )
        
        print("[VERIFY] Checking subcase status and rejection...")
        subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
        
        assert subcase is not None, "Subcase not found after update"
        assert subcase['status'] == 'SECTION_DENIED', \
            f"Expected status 'SECTION_DENIED', got '{subcase['status']}'"
        assert subcase['section_rejection_text'] == 'This incident does not fall under section jurisdiction. Recommend reassignment.', \
            "Section rejection text not saved correctly"
        
        print(f"  ✅ Status: {subcase['status']} (terminal)")
        print(f"  ✅ Rejection saved: {len(subcase['section_rejection_text'])} chars")
        
    finally:
        cleanup_test_incident(incident_id)


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

@test("3. Section Accept - Wrong Status Error")
def test_section_accept_wrong_status():
    """
    Test that section accept fails when subcase is not in SUBMITTED_TO_SECTION status.
    """
    from api_v2.services import case_response_service
    
    # Setup - create subcase in WRONG status
    incident_id = create_test_incident()
    subcase_id = create_test_subcase(incident_id, 'SECTION_ACCEPTED_PENDING_DEPT')
    user = MockUser(user_id=1)
    
    try:
        print("\n[EXECUTE] Attempting section accept on wrong status...")
        
        try:
            case_response_service.submit_section_response(
                subcase_id=subcase_id,
                explanation_text='This should fail',
                action_items=[],
                current_user=user
            )
            raise AssertionError("Expected Exception but none was raised")
            
        except Exception as e:
            error_msg = str(e)
            print(f"[VERIFY] Exception raised: {error_msg}")
            
            # Strict validation of error message
            assert "SUBMITTED_TO_SECTION" in error_msg, \
                f"Error message should mention required status 'SUBMITTED_TO_SECTION', got: {error_msg}"
            assert "SECTION_ACCEPTED_PENDING_DEPT" in error_msg, \
                f"Error message should mention current status 'SECTION_ACCEPTED_PENDING_DEPT', got: {error_msg}"
            
            print(f"  ✅ Correct exception raised with clear message")
        
    finally:
        cleanup_test_incident(incident_id)


@test("4. Section Reject - Wrong Status Error")
def test_section_reject_wrong_status():
    """
    Test that section reject fails when subcase is not in SUBMITTED_TO_SECTION status.
    """
    from api_v2.services import case_response_service
    
    # Setup - create subcase in WRONG status
    incident_id = create_test_incident()
    subcase_id = create_test_subcase(incident_id, 'DEPT_REJECTED')
    user = MockUser(user_id=1)
    
    try:
        print("\n[EXECUTE] Attempting section reject on wrong status...")
        
        try:
            case_response_service.reject_responsibility(
                subcase_id=subcase_id,
                rejection_text='This should fail',
                current_user=user
            )
            raise AssertionError("Expected Exception but none was raised")
            
        except Exception as e:
            error_msg = str(e)
            print(f"[VERIFY] Exception raised: {error_msg}")
            
            # Strict validation
            assert "SUBMITTED_TO_SECTION" in error_msg, \
                f"Error message should mention required status, got: {error_msg}"
            
            print(f"  ✅ Correct exception raised")
        
    finally:
        cleanup_test_incident(incident_id)


@test("5. Section Accept - None User Error")
def test_section_accept_none_user():
    """
    Test that section accept fails when current_user is None.
    """
    from api_v2.services import case_response_service
    
    # Setup
    incident_id = create_test_incident()
    subcase_id = create_test_subcase(incident_id, 'SUBMITTED_TO_SECTION')
    
    try:
        print("\n[EXECUTE] Attempting section accept with None user...")
        
        try:
            case_response_service.submit_section_response(
                subcase_id=subcase_id,
                explanation_text='This should fail',
                action_items=[],
                current_user=None  # None user
            )
            raise AssertionError("Expected Exception but none was raised")
            
        except Exception as e:
            error_msg = str(e)
            print(f"[VERIFY] Exception raised: {error_msg}")
            
            assert "current_user" in error_msg.lower() and "none" in error_msg.lower(), \
                f"Error message should mention current_user cannot be None, got: {error_msg}"
            
            print(f"  ✅ Correct exception raised")
        
    finally:
        cleanup_test_incident(incident_id)


@test("6. Section Reject - None User Error")
def test_section_reject_none_user():
    """
    Test that section reject fails when current_user is None.
    """
    from api_v2.services import case_response_service
    
    # Setup
    incident_id = create_test_incident()
    subcase_id = create_test_subcase(incident_id, 'SUBMITTED_TO_SECTION')
    
    try:
        print("\n[EXECUTE] Attempting section reject with None user...")
        
        try:
            case_response_service.reject_responsibility(
                subcase_id=subcase_id,
                rejection_text='This should fail',
                current_user=None
            )
            raise AssertionError("Expected Exception but none was raised")
            
        except Exception as e:
            error_msg = str(e)
            print(f"[VERIFY] Exception raised: {error_msg}")
            
            assert "current_user" in error_msg.lower() and "none" in error_msg.lower(), \
                f"Error message should mention current_user, got: {error_msg}"
            
            print(f"  ✅ Correct exception raised")
        
    finally:
        cleanup_test_incident(incident_id)


@test("7. Section Accept - Invalid Subcase ID")
def test_section_accept_invalid_subcase():
    """
    Test that section accept fails when subcase doesn't exist.
    """
    from api_v2.services import case_response_service
    
    user = MockUser(user_id=1)
    invalid_subcase_id = 999999
    
    print(f"\n[EXECUTE] Attempting section accept on non-existent subcase {invalid_subcase_id}...")
    
    try:
        case_response_service.submit_section_response(
            subcase_id=invalid_subcase_id,
            explanation_text='This should fail',
            action_items=[],
            current_user=user
        )
        raise AssertionError("Expected Exception but none was raised")
        
    except Exception as e:
        error_msg = str(e)
        print(f"[VERIFY] Exception raised: {error_msg}")
        
        assert "not found" in error_msg.lower(), \
            f"Error message should mention subcase not found, got: {error_msg}"
        
        print(f"  ✅ Correct exception raised")


@test("8. Section Reject - Invalid Subcase ID")
def test_section_reject_invalid_subcase():
    """
    Test that section reject fails when subcase doesn't exist.
    """
    from api_v2.services import case_response_service
    
    user = MockUser(user_id=1)
    invalid_subcase_id = 999999
    
    print(f"\n[EXECUTE] Attempting section reject on non-existent subcase {invalid_subcase_id}...")
    
    try:
        case_response_service.reject_responsibility(
            subcase_id=invalid_subcase_id,
            rejection_text='This should fail',
            current_user=user
        )
        raise AssertionError("Expected Exception but none was raised")
        
    except Exception as e:
        error_msg = str(e)
        print(f"[VERIFY] Exception raised: {error_msg}")
        
        assert "not found" in error_msg.lower(), \
            f"Error message should mention subcase not found, got: {error_msg}"
        
        print(f"  ✅ Correct exception raised")


# =============================================================================
# MAIN TEST EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("STEP 3.14 — PROMPT 1: SECTION-LEVEL WORKFLOW TESTS")
    print("Testing section-level actions in case_response_service.py")
    print("="*80)
    
    test_count = 0
    passed = 0
    failed = 0
    
    tests = [
        test_section_accept_happy_path,
        test_section_reject_happy_path,
        test_section_accept_wrong_status,
        test_section_reject_wrong_status,
        test_section_accept_none_user,
        test_section_reject_none_user,
        test_section_accept_invalid_subcase,
        test_section_reject_invalid_subcase,
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
        print("\n🎉 ALL TESTS PASSED! Prompt 1 implementation is complete.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the errors above.")
        sys.exit(1)




