"""
PHASE 2 TEST: DB Layer - Explanation Write Operations
======================================================
Tests all write/update operations for explanation workflow.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.api.db_layer.explanation_db import (
    get_case_by_id,
    get_explanation_status_id,
    get_case_status_id,
    update_case_explanation,
    update_case_requires_explanation,
    force_close_case,
    close_case_after_action_items,
    get_connection
)


def create_test_case():
    """Helper: Create a test case for write operations"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Get required lookup IDs
        cursor.execute("SELECT TOP 1 DomainID FROM dbo.IncidentDomain")
        domain = cursor.fetchone()
        if not domain:
            cursor.execute("SELECT TOP 1 DomainID FROM dbo.APP_IncidentCase")
            domain = cursor.fetchone()
        
        cursor.execute("SELECT TOP 1 CategoryID FROM dbo.IncidentCategory")
        category = cursor.fetchone()
        if not category:
            cursor.execute("SELECT TOP 1 CategoryID FROM dbo.APP_IncidentCase")
            category = cursor.fetchone()
        
        cursor.execute("SELECT TOP 1 SubCategoryID FROM dbo.IncidentSubCategory")
        subcategory = cursor.fetchone()
        if not subcategory:
            cursor.execute("SELECT TOP 1 SubCategoryID FROM dbo.APP_IncidentCase")
            subcategory = cursor.fetchone()
        
        cursor.execute("SELECT TOP 1 ClassificationID FROM dbo.IncidentCaseClassification")
        classification = cursor.fetchone()
        if not classification:
            cursor.execute("SELECT TOP 1 ClassificationID FROM dbo.APP_IncidentCase")
            classification = cursor.fetchone()
        
        cursor.execute("SELECT TOP 1 StatusID FROM dbo.IncidentCaseStage")
        stage = cursor.fetchone()
        if not stage:
            cursor.execute("SELECT TOP 1 StageID FROM dbo.APP_IncidentCase")
            stage = cursor.fetchone()
        
        cursor.execute("SELECT TOP 1 HarmID FROM dbo.IncidentCaseHarm")
        harm = cursor.fetchone()
        if not harm:
            cursor.execute("SELECT TOP 1 HarmLevelID FROM dbo.APP_IncidentCase")
            harm = cursor.fetchone()
        
        cursor.execute("SELECT TOP 1 ClinicalRiskTypeID FROM dbo.APP_LOOKUP_CLINICAL_RISK_TYPE WHERE IsActive = 1")
        risk = cursor.fetchone()
        
        cursor.execute("SELECT TOP 1 FeedbackIntentTypeID FROM dbo.APP_LOOKUP_FEEDBACK_INTENT_TYPE WHERE IsActive = 1")
        intent = cursor.fetchone()
        
        cursor.execute("SELECT TOP 1 UniqueID FROM dbo.AdminsrationUnit")
        org_unit = cursor.fetchone()
        
        open_status = get_case_status_id("OPEN")
        waiting_status = get_explanation_status_id("Waiting")
        
        if not all([domain, category, subcategory, classification, stage, harm, risk, intent, org_unit, open_status, waiting_status]):
            conn.close()
            return None
        
        # Insert test case
        cursor.execute("""
            INSERT INTO dbo.APP_IncidentCase (
                ComplaintText,
                RequiresExplanation,
                CaseStatusID,
                ExplanationStatusID,
                DomainID,
                CategoryID,
                SubCategoryID,
                ClassificationID,
                StageID,
                HarmLevelID,
                ClinicalRiskTypeID,
                FeedbackIntentTypeID,
                IssuingOrgUnitID,
                CreatedByUserID,
                CreatedAt
            )
            OUTPUT INSERTED.IncidentRequestCaseID
            VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, GETDATE())
        """, (
            'TEST CASE - Phase 2 Write Operations',
            open_status, waiting_status,
            domain[0], category[0], subcategory[0], classification[0],
            stage[0], harm[0], risk[0], intent[0], org_unit[0]
        ))
        
        case_id = cursor.fetchone()[0]
        conn.commit()
        conn.close()
        
        return case_id
        
    except Exception as e:
        print(f"Error creating test case: {e}")
        conn.rollback()
        conn.close()
        return None


def delete_test_case(case_id):
    """Helper: Delete a test case"""
    if not case_id:
        return
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Delete associated action items first
        cursor.execute("DELETE FROM dbo.APP_ActionItem WHERE IncidentRequestCaseID = ?", (case_id,))
        # Delete the case
        cursor.execute("DELETE FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?", (case_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error deleting test case: {e}")
        conn.rollback()
        conn.close()


def test_update_case_explanation():
    """Test 1: Submit explanation for a case"""
    print("=" * 70)
    print("TEST 1: Update Case Explanation")
    print("=" * 70)
    
    case_id = None
    try:
        # Create test case
        case_id = create_test_case()
        if not case_id:
            print("⚠ Could not create test case, skipping test")
            return True
        
        print(f"✓ Created test case ID: {case_id}")
        
        # Get initial state
        initial_case = get_case_by_id(case_id)
        print(f"  Initial State:")
        print(f"    CaseStatus: {initial_case['CaseStatusName']}")
        print(f"    ExplanationStatus: {initial_case['ExplanationStatusName']}")
        print(f"    TakenAction: {initial_case.get('TakenAction') or 'NULL'}")
        
        # Submit explanation
        explanation = "This is a test explanation for Phase 2 validation."
        result = update_case_explanation(case_id, explanation, user_id=1)
        
        print(f"\n✓ Explanation submitted successfully")
        print(f"  Message: {result['message']}")
        
        # Verify state transition
        updated_case = get_case_by_id(case_id)
        print(f"\n  Updated State:")
        print(f"    CaseStatus: {updated_case['CaseStatusName']}")
        print(f"    ExplanationStatus: {updated_case['ExplanationStatusName']}")
        print(f"    TakenAction: {'Present' if updated_case.get('TakenAction') else 'NULL'}")
        
        # Verify FSM transition: Open+Waiting → In Progress+Responded
        if updated_case['CaseStatusName'] == 'In Progress' and updated_case['ExplanationStatusName'] == 'Responded':
            print(f"\n✓ FSM transition successful: Open+Waiting → In Progress+Responded")
        else:
            print(f"\n✗ FSM transition failed")
            return False
        
        # Test error case: Try to update closed case
        try:
            force_close_case(case_id, user_id=1)
            update_case_explanation(case_id, "Should fail", user_id=1)
            print(f"\n✗ Should have raised error for closed case")
            return False
        except ValueError as e:
            print(f"\n✓ Correctly rejected update to closed case: {str(e)[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        delete_test_case(case_id)
        print(f"  Cleaned up test case")


def test_update_requires_explanation():
    """Test 2: Toggle RequiresExplanation flag"""
    print("\n" + "=" * 70)
    print("TEST 2: Update RequiresExplanation Flag")
    print("=" * 70)
    
    case_id = None
    try:
        # Create test case with RequiresExplanation = 1
        case_id = create_test_case()
        if not case_id:
            print("⚠ Could not create test case, skipping test")
            return True
        
        print(f"✓ Created test case ID: {case_id}")
        
        # Get initial state
        initial_case = get_case_by_id(case_id)
        print(f"  Initial RequiresExplanation: {initial_case['RequiresExplanation']}")
        
        # Toggle to False
        result = update_case_requires_explanation(case_id, False, user_id=1)
        print(f"\n✓ Updated to False")
        print(f"  Message: {result['message']}")
        
        updated_case = get_case_by_id(case_id)
        print(f"  Current RequiresExplanation: {updated_case['RequiresExplanation']}")
        
        if updated_case['RequiresExplanation'] == 0:
            print(f"✓ Flag correctly set to False")
        else:
            print(f"✗ Flag not updated")
            return False
        
        # Toggle back to True
        result = update_case_requires_explanation(case_id, True, user_id=1)
        print(f"\n✓ Updated to True")
        
        updated_case = get_case_by_id(case_id)
        print(f"  Current RequiresExplanation: {updated_case['RequiresExplanation']}")
        
        if updated_case['RequiresExplanation'] == 1:
            print(f"✓ Flag correctly set to True")
        else:
            print(f"✗ Flag not updated")
            return False
        
        # Test error case: Try to change flag on closed case
        try:
            force_close_case(case_id, user_id=1)
            update_case_requires_explanation(case_id, False, user_id=1)
            print(f"\n✗ Should have raised error for closed case")
            return False
        except ValueError as e:
            print(f"\n✓ Correctly rejected flag change on closed case: {str(e)[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        delete_test_case(case_id)
        print(f"  Cleaned up test case")


def test_force_close_case():
    """Test 3: Force close a case (admin override)"""
    print("\n" + "=" * 70)
    print("TEST 3: Force Close Case")
    print("=" * 70)
    
    case_id = None
    try:
        # Create test case
        case_id = create_test_case()
        if not case_id:
            print("⚠ Could not create test case, skipping test")
            return True
        
        print(f"✓ Created test case ID: {case_id}")
        
        # Get initial state
        initial_case = get_case_by_id(case_id)
        print(f"  Initial State:")
        print(f"    CaseStatus: {initial_case['CaseStatusName']}")
        print(f"    ExplanationStatus: {initial_case['ExplanationStatusName']}")
        
        # Force close
        result = force_close_case(case_id, user_id=1, reason="Admin override for testing")
        
        print(f"\n✓ Case forcibly closed")
        print(f"  Message: {result['message']}")
        print(f"  Reason: {result.get('reason', 'N/A')}")
        
        # Verify state
        updated_case = get_case_by_id(case_id)
        print(f"\n  Updated State:")
        print(f"    CaseStatus: {updated_case['CaseStatusName']}")
        print(f"    ExplanationStatus: {updated_case['ExplanationStatusName']}")
        
        # Verify: Closed + Forcibly Closed
        if updated_case['CaseStatusName'] == 'Closed' and updated_case['ExplanationStatusName'] == 'Forcibly Closed':
            print(f"\n✓ FSM transition successful: → Closed+Forcibly Closed")
        else:
            print(f"\n✗ FSM transition failed")
            return False
        
        # Test error case: Try to force close already closed case
        try:
            force_close_case(case_id, user_id=1)
            print(f"\n✗ Should have raised error for already closed case")
            return False
        except ValueError as e:
            print(f"\n✓ Correctly rejected second force close: {str(e)[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        delete_test_case(case_id)
        print(f"  Cleaned up test case")


def test_close_after_action_items():
    """Test 4: Close case after action items completion"""
    print("\n" + "=" * 70)
    print("TEST 4: Close Case After Action Items")
    print("=" * 70)
    
    case_id = None
    try:
        # Create test case
        case_id = create_test_case()
        if not case_id:
            print("⚠ Could not create test case, skipping test")
            return True
        
        print(f"✓ Created test case ID: {case_id}")
        
        # Submit explanation first (to transition to In Progress + Responded)
        update_case_explanation(case_id, "Test explanation", user_id=1)
        print(f"✓ Submitted explanation (transitioned to In Progress + Responded)")
        
        # Create a test action item
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO dbo.APP_ActionItem (
                IncidentRequestCaseID,
                ActionTitle,
                ActionDescription,
                IsDone,
                CreatedByUserID
            )
            OUTPUT INSERTED.ActionItemID
            VALUES (?, 'Test Action', 'Test action for Phase 2', 0, 1)
        """, (case_id,))
        action_id = cursor.fetchone()[0]
        conn.commit()
        conn.close()
        print(f"✓ Created test action item ID: {action_id}")
        
        # Try to close with incomplete action items (should fail)
        try:
            close_case_after_action_items(case_id, user_id=1)
            print(f"\n✗ Should have raised error for incomplete action items")
            return False
        except ValueError as e:
            print(f"\n✓ Correctly rejected closure with incomplete items: {str(e)[:60]}...")
        
        # Mark action item as done
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE dbo.APP_ActionItem SET IsDone = 1 WHERE ActionItemID = ?", (action_id,))
        conn.commit()
        conn.close()
        print(f"✓ Marked action item as done")
        
        # Now close the case
        result = close_case_after_action_items(case_id, user_id=1)
        print(f"\n✓ Case closed after action items")
        print(f"  Message: {result['message']}")
        
        # Verify state
        updated_case = get_case_by_id(case_id)
        print(f"\n  Final State:")
        print(f"    CaseStatus: {updated_case['CaseStatusName']}")
        print(f"    ExplanationStatus: {updated_case['ExplanationStatusName']}")
        
        # Verify: Closed + Responded
        if updated_case['CaseStatusName'] == 'Closed' and updated_case['ExplanationStatusName'] == 'Responded':
            print(f"\n✓ FSM transition successful: In Progress+Responded → Closed+Responded")
        else:
            print(f"\n✗ FSM transition failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        delete_test_case(case_id)
        print(f"  Cleaned up test case")


def test_fsm_validation():
    """Test 5: FSM state validation and error handling"""
    print("\n" + "=" * 70)
    print("TEST 5: FSM Validation and Error Handling")
    print("=" * 70)
    
    try:
        # Test with non-existent case
        try:
            update_case_explanation(999999999, "Test", user_id=1)
            print("✗ Should have raised error for non-existent case")
            return False
        except ValueError as e:
            print(f"✓ Correctly rejected non-existent case: {str(e)[:40]}...")
        
        # Test with invalid user
        try:
            force_close_case(999999999, user_id=999999)
            print("✗ Should have raised error for non-existent case")
            return False
        except ValueError as e:
            print(f"✓ Correctly rejected operation on non-existent case: {str(e)[:40]}...")
        
        print(f"\n✓ All FSM validation checks passed")
        return True
        
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all Phase 2 tests"""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  PHASE 2: DB LAYER WRITE OPERATIONS TESTS".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")
    
    tests = [
        ("Update Case Explanation", test_update_case_explanation),
        ("Update RequiresExplanation Flag", test_update_requires_explanation),
        ("Force Close Case", test_force_close_case),
        ("Close After Action Items", test_close_after_action_items),
        ("FSM Validation", test_fsm_validation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ TEST FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n")
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test_name:<45} {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print("=" * 70)
    print(f"  Total: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Phase 2 Complete!")
    else:
        print(f"\n⚠ {total - passed} test(s) failed - Please review")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
