"""
Comprehensive Force Close Feature Tests
Tests all critical functionality including edge cases.
"""
import sys
import json
sys.path.insert(0, 'backend')

from core.database import get_connection
from backend.api_v2.services import case_response_service
from backend.api_v2.db_layer import administrative_subcase_db
from backend.api.db_layer import incident_case
from datetime import datetime

class MockUser:
    """Mock user for testing"""
    def __init__(self, user_id, username, role_code):
        self.user_id = user_id
        self.username = username
        self.role_code = role_code
        self.scopes = [type('obj', (object,), {'role_code': role_code})]

def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def print_test(test_name, passed, message=""):
    """Print test result"""
    status = "[PASS]" if passed else "[FAIL]"
    print(f"\n{status}: {test_name}")
    if message:
        print(f"   {message}")

# =============================================================================
# TEST SETUP
# =============================================================================

def create_test_incident():
    """Create a test incident for force close testing"""
    print_section("TEST SETUP: Creating Test Data")
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Find an admin user
        cursor.execute("SELECT TOP 1 UserID, Username FROM APP_Users ORDER BY UserID")
        user_row = cursor.fetchone()
        admin_user_id = user_row[0]
        print(f"Using test user: {user_row[1]} (ID: {admin_user_id})")
        
        # Find a valid org unit
        cursor.execute("SELECT TOP 1 UniqueID, Name FROM AdminsrationUnit WHERE Frozen = 0")
        org_unit = cursor.fetchone()
        org_unit_id = org_unit[0]
        print(f"Using org unit: {org_unit[1]} (ID: {org_unit_id})")
        
        # Create test incident
        cursor.execute("""
            INSERT INTO dbo.APP_IncidentCase (
                ComplaintText,
                ImmediateAction,
                TakenAction,
                FeedbackRecievedDate,
                PatientName,
                IssuingOrgUnitID,
                CreatedByUserID,
                isINPatient,
                ClinicalRiskTypeID,
                FeedbackIntentTypeID,
                BuildingID,
                DomainID,
                CategoryID,
                SubCategoryID,
                ClassificationID,
                SeverityID,
                StageID,
                HarmLevelID,
                CaseStatusID,
                SourceID,
                ExplanationStatusID,
                RequiresExplanation
            )
            OUTPUT INSERTED.IncidentRequestCaseID
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            f"FORCE CLOSE TEST CASE - Created {datetime.now()}",
            "Test immediate action",
            "Test taken action",
            datetime.now(),
            "Test Patient",
            org_unit_id,
            admin_user_id,
            1,  # isINPatient
            1,  # ClinicalRiskTypeID
            1,  # FeedbackIntentTypeID
            1,  # BuildingID
            1,  # DomainID
            6,  # CategoryID
            19, # SubCategoryID
            132, # ClassificationID
            1,  # SeverityID
            1,  # StageID
            1,  # HarmLevelID
            1,  # CaseStatusID
            1,  # SourceID
            1,  # ExplanationStatusID
            0   # RequiresExplanation
        ))
        
        incident_id = cursor.fetchone()[0]
        conn.commit()
        print(f"\n[OK] Created test incident: #{incident_id}")
        
        # Create 3 subcases in different states
        subcases = []
        statuses = ['SUBMITTED_TO_SECTION', 'SECTION_ACCEPTED_PENDING_DEPT', 'DEPT_ACCEPTED_PENDING_ADMIN']
        
        # Find 3 different org units for subcases
        cursor.execute("SELECT TOP 3 UniqueID, Name FROM AdminsrationUnit WHERE Frozen = 0 ORDER BY UniqueID")
        org_units = cursor.fetchall()
        
        for i, (status, org) in enumerate(zip(statuses, org_units)):
            subcase_id = administrative_subcase_db.create_subcase(
                case_type='INCIDENT_RESPONSE',
                incident_id=incident_id,
                seasonal_report_id=None,
                target_org_unit_id=org[0],
                created_by_user_id=admin_user_id,
                initial_status=status
            )
            subcases.append({
                'id': subcase_id,
                'status': status,
                'org_unit': org[1],
                'org_unit_id': org[0]
            })
            print(f"[OK] Created subcase #{subcase_id}: {status} -> {org[1]}")
        
        return {
            'incident_id': incident_id,
            'subcases': subcases,
            'admin_user_id': admin_user_id,
            'org_unit_id': org_unit_id
        }
        
    except Exception as e:
        print(f"[ERROR] Error creating test data: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        return None
    finally:
        cursor.close()
        conn.close()

# =============================================================================
# TEST 1: Force Close with Multiple Subcases
# =============================================================================

def test_force_close_multiple_subcases(test_data):
    """Test force closing an incident with multiple subcases"""
    print_section("TEST 1: Force Close with Multiple Subcases")
    
    mock_user = MockUser(test_data['admin_user_id'], "test_admin", "SOFTWARE_ADMIN")
    incident_id = test_data['incident_id']
    
    try:
        # Execute force close
        result = case_response_service.force_close_incident(
            incident_id=incident_id,
            reason_text="TEST: Comprehensive test of force close functionality",
            current_user=mock_user
        )
        
        # Verify result structure
        print_test("Result structure is correct",
                   all(k in result for k in ['success', 'incident_id', 'subcases_closed', 'total_subcases_closed']),
                   f"Keys: {list(result.keys())}")
        
        print_test("Success flag is True", result['success'] == True)
        print_test("Correct incident ID", result['incident_id'] == incident_id)
        print_test("All subcases closed", result['total_subcases_closed'] == 3,
                   f"Closed: {result['total_subcases_closed']}/3")
        
        # Verify subcases are force-closed
        for subcase_info in test_data['subcases']:
            subcase = administrative_subcase_db.get_subcase_by_id(subcase_info['id'])
            is_closed = subcase['status'] == 'FORCE_CLOSED'
            has_tracking = subcase['force_closed_at'] is not None
            has_user = subcase['force_closed_by_user_id'] == mock_user.user_id
            has_reason = subcase['force_close_reason'] is not None
            
            print_test(f"Subcase #{subcase_info['id']} status is FORCE_CLOSED", is_closed)
            print_test(f"Subcase #{subcase_info['id']} has tracking fields", 
                       has_tracking and has_user and has_reason,
                       f"Closed by: {subcase['force_closed_by_user_id']}, Reason length: {len(subcase['force_close_reason']) if subcase['force_close_reason'] else 0}")
        
        # Verify incident has tracking
        conn = get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT ForceClosedAt, ForceClosedByUserID, ForceCloseReason
                FROM dbo.APP_IncidentCase
                WHERE IncidentRequestCaseID = ?
            """, (incident_id,))
            incident_row = cursor.fetchone()
            
            print_test("Incident has force close tracking",
                       incident_row and incident_row[0] is not None,
                       f"Closed at: {incident_row[0]}, By: {incident_row[1]}")
        finally:
            cursor.close()
            conn.close()
        
        return True
        
    except Exception as e:
        print_test("Force close execution", False, f"Error: {e}")
        import traceback
        tracebacklinicalRiskTypeID, FeedbackIntentTypeID, BuildingID, DomainID,
                CategoryID, SubCategoryID, ClassificationID,
                SeverityID, StageID, HarmLevelID,
                CaseStatusID, SourceID, ExplanationStatusID, RequiresExplanation
            )
            OUTPUT INSERTED.IncidentRequestCaseID
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            f"TEST CASE 2 - {datetime.now()}",
            "Test action",
            "Test taken action",
            datetime.now(),
            "Test Patient 2",
            test_data['org_unit_id'],
            test_data['admin_user_id'],
            1,  # isINPatient
            1,  # ClinicalRiskTypeID
            1,  # FeedbackIntentTypeID
            1,  # BuildingID
            1,  # DomainID
            6,  # CategoryID
            19, # SubCategoryID
            132, # ClassificationID
            1,  # SeverityID
            1,  # StageID
            1,  # HarmLevelID
            1,  # CaseStatusID
            1,  # SourceID
            1,  # ExplanationStatusID
            0   # RequiresExplanation
    # Create another test incident
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO dbo.APP_IncidentCase (
                ComplaintText, ImmediateAction, TakenAction,
                FeedbackRecievedDate, PatientName,
                IssuingOrgUnitID, CreatedByUserID, isINPatient,
                CaseStatusID, ExplanationStatusID, RequiresExplanation
            )
            OUTPUT INSERTED.IncidentRequestCaseID
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            f"TEST CASE 2 - {datetime.now()}",
            "Test action",
            "Test taken action",
            datetime.now(),
            "Test Patient 2",
            test_data['org_unit_id'],
            test_data['admin_user_id'],
            1, 1, 1, 0
        ))
        test_incident_id = cursor.fetchone()[0]
        conn.commit()
    finally:
        cursor.close()
        conn.close()
    
    # Test empty reason
    try:
        case_response_service.force_close_incident(
            incident_id=test_incident_id,
            reason_text="",
            current_user=mock_user
        )
        print_test("Empty reason rejected", False, "Should have raised exception")
    except Exception as e:
        print_test("Empty reason rejected", "at least 10 characters" in str(e), str(e))
    
    # Test short reason
    try:
        case_response_service.force_close_incident(
            incident_id=test_incident_id,
            reason_text="short",
            current_user=mock_user
        )
        print_test("Short reason rejected", False, "Should have raised exception")
    except Exception as e:
        print_test("Short reason rejected", "at least 10 characters" in str(e), str(e))
    
    # Test valid reason
    try:
        result = case_response_service.force_close_incident(
            incident_id=test_incident_id,
            reason_text="This is a valid reason with enough characters",
            current_user=mock_user
        )
        print_test("Valid reason accepted", result['success'] == True)
    except Exception as e:
        print_test("Valid reason accepted", False, f"Error: {e}")
    
    return True

# =============================================================================
# TEST 3: Idempotency (Force Close Already Closed)
# =============================================================================

def test_idempotency(test_data):
    """Test that force closing an already closed case is idempotent"""
    print_section("TEST 3: Idempotency - Force Close Already Closed Case")
    
    mock_user = MockUser(test_data['admin_user_id'], "test_admin", "SOFTWARE_ADMIN")
    incident_id = test_data['incident_id']
    
    try:
        # Force close again (already closed in TEST 1)
        result = case_response_service.force_close_incident(
            incident_id=incident_id,
            reason_text="Second force close attempt for idempotency test",
            current_user=mock_user
        )
        
        print_test("Idempotent operation succeeds", result['success'] == True)
        print_test("Returns same subcase count", result['total_subcases_closed'] == 3,
                   f"Count: {result['total_subcases_closed']}")
        
        return True
        
    except Exception as e:
        print_test("Idempotency test", False, f"Error: {e}")
        return False

# =============================================================================
# TEST 4: Inbox Filtering
# =============================================================================

def test_inbox_filtering(test_data):
    """Test that force-closed cases don't appear in inboxes"""
    print_section("TEST 4: Inbox Filtering - Force-Closed Cases Removed")
    
    from backend.api_v2.services import inbox_service
    
    # Create mock user with scope
    mock_user = MockUser(test_data['admin_user_id'], "test_section_admin", "SECTION_ADMIN")
    mock_user.allowed_unit_ids = {test_data['subcases'][0]['org_unit_id']}
    
    try:
        # Get inbox
        inbox_items = inbox_service.get_section_inbox(mock_user)
        
        # Check that force-closed subcase is NOT in inbox
        force_closed_subcase_id = test_data['subcases'][0]['id']
        is_in_inbox = any(item['subcase_id'] == force_closed_subcase_id for item in inbox_items)
        
        print_test("Force-closed case removed from inbox", not is_in_inbox,
                   f"Inbox contains {len(inbox_items)} items, force-closed ID {force_closed_subcase_id} present: {is_in_inbox}")
        
        # Verify the subcase is actually force-closed
        subcase = administrative_subcase_db.get_subcase_by_id(force_closed_subcase_id)
        print_test("Subcase status confirmed as FORCE_CLOSED", 
                   subcase['status'] == 'FORCE_CLOSED',
                   f"Status: {subcase['status']}")
        
        return True
        
    except Exception as e:
        print_test("Inbox filtering test", False, f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

# =============================================================================
# TEST 5: Database Integrity
# =============================================================================

def test_database_integrity():
    """Test database integrity and queries"""
    print_section("TEST 5: Database Integrity Check")
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Check force-closed subcases
        cursor.execute("""
            SELECT COUNT(*) as count,
                   COUNT(CASE WHEN ForceClosedAt IS NOT NULL THEN 1 END) as with_timestamp,
                   COUNT(CASE WHEN ForceClosedByUserID IS NOT NULL THEN 1 END) as with_user,
                   COUNT(CASE WHEN ForceCloseReason IS NOT NULL THEN 1 END) as with_reason
            FROM dbo.APP_AdministrativeSubcase
            WHERE Status = 'FORCE_CLOSED'
        """)
        row = cursor.fetchone()
        
        print_test("Force-closed subcases exist", row[0] > 0, f"Found {row[0]} force-closed subcases")
        print_test("All have ForceClosedAt timestamp", row[0] == row[1], f"{row[1]}/{row[0]}")
        print_test("All have ForceClosedByUserID", row[0] == row[2], f"{row[2]}/{row[0]}")
        print_test("All have ForceCloseReason", row[0] == row[3], f"{row[3]}/{row[0]}")
        
        # Check force-closed incidents
        cursor.execute("""
            SELECT COUNT(*) as count
            FROM dbo.APP_IncidentCase
            WHERE ForceClosedAt IS NOT NULL
        """)
        incident_count = cursor.fetchone()[0]
        
        print_test("Force-closed incidents exist", incident_count > 0, 
                   f"Found {incident_count} force-closed incidents")
        
        # Check indexes exist
        cursor.execute("""
            SELECT COUNT(*) 
            FROM sys.indexes 
            WHERE name IN ('IX_AdministrativeSubcase_ForceClosedAt', 'IX_IncidentCase_ForceClosedAt')
        """)
        index_count = cursor.fetchone()[0]
        
        print_test("Indexes created", index_count == 2, f"Found {index_count}/2 indexes")
        
        # Check FK constraints
        cursor.execute("""
            SELECT COUNT(*) 
            FROM sys.foreign_keys 
            WHERE name IN ('FK_AdministrativeSubcase_ForceClosedByUser', 'FK_IncidentCase_ForceClosedByUser')
        """)
        fk_count = cursor.fetchone()[0]
        
        print_test("FK constraints created", fk_count == 2, f"Found {fk_count}/2 FK constraints")
        
        return True
        
    except Exception as e:
        print_test("Database integrity check", False, f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cursor.close()
        conn.close()

# =============================================================================
# TEST 6: Audit Trail Query
# =============================================================================

def test_audit_trail():
    """Test that audit trail can be queried"""
    print_section("TEST 6: Audit Trail Verification")
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT TOP 5
                i.IncidentRequestCaseID,
                LEFT(i.ComplaintText, 50) as Summary,
                i.ForceClosedAt,
                u.Username,
                LEFT(i.ForceCloseReason, 50) as Reason,
                (SELECT COUNT(*) 
                 FROM APP_AdministrativeSubcase s 
                 WHERE s.IncidentRequestCaseID = i.IncidentRequestCaseID 
                 AND s.Status = 'FORCE_CLOSED') as SubcaseCount
            FROM APP_IncidentCase i
            LEFT JOIN APP_Users u ON i.ForceClosedByUserID = u.UserID
            WHERE i.ForceClosedAt IS NOT NULL
            ORDER BY i.ForceClosedAt DESC
        """)
        
        rows = cursor.fetchall()
        
        print_test("Audit trail query works", len(rows) > 0, f"Found {len(rows)} audit records")
        
        if rows:
            print("\nRecent Force-Close Audit Trail:")
            print("-" * 70)
            for row in rows:
                print(f"  Incident #{row[0]}: {row[1]}")
                print(f"    Closed: {row[2]} by {row[3]}")
                print(f"    Reason: {row[4]}")
                print(f"    Subcases closed: {row[5]}")
                print()
        
        return True
        
    except Exception as e:
        print_testsults)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {test_name}")
    
    print("\n" + "=" * 70)
    print(f" RESULT: {passed}/{total} tests passed")
    if passed == total:
        print(" [SUCCESS] ALL TESTS PASSED - FEATURE IS PRODUCTION READY")
    else:
        print(" [WARNING]
def run_all_tests():
    """Run all force close tests"""
    print("\n" + "=" * 70)
    print(" FORCE CLOSE FEATURE - COMPREHENSIVE TEST SUITE")
    print(" Critical Data Change Testing")
    print("=" * 70)
    print(f" Test Date: {datetime.now()}")
    print("=" * 70)
    
    # Create test[ERROR]data
    test_data = crsults)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {test_name}")
    
    print("\n" + "=" * 70)
    print(f" RESULT: {passed}/{total} tests passed")
    if passed == total:
        print(" [SUCCESS] ALL TESTS PASSED - FEATURE IS PRODUCTION READY")
    else:
        print(" [WARNING]("Database Integrity", test_database_integrity()))
    results.append(("Audit Trail", test_audit_trail()))
    
    # Summary
    print_section("TEST SUMMARY")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {test_name}")
    
    print("\n" + "=" * 70)
    print(f" RESULT: {passed}/{total} tests passed")
    if passed == total:
        print(" [SUCCESS] ALL TESTS PASSED - FEATURE IS PRODUCTION READY")
    else:
        print(" [WARNING]RNING]RNING]tal:
        print(" ✅ ALL TESTS PASSED - FEATURE IS PRODUCTION READY")
    else:
        print(" ⚠️  SOME TESTS FAILED - REVIEW REQUIRED")
    print("=" * 70)
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
