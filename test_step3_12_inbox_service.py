"""
STEP 3.12 Inbox Service Test Suite
Tests the inbox_service.py implementation across 3 layers:
- Layer 1: Static checks (imports, signatures)
- Layer 2: Unit tests (filter logic, action computation)
- Layer 3: Integration tests (real database)
"""

import sys
import os

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
            print(f"\n{'='*70}")
            print(f"TEST: {description}")
            print('='*70)
            try:
                func()
                print(f"✅ PASSED")
            except AssertionError as e:
                print(f"❌ FAILED: {str(e)}")
                raise
            except Exception as e:
                print(f"❌ ERROR: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
        return wrapper
    return decorator


class MockUser:
    """Mock user object for testing"""
    def __init__(self, role, section_id=None, department_id=None, user_id=1):
        self.role = role
        self.section_id = section_id
        self.department_id = department_id
        self.user_id = user_id


class MockSubcase:
    """Mock subcase object for testing"""
    def __init__(self, subcase_id, case_type, status, target_org_unit_id, 
                 incident_id=None, seasonal_report_id=None, created_at=None):
        self.SubcaseID = subcase_id
        self.CaseType = case_type
        self.Status = status
        self.TargetOrgUnitID = target_org_unit_id
        self.IncidentRequestCaseID = incident_id
        self.SeasonalReportID = seasonal_report_id
        self.CreatedAt = created_at


# =============================================================================
# LAYER 1: STATIC CHECKS
# =============================================================================

@test("Layer 1.1 - Import inbox_service module")
def test_import_inbox_service():
    """Verify that inbox_service module can be imported"""
    from api_v2.services import inbox_service
    print("  ✓ Module imported successfully")


@test("Layer 1.2 - Verify public functions exist")
def test_public_functions_exist():
    """Verify that all required public functions exist"""
    from api_v2.services import inbox_service
    
    assert hasattr(inbox_service, 'get_section_inbox'), "get_section_inbox not found"
    print("  ✓ get_section_inbox found")
    
    assert hasattr(inbox_service, 'get_department_inbox'), "get_department_inbox not found"
    print("  ✓ get_department_inbox found")
    
    assert hasattr(inbox_service, 'get_administration_inbox'), "get_administration_inbox not found"
    print("  ✓ get_administration_inbox found")


@test("Layer 1.3 - Verify internal helper functions exist")
def test_helper_functions_exist():
    """Verify that internal helper functions exist"""
    from api_v2.services import inbox_service
    
    assert hasattr(inbox_service, '_apply_scope_filter'), "_apply_scope_filter not found"
    print("  ✓ _apply_scope_filter found")
    
    assert hasattr(inbox_service, '_compute_allowed_actions'), "_compute_allowed_actions not found"
    print("  ✓ _compute_allowed_actions found")
    
    assert hasattr(inbox_service, '_build_inbox_item'), "_build_inbox_item not found"
    print("  ✓ _build_inbox_item found")


@test("Layer 1.4 - Check function signatures")
def test_function_signatures():
    """Verify function signatures are correct"""
    from api_v2.services import inbox_service
    import inspect
    
    sig1 = inspect.signature(inbox_service.get_section_inbox)
    print(f"  get_section_inbox: {sig1}")
    assert 'current_user' in sig1.parameters, "Missing current_user parameter"
    
    sig2 = inspect.signature(inbox_service.get_department_inbox)
    print(f"  get_department_inbox: {sig2}")
    assert 'current_user' in sig2.parameters, "Missing current_user parameter"
    
    sig3 = inspect.signature(inbox_service.get_administration_inbox)
    print(f"  get_administration_inbox: {sig3}")
    assert 'current_user' in sig3.parameters, "Missing current_user parameter"


# =============================================================================
# LAYER 2: UNIT TESTS (Mock Data)
# =============================================================================

@test("Layer 2.1 - Test role validation (Section)")
def test_section_role_validation():
    """Test that get_section_inbox rejects non-section users"""
    from api_v2.services.inbox_service import get_section_inbox
    
    wrong_user = MockUser(role='Department Administrator', section_id=1)
    
    try:
        get_section_inbox(wrong_user)
        raise AssertionError("Should have raised ValueError for wrong role")
    except ValueError as e:
        print(f"  ✓ Correctly rejected: {str(e)}")


@test("Layer 2.2 - Test role validation (Department)")
def test_department_role_validation():
    """Test that get_department_inbox rejects non-department users"""
    from api_v2.services.inbox_service import get_department_inbox
    
    wrong_user = MockUser(role='Section Administrator', department_id=1)
    
    try:
        get_department_inbox(wrong_user)
        raise AssertionError("Should have raised ValueError for wrong role")
    except ValueError as e:
        print(f"  ✓ Correctly rejected: {str(e)}")


@test("Layer 2.3 - Test role validation (Administration)")
def test_administration_role_validation():
    """Test that get_administration_inbox rejects non-admin users"""
    from api_v2.services.inbox_service import get_administration_inbox
    
    wrong_user = MockUser(role='Section Administrator')
    
    try:
        get_administration_inbox(wrong_user)
        raise AssertionError("Should have raised ValueError for wrong role")
    except ValueError as e:
        print(f"  ✓ Correctly rejected: {str(e)}")


@test("Layer 2.4 - Test scope filter (Section)")
def test_scope_filter_section():
    """Test that _apply_scope_filter correctly filters by section_id"""
    from api_v2.services.inbox_service import _apply_scope_filter
    
    user = MockUser(role='Section Administrator', section_id=1)
    
    subcases = [
        MockSubcase(1, 'INCIDENT_RESPONSE', 'SUBMITTED_TO_SECTION', target_org_unit_id=1),
        MockSubcase(2, 'INCIDENT_RESPONSE', 'SUBMITTED_TO_SECTION', target_org_unit_id=2),
        MockSubcase(3, 'INCIDENT_RESPONSE', 'SUBMITTED_TO_SECTION', target_org_unit_id=1),
    ]
    
    filtered = _apply_scope_filter(subcases, user)
    
    print(f"  Input: {len(subcases)} subcases")
    print(f"  Filtered: {len(filtered)} subcases")
    
    assert len(filtered) == 2, f"Expected 2 subcases for section_id=1, got {len(filtered)}"
    assert all(sc.TargetOrgUnitID == 1 for sc in filtered), "Filtered subcases should only have TargetOrgUnitID=1"
    print(f"  ✓ Correctly filtered to section_id=1")


@test("Layer 2.5 - Test scope filter (Department)")
def test_scope_filter_department():
    """Test that _apply_scope_filter correctly filters by department_id"""
    from api_v2.services.inbox_service import _apply_scope_filter
    
    user = MockUser(role='Department Administrator', department_id=3)
    
    subcases = [
        MockSubcase(1, 'INCIDENT_RESPONSE', 'SECTION_ACCEPTED_PENDING_DEPT', target_org_unit_id=2),
        MockSubcase(2, 'INCIDENT_RESPONSE', 'SECTION_ACCEPTED_PENDING_DEPT', target_org_unit_id=3),
        MockSubcase(3, 'INCIDENT_RESPONSE', 'SECTION_ACCEPTED_PENDING_DEPT', target_org_unit_id=3),
    ]
    
    filtered = _apply_scope_filter(subcases, user)
    
    print(f"  Input: {len(subcases)} subcases")
    print(f"  Filtered: {len(filtered)} subcases")
    
    assert len(filtered) == 2, f"Expected 2 subcases for department_id=3, got {len(filtered)}"
    assert all(sc.TargetOrgUnitID == 3 for sc in filtered), "Filtered subcases should only have TargetOrgUnitID=3"
    print(f"  ✓ Correctly filtered to department_id=3")


@test("Layer 2.6 - Test scope filter (Administration - no filter)")
def test_scope_filter_administration():
    """Test that _apply_scope_filter does NOT filter for administration"""
    from api_v2.services.inbox_service import _apply_scope_filter
    
    user = MockUser(role='Administration Administrator')
    
    subcases = [
        MockSubcase(1, 'INCIDENT_RESPONSE', 'DEPT_ACCEPTED_PENDING_ADMIN', target_org_unit_id=1),
        MockSubcase(2, 'INCIDENT_RESPONSE', 'DEPT_ACCEPTED_PENDING_ADMIN', target_org_unit_id=2),
        MockSubcase(3, 'INCIDENT_RESPONSE', 'DEPT_ACCEPTED_PENDING_ADMIN', target_org_unit_id=3),
    ]
    
    filtered = _apply_scope_filter(subcases, user)
    
    print(f"  Input: {len(subcases)} subcases")
    print(f"  Filtered: {len(filtered)} subcases")
    
    assert len(filtered) == 3, f"Expected all 3 subcases for administration, got {len(filtered)}"
    print(f"  ✓ Correctly returns all subcases (no filtering)")


@test("Layer 2.7 - Test compute allowed actions (Section)")
def test_compute_allowed_actions_section():
    """Test that _compute_allowed_actions returns correct actions for Section role"""
    from api_v2.services.inbox_service import _compute_allowed_actions
    
    user = MockUser(role='Section Administrator', section_id=1)
    subcase = MockSubcase(1, 'INCIDENT_RESPONSE', 'SUBMITTED_TO_SECTION', target_org_unit_id=1)
    
    actions = _compute_allowed_actions(subcase, user)
    
    print(f"  Actions: {actions}")
    
    assert 'view' in actions, "Should include 'view' action"
    assert 'accept' in actions, "Should include 'accept' action for SUBMITTED_TO_SECTION"
    assert 'reject' in actions, "Should include 'reject' action for SUBMITTED_TO_SECTION"
    print(f"  ✓ Correct actions for Section + SUBMITTED_TO_SECTION")


@test("Layer 2.8 - Test compute allowed actions (Department)")
def test_compute_allowed_actions_department():
    """Test that _compute_allowed_actions returns correct actions for Department role"""
    from api_v2.services.inbox_service import _compute_allowed_actions
    
    user = MockUser(role='Department Administrator', department_id=2)
    subcase = MockSubcase(2, 'INCIDENT_RESPONSE', 'SECTION_ACCEPTED_PENDING_DEPT', target_org_unit_id=2)
    
    actions = _compute_allowed_actions(subcase, user)
    
    print(f"  Actions: {actions}")
    
    assert 'view' in actions, "Should include 'view' action"
    assert 'accept' in actions, "Should include 'accept' action for SECTION_ACCEPTED_PENDING_DEPT"
    assert 'reject' in actions, "Should include 'reject' action for SECTION_ACCEPTED_PENDING_DEPT"
    print(f"  ✓ Correct actions for Department + SECTION_ACCEPTED_PENDING_DEPT")


@test("Layer 2.9 - Test compute allowed actions (Administration)")
def test_compute_allowed_actions_administration():
    """Test that _compute_allowed_actions returns correct actions for Administration role"""
    from api_v2.services.inbox_service import _compute_allowed_actions
    
    user = MockUser(role='Administration Administrator')
    subcase = MockSubcase(3, 'INCIDENT_RESPONSE', 'DEPT_ACCEPTED_PENDING_ADMIN', target_org_unit_id=1)
    
    actions = _compute_allowed_actions(subcase, user)
    
    print(f"  Actions: {actions}")
    
    assert 'view' in actions, "Should include 'view' action"
    assert 'accept' in actions, "Should include 'accept' action for DEPT_ACCEPTED_PENDING_ADMIN"
    assert 'reject' in actions, "Should include 'reject' action for DEPT_ACCEPTED_PENDING_ADMIN"
    print(f"  ✓ Correct actions for Administration + DEPT_ACCEPTED_PENDING_ADMIN")


@test("Layer 2.10 - Test compute allowed actions (wrong status)")
def test_compute_allowed_actions_wrong_status():
    """Test that _compute_allowed_actions returns only 'view' for wrong status"""
    from api_v2.services.inbox_service import _compute_allowed_actions
    
    user = MockUser(role='Section Administrator', section_id=1)
    # Wrong status for this role
    subcase = MockSubcase(1, 'INCIDENT_RESPONSE', 'SECTION_ACCEPTED_PENDING_DEPT', target_org_unit_id=1)
    
    actions = _compute_allowed_actions(subcase, user)
    
    print(f"  Actions: {actions}")
    
    assert 'view' in actions, "Should include 'view' action"
    assert 'accept' not in actions, "Should NOT include 'accept' for wrong status"
    assert 'reject' not in actions, "Should NOT include 'reject' for wrong status"
    print(f"  ✓ Correctly returns only 'view' for wrong status")


@test("Layer 2.11 - Test build inbox item")
def test_build_inbox_item():
    """Test that _build_inbox_item formats data correctly"""
    from api_v2.services.inbox_service import _build_inbox_item
    from datetime import datetime
    
    user = MockUser(role='Section Administrator', section_id=1)
    created_at = datetime(2026, 1, 30, 12, 0, 0)
    subcase = MockSubcase(
        subcase_id=100,
        case_type='INCIDENT_RESPONSE',
        status='SUBMITTED_TO_SECTION',
        target_org_unit_id=1,
        incident_id=500,
        seasonal_report_id=None,
        created_at=created_at
    )
    
    item = _build_inbox_item(subcase, user)
    
    print(f"  Item keys: {list(item.keys())}")
    
    assert item['subcase_id'] == 100, "subcase_id mismatch"
    assert item['case_type'] == 'INCIDENT_RESPONSE', "case_type mismatch"
    assert item['status'] == 'SUBMITTED_TO_SECTION', "status mismatch"
    assert item['target_org_unit_id'] == 1, "target_org_unit_id mismatch"
    assert item['incident_id'] == 500, "incident_id mismatch"
    assert item['seasonal_report_id'] is None, "seasonal_report_id should be None"
    assert item['created_at'] == created_at, "created_at mismatch"
    assert 'allowed_actions' in item, "allowed_actions missing"
    assert isinstance(item['allowed_actions'], list), "allowed_actions should be a list"
    
    print(f"  ✓ Item formatted correctly: {item}")


# =============================================================================
# LAYER 3: INTEGRATION TESTS (Real Database)
# =============================================================================

@test("Layer 3.1 - Verify DB layer functions are available")
def test_db_layer_functions_available():
    """Verify that required DB layer functions exist"""
    from api_v2.db_layer import administrative_subcase_db
    
    assert hasattr(administrative_subcase_db, 'get_subcases_pending_for_section'), \
        "get_subcases_pending_for_section not found in DB layer"
    print("  ✓ get_subcases_pending_for_section found")
    
    assert hasattr(administrative_subcase_db, 'get_subcases_pending_for_department'), \
        "get_subcases_pending_for_department not found in DB layer"
    print("  ✓ get_subcases_pending_for_department found")
    
    assert hasattr(administrative_subcase_db, 'get_subcases_pending_for_administration'), \
        "get_subcases_pending_for_administration not found in DB layer"
    print("  ✓ get_subcases_pending_for_administration found")


@test("Layer 3.2 - Check if test data exists (SubcaseID 53)")
def test_check_test_data_exists():
    """Check if SubcaseID 53 exists for testing"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT 
                SubcaseID,
                CaseType,
                Status,
                TargetOrgUnitID,
                SeasonalReportID
            FROM dbo.APP_AdministrativeSubcase
            WHERE SubcaseID = 53
        """)
        
        row = cursor.fetchone()
        
        if row:
            print(f"  ✓ SubcaseID 53 exists:")
            print(f"    - Type: {row.CaseType}")
            print(f"    - Status: {row.Status}")
            print(f"    - Target: {row.TargetOrgUnitID}")
            print(f"    - SeasonalReportID: {row.SeasonalReportID}")
        else:
            print(f"  ⚠️  SubcaseID 53 not found (integration tests may be limited)")
            
    finally:
        cursor.close()
        conn.close()


@test("Layer 3.3 - Test get_section_inbox with real data")
def test_get_section_inbox_integration():
    """Test get_section_inbox with real database data"""
    from api_v2.services.inbox_service import get_section_inbox
    
    # Create a mock user with Section Administrator role
    # Note: We don't know which section_id has data, so we'll try section_id=1
    user = MockUser(role='Section Administrator', section_id=1)
    
    try:
        inbox = get_section_inbox(user)
        
        print(f"  Retrieved {len(inbox)} inbox item(s)")
        
        # Verify structure of returned items
        if inbox:
            first_item = inbox[0]
            print(f"  First item keys: {list(first_item.keys())}")
            
            # Check required fields
            required_fields = [
                'subcase_id', 'case_type', 'incident_id', 'seasonal_report_id',
                'target_org_unit_id', 'status', 'created_at', 'allowed_actions'
            ]
            
            for field in required_fields:
                assert field in first_item, f"Missing required field: {field}"
            
            print(f"  ✓ All required fields present")
            
            # Verify all items are for this section
            for item in inbox:
                assert item['target_org_unit_id'] == 1, \
                    f"Item {item['subcase_id']} has wrong target_org_unit_id: {item['target_org_unit_id']}"
            
            print(f"  ✓ All items correctly filtered to section_id=1")
            
            # Verify all items have SUBMITTED_TO_SECTION status
            for item in inbox:
                assert item['status'] == 'SUBMITTED_TO_SECTION', \
                    f"Item {item['subcase_id']} has wrong status: {item['status']}"
            
            print(f"  ✓ All items have correct status (SUBMITTED_TO_SECTION)")
            
            # Verify allowed_actions
            for item in inbox:
                assert isinstance(item['allowed_actions'], list), \
                    f"Item {item['subcase_id']} allowed_actions is not a list"
                assert 'view' in item['allowed_actions'], \
                    f"Item {item['subcase_id']} missing 'view' action"
            
            print(f"  ✓ All items have valid allowed_actions")
        else:
            print(f"  ⚠️  No items in inbox (may be normal if no data for section_id=1)")
            
    except Exception as e:
        print(f"  Error: {str(e)}")
        raise


@test("Layer 3.4 - Test get_department_inbox with real data")
def test_get_department_inbox_integration():
    """Test get_department_inbox with real database data"""
    from api_v2.services.inbox_service import get_department_inbox
    
    # Create a mock user with Department Administrator role
    user = MockUser(role='Department Administrator', department_id=1)
    
    try:
        inbox = get_department_inbox(user)
        
        print(f"  Retrieved {len(inbox)} inbox item(s)")
        
        # Verify structure of returned items
        if inbox:
            first_item = inbox[0]
            
            # Check required fields
            required_fields = [
                'subcase_id', 'case_type', 'incident_id', 'seasonal_report_id',
                'target_org_unit_id', 'status', 'created_at', 'allowed_actions'
            ]
            
            for field in required_fields:
                assert field in first_item, f"Missing required field: {field}"
            
            print(f"  ✓ All required fields present")
            
            # Verify all items are for this department
            for item in inbox:
                assert item['target_org_unit_id'] == 1, \
                    f"Item {item['subcase_id']} has wrong target_org_unit_id: {item['target_org_unit_id']}"
            
            print(f"  ✓ All items correctly filtered to department_id=1")
            
            # Verify all items have SECTION_ACCEPTED_PENDING_DEPT status
            for item in inbox:
                assert item['status'] == 'SECTION_ACCEPTED_PENDING_DEPT', \
                    f"Item {item['subcase_id']} has wrong status: {item['status']}"
            
            print(f"  ✓ All items have correct status (SECTION_ACCEPTED_PENDING_DEPT)")
        else:
            print(f"  ⚠️  No items in inbox (may be normal if no data for department_id=1)")
            
    except Exception as e:
        print(f"  Error: {str(e)}")
        raise


@test("Layer 3.5 - Test get_administration_inbox with real data")
def test_get_administration_inbox_integration():
    """Test get_administration_inbox with real database data"""
    from api_v2.services.inbox_service import get_administration_inbox
    
    # Create a mock user with Administration Administrator role
    user = MockUser(role='Administration Administrator')
    
    try:
        inbox = get_administration_inbox(user)
        
        print(f"  Retrieved {len(inbox)} inbox item(s)")
        
        # Verify structure of returned items
        if inbox:
            first_item = inbox[0]
            
            # Check required fields
            required_fields = [
                'subcase_id', 'case_type', 'incident_id', 'seasonal_report_id',
                'target_org_unit_id', 'status', 'created_at', 'allowed_actions'
            ]
            
            for field in required_fields:
                assert field in first_item, f"Missing required field: {field}"
            
            print(f"  ✓ All required fields present")
            
            # Verify all items have DEPT_ACCEPTED_PENDING_ADMIN status
            for item in inbox:
                assert item['status'] == 'DEPT_ACCEPTED_PENDING_ADMIN', \
                    f"Item {item['subcase_id']} has wrong status: {item['status']}"
            
            print(f"  ✓ All items have correct status (DEPT_ACCEPTED_PENDING_ADMIN)")
        else:
            print(f"  ⚠️  No items in inbox (may be normal if no data at admin stage)")
            
    except Exception as e:
        print(f"  Error: {str(e)}")
        raise


# =============================================================================
# MAIN TEST EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("STEP 3.12 — INBOX SERVICE TEST SUITE")
    print("Testing inbox_service.py across 3 layers")
    print("="*80)
    
    layer1_passed = 0
    layer1_total = 0
    layer2_passed = 0
    layer2_total = 0
    layer3_passed = 0
    layer3_total = 0
    
    # LAYER 1: Static Checks
    print("\n" + "="*80)
    print("LAYER 1: STATIC CHECKS")
    print("="*80)
    
    tests_layer1 = [
        test_import_inbox_service,
        test_public_functions_exist,
        test_helper_functions_exist,
        test_function_signatures,
    ]
    
    for test_func in tests_layer1:
        layer1_total += 1
        try:
            test_func()
            layer1_passed += 1
        except Exception:
            pass
    
    # LAYER 2: Unit Tests
    print("\n" + "="*80)
    print("LAYER 2: UNIT TESTS (Mock Data)")
    print("="*80)
    
    tests_layer2 = [
        test_section_role_validation,
        test_department_role_validation,
        test_administration_role_validation,
        test_scope_filter_section,
        test_scope_filter_department,
        test_scope_filter_administration,
        test_compute_allowed_actions_section,
        test_compute_allowed_actions_department,
        test_compute_allowed_actions_administration,
        test_compute_allowed_actions_wrong_status,
        test_build_inbox_item,
    ]
    
    for test_func in tests_layer2:
        layer2_total += 1
        try:
            test_func()
            layer2_passed += 1
        except Exception:
            pass
    
    # LAYER 3: Integration Tests
    print("\n" + "="*80)
    print("LAYER 3: INTEGRATION TESTS (Real Database)")
    print("="*80)
    
    tests_layer3 = [
        test_db_layer_functions_available,
        test_check_test_data_exists,
        test_get_section_inbox_integration,
        test_get_department_inbox_integration,
        test_get_administration_inbox_integration,
    ]
    
    for test_func in tests_layer3:
        layer3_total += 1
        try:
            test_func()
            layer3_passed += 1
        except Exception:
            pass
    
    # SUMMARY
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Layer 1 (Static):      {layer1_passed}/{layer1_total} passed")
    print(f"Layer 2 (Unit):        {layer2_passed}/{layer2_total} passed")
    print(f"Layer 3 (Integration): {layer3_passed}/{layer3_total} passed")
    print(f"TOTAL:                 {layer1_passed + layer2_passed + layer3_passed}/{layer1_total + layer2_total + layer3_total} passed")
    
    total_passed = layer1_passed + layer2_passed + layer3_passed
    total_tests = layer1_total + layer2_total + layer3_total
    
    if total_passed == total_tests:
        print("\n✅ ALL TESTS PASSED! STEP 3.12 Prompt 1 is 100% complete!")
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed. Review output above.")
