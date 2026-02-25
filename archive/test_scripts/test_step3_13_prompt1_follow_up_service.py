"""
STEP 3.13 Prompt 1 Test — Follow-Up Service (Skeleton & Read API)

Tests the basic read-only functionality of follow_up_service.py:
1. Module exists and imports correctly
2. get_action_items_for_user function exists
3. Authentication checks work
4. Scope filtering works correctly (Phase 2.5 integration)
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


def test(description):
    """Test decorator"""
    def decorator(func):
        def wrapper():
            print(f"\n{'='*70}")
            print(f"TEST: {description}")
            print('='*70)
            try:
                func()
                print("✅ PASSED")
            except Exception as e:
                print(f"❌ FAILED: {str(e)}")
                import traceback
                traceback.print_exc()
        return wrapper
    return decorator


# Mock user class for testing
class MockUser:
    def __init__(self, user_id=None, allowed_unit_ids=None):
        self.user_id = user_id
        self.allowed_unit_ids = allowed_unit_ids if allowed_unit_ids else set()


@test("1. Module Import & Structure Verification")
def test_module_import():
    """Verify the follow_up_service module exists and can be imported."""
    from api_v2.services import follow_up_service
    
    print("  ✓ Module imported successfully")
    
    # Check for required function
    assert hasattr(follow_up_service, 'get_action_items_for_user'), \
        "get_action_items_for_user function not found"
    print("  ✓ get_action_items_for_user function exists")
    
    # Check for Unauthorized exception
    assert hasattr(follow_up_service, 'Unauthorized'), \
        "Unauthorized exception class not found"
    print("  ✓ Unauthorized exception class exists")


@test("2. Authentication Check — None User")
def test_auth_none_user():
    """Test that None user raises Unauthorized."""
    from api_v2.services.follow_up_service import get_action_items_for_user, Unauthorized
    
    try:
        result = get_action_items_for_user(None)
        raise AssertionError("Should have raised Unauthorized for None user")
    except Unauthorized as e:
        print(f"  ✓ Correctly raised Unauthorized: {str(e)}")


@test("3. Authentication Check — User Without user_id")
def test_auth_no_user_id():
    """Test that user without user_id raises Unauthorized."""
    from api_v2.services.follow_up_service import get_action_items_for_user, Unauthorized
    
    user_without_id = MockUser(user_id=None)
    
    try:
        result = get_action_items_for_user(user_without_id)
        raise AssertionError("Should have raised Unauthorized for user without user_id")
    except Unauthorized as e:
        print(f"  ✓ Correctly raised Unauthorized: {str(e)}")


@test("4. Empty Scope Check — User With No allowed_unit_ids")
def test_empty_scope():
    """Test that user with no allowed_unit_ids gets empty list."""
    from api_v2.services.follow_up_service import get_action_items_for_user
    
    user_no_scope = MockUser(user_id=999, allowed_unit_ids=set())
    
    result = get_action_items_for_user(user_no_scope)
    
    assert result == [], f"Expected empty list, got {result}"
    print(f"  ✓ User with no scope gets empty list: {result}")


@test("5. Real User Integration Test")
def test_real_user_integration():
    """
    Test with a real user from the database.
    This tests the full flow: DB fetch -> scope filtering -> return
    """
    from api_v2.services.follow_up_service import get_action_items_for_user
    from core.database import get_connection
    
    print("\n[SETUP] Finding a real user in the database...")
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Get a user with action items
        cursor.execute("""
            SELECT TOP 1 
                u.UserID,
                ai.ActionItemID,
                ai.SubcaseID,
                sc.TargetOrgUnitID
            FROM dbo.APP_Users u
            INNER JOIN dbo.APP_SubcaseActionItem ai ON u.UserID = ai.AssignedToUserID
            INNER JOIN dbo.APP_AdministrativeSubcase sc ON ai.SubcaseID = sc.SubcaseID
            WHERE u.UserID IS NOT NULL
        """)
        
        row = cursor.fetchone()
        
        if not row:
            print("  ⚠️  No users with action items found in database")
            print("  ℹ️  This is okay - test skipped")
            return
        
        user_id = row.UserID
        action_item_id = row.ActionItemID
        subcase_id = row.SubcaseID
        target_org_unit_id = row.TargetOrgUnitID
        
        print(f"  Found test user: UserID={user_id}")
        print(f"  Has ActionItem: ID={action_item_id}, SubcaseID={subcase_id}")
        print(f"  Subcase targets: OrgUnitID={target_org_unit_id}")
        
        # Test 1: User WITH scope (should see action item)
        print("\n[TEST 1] User with correct scope...")
        user_with_scope = MockUser(
            user_id=user_id,
            allowed_unit_ids={target_org_unit_id}
        )
        
        result = get_action_items_for_user(user_with_scope)
        
        print(f"  Result: {len(result)} action item(s) returned")
        assert len(result) > 0, "Expected at least 1 action item"
        
        # Verify the action item is in the results
        found = any(item['action_item_id'] == action_item_id for item in result)
        assert found, f"Action item {action_item_id} not found in results"
        print(f"  ✓ Action item {action_item_id} found in results")
        
        # Test 2: User WITHOUT scope (should see nothing)
        print("\n[TEST 2] User with wrong scope (should filter out)...")
        user_wrong_scope = MockUser(
            user_id=user_id,
            allowed_unit_ids={99999}  # Non-existent org unit
        )
        
        result_filtered = get_action_items_for_user(user_wrong_scope)
        
        print(f"  Result: {len(result_filtered)} action item(s) returned")
        assert len(result_filtered) == 0, "Expected 0 action items due to scope filter"
        print(f"  ✓ Scope filter correctly excluded all items")
        
    finally:
        cursor.close()
        conn.close()


@test("6. Function Signature Verification")
def test_function_signature():
    """Verify the function signature matches specification."""
    import inspect
    from api_v2.services.follow_up_service import get_action_items_for_user
    
    sig = inspect.signature(get_action_items_for_user)
    
    print(f"  Function signature: {sig}")
    
    # Check parameters
    params = list(sig.parameters.keys())
    assert params == ['current_user'], \
        f"Expected parameter 'current_user', got {params}"
    print(f"  ✓ Correct parameters: {params}")
    
    # Check return annotation if present
    if sig.return_annotation != inspect.Signature.empty:
        print(f"  ✓ Return annotation: {sig.return_annotation}")


@test("7. Docstring Verification")
def test_docstring():
    """Verify the function has proper documentation."""
    from api_v2.services.follow_up_service import get_action_items_for_user
    
    doc = get_action_items_for_user.__doc__
    
    assert doc is not None, "Function should have a docstring"
    assert "scope" in doc.lower(), "Docstring should mention scope"
    assert "action items" in doc.lower(), "Docstring should mention action items"
    
    print(f"  ✓ Docstring present and contains key concepts")
    print(f"\n  Docstring excerpt:")
    print(f"  {doc[:200]}...")


# =============================================================================
# MAIN TEST EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("STEP 3.13 PROMPT 1 — FOLLOW-UP SERVICE TEST SUITE")
    print("Testing: Skeleton & Read API (Phase 2.5 Scope Aligned)")
    print("="*80)
    
    # Module structure tests
    test_module_import()
    test_function_signature()
    test_docstring()
    
    # Authentication tests
    test_auth_none_user()
    test_auth_no_user_id()
    test_empty_scope()
    
    # Integration test
    test_real_user_integration()
    
    print("\n" + "="*80)
    print("TEST SUITE COMPLETE")
    print("="*80)
    print("\n✅ If all tests passed, Prompt 1 is complete!")
    print("📋 Next: Prompt 2 — Execution Actions")
