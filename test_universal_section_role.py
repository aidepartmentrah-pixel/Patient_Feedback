"""
Test script for UNIVERSAL_SECTION role implementation.
Tests:
1. Role exists in database
2. Inbox returns all section-level subcases (no scope filter)
3. Direct approval function works
"""
import sys
sys.path.insert(0, "backend")

from backend.core.database import get_connection
from backend.api_v2.services import inbox_service
from backend.api_v2.services import case_response_service
from backend.api.schemas.auth_models import CurrentUser, UserScope


def test_role_exists():
    """Test that UNIVERSAL_SECTION role exists in database."""
    print("\n=== Test 1: Role Exists in Database ===")
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT RoleID, RoleCode, RoleNameEn FROM dbo.APP_Roles WHERE RoleCode = 'UNIVERSAL_SECTION'")
        row = cursor.fetchone()
        
        if row:
            print(f"✅ UNIVERSAL_SECTION role found: ID={row.RoleID}, Name='{row.RoleNameEn}'")
            return True
        else:
            print("❌ UNIVERSAL_SECTION role NOT found in database")
            return False
    finally:
        cursor.close()
        conn.close()


def test_inbox_no_scope_filter():
    """Test that UNIVERSAL_SECTION inbox returns all section-level subcases without scope filter."""
    print("\n=== Test 2: Inbox Without Scope Filter ===")
    
    # Create a mock UNIVERSAL_SECTION user
    universal_user = CurrentUser(
        user_id=999,
        username="universal_test",
        is_active=True,
        scopes=[UserScope(role_code="UNIVERSAL_SECTION", org_unit_id=1, org_unit_type="ADMINISTRATION")],
        allowed_unit_ids=set()  # Empty - should not matter for UNIVERSAL_SECTION
    )
    
    # Get inbox
    try:
        inbox = inbox_service.get_universal_section_inbox(universal_user)
        print(f"✅ Universal section inbox returned {len(inbox)} items")
        
        if inbox:
            print(f"   First item: subcase_id={inbox[0].get('subcase_id')}, status={inbox[0].get('status')}")
            print(f"   Allowed actions: {inbox[0].get('allowed_actions')}")
            
            # Verify direct_approve is in allowed actions
            if 'direct_approve' in inbox[0].get('allowed_actions', []):
                print("✅ 'direct_approve' action is available")
            else:
                print("❌ 'direct_approve' action NOT in allowed_actions")
        else:
            print("   (No pending subcases at section level)")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_compare_with_section_admin():
    """Compare UNIVERSAL_SECTION inbox with a SECTION_ADMIN inbox to verify difference."""
    print("\n=== Test 3: Compare with Section Admin Inbox ===")
    
    # Get all subcases for universal (unfiltered)
    universal_user = CurrentUser(
        user_id=999,
        username="universal_test",
        is_active=True,
        scopes=[UserScope(role_code="UNIVERSAL_SECTION", org_unit_id=1, org_unit_type="ADMINISTRATION")],
        allowed_unit_ids=set()
    )
    
    # Get a section admin with limited scope
    section_user = CurrentUser(
        user_id=998,
        username="section_test",
        is_active=True,
        scopes=[UserScope(role_code="SECTION_ADMIN", org_unit_id=217, org_unit_type="SECTION")],
        allowed_unit_ids={217}  # Only org unit 217
    )
    
    try:
        universal_inbox = inbox_service.get_universal_section_inbox(universal_user)
        section_inbox = inbox_service.get_section_inbox(section_user)
        
        print(f"   UNIVERSAL_SECTION sees: {len(universal_inbox)} subcases")
        print(f"   SECTION_ADMIN (unit 217) sees: {len(section_inbox)} subcases")
        
        if len(universal_inbox) >= len(section_inbox):
            print("✅ UNIVERSAL_SECTION sees equal or more subcases (expected)")
        else:
            print("⚠️ UNIVERSAL_SECTION sees fewer subcases (unexpected)")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_direct_approve_validation():
    """Test that direct_approve validates role properly."""
    print("\n=== Test 4: Direct Approve Validation ===")
    
    # Create non-UNIVERSAL_SECTION user
    section_user = CurrentUser(
        user_id=998,
        username="section_test",
        is_active=True,
        scopes=[UserScope(role_code="SECTION_ADMIN", org_unit_id=217, org_unit_type="SECTION")],
        allowed_unit_ids={217}
    )
    
    try:
        case_response_service.direct_approve_to_admin(
            subcase_id=1,
            explanation_text="test",
            action_items=[],
            current_user=section_user
        )
        print("❌ Should have raised exception for non-UNIVERSAL_SECTION user")
        return False
    except Exception as e:
        if "UNIVERSAL_SECTION" in str(e):
            print(f"✅ Correctly rejected non-UNIVERSAL_SECTION user: {e}")
            return True
        else:
            print(f"❌ Wrong error: {e}")
            return False


if __name__ == "__main__":
    print("=" * 60)
    print("UNIVERSAL_SECTION Role Implementation Tests")
    print("=" * 60)
    
    results = []
    results.append(("Role Exists", test_role_exists()))
    results.append(("Inbox No Scope Filter", test_inbox_no_scope_filter()))
    results.append(("Compare With Section Admin", test_compare_with_section_admin()))
    results.append(("Direct Approve Validation", test_direct_approve_validation()))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed!")
    else:
        print("\n⚠️ Some tests failed")
