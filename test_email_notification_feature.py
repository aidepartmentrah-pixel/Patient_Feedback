"""
Test Email Notification Feature
Tests the email field addition and notification service.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from core.database import get_connection


def test_email_column_exists():
    """Test 1: Check if Email column exists in APP_Users table."""
    print("\n" + "="*60)
    print("TEST 1: Check Email column exists in APP_Users")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, CHARACTER_MAXIMUM_LENGTH
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = 'APP_Users' AND COLUMN_NAME = 'Email'
        """)
        row = cursor.fetchone()
        
        if row:
            print(f"✅ Email column exists:")
            print(f"   - Type: {row.DATA_TYPE}")
            print(f"   - Nullable: {row.IS_NULLABLE}")
            print(f"   - Max Length: {row.CHARACTER_MAXIMUM_LENGTH}")
            return True
        else:
            print("❌ Email column does NOT exist in APP_Users table")
            print("   Please run the migration: phase_email_add_user_email_column.sql")
            return False
            
    finally:
        cursor.close()
        conn.close()


def test_update_user_identity_with_email():
    """Test 2: Update user identity with email field."""
    print("\n" + "="*60)
    print("TEST 2: Update user identity with email")
    print("="*60)
    
    from backend.api.services.user_management_service import update_user_identity_service
    
    # Get a test user
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT TOP 1 UserID, Username FROM dbo.APP_Users WHERE Username != 'software_admin'")
        user = cursor.fetchone()
        
        if not user:
            print("⚠️ No test users found. Creating test user...")
            return False
        
        user_id = user.UserID
        username = user.Username
        test_email = f"test_{username}@hospital.local"
        
        print(f"   Testing with user: {username} (ID: {user_id})")
        print(f"   Setting email to: {test_email}")
        
        # Update identity with email
        update_user_identity_service(
            user_id=user_id,
            display_name=None,  # No change
            department_display_name=None,  # No change
            email=test_email
        )
        
        # Verify the update
        cursor.execute("SELECT Email FROM dbo.APP_Users WHERE UserID = ?", (user_id,))
        result = cursor.fetchone()
        
        if result and result.Email == test_email.lower():
            print(f"✅ Email updated successfully: {result.Email}")
            return True
        else:
            print(f"❌ Email update failed. Got: {result.Email if result else 'None'}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False
        
    finally:
        cursor.close()
        conn.close()


def test_list_users_includes_email():
    """Test 3: List users includes email in response."""
    print("\n" + "="*60)
    print("TEST 3: List users includes email field")
    print("="*60)
    
    from backend.api.services.user_management_service import list_users_for_settings_service
    
    try:
        users = list_users_for_settings_service()
        
        if not users:
            print("⚠️ No users returned")
            return False
        
        # Check if email field exists in response
        first_user = users[0]
        
        if 'email' in first_user:
            print(f"✅ Email field present in user response")
            print(f"   Sample user: {first_user.get('username')} - email: {first_user.get('email')}")
            return True
        else:
            print("❌ Email field NOT found in user response")
            print(f"   Fields present: {list(first_user.keys())}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def test_notification_service_mock():
    """Test 4: Notification service in mock mode."""
    print("\n" + "="*60)
    print("TEST 4: Notification service (mock mode)")
    print("="*60)
    
    try:
        from backend.api.services.notification_service import (
            send_notification,
            send_subcase_assignment_notification
        )
        from backend.config.notification_config import NOTIFICATION_MODE
        
        print(f"   Current mode: {NOTIFICATION_MODE}")
        
        # Test basic notification
        result = send_notification(
            to_email="test@hospital.local",
            subject="Test Subject",
            body="Test body content",
            run_async=False
        )
        
        if result:
            print("✅ send_notification() works in mock mode")
        else:
            print("❌ send_notification() failed")
            return False
        
        # Test subcase assignment notification
        result2 = send_subcase_assignment_notification(
            to_email="admin@hospital.local",
            case_id=12345
        )
        
        if result2:
            print("✅ send_subcase_assignment_notification() works")
        else:
            print("❌ send_subcase_assignment_notification() failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error importing notification service: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_get_section_admin_email():
    """Test 5: Get section admin email lookup."""
    print("\n" + "="*60)
    print("TEST 5: Get section admin email lookup")
    print("="*60)
    
    try:
        from backend.api.services.notification_service import get_section_admin_email
        
        # Get a valid org unit
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT TOP 1 urs.OrgUnitID
            FROM dbo.APP_UserRoleScope urs
            INNER JOIN dbo.APP_Roles r ON urs.RoleID = r.RoleID
            WHERE r.RoleCode = 'SECTION_ADMIN'
        """)
        row = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if not row:
            print("⚠️ No section admin assignments found")
            return True  # Not a failure, just no data
        
        org_unit_id = row.OrgUnitID
        print(f"   Testing with org_unit_id: {org_unit_id}")
        
        email = get_section_admin_email(org_unit_id)
        
        if email:
            print(f"✅ Found admin email: {email}")
        else:
            print(f"⚠️ No email found for admin (may not have email set yet)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("EMAIL NOTIFICATION FEATURE - TEST SUITE")
    print("="*60)
    
    results = []
    
    # Test 1: Check column
    results.append(("Email column exists", test_email_column_exists()))
    
    # Only proceed with other tests if column exists
    if results[0][1]:
        results.append(("Update identity with email", test_update_user_identity_with_email()))
        results.append(("List users includes email", test_list_users_includes_email()))
    else:
        print("\n⚠️ Skipping remaining tests - Email column must be created first")
    
    # Test notification service (doesn't depend on column)
    results.append(("Notification service mock", test_notification_service_mock()))
    results.append(("Get section admin email", test_get_section_admin_email()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print("\n⚠️ Some tests failed. Check output above.")
        return 1


if __name__ == "__main__":
    exit(main())
