"""Debug test A2 - Section Rejection"""
import sys
sys.path.insert(0, r'c:\Users\IT\Documents\GitHub Repository\Patient_Feedback')

from test_workflow_comprehensive import create_test_subcase, create_test_user, cleanup_test_subcase, get_db_cursor
from backend.api_v2.routers.workflow_router import act_on_case

print("="*60)
print("DEBUGGING TEST A2: Section Rejection")
print("="*60)

subcase_id = create_test_subcase()
print(f"Created subcase {subcase_id}")

section_admin = create_test_user('SECTION_ADMIN', 2, 'Section')
print(f"Created section admin user")

print("\n--- Calling act_on_case with REJECT action ---")
try:
    # Add some debugging to the router
    import backend.api_v2.routers.workflow_router as router_module
    original_func = router_module.case_response_service.reject_responsibility
    
    def debug_reject_responsibility(*args, **kwargs):
        print(f"  → reject_responsibility called with args={args}, kwargs={kwargs}")
        try:
            result = original_func(*args, **kwargs)
            print(f"  → reject_responsibility SUCCESS")
            return result
        except Exception as e:
            print(f"  → reject_responsibility FAILED: {e}")
            raise
    
    router_module.case_response_service.reject_responsibility = debug_reject_responsibility
    
    response = act_on_case(
        subcase_id=subcase_id,
        body={
            'action': 'REJECT',
            'rejection_text': 'This is not our department responsibility'
        },
        current_user=section_admin
    )
    print(f"Response: {response}")
    print(f"Success: {response.get('success')}")
except Exception as e:
    print(f"ERROR during act_on_case: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Checking database ---")
conn, cursor = get_db_cursor()
try:
    cursor.execute("SELECT Status, SectionRejectionText FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = ?", (subcase_id,))
    result = cursor.fetchone()
    if result:
        print(f"Status: {result.Status}")
        print(f"SectionRejectionText: '{result.SectionRejectionText}'")
        
        if result.Status == 'SECTION_DENIED':
            print("✓ Status changed correctly")
        else:
            print(f"✗ Status is wrong - expected SECTION_DENIED, got {result.Status}")
        
        if result.SectionRejectionText and len(result.SectionRejectionText) > 0:
            print("✓ Rejection text saved correctly")
        else:
            print("✗ Rejection text is missing or empty")
    else:
        print("✗ Subcase not found in database")
finally:
    cursor.close()
    conn.close()

cleanup_test_subcase(subcase_id)
print("\nTest complete")
