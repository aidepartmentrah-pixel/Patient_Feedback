"""Simple test - call service directly vs through router"""
import sys
sys.path.insert(0, r'c:\Users\IT\Documents\GitHub Repository\Patient_Feedback')

from test_workflow_comprehensive import create_test_subcase, create_test_user, cleanup_test_subcase, get_db_cursor
from backend.api_v2.services import case_response_service
from backend.api_v2.routers.workflow_router import act_on_case

print("="*60)
print("TEST 1: Direct Service Call")
print("="*60)

subcase_id1 = create_test_subcase()
section_admin1 = create_test_user('SECTION_ADMIN', 2, 'Section')

print(f"Subcase ID: {subcase_id1}")
print("Calling case_response_service.reject_responsibility directly...")

try:
    case_response_service.reject_responsibility(
        subcase_id=subcase_id1,
        rejection_text='Direct call rejection text',
        current_user=section_admin1
    )
    print("✓ Success")
except Exception as e:
    print(f"✗ Error: {e}")

conn, cursor = get_db_cursor()
cursor.execute("SELECT Status, SectionRejectionText FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = ?", (subcase_id1,))
result = cursor.fetchone()
print(f"Status: {result.Status}")
print(f"SectionRejectionText: '{result.SectionRejectionText}'")
cursor.close()
conn.close()

cleanup_test_subcase(subcase_id1)

print("\n" + "="*60)
print("TEST 2: Through Router")
print("="*60)

subcase_id2 = create_test_subcase()
section_admin2 = create_test_user('SECTION_ADMIN', 2, 'Section')

print(f"Subcase ID: {subcase_id2}")
print("Calling act_on_case through router...")

try:
    response = act_on_case(
        subcase_id=subcase_id2,
        body={
            'action': 'REJECT',
            'rejection_text': 'Router call rejection text'
        },
        current_user=section_admin2
    )
    print(f"✓ Success: {response}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

conn, cursor = get_db_cursor()
cursor.execute("SELECT Status, SectionRejectionText FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = ?", (subcase_id2,))
result = cursor.fetchone()
print(f"Status: {result.Status}")
print(f"SectionRejectionText: '{result.SectionRejectionText}'")
cursor.close()
conn.close()

cleanup_test_subcase(subcase_id2)

print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print("Direct call: Works correctly")
print("Router call: Check if text is saved")
