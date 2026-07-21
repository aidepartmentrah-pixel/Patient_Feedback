"""
Investigate missing complaints #14 and #15
"""
import sys
sys.path.insert(0, r"C:\Users\IT\Documents\GitHub Repository\Patient_Feedback\backend")

from api.services.monthly_report_service import monthly_report_service

# Get all complaints
result = monthly_report_service.generate_monthly_report(
    year=2025, month=12, start_date=None, end_date=None,
    mode="detailed", scope=None,
    administration_ids=None, department_ids=None, section_ids=None,
    page=1, page_size=9999
)

complaints = result.get("complaints", [])

# Find complaints 14 and 15
for c in complaints:
    if c["id"] in [14, 15]:
        print(f"\nComplaint #{c['id']}:")
        print(f"  Text: {c.get('complaint_text', '')[:100]}")
        print(f"  Target Departments ({len(c.get('target_departments', []))}):")
        for td in c.get("target_departments", []):
            print(f"    - Section ID: {td.get('section_id')}")
            print(f"      Section Name: {td.get('section_name')}")
            print(f"      Dept ID: {td.get('department_id')}")
            print(f"      Dept Name: {td.get('department_name')}")
            print(f"      Admin ID: {td.get('administration_id')}")
            print(f"      Admin Name: {td.get('administration_name')}")
            print(f"      Is Primary: {td.get('is_primary')}")
            print()
