"""
Check complaints for target department assignments
"""
import sys
sys.path.insert(0, r"C:\Users\IT\Documents\GitHub Repository\Patient_Feedback\backend")

from api.services.monthly_report_service import monthly_report_service

# Get all complaints for December 2025
result = monthly_report_service.generate_monthly_report(
    year=2025,
    month=12,
    start_date=None,
    end_date=None,
    mode="detailed",
    scope=None,
    administration_ids=None,
    department_ids=None,
    section_ids=None,
    page=1,
    page_size=9999
)

complaints = result.get("complaints", [])

print(f"Total complaints: {len(complaints)}")
print("=" * 80)

# Categorize by target department status
no_targets = []
single_target = []
multi_target = []

for c in complaints:
    targets = c.get("target_departments", [])
    if not targets or len(targets) == 0:
        no_targets.append(c)
    elif len(targets) == 1:
        single_target.append(c)
    else:
        multi_target.append(c)

print(f"\n✓ Complaints with NO target departments: {len(no_targets)}")
print(f"✓ Complaints with 1 target department: {len(single_target)}")
print(f"✓ Complaints with multiple target departments: {len(multi_target)}")

if no_targets:
    print(f"\n⚠️ WARNING: {len(no_targets)} complaints have no target departments!")
    print("These will NOT appear in any department/administration reports")
    print("\nSample IDs without targets:")
    for c in no_targets[:10]:
        print(f"  - Complaint #{c['id']}: {c.get('complaint_text', '')[:50]}...")

print("\n" + "=" * 80)
print("CONCLUSION:")
if no_targets:
    print(f"❌ {len(no_targets)} complaints are missing target departments")
    print("   These complaints won't appear in organizational unit reports")
else:
    print("✓ All complaints have target departments assigned")
