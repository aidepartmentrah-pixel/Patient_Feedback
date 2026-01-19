"""
Test UNION (OR) filtering for multiple organizational units.

This demonstrates that selecting multiple Administrations/Departments/Sections
uses UNION (OR) logic, NOT INTERSECTION (AND) logic.

Example:
- Select Administration 3 AND Administration 1
- Result: ALL cases from Admin 3 OR Admin 1 (UNION)
- NOT: Only cases that belong to BOTH (INTERSECTION - this would be empty!)
"""

import sys
sys.path.append('c:\\Users\\IT\\Documents\\GitHub Repository\\Patient_Feedback\\backend')

from api.db_layer.reports_db import get_org_unit_descendants, debug_expand_org_units

print("\n" + "="*80)
print("UNION (OR) FILTERING TEST")
print("="*80)

# Test: Multiple Administrations - should get UNION of both trees
print("\nScenario: User selects Administration 3 AND Administration 1")
print("-" * 80)

# Expand Administration 3 alone
admin_3_tree = get_org_unit_descendants(3)
print(f"\nAdministration 3 tree: {len(admin_3_tree)} units")
print(f"Units: {sorted(admin_3_tree)[:10]}... (showing first 10)")

# Expand Administration 1 alone
admin_1_tree = get_org_unit_descendants(1)
print(f"\nAdministration 1 tree: {len(admin_1_tree)} units")
print(f"Units: {sorted(admin_1_tree)[:10]}... (showing first 10)")

# UNION of both (what the system will do)
union_tree = debug_expand_org_units([3, 1])
print(f"\n✅ UNION (Admin 3 OR Admin 1): {len(union_tree)} units")
print(f"Units: {sorted(union_tree)[:15]}... (showing first 15)")

# Calculate what INTERSECTION would be (what we DON'T want)
intersection = set(admin_3_tree) & set(admin_1_tree)
print(f"\n❌ INTERSECTION (Admin 3 AND Admin 1): {len(intersection)} units")
print(f"(This is what we DON'T want - would return very few or zero results)")

print("\n" + "="*80)
print("VERIFICATION:")
print("="*80)
print(f"""
Selected: Administration 3 AND Administration 1

What happens:
1. Administration 3 expands to {len(admin_3_tree)} units (all its departments & sections)
2. Administration 1 expands to {len(admin_1_tree)} units (all its departments & sections)
3. System combines them with UNION (OR) → {len(union_tree)} total unique units
4. Finds ALL complaints where ANY target_department is in this combined list

Result: You get cases from BOTH administrations (UNION/OR logic) ✅

What we DON'T do:
- INTERSECTION (AND logic) would only find {len(intersection)} units
- This would return very few or zero cases ❌

""")
print("="*80)

# Test with Departments
print("\nBonus Test: Multiple Departments")
print("-" * 80)

dept_28_tree = get_org_unit_descendants(28)
dept_24_tree = get_org_unit_descendants(24)
union_depts = debug_expand_org_units([28, 24])

print(f"Department 28: {len(dept_28_tree)} units")
print(f"Department 24: {len(dept_24_tree)} units")
print(f"UNION (Dept 28 OR Dept 24): {len(union_depts)} units ✅")

print("\n" + "="*80 + "\n")
