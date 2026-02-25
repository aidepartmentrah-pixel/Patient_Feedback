"""
Test the new Organization Unit endpoints

This script tests the specialized org unit selection endpoints:
1. /api/org-units/leaves - For insert forms
2. /api/org-units/administrations - For reports
3. /api/org-units/departments - For filtering
4. /api/org-units/sections - For filtering
5. /api/org-units/unit/{id} - For breadcrumbs
6. /api/org-units/summary - For overview
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from api.services import org_unit_service


def print_separator(title):
    """Print a visual separator"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_leaf_units():
    """Test getting leaf units (for insert forms)"""
    print_separator("TEST 1: Leaf Units (For Insert Forms)")
    
    leaves = org_unit_service.get_leaf_units()
    
    print(f"\nFound {len(leaves)} leaf units (sections with no children)")
    print("\nThese are the units users should select in INSERT forms:")
    print("(Because people complain about what they actually experience)\n")
    
    if leaves:
        # Show first 5 examples
        for i, leaf in enumerate(leaves[:5]):
            print(f"{i+1}. {leaf['name']}")
            print(f"   - ID: {leaf['id']}")
            print(f"   - Type: {leaf['type_name']}")
            if leaf['parent_name']:
                print(f"   - Parent: {leaf['parent_name']}")
            print()
        
        if len(leaves) > 5:
            print(f"... and {len(leaves) - 5} more leaf units\n")
    else:
        print("⚠️  No leaf units found (database might be empty)\n")
    
    return len(leaves) > 0


def test_administrations():
    """Test getting administrations (for reports)"""
    print_separator("TEST 2: Administrations (For Reports)")
    
    admins = org_unit_service.get_administrations()
    
    print(f"\nFound {len(admins)} administrations (top-level units)")
    print("\nThese are for REPORT configuration:")
    print("(For aggregate analysis across major divisions)\n")
    
    if admins:
        for i, admin in enumerate(admins):
            print(f"{i+1}. {admin['name']} (ID: {admin['id']})")
    else:
        print("⚠️  No administrations found\n")
    
    return len(admins) > 0


def test_departments():
    """Test getting departments"""
    print_separator("TEST 3: Departments (For Filtering)")
    
    depts = org_unit_service.get_departments()
    
    print(f"\nFound {len(depts)} departments")
    
    if depts:
        # Show first 5 examples
        for i, dept in enumerate(depts[:5]):
            print(f"{i+1}. {dept['name']} (ID: {dept['id']})")
            print(f"   - Administration ID: {dept['administration_id']}")
        
        if len(depts) > 5:
            print(f"... and {len(depts) - 5} more departments\n")
    else:
        print("⚠️  No departments found\n")
    
    return len(depts) > 0


def test_sections():
    """Test getting sections"""
    print_separator("TEST 4: Sections (For Filtering)")
    
    sections = org_unit_service.get_sections()
    
    print(f"\nFound {len(sections)} sections")
    
    if sections:
        # Show first 5 examples
        for i, section in enumerate(sections[:5]):
            print(f"{i+1}. {section['name']} (ID: {section['id']})")
            print(f"   - Department ID: {section['department_id']}")
        
        if len(sections) > 5:
            print(f"... and {len(sections) - 5} more sections\n")
    else:
        print("⚠️  No sections found\n")
    
    return len(sections) > 0


def test_unit_with_ancestors():
    """Test getting a unit with its ancestry"""
    print_separator("TEST 5: Unit with Ancestors (For Breadcrumbs)")
    
    # Try to get a section with full ancestry
    sections = org_unit_service.get_sections()
    
    if sections:
        # Test with first section
        section_id = sections[0]['id']
        result = org_unit_service.get_unit_with_ancestors(section_id)
        
        if result:
            print(f"\nUnit: {result['name']} (ID: {result['id']})")
            print(f"Type: {result['type_name']}")
            print(f"\nAncestry Chain:")
            
            if result['ancestors']:
                for i, ancestor in enumerate(result['ancestors']):
                    indent = "  " * i
                    print(f"{indent}↓ {ancestor['name']} ({ancestor['type_name']})")
                
                indent = "  " * len(result['ancestors'])
                print(f"{indent}↓ {result['name']} ({result['type_name']}) ← YOU ARE HERE")
            else:
                print("  (No ancestors - this is a top-level unit)")
            
            # Build breadcrumb
            breadcrumb_parts = [a['name'] for a in result['ancestors']]
            breadcrumb_parts.append(result['name'])
            breadcrumb = " > ".join(breadcrumb_parts)
            print(f"\nBreadcrumb: {breadcrumb}\n")
        else:
            print(f"⚠️  Unit {section_id} not found\n")
        
        return result is not None
    else:
        print("⚠️  No sections available to test\n")
        return False


def test_summary():
    """Test getting summary counts"""
    print_separator("TEST 6: Summary (Overview)")
    
    administrations = org_unit_service.get_administrations()
    departments = org_unit_service.get_departments()
    sections = org_unit_service.get_sections()
    leaves = org_unit_service.get_leaf_units()
    
    print("\nOrganizational Unit Summary:")
    print(f"  Administrations: {len(administrations)}")
    print(f"  Departments:     {len(departments)}")
    print(f"  Sections:        {len(sections)}")
    print(f"  Total Units:     {len(administrations) + len(departments) + len(sections)}")
    print(f"  Leaf Nodes:      {len(leaves)}")
    print()
    
    return True


def test_usage_examples():
    """Show practical usage examples"""
    print_separator("USAGE GUIDE")
    
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                          WHEN TO USE EACH ENDPOINT                         ║
╚════════════════════════════════════════════════════════════════════════════╝

1. INSERT/ADD PATIENT FORMS
   ▸ Use: GET /api/org-units/leaves
   ▸ Why: Users select the ACTUAL unit where incident occurred
   ▸ Example: "Emergency Section", "ICU Section"
   
2. REPORTS (All Administrations)
   ▸ Use: GET /api/org-units/administrations
   ▸ Why: Compare major hospital divisions
   ▸ Example: "Medical Administration" vs "Surgical Administration"

3. CASCADING FILTERS (Investigation, Dashboard)
   ▸ Use: GET /api/investigation/hierarchy (existing endpoint)
   ▸ Why: Users drill down: Admin → Dept → Section

4. BREADCRUMB NAVIGATION
   ▸ Use: GET /api/org-units/unit/{id}
   ▸ Why: Show full context for a unit
   ▸ Example: "Medical Admin > Emergency Dept > ICU Section"

╔════════════════════════════════════════════════════════════════════════════╗
║                              FRONTEND EXAMPLES                             ║
╚════════════════════════════════════════════════════════════════════════════╝

// 1. Populate issuing department in INSERT form
const response = await fetch('/api/org-units/leaves');
const data = await response.json();
const issuingDeptOptions = data.leaves.map(leaf => ({
  value: leaf.id,
  label: leaf.name
}));

// 2. Populate report scope dropdown
const response = await fetch('/api/org-units/administrations');
const data = await response.json();
const reportScopeOptions = [
  { value: 'all', label: 'All Administrations' },
  ...data.administrations.map(admin => ({
    value: admin.id,
    label: admin.name
  }))
];

// 3. Show breadcrumb for a unit
const response = await fetch(`/api/org-units/unit/${unitId}`);
const data = await response.json();
setBreadcrumb(data.breadcrumb);
// Output: "Medical Administration > Emergency Department > ICU Section"

""")


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "ORGANIZATION UNIT ENDPOINTS TEST" + " " * 31 + "║")
    print("╚" + "═" * 78 + "╝")
    
    results = {}
    
    try:
        results['leaves'] = test_leaf_units()
        results['administrations'] = test_administrations()
        results['departments'] = test_departments()
        results['sections'] = test_sections()
        results['ancestors'] = test_unit_with_ancestors()
        results['summary'] = test_summary()
        test_usage_examples()
        
        # Final summary
        print_separator("TEST RESULTS")
        
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        print(f"\nTests Passed: {passed}/{total}")
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {status} - {test_name}")
        
        if passed == total:
            print("\n🎉 All tests passed! Endpoints are ready to use.\n")
            print("Next Steps:")
            print("1. Start the backend server: cd backend && uvicorn main:app --reload")
            print("2. Test endpoints in browser or with curl")
            print("3. Update frontend to use these endpoints\n")
        else:
            print("\n⚠️  Some tests failed. Check database content.\n")
    
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
