"""
Quick Test: Show Real Data from All 4 Endpoints

Run this to see actual data from your database for each endpoint type.
"""

import requests
import json


BASE_URL = "http://localhost:8000"


def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_leaves():
    """Test endpoint #1: Leaf nodes for INSERT forms"""
    print_section("1. LEAF NODES (for INSERT forms)")
    
    url = f"{BASE_URL}/api/org-units/leaves"
    print(f"\nEndpoint: GET {url}")
    
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        
        print(f"\n✅ SUCCESS: Returns {data['count']} leaf units")
        print("\nThese are the units for INSERT/ADD PATIENT forms:")
        print("(Users select where incident ACTUALLY happened)\n")
        
        # Show first 5
        for i, leaf in enumerate(data['leaves'][:5]):
            print(f"{i+1}. {leaf['name']}")
            print(f"   ID: {leaf['id']}")
            print(f"   Type: {leaf['type_name']}")
            if leaf['parent_name']:
                print(f"   Parent: {leaf['parent_name']}")
            print()
        
        if data['count'] > 5:
            print(f"... and {data['count'] - 5} more\n")
        
        return True
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


def test_administrations():
    """Test endpoint #2 & #3: Administrations for REPORTS and selection"""
    print_section("2 & 3. ALL ADMINISTRATIONS (for REPORTS & selection)")
    
    url = f"{BASE_URL}/api/org-units/administrations"
    print(f"\nEndpoint: GET {url}")
    
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        
        print(f"\n✅ SUCCESS: Returns {data['count']} administrations")
        print("\nUse this for:")
        print("  - Report Configuration ('All Administrations' option)")
        print("  - User Assignment (select administration)")
        print("  - High-level filtering\n")
        
        print("All Administrations:")
        for i, admin in enumerate(data['administrations']):
            print(f"{i+1}. {admin['name']} (ID: {admin['id']})")
        
        print()
        return True
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


def test_departments():
    """Test endpoint #4: Departments"""
    print_section("4. SELECT DEPARTMENTS (only departments)")
    
    url = f"{BASE_URL}/api/org-units/departments"
    print(f"\nEndpoint: GET {url}")
    
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        
        print(f"\n✅ SUCCESS: Returns {data['count']} departments")
        print("\nUse this for:")
        print("  - Department filters")
        print("  - User assignment to department")
        print("  - Department-level reports\n")
        
        print("First 10 Departments:")
        for i, dept in enumerate(data['departments'][:10]):
            print(f"{i+1}. {dept['name']} (ID: {dept['id']}, Admin: {dept['administration_id']})")
        
        if data['count'] > 10:
            print(f"\n... and {data['count'] - 10} more\n")
        
        return True
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


def test_summary():
    """Test summary endpoint"""
    print_section("BONUS: Summary Stats")
    
    url = f"{BASE_URL}/api/org-units/summary"
    print(f"\nEndpoint: GET {url}")
    
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        
        print("\n✅ SUCCESS")
        print("\nYour Database Contains:")
        print(f"  Administrations: {data['administrations']}")
        print(f"  Departments:     {data['departments']}")
        print(f"  Sections:        {data['sections']}")
        print(f"  Total Units:     {data['total']}")
        print(f"  Leaf Nodes:      {data['leaves']}")
        print()
        
        return True
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


def show_usage_examples():
    """Show frontend usage examples"""
    print_section("FRONTEND USAGE EXAMPLES")
    
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                          COPY-PASTE FOR YOUR UI                            ║
╚════════════════════════════════════════════════════════════════════════════╝

1. ADD PATIENT FORM (Issuing Department):
   ────────────────────────────────────────
   const response = await fetch('http://localhost:8000/api/org-units/leaves');
   const data = await response.json();
   
   issuingDeptOptions.value = data.leaves.map(leaf => ({
     value: leaf.id,
     label: leaf.name
   }));


2. REPORT CONFIGURATION (Report Scope):
   ────────────────────────────────────────
   const response = await fetch('http://localhost:8000/api/org-units/administrations');
   const data = await response.json();
   
   reportScopeOptions.value = [
     { value: 'all', label: 'All Administrations' },
     ...data.administrations.map(admin => ({
       value: admin.id,
       label: admin.name
     }))
   ];


3. USER ASSIGNMENT (Administration):
   ────────────────────────────────────────
   const response = await fetch('http://localhost:8000/api/org-units/administrations');
   const data = await response.json();
   
   adminOptions.value = data.administrations.map(admin => ({
     value: admin.id,
     text: admin.name
   }));


4. DEPARTMENT FILTER (Multi-select):
   ────────────────────────────────────────
   const response = await fetch('http://localhost:8000/api/org-units/departments');
   const data = await response.json();
   
   departmentFilters.value = data.departments.map(dept => ({
     id: dept.id,
     name: dept.name,
     checked: false
   }));

╔════════════════════════════════════════════════════════════════════════════╗
║                            TEST IN BROWSER                                 ║
╚════════════════════════════════════════════════════════════════════════════╝

Open these URLs:

1. Leaf Nodes:       http://localhost:8000/api/org-units/leaves
2. Administrations:  http://localhost:8000/api/org-units/administrations
3. Departments:      http://localhost:8000/api/org-units/departments
4. Summary:          http://localhost:8000/api/org-units/summary
5. API Docs:         http://localhost:8000/docs

""")


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 22 + "YOUR 4 ENDPOINTS - LIVE TEST" + " " * 28 + "║")
    print("╚" + "═" * 78 + "╝")
    
    print(f"\nBase URL: {BASE_URL}")
    print("Testing connection...")
    
    results = {}
    
    results['leaves'] = test_leaves()
    results['administrations'] = test_administrations()
    results['departments'] = test_departments()
    results['summary'] = test_summary()
    
    show_usage_examples()
    
    # Final summary
    print("=" * 80)
    print("  TEST RESULTS")
    print("=" * 80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nEndpoints Working: {passed}/{total}\n")
    
    for name, result in results.items():
        status = "✅ WORKING" if result else "❌ FAILED"
        print(f"  {status} - {name}")
    
    if passed == total:
        print("\n" + "🎉" * 40)
        print("\n  ALL ENDPOINTS WORKING!")
        print("  Your backend is ready for frontend integration!\n")
        print("🎉" * 40 + "\n")
    else:
        print("\n⚠️  Some endpoints failed. Check if server is running.\n")
        print("Start server: cd backend && uvicorn main:app --reload\n")


if __name__ == "__main__":
    main()
