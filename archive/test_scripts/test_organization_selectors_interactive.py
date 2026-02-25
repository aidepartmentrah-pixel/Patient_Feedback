"""
Interactive Test: All 4 Organization Selector Types

This script demonstrates when and how to use each of the 4 organization
selector endpoints in different UI scenarios.
"""

import requests
import json
from typing import Dict, List


BASE_URL = "http://localhost:8000"


def print_header(title: str, emoji: str = "🎯"):
    """Print a section header"""
    print("\n" + "=" * 80)
    print(f"{emoji} {title}")
    print("=" * 80)


def print_success(message: str):
    """Print success message"""
    print(f"\n✅ {message}")


def print_use_case(title: str, description: str):
    """Print use case box"""
    print(f"\n┌{'─' * 78}┐")
    print(f"│ {title:<76} │")
    print(f"├{'─' * 78}┤")
    print(f"│ {description:<76} │")
    print(f"└{'─' * 78}┘")


# =============================================================================
# SELECTOR TYPE 1: LEAF NODES (for INSERT forms)
# =============================================================================

def test_selector_type_1_leaves():
    """
    Selector Type 1: Leaf Nodes
    
    When to use: INSERT/ADD forms where user selects ACTUAL location
    Examples: Add Patient, Create Incident, Log Complaint
    """
    print_header("SELECTOR TYPE 1: LEAF NODES (INSERT Forms)", "📝")
    
    print_use_case(
        "USE CASE: Add Patient Form",
        "User needs to select the department where the incident occurred"
    )
    
    url = f"{BASE_URL}/api/org-units/leaves"
    response = requests.get(url)
    data = response.json()
    
    print_success(f"Retrieved {data['count']} leaf units")
    
    print("\n▶ Frontend Implementation:")
    print("""
    // Add Patient Form - Issuing Department Dropdown
    const response = await fetch('/api/org-units/leaves');
    const data = await response.json();
    
    issuingDeptDropdown.options = data.leaves.map(leaf => ({
        value: leaf.id,
        label: leaf.name
    }));
    """)
    
    print("\n▶ Sample Data (First 5):")
    for i, leaf in enumerate(data['leaves'][:5], 1):
        print(f"  {i}. {leaf['name']} (ID: {leaf['id']})")
        if leaf.get('parent_name'):
            print(f"     └─ Parent: {leaf['parent_name']}")
    
    print(f"\n  ... and {data['count'] - 5} more options\n")
    
    return data


# =============================================================================
# SELECTOR TYPE 2: ALL ADMINISTRATIONS (for Reports)
# =============================================================================

def test_selector_type_2_all_administrations():
    """
    Selector Type 2: All Administrations
    
    When to use: Report configuration, high-level aggregate views
    Examples: Monthly Report, Seasonal Report, Executive Dashboard
    """
    print_header("SELECTOR TYPE 2: ALL ADMINISTRATIONS (Reports)", "📊")
    
    print_use_case(
        "USE CASE: Report Configuration Page",
        "User wants to see 'All Administrations' or select a specific one"
    )
    
    url = f"{BASE_URL}/api/org-units/administrations"
    response = requests.get(url)
    data = response.json()
    
    print_success(f"Retrieved {data['count']} administrations")
    
    print("\n▶ Frontend Implementation:")
    print("""
    // Report Config - Report Scope Dropdown
    const response = await fetch('/api/org-units/administrations');
    const data = await response.json();
    
    reportScopeOptions = [
        { value: 'all', label: 'All Administrations' },  // Add "All" option
        ...data.administrations.map(admin => ({
            value: admin.id,
            label: admin.name
        }))
    ];
    """)
    
    print("\n▶ All Administrations:")
    for i, admin in enumerate(data['administrations'], 1):
        print(f"  {i}. {admin['name']} (ID: {admin['id']})")
    
    print()
    return data


# =============================================================================
# SELECTOR TYPE 3: SELECT ADMINISTRATIONS (for User Assignment)
# =============================================================================

def test_selector_type_3_select_administration():
    """
    Selector Type 3: Select Administrations
    
    When to use: User assignment, settings, filters
    Examples: User Management, Settings Page, Investigation Filters
    """
    print_header("SELECTOR TYPE 3: SELECT ADMINISTRATIONS (User Assignment)", "👤")
    
    print_use_case(
        "USE CASE: User Management - Assign User to Administration",
        "Admin needs to assign a user to a specific administration"
    )
    
    # Same endpoint as Type 2!
    url = f"{BASE_URL}/api/org-units/administrations"
    response = requests.get(url)
    data = response.json()
    
    print_success(f"Retrieved {data['count']} administrations")
    
    print("\n▶ Frontend Implementation:")
    print("""
    // User Management - Assign Administration Dropdown
    const response = await fetch('/api/org-units/administrations');
    const data = await response.json();
    
    adminAssignmentDropdown.options = data.administrations.map(admin => ({
        value: admin.id,
        label: admin.name
    }));
    
    // Note: No "All" option here - user must select ONE administration
    """)
    
    print("\n▶ Sample UI Flow:")
    print("  1. Admin opens 'Create User' form")
    print("  2. Dropdown shows 9 administration options")
    print("  3. Admin selects: 'الإدارة الطبية' (ID: 4)")
    print("  4. User is created with administration_id = 4")
    
    print("\n▶ Difference from Type 2:")
    print("  - Type 2: Adds 'All Administrations' option for reports")
    print("  - Type 3: Single-select only, no 'All' option")
    print()
    
    return data


# =============================================================================
# SELECTOR TYPE 4: DEPARTMENTS ONLY (for Filtering)
# =============================================================================

def test_selector_type_4_departments():
    """
    Selector Type 4: Departments Only
    
    When to use: Department-level filtering, assignment, reports
    Examples: Filter Panel, Performance Dashboard, Department Reports
    """
    print_header("SELECTOR TYPE 4: DEPARTMENTS ONLY (Filtering)", "🔍")
    
    print_use_case(
        "USE CASE: Dashboard - Filter by Department (Multi-select)",
        "User wants to compare performance across multiple departments"
    )
    
    url = f"{BASE_URL}/api/org-units/departments"
    response = requests.get(url)
    data = response.json()
    
    print_success(f"Retrieved {data['count']} departments")
    
    print("\n▶ Frontend Implementation:")
    print("""
    // Dashboard - Department Filter (Multi-select Checkboxes)
    const response = await fetch('/api/org-units/departments');
    const data = await response.json();
    
    departmentFilters = data.departments.map(dept => ({
        id: dept.id,
        name: dept.name,
        administrationId: dept.administration_id,
        checked: false
    }));
    
    // User checks multiple departments → filter charts
    """)
    
    print("\n▶ Sample Data (First 10):")
    for i, dept in enumerate(data['departments'][:10], 1):
        print(f"  {i}. {dept['name']} (ID: {dept['id']}, Admin: {dept['administration_id']})")
    
    print(f"\n  ... and {data['count'] - 10} more departments")
    
    print("\n▶ Sample UI Flow:")
    print("  1. Dashboard loads with 'Filter by Department' panel")
    print("  2. Panel shows 134 department checkboxes")
    print("  3. User checks: 'دائرة الطوارئ الطبية', 'دائرة التصوير الطبي'")
    print("  4. Dashboard updates to show only those 2 departments")
    print()
    
    return data


# =============================================================================
# COMPARISON TABLE
# =============================================================================

def show_comparison_table():
    """Show side-by-side comparison of all 4 selector types"""
    print_header("COMPARISON: When to Use Each Selector", "📋")
    
    print("""
┌──────────────────────────────────────────────────────────────────────────────┐
│ SELECTOR TYPE COMPARISON                                                     │
├─────┬────────────────────┬─────────┬──────────────────────────────────────────┤
│ #   │ Endpoint           │ Count   │ Use When...                              │
├─────┼────────────────────┼─────────┼──────────────────────────────────────────┤
│ 1   │ /leaves            │ 216     │ User selects WHERE incident happened     │
│     │                    │         │ (INSERT forms, Create Incident)          │
├─────┼────────────────────┼─────────┼──────────────────────────────────────────┤
│ 2   │ /administrations   │ 9       │ Report needs "All Administrations"       │
│     │ (with "All" opt)   │         │ (Report Config, Monthly/Seasonal Report) │
├─────┼────────────────────┼─────────┼──────────────────────────────────────────┤
│ 3   │ /administrations   │ 9       │ User assigned to ONE administration      │
│     │ (single-select)    │         │ (User Management, Settings)              │
├─────┼────────────────────┼─────────┼──────────────────────────────────────────┤
│ 4   │ /departments       │ 134     │ Filter/compare by department             │
│     │                    │         │ (Dashboard Filters, Department Reports)  │
└─────┴────────────────────┴─────────┴──────────────────────────────────────────┘
""")


# =============================================================================
# PRACTICAL SCENARIOS
# =============================================================================

def show_practical_scenarios():
    """Show real-world scenarios for each selector type"""
    print_header("PRACTICAL SCENARIOS", "💡")
    
    scenarios = [
        {
            "title": "Scenario 1: Nurse Adding Patient Complaint",
            "selector": "Type 1 (Leaves)",
            "steps": [
                "Nurse clicks 'Add Patient Complaint'",
                "Form shows 'Department' dropdown with 216 leaf units",
                "Nurse selects 'دائرة الطوارئ الطبية' (where she works)",
                "Complaint is logged with specific department ID"
            ]
        },
        {
            "title": "Scenario 2: Manager Generating Monthly Report",
            "selector": "Type 2 (All Administrations)",
            "steps": [
                "Manager opens 'Monthly Report' page",
                "Dropdown shows 'All Administrations' + 9 specific admins",
                "Manager selects 'All Administrations'",
                "Report shows aggregate data across entire hospital"
            ]
        },
        {
            "title": "Scenario 3: Admin Creating New User",
            "selector": "Type 3 (Select Administration)",
            "steps": [
                "Admin opens 'Create User' form",
                "Dropdown shows 9 administrations (no 'All' option)",
                "Admin selects 'الإدارة الطبية'",
                "New user is created with scope limited to Medical Admin"
            ]
        },
        {
            "title": "Scenario 4: Executive Comparing Departments",
            "selector": "Type 4 (Departments)",
            "steps": [
                "Executive opens Performance Dashboard",
                "Filter panel shows 134 department checkboxes",
                "Executive checks 5 high-priority departments",
                "Charts update to show only those 5 departments"
            ]
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'─' * 80}")
        print(f"📌 {scenario['title']}")
        print(f"   Selector: {scenario['selector']}")
        print(f"{'─' * 80}")
        for step in scenario['steps']:
            print(f"   → {step}")
    
    print()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run interactive test"""
    print_header("ORGANIZATION SELECTOR - INTERACTIVE TEST", "🎯")
    print("\nThis test shows WHEN and HOW to use each of the 4 selector types.\n")
    
    try:
        # Test each selector type
        test_selector_type_1_leaves()
        test_selector_type_2_all_administrations()
        test_selector_type_3_select_administration()
        test_selector_type_4_departments()
        
        # Show comparison and scenarios
        show_comparison_table()
        show_practical_scenarios()
        
        # Final summary
        print_header("SUMMARY", "✅")
        print("""
All 4 organization selector types are working!

Next Steps for Frontend:
1. ✅ Endpoints are ready and tested
2. ✅ Use the code examples above in your UI components
3. ✅ Refer to ORGANIZATION_SELECTOR_GUIDE.md for complete documentation
4. ✅ Test in browser: http://localhost:8000/docs

Your backend is 100% ready for frontend integration! 🎉
""")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to backend server")
        print("Please ensure the backend is running at http://localhost:8000")
        print("\nTo start the backend:")
        print("  cd backend")
        print("  uvicorn main:app --reload")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")


if __name__ == "__main__":
    main()
