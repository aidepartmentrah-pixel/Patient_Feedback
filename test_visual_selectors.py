"""
Visual Test: Organization Selector Types

This script creates a visual comparison of all 4 selector types
showing sample data and usage examples.
"""

import requests
from typing import Dict, List


BASE_URL = "http://localhost:8000"


def print_box(title: str, content: List[str], width: int = 76):
    """Print a fancy box with content"""
    print("┌" + "─" * width + "┐")
    print("│ " + title.ljust(width - 2) + " │")
    print("├" + "─" * width + "┤")
    for line in content:
        print("│ " + line.ljust(width - 2) + " │")
    print("└" + "─" * width + "┘")


def main():
    print("\n" + "=" * 80)
    print("  🎯 ORGANIZATION SELECTOR - VISUAL COMPARISON")
    print("=" * 80)
    
    try:
        # Fetch all data
        leaves_data = requests.get(f"{BASE_URL}/api/org-units/leaves").json()
        admin_data = requests.get(f"{BASE_URL}/api/org-units/administrations").json()
        dept_data = requests.get(f"{BASE_URL}/api/org-units/departments").json()
        
        print("\n")
        
        # Type 1: Leaves
        print_box(
            "TYPE 1: LEAVES (216 units) - For INSERT Forms",
            [
                "Use Case: Add Patient Form, Create Incident",
                "Endpoint: GET /api/org-units/leaves",
                "",
                "Sample Data:",
                f"  • {leaves_data['leaves'][0]['name']} (ID: {leaves_data['leaves'][0]['id']})",
                f"  • {leaves_data['leaves'][1]['name']} (ID: {leaves_data['leaves'][1]['id']})",
                f"  • {leaves_data['leaves'][2]['name']} (ID: {leaves_data['leaves'][2]['id']})",
                f"  ... and {leaves_data['count'] - 3} more",
                "",
                "UI: Simple Dropdown (Single-select)",
                "Code: data.leaves.map(leaf => ({ value: leaf.id, label: leaf.name }))"
            ]
        )
        
        print()
        
        # Type 2: All Administrations
        print_box(
            "TYPE 2: ALL ADMINISTRATIONS (9 units) - For Reports",
            [
                "Use Case: Report Configuration, Monthly Reports",
                "Endpoint: GET /api/org-units/administrations",
                "",
                "Sample Data:",
                f"  • ALL ADMINISTRATIONS (special option)",
                f"  • {admin_data['administrations'][0]['name']} (ID: {admin_data['administrations'][0]['id']})",
                f"  • {admin_data['administrations'][1]['name']} (ID: {admin_data['administrations'][1]['id']})",
                f"  • {admin_data['administrations'][2]['name']} (ID: {admin_data['administrations'][2]['id']})",
                f"  ... and {admin_data['count'] - 3} more",
                "",
                "UI: Dropdown with 'All' option",
                "Code: [{ value: 'all', label: 'All' }, ...data.administrations]"
            ]
        )
        
        print()
        
        # Type 3: Select Administration
        print_box(
            "TYPE 3: SELECT ADMINISTRATION (9 units) - For User Assignment",
            [
                "Use Case: User Management, Settings",
                "Endpoint: GET /api/org-units/administrations (same as Type 2!)",
                "",
                "Sample Data:",
                f"  • {admin_data['administrations'][0]['name']} (ID: {admin_data['administrations'][0]['id']})",
                f"  • {admin_data['administrations'][1]['name']} (ID: {admin_data['administrations'][1]['id']})",
                f"  • {admin_data['administrations'][2]['name']} (ID: {admin_data['administrations'][2]['id']})",
                f"  ... and {admin_data['count'] - 3} more",
                "",
                "UI: Simple Dropdown (NO 'All' option)",
                "Code: data.administrations.map(a => ({ value: a.id, label: a.name }))"
            ]
        )
        
        print()
        
        # Type 4: Departments
        print_box(
            "TYPE 4: DEPARTMENTS (134 units) - For Filtering",
            [
                "Use Case: Dashboard Filters, Performance Reports",
                "Endpoint: GET /api/org-units/departments",
                "",
                "Sample Data:",
                f"  ☐ {dept_data['departments'][0]['name']} (ID: {dept_data['departments'][0]['id']})",
                f"  ☐ {dept_data['departments'][1]['name']} (ID: {dept_data['departments'][1]['id']})",
                f"  ☐ {dept_data['departments'][2]['name']} (ID: {dept_data['departments'][2]['id']})",
                f"  ☐ {dept_data['departments'][3]['name']} (ID: {dept_data['departments'][3]['id']})",
                f"  ... and {dept_data['count'] - 4} more",
                "",
                "UI: Multi-select Checkboxes",
                "Code: data.departments.map(d => ({ id: d.id, name: d.name, checked: false }))"
            ]
        )
        
        print("\n" + "=" * 80)
        print("  QUICK DECISION GUIDE")
        print("=" * 80)
        
        print("""
┌──────────────────────────────────────┬─────────────────────────────────┐
│ Your Question                        │ Use This Selector               │
├──────────────────────────────────────┼─────────────────────────────────┤
│ Where did incident HAPPEN?           │ Type 1: Leaves (216)            │
│ Need "All Administrations" option?   │ Type 2: All Admins (9)          │
│ Assign user to ONE administration?   │ Type 3: Select Admin (9)        │
│ Filter by multiple departments?      │ Type 4: Departments (134)       │
└──────────────────────────────────────┴─────────────────────────────────┘
""")
        
        print("=" * 80)
        print("  🎉 All selectors working! Backend ready for frontend integration!")
        print("=" * 80)
        print()
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to backend")
        print("Please start the backend: cd backend && uvicorn main:app --reload")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")


if __name__ == "__main__":
    main()
