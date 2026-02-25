"""
Test Organization Unit Endpoints via HTTP

Tests the actual API endpoints to ensure they work correctly
when the backend server is running.

Run this after starting the server:
  cd backend && uvicorn main:app --reload
"""

import requests
import json
from typing import Dict, Any


BASE_URL = "http://localhost:8000"


def print_separator(title: str):
    """Print a visual separator"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_endpoint(endpoint: str, method: str = "GET", description: str = "") -> Dict[Any, Any]:
    """Test an API endpoint"""
    url = f"{BASE_URL}{endpoint}"
    
    print(f"\n{description}")
    print(f"▸ Endpoint: {method} {endpoint}")
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=5)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        print(f"▸ Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Show summary based on endpoint
            if "/leaves" in endpoint:
                count = data.get("count", 0)
                print(f"▸ Result: {count} leaf units")
                if data.get("leaves"):
                    print(f"▸ First: {data['leaves'][0]['name']} (ID: {data['leaves'][0]['id']})")
            
            elif "/administrations" in endpoint:
                count = data.get("count", 0)
                print(f"▸ Result: {count} administrations")
                if data.get("administrations"):
                    names = [admin['name'] for admin in data['administrations'][:3]]
                    print(f"▸ Examples: {', '.join(names)}")
            
            elif "/departments" in endpoint:
                count = data.get("count", 0)
                print(f"▸ Result: {count} departments")
            
            elif "/sections" in endpoint:
                count = data.get("count", 0)
                print(f"▸ Result: {count} sections")
            
            elif "/summary" in endpoint:
                print(f"▸ Result:")
                print(f"   - Administrations: {data.get('administrations', 0)}")
                print(f"   - Departments: {data.get('departments', 0)}")
                print(f"   - Sections: {data.get('sections', 0)}")
                print(f"   - Leaf Nodes: {data.get('leaves', 0)}")
                print(f"   - Total: {data.get('total', 0)}")
            
            elif "/unit/" in endpoint:
                print(f"▸ Result: {data.get('name', 'Unknown')}")
                if data.get('breadcrumb'):
                    print(f"▸ Breadcrumb: {data['breadcrumb']}")
            
            print("✅ SUCCESS")
            return data
        else:
            print(f"❌ FAILED: {response.text}")
            return {}
    
    except requests.exceptions.ConnectionError:
        print("❌ FAILED: Could not connect to server")
        print("   Make sure the backend is running: cd backend && uvicorn main:app --reload")
        return {}
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return {}


def main():
    """Run all API endpoint tests"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 18 + "ORGANIZATION UNIT API TEST" + " " * 34 + "║")
    print("╚" + "═" * 78 + "╝")
    
    print(f"\nBase URL: {BASE_URL}")
    print("Expected: Backend server running on http://localhost:8000")
    
    # Test 1: Leaf units (for insert forms)
    print_separator("TEST 1: Leaf Units (For INSERT Forms)")
    test_endpoint(
        "/api/org-units/leaves",
        description="Get all leaf units (sections with no children)"
    )
    
    # Test 2: Administrations (for reports)
    print_separator("TEST 2: Administrations (For REPORTS)")
    test_endpoint(
        "/api/org-units/administrations",
        description="Get all top-level administrations"
    )
    
    # Test 3: Departments
    print_separator("TEST 3: Departments")
    test_endpoint(
        "/api/org-units/departments",
        description="Get all departments"
    )
    
    # Test 4: Sections
    print_separator("TEST 4: Sections")
    test_endpoint(
        "/api/org-units/sections",
        description="Get all sections"
    )
    
    # Test 5: Summary
    print_separator("TEST 5: Summary")
    test_endpoint(
        "/api/org-units/summary",
        description="Get overview of all organizational units"
    )
    
    # Test 6: Unit with ancestors (need a valid unit ID)
    print_separator("TEST 6: Unit with Ancestors")
    
    # First get a section to use as example
    leaves_data = test_endpoint(
        "/api/org-units/leaves",
        description="Get leaves to find a unit ID"
    )
    
    if leaves_data.get("leaves"):
        unit_id = leaves_data["leaves"][0]["id"]
        test_endpoint(
            f"/api/org-units/unit/{unit_id}",
            description=f"Get unit {unit_id} with full ancestry chain"
        )
    
    # Final summary
    print_separator("COMPLETE")
    print("\n✅ API endpoint tests complete!")
    print("\nYou can now:")
    print("1. View API docs: http://localhost:8000/docs")
    print("2. Test endpoints in browser")
    print("3. Update frontend to use these endpoints")
    print()


if __name__ == "__main__":
    main()
