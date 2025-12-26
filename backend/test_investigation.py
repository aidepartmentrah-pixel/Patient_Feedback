"""
Quick test script for investigation endpoints.
Run this to test the investigation service directly.
"""

import sys
sys.path.insert(0, ".")

from api.services.investigation_service import (
    get_investigation_tree,
    get_available_seasons,
    get_organizational_hierarchy
)

def test_seasons():
    """Test available seasons endpoint."""
    print("\n" + "="*80)
    print("Testing Available Seasons")
    print("="*80)
    
    try:
        result = get_available_seasons()
        print(f"✓ Found {len(result['seasons'])} seasons")
        print(f"  Current season: {result['current_season']}")
        
        for season in result['seasons'][:3]:  # Show first 3
            print(f"  - {season['season_id']}: {season['season_label']}")
        
        return result['seasons'][0]['season_id'] if result['seasons'] else None
    
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def test_hierarchy():
    """Test organizational hierarchy endpoint."""
    print("\n" + "="*80)
    print("Testing Organizational Hierarchy")
    print("="*80)
    
    try:
        result = get_organizational_hierarchy()
        print(f"✓ Found {len(result['administrations'])} administrations")
        print(f"✓ Found {len(result['departments'])} departments")
        print(f"✓ Found {len(result['sections'])} sections")
        
        # Show first admin
        if result['administrations']:
            admin = result['administrations'][0]
            print(f"\n  Sample Administration:")
            print(f"    ID: {admin['id']}")
            print(f"    Name: {admin['name_en']}")
        
        # Show first dept
        if result['departments']:
            dept = result['departments'][0]
            print(f"\n  Sample Department:")
            print(f"    ID: {dept['id']}")
            print(f"    Name: {dept['name_en']}")
            print(f"    Admin ID: {dept['administration_id']}")
        
        return result
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_tree(season_id, admin_id=None, dept_id=None, section_id=None):
    """Test investigation tree endpoint."""
    scope_desc = "Hospital-Wide"
    if section_id:
        scope_desc = f"Section {section_id}"
    elif dept_id:
        scope_desc = f"Department {dept_id}"
    elif admin_id:
        scope_desc = f"Administration {admin_id}"
    
    print("\n" + "="*80)
    print(f"Testing Investigation Tree - {scope_desc}")
    print("="*80)
    
    try:
        result = get_investigation_tree(
            season=str(season_id),
            tree_type="incident_count",
            administration_id=admin_id,
            department_id=dept_id,
            section_id=section_id,
        )
        
        print(f"✓ Season: {result['season_label']}")
        print(f"✓ Scope Level: {result['scope']['level']}")
        print(f"✓ Tree Nodes: {len(result['tree'])}")
        print(f"✓ Total Incidents: {result['summary']['total_incidents']}")
        
        # Show first node
        if result['tree']:
            node = result['tree'][0]
            print(f"\n  First Node:")
            print(f"    ID: {node['node_id']}")
            print(f"    Name: {node['node_name']}")
            print(f"    Type: {node['node_type']}")
            print(f"    Value: {node['value']}")
            print(f"    Children: {len(node['children'])}")
        
        return True
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("INVESTIGATION SERVICE TEST SUITE")
    print("="*80)
    
    # Test 1: Get available seasons
    season_id = test_seasons()
    if not season_id:
        print("\n✗ Cannot proceed without seasons data")
        return
    
    # Test 2: Get organizational hierarchy
    hierarchy = test_hierarchy()
    if not hierarchy:
        print("\n✗ Cannot proceed without hierarchy data")
        return
    
    # Test 3: Hospital-wide tree
    print("\n\n" + "="*80)
    print("TEST: Hospital-Wide Incident Count")
    print("="*80)
    test_tree(season_id)
    
    # Test 4: Administration-specific tree
    if hierarchy['administrations']:
        admin_id = hierarchy['administrations'][0]['id']
        print("\n\n" + "="*80)
        print(f"TEST: Administration {admin_id} Incident Count")
        print("="*80)
        test_tree(season_id, admin_id=admin_id)
    
    # Test 5: Department-specific tree
    if hierarchy['departments']:
        dept_id = hierarchy['departments'][0]['id']
        print("\n\n" + "="*80)
        print(f"TEST: Department {dept_id} Incident Count")
        print("="*80)
        test_tree(season_id, dept_id=dept_id)
    
    # Test 6: Section-specific tree
    if hierarchy['sections']:
        section_id = hierarchy['sections'][0]['id']
        print("\n\n" + "="*80)
        print(f"TEST: Section {section_id} Incident Count")
        print("="*80)
        test_tree(season_id, section_id=section_id)
    
    print("\n" + "="*80)
    print("TEST SUITE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
