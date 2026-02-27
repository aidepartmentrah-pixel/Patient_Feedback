"""
Central Org Tree Service

Single source of truth for traversing the AdminsrationUnit hierarchy.
Loads the full org tree into memory and provides efficient traversal functions.

Design:
- Load full tree once using admin_units.get_admin_unit_tree()
- Normalize into in-memory maps for fast lookups
- Provide minimal, necessary API: get_full_tree, get_descendants, get_ancestors, is_ancestor
- All traversal is done in-memory using DFS/BFS
- Deterministic, side-effect free, safe to call from any service
"""

from typing import Optional
from ..db_layer import admin_units


# =========================================================
# IN-MEMORY TREE STATE
# =========================================================

_tree_cache: Optional[dict] = None


def _row_to_dict(row) -> dict:
    """
    Normalize pyodbc row to dict.
    """
    if isinstance(row, dict):
        return row
    
    return {
        "UniqueID": row.UniqueID,
        "ParentID": row.ParentID,
        "Type": row.Type,
        "Name": row.Name,
    }


def _load_tree():
    """
    Load the full org tree from DB and normalize into in-memory structures.
    
    Returns:
        dict with:
            - nodes_by_id: dict[int, dict]
            - children_map: dict[int, list[int]]
            - raw_tree: list[dict]
    """
    raw_rows = admin_units.get_admin_unit_tree()
    raw_tree = [_row_to_dict(row) for row in raw_rows]
    
    nodes_by_id = {}
    children_map = {}
    
    # Build node index
    for node in raw_tree:
        node_id = node["UniqueID"]
        nodes_by_id[node_id] = node
        children_map[node_id] = []
    
    # Build parent-child relationships
    for node in raw_tree:
        node_id = node["UniqueID"]
        parent_id = node["ParentID"]
        
        # Skip root nodes (where ParentID == UniqueID)
        if parent_id != node_id and parent_id in children_map:
            children_map[parent_id].append(node_id)
    
    return {
        "nodes_by_id": nodes_by_id,
        "children_map": children_map,
        "raw_tree": raw_tree,
    }


def _get_tree():
    """
    Get the cached tree or load it if not yet loaded.
    """
    global _tree_cache
    
    if _tree_cache is None:
        _tree_cache = _load_tree()
    
    return _tree_cache


# =========================================================
# PUBLIC API
# =========================================================

def get_full_tree() -> list[dict]:
    """
    Return the raw org unit tree as a list of dicts.
    
    Each dict contains:
        - UniqueID: int
        - ParentID: int
        - Type: str
        - Name: str
    
    Returns:
        list[dict]: All org units in the tree
    """
    tree = _get_tree()
    return tree["raw_tree"]


def get_descendants(root_id: int) -> set[int]:
    """
    Return the set of root_id and all its descendant org unit IDs.
    
    Uses DFS to traverse the tree from root_id downward.
    
    Args:
        root_id: The root org unit ID
        
    Returns:
        set[int]: Set containing root_id and all descendant IDs
    """
    tree = _get_tree()
    nodes_by_id = tree["nodes_by_id"]
    children_map = tree["children_map"]
    
    # If node doesn't exist, return empty set
    if root_id not in nodes_by_id:
        return set()
    
    descendants = set()
    stack = [root_id]
    
    # DFS traversal
    while stack:
        current_id = stack.pop()
        descendants.add(current_id)
        
        # Add all children to stack
        for child_id in children_map.get(current_id, []):
            stack.append(child_id)
    
    return descendants


def get_ancestors(node_id: int) -> set[int]:
    """
    Return the set of ancestors of node_id up to the root.
    
    Does NOT include node_id itself.
    
    Args:
        node_id: The org unit ID to find ancestors for
        
    Returns:
        set[int]: Set of ancestor IDs (not including node_id)
    """
    tree = _get_tree()
    nodes_by_id = tree["nodes_by_id"]
    
    # If node doesn't exist, return empty set
    if node_id not in nodes_by_id:
        return set()
    
    ancestors = set()
    current_id = node_id
    
    # Walk up the tree until we reach a root (where ParentID == UniqueID)
    while True:
        node = nodes_by_id.get(current_id)
        if not node:
            break
        
        parent_id = node["ParentID"]
        
        # Stop if we've reached a root node
        if parent_id == current_id:
            break
        
        # Stop if we've seen this parent before (circular reference protection)
        if parent_id in ancestors:
            break
        
        ancestors.add(parent_id)
        current_id = parent_id
    
    return ancestors


def is_ancestor(parent_id: int, child_id: int) -> bool:
    """
    Return True if parent_id is an ancestor of child_id in the org tree.
    
    Args:
        parent_id: The potential ancestor org unit ID
        child_id: The child org unit ID
        
    Returns:
        bool: True if parent_id is an ancestor of child_id
    """
    # Get all ancestors of child_id
    ancestors = get_ancestors(child_id)
    return parent_id in ancestors


def clear_cache():
    """
    Clear the cached tree data.
    Useful for testing or if the tree structure changes at runtime.
    """
    global _tree_cache
    _tree_cache = None


# =========================================================
# TESTING / DEBUG
# =========================================================

if __name__ == "__main__":
    """
    Manual test block to verify tree traversal logic.
    """
    print("=" * 60)
    print("CENTRAL ORG TREE SERVICE - TEST")
    print("=" * 60)
    
    # Load the tree
    full_tree = get_full_tree()
    print(f"\nLoaded {len(full_tree)} org units")
    
    # Find an Administration node (root where ParentID == UniqueID)
    admin_nodes = [n for n in full_tree if n["ParentID"] == n["UniqueID"]]
    if admin_nodes:
        admin_node = admin_nodes[0]
        admin_id = admin_node["UniqueID"]
        admin_name = admin_node["Name"]
        
        print(f"\nTesting with Administration: {admin_name} (ID: {admin_id})")
        
        # Get all descendants
        admin_descendants = get_descendants(admin_id)
        print(f"   → Descendants: {len(admin_descendants)} units (including self)")
        
        # Find a department under this administration
        dept_nodes = [n for n in full_tree 
                      if n["ParentID"] == admin_id and n["UniqueID"] != admin_id]
        
        if dept_nodes:
            dept_node = dept_nodes[0]
            dept_id = dept_node["UniqueID"]
            dept_name = dept_node["Name"]
            
            print(f"\nTesting with Department: {dept_name} (ID: {dept_id})")
            
            # Get department descendants
            dept_descendants = get_descendants(dept_id)
            print(f"   → Descendants: {len(dept_descendants)} units (including self)")
            
            # Check ancestry
            is_admin_ancestor = is_ancestor(admin_id, dept_id)
            print(f"   → Is {admin_name} ancestor of {dept_name}? {is_admin_ancestor}")
            
            # Get department ancestors
            dept_ancestors = get_ancestors(dept_id)
            print(f"   → Ancestors: {dept_ancestors}")
            
            # Find a section under this department
            section_nodes = [n for n in full_tree 
                           if n["ParentID"] == dept_id and n["UniqueID"] != dept_id]
            
            if section_nodes:
                section_node = section_nodes[0]
                section_id = section_node["UniqueID"]
                section_name = section_node["Name"]
                
                print(f"\nTesting with Section: {section_name} (ID: {section_id})")
                
                # Get section descendants (should only be itself)
                section_descendants = get_descendants(section_id)
                print(f"   → Descendants: {len(section_descendants)} units (should be 1 - only itself)")
                
                # Check ancestry
                is_admin_ancestor_of_section = is_ancestor(admin_id, section_id)
                is_dept_ancestor_of_section = is_ancestor(dept_id, section_id)
                print(f"   → Is {admin_name} ancestor of {section_name}? {is_admin_ancestor_of_section}")
                print(f"   → Is {dept_name} ancestor of {section_name}? {is_dept_ancestor_of_section}")
                
                # Get section ancestors
                section_ancestors = get_ancestors(section_id)
                print(f"   → Ancestors: {section_ancestors}")
    
    print("\n" + "=" * 60)
    print("All tests completed")
    print("=" * 60)
