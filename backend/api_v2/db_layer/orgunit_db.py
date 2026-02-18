"""
Organization Unit Database Layer (API V2)
Handles SQL operations for AdminsrationUnit table.

This is a pure DB access helper for organization hierarchy resolution.
NO business logic. NO authorization. ONLY SQL operations.
READ ONLY - no mutations.
"""

from typing import Dict, Any, List, Optional, Set
from core.database import get_connection


# ============================================================
# READ OPERATIONS
# ============================================================

def get_all_orgunits() -> List[Dict[str, Any]]:
    """
    Get all organizational units from the database.
    
    Returns the full hierarchy:
        - Administration (root)
        - Departments (children of Administration)
        - Sections (children of Departments)
    
    Returns:
        List of dictionaries containing all orgunit records
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT 
                UniqueID,
                ParentID,
                Type,
                Name
            FROM dbo.AdminsrationUnit
            ORDER BY UniqueID
        """)
        
        rows = cursor.fetchall()
        
        # Convert to list of dictionaries
        orgunits = []
        for row in rows:
            orgunits.append({
                "UniqueID": row.UniqueID,
                "ParentID": row.ParentID,
                "Type": row.Type,
                "Name": row.Name
            })
        
        return orgunits
        
    finally:
        cursor.close()
        conn.close()


def get_descendant_orgunit_ids(root_orgunit_id: int) -> Set[int]:
    """
    Get all descendant organizational unit IDs for a given root orgunit.
    
    This function resolves the organizational hierarchy and returns:
        - The root orgunit ID itself
        - All direct children
        - All grandchildren
        - All deeper descendants
    
    Examples:
        - Administration root: Returns all departments and sections below it
        - Department root: Returns the department itself and all sections below it
        - Section root: Returns only the section itself (leaf node)
    
    Implementation:
        1. Loads all orgunits from database
        2. Builds an in-memory parent → children map
        3. Traverses the hierarchy starting from root
        4. Collects all descendant IDs
    
    Args:
        root_orgunit_id: The ID of the root organizational unit
        
    Returns:
        Set of organizational unit IDs including root and all descendants
    """
    # Step 1: Load all orgunits
    all_orgunits = get_all_orgunits()
    
    # Step 2: Build parent → children map
    children_map = {}
    for orgunit in all_orgunits:
        parent_id = orgunit.get("ParentID")
        unique_id = orgunit.get("UniqueID")
        
        if parent_id is not None:
            if parent_id not in children_map:
                children_map[parent_id] = []
            children_map[parent_id].append(unique_id)
    
    # Step 3: Traverse hierarchy and collect descendants
    descendants = set()
    
    def collect_descendants(node_id: int):
        """Recursive helper to collect all descendants."""
        # Add current node
        descendants.add(node_id)
        
        # Add all children and their descendants
        if node_id in children_map:
            for child_id in children_map[node_id]:
                collect_descendants(child_id)
    
    # Start traversal from root
    collect_descendants(root_orgunit_id)
    
    return descendants
