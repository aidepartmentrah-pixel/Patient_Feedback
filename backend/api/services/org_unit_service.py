"""
Organizational Unit Service

Provides specialized organizational unit selection functions for different use cases:
1. Leaf nodes (sections) for insert forms
2. Top-level administrations for reports
3. Filtered lists by type

This service complements the existing hierarchical services by providing
flat, filtered lists of organizational units.
"""

from typing import List, Dict, Optional
from ..db_layer import admin_units
from ..constants.org_unit_types import (
    ORG_TYPE_ADMINISTRATION,
    ORG_TYPE_DEPARTMENT,
    ORG_TYPE_SECTION,
    ORG_TYPE_NAME_MAP
)


def get_leaf_units() -> List[Dict]:
    """
    Get all leaf organizational units (units with no children).
    
    Use Case: Insert/Add forms where users select the ACTUAL department
    where an incident occurred (not abstract administrative groupings).
    
    Returns:
        List of dicts with:
        - id: Unit ID
        - name: Unit name
        - name_ar: Unit name (Arabic)
        - parent_id: Parent unit ID
        - parent_name: Parent unit name (for context)
        - type: Unit type code
        - type_name: Unit type name (SECTION, DEPARTMENT, etc.)
    
    Example:
        [
            {
                "id": 45,
                "name": "Emergency Section",
                "name_ar": "قسم الطوارئ",
                "parent_id": 10,
                "parent_name": "Emergency Department",
                "type": 324,
                "type_name": "SECTION"
            }
        ]
    """
    # Get leaf nodes from database
    leaf_rows = admin_units.get_admin_unit_leaves()
    
    # Get all units for parent name lookup
    all_units_rows = admin_units.get_admin_unit_tree()
    units_by_id = {u.UniqueID: u for u in all_units_rows}
    
    result = []
    for leaf in leaf_rows:
        leaf_id = leaf.UniqueID
        parent_id = leaf.ParentID
        
        # Get parent info
        parent_name = None
        if parent_id != leaf_id and parent_id in units_by_id:
            parent_name = units_by_id[parent_id].Name
        
        # Get type name
        type_code = leaf.Type
        type_name = ORG_TYPE_NAME_MAP.get(type_code, "UNKNOWN")
        
        result.append({
            "id": leaf_id,
            "name": leaf.Name,
            "name_ar": leaf.Name,  # Assuming Name field contains Arabic/bilingual
            "parent_id": parent_id if parent_id != leaf_id else None,
            "parent_name": parent_name,
            "type": type_code,
            "type_name": type_name
        })
    
    return result


def get_administrations() -> List[Dict]:
    """
    Get all top-level administration units only.
    
    Use Case: Reports and aggregate analysis where you need to select
    major divisions of the hospital (e.g., "Medical Administration",
    "Surgical Administration").
    
    Returns:
        List of dicts with:
        - id: Administration ID
        - name: Administration name
        - name_ar: Administration name (Arabic)
    
    Example:
        [
            {
                "id": 1,
                "name": "Medical Administration",
                "name_ar": "الإدارة الطبية"
            }
        ]
    """
    all_units = admin_units.get_admin_unit_tree()
    
    result = []
    for unit in all_units:
        # Administration: ParentID == UniqueID (self-referencing root)
        if unit.ParentID == unit.UniqueID:
            result.append({
                "id": unit.UniqueID,
                "name": unit.Name,
                "name_ar": unit.Name
            })
    
    return result


def get_departments() -> List[Dict]:
    """
    Get all department units.
    
    Use Case: Mid-level filtering or reporting by department.
    
    Returns:
        List of dicts with:
        - id: Department ID
        - name: Department name
        - name_ar: Department name (Arabic)
        - administration_id: Parent administration ID
    
    Example:
        [
            {
                "id": 10,
                "name": "Emergency Department",
                "name_ar": "قسم الطوارئ",
                "administration_id": 1
            }
        ]
    """
    all_units = admin_units.get_admin_unit_tree()
    
    # Build lookup for parent checking
    units_by_id = {u.UniqueID: u for u in all_units}
    
    result = []
    for unit in all_units:
        unit_id = unit.UniqueID
        parent_id = unit.ParentID
        
        # Skip administrations (self-referencing)
        if parent_id == unit_id:
            continue
        
        # Check if parent is an administration
        if parent_id in units_by_id:
            parent = units_by_id[parent_id]
            if parent.ParentID == parent.UniqueID:
                # Parent is an administration, so this is a department
                result.append({
                    "id": unit_id,
                    "name": unit.Name,
                    "name_ar": unit.Name,
                    "administration_id": parent_id
                })
    
    return result


def get_section_parents() -> List[Dict]:
    """
    Get all valid parent units for section creation.
    
    Use Case: Parent dropdown when creating a new section.
    Only ADMINISTRATION (Type=323) and DEPARTMENT (Type=325) units
    can be parents of sections. SECTION units (Type=324) cannot be parents.
    
    Returns:
        List of dicts with:
        - id: Unit ID
        - name: Unit name
        - name_ar: Unit name (Arabic)
        - type: Type code (323 or 325)
        - type_name: Type name (ADMINISTRATION or DEPARTMENT)
    
    Example:
        [
            {
                "id": 1,
                "name": "Medical Administration",
                "name_ar": "الإدارة الطبية",
                "type": 323,
                "type_name": "ADMINISTRATION"
            },
            {
                "id": 10,
                "name": "Emergency Department",
                "name_ar": "قسم الطوارئ",
                "type": 325,
                "type_name": "DEPARTMENT"
            }
        ]
    """
    all_units = admin_units.get_admin_unit_tree()
    
    # Valid parent types for sections
    valid_types = [ORG_TYPE_ADMINISTRATION, ORG_TYPE_DEPARTMENT]
    
    result = []
    for unit in all_units:
        unit_type = unit.Type
        if unit_type in valid_types:
            type_name = ORG_TYPE_NAME_MAP.get(unit_type, "UNKNOWN")
            result.append({
                "id": unit.UniqueID,
                "name": unit.Name,
                "name_ar": unit.Name,
                "type": unit_type,
                "type_name": type_name
            })
    
    return result


def get_sections() -> List[Dict]:
    """
    Get all section units (children of departments).
    
    Use Case: Section-level filtering or reporting.
    
    Returns:
        List of dicts with:
        - id: Section ID
        - name: Section name
        - name_ar: Section name (Arabic)
        - department_id: Parent department ID
    
    Example:
        [
            {
                "id": 45,
                "name": "Emergency Section",
                "name_ar": "قسم الطوارئ",
                "department_id": 10
            }
        ]
    """
    all_units = admin_units.get_admin_unit_tree()
    
    # Build lookup for parent checking
    units_by_id = {u.UniqueID: u for u in all_units}
    
    result = []
    for unit in all_units:
        unit_id = unit.UniqueID
        parent_id = unit.ParentID
        
        # Skip administrations (self-referencing)
        if parent_id == unit_id:
            continue
        
        # Check if parent is a department
        if parent_id in units_by_id:
            parent = units_by_id[parent_id]
            parent_parent_id = parent.ParentID
            
            # If parent's parent is an administration, then parent is a department
            # and this unit is a section
            if parent_parent_id != parent.UniqueID and parent_parent_id in units_by_id:
                grandparent = units_by_id[parent_parent_id]
                if grandparent.ParentID == grandparent.UniqueID:
                    # Grandparent is administration, parent is department, this is section
                    result.append({
                        "id": unit_id,
                        "name": unit.Name,
                        "name_ar": unit.Name,
                        "department_id": parent_id
                    })
    
    return result


def get_units_by_type(unit_type: str) -> List[Dict]:
    """
    Get organizational units filtered by type.
    
    Args:
        unit_type: One of "ADMINISTRATION", "DEPARTMENT", or "SECTION"
    
    Returns:
        List of organizational units of the specified type
    
    Raises:
        ValueError: If unit_type is invalid
    """
    unit_type_upper = unit_type.upper()
    
    if unit_type_upper == "ADMINISTRATION":
        return get_administrations()
    elif unit_type_upper == "DEPARTMENT":
        return get_departments()
    elif unit_type_upper == "SECTION":
        return get_sections()
    else:
        raise ValueError(
            f"Invalid unit_type: {unit_type}. "
            f"Must be one of: ADMINISTRATION, DEPARTMENT, SECTION"
        )


def get_unit_with_ancestors(unit_id: int) -> Optional[Dict]:
    """
    Get a single organizational unit with its full ancestry chain.
    
    Use Case: Display breadcrumb navigation or full context for a unit.
    
    Args:
        unit_id: Organizational unit ID
    
    Returns:
        Dict with:
        - id: Unit ID
        - name: Unit name
        - type: Unit type code
        - type_name: Unit type name
        - ancestors: List of ancestor units from root to parent
    
    Example:
        {
            "id": 45,
            "name": "Emergency Section",
            "type": 324,
            "type_name": "SECTION",
            "ancestors": [
                {"id": 1, "name": "Medical Administration", "type_name": "ADMINISTRATION"},
                {"id": 10, "name": "Emergency Department", "type_name": "DEPARTMENT"}
            ]
        }
    
    Returns None if unit not found.
    """
    all_units = admin_units.get_admin_unit_tree()
    units_by_id = {u.UniqueID: u for u in all_units}
    
    if unit_id not in units_by_id:
        return None
    
    unit = units_by_id[unit_id]
    
    # Build ancestor chain
    ancestors = []
    current_id = unit.ParentID
    visited = set([unit_id])  # Prevent infinite loops
    
    while current_id != unit_id and current_id in units_by_id and current_id not in visited:
        visited.add(current_id)
        ancestor = units_by_id[current_id]
        
        ancestors.insert(0, {
            "id": ancestor.UniqueID,
            "name": ancestor.Name,
            "type": ancestor.Type,
            "type_name": ORG_TYPE_NAME_MAP.get(ancestor.Type, "UNKNOWN")
        })
        
        # Move to next ancestor
        if ancestor.ParentID == ancestor.UniqueID:
            # Reached root (administration)
            break
        current_id = ancestor.ParentID
    
    return {
        "id": unit.UniqueID,
        "name": unit.Name,
        "type": unit.Type,
        "type_name": ORG_TYPE_NAME_MAP.get(unit.Type, "UNKNOWN"),
        "ancestors": ancestors
    }
