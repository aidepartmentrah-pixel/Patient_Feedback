"""
Organizational Unit Router

Provides specialized endpoints for organizational unit selection:
- Leaf nodes for insert forms
- Administrations for reports
- Departments and sections for filtering

These endpoints complement the existing hierarchical endpoints by providing
flat, filtered lists optimized for specific UI components.
"""

from fastapi import APIRouter, HTTPException, Path
from typing import List, Dict

from ..services import org_unit_service


router = APIRouter(prefix="/api/org-units", tags=["Organizational Units"])


@router.get("/leaves")
def get_leaf_units():
    """
    Get all leaf organizational units (units with no children).
    
    **Use Case**: Insert/Add Patient forms, Incident creation forms
    
    **Why**: Users need to select the ACTUAL department where an incident
    occurred (e.g., "Emergency Section"), not abstract administrative
    groupings (e.g., "Medical Administration").
    
    **Returns**:
    ```json
    {
      "leaves": [
        {
          "id": 45,
          "name": "Emergency Section",
          "name_ar": "قسم الطوارئ",
          "parent_id": 10,
          "parent_name": "Emergency Department",
          "type": 324,
          "type_name": "SECTION"
        }
      ],
      "count": 1
    }
    ```
    
    **Frontend Usage**:
    ```javascript
    // Populate issuing department dropdown
    const response = await fetch('/api/org-units/leaves');
    const data = await response.json();
    const options = data.leaves.map(leaf => ({
      value: leaf.id,
      label: leaf.name
    }));
    ```
    """
    try:
        leaves = org_unit_service.get_leaf_units()
        return {
            "leaves": leaves,
            "count": len(leaves)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve leaf units: {str(e)}"
        )


@router.get("/administrations")
def get_administrations():
    """
    Get all top-level administration units only.
    
    **Use Case**: Report configuration, aggregate analysis, high-level filtering
    
    **Why**: Reports need to compare major hospital divisions
    (e.g., "Medical Administration" vs "Surgical Administration").
    
    **Returns**:
    ```json
    {
      "administrations": [
        {
          "id": 1,
          "name": "Medical Administration",
          "name_ar": "الإدارة الطبية"
        }
      ],
      "count": 1
    }
    ```
    
    **Frontend Usage**:
    ```javascript
    // Populate report scope dropdown
    const response = await fetch('/api/org-units/administrations');
    const data = await response.json();
    const options = [
      { value: 'all', label: 'All Administrations' },
      ...data.administrations.map(admin => ({
        value: admin.id,
        label: admin.name
      }))
    ];
    ```
    """
    try:
        administrations = org_unit_service.get_administrations()
        return {
            "administrations": administrations,
            "count": len(administrations)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve administrations: {str(e)}"
        )


@router.get("/departments")
def get_departments():
    """
    Get all department units.
    
    **Use Case**: Mid-level filtering or reporting by department.
    
    **Returns**:
    ```json
    {
      "departments": [
        {
          "id": 10,
          "name": "Emergency Department",
          "name_ar": "قسم الطوارئ",
          "administration_id": 1
        }
      ],
      "count": 1
    }
    ```
    
    **Frontend Usage**:
    ```javascript
    // Populate department filter dropdown
    const response = await fetch('/api/org-units/departments');
    const data = await response.json();
    const options = data.departments.map(dept => ({
      value: dept.id,
      label: dept.name,
      administrationId: dept.administration_id
    }));
    ```
    """
    try:
        departments = org_unit_service.get_departments()
        return {
            "departments": departments,
            "count": len(departments)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve departments: {str(e)}"
        )


@router.get("/section-parents")
def get_section_parents():
    """
    Get all valid parent units for section creation.
    
    **Use Case**: Parent dropdown when creating a new section.
    Only ADMINISTRATION (Type=323) and DEPARTMENT (Type=325) units
    can be parents of sections. SECTION units (Type=324) cannot be parents.
    
    **Returns**:
    ```json
    {
      "parents": [
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
      ],
      "count": 2
    }
    ```
    
    **Frontend Usage**:
    ```javascript
    // Populate parent dropdown for section creation
    const response = await fetch('/api/org-units/section-parents');
    const data = await response.json();
    const options = data.parents.map(parent => ({
      value: parent.id,
      label: `${parent.name} (${parent.type_name})`
    }));
    ```
    """
    try:
        parents = org_unit_service.get_section_parents()
        return {
            "parents": parents,
            "count": len(parents)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve section parents: {str(e)}"
        )


@router.get("/all-targets")
def get_all_target_units():
    """
    Get all active org units available as complaint targets:
    Administration (type 323), Department (type 325), Section (type 324).

    Used by the Insert page target dropdown so a complaint can be issued
    against any level of the org hierarchy.

    Returns:
    ```json
    {
      "units": [
        {"id": 1, "name": "Medical Administration", "type": 323, "type_label": "Administration"},
        {"id": 10, "name": "Emergency Department",  "type": 325, "type_label": "Department"},
        {"id": 45, "name": "ER Reception",          "type": 324, "type_label": "Section"}
      ],
      "count": 3
    }
    ```
    """
    try:
        units = org_unit_service.get_all_target_units()
        return {"units": units, "count": len(units)}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve target units: {str(e)}"
        )


@router.get("/sections")
def get_sections():
    """
    Get all section units (children of departments).
    
    **Use Case**: Section-level filtering or reporting.
    
    **Returns**:
    ```json
    {
      "sections": [
        {
          "id": 45,
          "name": "Emergency Section",
          "name_ar": "قسم الطوارئ",
          "department_id": 10
        }
      ],
      "count": 1
    }
    ```
    
    **Frontend Usage**:
    ```javascript
    // Populate section filter dropdown
    const response = await fetch('/api/org-units/sections');
    const data = await response.json();
    const options = data.sections.map(section => ({
      value: section.id,
      label: section.name,
      departmentId: section.department_id
    }));
    ```
    """
    try:
        sections = org_unit_service.get_sections()
        return {
            "sections": sections,
            "count": len(sections)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve sections: {str(e)}"
        )


@router.get("/unit/{unit_id}")
def get_unit_with_ancestors(
    unit_id: int = Path(..., ge=1, description="Organizational unit ID")
):
    """
    Get a single organizational unit with its full ancestry chain.
    
    **Use Case**: Display breadcrumb navigation or full context for a unit.
    
    **Example**: GET /api/org-units/unit/45
    
    **Returns**:
    ```json
    {
      "id": 45,
      "name": "Emergency Section",
      "type": 324,
      "type_name": "SECTION",
      "ancestors": [
        {
          "id": 1,
          "name": "Medical Administration",
          "type": 323,
          "type_name": "ADMINISTRATION"
        },
        {
          "id": 10,
          "name": "Emergency Department",
          "type": 325,
          "type_name": "DEPARTMENT"
        }
      ],
      "breadcrumb": "Medical Administration > Emergency Department > Emergency Section"
    }
    ```
    
    **Frontend Usage**:
    ```javascript
    // Display breadcrumb for a unit
    const response = await fetch(`/api/org-units/unit/${unitId}`);
    const data = await response.json();
    console.log(data.breadcrumb);
    // Output: "Medical Administration > Emergency Department > Emergency Section"
    ```
    """
    try:
        result = org_unit_service.get_unit_with_ancestors(unit_id)
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Organizational unit with ID {unit_id} not found"
            )
        
        # Build breadcrumb string
        breadcrumb_parts = [ancestor["name"] for ancestor in result["ancestors"]]
        breadcrumb_parts.append(result["name"])
        breadcrumb = " > ".join(breadcrumb_parts)
        
        result["breadcrumb"] = breadcrumb
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve unit: {str(e)}"
        )


@router.get("/summary")
def get_org_units_summary():
    """
    Get a summary count of organizational units by type.
    
    **Use Case**: Admin dashboard, system overview.
    
    **Returns**:
    ```json
    {
      "administrations": 5,
      "departments": 23,
      "sections": 87,
      "total": 115,
      "leaves": 87
    }
    ```
    """
    try:
        administrations = org_unit_service.get_administrations()
        departments = org_unit_service.get_departments()
        sections = org_unit_service.get_sections()
        leaves = org_unit_service.get_leaf_units()
        
        return {
            "administrations": len(administrations),
            "departments": len(departments),
            "sections": len(sections),
            "total": len(administrations) + len(departments) + len(sections),
            "leaves": len(leaves)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve summary: {str(e)}"
        )
