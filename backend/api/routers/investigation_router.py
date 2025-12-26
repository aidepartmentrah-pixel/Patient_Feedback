from fastapi import APIRouter, Query, HTTPException
import traceback
from typing import Literal
from ..services.investigation_service import (
    get_investigation_tree,
    get_available_seasons,
    get_organizational_hierarchy,
)

router = APIRouter(
    prefix="/api/investigation",
    tags=["Investigation"],
)


# =========================================================
# INVESTIGATION TREE ENDPOINT
# =========================================================

@router.get("/tree")
def fetch_investigation_tree(
    season: str = Query(
        ...,
        description="Season identifier (e.g., '2024-Q4' or season ID)"
    ),
    tree_type: Literal[
        "incident_count",
        "domain_distribution_numbers",
        "domain_distribution_percentage",
        "severity_distribution_numbers",
        "severity_distribution_percentage",
        "red_flag_incidents",
        "never_event_incidents"
    ] = Query(
        ...,
        description="Type of aggregation/visualization for the tree"
    ),
    administration_id: int | None = Query(
        None,
        description="Filter to specific administration (Idara)"
    ),
    department_id: int | None = Query(
        None,
        description="Filter to specific department (Dayra)"
    ),
    section_id: int | None = Query(
        None,
        description="Filter to specific section (Qism)"
    ),
):
    """
    Fetch hierarchical investigation tree with aggregated incident data.
    
    **Tree Types:**
    - `incident_count`: Total incident count per organizational node
    - `domain_distribution_numbers`: Domain breakdown (absolute counts)
    - `domain_distribution_percentage`: Domain breakdown (percentages)
    - `severity_distribution_numbers`: Severity breakdown (absolute counts)
    - `severity_distribution_percentage`: Severity breakdown (percentages)
    - `red_flag_incidents`: Red flag incident count per node
    - `never_event_incidents`: Never event incident count per node
    
    **Organizational Filters (Hierarchical):**
    - No filter: Show entire hospital hierarchy (all administrations)
    - administration_id: Show that administration and its departments/sections
    - department_id: Show that department and its sections
    - section_id: Show only that section
    
    **Response includes:**
    - Nested tree structure with aggregated metrics
    - Node metadata (ID, name, type, level)
    - Aggregated values based on tree_type
    - Summary statistics for entire scope
    """
    
    print("=" * 80)
    print("INVESTIGATION TREE REQUEST RECEIVED:")
    print(f"  season: {season}")
    print(f"  tree_type: {tree_type}")
    print(f"  administration_id: {administration_id}")
    print(f"  department_id: {department_id}")
    print(f"  section_id: {section_id}")
    print("=" * 80)
    
    # -------------------------
    # Call service
    # -------------------------
    try:
        return get_investigation_tree(
            season=season,
            tree_type=tree_type,
            administration_id=administration_id,
            department_id=department_id,
            section_id=section_id,
        )
    
    except ValueError as e:
        # Handle specific business logic errors
        print(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "message": str(e),
                "message_ar": str(e)  # You can add Arabic translation here
            }
        )
    
    except Exception as e:
        print(f"Investigation tree error:")
        print(f"  season: {season}")
        print(f"  tree_type: {tree_type}")
        print(f"  scope: administration={administration_id}, dept={department_id}, section={section_id}")
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# =========================================================
# AVAILABLE SEASONS ENDPOINT
# =========================================================

@router.get("/seasons")
def fetch_available_seasons():
    """
    Fetch list of available investigation periods/seasons.
    
    Used to populate the season selector dropdown in the UI.
    
    **Response includes:**
    - Array of seasons with IDs, labels, and date ranges
    - Current season flag (is_current)
    - Current season ID
    
    **Example seasons:**
    - "2024-Q4" (Oct-Dec 2024)
    - "2024-Q3" (Jul-Sep 2024)
    - Custom periods based on Season table
    """
    
    print("=" * 80)
    print("AVAILABLE SEASONS REQUEST RECEIVED")
    print("=" * 80)
    
    try:
        return get_available_seasons()
    
    except Exception as e:
        print(f"Available seasons error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# =========================================================
# ORGANIZATIONAL HIERARCHY ENDPOINT
# =========================================================

@router.get("/hierarchy")
def fetch_organizational_hierarchy():
    """
    Fetch organizational structure for cascading selectors.
    
    Returns the three-level organizational hierarchy:
    - Administrations (Idara) - Top level
    - Departments (Dayra) - Middle level
    - Sections (Qism) - Leaf level
    
    Used to populate cascading dropdowns in the filter UI.
    
    **Response includes:**
    - Array of administrations with IDs and names
    - Array of departments with parent administration IDs
    - Array of sections with parent department IDs
    - Names in both English and Arabic
    
    **Can be cached on frontend** (structure changes infrequently)
    """
    
    print("=" * 80)
    print("ORGANIZATIONAL HIERARCHY REQUEST RECEIVED")
    print("=" * 80)
    
    try:
        return get_organizational_hierarchy()
    
    except Exception as e:
        print(f"Organizational hierarchy error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
