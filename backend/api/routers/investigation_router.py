from fastapi import APIRouter, Query, HTTPException
import traceback
from datetime import date as DateType
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
    start_date: DateType = Query(
        ...,
        description="Start date of the investigation period (YYYY-MM-DD)"
    ),
    end_date: DateType = Query(
        ...,
        description="End date of the investigation period (YYYY-MM-DD)"
    ),
    tree_type: Literal[
        "incident_count",
        "domain_distribution_numbers",
        "domain_distribution_percentage",
        "severity_distribution_numbers",
        "severity_distribution_percentage",
        "red_flag_incidents",
        "never_event_incidents",
        "notice_count",
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

    **Period:**
    - start_date / end_date: ISO date strings (YYYY-MM-DD) defining the investigation window.
      The frontend computes these from seasonal, yearly, or custom period selection.

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
    """

    print("=" * 80)
    print("INVESTIGATION TREE REQUEST RECEIVED:")
    print(f"  start_date: {start_date}")
    print(f"  end_date: {end_date}")
    print(f"  tree_type: {tree_type}")
    print(f"  administration_id: {administration_id}")
    print(f"  department_id: {department_id}")
    print(f"  section_id: {section_id}")
    print("=" * 80)

    if start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail={"error": "validation_error", "message": "start_date must be on or before end_date"}
        )

    try:
        return get_investigation_tree(
            start_date=start_date,
            end_date=end_date,
            tree_type=tree_type,
            administration_id=administration_id,
            department_id=department_id,
            section_id=section_id,
        )

    except ValueError as e:
        print(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={"error": "validation_error", "message": str(e)}
        )

    except Exception as e:
        print(f"Investigation tree error: {str(e)}")
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
    Fetch list of available investigation seasons/periods.

    Used to populate the seasonal period selector in the UI.
    Each season includes start_date and end_date so the frontend
    can compute the date range for the /tree request.
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
