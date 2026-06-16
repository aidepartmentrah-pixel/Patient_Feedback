from datetime import date, timedelta
from fastapi import APIRouter, Query, HTTPException, Depends
import traceback
from ..services.dashboard_service import get_dashboard_stats, get_dashboard_hierarchy, get_operational_summary
from ..dependencies.user_context import get_current_user
from ..schemas.auth_models import CurrentUser
from ..utils.guards import require_unit_in_scope

router = APIRouter(
    prefix="/api/dashboard",
    tags=["Dashboard"],
)

# =========================================================
# DASHBOARD STATS ENDPOINT
# =========================================================

@router.get("/stats")
def dashboard_stats(
    current_user: CurrentUser = Depends(get_current_user),
    scope: str = Query(..., description="hospital | administration | department | section"),
    administration_id: int | None = Query(None),
    department_id: int | None = Query(None),
    section_id: int | None = Query(None),
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    classification_chart_type: str = Query("bar", description="bar | pie | donut | line"),
    stage_chart_type: str = Query("bar", description="bar | pie | donut | line"),
    department_chart_type: str = Query("bar", description="bar | pie | donut | line"),
):
    # Log all incoming parameters
    print("=" * 80)
    print(f"DASHBOARD REQUEST RECEIVED:")
    print(f"  scope: {scope} (type: {type(scope)})")
    print(f"  administration_id: {administration_id} (type: {type(administration_id)})")
    print(f"  department_id: {department_id} (type: {type(department_id)})")
    print(f"  section_id: {section_id} (type: {type(section_id)})")
    print(f"  start_date: {start_date}")
    print(f"  end_date: {end_date}")
    print("=" * 80)
    
    # -------------------------
    # Validate scope logic
    # -------------------------
    if scope not in {"hospital", "administration", "department", "section"}:
        print(f"ERROR: Invalid scope '{scope}'")
        raise HTTPException(status_code=400, detail="Invalid scope")

    if scope == "administration" and administration_id is None:
        print(f"ERROR: administration_id required for administration scope")
        raise HTTPException(status_code=400, detail="administration_id required")

    if scope == "department" and department_id is None:
        print(f"ERROR: department_id required for department scope")
        raise HTTPException(status_code=400, detail="department_id required")

    if scope == "section" and section_id is None:
        print(f"ERROR: section_id required for section scope")
        raise HTTPException(status_code=400, detail="section_id required")
    
    # -------------------------
    # Enforce organizational scope (Phase 2.5)
    # Validate client-provided org unit IDs against user's allowed scope
    # -------------------------
    if scope == "administration" and administration_id is not None:
        require_unit_in_scope(current_user, administration_id)
    
    if scope == "department" and department_id is not None:
        require_unit_in_scope(current_user, department_id)
    
    if scope == "section" and section_id is not None:
        require_unit_in_scope(current_user, section_id)

    # -------------------------
    # Default date handling
    # -------------------------
    if end_date is None:
        end_date = date.today()

    if start_date is None:
        start_date = end_date - timedelta(days=30)

    # -------------------------
    # Call service
    # -------------------------
    try:
        return get_dashboard_stats(
            current_user=current_user,
            scope=scope,
            administration_id=administration_id,
            department_id=department_id,
            section_id=section_id,
            start_date=start_date,
            end_date=end_date,
            classification_chart_type=classification_chart_type,
            stage_chart_type=stage_chart_type,
            department_chart_type=department_chart_type,
        )
    except Exception as e:
        print(f"Dashboard error - scope: {scope}, admin: {administration_id}, dept: {department_id}, section: {section_id}")
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# =========================================================
# DASHBOARD HIERARCHY ENDPOINT
# =========================================================


# This Works here
@router.get("/hierarchy")
def dashboard_hierarchy(
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get organizational hierarchy filtered by user's scope."""
    return get_dashboard_hierarchy(current_user)


# =========================================================
# DASHBOARD DATE BOUNDS ENDPOINT (PHASE DR-B)
# =========================================================

@router.get("/date-bounds")
def dashboard_date_bounds(
    current_user: CurrentUser = Depends(get_current_user),
    scope: str = Query(..., description="hospital | administration | department | section"),
    administration_id: int | None = Query(None),
    department_id: int | None = Query(None),
    section_id: int | None = Query(None),
):
    """
    Get minimum and maximum incident CreatedAt dates within dashboard scope.
    
    Returns DATE only (YYYY-MM-DD), not datetime.
    Uses identical scope resolution and RBAC filtering as dashboard stats.
    """
    
    # -------------------------
    # Validate scope logic (same as stats endpoint)
    # -------------------------
    if scope not in {"hospital", "administration", "department", "section"}:
        raise HTTPException(status_code=400, detail="Invalid scope")

    if scope == "administration" and administration_id is None:
        raise HTTPException(status_code=400, detail="administration_id required")

    if scope == "department" and department_id is None:
        raise HTTPException(status_code=400, detail="department_id required")

    if scope == "section" and section_id is None:
        raise HTTPException(status_code=400, detail="section_id required")
    
    # -------------------------
    # Enforce organizational scope (same as stats endpoint)
    # -------------------------
    if scope == "administration" and administration_id is not None:
        require_unit_in_scope(current_user, administration_id)
    
    if scope == "department" and department_id is not None:
        require_unit_in_scope(current_user, department_id)
    
    if scope == "section" and section_id is not None:
        require_unit_in_scope(current_user, section_id)

    # -------------------------
    # Determine Requested Scope (INLINE - copied from get_dashboard_stats logic)
    # -------------------------
    from ..services import org_tree_service
    
    if scope == "section" and section_id is not None:
        requested_unit_ids = {section_id}
    
    elif scope == "department" and department_id is not None:
        requested_unit_ids = org_tree_service.get_descendants(department_id)
    
    elif scope == "administration" and administration_id is not None:
        requested_unit_ids = org_tree_service.get_descendants(administration_id)
    
    else:
        requested_unit_ids = current_user.allowed_unit_ids
    
    # -------------------------
    # RBAC Safety: Intersect with allowed scope
    # -------------------------
    scope_unit_ids = list(requested_unit_ids & current_user.allowed_unit_ids)

    # -------------------------
    # Call service
    # -------------------------
    try:
        from ..services.dashboard_service import get_dashboard_date_bounds_for_units
        result = get_dashboard_date_bounds_for_units(scope_unit_ids)
        
        # Convert date objects to ISO strings for JSON response
        return {
            "min_date": result["min_date"].isoformat() if result["min_date"] else None,
            "max_date": result["max_date"].isoformat() if result["max_date"] else None,
        }
    except Exception as e:
        print(f"Dashboard date bounds error - scope: {scope}, admin: {administration_id}, dept: {department_id}, section: {section_id}")
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# =========================================================
# OPERATIONAL DASHBOARD SUMMARY ENDPOINT (HCAT Perf Monitoring - Session 2)
# =========================================================

@router.get("/operational-summary")
def dashboard_operational_summary(
    current_user: CurrentUser = Depends(get_current_user),
    scope: str = Query(..., description="hospital | administration | department | section"),
    administration_id: int | None = Query(None),
    department_id: int | None = Query(None),
    section_id: int | None = Query(None),
):
    """
    Operational dashboard summary: open / closed / force closed / late replies /
    currently overdue / extra time granted.

    Uses identical scope resolution and RBAC filtering as dashboard stats.
    """

    # -------------------------
    # Validate scope logic (same as stats endpoint)
    # -------------------------
    if scope not in {"hospital", "administration", "department", "section"}:
        raise HTTPException(status_code=400, detail="Invalid scope")

    if scope == "administration" and administration_id is None:
        raise HTTPException(status_code=400, detail="administration_id required")

    if scope == "department" and department_id is None:
        raise HTTPException(status_code=400, detail="department_id required")

    if scope == "section" and section_id is None:
        raise HTTPException(status_code=400, detail="section_id required")

    # -------------------------
    # Enforce organizational scope (same as stats endpoint)
    # -------------------------
    if scope == "administration" and administration_id is not None:
        require_unit_in_scope(current_user, administration_id)

    if scope == "department" and department_id is not None:
        require_unit_in_scope(current_user, department_id)

    if scope == "section" and section_id is not None:
        require_unit_in_scope(current_user, section_id)

    # -------------------------
    # Determine Requested Scope (INLINE - copied from get_dashboard_stats logic)
    # -------------------------
    from ..services import org_tree_service

    if scope == "section" and section_id is not None:
        requested_unit_ids = {section_id}

    elif scope == "department" and department_id is not None:
        requested_unit_ids = org_tree_service.get_descendants(department_id)

    elif scope == "administration" and administration_id is not None:
        requested_unit_ids = org_tree_service.get_descendants(administration_id)

    else:
        requested_unit_ids = current_user.allowed_unit_ids

    # -------------------------
    # RBAC Safety: Intersect with allowed scope
    # -------------------------
    scope_unit_ids = list(requested_unit_ids & current_user.allowed_unit_ids)

    # -------------------------
    # Call service
    # -------------------------
    try:
        return get_operational_summary(scope_unit_ids)
    except Exception as e:
        print(f"Dashboard operational summary error - scope: {scope}, admin: {administration_id}, dept: {department_id}, section: {section_id}")
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# =========================================================
# DEBUG ENDPOINT: Check classifications mapping
# =========================================================

@router.get("/debug/classifications")
def debug_classifications():
    from ..db_layer import lookups
    
    classifications = lookups.get_classifications()
    return {
        "total_count": len(classifications),
        "classifications": classifications
    }


# =========================================================
# DEBUG ENDPOINT: Check stage histogram data
# =========================================================

@router.get("/debug/stage-histogram")
def debug_stage_histogram(
    scope: str = Query("hospital"),
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
):
    from datetime import timedelta
    from ..services.dashboard_service import _resolve_scope, _fetch_incidents_in_scope, _histogram_with_names
    from ..db_layer import lookups
    from collections import Counter
    
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=30)
    
    scope_unit_ids, _ = _resolve_scope(scope, None, None, None)
    incidents = _fetch_incidents_in_scope(scope_unit_ids, start_date, end_date)
    
    stages = lookups.get_case_stages()
    stage_map = {s["StageID"]: s["StageName"] for s in stages}
    
    counter = Counter(i["StageID"] for i in incidents if i.get("StageID"))
    
    return {
        "total_incidents": len(incidents),
        "raw_stage_counts": dict(counter),
        "stage_lookup_map": stage_map,
        "final_histogram": _histogram_with_names(incidents, "StageID", stage_map),
        "sample_incident_stages": [i.get("StageID") for i in incidents[:10]]
    }