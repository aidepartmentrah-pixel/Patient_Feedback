from datetime import date, timedelta
from fastapi import APIRouter, Query, HTTPException
import traceback
from ..services.dashboard_service import get_dashboard_stats, get_dashboard_hierarchy

router = APIRouter(
    prefix="/api/dashboard",
    tags=["Dashboard"],
)

# =========================================================
# DASHBOARD STATS ENDPOINT
# =========================================================

@router.get("/stats")
def dashboard_stats(
    scope: str = Query(..., description="hospital | administration | department | section"),
    administration_id: int | None = Query(None),
    department_id: int | None = Query(None),
    section_id: int | None = Query(None),
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
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
            scope=scope,
            administration_id=administration_id,
            department_id=department_id,
            section_id=section_id,
            start_date=start_date,
            end_date=end_date,
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
def dashboard_hierarchy():
    return get_dashboard_hierarchy()
