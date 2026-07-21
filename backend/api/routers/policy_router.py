"""
Policy Router — 4-card organizational unit policy management API.

Cards:
    1. Hospital
    2. Sections      (all sections share identical standardized values)
    3. Departments   (all departments share identical standardized values)
    4. Administrations (all administrations share identical standardized values)

Loading:  GET  /api/org-policy/levels          → returns all 4 cards
Saving:   PUT  /api/org-policy/hospital        → hospital policy
          PUT  /api/org-policy/sections        → overwrites ALL section rows
          PUT  /api/org-policy/departments     → overwrites ALL department rows
          PUT  /api/org-policy/administrations → overwrites ALL administration rows
Evaluate: POST /api/org-policy/evaluate        → returns policy_snapshot for date range
Target:   GET  /api/org-policy/target          → scope-driven domain targets (Target Analysis, Session 2)
"""

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..dependencies.user_context import get_current_user
from ..schemas.auth_models import CurrentUser
from ..utils.guards import require_logged_in, require_software_admin, require_unit_in_scope
from ..services.policy_service import (
    get_org_level_policies,
    save_hospital_policy,
    save_sections_policy,
    save_departments_policy,
    save_administrations_policy,
    get_domain_targets_for_scope,
)
from ..services.policy_evaluator import compute_policy_snapshot
from ..services.section_compliance_service import get_section_compliance


router = APIRouter(prefix="/api/org-policy", tags=["Org Policy"])


# ──────────────────────────────────────────────────────────────
# Request models
# ──────────────────────────────────────────────────────────────

class SectionPolicyRequest(BaseModel):
    all_limit:    int = Field(..., ge=0, description="Low-severity incidents per HCAT classification (LowSeverityLimit)")
    medium_limit: int = Field(..., ge=0, description="Medium incidents per HCAT classification (MediumSeverityLimit)")
    high_limit:   int = Field(..., ge=0, description="High incidents per HCAT classification (HighSeverityLimit)")


class DomainPolicyRequest(BaseModel):
    clinical_domain_limit:    int = Field(..., ge=0)
    management_domain_limit:  int = Field(..., ge=0)
    relational_domain_limit:  int = Field(..., ge=0)


class HospitalPolicyRequest(BaseModel):
    low_severity_limit:       int = Field(..., ge=0)
    medium_severity_limit:    int = Field(..., ge=0)
    high_severity_limit:      int = Field(..., ge=0)
    clinical_domain_limit:    int = Field(..., ge=0)
    management_domain_limit:  int = Field(..., ge=0)
    relational_domain_limit:  int = Field(..., ge=0)


class EvaluateRequest(BaseModel):
    date_from: date
    date_to:   date


# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────

@router.get("/levels")
async def get_policy_levels(
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Load all 4 policy cards for the Settings page.
    Returns one representative active row per level.
    Falls back to safe defaults when no row exists.
    """
    require_logged_in(current_user)
    require_software_admin(current_user)
    try:
        return {"success": True, **get_org_level_policies()}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "fetch_failed", "message": str(e),
                    "message_ar": f"فشل جلب سياسات الوحدات التنظيمية: {str(e)}"},
        )


@router.put("/hospital")
async def update_hospital_policy(
    request: HospitalPolicyRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Save hospital-level policy (single global row)."""
    require_logged_in(current_user)
    require_software_admin(current_user)
    try:
        save_hospital_policy(
            low_severity_limit=request.low_severity_limit,
            medium_severity_limit=request.medium_severity_limit,
            high_severity_limit=request.high_severity_limit,
            clinical_domain_limit=request.clinical_domain_limit,
            management_domain_limit=request.management_domain_limit,
            relational_domain_limit=request.relational_domain_limit,
            user_id=current_user.user_id,
        )
        return {"success": True, "message": "Hospital policy saved"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "update_failed", "message": str(e),
                    "message_ar": f"فشل حفظ سياسة المستشفى: {str(e)}"},
        )


@router.put("/sections")
async def update_sections_policy(
    request: SectionPolicyRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Overwrite ALL section policy rows with identical standardized values.
    Missing rows are created automatically.

    Semantic mapping:
        all_limit    → LowSeverityLimit    (low-severity incidents per HCAT classification)
        medium_limit → MediumSeverityLimit (medium incidents per HCAT classification)
        high_limit   → HighSeverityLimit   (high incidents per HCAT classification)
    """
    require_logged_in(current_user)
    require_software_admin(current_user)
    try:
        count = save_sections_policy(
            all_limit=request.all_limit,
            medium_limit=request.medium_limit,
            high_limit=request.high_limit,
            user_id=current_user.user_id,
        )
        return {"success": True, "sections_updated": count}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "update_failed", "message": str(e),
                    "message_ar": f"فشل حفظ سياسة الأقسام: {str(e)}"},
        )


@router.put("/departments")
async def update_departments_policy(
    request: DomainPolicyRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Overwrite ALL department policy rows with identical standardized domain limits.
    Missing rows are created automatically.
    """
    require_logged_in(current_user)
    require_software_admin(current_user)
    try:
        count = save_departments_policy(
            clinical_limit=request.clinical_domain_limit,
            management_limit=request.management_domain_limit,
            relational_limit=request.relational_domain_limit,
            user_id=current_user.user_id,
        )
        return {"success": True, "departments_updated": count}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "update_failed", "message": str(e),
                    "message_ar": f"فشل حفظ سياسة الدوائر: {str(e)}"},
        )


@router.put("/administrations")
async def update_administrations_policy(
    request: DomainPolicyRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Overwrite ALL administration policy rows with identical standardized domain limits.
    Missing rows are created automatically.
    """
    require_logged_in(current_user)
    require_software_admin(current_user)
    try:
        count = save_administrations_policy(
            clinical_limit=request.clinical_domain_limit,
            management_limit=request.management_domain_limit,
            relational_limit=request.relational_domain_limit,
            user_id=current_user.user_id,
        )
        return {"success": True, "administrations_updated": count}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "update_failed", "message": str(e),
                    "message_ar": f"فشل حفظ سياسة الإدارات: {str(e)}"},
        )


@router.post("/evaluate")
async def evaluate_policy_snapshot(
    request: EvaluateRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Compute the policy_snapshot for the given date range.

    Returns the contract object consumed by the reporting session:
    {
        "date_from": "...",
        "date_to":   "...",
        "sections":        [...],
        "departments":     [...],
        "administrations": [...],
        "hospital":        {...}
    }

    Each section entry covers one (section × HCAT classification) pair.
    Each department/administration entry covers domain-level counts.
    """
    require_logged_in(current_user)
    try:
        snapshot = compute_policy_snapshot(request.date_from, request.date_to)
        return {"success": True, **snapshot}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "evaluation_failed", "message": str(e),
                    "message_ar": f"فشل تقييم سياسة الامتثال: {str(e)}"},
        )


@router.get("/target")
async def get_domain_targets(
    scope: str = Query(..., description="hospital | administration | department | section"),
    administration_id: int | None = Query(None),
    department_id: int | None = Query(None),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Scope-driven domain target retrieval for Target Analysis (Session 2).

    Returns raw incident-count ceilings (Clinical/Management/Relational),
    not percentages — the schema has no percentage-of-total field.
    Section scope always returns has_domain_targets=false: section policy
    is classification-based, not domain-based, and must never show domain
    target lines (mandatory rule, Session 2).
    """
    require_logged_in(current_user)

    if scope not in {"hospital", "administration", "department", "section"}:
        raise HTTPException(status_code=400, detail="Invalid scope")

    if scope == "administration" and administration_id is not None:
        require_unit_in_scope(current_user, administration_id)

    if scope == "department" and department_id is not None:
        require_unit_in_scope(current_user, department_id)

    try:
        targets = get_domain_targets_for_scope(scope, administration_id, department_id)
        return {"success": True, "scope": scope, **targets}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "fetch_failed", "message": str(e),
                    "message_ar": f"فشل جلب أهداف النطاق: {str(e)}"},
        )


@router.get("/section-compliance")
async def get_section_compliance_endpoint(
    scope: str = Query(..., description="section | hospital"),
    start_date: str = Query(..., description="Start date YYYY-MM-DD"),
    end_date: str = Query(..., description="End date YYYY-MM-DD"),
    section_id: int | None = Query(None, description="Required when scope=section"),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Section Compliance Analysis — HCAT Iteration 4, Session 3.

    scope=section  → compliance for one specific section.
    scope=hospital → all sections aggregated by classification (Option B).

    Only section and hospital scopes are valid here.
    Administration and Department scopes do not use classification-based policy.
    """
    require_logged_in(current_user)

    if scope not in {"section", "hospital"}:
        raise HTTPException(status_code=400, detail="scope must be 'section' or 'hospital'")

    if scope == "section":
        if section_id is None:
            raise HTTPException(status_code=400, detail="section_id required for section scope")
        require_unit_in_scope(current_user, section_id)

    try:
        date_from = date.fromisoformat(start_date)
        date_to   = date.fromisoformat(end_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Dates must be in YYYY-MM-DD format")

    try:
        result = get_section_compliance(
            scope=scope,
            date_from=date_from,
            date_to=date_to,
            section_id=section_id,
        )
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "compliance_fetch_failed", "message": str(e)},
        )
