"""
Section Compliance Service — HCAT Iteration 4, Session 3.

Computes classification-level compliance for:
  - Section scope: one specific section vs its own policy targets
  - Hospital scope: all sections aggregated per classification (Option B),
    compared against the shared standardized section target

Targets come from APP_OrgUnitPolicy (LowSeverityLimit = low-severity target,
MediumSeverityLimit = medium target, HighSeverityLimit = high target).
All sections share identical targets due to the bulk-save convention
(confirmed in Session 0 discovery).

Date filtering uses FeedbackRecievedDate with an inclusive range,
consistent with policy_evaluator.py.
"""

from datetime import date
from typing import Any, Dict, List, Optional

from core.database import get_connection
from ..constants.org_unit_types import ORG_TYPE_SECTION
from ..db_layer.org_unit_policy import (
    get_policy_by_unit_id,
    get_representative_policy_for_type,
)
from ..db_layer.reports_db import build_org_filter_condition


# ──────────────────────────────────────────────────────────────
# Compliance status engine (server-side single source of truth)
# ──────────────────────────────────────────────────────────────

def _compliance_status(
    low: int,
    medium: int,
    high: int,
    low_limit: Optional[int],
    medium_limit: Optional[int],
    high_limit: Optional[int],
) -> str:
    """
    Binary status — any of the three metrics (Low/Medium/High) exceeding
    its target is a Violation. Which specific metric(s) failed is derived
    by the frontend directly from the actual/target pairs already present
    in each row (no separate severity-tier vocabulary surfaced to users).
    """
    high_violated   = high_limit   is not None and high   > high_limit
    medium_violated = medium_limit is not None and medium > medium_limit
    low_violated    = low_limit    is not None and low    > low_limit

    if high_violated or medium_violated or low_violated:
        return "Violation"
    return "Compliant"


# ──────────────────────────────────────────────────────────────
# DB queries
# ──────────────────────────────────────────────────────────────

def _query_single_section(
    section_id: int,
    date_from: date,
    date_to: date,
) -> List[Dict[str, Any]]:
    """Classification counts for one section (target-department filtered)."""
    org_filter = build_org_filter_condition(None, None, None, section_id)
    sql = f"""
        SELECT
            ic.ClassificationID,
            cl.Classification_AR  AS classification_name,
            cl.Classification_EN  AS classification_name_en,
            COUNT(DISTINCT ic.IncidentRequestCaseID)                              AS total_cases,
            COUNT(DISTINCT CASE WHEN ic.SeverityID = 1
                                THEN ic.IncidentRequestCaseID END)                AS low_cases,
            COUNT(DISTINCT CASE WHEN ic.SeverityID = 2
                                THEN ic.IncidentRequestCaseID END)                AS medium_cases,
            COUNT(DISTINCT CASE WHEN ic.SeverityID = 3
                                THEN ic.IncidentRequestCaseID END)                AS high_cases
        FROM dbo.APP_IncidentCase ic
        INNER JOIN dbo.APP_IncidentCaseTargetDepartment td
            ON ic.IncidentRequestCaseID = td.IncidentRequestCaseID
        LEFT JOIN dbo.APP_LOOKUP_CLASSIFICATION cl
            ON ic.ClassificationID = cl.ClassificationID
        WHERE ic.FeedbackRecievedDate >= ?
          AND ic.FeedbackRecievedDate <= ?
          AND ic.ClassificationID IS NOT NULL
          AND {org_filter}
        GROUP BY ic.ClassificationID, cl.Classification_AR, cl.Classification_EN
        ORDER BY total_cases DESC
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(sql, (date_from, date_to))
        rows = cursor.fetchall()
        return [
            {
                "classification_id":      row[0],
                "classification_name":    row[1] or "",
                "classification_name_en": row[2] or "",
                "total_cases":            row[3] or 0,
                "low_cases":              row[4] or 0,
                "medium_cases":           row[5] or 0,
                "high_cases":             row[6] or 0,
            }
            for row in rows
        ]
    finally:
        cursor.close()
        conn.close()


def _query_hospital_all_sections(
    date_from: date,
    date_to: date,
) -> List[Dict[str, Any]]:
    """
    Classification counts aggregated across ALL sections hospital-wide
    (Option B — one row per classification, not per section).
    Filters target departments to section-type units only (Type = 324).
    """
    sql = """
        SELECT
            ic.ClassificationID,
            cl.Classification_AR  AS classification_name,
            cl.Classification_EN  AS classification_name_en,
            COUNT(DISTINCT ic.IncidentRequestCaseID)                              AS total_cases,
            COUNT(DISTINCT CASE WHEN ic.SeverityID = 1
                                THEN ic.IncidentRequestCaseID END)                AS low_cases,
            COUNT(DISTINCT CASE WHEN ic.SeverityID = 2
                                THEN ic.IncidentRequestCaseID END)                AS medium_cases,
            COUNT(DISTINCT CASE WHEN ic.SeverityID = 3
                                THEN ic.IncidentRequestCaseID END)                AS high_cases
        FROM dbo.APP_IncidentCase ic
        INNER JOIN dbo.APP_IncidentCaseTargetDepartment td
            ON ic.IncidentRequestCaseID = td.IncidentRequestCaseID
        INNER JOIN dbo.AdminsrationUnit au
            ON td.DepartmentID = au.UniqueID
        LEFT JOIN dbo.APP_LOOKUP_CLASSIFICATION cl
            ON ic.ClassificationID = cl.ClassificationID
        WHERE ic.FeedbackRecievedDate >= ?
          AND ic.FeedbackRecievedDate <= ?
          AND ic.ClassificationID IS NOT NULL
          AND au.Type = 324
        GROUP BY ic.ClassificationID, cl.Classification_AR, cl.Classification_EN
        ORDER BY total_cases DESC
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(sql, (date_from, date_to))
        rows = cursor.fetchall()
        return [
            {
                "classification_id":      row[0],
                "classification_name":    row[1] or "",
                "classification_name_en": row[2] or "",
                "total_cases":            row[3] or 0,
                "low_cases":              row[4] or 0,
                "medium_cases":           row[5] or 0,
                "high_cases":             row[6] or 0,
            }
            for row in rows
        ]
    finally:
        cursor.close()
        conn.close()


# ──────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────

def get_section_compliance(
    scope: str,
    date_from: date,
    date_to: date,
    section_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Return compliance data for the Section Compliance module.

    scope=section  → one section's classifications vs its own policy targets.
    scope=hospital → all sections aggregated per classification (Option B),
                     compared against the shared standardized section target.

    Returns:
        {
            "scope": str,
            "date_from": str,
            "date_to": str,
            "low_limit": int | None,
            "medium_limit": int | None,
            "high_limit": int | None,
            "has_policy": bool,
            "rows": [ { classification, actuals, targets, status } ]
        }
    """
    # Load policy targets
    if scope == "section" and section_id is not None:
        policy = get_policy_by_unit_id(section_id)
    else:
        # Hospital scope: use the shared standardized section target
        policy = get_representative_policy_for_type(ORG_TYPE_SECTION)

    low_limit    = policy.get("LowSeverityLimit")    if policy else None
    medium_limit = policy.get("MediumSeverityLimit") if policy else None
    high_limit   = policy.get("HighSeverityLimit")   if policy else None
    has_policy   = policy is not None

    # Load actuals
    if scope == "section" and section_id is not None:
        raw = _query_single_section(section_id, date_from, date_to)
    else:
        raw = _query_hospital_all_sections(date_from, date_to)

    # Build output rows with compliance status
    rows = []
    for r in raw:
        status = _compliance_status(
            r["low_cases"],  r["medium_cases"],  r["high_cases"],
            low_limit,       medium_limit,        high_limit,
        )
        rows.append({
            "classification_id":      r["classification_id"],
            "classification_name":    r["classification_name"],
            "classification_name_en": r["classification_name_en"],
            "total_actual":           r["total_cases"],
            "low_actual":             r["low_cases"],
            "low_target":             low_limit,
            "medium_actual":          r["medium_cases"],
            "medium_target":          medium_limit,
            "high_actual":            r["high_cases"],
            "high_target":            high_limit,
            "compliance_status":      status,
        })

    return {
        "scope":         scope,
        "date_from":     date_from.isoformat(),
        "date_to":       date_to.isoformat(),
        "low_limit":     low_limit,
        "medium_limit":  medium_limit,
        "high_limit":    high_limit,
        "has_policy":    has_policy,
        "rows":          rows,
    }
