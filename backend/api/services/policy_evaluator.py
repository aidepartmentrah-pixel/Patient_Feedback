"""
Policy Evaluator — Computes the authoritative policy_snapshot.

This is the TRUTH engine for the reporting layer.
The reporting session renders this output; it never recomputes policy logic.

Section policy   → evaluated per HCAT Classification (ClassificationID).
Department policy → evaluated per domain (Clinical / Management / Relational).
Administration   → same as Department.
Hospital         → whole-hospital domain totals.

Failure formula (STRICTLY GREATER THAN):
    Section classification fails if:
        low_cases    > LowSeverityLimit    (low-severity incidents per classification)
        medium_cases > MediumSeverityLimit
        high_cases   > HighSeverityLimit

    Department / Administration fails if:
        clinical_cases    > ClinicalDomainLimit
        management_cases  > ManagementDomainLimit
        relational_cases  > RelationalDomainLimit
"""

from datetime import date
from typing import Any, Dict, List

from core.database import get_connection
from ..constants.org_unit_types import (
    ORG_TYPE_ADMINISTRATION,
    ORG_TYPE_DEPARTMENT,
    ORG_TYPE_SECTION,
)
from ..db_layer.admin_units import get_units_by_type
from ..db_layer.org_unit_policy import (
    HOSPITAL_POLICY_UNIT_ID,
    get_all_policy_rows,
    get_hospital_policy,
)
from ..db_layer.reports_db import build_org_filter_condition


# ──────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────

def compute_policy_snapshot(date_from: date, date_to: date) -> dict:
    """
    Evaluate all org units against their stored policies for [date_from, date_to].

    Returns the stable policy_snapshot contract dict consumed by the reporting
    session.  Keys: sections, departments, administrations, hospital.
    """
    policy_map = get_all_policy_rows()

    return {
        "date_from":       date_from.isoformat(),
        "date_to":         date_to.isoformat(),
        "sections":        _evaluate_sections(date_from, date_to, policy_map),
        "departments":     _evaluate_domain_level(
                               date_from, date_to, policy_map,
                               ORG_TYPE_DEPARTMENT, "department"),
        "administrations": _evaluate_domain_level(
                               date_from, date_to, policy_map,
                               ORG_TYPE_ADMINISTRATION, "administration"),
        "hospital":        _evaluate_hospital(date_from, date_to),
    }


# ──────────────────────────────────────────────────────────────
# Section evaluation — per HCAT classification
# ──────────────────────────────────────────────────────────────

def _evaluate_sections(
    date_from: date,
    date_to: date,
    policy_map: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    sections = get_units_by_type(ORG_TYPE_SECTION)
    results: List[Dict[str, Any]] = []

    for section in sections:
        uid  = section["id"]
        name = section["name"]
        pol  = policy_map.get(uid)

        low_limit    = pol["LowSeverityLimit"]    if pol else None
        medium_limit = pol["MediumSeverityLimit"] if pol else None
        high_limit   = pol["HighSeverityLimit"]   if pol else None

        classifications = _query_section_classification_counts(uid, date_from, date_to)

        for cls in classifications:
            low    = cls["low_cases"]
            medium = cls["medium_cases"]
            high   = cls["high_cases"]

            violated: List[str] = []
            if low_limit    is not None and low    > low_limit:
                violated.append(f"low>{low_limit}")
            if medium_limit is not None and medium > medium_limit:
                violated.append(f"medium>{medium_limit}")
            if high_limit   is not None and high   > high_limit:
                violated.append(f"high>{high_limit}")

            results.append({
                "section_id":              uid,
                "section_name":            name,
                "classification_id":       cls["classification_id"],
                "classification_name":     cls["classification_name"],
                "classification_name_en":  cls["classification_name_en"],
                "total_cases":             cls["total_cases"],
                "low_cases":               low,
                "medium_cases":            medium,
                "high_cases":              high,
                "low_limit":               low_limit,
                "medium_limit":            medium_limit,
                "high_limit":              high_limit,
                "is_violating":            len(violated) > 0,
                "violated_rules":          violated,
                "has_policy":              pol is not None,
            })

    return results


# ──────────────────────────────────────────────────────────────
# Department / Administration evaluation — per domain
# ──────────────────────────────────────────────────────────────

def _evaluate_domain_level(
    date_from: date,
    date_to: date,
    policy_map: Dict[int, Dict[str, Any]],
    unit_type: int,
    label: str,
) -> List[Dict[str, Any]]:
    units = get_units_by_type(unit_type)
    results: List[Dict[str, Any]] = []

    for unit in units:
        uid  = unit["id"]
        name = unit["name"]
        pol  = policy_map.get(uid)

        clin_limit = pol["ClinicalDomainLimit"]    if pol else None
        mgmt_limit = pol["ManagementDomainLimit"]  if pol else None
        rel_limit  = pol["RelationalDomainLimit"]  if pol else None

        counts = _query_unit_domain_counts(uid, unit_type, date_from, date_to)

        violated_domains: List[str] = []
        if clin_limit is not None and counts["clinical"]    > clin_limit:
            violated_domains.append("clinical")
        if mgmt_limit is not None and counts["management"]  > mgmt_limit:
            violated_domains.append("management")
        if rel_limit  is not None and counts["relational"]  > rel_limit:
            violated_domains.append("relational")

        results.append({
            f"{label}_id":          uid,
            f"{label}_name":        name,
            "clinical_cases":       counts["clinical"],
            "management_cases":     counts["management"],
            "relational_cases":     counts["relational"],
            "clinical_limit":       clin_limit,
            "management_limit":     mgmt_limit,
            "relational_limit":     rel_limit,
            "is_violating":         len(violated_domains) > 0,
            "violated_domains":     violated_domains,
            "has_policy":           pol is not None,
        })

    return results


# ──────────────────────────────────────────────────────────────
# Hospital evaluation
# ──────────────────────────────────────────────────────────────

def _evaluate_hospital(date_from: date, date_to: date) -> Dict[str, Any]:
    pol    = get_hospital_policy()
    counts = _query_hospital_domain_counts(date_from, date_to)

    return {
        "total_cases":       counts["total"],
        "clinical_cases":    counts["clinical"],
        "management_cases":  counts["management"],
        "relational_cases":  counts["relational"],
        "has_policy":        pol is not None,
        "policy": {
            "low_severity_limit":      pol["LowSeverityLimit"]    if pol else None,
            "medium_severity_limit":   pol["MediumSeverityLimit"] if pol else None,
            "high_severity_limit":     pol["HighSeverityLimit"]   if pol else None,
            "clinical_domain_limit":   pol["ClinicalDomainLimit"]   if pol else None,
            "management_domain_limit": pol["ManagementDomainLimit"] if pol else None,
            "relational_domain_limit": pol["RelationalDomainLimit"] if pol else None,
        } if pol else None,
    }


# ──────────────────────────────────────────────────────────────
# DB query helpers
# ──────────────────────────────────────────────────────────────

def _query_section_classification_counts(
    section_id: int,
    date_from: date,
    date_to: date,
) -> List[Dict[str, Any]]:
    """
    Count incidents per HCAT classification for a single section.
    Uses the same tree-aware target-department filter as monthly reports.
    Severity IDs: 1=Low, 2=Medium, 3=High.
    """
    org_filter = build_org_filter_condition(None, None, None, section_id)

    sql = f"""
        SELECT
            ic.ClassificationID,
            cl.Classification_AR  AS classification_name,
            cl.Classification_EN  AS classification_name_en,
            COUNT(DISTINCT ic.IncidentRequestCaseID) AS total_cases,
            COUNT(DISTINCT CASE WHEN ic.SeverityID = 1
                                THEN ic.IncidentRequestCaseID END) AS low_cases,
            COUNT(DISTINCT CASE WHEN ic.SeverityID = 2
                                THEN ic.IncidentRequestCaseID END) AS medium_cases,
            COUNT(DISTINCT CASE WHEN ic.SeverityID = 3
                                THEN ic.IncidentRequestCaseID END) AS high_cases
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
        ORDER BY ic.ClassificationID
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


def _query_unit_domain_counts(
    unit_id: int,
    unit_type: int,
    date_from: date,
    date_to: date,
) -> Dict[str, int]:
    """
    Count cases by domain (Clinical/Management/Relational) for a dept or admin.
    DomainID: 1=Clinical, 2=Management, 3=Relational.
    """
    if unit_type == ORG_TYPE_SECTION:
        org_filter = build_org_filter_condition(None, None, None, unit_id)
    elif unit_type == ORG_TYPE_DEPARTMENT:
        org_filter = build_org_filter_condition(None, None, unit_id, None)
    else:
        org_filter = build_org_filter_condition(None, unit_id, None, None)

    sql = f"""
        SELECT
            COUNT(DISTINCT CASE WHEN ic.DomainID = 1
                                THEN ic.IncidentRequestCaseID END) AS clinical_cases,
            COUNT(DISTINCT CASE WHEN ic.DomainID = 2
                                THEN ic.IncidentRequestCaseID END) AS management_cases,
            COUNT(DISTINCT CASE WHEN ic.DomainID = 3
                                THEN ic.IncidentRequestCaseID END) AS relational_cases
        FROM dbo.APP_IncidentCase ic
        INNER JOIN dbo.APP_IncidentCaseTargetDepartment td
            ON ic.IncidentRequestCaseID = td.IncidentRequestCaseID
        WHERE ic.FeedbackRecievedDate >= ?
          AND ic.FeedbackRecievedDate <= ?
          AND ic.DomainID IS NOT NULL
          AND {org_filter}
    """

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(sql, (date_from, date_to))
        row = cursor.fetchone()
        if not row:
            return {"clinical": 0, "management": 0, "relational": 0}
        return {
            "clinical":   row[0] or 0,
            "management": row[1] or 0,
            "relational": row[2] or 0,
        }
    finally:
        cursor.close()
        conn.close()


def _query_hospital_domain_counts(date_from: date, date_to: date) -> Dict[str, int]:
    """Count cases by domain for the entire hospital (no org filter)."""
    sql = """
        SELECT
            COUNT(DISTINCT ic.IncidentRequestCaseID) AS total_cases,
            COUNT(DISTINCT CASE WHEN ic.DomainID = 1
                                THEN ic.IncidentRequestCaseID END) AS clinical_cases,
            COUNT(DISTINCT CASE WHEN ic.DomainID = 2
                                THEN ic.IncidentRequestCaseID END) AS management_cases,
            COUNT(DISTINCT CASE WHEN ic.DomainID = 3
                                THEN ic.IncidentRequestCaseID END) AS relational_cases
        FROM dbo.APP_IncidentCase ic
        WHERE ic.FeedbackRecievedDate >= ?
          AND ic.FeedbackRecievedDate <= ?
          AND ic.DomainID IS NOT NULL
    """

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(sql, (date_from, date_to))
        row = cursor.fetchone()
        if not row:
            return {"total": 0, "clinical": 0, "management": 0, "relational": 0}
        return {
            "total":      row[0] or 0,
            "clinical":   row[1] or 0,
            "management": row[2] or 0,
            "relational": row[3] or 0,
        }
    finally:
        cursor.close()
        conn.close()
