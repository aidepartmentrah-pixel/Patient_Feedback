"""
Workflow Activity Report Database Layer

Queries for Section Workflow Activity Report:
- Fetches cases with their subcases (responses/explanations) and action items
- Filtered by date range and organizational scope
- NO business logic — pure SQL
"""

from typing import List, Dict, Any, Optional
from datetime import date
from core.database import get_connection
from .reports_db import get_org_unit_descendants


def get_workflow_activity_cases(
    start_date: date,
    end_date: date,
    unit_ids: List[int],
) -> List[Dict[str, Any]]:
    """
    Fetch cases with their subcases and action items for the Workflow Activity report.

    Args:
        start_date: Inclusive start date (filters on APP_IncidentCase.FeedbackRecievedDate)
        end_date:   Inclusive end date
        unit_ids:   Expanded list of org unit IDs to include (already scope-expanded by caller)

    Returns:
        List of case dicts. Each dict has:
        {
            "case_id": int,
            "patient_name": str | None,
            "feedback_date": date | None,
            "complaint_text": str | None,
            "subcases": [
                {
                    "subcase_id": int,
                    "status": str,
                    "target_org_unit_id": int,
                    "target_org_unit_name": str | None,
                    "section_explanation": str | None,
                    "department_explanation": str | None,
                    "administration_explanation": str | None,
                    "action_items": [
                        {
                            "action_item_id": int,
                            "title": str,
                            "description": str | None,
                            "due_date": date | None,
                            "status": str,
                            "completed_at": datetime | None,
                        }
                    ]
                }
            ]
        }
    """
    if not unit_ids:
        return []

    conn = get_connection()
    cursor = conn.cursor()

    try:
        placeholders = ",".join("?" * len(unit_ids))

        # Step 1: Fetch cases in scope and date range
        case_query = f"""
            SELECT
                c.IncidentRequestCaseID,
                c.PatientName,
                CAST(c.FeedbackRecievedDate AS DATE) AS FeedbackDate,
                c.ComplaintText
            FROM dbo.APP_IncidentCase c
            WHERE
                c.IssuingOrgUnitID IN ({placeholders})
                AND CAST(c.FeedbackRecievedDate AS DATE) BETWEEN ? AND ?
            ORDER BY c.FeedbackRecievedDate DESC
        """
        params = unit_ids + [start_date, end_date]
        cursor.execute(case_query, params)
        case_rows = cursor.fetchall()

        if not case_rows:
            return []

        case_ids = [row.IncidentRequestCaseID for row in case_rows]

        # Step 2: Fetch subcases for all retrieved cases
        sc_placeholders = ",".join("?" * len(case_ids))
        subcase_query = f"""
            SELECT
                s.SubcaseID,
                s.IncidentRequestCaseID,
                s.Status,
                s.TargetOrgUnitID,
                org.Name AS TargetOrgUnitName,
                s.SectionExplanationText,
                s.DepartmentExplanationText,
                s.AdministrationExplanationText
            FROM dbo.APP_AdministrativeSubcase s
            LEFT JOIN dbo.AdminsrationUnit org ON s.TargetOrgUnitID = org.UniqueID
            WHERE s.IncidentRequestCaseID IN ({sc_placeholders})
            ORDER BY s.SubcaseID ASC
        """
        cursor.execute(subcase_query, case_ids)
        subcase_rows = cursor.fetchall()

        subcase_ids = [row.SubcaseID for row in subcase_rows]

        # Step 3: Fetch action items for all retrieved subcases (if any)
        action_items_by_subcase: Dict[int, List[Dict]] = {}
        if subcase_ids:
            ai_placeholders = ",".join("?" * len(subcase_ids))
            action_item_query = f"""
                SELECT
                    ai.ActionItemID,
                    ai.SubcaseID,
                    ai.Title,
                    ai.Description,
                    ai.DueDate,
                    ai.Status,
                    ai.CompletedAt
                FROM dbo.APP_SubcaseActionItem ai
                WHERE ai.SubcaseID IN ({ai_placeholders})
                  AND ai.Status != 'CANCELLED'
                ORDER BY ai.DueDate ASC
            """
            cursor.execute(action_item_query, subcase_ids)
            for ai_row in cursor.fetchall():
                bucket = action_items_by_subcase.setdefault(ai_row.SubcaseID, [])
                bucket.append({
                    "action_item_id": ai_row.ActionItemID,
                    "title": ai_row.Title,
                    "description": ai_row.Description,
                    "due_date": ai_row.DueDate,
                    "status": ai_row.Status,
                    "completed_at": ai_row.CompletedAt,
                })

        # Step 4: Group subcases by case_id
        subcases_by_case: Dict[int, List[Dict]] = {}
        for sc in subcase_rows:
            bucket = subcases_by_case.setdefault(sc.IncidentRequestCaseID, [])
            bucket.append({
                "subcase_id": sc.SubcaseID,
                "status": sc.Status,
                "target_org_unit_id": sc.TargetOrgUnitID,
                "target_org_unit_name": sc.TargetOrgUnitName,
                "section_explanation": sc.SectionExplanationText,
                "department_explanation": sc.DepartmentExplanationText,
                "administration_explanation": sc.AdministrationExplanationText,
                "action_items": action_items_by_subcase.get(sc.SubcaseID, []),
            })

        # Step 5: Assemble final result
        result = []
        for row in case_rows:
            result.append({
                "case_id": row.IncidentRequestCaseID,
                "patient_name": row.PatientName,
                "feedback_date": row.FeedbackDate,
                "complaint_text": row.ComplaintText,
                "subcases": subcases_by_case.get(row.IncidentRequestCaseID, []),
            })

        return result

    finally:
        cursor.close()
        conn.close()


def expand_scope_to_unit_ids(
    scope: str,
    administration_ids: Optional[List[int]],
    department_ids: Optional[List[int]],
    section_ids: Optional[List[int]],
    hospital_id: int = 1,
) -> List[int]:
    """
    Resolve scope selection to a flat list of org unit IDs for DB filtering.

    For hospital scope: returns all descendants of hospital_id.
    For admin/dept/section with specific IDs: expands each to include descendants.
    For admin/dept/section with no IDs (all): expands from hospital root.
    """
    if scope == "hospital":
        return get_org_unit_descendants(hospital_id)

    target_ids: List[int] = []

    if scope == "administration":
        roots = administration_ids or []
    elif scope == "department":
        roots = department_ids or []
    elif scope == "section":
        roots = section_ids or []
    else:
        roots = []

    if not roots:
        # "all" within scope — expand from hospital root
        return get_org_unit_descendants(hospital_id)

    expanded: set = set()
    for uid in roots:
        expanded.update(get_org_unit_descendants(uid))
    return list(expanded)
