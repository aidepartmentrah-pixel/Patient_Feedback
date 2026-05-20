"""
Workflow Activity Report Service

Orchestrates data fetching for the Section Workflow Activity Report.
Accepts date range + org scope, delegates to DB layer, returns shaped data dict.

NO Word generation here — formatting is in workflow_activity_word_formatter.py.
"""

from datetime import date, datetime
from typing import Dict, Any, Optional, List

from ..db_layer.workflow_activity_db import (
    get_workflow_activity_cases,
    expand_scope_to_unit_ids,
)


def build_workflow_activity_report(
    start_date: date,
    end_date: date,
    scope: str,
    administration_ids: Optional[List[int]],
    department_ids: Optional[List[int]],
    section_ids: Optional[List[int]],
    hospital_id: int = 1,
    generated_by: str = "System",
) -> Dict[str, Any]:
    """
    Build the data payload for a Workflow Activity report.

    Args:
        start_date:          Report start date (inclusive)
        end_date:            Report end date (inclusive)
        scope:               'hospital' | 'administration' | 'department' | 'section'
        administration_ids:  Selected administration IDs (None or [] means all)
        department_ids:      Selected department IDs (None or [] means all)
        section_ids:         Selected section IDs (None or [] means all)
        hospital_id:         Root hospital ID for scope expansion (default 1)
        generated_by:        Display name for report header

    Returns:
        {
            "meta": {
                "start_date": date,
                "end_date": date,
                "scope": str,
                "generated_at": date,
                "generated_by": str,
                "total_cases": int,
                "total_subcases": int,
                "total_action_items": int,
            },
            "cases": [
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
                                    "is_overdue": bool,
                                    "days_overdue": int | None,
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    """
    unit_ids = expand_scope_to_unit_ids(
        scope=scope,
        administration_ids=administration_ids,
        department_ids=department_ids,
        section_ids=section_ids,
        hospital_id=hospital_id,
    )

    cases = get_workflow_activity_cases(
        start_date=start_date,
        end_date=end_date,
        unit_ids=unit_ids,
    )

    today = date.today()
    total_subcases = 0
    total_action_items = 0

    for case in cases:
        total_subcases += len(case["subcases"])
        for subcase in case["subcases"]:
            for ai in subcase["action_items"]:
                total_action_items += 1
                # Compute overdue flag at service layer
                due = ai["due_date"]
                completed = ai["completed_at"]
                if due and completed is None and (due.date() if isinstance(due, datetime) else due) < today:
                    days = (today - (due.date() if isinstance(due, datetime) else due)).days
                    ai["is_overdue"] = True
                    ai["days_overdue"] = days
                else:
                    ai["is_overdue"] = False
                    ai["days_overdue"] = None

    return {
        "meta": {
            "start_date": start_date,
            "end_date": end_date,
            "scope": scope,
            "generated_at": today,
            "generated_by": generated_by,
            "total_cases": len(cases),
            "total_subcases": total_subcases,
            "total_action_items": total_action_items,
        },
        "cases": cases,
    }
