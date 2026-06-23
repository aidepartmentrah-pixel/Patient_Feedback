"""
Workflow Performance Report Service (HCAT Reporting Refactor - Session 1)

Orchestrates the unified Workflow Performance Report (Response Performance +
Current Delay) for the Workflow Page export. Pure aggregation layer - it
calls the existing get_response_performance_report() and
get_current_delay_report() exactly as the old Reporting Page cards do and
assembles a Unified Report Model. No new metrics, calculations, or business
logic are introduced here; formatting is in
workflow_performance_word_formatter.py.
"""

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from .response_performance_service import get_response_performance_report
from .current_delay_service import get_current_delay_report


def build_workflow_performance_report(
    scope_unit_ids: List[int],
    date_from: datetime,
    date_to_exclusive: datetime,
    level: Optional[str] = None,
    target_unit_id: Optional[int] = None,
    generated_by: str = "System",
) -> Dict[str, Any]:
    """
    Build the Unified Report Model for the Workflow Performance Report.

    Args:
        scope_unit_ids: org unit IDs the requesting user is allowed to see
            (already RBAC-intersected by the caller).
        date_from: inclusive lower bound, used only by the Response
            Performance section (Current Delay is a live/current-state
            report with no date range, same as the old card).
        date_to_exclusive: exclusive upper bound for the Response
            Performance section.
        level: "All" | "Administration" | "Department" | "Section".
        target_unit_id: restrict the report to a single org unit.
        generated_by: display name for the report footer.

    Returns:
        {
            "meta": {
                "date_from": date, "date_to": date, "level": str,
                "target_unit_id": int | None, "generated_at": datetime,
                "generated_by": str,
            },
            "response_performance": [...rows, same shape as
                get_response_performance_report()...],
            "current_delay": [...rows, same shape as
                get_current_delay_report()...],
        }
    """
    response_performance = get_response_performance_report(
        scope_unit_ids=scope_unit_ids,
        date_from=date_from,
        date_to_exclusive=date_to_exclusive,
        level=level,
        target_unit_id=target_unit_id,
    )

    current_delay = get_current_delay_report(
        scope_unit_ids=scope_unit_ids,
        level=level,
        target_unit_id=target_unit_id,
    )

    inclusive_date_to = date_to_exclusive - timedelta(days=1)

    return {
        "meta": {
            "date_from": date_from.date() if isinstance(date_from, datetime) else date_from,
            "date_to": inclusive_date_to.date() if isinstance(inclusive_date_to, datetime) else inclusive_date_to,
            "level": level or "All",
            "target_unit_id": target_unit_id,
            "generated_at": datetime.now(),
            "generated_by": generated_by,
        },
        "response_performance": response_performance,
        "current_delay": current_delay,
    }
