"""
Action Log Classification Service (API V2)
Pure Python logic for grouping and computing action item metrics.

PHASE F — ACTION LOG REPORT
No database access. No business rules. Pure data transformation.

Classification rules (LOCKED BY DECISION BLOCK):
- COMPLETED: StatusCode IN ('DONE', 'VERIFIED')
- NOT COMPLETED: All other statuses (CANCELLED already excluded by DB)
- OVERDUE: DueDate < today AND CompletedAt IS NULL
"""

from typing import Dict, Any, List
from datetime import date


# ============================================================================
# CLASSIFICATION LOGIC
# ============================================================================

def classify_action_items(rows: List[Dict[str, Any]], today: date) -> Dict[str, Any]:
    """
    Classify flat action item rows into completed vs not completed groups.
    
    This is pure data transformation with no side effects.
    All rules are LOCKED by Phase F decision block.
    
    Classification Rules:
    - COMPLETED: status_code IN ('DONE', 'VERIFIED')
    - NOT COMPLETED: all other statuses
      (CANCELLED already excluded by DB query)
    
    Overdue Computation:
    - is_overdue = (due_date IS NOT NULL 
                    AND due_date < today 
                    AND completed_at IS NULL)
    - days_overdue = (today - due_date).days if overdue else None
    
    Sorting:
    - completed_items: order from DB (by DueDate)
    - not_completed_items:
        1. Overdue items first (by days_overdue descending)
        2. Not overdue items (by DueDate ascending, nulls last)
    
    Args:
        rows: Flat list of action item dicts from DB layer
              Expected fields: status_code, due_date, completed_at, etc.
        today: Current date for overdue computation
        
    Returns:
        Dict with structure:
        {
            "completed_items": list[dict],  # DONE/VERIFIED items
            "not_completed_items": list[dict],  # Everything else
            "totals": {
                "completed_count": int,
                "not_completed_count": int,
                "overdue_count": int
            }
        }
        
        Each item dict is augmented with:
        - is_overdue: bool
        - days_overdue: int | None
    """
    completed_items = []
    not_completed_items = []
    overdue_count = 0
    
    # Process each row
    for row in rows:
        # Create a copy to avoid mutating input
        item = dict(row)
        
        # Extract fields
        status_code = item.get("status_code", "").upper()
        due_date = item.get("due_date")
        completed_at = item.get("completed_at")
        
        # Compute overdue status
        is_overdue = False
        days_overdue = None
        
        if due_date is not None and completed_at is None:
            if due_date < today:
                is_overdue = True
                days_overdue = (today - due_date).days
                overdue_count += 1
        
        # Attach computed fields
        item["is_overdue"] = is_overdue
        item["days_overdue"] = days_overdue
        
        # Classify: DONE vs NOT DONE
        if status_code in ('DONE', 'VERIFIED'):
            completed_items.append(item)
        else:
            not_completed_items.append(item)
    
    # Sort not_completed_items:
    # 1. Overdue items first (most overdue first)
    # 2. Then not overdue items by due date ascending (nulls last)
    def sort_key(item):
        is_overdue = item["is_overdue"]
        days_overdue = item["days_overdue"]
        due_date = item.get("due_date")
        
        if is_overdue:
            # Overdue items: sort by days_overdue descending (most overdue first)
            # Use negative to get descending order
            return (0, -days_overdue if days_overdue else 0, due_date or date.max)
        else:
            # Not overdue: sort by due date ascending (nulls last)
            return (1, due_date or date.max, 0)
    
    not_completed_items.sort(key=sort_key)
    
    # Build response
    return {
        "completed_items": completed_items,
        "not_completed_items": not_completed_items,
        "totals": {
            "completed_count": len(completed_items),
            "not_completed_count": len(not_completed_items),
            "overdue_count": overdue_count
        }
    }


def is_action_item_done(status_code: str) -> bool:
    """
    Helper: Check if an action item status is considered "done".
    
    LOCKED RULE: DONE = status in ('DONE', 'VERIFIED')
    
    Args:
        status_code: Action item status code
        
    Returns:
        True if status is DONE or VERIFIED
    """
    return status_code.upper() in ('DONE', 'VERIFIED')


def is_action_item_overdue(due_date: date | None, completed_at: Any, today: date) -> bool:
    """
    Helper: Check if an action item is overdue.
    
    LOCKED RULE: overdue = (due_date < today AND completed_at IS NULL)
    
    Args:
        due_date: Action item due date (None if not set)
        completed_at: Completion timestamp (None if not completed)
        today: Current date
        
    Returns:
        True if overdue
    """
    if due_date is None:
        return False
    
    if completed_at is not None:
        return False
    
    return due_date < today
