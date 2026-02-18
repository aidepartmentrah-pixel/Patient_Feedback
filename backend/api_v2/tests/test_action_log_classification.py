"""
📋 PHASE F — TEST F-B4 — ACTION LOG CLASSIFICATION SERVICE

Pure unit tests for action item classification logic.
NO database dependency - uses synthetic data.

Tests verify:
- DONE vs NOT DONE grouping
- Overdue computation
- Sorting rules
- Totals calculation
"""

import pytest
from datetime import date, datetime
from backend.api_v2.services.action_log_classification_service import (
    classify_action_items,
    is_action_item_done,
    is_action_item_overdue
)


# ============================================================================
# TEST 1 — DONE GROUPING
# ============================================================================

def test_done_grouping():
    """
    Test that DONE and VERIFIED statuses are classified as completed.
    """
    today = date(2026, 2, 6)
    
    rows = [
        {
            "action_item_id": 1,
            "title": "Task 1",
            "status_code": "DONE",
            "due_date": date(2026, 2, 1),
            "completed_at": datetime(2026, 2, 1, 10, 0, 0)
        },
        {
            "action_item_id": 2,
            "title": "Task 2",
            "status_code": "VERIFIED",
            "due_date": date(2026, 2, 2),
            "completed_at": datetime(2026, 2, 2, 15, 0, 0)
        }
    ]
    
    result = classify_action_items(rows, today)
    
    # Both should be in completed_items
    assert len(result["completed_items"]) == 2
    assert len(result["not_completed_items"]) == 0
    
    # Check they're the right items
    completed_ids = [item["action_item_id"] for item in result["completed_items"]]
    assert 1 in completed_ids
    assert 2 in completed_ids
    
    # Check totals
    assert result["totals"]["completed_count"] == 2
    assert result["totals"]["not_completed_count"] == 0
    
    print("✅ DONE and VERIFIED correctly grouped as completed")


# ============================================================================
# TEST 2 — NOT DONE GROUPING
# ============================================================================

def test_not_done_grouping():
    """
    Test that non-terminal statuses are classified as not completed.
    """
    today = date(2026, 2, 6)
    
    rows = [
        {
            "action_item_id": 10,
            "title": "In Progress Task",
            "status_code": "IN_PROGRESS",
            "due_date": date(2026, 2, 10),
            "completed_at": None
        },
        {
            "action_item_id": 11,
            "title": "Draft Task",
            "status_code": "DRAFT",
            "due_date": date(2026, 2, 15),
            "completed_at": None
        },
        {
            "action_item_id": 12,
            "title": "Admin Approved Task",
            "status_code": "ADMIN_APPROVED",
            "due_date": date(2026, 2, 20),
            "completed_at": None
        }
    ]
    
    result = classify_action_items(rows, today)
    
    # All should be in not_completed_items
    assert len(result["completed_items"]) == 0
    assert len(result["not_completed_items"]) == 3
    
    # Check totals
    assert result["totals"]["completed_count"] == 0
    assert result["totals"]["not_completed_count"] == 3
    
    print("✅ Non-terminal statuses correctly grouped as not completed")


# ============================================================================
# TEST 3 — OVERDUE DETECTION
# ============================================================================

def test_overdue_detection():
    """
    Test that overdue status is correctly computed.
    DueDate < today AND CompletedAt NULL -> overdue
    """
    today = date(2026, 2, 6)
    
    rows = [
        {
            "action_item_id": 20,
            "title": "Overdue Task",
            "status_code": "IN_PROGRESS",
            "due_date": date(2026, 2, 1),  # 5 days ago
            "completed_at": None
        }
    ]
    
    result = classify_action_items(rows, today)
    
    # Should be in not_completed_items (status is IN_PROGRESS)
    assert len(result["not_completed_items"]) == 1
    
    item = result["not_completed_items"][0]
    
    # Check overdue computation
    assert item["is_overdue"] is True
    assert item["days_overdue"] == 5
    
    # Check totals
    assert result["totals"]["overdue_count"] == 1
    
    print("✅ Overdue status correctly computed: 5 days overdue")


# ============================================================================
# TEST 4 — NOT OVERDUE WHEN COMPLETED
# ============================================================================

def test_not_overdue_when_completed():
    """
    Test that completed items are NOT marked overdue even if past due date.
    """
    today = date(2026, 2, 6)
    
    rows = [
        {
            "action_item_id": 30,
            "title": "Completed Late Task",
            "status_code": "DONE",
            "due_date": date(2026, 2, 1),  # Past due
            "completed_at": datetime(2026, 2, 5, 10, 0, 0)  # But completed
        }
    ]
    
    result = classify_action_items(rows, today)
    
    # Should be in completed_items
    assert len(result["completed_items"]) == 1
    
    item = result["completed_items"][0]
    
    # Should NOT be overdue (because completed)
    assert item["is_overdue"] is False
    assert item["days_overdue"] is None
    
    # Check totals
    assert result["totals"]["overdue_count"] == 0
    
    print("✅ Completed items not marked as overdue")


# ============================================================================
# TEST 5 — SORTING RULE
# ============================================================================

def test_not_completed_sorting():
    """
    Test that not_completed_items are sorted correctly:
    1. Overdue items first (most overdue first)
    2. Then not overdue items by due date ascending
    """
    today = date(2026, 2, 6)
    
    rows = [
        {
            "action_item_id": 40,
            "title": "Due Soon",
            "status_code": "IN_PROGRESS",
            "due_date": date(2026, 2, 8),  # Not overdue, due in 2 days
            "completed_at": None
        },
        {
            "action_item_id": 41,
            "title": "Overdue 10 days",
            "status_code": "DRAFT",
            "due_date": date(2026, 1, 27),  # Overdue by 10 days
            "completed_at": None
        },
        {
            "action_item_id": 42,
            "title": "Due Later",
            "status_code": "ADMIN_APPROVED",
            "due_date": date(2026, 2, 20),  # Not overdue, due in 14 days
            "completed_at": None
        },
        {
            "action_item_id": 43,
            "title": "Overdue 3 days",
            "status_code": "IN_PROGRESS",
            "due_date": date(2026, 2, 3),  # Overdue by 3 days
            "completed_at": None
        }
    ]
    
    result = classify_action_items(rows, today)
    
    # All should be not completed
    assert len(result["not_completed_items"]) == 4
    
    # Check order: overdue first (most overdue), then by due date
    sorted_items = result["not_completed_items"]
    
    # Expected order:
    # 1. ID 41 (overdue 10 days)
    # 2. ID 43 (overdue 3 days)
    # 3. ID 40 (due 2026-02-08)
    # 4. ID 42 (due 2026-02-20)
    
    assert sorted_items[0]["action_item_id"] == 41, "Most overdue should be first"
    assert sorted_items[0]["days_overdue"] == 10
    
    assert sorted_items[1]["action_item_id"] == 43, "Second most overdue"
    assert sorted_items[1]["days_overdue"] == 3
    
    assert sorted_items[2]["action_item_id"] == 40, "Earlier due date"
    assert sorted_items[2]["is_overdue"] is False
    
    assert sorted_items[3]["action_item_id"] == 42, "Later due date"
    assert sorted_items[3]["is_overdue"] is False
    
    print("✅ Sorting correct: overdue first (10d, 3d), then by due date")


# ============================================================================
# TEST 6 — TOTALS CORRECT
# ============================================================================

def test_totals_calculation():
    """
    Test that totals are correctly computed.
    """
    today = date(2026, 2, 6)
    
    rows = [
        # 2 completed
        {
            "action_item_id": 50,
            "status_code": "DONE",
            "due_date": date(2026, 2, 1),
            "completed_at": datetime(2026, 2, 1, 10, 0, 0)
        },
        {
            "action_item_id": 51,
            "status_code": "VERIFIED",
            "due_date": date(2026, 2, 2),
            "completed_at": datetime(2026, 2, 2, 15, 0, 0)
        },
        # 3 not completed (2 overdue, 1 not overdue)
        {
            "action_item_id": 52,
            "status_code": "IN_PROGRESS",
            "due_date": date(2026, 2, 1),  # Overdue
            "completed_at": None
        },
        {
            "action_item_id": 53,
            "status_code": "DRAFT",
            "due_date": date(2026, 2, 3),  # Overdue
            "completed_at": None
        },
        {
            "action_item_id": 54,
            "status_code": "ADMIN_APPROVED",
            "due_date": date(2026, 2, 10),  # Not overdue
            "completed_at": None
        }
    ]
    
    result = classify_action_items(rows, today)
    
    # Check counts
    assert result["totals"]["completed_count"] == 2
    assert result["totals"]["not_completed_count"] == 3
    assert result["totals"]["overdue_count"] == 2
    
    # Verify list counts match totals
    assert len(result["completed_items"]) == result["totals"]["completed_count"]
    assert len(result["not_completed_items"]) == result["totals"]["not_completed_count"]
    
    print("✅ Totals correctly calculated: 2 completed, 3 not completed, 2 overdue")


# ============================================================================
# TEST 7 — NULL DUE DATE HANDLING
# ============================================================================

def test_null_due_date_handling():
    """
    Test that items with NULL due date are NOT marked overdue.
    """
    today = date(2026, 2, 6)
    
    rows = [
        {
            "action_item_id": 60,
            "title": "No Due Date",
            "status_code": "IN_PROGRESS",
            "due_date": None,  # NULL due date
            "completed_at": None
        }
    ]
    
    result = classify_action_items(rows, today)
    
    # Should be in not_completed_items
    assert len(result["not_completed_items"]) == 1
    
    item = result["not_completed_items"][0]
    
    # Should NOT be overdue
    assert item["is_overdue"] is False
    assert item["days_overdue"] is None
    
    # Check totals
    assert result["totals"]["overdue_count"] == 0
    
    print("✅ NULL due date correctly handled (not overdue)")


# ============================================================================
# TEST 8 — HELPER FUNCTIONS
# ============================================================================

def test_is_action_item_done_helper():
    """
    Test the is_action_item_done helper function.
    """
    assert is_action_item_done("DONE") is True
    assert is_action_item_done("VERIFIED") is True
    assert is_action_item_done("done") is True  # Case insensitive
    assert is_action_item_done("verified") is True
    
    assert is_action_item_done("IN_PROGRESS") is False
    assert is_action_item_done("DRAFT") is False
    assert is_action_item_done("ADMIN_APPROVED") is False
    
    print("✅ is_action_item_done helper works correctly")


def test_is_action_item_overdue_helper():
    """
    Test the is_action_item_overdue helper function.
    """
    today = date(2026, 2, 6)
    
    # Past due, not completed -> overdue
    assert is_action_item_overdue(date(2026, 2, 1), None, today) is True
    
    # Past due, but completed -> not overdue
    assert is_action_item_overdue(
        date(2026, 2, 1), 
        datetime(2026, 2, 5, 10, 0, 0), 
        today
    ) is False
    
    # Future due -> not overdue
    assert is_action_item_overdue(date(2026, 2, 10), None, today) is False
    
    # NULL due date -> not overdue
    assert is_action_item_overdue(None, None, today) is False
    
    print("✅ is_action_item_overdue helper works correctly")


# ============================================================================
# TEST 9 — EMPTY INPUT
# ============================================================================

def test_empty_input():
    """
    Test that empty input returns empty lists and zero totals.
    """
    today = date(2026, 2, 6)
    rows = []
    
    result = classify_action_items(rows, today)
    
    assert len(result["completed_items"]) == 0
    assert len(result["not_completed_items"]) == 0
    assert result["totals"]["completed_count"] == 0
    assert result["totals"]["not_completed_count"] == 0
    assert result["totals"]["overdue_count"] == 0
    
    print("✅ Empty input handled correctly")


# ============================================================================
# TEST 10 — MIXED SCENARIO
# ============================================================================

def test_mixed_scenario():
    """
    Test a realistic mixed scenario with various statuses and dates.
    """
    today = date(2026, 2, 6)
    
    rows = [
        # Completed items
        {"action_item_id": 100, "status_code": "DONE", "due_date": date(2026, 1, 15), "completed_at": datetime(2026, 1, 14, 10, 0, 0)},
        {"action_item_id": 101, "status_code": "VERIFIED", "due_date": date(2026, 1, 20), "completed_at": datetime(2026, 1, 19, 15, 0, 0)},
        
        # Not completed - overdue
        {"action_item_id": 102, "status_code": "IN_PROGRESS", "due_date": date(2026, 1, 25), "completed_at": None},
        {"action_item_id": 103, "status_code": "DRAFT", "due_date": date(2026, 2, 1), "completed_at": None},
        
        # Not completed - not overdue
        {"action_item_id": 104, "status_code": "ADMIN_APPROVED", "due_date": date(2026, 2, 15), "completed_at": None},
        {"action_item_id": 105, "status_code": "IN_PROGRESS", "due_date": date(2026, 2, 20), "completed_at": None},
        
        # Not completed - null due date
        {"action_item_id": 106, "status_code": "DRAFT", "due_date": None, "completed_at": None}
    ]
    
    result = classify_action_items(rows, today)
    
    # Check totals
    assert result["totals"]["completed_count"] == 2
    assert result["totals"]["not_completed_count"] == 5
    assert result["totals"]["overdue_count"] == 2
    
    # Verify overdue items are at the top of not_completed list
    not_completed = result["not_completed_items"]
    assert not_completed[0]["is_overdue"] is True  # ID 102 (12 days overdue)
    assert not_completed[1]["is_overdue"] is True  # ID 103 (5 days overdue)
    assert not_completed[2]["is_overdue"] is False  # ID 104 or 105
    
    print("✅ Mixed scenario processed correctly")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
