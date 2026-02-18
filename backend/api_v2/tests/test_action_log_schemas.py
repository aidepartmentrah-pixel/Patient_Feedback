"""
📋 PHASE F — TEST F-B1 — ACTION LOG CONTRACT TESTS (API V2)

Unit tests for Action Log Report schema contract validation.

Tests ensure:
- Pydantic models validate correctly
- Optional fields work as expected
- Overdue computation logic is accurate
- Report data structures have correct shape
- Response wrappers serialize properly

NO DATABASE REQUIRED - Pure unit tests
"""

import pytest
from datetime import date, datetime
from backend.api_v2.schemas.action_log_schemas import (
    ActionLogReportRequest,
    ActionLogItem,
    ActionLogReportData,
    ActionLogReportResponse,
    compute_overdue_fields
)


# ============================================================================
# TEST 1 — PYDANTIC MODEL INSTANTIATION
# ============================================================================

def test_action_log_item_instantiation():
    """
    Test that ActionLogItem can be instantiated with realistic values.
    Validates all fields are accessible and types validated.
    """
    item = ActionLogItem(
        action_item_id=101,
        subcase_id=42,
        title="متابعة التقرير الطبي",
        description="يجب إكمال التقرير خلال 5 أيام",
        status="IN_PROGRESS",
        due_date=date(2026, 2, 10),
        started_at=datetime(2026, 2, 5, 9, 0, 0),
        completed_at=None,
        verified_at=None,
        assigned_to_user_id=7,
        assigned_to_display_name="د. أحمد محمد",
        target_org_unit_id=3,
        target_org_unit_name="قسم الطوارئ",
        is_overdue=False,
        days_overdue=None
    )
    
    # Assert all fields accessible
    assert item.action_item_id == 101
    assert item.subcase_id == 42
    assert item.title == "متابعة التقرير الطبي"
    assert item.description == "يجب إكمال التقرير خلال 5 أيام"
    assert item.status == "IN_PROGRESS"
    assert item.due_date == date(2026, 2, 10)
    assert item.started_at == datetime(2026, 2, 5, 9, 0, 0)
    assert item.completed_at is None
    assert item.verified_at is None
    assert item.assigned_to_user_id == 7
    assert item.assigned_to_display_name == "د. أحمد محمد"
    assert item.target_org_unit_id == 3
    assert item.target_org_unit_name == "قسم الطوارئ"
    assert item.is_overdue is False
    assert item.days_overdue is None
    
    # Validate model serializes to dict
    item_dict = item.model_dump()
    assert isinstance(item_dict, dict)
    assert item_dict["action_item_id"] == 101


# ============================================================================
# TEST 2 — OPTIONAL FIELDS ALLOWED
# ============================================================================

def test_action_log_item_optional_fields():
    """
    Test that ActionLogItem accepts None for optional fields.
    This simulates action items with minimal data.
    """
    item = ActionLogItem(
        action_item_id=202,
        subcase_id=88,
        title="عنصر بدون تفاصيل",
        description=None,  # Optional
        status="DRAFT",
        due_date=None,  # Optional
        started_at=None,
        completed_at=None,
        verified_at=None,
        assigned_to_user_id=None,  # Optional
        assigned_to_display_name=None,  # Optional
        target_org_unit_id=None,  # Optional
        target_org_unit_name=None,  # Optional
        is_overdue=False,
        days_overdue=None
    )
    
    # Assert None values accepted
    assert item.description is None
    assert item.due_date is None
    assert item.assigned_to_user_id is None
    assert item.assigned_to_display_name is None
    assert item.target_org_unit_id is None
    assert item.target_org_unit_name is None
    
    # Validate model still valid
    assert item.action_item_id == 202
    assert item.title == "عنصر بدون تفاصيل"


# ============================================================================
# TEST 3 — OVERDUE HELPER LOGIC
# ============================================================================

def test_compute_overdue_fields_case_a_overdue():
    """
    CASE A: due_date in past, not completed -> overdue
    """
    today = date(2026, 2, 6)
    due_date = date(2026, 2, 1)
    completed_at = None
    
    is_overdue, days_overdue = compute_overdue_fields(due_date, completed_at, today)
    
    assert is_overdue is True
    assert days_overdue == 5  # 2026-02-06 minus 2026-02-01


def test_compute_overdue_fields_case_b_completed():
    """
    CASE B: due_date in past, but completed -> NOT overdue
    """
    today = date(2026, 2, 6)
    due_date = date(2026, 2, 1)
    completed_at = datetime(2026, 2, 3, 14, 30, 0)  # Completed on time
    
    is_overdue, days_overdue = compute_overdue_fields(due_date, completed_at, today)
    
    assert is_overdue is False
    assert days_overdue is None


def test_compute_overdue_fields_case_c_no_due_date():
    """
    CASE C: due_date is None -> NOT overdue
    """
    today = date(2026, 2, 6)
    due_date = None
    completed_at = None
    
    is_overdue, days_overdue = compute_overdue_fields(due_date, completed_at, today)
    
    assert is_overdue is False
    assert days_overdue is None


def test_compute_overdue_fields_future_due_date():
    """
    EDGE CASE: due_date in future -> NOT overdue
    """
    today = date(2026, 2, 6)
    due_date = date(2026, 2, 20)  # Future date
    completed_at = None
    
    is_overdue, days_overdue = compute_overdue_fields(due_date, completed_at, today)
    
    assert is_overdue is False
    assert days_overdue is None


# ============================================================================
# TEST 4 — REPORT DATA SHAPE
# ============================================================================

def test_action_log_report_data_structure():
    """
    Test that ActionLogReportData has correct shape with consistent totals.
    """
    completed_item_1 = ActionLogItem(
        action_item_id=1,
        subcase_id=10,
        title="Completed Task 1",
        description=None,
        status="DONE",
        due_date=date(2026, 1, 15),
        started_at=None,
        completed_at=datetime(2026, 1, 14, 10, 0, 0),
        verified_at=None,
        assigned_to_user_id=5,
        assigned_to_display_name="User A",
        target_org_unit_id=1,
        target_org_unit_name="Section A",
        is_overdue=False,
        days_overdue=None
    )
    
    completed_item_2 = ActionLogItem(
        action_item_id=2,
        subcase_id=11,
        title="Verified Task",
        description=None,
        status="VERIFIED",
        due_date=date(2026, 1, 20),
        started_at=None,
        completed_at=datetime(2026, 1, 19, 15, 0, 0),
        verified_at=datetime(2026, 1, 20, 9, 0, 0),
        assigned_to_user_id=6,
        assigned_to_display_name="User B",
        target_org_unit_id=2,
        target_org_unit_name="Section B",
        is_overdue=False,
        days_overdue=None
    )
    
    not_completed_item_1 = ActionLogItem(
        action_item_id=3,
        subcase_id=12,
        title="In Progress Task",
        description=None,
        status="IN_PROGRESS",
        due_date=date(2026, 2, 5),
        started_at=datetime(2026, 2, 1, 8, 0, 0),
        completed_at=None,
        verified_at=None,
        assigned_to_user_id=7,
        assigned_to_display_name="User C",
        target_org_unit_id=1,
        target_org_unit_name="Section A",
        is_overdue=True,
        days_overdue=1
    )
    
    not_completed_item_2 = ActionLogItem(
        action_item_id=4,
        subcase_id=13,
        title="Draft Task",
        description=None,
        status="DRAFT",
        due_date=date(2026, 2, 10),
        started_at=None,
        completed_at=None,
        verified_at=None,
        assigned_to_user_id=8,
        assigned_to_display_name="User D",
        target_org_unit_id=3,
        target_org_unit_name="Section C",
        is_overdue=False,
        days_overdue=None
    )
    
    not_completed_item_3 = ActionLogItem(
        action_item_id=5,
        subcase_id=14,
        title="Overdue Task",
        description=None,
        status="ADMIN_APPROVED",
        due_date=date(2026, 1, 30),
        started_at=None,
        completed_at=None,
        verified_at=None,
        assigned_to_user_id=9,
        assigned_to_display_name="User E",
        target_org_unit_id=2,
        target_org_unit_name="Section B",
        is_overdue=True,
        days_overdue=7
    )
    
    report_data = ActionLogReportData(
        season_id=1,
        season_name="Q1 2026",
        start_date=date(2026, 1, 1),
        end_date=date(2026, 3, 31),
        generated_at=datetime(2026, 2, 6, 12, 0, 0),
        completed_items=[completed_item_1, completed_item_2],
        not_completed_items=[not_completed_item_1, not_completed_item_2, not_completed_item_3],
        totals={
            "completed_count": 2,
            "not_completed_count": 3,
            "overdue_count": 2
        }
    )
    
    # Assert structure
    assert report_data.season_id == 1
    assert report_data.season_name == "Q1 2026"
    assert report_data.start_date == date(2026, 1, 1)
    assert report_data.end_date == date(2026, 3, 31)
    assert isinstance(report_data.generated_at, datetime)
    
    # Assert list lengths
    assert len(report_data.completed_items) == 2
    assert len(report_data.not_completed_items) == 3
    
    # Assert totals consistent
    assert report_data.totals["completed_count"] == 2
    assert report_data.totals["not_completed_count"] == 3
    assert report_data.totals["overdue_count"] == 2
    
    # Assert totals keys exist
    assert "completed_count" in report_data.totals
    assert "not_completed_count" in report_data.totals
    assert "overdue_count" in report_data.totals
    
    # Validate serialization
    data_dict = report_data.model_dump()
    assert isinstance(data_dict, dict)
    assert data_dict["season_id"] == 1


# ============================================================================
# TEST 5 — RESPONSE WRAPPER
# ============================================================================

def test_action_log_response_success():
    """
    Test ActionLogReportResponse with success case.
    """
    report_data = ActionLogReportData(
        season_id=2,
        season_name="Q2 2026",
        start_date=date(2026, 4, 1),
        end_date=date(2026, 6, 30),
        generated_at=datetime(2026, 5, 10, 14, 0, 0),
        completed_items=[],
        not_completed_items=[],
        totals={
            "completed_count": 0,
            "not_completed_count": 0,
            "overdue_count": 0
        }
    )
    
    response = ActionLogReportResponse(
        success=True,
        data=report_data,
        error=None
    )
    
    # Assert success structure
    assert response.success is True
    assert response.data is not None
    assert response.data.season_id == 2
    assert response.error is None
    
    # Validate serialization
    response_dict = response.model_dump()
    assert isinstance(response_dict, dict)
    assert response_dict["success"] is True
    assert response_dict["data"] is not None
    
    # Validate JSON serialization
    response_json = response.model_dump_json()
    assert isinstance(response_json, str)
    assert '"success":true' in response_json.lower() or '"success": true' in response_json.lower()


def test_action_log_response_failure():
    """
    Test ActionLogReportResponse with failure case.
    """
    response = ActionLogReportResponse(
        success=False,
        data=None,
        error="Season not found"
    )
    
    # Assert failure structure
    assert response.success is False
    assert response.data is None
    assert response.error == "Season not found"
    
    # Validate serialization
    response_dict = response.model_dump()
    assert isinstance(response_dict, dict)
    assert response_dict["success"] is False
    assert response_dict["error"] == "Season not found"
    
    # Validate JSON serialization
    response_json = response.model_dump_json()
    assert isinstance(response_json, str)
    assert '"success":false' in response_json.lower() or '"success": false' in response_json.lower()


def test_action_log_request_validation():
    """
    Test ActionLogReportRequest validates season_id correctly.
    """
    request = ActionLogReportRequest(season_id=5)
    
    assert request.season_id == 5
    
    # Validate serialization
    request_dict = request.model_dump()
    assert request_dict["season_id"] == 5


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
