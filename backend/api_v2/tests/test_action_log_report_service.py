"""
📋 PHASE F — TEST F-B5 — ACTION LOG REPORT BUILDER SERVICE

Unit tests with mocked dependencies.
Tests orchestration logic without hitting real database.

Mocks:
- season_service.resolve_season_date_range
- action_item_subcase_db.get_action_items_by_due_date_range
- action_log_classification_service.classify_action_items
- current_user (scope)
"""

import pytest
from datetime import date
from unittest.mock import Mock, patch, MagicMock
from backend.api_v2.services.action_log_report_service import build_action_log_report
from backend.api_v2.services.season_service import SeasonNotFoundError


# ============================================================================
# TEST 1 — SEASON NOT FOUND
# ============================================================================

@patch('backend.api_v2.services.action_log_report_service.season_service.resolve_season_date_range')
def test_season_not_found_raises_error(mock_resolve_season):
    """
    Test that SeasonNotFoundError is raised when season doesn't exist.
    """
    # Mock season resolver to raise not found error
    mock_resolve_season.side_effect = SeasonNotFoundError("Season with ID 99999 not found")
    
    # Create mock user and connection
    mock_user = Mock()
    mock_user.allowed_unit_ids = {1, 2}
    mock_user.display_name = "Test User"
    mock_conn = Mock()
    
    # Should raise SeasonNotFoundError
    with pytest.raises(SeasonNotFoundError) as exc_info:
        build_action_log_report(mock_conn, 99999, mock_user, date(2026, 2, 6))
    
    assert "not found" in str(exc_info.value).lower()
    
    print("✅ SeasonNotFoundError correctly raised for invalid season_id")


# ============================================================================
# TEST 2 — CALLS DB WITH CORRECT DATES
# ============================================================================

@patch('backend.api_v2.services.action_log_report_service.action_log_classification_service.classify_action_items')
@patch('backend.api_v2.services.action_log_report_service.action_item_subcase_db.get_action_items_by_due_date_range')
@patch('backend.api_v2.services.action_log_report_service.season_service.resolve_season_date_range')
def test_calls_db_with_correct_dates(mock_resolve_season, mock_get_items, mock_classify):
    """
    Test that DB query is called with correct start and end dates from season.
    """
    # Mock season resolver
    mock_resolve_season.return_value = {
        "season_id": 5,
        "season_name": "Q1 2026",
        "start_date": date(2026, 1, 1),
        "end_date": date(2026, 3, 31)
    }
    
    # Mock DB query to return empty list
    mock_get_items.return_value = []
    
    # Mock classification
    mock_classify.return_value = {
        "completed_items": [],
        "not_completed_items": [],
        "totals": {"completed_count": 0, "not_completed_count": 0, "overdue_count": 0}
    }
    
    # Create mock user and connection
    mock_user = Mock()
    mock_user.allowed_unit_ids = {1, 2}
    mock_user.display_name = "Test User"
    mock_conn = Mock()
    
    # Call service
    result = build_action_log_report(mock_conn, 5, mock_user, date(2026, 2, 6))
    
    # Assert DB query was called with correct dates
    mock_get_items.assert_called_once_with(
        mock_conn,
        date(2026, 1, 1),
        date(2026, 3, 31)
    )
    
    print("✅ DB query called with correct date range from season")


# ============================================================================
# TEST 3 — SCOPE FILTERING APPLIED
# ============================================================================

@patch('backend.api_v2.services.action_log_report_service.action_log_classification_service.classify_action_items')
@patch('backend.api_v2.services.action_log_report_service.action_item_subcase_db.get_action_items_by_due_date_range')
@patch('backend.api_v2.services.action_log_report_service.season_service.resolve_season_date_range')
def test_scope_filtering_applied(mock_resolve_season, mock_get_items, mock_classify):
    """
    Test that only action items in user's allowed org units are passed to classification.
    """
    # Mock season resolver
    mock_resolve_season.return_value = {
        "season_id": 5,
        "season_name": "Q1 2026",
        "start_date": date(2026, 1, 1),
        "end_date": date(2026, 3, 31)
    }
    
    # Mock DB query to return rows with different org units
    mock_get_items.return_value = [
        {"action_item_id": 1, "target_org_unit_id": 1, "status_code": "DONE"},
        {"action_item_id": 2, "target_org_unit_id": 2, "status_code": "IN_PROGRESS"},
        {"action_item_id": 3, "target_org_unit_id": 3, "status_code": "DRAFT"},  # Not in scope
        {"action_item_id": 4, "target_org_unit_id": 1, "status_code": "VERIFIED"},
        {"action_item_id": 5, "target_org_unit_id": 5, "status_code": "ADMIN_APPROVED"}  # Not in scope
    ]
    
    # Mock classification
    mock_classify.return_value = {
        "completed_items": [],
        "not_completed_items": [],
        "totals": {"completed_count": 0, "not_completed_count": 0, "overdue_count": 0}
    }
    
    # Create mock user with limited scope
    mock_user = Mock()
    mock_user.allowed_unit_ids = {1, 2}  # Only org units 1 and 2
    mock_user.display_name = "Test User"
    mock_conn = Mock()
    
    # Call service
    result = build_action_log_report(mock_conn, 5, mock_user, date(2026, 2, 6))
    
    # Assert classification was called with filtered rows only
    assert mock_classify.called
    filtered_rows = mock_classify.call_args[0][0]
    
    # Should only have items from org units 1 and 2
    assert len(filtered_rows) == 3
    for row in filtered_rows:
        assert row["target_org_unit_id"] in {1, 2}
    
    # Check specific IDs
    filtered_ids = [row["action_item_id"] for row in filtered_rows]
    assert 1 in filtered_ids
    assert 2 in filtered_ids
    assert 4 in filtered_ids
    assert 3 not in filtered_ids  # Excluded
    assert 5 not in filtered_ids  # Excluded
    
    print("✅ Scope filtering correctly applied (3 out of 5 items passed)")


# ============================================================================
# TEST 4 — CLASSIFICATION CALLED
# ============================================================================

@patch('backend.api_v2.services.action_log_report_service.action_log_classification_service.classify_action_items')
@patch('backend.api_v2.services.action_log_report_service.action_item_subcase_db.get_action_items_by_due_date_range')
@patch('backend.api_v2.services.action_log_report_service.season_service.resolve_season_date_range')
def test_classification_called_with_filtered_rows(mock_resolve_season, mock_get_items, mock_classify):
    """
    Test that classification service is called once with filtered rows.
    """
    # Mock season resolver
    mock_resolve_season.return_value = {
        "season_id": 5,
        "season_name": "Q1 2026",
        "start_date": date(2026, 1, 1),
        "end_date": date(2026, 3, 31)
    }
    
    # Mock DB query
    mock_get_items.return_value = [
        {"action_item_id": 1, "target_org_unit_id": 1, "status_code": "DONE"}
    ]
    
    # Mock classification
    mock_classify.return_value = {
        "completed_items": [{"action_item_id": 1}],
        "not_completed_items": [],
        "totals": {"completed_count": 1, "not_completed_count": 0, "overdue_count": 0}
    }
    
    # Create mock user and connection
    mock_user = Mock()
    mock_user.allowed_unit_ids = {1}
    mock_user.display_name = "Test User"
    mock_conn = Mock()
    
    today = date(2026, 2, 6)
    
    # Call service
    result = build_action_log_report(mock_conn, 5, mock_user, today)
    
    # Assert classification was called once
    mock_classify.assert_called_once()
    
    # Check arguments: (rows, today)
    call_args = mock_classify.call_args[0]
    assert len(call_args) == 2
    assert isinstance(call_args[0], list)  # rows
    assert call_args[1] == today  # today date
    
    print("✅ Classification service called once with correct arguments")


# ============================================================================
# TEST 5 — OUTPUT STRUCTURE
# ============================================================================

@patch('backend.api_v2.services.action_log_report_service.action_log_classification_service.classify_action_items')
@patch('backend.api_v2.services.action_log_report_service.action_item_subcase_db.get_action_items_by_due_date_range')
@patch('backend.api_v2.services.action_log_report_service.season_service.resolve_season_date_range')
def test_output_structure_correct(mock_resolve_season, mock_get_items, mock_classify):
    """
    Test that returned dict has correct structure.
    """
    # Mock season resolver
    mock_resolve_season.return_value = {
        "season_id": 5,
        "season_name": "Q1 2026",
        "start_date": date(2026, 1, 1),
        "end_date": date(2026, 3, 31)
    }
    
    # Mock DB query
    mock_get_items.return_value = []
    
    # Mock classification
    mock_classify.return_value = {
        "completed_items": [{"id": 1}],
        "not_completed_items": [{"id": 2}],
        "totals": {"completed_count": 1, "not_completed_count": 1, "overdue_count": 0}
    }
    
    # Create mock user and connection
    mock_user = Mock()
    mock_user.allowed_unit_ids = {1}
    mock_user.display_name = "Test User"
    mock_conn = Mock()
    
    # Call service
    result = build_action_log_report(mock_conn, 5, mock_user, date(2026, 2, 6))
    
    # Assert structure
    assert "meta" in result
    assert "completed_items" in result
    assert "not_completed_items" in result
    assert "totals" in result
    
    # Check types
    assert isinstance(result["meta"], dict)
    assert isinstance(result["completed_items"], list)
    assert isinstance(result["not_completed_items"], list)
    assert isinstance(result["totals"], dict)
    
    print("✅ Output structure correct (meta, completed_items, not_completed_items, totals)")


# ============================================================================
# TEST 6 — META FIELDS PRESENT
# ============================================================================

@patch('backend.api_v2.services.action_log_report_service.action_log_classification_service.classify_action_items')
@patch('backend.api_v2.services.action_log_report_service.action_item_subcase_db.get_action_items_by_due_date_range')
@patch('backend.api_v2.services.action_log_report_service.season_service.resolve_season_date_range')
def test_meta_fields_present(mock_resolve_season, mock_get_items, mock_classify):
    """
    Test that meta dict contains all required fields.
    """
    # Mock season resolver
    mock_resolve_season.return_value = {
        "season_id": 5,
        "season_name": "Q1 2026",
        "start_date": date(2026, 1, 1),
        "end_date": date(2026, 3, 31)
    }
    
    # Mock DB query
    mock_get_items.return_value = []
    
    # Mock classification
    mock_classify.return_value = {
        "completed_items": [],
        "not_completed_items": [],
        "totals": {"completed_count": 0, "not_completed_count": 0, "overdue_count": 0}
    }
    
    # Create mock user and connection
    mock_user = Mock()
    mock_user.allowed_unit_ids = {1}
    mock_user.display_name = "Dr. Ahmed"
    mock_conn = Mock()
    
    today = date(2026, 2, 6)
    
    # Call service
    result = build_action_log_report(mock_conn, 5, mock_user, today)
    
    meta = result["meta"]
    
    # Assert all required fields present
    assert "season_id" in meta
    assert "season_name" in meta
    assert "start_date" in meta
    assert "end_date" in meta
    assert "generated_at" in meta
    assert "generated_by" in meta
    
    # Assert values
    assert meta["season_id"] == 5
    assert meta["season_name"] == "Q1 2026"
    assert meta["start_date"] == date(2026, 1, 1)
    assert meta["end_date"] == date(2026, 3, 31)
    assert meta["generated_at"] == today
    assert meta["generated_by"] == "Dr. Ahmed"
    
    print("✅ Meta fields all present with correct values")


# ============================================================================
# TEST 7 — EMPTY SCOPE RETURNS EMPTY REPORT
# ============================================================================

@patch('backend.api_v2.services.action_log_report_service.action_log_classification_service.classify_action_items')
@patch('backend.api_v2.services.action_log_report_service.action_item_subcase_db.get_action_items_by_due_date_range')
@patch('backend.api_v2.services.action_log_report_service.season_service.resolve_season_date_range')
def test_empty_scope_returns_empty_report(mock_resolve_season, mock_get_items, mock_classify):
    """
    Test that if user has no allowed org units, report is empty.
    """
    # Mock season resolver
    mock_resolve_season.return_value = {
        "season_id": 5,
        "season_name": "Q1 2026",
        "start_date": date(2026, 1, 1),
        "end_date": date(2026, 3, 31)
    }
    
    # Mock DB query returns items
    mock_get_items.return_value = [
        {"action_item_id": 1, "target_org_unit_id": 1, "status_code": "DONE"},
        {"action_item_id": 2, "target_org_unit_id": 2, "status_code": "IN_PROGRESS"}
    ]
    
    # Mock classification for empty input
    mock_classify.return_value = {
        "completed_items": [],
        "not_completed_items": [],
        "totals": {"completed_count": 0, "not_completed_count": 0, "overdue_count": 0}
    }
    
    # Create mock user with NO allowed org units
    mock_user = Mock()
    mock_user.allowed_unit_ids = set()  # Empty scope
    mock_user.display_name = "Limited User"
    mock_conn = Mock()
    
    # Call service
    result = build_action_log_report(mock_conn, 5, mock_user, date(2026, 2, 6))
    
    # Classification should be called with empty list
    filtered_rows = mock_classify.call_args[0][0]
    assert len(filtered_rows) == 0
    
    # Result should be empty
    assert len(result["completed_items"]) == 0
    assert len(result["not_completed_items"]) == 0
    
    print("✅ Empty scope correctly results in empty report")


# ============================================================================
# TEST 8 — USER WITHOUT DISPLAY NAME HANDLED
# ============================================================================

@patch('backend.api_v2.services.action_log_report_service.action_log_classification_service.classify_action_items')
@patch('backend.api_v2.services.action_log_report_service.action_item_subcase_db.get_action_items_by_due_date_range')
@patch('backend.api_v2.services.action_log_report_service.season_service.resolve_season_date_range')
def test_user_without_display_name_handled(mock_resolve_season, mock_get_items, mock_classify):
    """
    Test that user without display_name attribute is handled gracefully.
    """
    # Mock season resolver
    mock_resolve_season.return_value = {
        "season_id": 5,
        "season_name": "Q1 2026",
        "start_date": date(2026, 1, 1),
        "end_date": date(2026, 3, 31)
    }
    
    # Mock DB query
    mock_get_items.return_value = []
    
    # Mock classification
    mock_classify.return_value = {
        "completed_items": [],
        "not_completed_items": [],
        "totals": {"completed_count": 0, "not_completed_count": 0, "overdue_count": 0}
    }
    
    # Create mock user WITHOUT display_name
    mock_user = Mock(spec=['allowed_unit_ids'])  # Only has allowed_unit_ids
    mock_user.allowed_unit_ids = {1}
    mock_conn = Mock()
    
    # Call service (should not raise error)
    result = build_action_log_report(mock_conn, 5, mock_user, date(2026, 2, 6))
    
    # generated_by should be None
    assert result["meta"]["generated_by"] is None
    
    print("✅ User without display_name handled gracefully (generated_by = None)")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
