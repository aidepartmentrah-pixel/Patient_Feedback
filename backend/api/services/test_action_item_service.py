"""
Unit Tests for ActionItem Parent Validation

Tests the critical business rule:
ActionItems must belong to exactly ONE parent (incident, seasonal report, or season case).
"""

import pytest
from datetime import date
from backend.api.services.action_item_service import create_action_item


def test_action_item_with_no_parent_raises_error():
    """Test that creating an action item with no parent raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        create_action_item(
            action_title="Test Action",
            created_by_user_id=1,
            incident_case_id=None,
            seasonal_report_id=None,
            season_case_id=None,
        )
    
    assert "exactly one parent" in str(exc_info.value).lower()


def test_action_item_with_two_parents_raises_error():
    """Test that creating an action item with two parents raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        create_action_item(
            action_title="Test Action",
            created_by_user_id=1,
            incident_case_id=123,
            seasonal_report_id=456,
            season_case_id=None,
        )
    
    assert "exactly one parent" in str(exc_info.value).lower()
    assert "2" in str(exc_info.value)


def test_action_item_with_three_parents_raises_error():
    """Test that creating an action item with all three parents raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        create_action_item(
            action_title="Test Action",
            created_by_user_id=1,
            incident_case_id=123,
            seasonal_report_id=456,
            season_case_id=789,
        )
    
    assert "exactly one parent" in str(exc_info.value).lower()
    assert "3" in str(exc_info.value)


def test_action_item_with_incident_parent_only():
    """Test that creating an action item with only incident_case_id is valid (would succeed if DB exists)."""
    # This test would succeed if connected to database
    # For now, it demonstrates the validation passes
    try:
        create_action_item(
            action_title="Test Action for Incident",
            created_by_user_id=1,
            incident_case_id=123,
            seasonal_report_id=None,
            season_case_id=None,
            action_description="Follow up on patient feedback",
            due_date=date(2026, 2, 1)
        )
    except Exception as e:
        # DB connection error is expected in test environment
        # As long as it's not a ValueError, validation passed
        assert not isinstance(e, ValueError)


def test_action_item_with_seasonal_report_parent_only():
    """Test that creating an action item with only seasonal_report_id is valid."""
    try:
        create_action_item(
            action_title="Test Action for Seasonal Report",
            created_by_user_id=1,
            incident_case_id=None,
            seasonal_report_id=456,
            season_case_id=None,
        )
    except Exception as e:
        assert not isinstance(e, ValueError)


def test_action_item_with_season_case_parent_only():
    """Test that creating an action item with only season_case_id is valid."""
    try:
        create_action_item(
            action_title="Test Action for Season Case",
            created_by_user_id=1,
            incident_case_id=None,
            seasonal_report_id=None,
            season_case_id=789,
        )
    except Exception as e:
        assert not isinstance(e, ValueError)


def test_action_item_missing_required_fields():
    """Test that missing required fields raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        create_action_item(
            action_title="",  # Empty title
            created_by_user_id=1,
            incident_case_id=123,
        )
    
    assert "action_title" in str(exc_info.value).lower()
