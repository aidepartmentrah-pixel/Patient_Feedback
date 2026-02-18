"""
📋 PHASE F — TEST F-B7 — ACTION LOG EXPORT ENDPOINT

Tests for Action Log export endpoint (API v2).
Uses FastAPI TestClient with mocked services.

NO real database. NO real Word generation.
All services mocked.

Tests verify:
- Endpoint returns Word file
- Services called with correct params
- Authentication required
- Error handling
"""

import pytest
from datetime import date
from unittest.mock import Mock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.api_v2.routers.action_log_router import router


# ============================================================================
# TEST CLIENT & FIXTURES
# ============================================================================

# Create minimal test app with just action_log_router
app = FastAPI()
app.include_router(router)
client = TestClient(app)


@pytest.fixture
def mock_user():
    """
    Mock authenticated user.
    """
    user = Mock()
    user.user_id = 1
    user.display_name = "Test User"
    user.allowed_unit_ids = {1, 2, 3}
    user.scopes = [Mock(role_code="WORKER")]
    return user


@pytest.fixture
def sample_report_data():
    """
    Sample report data returned by build_action_log_report.
    """
    return {
        "meta": {
            "season_id": 5,
            "season_name": "Q1 2026",
            "start_date": date(2026, 1, 1),
            "end_date": date(2026, 3, 31),
            "generated_at": date(2026, 2, 6),
            "generated_by": "Test User"
        },
        "completed_items": [
            {
                "action_item_id": 101,
                "title": "إجراء منجز",
                "assigned_to_display_name": "د. أحمد",
                "org_unit_name": "قسم الطوارئ",
                "due_date": date(2026, 1, 15),
                "completed_at": date(2026, 1, 14),
                "is_overdue": False,
                "days_overdue": None
            }
        ],
        "not_completed_items": [],
        "totals": {
            "completed_count": 1,
            "not_completed_count": 0,
            "overdue_count": 0
        }
    }


# ============================================================================
# TEST 1 — ENDPOINT SUCCESS
# ============================================================================

@patch('backend.api_v2.routers.action_log_router.generate_action_log_word')
@patch('backend.api_v2.routers.action_log_router.build_action_log_report')
@patch('backend.api_v2.routers.action_log_router.get_db_connection')
def test_endpoint_success(
    mock_get_db_connection,
    mock_build_report,
    mock_generate_word,
    mock_user,
    sample_report_data
):
    """
    Test that endpoint returns valid Word file with correct headers.
    """
    # Override get_current_user dependency
    from backend.api.dependencies.user_context import get_current_user
    app.dependency_overrides[get_current_user] = lambda: mock_user
    
    # Mock database connection
    mock_conn = Mock()
    mock_get_db_connection.return_value = mock_conn
    
    # Mock report builder
    mock_build_report.return_value = sample_report_data
    
    # Mock Word generator
    fake_doc_bytes = b"FAKE_DOCX_CONTENT_123"
    mock_generate_word.return_value = fake_doc_bytes
    
    try:
        # Call endpoint
        response = client.get("/api/v2/action-log/export?season_id=5")
        
        # Assert response
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        assert "attachment" in response.headers["content-disposition"]
        assert "action_log_season_5.docx" in response.headers["content-disposition"]
        assert response.content == fake_doc_bytes
        
        # Assert connection was closed
        mock_conn.close.assert_called_once()
        
        print("✅ Endpoint returns Word file with correct headers")
    finally:
        # Clean up dependency overrides
        app.dependency_overrides.clear()


# ============================================================================
# TEST 2 — SERVICE CALLED WITH CORRECT PARAMS
# ============================================================================

@patch('backend.api_v2.routers.action_log_router.generate_action_log_word')
@patch('backend.api_v2.routers.action_log_router.build_action_log_report')
@patch('backend.api_v2.routers.action_log_router.get_db_connection')
def test_service_called_with_correct_params(
    mock_get_db_connection,
    mock_build_report,
    mock_generate_word,
    mock_user,
    sample_report_data
):
    """
    Test that build_action_log_report is called with correct parameters.
    """
    # Override get_current_user dependency
    from backend.api.dependencies.user_context import get_current_user
    app.dependency_overrides[get_current_user] = lambda: mock_user
    
    # Mock database connection
    mock_conn = Mock()
    mock_get_db_connection.return_value = mock_conn
    
    # Mock report builder
    mock_build_report.return_value = sample_report_data
    
    # Mock Word generator
    mock_generate_word.return_value = b"FAKE_DOCX"
    
    try:
        # Call endpoint
        response = client.get("/api/v2/action-log/export?season_id=7")
        
        # Assert service was called with correct params
        assert mock_build_report.call_count == 1
        call_args = mock_build_report.call_args
        
        # Check positional/keyword arguments
        assert call_args.kwargs["conn"] == mock_conn or call_args.args[0] == mock_conn
        assert call_args.kwargs["season_id"] == 7 or call_args.args[1] == 7
        assert call_args.kwargs["current_user"] == mock_user or call_args.args[2] == mock_user
        # today should be date.today() - just check it's a date
        today_arg = call_args.kwargs.get("today") or call_args.args[3]
        assert isinstance(today_arg, date)
        
        print("✅ Service called with correct parameters")
    finally:
        # Clean up dependency overrides
        app.dependency_overrides.clear()


# ============================================================================
# TEST 3 — WORD GENERATOR CALLED WITH REPORT DATA
# ============================================================================

@patch('backend.api_v2.routers.action_log_router.generate_action_log_word')
@patch('backend.api_v2.routers.action_log_router.build_action_log_report')
@patch('backend.api_v2.routers.action_log_router.get_db_connection')
def test_word_generator_called_with_report_data(
    mock_get_db_connection,
    mock_build_report,
    mock_generate_word,
    mock_user,
    sample_report_data
):
    """
    Test that generate_action_log_word is called with report data.
    """
    # Override get_current_user dependency
    from backend.api.dependencies.user_context import get_current_user
    app.dependency_overrides[get_current_user] = lambda: mock_user
    
    # Mock database connection
    mock_conn = Mock()
    mock_get_db_connection.return_value = mock_conn
    
    # Mock report builder
    mock_build_report.return_value = sample_report_data
    
    # Mock Word generator
    mock_generate_word.return_value = b"FAKE_DOCX"
    
    try:
        # Call endpoint
        response = client.get("/api/v2/action-log/export?season_id=5")
        
        # Assert Word generator was called with report data
        mock_generate_word.assert_called_once_with(sample_report_data)
        
        print("✅ Word generator called with report data")
    finally:
        # Clean up dependency overrides
        app.dependency_overrides.clear()


# ============================================================================
# TEST 4 — SEASON NOT FOUND ERROR
# ============================================================================

@patch('backend.api_v2.routers.action_log_router.build_action_log_report')
@patch('backend.api_v2.routers.action_log_router.get_db_connection')
def test_season_not_found_error(
    mock_get_db_connection,
    mock_build_report,
    mock_user
):
    """
    Test that endpoint handles season not found error.
    """
    # Override get_current_user dependency
    from backend.api.dependencies.user_context import get_current_user
    app.dependency_overrides[get_current_user] = lambda: mock_user
    
    # Mock database connection
    mock_conn = Mock()
    mock_get_db_connection.return_value = mock_conn
    
    # Mock report builder to raise error
    from backend.api_v2.services.season_service import SeasonNotFoundError
    mock_build_report.side_effect = SeasonNotFoundError("Season with ID 99999 not found")
    
    try:
        # Call endpoint - exception will propagate through TestClient
        # FastAPI will convert to 500 (or we can add exception handlers later)
        response = client.get("/api/v2/action-log/export?season_id=99999")
        
        # Should get error response
        assert response.status_code >= 400
        
        # Connection should still be closed
        mock_conn.close.assert_called_once()
        
        print(f"✅ Season not found error handled (status: {response.status_code})")
    except Exception as e:
        # If exception propagates through TestClient (which it does for custom exceptions)
        # This is expected behavior - the connection cleanup happens in finally block
        mock_conn.close.assert_called_once()
        print(f"✅ Season not found error raised as expected: {type(e).__name__}")
    finally:
        # Clean up dependency overrides
        app.dependency_overrides.clear()


# ============================================================================
# TEST 5 — AUTH REQUIRED
# ============================================================================

def test_auth_required():
    """
    Test that endpoint requires authentication.
    """
    # Override get_current_user to raise 401
    from backend.api.dependencies.user_context import get_current_user
    from fastapi import HTTPException
    
    def mock_auth_fail():
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    app.dependency_overrides[get_current_user] = mock_auth_fail
    
    try:
        # Call endpoint
        response = client.get("/api/v2/action-log/export?season_id=5")
        
        # Assert 401
        assert response.status_code == 401
        
        print("✅ Authentication required")
    finally:
        # Clean up dependency overrides
        app.dependency_overrides.clear()


# ============================================================================
# TEST 6 — CONNECTION CLOSED ON ERROR
# ============================================================================

@patch('backend.api_v2.routers.action_log_router.build_action_log_report')
@patch('backend.api_v2.routers.action_log_router.get_db_connection')
def test_connection_closed_on_error(
    mock_get_db_connection,
    mock_build_report,
    mock_user
):
    """
    Test that database connection is closed even when error occurs.
    """
    # Override get_current_user dependency
    from backend.api.dependencies.user_context import get_current_user
    app.dependency_overrides[get_current_user] = lambda: mock_user
    
    # Mock database connection
    mock_conn = Mock()
    mock_get_db_connection.return_value = mock_conn
    
    # Mock report builder to raise generic error
    mock_build_report.side_effect = ValueError("Some error")
    
    try:
        # Call endpoint (will fail)
        try:
            response = client.get("/api/v2/action-log/export?season_id=5")
        except:
            pass
        
        # Assert connection was closed despite error
        mock_conn.close.assert_called_once()
        
        print("✅ Connection closed on error")
    finally:
        # Clean up dependency overrides
        app.dependency_overrides.clear()


# ============================================================================
# TEST 7 — MISSING SEASON_ID PARAM
# ============================================================================

@patch('backend.api_v2.routers.action_log_router.get_db_connection')
def test_missing_season_id_param(mock_get_db_connection, mock_user):
    """
    Test that endpoint requires season_id query param.
    """
    # Override get_current_user dependency
    from backend.api.dependencies.user_context import get_current_user
    app.dependency_overrides[get_current_user] = lambda: mock_user
    
    # Mock connection (should not be called)
    mock_conn = Mock()
    mock_get_db_connection.return_value = mock_conn
    
    try:
        # Call endpoint without season_id (FastAPI should return 422 validation error)
        response = client.get("/api/v2/action-log/export")
        
        # Assert validation error
        assert response.status_code == 422
        
        print("✅ Missing season_id validation")
    finally:
        # Clean up dependency overrides
        app.dependency_overrides.clear()


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
