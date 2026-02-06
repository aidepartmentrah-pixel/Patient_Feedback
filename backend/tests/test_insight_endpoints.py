"""
PHASE 4B - Insight Endpoint Smoke Tests

Basic smoke tests to verify insight endpoints are reachable and return 200.
Uses session-based authentication (no JWT).

Tests:
- GET /api/v2/insight/kpi-summary
- POST /api/v2/insight/distribution
- POST /api/v2/insight/trend
- GET /api/v2/insight/stuck
"""

import sys
import os
import pytest

# Add backend directory to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from fastapi.testclient import TestClient
import main

app = main.app
client = TestClient(app)


@pytest.fixture
def authenticated_client():
    """
    Fixture to provide an authenticated test client.
    
    Authenticates using the session-based auth system.
    Returns a TestClient with an active session.
    """
    # Login with a test user
    # Using admin credentials that should exist in the system
    login_response = client.post(
        "/api/auth/login",
        json={
            "username": "software_admin",
            "password": "admin123"
        }
    )
    
    # Verify login succeeded
    if login_response.status_code != 200:
        pytest.skip(f"Authentication failed: {login_response.status_code}. "
                   f"Test user may not exist. Response: {login_response.json()}")
    
    # Client now has session cookie - return it
    return client


def test_kpi_summary_smoke(authenticated_client):
    """
    Test 1: GET /api/v2/insight/kpi-summary smoke test.
    
    Verifies:
    - Endpoint is reachable
    - Returns 200 status
    - Response is a dict
    """
    response = authenticated_client.get("/api/v2/insight/kpi-summary")
    
    # Assert endpoint is reachable
    assert response.status_code == 200, \
        f"Expected status 200, got {response.status_code}. Response: {response.text}"
    
    # Assert response is dict
    data = response.json()
    assert isinstance(data, dict), \
        f"Expected dict response, got {type(data)}"
    
    # Basic shape check (should have expected keys)
    assert "total_subcases" in data, "Response missing 'total_subcases' key"
    assert "by_status" in data, "Response missing 'by_status' key"
    assert "action_items" in data, "Response missing 'action_items' key"


def test_distribution_status_smoke(authenticated_client):
    """
    Test 2: POST /api/v2/insight/distribution smoke test.
    
    Verifies:
    - Endpoint is reachable
    - Returns 200 status
    - Response is a list
    """
    response = authenticated_client.post(
        "/api/v2/insight/distribution",
        json={"dimension": "status"}
    )
    
    # Assert endpoint is reachable
    assert response.status_code == 200, \
        f"Expected status 200, got {response.status_code}. Response: {response.text}"
    
    # Assert response is list
    data = response.json()
    assert isinstance(data, list), \
        f"Expected list response, got {type(data)}"


def test_distribution_org_unit_smoke(authenticated_client):
    """
    Test 3: POST /api/v2/insight/distribution with org_unit dimension.
    
    Verifies endpoint works with different dimension values.
    """
    response = authenticated_client.post(
        "/api/v2/insight/distribution",
        json={"dimension": "org_unit"}
    )
    
    # Assert endpoint is reachable
    assert response.status_code == 200, \
        f"Expected status 200, got {response.status_code}. Response: {response.text}"
    
    # Assert response is list
    data = response.json()
    assert isinstance(data, list), \
        f"Expected list response, got {type(data)}"


def test_trend_month_smoke(authenticated_client):
    """
    Test 4: POST /api/v2/insight/trend smoke test.
    
    Verifies:
    - Endpoint is reachable
    - Returns 200 status
    - Response is a list
    """
    response = authenticated_client.post(
        "/api/v2/insight/trend",
        json={"bucket": "month"}
    )
    
    # Assert endpoint is reachable
    assert response.status_code == 200, \
        f"Expected status 200, got {response.status_code}. Response: {response.text}"
    
    # Assert response is list
    data = response.json()
    assert isinstance(data, list), \
        f"Expected list response, got {type(data)}"


def test_trend_year_smoke(authenticated_client):
    """
    Test 5: POST /api/v2/insight/trend with year bucket.
    
    Verifies endpoint works with different bucket values.
    """
    response = authenticated_client.post(
        "/api/v2/insight/trend",
        json={"bucket": "year"}
    )
    
    # Assert endpoint is reachable
    assert response.status_code == 200, \
        f"Expected status 200, got {response.status_code}. Response: {response.text}"
    
    # Assert response is list
    data = response.json()
    assert isinstance(data, list), \
        f"Expected list response, got {type(data)}"


def test_trend_day_smoke(authenticated_client):
    """
    Test 6: POST /api/v2/insight/trend with day bucket.
    
    Verifies endpoint works with day-level granularity.
    """
    response = authenticated_client.post(
        "/api/v2/insight/trend",
        json={"bucket": "day"}
    )
    
    # Assert endpoint is reachable
    assert response.status_code == 200, \
        f"Expected status 200, got {response.status_code}. Response: {response.text}"
    
    # Assert response is list
    data = response.json()
    assert isinstance(data, list), \
        f"Expected list response, got {type(data)}"


def test_stuck_smoke(authenticated_client):
    """
    Test 7: GET /api/v2/insight/stuck smoke test.
    
    Verifies:
    - Endpoint is reachable
    - Returns 200 status
    - Response is a list
    """
    response = authenticated_client.get("/api/v2/insight/stuck?days_threshold=7")
    
    # Assert endpoint is reachable
    assert response.status_code == 200, \
        f"Expected status 200, got {response.status_code}. Response: {response.text}"
    
    # Assert response is list
    data = response.json()
    assert isinstance(data, list), \
        f"Expected list response, got {type(data)}"


def test_stuck_low_threshold_smoke(authenticated_client):
    """
    Test 8: GET /api/v2/insight/stuck with low threshold.
    
    Verifies endpoint works with low threshold value (catches more cases).
    """
    response = authenticated_client.get("/api/v2/insight/stuck?days_threshold=1")
    
    # Assert endpoint is reachable
    assert response.status_code == 200, \
        f"Expected status 200, got {response.status_code}. Response: {response.text}"
    
    # Assert response is list
    data = response.json()
    assert isinstance(data, list), \
        f"Expected list response, got {type(data)}"


def test_stuck_custom_threshold_smoke(authenticated_client):
    """
    Test 9: GET /api/v2/insight/stuck with custom threshold.
    
    Verifies endpoint works with different threshold values.
    """
    response = authenticated_client.get("/api/v2/insight/stuck?days_threshold=30")
    
    # Assert endpoint is reachable
    assert response.status_code == 200, \
        f"Expected status 200, got {response.status_code}. Response: {response.text}"
    
    # Assert response is list
    data = response.json()
    assert isinstance(data, list), \
        f"Expected list response, got {type(data)}"


def test_all_endpoints_require_authentication():
    """
    Test 10: Verify all endpoints require authentication.
    
    Attempts to access endpoints without authentication.
    Should return 401 Unauthorized.
    """
    unauthenticated_client = TestClient(app)
    
    # Test KPI summary
    response = unauthenticated_client.get("/api/v2/insight/kpi-summary")
    assert response.status_code == 401, \
        "KPI summary endpoint should require authentication"
    
    # Test distribution
    response = unauthenticated_client.post(
        "/api/v2/insight/distribution",
        json={"dimension": "status"}
    )
    assert response.status_code == 401, \
        "Distribution endpoint should require authentication"
    
    # Test trend
    response = unauthenticated_client.post(
        "/api/v2/insight/trend",
        json={"bucket": "month"}
    )
    assert response.status_code == 401, \
        "Trend endpoint should require authentication"
    
    # Test stuck
    response = unauthenticated_client.get("/api/v2/insight/stuck")
    assert response.status_code == 401, \
        "Stuck endpoint should require authentication"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
