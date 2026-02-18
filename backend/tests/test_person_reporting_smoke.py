"""
PHASE D — B-D12 — BACKEND SMOKE TESTS FOR PERSON REPORTING
===========================================================

Phase D smoke coverage — wiring validation only.

Lightweight smoke tests validating endpoint behavior and service wiring.
No heavy integration tests. No test DB fixtures required.
Focus on contract and wiring validation.

Tests:
1. Worker profile endpoint reachable - call /api/workers/{id}/profile
2. Doctor seasonal word endpoint reachable
3. Worker seasonal word endpoint reachable
4. Unauthorized blocked - call without auth
5. Response type check - verify content-type contains wordprocessingml.document
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
    
    Phase D smoke coverage — wiring validation only.
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


def test_worker_profile_endpoint_reachable(authenticated_client):
    """
    TEST 1 — Worker profile endpoint reachable
    
    Phase D smoke coverage — wiring validation only.
    
    Verifies:
    - Endpoint /api/workers/{id}/profile is reachable
    - Returns 200 or 404 (worker found or not found)
    - Does NOT return 500 (server error)
    
    Accepts:
    - 200: Worker found, profile returned
    - 404: Worker not found (valid response)
    - 403: Forbidden (out of scope, valid response)
    
    Fails on:
    - 500: Server error (wiring problem)
    """
    # Use employee ID 1 (likely exists in most systems)
    response = authenticated_client.get("/api/workers/1/profile")
    
    # Assert endpoint is reachable and not returning server error
    assert response.status_code in [200, 403, 404], \
        f"Expected status 200/403/404, got {response.status_code}. " \
        f"Response: {response.text}. " \
        f"Status 500 indicates wiring problem."
    
    # If 200, verify response structure
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, dict), \
            f"Expected dict response, got {type(data)}"
        
        # Basic shape check - worker profile can have various structures
        assert "employee_id" in data or "worker_identity" in data or "worker" in data, \
            "Response missing worker identity fields"


def test_doctor_seasonal_word_endpoint_reachable(authenticated_client):
    """
    TEST 2 — Doctor seasonal word endpoint reachable
    
    Phase D smoke coverage — wiring validation only.
    
    Verifies:
    - Endpoint /api/person-reports/doctor/{id}/seasonal-word is reachable
    - Returns 200, 403, or 404 (NOT 500)
    - Endpoint accepts date query parameters
    
    Accepts:
    - 200: Report generated successfully
    - 403: Forbidden (insufficient role or out of scope)
    - 404: Doctor not found
    
    Fails on:
    - 500: Server error (wiring problem)
    """
    # Use doctor ID 1 with sample date range
    response = authenticated_client.get(
        "/api/person-reports/doctor/1/seasonal-word",
        params={
            "season_start": "2024-01-01",
            "season_end": "2024-12-31"
        }
    )
    
    # Assert endpoint is reachable and not returning server error
    assert response.status_code in [200, 403, 404], \
        f"Expected status 200/403/404, got {response.status_code}. " \
        f"Response: {response.text[:500]}. " \
        f"Status 500 indicates wiring problem."


def test_worker_seasonal_word_endpoint_reachable(authenticated_client):
    """
    TEST 3 — Worker seasonal word endpoint reachable
    
    Phase D smoke coverage — wiring validation only.
    
    Verifies:
    - Endpoint /api/person-reports/worker/{id}/seasonal-word is reachable
    - Returns 200, 403, or 404 (NOT 500)
    - Endpoint accepts date query parameters
    
    Accepts:
    - 200: Report generated successfully
    - 403: Forbidden (insufficient role or out of scope)
    - 404: Worker not found
    
    Fails on:
    - 500: Server error (wiring problem)
    """
    # Use employee ID 1 with sample date range
    response = authenticated_client.get(
        "/api/person-reports/worker/1/seasonal-word",
        params={
            "season_start": "2024-01-01",
            "season_end": "2024-12-31"
        }
    )
    
    # Assert endpoint is reachable and not returning server error
    assert response.status_code in [200, 403, 404], \
        f"Expected status 200/403/404, got {response.status_code}. " \
        f"Response: {response.text[:500]}. " \
        f"Status 500 indicates wiring problem."


def test_unauthorized_blocked():
    """
    TEST 4 — Unauthorized blocked
    
    Phase D smoke coverage — wiring validation only.
    
    Verifies:
    - Endpoints require authentication
    - Unauthenticated requests are blocked
    - Returns 401 or 403 (NOT 200 or 500)
    
    Accepts:
    - 401: Unauthorized (not authenticated)
    - 403: Forbidden (authenticated but no permission)
    
    Fails on:
    - 200: Should not allow unauthenticated access
    - 500: Server error (wiring problem)
    """
    # Create unauthenticated client (no login)
    unauth_client = TestClient(app)
    
    # Test doctor seasonal endpoint
    response_doctor = unauth_client.get(
        "/api/person-reports/doctor/1/seasonal-word",
        params={
            "season_start": "2024-01-01",
            "season_end": "2024-12-31"
        }
    )
    
    # Assert authentication is required
    assert response_doctor.status_code in [401, 403], \
        f"Expected status 401/403 for unauthenticated request, got {response_doctor.status_code}. " \
        f"Endpoint should require authentication."
    
    # Test worker seasonal endpoint
    response_worker = unauth_client.get(
        "/api/person-reports/worker/1/seasonal-word",
        params={
            "season_start": "2024-01-01",
            "season_end": "2024-12-31"
        }
    )
    
    # Assert authentication is required
    assert response_worker.status_code in [401, 403], \
        f"Expected status 401/403 for unauthenticated request, got {response_worker.status_code}. " \
        f"Endpoint should require authentication."


def test_response_type_check(authenticated_client):
    """
    TEST 5 — Response type check
    
    Phase D smoke coverage — wiring validation only.
    
    Verifies:
    - If Word document is returned (200), content-type is correct
    - Content-Type header contains: wordprocessingml.document
    - StreamingResponse configured properly
    
    Note: This test may get 403/404 if the test user doesn't have access
    or the employee doesn't exist. That's acceptable. We're testing the
    contract when the response IS successful.
    """
    # Try doctor seasonal endpoint
    response_doctor = authenticated_client.get(
        "/api/person-reports/doctor/1/seasonal-word",
        params={
            "season_start": "2024-01-01",
            "season_end": "2024-12-31"
        }
    )
    
    # If response is successful (200), verify content-type
    if response_doctor.status_code == 200:
        content_type = response_doctor.headers.get("content-type", "")
        assert "wordprocessingml.document" in content_type, \
            f"Expected content-type to contain 'wordprocessingml.document', " \
            f"got: {content_type}"
        
        # Verify Content-Disposition header exists
        content_disposition = response_doctor.headers.get("content-disposition", "")
        assert "attachment" in content_disposition.lower(), \
            f"Expected Content-Disposition to contain 'attachment', " \
            f"got: {content_disposition}"
    
    # Try worker seasonal endpoint
    response_worker = authenticated_client.get(
        "/api/person-reports/worker/1/seasonal-word",
        params={
            "season_start": "2024-01-01",
            "season_end": "2024-12-31"
        }
    )
    
    # If response is successful (200), verify content-type
    if response_worker.status_code == 200:
        content_type = response_worker.headers.get("content-type", "")
        assert "wordprocessingml.document" in content_type, \
            f"Expected content-type to contain 'wordprocessingml.document', " \
            f"got: {content_type}"
        
        # Verify Content-Disposition header exists
        content_disposition = response_worker.headers.get("content-disposition", "")
        assert "attachment" in content_disposition.lower(), \
            f"Expected Content-Disposition to contain 'attachment', " \
            f"got: {content_disposition}"
    
    # If neither endpoint returned 200, skip content-type check but pass test
    # (wiring is still valid even if access is denied)
    if response_doctor.status_code != 200 and response_worker.status_code != 200:
        pytest.skip(
            f"Both endpoints returned non-200 status "
            f"(doctor: {response_doctor.status_code}, worker: {response_worker.status_code}). "
            f"Cannot verify content-type without successful response. "
            f"This is acceptable - test user may lack access or employees may not exist."
        )


if __name__ == "__main__":
    """
    Run smoke tests directly.
    
    Usage:
        python test_person_reporting_smoke.py
    """
    pytest.main([__file__, "-v", "-s"])
