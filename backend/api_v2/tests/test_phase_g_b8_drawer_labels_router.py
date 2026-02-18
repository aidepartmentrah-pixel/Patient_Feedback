"""
Phase G - G-B8: Drawer Labels Router Tests

Tests all endpoints in drawer_labels_router.py:
1. POST /api/v2/drawer-labels - Create label
2. GET /api/v2/drawer-labels - List active labels
3. DELETE /api/v2/drawer-labels/{label_id} - Disable label

Tests include:
- Success scenarios for all endpoints
- Validation errors (short names, duplicates)
- Authorization checks (forbidden roles, unauthorized)
- Soft delete behavior (disabled labels not returned)

Tests run against real database with TestClient integration.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock
import uuid
from fastapi.testclient import TestClient

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

from main import app
from backend.api.dependencies.user_context import get_current_user
from backend.api.schemas.auth_models import CurrentUser
import pyodbc
import os


# ==================== TEST CLIENT ====================
client = TestClient(app)


# ==================== HELPER FUNCTIONS ====================

def create_mock_user(user_id=1, username="test_user", roles=None):
    """Create mock authenticated user."""
    if roles is None:
        roles = ["SOFTWARE_ADMIN"]
    
    mock_user = Mock(spec=CurrentUser)
    mock_user.user_id = user_id
    mock_user.username = username
    mock_user.roles = roles
    mock_user.section_id = 1
    mock_user.organizational_unit_id = 1
    
    return mock_user


# ==================== FIXTURES ====================

@pytest.fixture
def auth_client_admin():
    """TestClient with SOFTWARE_ADMIN authentication."""
    def override_get_current_user():
        return create_mock_user(user_id=1, username="test_admin", roles=["SOFTWARE_ADMIN"])
    
    app.dependency_overrides[get_current_user] = override_get_current_user
    yield client
    app.dependency_overrides.clear()


@pytest.fixture
def auth_client_worker():
    """TestClient with WORKER authentication."""
    def override_get_current_user():
        return create_mock_user(user_id=2, username="test_worker", roles=["WORKER"])
    
    app.dependency_overrides[get_current_user] = override_get_current_user
    yield client
    app.dependency_overrides.clear()


@pytest.fixture
def auth_client_forbidden():
    """TestClient with DOCTOR authentication (forbidden role)."""
    def override_get_current_user():
        return create_mock_user(user_id=3, username="test_doctor", roles=["DOCTOR"])
    
    app.dependency_overrides[get_current_user] = override_get_current_user
    yield client
    app.dependency_overrides.clear()


@pytest.fixture
def unauth_client():
    """TestClient without authentication."""
    return client


# ==================== HELPER FUNCTIONS FOR CLEANUP ====================

def cleanup_label(label_id):
    """Delete test label from database."""
    try:
        from api_v2.db_layer.drawer_label_db import get_db_connection
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM DrawerLabels WHERE DrawerLabelId = ?", (label_id,))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Warning: Failed to cleanup label {label_id}: {e}")


# ==================== TESTS ====================

def test_create_label_success(auth_client_admin):
    """Test POST /api/v2/drawer-labels - Success scenario."""
    unique_label = f"TestLabel_{uuid.uuid4().hex[:8]}"
    
    response = auth_client_admin.post(
        "/api/v2/drawer-labels",
        json={"label_name": unique_label}
    )
    
    assert response.status_code == 201
    data = response.json()
    assert "label_id" in data
    assert isinstance(data["label_id"], int)
    assert data["success"] is True
    
    # Cleanup
    cleanup_label(data["label_id"])


def test_create_label_trimmed_input_success(auth_client_admin):
    """Test POST /api/v2/drawer-labels - Trimmed input success."""
    unique_label = f"  Trimmed_{uuid.uuid4().hex[:8]}  "
    
    response = auth_client_admin.post(
        "/api/v2/drawer-labels",
        json={"label_name": unique_label}
    )
    
    assert response.status_code == 201
    data = response.json()
    assert "label_id" in data
    
    # Cleanup
    cleanup_label(data["label_id"])


def test_create_label_short_name_error(auth_client_admin):
    """Test POST /api/v2/drawer-labels - Short name returns 400."""
    response = auth_client_admin.post(
        "/api/v2/drawer-labels",
        json={"label_name": "A"}
    )
    
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "2" in data["detail"].lower()  # Mentions minimum length


def test_create_label_duplicate_error(auth_client_admin):
    """Test POST /api/v2/drawer-labels - Duplicate name returns 400."""
    unique_label = f"Duplicate_{uuid.uuid4().hex[:8]}"
    
    # Create first label
    response1 = auth_client_admin.post(
        "/api/v2/drawer-labels",
        json={"label_name": unique_label}
    )
    assert response1.status_code == 201
    label_id = response1.json()["label_id"]
    
    try:
        # Try to create duplicate
        response2 = auth_client_admin.post(
            "/api/v2/drawer-labels",
            json={"label_name": unique_label}
        )
        assert response2.status_code == 400
        data = response2.json()
        assert "detail" in data
        assert "already exists" in data["detail"].lower() or "duplicate" in data["detail"].lower()
    finally:
        # Cleanup
        cleanup_label(label_id)


def test_get_labels_returns_created_label(auth_client_admin):
    """Test GET /api/v2/drawer-labels - Returns created label."""
    unique_label = f"GetTest_{uuid.uuid4().hex[:8]}"
    
    # Create label
    create_response = auth_client_admin.post(
        "/api/v2/drawer-labels",
        json={"label_name": unique_label}
    )
    assert create_response.status_code == 201
    label_id = create_response.json()["label_id"]
    
    try:
        # List labels
        list_response = auth_client_admin.get("/api/v2/drawer-labels")
        assert list_response.status_code == 200
        data = list_response.json()
        assert "labels" in data
        assert "total" in data
        assert isinstance(data["labels"], list)
        
        # Find created label in list
        found = any(label["label_id"] == label_id for label in data["labels"])
        assert found, f"Created label {label_id} not found in list"
    finally:
        # Cleanup
        cleanup_label(label_id)


def test_disable_label_success(auth_client_admin):
    """Test DELETE /api/v2/drawer-labels/{label_id} - Success scenario."""
    unique_label = f"DisableTest_{uuid.uuid4().hex[:8]}"
    
    # Create label
    create_response = auth_client_admin.post(
        "/api/v2/drawer-labels",
        json={"label_name": unique_label}
    )
    assert create_response.status_code == 201
    label_id = create_response.json()["label_id"]
    
    try:
        # Disable label
        delete_response = auth_client_admin.delete(f"/api/v2/drawer-labels/{label_id}")
        assert delete_response.status_code == 200
        data = delete_response.json()
        assert data["success"] is True
    finally:
        # Cleanup
        cleanup_label(label_id)


def test_get_labels_does_not_return_disabled_label(auth_client_admin):
    """Test GET /api/v2/drawer-labels - Disabled label not returned."""
    unique_label = f"HiddenTest_{uuid.uuid4().hex[:8]}"
    
    # Create label
    create_response = auth_client_admin.post(
        "/api/v2/drawer-labels",
        json={"label_name": unique_label}
    )
    assert create_response.status_code == 201
    label_id = create_response.json()["label_id"]
    
    try:
        # Disable label
        delete_response = auth_client_admin.delete(f"/api/v2/drawer-labels/{label_id}")
        assert delete_response.status_code == 200
        
        # List labels - should not include disabled label
        list_response = auth_client_admin.get("/api/v2/drawer-labels")
        assert list_response.status_code == 200
        data = list_response.json()
        
        # Ensure disabled label is not in list
        found = any(label["label_id"] == label_id for label in data["labels"])
        assert not found, f"Disabled label {label_id} should not be in list"
    finally:
        # Cleanup
        cleanup_label(label_id)


def test_forbidden_role_access(auth_client_forbidden):
    """Test all endpoints - Forbidden role returns 403."""
    # POST create label
    response1 = auth_client_forbidden.post(
        "/api/v2/drawer-labels",
        json={"label_name": "ForbiddenTest"}
    )
    assert response1.status_code == 403
    
    # GET list labels
    response2 = auth_client_forbidden.get("/api/v2/drawer-labels")
    assert response2.status_code == 403
    
    # DELETE disable label
    response3 = auth_client_forbidden.delete("/api/v2/drawer-labels/999")
    assert response3.status_code == 403


def test_unauthorized_access(unauth_client):
    """Test all endpoints - Unauthorized returns 401."""
    # POST create label
    response1 = unauth_client.post(
        "/api/v2/drawer-labels",
        json={"label_name": "UnauthorizedTest"}
    )
    assert response1.status_code == 401
    
    # GET list labels
    response2 = unauth_client.get("/api/v2/drawer-labels")
    assert response2.status_code == 401
    
    # DELETE disable label
    response3 = unauth_client.delete("/api/v2/drawer-labels/999")
    assert response3.status_code == 401
