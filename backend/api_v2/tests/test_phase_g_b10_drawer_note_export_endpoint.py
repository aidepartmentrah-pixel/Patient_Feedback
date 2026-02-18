"""
Phase G - G-B10: Drawer Notes Export Endpoint Tests

Integration tests for Drawer Notes Word export endpoint.

Test Coverage:
- Export with worker role (success - 200)
- Export document content verification
- Export with forbidden role (403)
- Export without authentication (401)

Uses real database and TestClient, no mocks.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock
import uuid
from io import BytesIO

from fastapi.testclient import TestClient

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

from main import app
from backend.api.dependencies.user_context import get_current_user
from backend.api.schemas.auth_models import CurrentUser
from api_v2.db_layer import drawer_note_db, drawer_label_db
from docx import Document


# ============================================================
# TEST CLIENT
# ============================================================
client = TestClient(app)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_mock_user(user_id=1, username="test_user", roles=None):
    """Create mock authenticated user."""
    if roles is None:
        roles = ["WORKER"]
    
    mock_user = Mock(spec=CurrentUser)
    mock_user.user_id = user_id
    mock_user.username = username
    mock_user.roles = roles
    mock_user.section_id = 1
    mock_user.organizational_unit_id = 1
    
    return mock_user


def override_auth(user):
    """Override authentication dependency with mock user."""
    def _get_mock_user():
        return user
    app.dependency_overrides[get_current_user] = _get_mock_user


def clear_auth():
    """Clear authentication override."""
    app.dependency_overrides.clear()


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def test_data_cleanup():
    """Track and cleanup test data."""
    created_label_ids = []
    created_note_ids = []
    
    yield created_label_ids, created_note_ids
    
    # Cleanup
    conn = drawer_note_db.get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Delete note-label links
        for note_id in created_note_ids:
            cursor.execute("DELETE FROM dbo.APP_DrawerNoteLabelLink WHERE NoteID = ?", (note_id,))
        
        # Delete notes
        for note_id in created_note_ids:
            cursor.execute("DELETE FROM dbo.APP_DrawerNote WHERE NoteID = ?", (note_id,))
        
        # Delete labels
        for label_id in created_label_ids:
            cursor.execute("DELETE FROM dbo.APP_DrawerLabel WHERE LabelID = ?", (label_id,))
        
        conn.commit()
    except Exception as e:
        print(f"Cleanup warning: {e}")
    finally:
        cursor.close()
        conn.close()


# ============================================================
# TESTS
# ============================================================

def test_export_worker_success(test_data_cleanup):
    """Test 1: GET /export/word - worker role - success (200)."""
    created_label_ids, created_note_ids = test_data_cleanup
    
    # Create test data
    unique_suffix = uuid.uuid4().hex[:8]
    label_name = f"ExportLabel_{unique_suffix}"
    label_id = drawer_label_db.insert_label(label_name)
    created_label_ids.append(label_id)
    
    note_text = f"Export test note {unique_suffix}"
    note_id = drawer_note_db.insert_note(note_text, 1, "Export Author")
    drawer_note_db.attach_labels_to_note(note_id, [label_id])
    created_note_ids.append(note_id)
    
    try:
        # Override auth with worker user
        worker_user = create_mock_user(roles=["WORKER"])
        override_auth(worker_user)
        
        # Call export endpoint
        response = client.get("/api/v2/drawer-notes/export/word")
        
        # Assertions
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        # Check content type
        content_type = response.headers.get("content-type")
        assert content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document", \
            f"Expected Word document MIME type, got {content_type}"
        
        # Check content disposition
        content_disposition = response.headers.get("content-disposition")
        assert content_disposition is not None, "Content-Disposition header missing"
        assert "attachment" in content_disposition, "Should be attachment"
        assert "drawer_notes_export.docx" in content_disposition, "Should have filename"
        
        # Check body length
        assert len(response.content) > 0, "Response body should not be empty"
        
    finally:
        clear_auth()


def test_export_document_loads_and_contains_content(test_data_cleanup):
    """Test 2: Load exported document and verify content."""
    created_label_ids, created_note_ids = test_data_cleanup
    
    # Create test data with unique identifiers
    unique_suffix = uuid.uuid4().hex[:8]
    label_name = f"ContentLabel_{unique_suffix}"
    label_id = drawer_label_db.insert_label(label_name)
    created_label_ids.append(label_id)
    
    note_text = f"Content verification note {unique_suffix}"
    author_name = f"Content Author {unique_suffix}"
    note_id = drawer_note_db.insert_note(note_text, 1, author_name)
    drawer_note_db.attach_labels_to_note(note_id, [label_id])
    created_note_ids.append(note_id)
    
    try:
        # Override auth with worker user
        worker_user = create_mock_user(roles=["WORKER"])
        override_auth(worker_user)
        
        # Call export endpoint
        response = client.get("/api/v2/drawer-notes/export/word")
        
        assert response.status_code == 200
        
        # Load document with python-docx
        doc = Document(BytesIO(response.content))
        
        # Extract all text
        full_text = "\n".join([para.text for para in doc.paragraphs])
        
        # Verify content
        assert "Drawer Notes Registry" in full_text, "Should contain document title"
        assert note_text in full_text, "Should contain note text"
        
    finally:
        clear_auth()


def test_export_forbidden_role(test_data_cleanup):
    """Test 3: GET /export/word - forbidden role - 403."""
    created_label_ids, created_note_ids = test_data_cleanup
    
    try:
        # Override auth with DOCTOR user (forbidden role)
        doctor_user = create_mock_user(roles=["DOCTOR"])
        override_auth(doctor_user)
        
        # Call export endpoint
        response = client.get("/api/v2/drawer-notes/export/word")
        
        # Should return 403 Forbidden
        assert response.status_code == 403, f"Expected 403, got {response.status_code}"
        
    finally:
        clear_auth()


def test_export_no_authentication():
    """Test 4: GET /export/word - no authentication - 401."""
    # No auth override
    clear_auth()
    
    # Call export endpoint
    response = client.get("/api/v2/drawer-notes/export/word")
    
    # Should return 401 Unauthorized
    assert response.status_code == 401, f"Expected 401, got {response.status_code}"


def test_export_with_software_admin_role(test_data_cleanup):
    """Test 5: GET /export/word - SOFTWARE_ADMIN role - success (200)."""
    created_label_ids, created_note_ids = test_data_cleanup
    
    # Create minimal test data
    unique_suffix = uuid.uuid4().hex[:8]
    label_id = drawer_label_db.insert_label(f"AdminLabel_{unique_suffix}")
    created_label_ids.append(label_id)
    
    note_id = drawer_note_db.insert_note(f"Admin test {unique_suffix}", 1, "Admin")
    drawer_note_db.attach_labels_to_note(note_id, [label_id])
    created_note_ids.append(note_id)
    
    try:
        # Override auth with SOFTWARE_ADMIN user
        admin_user = create_mock_user(roles=["SOFTWARE_ADMIN"])
        override_auth(admin_user)
        
        # Call export endpoint
        response = client.get("/api/v2/drawer-notes/export/word")
        
        # Should succeed
        assert response.status_code == 200
        assert len(response.content) > 0
        
    finally:
        clear_auth()
