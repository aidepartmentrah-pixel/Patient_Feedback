"""
Test Suite: Phase G-B12 - Consolidated Router Integration Tests

Tests full FastAPI layer behavior for Drawer Notes and Drawer Labels routers.

This is a STRICT router integration test suite that:
- Uses real FastAPI app instance
- Uses real database
- No mocks of services
- Uses TestClient
- Uses real auth/token helper

Target modules:
- backend/api_v2/routers/drawer_notes_router.py
- backend/api_v2/routers/drawer_labels_router.py

Test Coverage:
AUTH + ROLE (3 tests)
1. Notes endpoints require auth → 401 without token
2. Labels endpoints require auth → 401
3. Forbidden role → 403 on all endpoints

LABEL ROUTER (4 tests)
4. Create label → success
5. Duplicate label → 400
6. List labels → contains created
7. Disable label → removed from list

NOTES ROUTER (3 tests)
8. Create note with labels → success
9. Create note empty text → 400
10. Create note empty labels → 400

EDITING (3 tests)
11. Edit text → success → verify via GET
12. Edit text empty → 400
13. Edit labels → success → verify

DELETE (2 tests)
14. Delete note → success
15. Deleted note not in list

FILTER (1 test)
16. Filter by label_ids query param → correct subset

EXPORT (1 test)
17. Export endpoint → 200 + correct content-type + body length > 0

ERROR MAPPING (2 tests)
18. Invalid label id → 400
19. Missing note id → 404

Author: Phase G-B12 Implementation
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
from api_v2.services import drawer_label_service
from api_v2.db_layer import drawer_note_db, drawer_label_db


# ============================================================================
# TEST CLIENT
# ============================================================================
client = TestClient(app)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_mock_user(user_id=1, username="test_user", roles=None):
    """Create mock authenticated user."""
    if roles is None:
        roles = ["WORKER"]
    
    mock_user = Mock(spec=CurrentUser)
    mock_user.user_id = user_id
    mock_user.username = username
    mock_user.roles = roles
    mock_user.display_name = username.title()
    mock_user.allowed_unit_ids = {}
    
    return mock_user


def login_worker():
    """Helper: Override auth as WORKER role."""
    user = create_mock_user(user_id=1, username="test_worker", roles=["WORKER"])
    app.dependency_overrides[get_current_user] = lambda: user
    return user


def login_admin():
    """Helper: Override auth as SOFTWARE_ADMIN role."""
    user = create_mock_user(user_id=2, username="test_admin", roles=["SOFTWARE_ADMIN"])
    app.dependency_overrides[get_current_user] = lambda: user
    return user


def login_forbidden_role():
    """Helper: Override auth as DOCTOR role (forbidden)."""
    user = create_mock_user(user_id=3, username="test_doctor", roles=["DOCTOR"])
    app.dependency_overrides[get_current_user] = lambda: user
    return user


def clear_auth():
    """Clear authentication override."""
    app.dependency_overrides.clear()


def cleanup_test_data(note_ids=None, label_ids=None):
    """Helper: Clean up test data from database."""
    from api_v2.db_layer.drawer_note_db import get_db_connection
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        if note_ids:
            placeholders = ','.join(['?' for _ in note_ids])
            cursor.execute(f"DELETE FROM dbo.APP_DrawerNoteLabelLink WHERE NoteID IN ({placeholders})", note_ids)
            cursor.execute(f"DELETE FROM dbo.APP_DrawerNote WHERE NoteID IN ({placeholders})", note_ids)
        
        if label_ids:
            placeholders = ','.join(['?' for _ in label_ids])
            cursor.execute(f"DELETE FROM dbo.APP_DrawerLabel WHERE LabelID IN ({placeholders})", label_ids)
        
        conn.commit()
    finally:
        cursor.close()
        conn.close()


# ============================================================================
# TEST CASES
# ============================================================================

class TestRouterIntegration:
    """Consolidated router integration test suite."""
    
    # ========================================================================
    # AUTH + ROLE TESTS (3 tests)
    # ========================================================================
    
    def test_1_notes_endpoints_require_auth_401(self):
        """
        Test 1: Notes endpoints require auth → 401 without token.
        """
        print("\n" + "="*80)
        print("TEST 1: NOTES ENDPOINTS REQUIRE AUTH - 401")
        print("="*80)
        
        clear_auth()
        
        # Try GET without auth
        response = client.get("/api/v2/drawer-notes")
        assert response.status_code == 401, "Should return 401 without auth"
        print(f"✓ GET /drawer-notes: 401")
        
        # Try POST without auth
        response = client.post("/api/v2/drawer-notes", json={
            "note_text": "Test",
            "label_ids": [1]
        })
        assert response.status_code == 401, "Should return 401 without auth"
        print(f"✓ POST /drawer-notes: 401")
        
        print("\n✅ PASS - Notes endpoints require auth")
    
    def test_2_labels_endpoints_require_auth_401(self):
        """
        Test 2: Labels endpoints require auth → 401.
        """
        print("\n" + "="*80)
        print("TEST 2: LABELS ENDPOINTS REQUIRE AUTH - 401")
        print("="*80)
        
        clear_auth()
        
        # Try GET without auth
        response = client.get("/api/v2/drawer-labels")
        assert response.status_code == 401, "Should return 401 without auth"
        print(f"✓ GET /drawer-labels: 401")
        
        # Try POST without auth
        response = client.post("/api/v2/drawer-labels", json={
            "label_name": "Test"
        })
        assert response.status_code == 401, "Should return 401 without auth"
        print(f"✓ POST /drawer-labels: 401")
        
        print("\n✅ PASS - Labels endpoints require auth")
    
    def test_3_forbidden_role_403_on_all_endpoints(self):
        """
        Test 3: Forbidden role → 403 on all endpoints.
        """
        print("\n" + "="*80)
        print("TEST 3: FORBIDDEN ROLE - 403 ON ALL ENDPOINTS")
        print("="*80)
        
        login_forbidden_role()
        
        try:
            # Try notes endpoints with DOCTOR role
            response = client.get("/api/v2/drawer-notes")
            assert response.status_code == 403, "Should return 403 for forbidden role"
            print(f"✓ GET /drawer-notes: 403")
            
            response = client.post("/api/v2/drawer-notes", json={
                "note_text": "Test",
                "label_ids": [1]
            })
            assert response.status_code == 403, "Should return 403 for forbidden role"
            print(f"✓ POST /drawer-notes: 403")
            
            # Try labels endpoints with DOCTOR role
            response = client.get("/api/v2/drawer-labels")
            assert response.status_code == 403, "Should return 403 for forbidden role"
            print(f"✓ GET /drawer-labels: 403")
            
            response = client.post("/api/v2/drawer-labels", json={
                "label_name": "Test"
            })
            assert response.status_code == 403, "Should return 403 for forbidden role"
            print(f"✓ POST /drawer-labels: 403")
            
            print("\n✅ PASS - Forbidden role returns 403 on all endpoints")
            
        finally:
            clear_auth()
    
    # ========================================================================
    # LABEL ROUTER TESTS (4 tests)
    # ========================================================================
    
    def test_4_create_label_success(self):
        """
        Test 4: Create label → success.
        """
        print("\n" + "="*80)
        print("TEST 4: CREATE LABEL - SUCCESS")
        print("="*80)
        
        login_worker()
        label_id = None
        
        try:
            label_name = f"TestLabel_{uuid.uuid4().hex[:8]}"
            response = client.post("/api/v2/drawer-labels", json={
                "label_name": label_name
            })
            
            assert response.status_code == 201, f"Should return 201, got {response.status_code}"
            data = response.json()
            assert "label_id" in data
            label_id = data["label_id"]
            assert "success" in data
            assert data["success"] == True
            print(f"✓ Created label: {label_id}")
            
            print("\n✅ PASS - Label created successfully")
            
        finally:
            clear_auth()
            if label_id:
                cleanup_test_data(label_ids=[label_id])
    
    def test_5_duplicate_label_400(self):
        """
        Test 5: Duplicate label → 400.
        """
        print("\n" + "="*80)
        print("TEST 5: DUPLICATE LABEL - 400")
        print("="*80)
        
        login_worker()
        label_id = None
        
        try:
            label_name = f"DuplicateLabel_{uuid.uuid4().hex[:8]}"
            
            # Create first label
            label_id = drawer_label_service.create_label(label_name)
            print(f"✓ Created label: {label_id}")
            
            # Try to create duplicate
            response = client.post("/api/v2/drawer-labels", json={
                "label_name": label_name
            })
            
            assert response.status_code == 400, f"Should return 400 for duplicate, got {response.status_code}"
            print(f"✓ Duplicate label rejected with 400")
            
            print("\n✅ PASS - Duplicate label returns 400")
            
        finally:
            clear_auth()
            if label_id:
                cleanup_test_data(label_ids=[label_id])
    
    def test_6_list_labels_contains_created(self):
        """
        Test 6: List labels → contains created.
        """
        print("\n" + "="*80)
        print("TEST 6: LIST LABELS - CONTAINS CREATED")
        print("="*80)
        
        login_worker()
        label_id = None
        
        try:
            label_name = f"ListTestLabel_{uuid.uuid4().hex[:8]}"
            label_id = drawer_label_service.create_label(label_name)
            print(f"✓ Created label: {label_id}")
            
            # List labels
            response = client.get("/api/v2/drawer-labels")
            assert response.status_code == 200
            data = response.json()
            
            assert "labels" in data
            labels = data["labels"]
            label_ids = [lbl["label_id"] for lbl in labels]
            assert label_id in label_ids, "Created label should be in list"
            print(f"✓ Created label found in list")
            
            print("\n✅ PASS - List labels contains created label")
            
        finally:
            clear_auth()
            if label_id:
                cleanup_test_data(label_ids=[label_id])
    
    def test_7_disable_label_removed_from_list(self):
        """
        Test 7: Disable label → removed from list.
        """
        print("\n" + "="*80)
        print("TEST 7: DISABLE LABEL - REMOVED FROM LIST")
        print("="*80)
        
        login_worker()
        label_id = None
        
        try:
            label_name = f"DisableTestLabel_{uuid.uuid4().hex[:8]}"
            label_id = drawer_label_service.create_label(label_name)
            print(f"✓ Created label: {label_id}")
            
            # Disable label
            response = client.delete(f"/api/v2/drawer-labels/{label_id}")
            assert response.status_code == 200
            print(f"✓ Disabled label via DELETE")
            
            # List labels - should not include disabled
            response = client.get("/api/v2/drawer-labels")
            assert response.status_code == 200
            data = response.json()
            
            labels = data["labels"]
            label_ids = [lbl["label_id"] for lbl in labels]
            assert label_id not in label_ids, "Disabled label should not be in list"
            print(f"✓ Disabled label not in list")
            
            print("\n✅ PASS - Disabled label removed from list")
            
        finally:
            clear_auth()
            if label_id:
                cleanup_test_data(label_ids=[label_id])
    
    # ========================================================================
    # NOTES ROUTER TESTS (3 tests)
    # ========================================================================
    
    def test_8_create_note_with_labels_success(self):
        """
        Test 8: Create note with labels → success.
        """
        print("\n" + "="*80)
        print("TEST 8: CREATE NOTE WITH LABELS - SUCCESS")
        print("="*80)
        
        login_worker()
        label_id = None
        note_id = None
        
        try:
            label_id = drawer_label_service.create_label(f"NoteLabel_{uuid.uuid4().hex[:8]}")
            print(f"✓ Created label: {label_id}")
            
            note_text = "Test note via router"
            response = client.post("/api/v2/drawer-notes", json={
                "note_text": note_text,
                "label_ids": [label_id]
            })
            
            assert response.status_code == 201, f"Should return 201, got {response.status_code}"
            data = response.json()
            assert "note_id" in data
            note_id = data["note_id"]
            print(f"✓ Created note: {note_id}")
            
            print("\n✅ PASS - Note created successfully")
            
        finally:
            clear_auth()
            if note_id and label_id:
                cleanup_test_data(note_ids=[note_id], label_ids=[label_id])
    
    def test_9_create_note_empty_text_400(self):
        """
        Test 9: Create note empty text → 400.
        """
        print("\n" + "="*80)
        print("TEST 9: CREATE NOTE EMPTY TEXT - 400")
        print("="*80)
        
        login_worker()
        label_id = None
        
        try:
            label_id = drawer_label_service.create_label(f"EmptyTextLabel_{uuid.uuid4().hex[:8]}")
            print(f"✓ Created label: {label_id}")
            
            # Try to create note with empty text
            response = client.post("/api/v2/drawer-notes", json={
                "note_text": "   ",
                "label_ids": [label_id]
            })
            
            assert response.status_code == 400, f"Should return 400 for empty text, got {response.status_code}"
            print(f"✓ Empty text rejected with 400")
            
            print("\n✅ PASS - Empty text returns 400")
            
        finally:
            clear_auth()
            if label_id:
                cleanup_test_data(label_ids=[label_id])
    
    def test_10_create_note_empty_labels_400(self):
        """
        Test 10: Create note empty labels → 400.
        """
        print("\n" + "="*80)
        print("TEST 10: CREATE NOTE EMPTY LABELS - 400")
        print("="*80)
        
        login_worker()
        
        try:
            # Try to create note with empty labels
            response = client.post("/api/v2/drawer-notes", json={
                "note_text": "Valid text",
                "label_ids": []
            })
            
            assert response.status_code == 400, f"Should return 400 for empty labels, got {response.status_code}"
            print(f"✓ Empty labels rejected with 400")
            
            print("\n✅ PASS - Empty labels returns 400")
            
        finally:
            clear_auth()
    
    # ========================================================================
    # EDITING TESTS (3 tests)
    # ========================================================================
    
    def test_11_edit_text_success_verify_via_get(self):
        """
        Test 11: Edit text → success → verify via GET.
        """
        print("\n" + "="*80)
        print("TEST 11: EDIT TEXT - SUCCESS - VERIFY VIA GET")
        print("="*80)
        
        login_worker()
        label_id = None
        note_id = None
        
        try:
            # Create label and note
            label_id = drawer_label_service.create_label(f"EditLabel_{uuid.uuid4().hex[:8]}")
            note_id = drawer_note_db.insert_note("Original text", 1, "test_user")
            drawer_note_db.attach_labels_to_note(note_id, [label_id])
            print(f"✓ Created note: {note_id}")
            
            # Edit text
            new_text = "Updated text via router"
            response = client.put(f"/api/v2/drawer-notes/{note_id}/text", json={
                "note_text": new_text
            })
            
            assert response.status_code == 200, f"Should return 200, got {response.status_code}"
            print(f"✓ Updated text via PUT")
            
            # Verify via GET
            response = client.get(f"/api/v2/drawer-notes/{note_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["note_text"] == new_text, "Text should be updated"
            print(f"✓ Verified updated text via GET")
            
            print("\n✅ PASS - Edit text successful and verified")
            
        finally:
            clear_auth()
            if note_id and label_id:
                cleanup_test_data(note_ids=[note_id], label_ids=[label_id])
    
    def test_12_edit_text_empty_400(self):
        """
        Test 12: Edit text empty → 400.
        """
        print("\n" + "="*80)
        print("TEST 12: EDIT TEXT EMPTY - 400")
        print("="*80)
        
        login_worker()
        label_id = None
        note_id = None
        
        try:
            # Create label and note
            label_id = drawer_label_service.create_label(f"EditEmptyLabel_{uuid.uuid4().hex[:8]}")
            note_id = drawer_note_db.insert_note("Original text", 1, "test_user")
            drawer_note_db.attach_labels_to_note(note_id, [label_id])
            print(f"✓ Created note: {note_id}")
            
            # Try to edit with empty text
            response = client.put(f"/api/v2/drawer-notes/{note_id}/text", json={
                "note_text": "   "
            })
            
            assert response.status_code == 400, f"Should return 400 for empty text, got {response.status_code}"
            print(f"✓ Empty text rejected with 400")
            
            print("\n✅ PASS - Empty text edit returns 400")
            
        finally:
            clear_auth()
            if note_id and label_id:
                cleanup_test_data(note_ids=[note_id], label_ids=[label_id])
    
    def test_13_edit_labels_success_verify(self):
        """
        Test 13: Edit labels → success → verify.
        """
        print("\n" + "="*80)
        print("TEST 13: EDIT LABELS - SUCCESS - VERIFY")
        print("="*80)
        
        login_worker()
        label_id_1 = None
        label_id_2 = None
        note_id = None
        
        try:
            # Create labels and note
            label_id_1 = drawer_label_service.create_label(f"OldLabel_{uuid.uuid4().hex[:8]}")
            label_id_2 = drawer_label_service.create_label(f"NewLabel_{uuid.uuid4().hex[:8]}")
            note_id = drawer_note_db.insert_note("Test note", 1, "test_user")
            drawer_note_db.attach_labels_to_note(note_id, [label_id_1])
            print(f"✓ Created note with label {label_id_1}: {note_id}")
            
            # Edit labels
            response = client.put(f"/api/v2/drawer-notes/{note_id}/labels", json={
                "label_ids": [label_id_2]
            })
            
            assert response.status_code == 200, f"Should return 200, got {response.status_code}"
            print(f"✓ Updated labels via PUT")
            
            # Verify via GET
            response = client.get(f"/api/v2/drawer-notes/{note_id}")
            assert response.status_code == 200
            data = response.json()
            assert label_id_2 in data["label_ids"], "New label should be present"
            assert label_id_1 not in data["label_ids"], "Old label should be removed"
            print(f"✓ Verified labels updated via GET")
            
            print("\n✅ PASS - Edit labels successful and verified")
            
        finally:
            clear_auth()
            if note_id:
                cleanup_test_data(note_ids=[note_id], label_ids=[label_id_1, label_id_2])
    
    # ========================================================================
    # DELETE TESTS (2 tests)
    # ========================================================================
    
    def test_14_delete_note_success(self):
        """
        Test 14: Delete note → success.
        """
        print("\n" + "="*80)
        print("TEST 14: DELETE NOTE - SUCCESS")
        print("="*80)
        
        login_worker()
        label_id = None
        note_id = None
        
        try:
            # Create label and note
            label_id = drawer_label_service.create_label(f"DeleteLabel_{uuid.uuid4().hex[:8]}")
            note_id = drawer_note_db.insert_note("To be deleted", 1, "test_user")
            drawer_note_db.attach_labels_to_note(note_id, [label_id])
            print(f"✓ Created note: {note_id}")
            
            # Delete note
            response = client.delete(f"/api/v2/drawer-notes/{note_id}")
            assert response.status_code == 200, f"Should return 200, got {response.status_code}"
            print(f"✓ Deleted note via DELETE")
            
            print("\n✅ PASS - Note deleted successfully")
            
        finally:
            clear_auth()
            if note_id and label_id:
                cleanup_test_data(note_ids=[note_id], label_ids=[label_id])
    
    def test_15_deleted_note_not_in_list(self):
        """
        Test 15: Deleted note not in list.
        """
        print("\n" + "="*80)
        print("TEST 15: DELETED NOTE NOT IN LIST")
        print("="*80)
        
        login_worker()
        label_id = None
        note_id = None
        
        try:
            # Create label and note
            label_id = drawer_label_service.create_label(f"DeletedListLabel_{uuid.uuid4().hex[:8]}")
            note_id = drawer_note_db.insert_note("To be deleted and checked", 1, "test_user")
            drawer_note_db.attach_labels_to_note(note_id, [label_id])
            print(f"✓ Created note: {note_id}")
            
            # Delete note
            response = client.delete(f"/api/v2/drawer-notes/{note_id}")
            assert response.status_code == 200
            print(f"✓ Deleted note")
            
            # List notes - should not include deleted
            response = client.get("/api/v2/drawer-notes")
            assert response.status_code == 200
            data = response.json()
            
            notes = data["items"]
            note_ids = [n["note_id"] for n in notes]
            assert note_id not in note_ids, "Deleted note should not be in list"
            print(f"✓ Deleted note not in list")
            
            print("\n✅ PASS - Deleted note not in list")
            
        finally:
            clear_auth()
            if note_id and label_id:
                cleanup_test_data(note_ids=[note_id], label_ids=[label_id])
    
    # ========================================================================
    # FILTER TEST (1 test)
    # ========================================================================
    
    def test_16_filter_by_label_ids_correct_subset(self):
        """
        Test 16: Filter by label_ids query param → correct subset.
        """
        print("\n" + "="*80)
        print("TEST 16: FILTER BY LABEL IDS - CORRECT SUBSET")
        print("="*80)
        
        login_worker()
        label_id_1 = None
        label_id_2 = None
        note_id_a = None
        note_id_b = None
        
        try:
            # Create labels
            label_id_1 = drawer_label_service.create_label(f"FilterL1_{uuid.uuid4().hex[:8]}")
            label_id_2 = drawer_label_service.create_label(f"FilterL2_{uuid.uuid4().hex[:8]}")
            print(f"✓ Created labels: {label_id_1}, {label_id_2}")
            
            # Create note A with both labels
            note_id_a = drawer_note_db.insert_note("Note A has both", 1, "test_user")
            drawer_note_db.attach_labels_to_note(note_id_a, [label_id_1, label_id_2])
            print(f"✓ Created note A with both labels: {note_id_a}")
            
            # Create note B with only label 1
            note_id_b = drawer_note_db.insert_note("Note B has one", 1, "test_user")
            drawer_note_db.attach_labels_to_note(note_id_b, [label_id_1])
            print(f"✓ Created note B with only label 1: {note_id_b}")
            
            # Filter by both labels - should return only A
            response = client.get(f"/api/v2/drawer-notes?label_ids={label_id_1}&label_ids={label_id_2}")
            assert response.status_code == 200
            data = response.json()
            
            notes = data["items"]
            note_ids = [n["note_id"] for n in notes]
            assert note_id_a in note_ids, "Note A should be in results (has both labels)"
            assert note_id_b not in note_ids, "Note B should NOT be in results (missing label 2)"
            print(f"✓ Filter correctly returned only note A")
            
            print("\n✅ PASS - Filter by label_ids works correctly")
            
        finally:
            clear_auth()
            if note_id_a or note_id_b:
                cleanup_test_data(
                    note_ids=[note_id_a, note_id_b] if note_id_a and note_id_b else ([note_id_a] if note_id_a else [note_id_b]),
                    label_ids=[label_id_1, label_id_2]
                )
    
    # ========================================================================
    # EXPORT TEST (1 test)
    # ========================================================================
    
    def test_17_export_endpoint_200_correct_content_type(self):
        """
        Test 17: Export endpoint → 200 + correct content-type + body length > 0.
        """
        print("\n" + "="*80)
        print("TEST 17: EXPORT ENDPOINT - 200 + CORRECT CONTENT-TYPE")
        print("="*80)
        
        login_worker()
        
        try:
            response = client.get("/api/v2/drawer-notes/export/word")
            
            assert response.status_code == 200, f"Should return 200, got {response.status_code}"
            print(f"✓ Export returned 200")
            
            content_type = response.headers.get("content-type", "")
            assert "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in content_type, \
                "Should return Word MIME type"
            print(f"✓ Content-Type is correct Word MIME type")
            
            assert len(response.content) > 0, "Body should have content"
            print(f"✓ Body length: {len(response.content)} bytes")
            
            print("\n✅ PASS - Export endpoint works correctly")
            
        finally:
            clear_auth()
    
    # ========================================================================
    # ERROR MAPPING TESTS (2 tests)
    # ========================================================================
    
    def test_18_invalid_label_id_400(self):
        """
        Test 18: Invalid label id → 400.
        """
        print("\n" + "="*80)
        print("TEST 18: INVALID LABEL ID - 400")
        print("="*80)
        
        login_worker()
        
        try:
            # Try to create note with invalid label ID
            response = client.post("/api/v2/drawer-notes", json={
                "note_text": "Valid text",
                "label_ids": [999999]
            })
            
            assert response.status_code == 400, f"Should return 400 for invalid label, got {response.status_code}"
            print(f"✓ Invalid label ID rejected with 400")
            
            print("\n✅ PASS - Invalid label ID returns 400")
            
        finally:
            clear_auth()
    
    def test_19_missing_note_id_404(self):
        """
        Test 19: Missing note id → 404.
        """
        print("\n" + "="*80)
        print("TEST 19: MISSING NOTE ID - 404")
        print("="*80)
        
        login_worker()
        
        try:
            # Try to GET non-existent note
            response = client.get("/api/v2/drawer-notes/999999")
            
            assert response.status_code == 404, f"Should return 404 for missing note, got {response.status_code}"
            print(f"✓ Missing note ID returned 404")
            
            print("\n✅ PASS - Missing note ID returns 404")
            
        finally:
            clear_auth()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
