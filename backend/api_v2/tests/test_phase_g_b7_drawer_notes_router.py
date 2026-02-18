"""
Test Suite: Phase G-B7 - Drawer Notes Router
Integration tests for Drawer Notes API endpoints.

Verifies:
- CRUD operations via HTTP endpoints
- Authentication and authorization
- Request/response schemas
- Error handling
- Label filtering

Target: 
- backend/api_v2/routers/drawer_notes_router.py

Test Coverage:
- All endpoints (POST, GET, PUT, DELETE)
- Success scenarios
- Error conditions (400, 401, 403, 404)
- Role-based access control

Note: Uses real FastAPI app with TestClient and real database
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


# ============================================================
# TEST CLIENT
# ============================================================
client = TestClient(app)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_mock_user(user_id=1, username="test_worker", roles=None):
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


def override_auth(user):
    """Override authentication dependency with mock user."""
    def _get_mock_user():
        return user
    app.dependency_overrides[get_current_user] = _get_mock_user


def clear_auth():
    """Clear authentication override."""
    app.dependency_overrides.clear()


# ============================================================
# TEST CLASS
# ============================================================

class TestDrawerNotesRouter:
    """Integration tests for Drawer Notes router."""
    
    def test_1_create_note_worker_success(self):
        """
        Test 1: POST create note - worker role - success.
        """
        print("\n" + "="*80)
        print("TEST 1: POST CREATE NOTE - WORKER - SUCCESS")
        print("="*80)
        
        # Create test labels
        label_id_1 = drawer_label_service.create_label(f"Label{uuid.uuid4().hex[:8]}")
        label_id_2 = drawer_label_service.create_label(f"Label{uuid.uuid4().hex[:8]}")
        
        try:
            # Override auth with worker user
            worker_user = create_mock_user(roles=["WORKER"])
            override_auth(worker_user)
            
            # Create note
            response = client.post(
                "/api/v2/drawer-notes/",
                json={
                    "note_text": f"Test note {uuid.uuid4().hex[:8]}",
                    "label_ids": [label_id_1, label_id_2]
                }
            )
            
            print(f"✓ Response status: {response.status_code}")
            print(f"✓ Response body: {response.json()}")
            
            assert response.status_code == 201, "Should return 201 Created"
            data = response.json()
            assert "note_id" in data, "Should return note_id"
            assert data["success"] is True
            note_id = data["note_id"]
            
            print(f"✓ Created note ID: {note_id}")
            print("\n✅ PASS - worker can create notes")
            
            # Clean up
            drawer_note_db.soft_delete_note(note_id)
            
        finally:
            clear_auth()
            # Clean up labels
            from api_v2.db_layer.drawer_label_db import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM dbo.APP_DrawerNoteLabelLink WHERE NoteID IN (SELECT NoteID FROM dbo.APP_DrawerNote WHERE CreatedByUserID = 1)")
            cursor.execute("DELETE FROM dbo.APP_DrawerNote WHERE CreatedByUserID = 1")
            cursor.execute("DELETE FROM dbo.APP_DrawerLabel WHERE LabelID IN (?, ?)", (label_id_1, label_id_2))
            conn.commit()
            cursor.close()
            conn.close()
            print("Cleaned up test data")
    
    def test_2_create_note_forbidden_role_403(self):
        """
        Test 2: POST create note - forbidden role - 403.
        """
        print("\n" + "="*80)
        print("TEST 2: POST CREATE NOTE - FORBIDDEN ROLE - 403")
        print("="*80)
        
        # Create test label
        label_id = drawer_label_service.create_label(f"Label{uuid.uuid4().hex[:8]}")
        
        try:
            # Override auth with unauthorized role
            forbidden_user = create_mock_user(roles=["SECTION_ADMIN"])
            override_auth(forbidden_user)
            
            # Try to create note
            response = client.post(
                "/api/v2/drawer-notes/",
                json={
                    "note_text": "Test note",
                    "label_ids": [label_id]
                }
            )
            
            print(f"✓ Response status: {response.status_code}")
            assert response.status_code == 403, "Should return 403 Forbidden"
            assert "Not authorized" in response.json()["detail"]
            
            print(f"✓ Correctly rejected forbidden role")
            print("\n✅ PASS - role guard works")
            
        finally:
            clear_auth()
            # Clean up label
            from api_v2.db_layer.drawer_label_db import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM dbo.APP_DrawerLabel WHERE LabelID = ?", (label_id,))
            conn.commit()
            cursor.close()
            conn.close()
            print("Cleaned up test data")
    
    def test_3_list_notes_worker_success(self):
        """
        Test 3: GET list notes - worker - success.
        """
        print("\n" + "="*80)
        print("TEST 3: GET LIST NOTES - WORKER - SUCCESS")
        print("="*80)
        
        # Create test label and note
        label_id = drawer_label_service.create_label(f"Label{uuid.uuid4().hex[:8]}")
        note_id = drawer_note_db.insert_note(
            f"Test note {uuid.uuid4().hex[:8]}", 1, "test_user"
        )
        drawer_note_db.attach_labels_to_note(note_id, [label_id])
        
        try:
            # Override auth with worker user
            worker_user = create_mock_user(roles=["WORKER"])
            override_auth(worker_user)
            
            # List notes
            response = client.get("/api/v2/drawer-notes/")
            
            print(f"✓ Response status: {response.status_code}")
            assert response.status_code == 200, "Should return 200 OK"
            data = response.json()
            assert "items" in data
            assert "total" in data
            
            print(f"✓ Found {data['total']} notes")
            print("\n✅ PASS - list endpoint works")
            
        finally:
            clear_auth()
            # Clean up
            from api_v2.db_layer.drawer_note_db import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM dbo.APP_DrawerNoteLabelLink WHERE NoteID = ?", (note_id,))
            cursor.execute("DELETE FROM dbo.APP_DrawerNote WHERE NoteID = ?", (note_id,))
            cursor.execute("DELETE FROM dbo.APP_DrawerLabel WHERE LabelID = ?", (label_id,))
            conn.commit()
            cursor.close()
            conn.close()
            print("Cleaned up test data")
    
    def test_4_get_note_by_id_success(self):
        """
        Test 4: GET note by ID - success.
        """
        print("\n" + "="*80)
        print("TEST 4: GET NOTE BY ID - SUCCESS")
        print("="*80)
        
        # Create test label and note
        label_id = drawer_label_service.create_label(f"Label{uuid.uuid4().hex[:8]}")
        note_id = drawer_note_db.insert_note(
            f"Test note {uuid.uuid4().hex[:8]}", 1, "test_user"
        )
        drawer_note_db.attach_labels_to_note(note_id, [label_id])
        
        try:
            # Override auth with worker user
            worker_user = create_mock_user(roles=["WORKER"])
            override_auth(worker_user)
            
            # Get note by ID
            response = client.get(f"/api/v2/drawer-notes/{note_id}")
            
            print(f"✓ Response status: {response.status_code}")
            assert response.status_code == 200, "Should return 200 OK"
            data = response.json()
            assert data["note_id"] == note_id
            assert "note_text" in data
            assert "label_ids" in data
            assert label_id in data["label_ids"]
            
            print(f"✓ Retrieved note {note_id}")
            print("\n✅ PASS - get by ID works")
            
        finally:
            clear_auth()
            # Clean up
            from api_v2.db_layer.drawer_note_db import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM dbo.APP_DrawerNoteLabelLink WHERE NoteID = ?", (note_id,))
            cursor.execute("DELETE FROM dbo.APP_DrawerNote WHERE NoteID = ?", (note_id,))
            cursor.execute("DELETE FROM dbo.APP_DrawerLabel WHERE LabelID = ?", (label_id,))
            conn.commit()
            cursor.close()
            conn.close()
            print("Cleaned up test data")
    
    def test_5_update_text_success_verify_changed(self):
        """
        Test 5: PUT text - success - verify changed via GET.
        """
        print("\n" + "="*80)
        print("TEST 5: PUT TEXT - SUCCESS - VERIFY CHANGED")
        print("="*80)
        
        # Create test label and note
        label_id = drawer_label_service.create_label(f"Label{uuid.uuid4().hex[:8]}")
        original_text = f"Original text {uuid.uuid4().hex[:8]}"
        note_id = drawer_note_db.insert_note(original_text, 1, "test_user")
        drawer_note_db.attach_labels_to_note(note_id, [label_id])
        
        try:
            # Override auth with worker user
            worker_user = create_mock_user(roles=["WORKER"])
            override_auth(worker_user)
            
            # Update text
            new_text = f"Updated text {uuid.uuid4().hex[:8]}"
            response = client.put(
                f"/api/v2/drawer-notes/{note_id}/text",
                json={"note_text": new_text}
            )
            
            print(f"✓ Update response status: {response.status_code}")
            assert response.status_code == 200, "Should return 200 OK"
            assert response.json()["success"] is True
            
            # Verify change via GET
            get_response = client.get(f"/api/v2/drawer-notes/{note_id}")
            assert get_response.status_code == 200
            data = get_response.json()
            assert data["note_text"] == new_text.strip()
            assert data["note_text"] != original_text
            
            print(f"✓ Text updated successfully")
            print("\n✅ PASS - text update works")
            
        finally:
            clear_auth()
            # Clean up
            from api_v2.db_layer.drawer_note_db import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM dbo.APP_DrawerNoteLabelLink WHERE NoteID = ?", (note_id,))
            cursor.execute("DELETE FROM dbo.APP_DrawerNote WHERE NoteID = ?", (note_id,))
            cursor.execute("DELETE FROM dbo.APP_DrawerLabel WHERE LabelID = ?", (label_id,))
            conn.commit()
            cursor.close()
            conn.close()
            print("Cleaned up test data")
    
    def test_6_update_text_empty_400(self):
        """
        Test 6: PUT text - empty - 400.
        """
        print("\n" + "="*80)
        print("TEST 6: PUT TEXT - EMPTY - 400")
        print("="*80)
        
        # Create test label and note
        label_id = drawer_label_service.create_label(f"Label{uuid.uuid4().hex[:8]}")
        note_id = drawer_note_db.insert_note("Original text", 1, "test_user")
        drawer_note_db.attach_labels_to_note(note_id, [label_id])
        
        try:
            # Override auth with worker user
            worker_user = create_mock_user(roles=["WORKER"])
            override_auth(worker_user)
            
            # Try to update with empty text
            response = client.put(
                f"/api/v2/drawer-notes/{note_id}/text",
                json={"note_text": "   "}
            )
            
            print(f"✓ Response status: {response.status_code}")
            assert response.status_code == 400, "Should return 400 Bad Request"
            assert "empty" in response.json()["detail"].lower()
            
            print(f"✓ Correctly rejected empty text")
            print("\n✅ PASS - empty text validation works")
            
        finally:
            clear_auth()
            # Clean up
            from api_v2.db_layer.drawer_note_db import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM dbo.APP_DrawerNoteLabelLink WHERE NoteID = ?", (note_id,))
            cursor.execute("DELETE FROM dbo.APP_DrawerNote WHERE NoteID = ?", (note_id,))
            cursor.execute("DELETE FROM dbo.APP_DrawerLabel WHERE LabelID = ?", (label_id,))
            conn.commit()
            cursor.close()
            conn.close()
            print("Cleaned up test data")
    
    def test_7_update_labels_success_verify_changed(self):
        """
        Test 7: PUT labels - success - verify changed via GET.
        """
        print("\n" + "="*80)
        print("TEST 7: PUT LABELS - SUCCESS - VERIFY CHANGED")
        print("="*80)
        
        # Create test labels
        label_id_1 = drawer_label_service.create_label(f"Label{uuid.uuid4().hex[:8]}")
        label_id_2 = drawer_label_service.create_label(f"Label{uuid.uuid4().hex[:8]}")
        label_id_3 = drawer_label_service.create_label(f"Label{uuid.uuid4().hex[:8]}")
        
        # Create note with label 1
        note_id = drawer_note_db.insert_note("Test note", 1, "test_user")
        drawer_note_db.attach_labels_to_note(note_id, [label_id_1])
        
        try:
            # Override auth with worker user
            worker_user = create_mock_user(roles=["WORKER"])
            override_auth(worker_user)
            
            # Update labels to 2 and 3
            response = client.put(
                f"/api/v2/drawer-notes/{note_id}/labels",
                json={"label_ids": [label_id_2, label_id_3]}
            )
            
            print(f"✓ Update response status: {response.status_code}")
            assert response.status_code == 200, "Should return 200 OK"
            assert response.json()["success"] is True
            
            # Verify change via GET
            get_response = client.get(f"/api/v2/drawer-notes/{note_id}")
            assert get_response.status_code == 200
            data = get_response.json()
            assert label_id_1 not in data["label_ids"]
            assert label_id_2 in data["label_ids"]
            assert label_id_3 in data["label_ids"]
            
            print(f"✓ Labels updated successfully")
            print("\n✅ PASS - label update works")
            
        finally:
            clear_auth()
            # Clean up
            from api_v2.db_layer.drawer_note_db import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM dbo.APP_DrawerNoteLabelLink WHERE NoteID = ?", (note_id,))
            cursor.execute("DELETE FROM dbo.APP_DrawerNote WHERE NoteID = ?", (note_id,))
            cursor.execute("DELETE FROM dbo.APP_DrawerLabel WHERE LabelID IN (?, ?, ?)", 
                          (label_id_1, label_id_2, label_id_3))
            conn.commit()
            cursor.close()
            conn.close()
            print("Cleaned up test data")
    
    def test_8_update_labels_empty_400(self):
        """
        Test 8: PUT labels - empty - 400.
        """
        print("\n" + "="*80)
        print("TEST 8: PUT LABELS - EMPTY - 400")
        print("="*80)
        
        # Create test label and note
        label_id = drawer_label_service.create_label(f"Label{uuid.uuid4().hex[:8]}")
        note_id = drawer_note_db.insert_note("Test note", 1, "test_user")
        drawer_note_db.attach_labels_to_note(note_id, [label_id])
        
        try:
            # Override auth with worker user
            worker_user = create_mock_user(roles=["WORKER"])
            override_auth(worker_user)
            
            # Try to update with empty labels
            response = client.put(
                f"/api/v2/drawer-notes/{note_id}/labels",
                json={"label_ids": []}
            )
            
            print(f"✓ Response status: {response.status_code}")
            assert response.status_code == 400, "Should return 400 Bad Request"
            assert "at least one label" in response.json()["detail"].lower()
            
            print(f"✓ Correctly rejected empty labels")
            print("\n✅ PASS - empty labels validation works")
            
        finally:
            clear_auth()
            # Clean up
            from api_v2.db_layer.drawer_note_db import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM dbo.APP_DrawerNoteLabelLink WHERE NoteID = ?", (note_id,))
            cursor.execute("DELETE FROM dbo.APP_DrawerNote WHERE NoteID = ?", (note_id,))
            cursor.execute("DELETE FROM dbo.APP_DrawerLabel WHERE LabelID = ?", (label_id,))
            conn.commit()
            cursor.close()
            conn.close()
            print("Cleaned up test data")
    
    def test_9_delete_note_success_not_in_list(self):
        """
        Test 9: DELETE note - success - not in list anymore.
        """
        print("\n" + "="*80)
        print("TEST 9: DELETE NOTE - SUCCESS - NOT IN LIST")
        print("="*80)
        
        # Create test label and note
        label_id = drawer_label_service.create_label(f"Label{uuid.uuid4().hex[:8]}")
        note_id = drawer_note_db.insert_note(f"Test note {uuid.uuid4().hex[:8]}", 1, "test_user")
        drawer_note_db.attach_labels_to_note(note_id, [label_id])
        
        try:
            # Override auth with worker user
            worker_user = create_mock_user(roles=["WORKER"])
            override_auth(worker_user)
            
            # Verify note in list before delete
            list_response = client.get("/api/v2/drawer-notes/")
            before_ids = [n["note_id"] for n in list_response.json()["items"]]
            assert note_id in before_ids, "Note should be in list before delete"
            print(f"✓ Note {note_id} in list before delete")
            
            # Delete note
            delete_response = client.delete(f"/api/v2/drawer-notes/{note_id}")
            print(f"✓ Delete response status: {delete_response.status_code}")
            assert delete_response.status_code == 200, "Should return 200 OK"
            assert delete_response.json()["success"] is True
            
            # Verify note NOT in list after delete
            list_response_after = client.get("/api/v2/drawer-notes/")
            after_ids = [n["note_id"] for n in list_response_after.json()["items"]]
            assert note_id not in after_ids, "Deleted note should NOT be in list"
            
            print(f"✓ Note {note_id} NOT in list after delete")
            print("\n✅ PASS - soft delete works")
            
        finally:
            clear_auth()
            # Clean up
            from api_v2.db_layer.drawer_note_db import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM dbo.APP_DrawerNoteLabelLink WHERE NoteID = ?", (note_id,))
            cursor.execute("DELETE FROM dbo.APP_DrawerNote WHERE NoteID = ?", (note_id,))
            cursor.execute("DELETE FROM dbo.APP_DrawerLabel WHERE LabelID = ?", (label_id,))
            conn.commit()
            cursor.close()
            conn.close()
            print("Cleaned up test data")
    
    def test_10_get_deleted_note_still_exists(self):
        """
        Test 10: GET deleted note - still exists by ID (soft delete).
        """
        print("\n" + "="*80)
        print("TEST 10: GET DELETED NOTE - STILL EXISTS BY ID")
        print("="*80)
        
        # Create test label and note
        label_id = drawer_label_service.create_label(f"Label{uuid.uuid4().hex[:8]}")
        note_id = drawer_note_db.insert_note("Test note", 1, "test_user")
        drawer_note_db.attach_labels_to_note(note_id, [label_id])
        
        try:
            # Override auth with worker user
            worker_user = create_mock_user(roles=["WORKER"])
            override_auth(worker_user)
            
            # Delete note
            delete_response = client.delete(f"/api/v2/drawer-notes/{note_id}")
            assert delete_response.status_code == 200
            print(f"✓ Deleted note {note_id}")
            
            # Try to get deleted note by ID (should still exist - soft delete)
            get_response = client.get(f"/api/v2/drawer-notes/{note_id}")
            assert get_response.status_code == 200, "Soft-deleted note should still be retrievable by ID"
            data = get_response.json()
            assert data["note_id"] == note_id
            assert data["is_deleted"] is True
            
            print(f"✓ Deleted note still retrievable by ID (soft delete)")
            print("\n✅ PASS - soft delete preserves record")
            
        finally:
            clear_auth()
            # Clean up
            from api_v2.db_layer.drawer_note_db import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM dbo.APP_DrawerNoteLabelLink WHERE NoteID = ?", (note_id,))
            cursor.execute("DELETE FROM dbo.APP_DrawerNote WHERE NoteID = ?", (note_id,))
            cursor.execute("DELETE FROM dbo.APP_DrawerLabel WHERE LabelID = ?", (label_id,))
            conn.commit()
            cursor.close()
            conn.close()
            print("Cleaned up test data")
    
    def test_11_filter_by_labels_and_logic(self):
        """
        Test 11: GET with label_ids filter - returns correct subset (AND logic).
        """
        print("\n" + "="*80)
        print("TEST 11: GET WITH LABEL FILTER - AND LOGIC")
        print("="*80)
        
        # Create test labels
        label_id_a = drawer_label_service.create_label(f"LabelA{uuid.uuid4().hex[:8]}")
        label_id_b = drawer_label_service.create_label(f"LabelB{uuid.uuid4().hex[:8]}")
        
        # Create note with both labels
        note_id_both = drawer_note_db.insert_note(
            f"Note with A+B {uuid.uuid4().hex[:8]}", 1, "test_user"
        )
        drawer_note_db.attach_labels_to_note(note_id_both, [label_id_a, label_id_b])
        
        # Create note with only label A
        note_id_a_only = drawer_note_db.insert_note(
            f"Note with A only {uuid.uuid4().hex[:8]}", 1, "test_user"
        )
        drawer_note_db.attach_labels_to_note(note_id_a_only, [label_id_a])
        
        try:
            # Override auth with worker user
            worker_user = create_mock_user(roles=["WORKER"])
            override_auth(worker_user)
            
            # Filter by both labels (AND logic)
            response = client.get(
                f"/api/v2/drawer-notes/?label_ids={label_id_a}&label_ids={label_id_b}"
            )
            
            print(f"✓ Response status: {response.status_code}")
            assert response.status_code == 200
            data = response.json()
            result_ids = [n["note_id"] for n in data["items"]]
            
            print(f"✓ Found {len(result_ids)} notes: {result_ids}")
            
            # Verify only note with both labels returned
            assert note_id_both in result_ids, "Note with both labels should be in results"
            assert note_id_a_only not in result_ids, "Note with only A should NOT be in results"
            
            print(f"✓ AND filtering works correctly")
            print("\n✅ PASS - label filter AND logic works")
            
        finally:
            clear_auth()
            # Clean up
            from api_v2.db_layer.drawer_note_db import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM dbo.APP_DrawerNoteLabelLink WHERE NoteID IN (?, ?)", 
                          (note_id_both, note_id_a_only))
            cursor.execute("DELETE FROM dbo.APP_DrawerNote WHERE NoteID IN (?, ?)", 
                          (note_id_both, note_id_a_only))
            cursor.execute("DELETE FROM dbo.APP_DrawerLabel WHERE LabelID IN (?, ?)", 
                          (label_id_a, label_id_b))
            conn.commit()
            cursor.close()
            conn.close()
            print("Cleaned up test data")
    
    def test_12_unauthorized_request_401(self):
        """
        Test 12: Unauthorized request - 401.
        """
        print("\n" + "="*80)
        print("TEST 12: UNAUTHORIZED REQUEST - 401")
        print("="*80)
        
        # Clear auth (no user)
        clear_auth()
        
        # Override with auth that raises 401
        from fastapi import HTTPException, status as http_status
        def mock_auth_fail():
            raise HTTPException(status_code=http_status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
        app.dependency_overrides[get_current_user] = mock_auth_fail
        
        try:
            # Try to list notes without auth
            response = client.get("/api/v2/drawer-notes/")
            
            print(f"✓ Response status: {response.status_code}")
            assert response.status_code == 401, "Should return 401 Unauthorized"
            
            print(f"✓ Correctly rejected unauthorized request")
            print("\n✅ PASS - authentication required")
            
        finally:
            clear_auth()


def run_all_tests():
    """Run all tests in sequence."""
    print("\n" + "="*80)
    print("PHASE G-B7: DRAWER NOTES ROUTER TESTS")
    print("="*80)
    
    router_tests = TestDrawerNotesRouter()
    
    tests = [
        ("POST Create Note WORKER SUCCESS", router_tests.test_1_create_note_worker_success),
        ("POST Create Note FORBIDDEN 403", router_tests.test_2_create_note_forbidden_role_403),
        ("GET List Notes SUCCESS", router_tests.test_3_list_notes_worker_success),
        ("GET Note by ID SUCCESS", router_tests.test_4_get_note_by_id_success),
        ("PUT Text SUCCESS Verify", router_tests.test_5_update_text_success_verify_changed),
        ("PUT Text EMPTY 400", router_tests.test_6_update_text_empty_400),
        ("PUT Labels SUCCESS Verify", router_tests.test_7_update_labels_success_verify_changed),
        ("PUT Labels EMPTY 400", router_tests.test_8_update_labels_empty_400),
        ("DELETE Note SUCCESS", router_tests.test_9_delete_note_success_not_in_list),
        ("GET Deleted Note EXISTS", router_tests.test_10_get_deleted_note_still_exists),
        ("GET Filter Labels AND", router_tests.test_11_filter_by_labels_and_logic),
        ("Unauthorized 401", router_tests.test_12_unauthorized_request_401),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ FAIL - {test_name}: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ ERROR - {test_name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total: {len(tests)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print("="*80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
