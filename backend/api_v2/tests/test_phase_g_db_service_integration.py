"""
Test Suite: Phase G-B11 - Consolidated DB + Service Integration Tests

Tests drawer notes DB layer and Service layer working together correctly.

This is a STRICT integration test suite that:
- Uses real SQL Server database
- No mocks
- Tests real db_layer + service layer
- Uses isolated test data per test
- Each test cleans up its data

Target modules:
- backend/api_v2/services/drawer_note_service.py
- backend/api_v2/services/drawer_label_service.py
- backend/api_v2/db_layer/drawer_note_db.py
- backend/api_v2/db_layer/drawer_label_db.py

Test Coverage:
1. Create note with multiple labels - success
2. Edit note text - persisted
3. Replace note labels - persisted
4. Reject create note with inactive label
5. Reject edit labels with inactive label
6. Soft delete note - effects visible
7. Reject edit text on deleted note
8. Label validation - partial invalid set fails
9. Filter by labels - ALL labels semantics
10. Pagination works

Author: Phase G-B11 Implementation
"""

import pytest
import sys
from pathlib import Path
import uuid

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

from api_v2.services import drawer_note_service
from api_v2.services import drawer_label_service
from api_v2.db_layer import drawer_note_db
from api_v2.db_layer import drawer_label_db


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_test_label(label_name=None):
    """
    Helper: Create a test label via label service.
    
    Args:
        label_name (str, optional): Label name. Defaults to random.
    
    Returns:
        int: Label ID
    """
    if label_name is None:
        label_name = f"TestLabel_{uuid.uuid4().hex[:8]}"
    return drawer_label_service.create_label(label_name)


def create_test_note(note_text=None, label_ids=None):
    """
    Helper: Create a test note via note service.
    
    Args:
        note_text (str, optional): Note text. Defaults to random.
        label_ids (list, optional): Label IDs. Defaults to creating one label.
    
    Returns:
        int: Note ID
    """
    if note_text is None:
        note_text = f"Test note {uuid.uuid4().hex[:8]}"
    
    if label_ids is None:
        label_id = create_test_label()
        label_ids = [label_id]
    
    return drawer_note_service.create_note_with_labels(
        note_text=note_text,
        label_ids=label_ids,
        created_by_user_id=1,
        created_by_name="test_user"
    )


def cleanup_test_data(note_ids=None, label_ids=None):
    """
    Helper: Clean up test data from database.
    
    Args:
        note_ids (list, optional): Note IDs to delete
        label_ids (list, optional): Label IDs to delete
    """
    from api_v2.db_layer.drawer_note_db import get_db_connection
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Delete note-label links
        if note_ids:
            placeholders = ','.join(['?' for _ in note_ids])
            cursor.execute(f"DELETE FROM dbo.APP_DrawerNoteLabelLink WHERE NoteID IN ({placeholders})", note_ids)
        
        # Delete notes
        if note_ids:
            placeholders = ','.join(['?' for _ in note_ids])
            cursor.execute(f"DELETE FROM dbo.APP_DrawerNote WHERE NoteID IN ({placeholders})", note_ids)
        
        # Delete labels
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

class TestDBServiceIntegration:
    """Consolidated integration test suite for DB + Service layers."""
    
    def test_1_create_note_with_multiple_labels_success(self):
        """
        Test 1: Create note with multiple labels - verify DB persistence.
        
        Verifies:
        - Note created via service
        - DB note row exists
        - Link table has correct number of rows
        """
        print("\n" + "="*80)
        print("TEST 1: CREATE NOTE WITH MULTIPLE LABELS - SUCCESS")
        print("="*80)
        
        # Create 2 test labels
        label_id_1 = create_test_label("Label_1")
        label_id_2 = create_test_label("Label_2")
        print(f"✓ Created labels: {label_id_1}, {label_id_2}")
        
        note_id = None
        try:
            # Create note with both labels
            note_text = "Integration test note with 2 labels"
            note_id = drawer_note_service.create_note_with_labels(
                note_text=note_text,
                label_ids=[label_id_1, label_id_2],
                created_by_user_id=1,
                created_by_name="test_integration"
            )
            print(f"✓ Created note: {note_id}")
            
            # Verify DB note row exists
            note = drawer_note_db.get_note_by_id(note_id)
            assert note is not None, "Note should exist in database"
            assert note['note_text'] == note_text
            print(f"✓ Verified note exists in DB")
            
            # Verify link table has 2 rows
            label_ids_from_db = drawer_note_db.get_note_label_ids(note_id)
            assert len(label_ids_from_db) == 2, "Should have 2 label links"
            assert label_id_1 in label_ids_from_db
            assert label_id_2 in label_ids_from_db
            print(f"✓ Verified 2 links in APP_DrawerNoteLabelLink")
            
            print("\n✅ PASS - Note created with multiple labels and persisted correctly")
            
        finally:
            cleanup_test_data(
                note_ids=[note_id] if note_id else None,
                label_ids=[label_id_1, label_id_2]
            )
    
    def test_2_edit_note_text_persisted(self):
        """
        Test 2: Edit note text - verify persistence in DB.
        
        Verifies:
        - Note text updated via service
        - DB reflects new text when queried directly
        """
        print("\n" + "="*80)
        print("TEST 2: EDIT NOTE TEXT - PERSISTED")
        print("="*80)
        
        label_id = create_test_label()
        note_id = create_test_note(label_ids=[label_id])
        print(f"✓ Created test note: {note_id}")
        
        try:
            # Update text via service
            new_text = "Updated text via integration test"
            drawer_note_service.edit_note_text(note_id, new_text)
            print(f"✓ Updated note text via service")
            
            # Read via DB layer directly
            note = drawer_note_db.get_note_by_id(note_id)
            assert note is not None
            assert note['note_text'] == new_text, "DB should reflect updated text"
            print(f"✓ Verified text persisted in DB: '{note['note_text']}'")
            
            print("\n✅ PASS - Note text edit persisted correctly")
            
        finally:
            cleanup_test_data(note_ids=[note_id], label_ids=[label_id])
    
    def test_3_replace_note_labels_persisted(self):
        """
        Test 3: Replace note labels - verify old links removed, new links present.
        
        Verifies:
        - Old label links removed from DB
        - New label links added to DB
        """
        print("\n" + "="*80)
        print("TEST 3: REPLACE NOTE LABELS - PERSISTED")
        print("="*80)
        
        # Create labels and note
        label_id_1 = create_test_label("OldLabel1")
        label_id_2 = create_test_label("OldLabel2")
        label_id_3 = create_test_label("NewLabel3")
        print(f"✓ Created labels: {label_id_1}, {label_id_2}, {label_id_3}")
        
        note_id = create_test_note(label_ids=[label_id_1, label_id_2])
        print(f"✓ Created note with labels 1 and 2: {note_id}")
        
        try:
            # Replace labels with new set
            drawer_note_service.edit_note_labels(note_id, [label_id_3])
            print(f"✓ Replaced labels via service (now only label 3)")
            
            # Verify old links removed, new links present
            label_ids_from_db = drawer_note_db.get_note_label_ids(note_id)
            assert len(label_ids_from_db) == 1, "Should have only 1 label"
            assert label_id_3 in label_ids_from_db, "New label should be present"
            assert label_id_1 not in label_ids_from_db, "Old label 1 should be removed"
            assert label_id_2 not in label_ids_from_db, "Old label 2 should be removed"
            print(f"✓ Verified old links removed, new link present")
            
            print("\n✅ PASS - Label replacement persisted correctly")
            
        finally:
            cleanup_test_data(
                note_ids=[note_id],
                label_ids=[label_id_1, label_id_2, label_id_3]
            )
    
    def test_4_reject_create_note_with_inactive_label(self):
        """
        Test 4: Reject create note with inactive label.
        
        Verifies:
        - Create note fails when label is inactive
        - Raises ValueError
        """
        print("\n" + "="*80)
        print("TEST 4: REJECT CREATE NOTE WITH INACTIVE LABEL")
        print("="*80)
        
        label_id = create_test_label()
        print(f"✓ Created label: {label_id}")
        
        try:
            # Disable label via service
            drawer_label_service.disable_label(label_id)
            print(f"✓ Disabled label via service")
            
            # Try to create note with inactive label - should fail
            with pytest.raises(ValueError, match="Invalid or inactive label IDs"):
                drawer_note_service.create_note_with_labels(
                    note_text="Should not be created",
                    label_ids=[label_id],
                    created_by_user_id=1,
                    created_by_name="test_user"
                )
            print(f"✓ Correctly rejected note creation with inactive label")
            
            print("\n✅ PASS - Create note rejected with inactive label")
            
        finally:
            cleanup_test_data(label_ids=[label_id])
    
    def test_5_reject_edit_labels_with_inactive_label(self):
        """
        Test 5: Reject edit labels with inactive label.
        
        Verifies:
        - Edit labels fails when new label is inactive
        - Raises ValueError
        """
        print("\n" + "="*80)
        print("TEST 5: REJECT EDIT LABELS WITH INACTIVE LABEL")
        print("="*80)
        
        label_id_active = create_test_label("ActiveLabel")
        label_id_inactive = create_test_label("InactiveLabel")
        print(f"✓ Created labels: {label_id_active} (active), {label_id_inactive} (will disable)")
        
        note_id = create_test_note(label_ids=[label_id_active])
        print(f"✓ Created note: {note_id}")
        
        try:
            # Disable second label
            drawer_label_service.disable_label(label_id_inactive)
            print(f"✓ Disabled label {label_id_inactive}")
            
            # Try to edit note labels to include inactive label - should fail
            with pytest.raises(ValueError, match="Invalid or inactive label IDs"):
                drawer_note_service.edit_note_labels(note_id, [label_id_inactive])
            print(f"✓ Correctly rejected label edit with inactive label")
            
            print("\n✅ PASS - Edit labels rejected with inactive label")
            
        finally:
            cleanup_test_data(
                note_ids=[note_id],
                label_ids=[label_id_active, label_id_inactive]
            )
    
    def test_6_soft_delete_note_effects_visible(self):
        """
        Test 6: Soft delete note - verify effects.
        
        Verifies:
        - list_notes returns none after soft delete
        - get_note_by_id still returns row with is_deleted=1
        """
        print("\n" + "="*80)
        print("TEST 6: SOFT DELETE NOTE - EFFECTS VISIBLE")
        print("="*80)
        
        label_id = create_test_label()
        note_id = create_test_note(label_ids=[label_id])
        print(f"✓ Created note: {note_id}")
        
        try:
            # Soft delete via service
            drawer_note_service.soft_delete_note(note_id)
            print(f"✓ Soft deleted note")
            
            # Verify list_notes does not return it
            all_notes = drawer_note_service.list_notes(limit=1000)
            note_ids_in_list = [n['note_id'] for n in all_notes]
            assert note_id not in note_ids_in_list, "Deleted note should not appear in list"
            print(f"✓ Verified note not in list_notes results")
            
            # Verify get_note_by_id still returns it with is_deleted=1
            note = drawer_note_db.get_note_by_id(note_id)
            assert note is not None, "Deleted note should still exist in DB"
            assert note['is_deleted'] == 1, "Note should have is_deleted=1"
            print(f"✓ Verified get_note_by_id returns note with is_deleted=1")
            
            print("\n✅ PASS - Soft delete effects verified")
            
        finally:
            cleanup_test_data(note_ids=[note_id], label_ids=[label_id])
    
    def test_7_reject_edit_text_on_deleted_note(self):
        """
        Test 7: Reject edit text on deleted note.
        
        Verifies:
        - Edit text fails on deleted note
        - Raises ValueError
        """
        print("\n" + "="*80)
        print("TEST 7: REJECT EDIT TEXT ON DELETED NOTE")
        print("="*80)
        
        label_id = create_test_label()
        note_id = create_test_note(label_ids=[label_id])
        print(f"✓ Created note: {note_id}")
        
        try:
            # Soft delete
            drawer_note_service.soft_delete_note(note_id)
            print(f"✓ Soft deleted note")
            
            # Try to edit text - should fail
            with pytest.raises(ValueError, match="Cannot edit deleted note"):
                drawer_note_service.edit_note_text(note_id, "Should not be allowed")
            print(f"✓ Correctly rejected edit on deleted note")
            
            print("\n✅ PASS - Edit text rejected on deleted note")
            
        finally:
            cleanup_test_data(note_ids=[note_id], label_ids=[label_id])
    
    def test_8_label_validation_partial_invalid_set_fails(self):
        """
        Test 8: Label validation - partial invalid set fails.
        
        Verifies:
        - Mix of valid + invalid label IDs fails
        - Raises ValueError
        """
        print("\n" + "="*80)
        print("TEST 8: LABEL VALIDATION - PARTIAL INVALID SET FAILS")
        print("="*80)
        
        label_id_valid = create_test_label()
        label_id_invalid = 999999  # Non-existent
        print(f"✓ Created valid label: {label_id_valid}")
        print(f"✓ Using invalid label ID: {label_id_invalid}")
        
        try:
            # Try to create note with mix of valid and invalid labels - should fail
            with pytest.raises(ValueError, match="Invalid or inactive label IDs"):
                drawer_note_service.create_note_with_labels(
                    note_text="Should not be created",
                    label_ids=[label_id_valid, label_id_invalid],
                    created_by_user_id=1,
                    created_by_name="test_user"
                )
            print(f"✓ Correctly rejected mixed valid/invalid label set")
            
            print("\n✅ PASS - Partial invalid label set rejected")
            
        finally:
            cleanup_test_data(label_ids=[label_id_valid])
    
    def test_9_filter_by_labels_all_labels_semantics(self):
        """
        Test 9: Filter by labels - ALL labels semantics (AND logic).
        
        Verifies:
        - Note A with labels {1,2}
        - Note B with labels {1}
        - Filter by {1,2} returns only A
        """
        print("\n" + "="*80)
        print("TEST 9: FILTER BY LABELS - ALL LABELS SEMANTICS (AND)")
        print("="*80)
        
        # Create labels
        label_id_1 = create_test_label("FilterLabel1")
        label_id_2 = create_test_label("FilterLabel2")
        print(f"✓ Created labels: {label_id_1}, {label_id_2}")
        
        # Create note A with labels {1, 2}
        note_id_a = create_test_note(
            note_text="Note A has both labels",
            label_ids=[label_id_1, label_id_2]
        )
        print(f"✓ Created note A with labels {{1,2}}: {note_id_a}")
        
        # Create note B with labels {1}
        note_id_b = create_test_note(
            note_text="Note B has only label 1",
            label_ids=[label_id_1]
        )
        print(f"✓ Created note B with labels {{1}}: {note_id_b}")
        
        try:
            # Filter by {1, 2} - should return only A
            results = drawer_note_service.list_notes(
                label_ids=[label_id_1, label_id_2],
                limit=100
            )
            result_note_ids = [n['note_id'] for n in results]
            
            assert note_id_a in result_note_ids, "Note A should be in results (has both labels)"
            assert note_id_b not in result_note_ids, "Note B should NOT be in results (missing label 2)"
            print(f"✓ Filter by {{1,2}} correctly returned only note A")
            
            print("\n✅ PASS - ALL labels (AND) filtering works correctly")
            
        finally:
            cleanup_test_data(
                note_ids=[note_id_a, note_id_b],
                label_ids=[label_id_1, label_id_2]
            )
    
    def test_10_pagination_works(self):
        """
        Test 10: Pagination works correctly.
        
        Verifies:
        - Create multiple notes
        - list with limit/offset returns correct counts
        """
        print("\n" + "="*80)
        print("TEST 10: PAGINATION WORKS")
        print("="*80)
        
        label_id = create_test_label()
        print(f"✓ Created label: {label_id}")
        
        # Create 5 notes
        note_ids = []
        for i in range(5):
            note_id = create_test_note(
                note_text=f"Pagination test note {i+1}",
                label_ids=[label_id]
            )
            note_ids.append(note_id)
        print(f"✓ Created 5 notes: {note_ids}")
        
        try:
            # List with limit=2, offset=0 - should get 2 notes
            page_1 = drawer_note_service.list_notes(limit=2, offset=0)
            assert len(page_1) == 2, "First page should have 2 notes"
            print(f"✓ Page 1 (limit=2, offset=0): {len(page_1)} notes")
            
            # List with limit=2, offset=2 - should get 2 notes
            page_2 = drawer_note_service.list_notes(limit=2, offset=2)
            assert len(page_2) == 2, "Second page should have 2 notes"
            print(f"✓ Page 2 (limit=2, offset=2): {len(page_2)} notes")
            
            # List with limit=2, offset=4 - should get 1 note (or more if other notes exist)
            page_3 = drawer_note_service.list_notes(limit=2, offset=4)
            assert len(page_3) >= 1, "Third page should have at least 1 note"
            print(f"✓ Page 3 (limit=2, offset=4): {len(page_3)} notes")
            
            # Verify no overlap between pages
            page_1_ids = {n['note_id'] for n in page_1}
            page_2_ids = {n['note_id'] for n in page_2}
            overlap = page_1_ids & page_2_ids
            assert len(overlap) == 0, "Pages should not have overlapping notes"
            print(f"✓ Verified no overlap between pages")
            
            print("\n✅ PASS - Pagination works correctly")
            
        finally:
            cleanup_test_data(note_ids=note_ids, label_ids=[label_id])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
