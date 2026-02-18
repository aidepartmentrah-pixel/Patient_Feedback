"""
Test Suite: Phase G-B5 - Drawer Note Service Layer
Tests all business logic functions for drawer notes service.

Verifies:
- Note creation with validation
- Note editing with validation
- Label management validation
- Soft delete operations
- Listing and filtering

Target: 
- backend/api_v2/services/drawer_note_service.py

Test Coverage:
- All service functions
- Success scenarios
- Error conditions
- Business rule enforcement

Note: Uses real database connection (no mocks)
"""

import pytest
import sys
from pathlib import Path
import uuid

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

from api_v2.services import drawer_note_service
from api_v2.db_layer import drawer_note_db
from api_v2.db_layer import drawer_label_db


class TestDrawerNoteService:
    """Test suite for drawer note service layer functions."""
    
    def test_1_create_note_with_labels_success(self):
        """
        Test 1: Verify create_note_with_labels creates note with labels.
        """
        print("\n" + "="*80)
        print("TEST 1: CREATE NOTE WITH LABELS - SUCCESS")
        print("="*80)
        
        # Create test labels
        label_id_1 = drawer_label_db.insert_label(f"label_{uuid.uuid4().hex[:8]}")
        label_id_2 = drawer_label_db.insert_label(f"label_{uuid.uuid4().hex[:8]}")
        
        try:
            print(f"✓ Created test labels: {label_id_1}, {label_id_2}")
            
            # Create note with labels
            test_text = f"Test note {uuid.uuid4().hex[:8]}"
            note_id = drawer_note_service.create_note_with_labels(
                note_text=test_text,
                label_ids=[label_id_1, label_id_2],
                created_by_user_id=1,
                created_by_name="test_user"
            )
            
            print(f"✓ Created note with ID: {note_id}")
            assert note_id is not None, "Should return note ID"
            assert note_id > 0, "Note ID should be positive"
            
            # Verify note exists in DB
            note = drawer_note_db.get_note_by_id(note_id)
            assert note is not None, "Note should exist in database"
            assert note['note_text'] == test_text.strip()
            
            # Verify labels attached
            label_ids = drawer_note_db.get_note_label_ids(note_id)
            assert len(label_ids) == 2, "Should have 2 labels"
            assert label_id_1 in label_ids, "Label 1 should be attached"
            assert label_id_2 in label_ids, "Label 2 should be attached"
            
            print(f"✓ Verified note and labels in database")
            print("\n✅ PASS - create_note_with_labels success")
            
        finally:
            # Clean up
            if 'note_id' in locals():
                from api_v2.db_layer.drawer_note_db import get_db_connection
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("DELETE FROM dbo.APP_DrawerNoteLabelLink WHERE NoteID = ?", (note_id,))
                cursor.execute("DELETE FROM dbo.APP_DrawerNote WHERE NoteID = ?", (note_id,))
                cursor.execute("DELETE FROM dbo.APP_DrawerLabel WHERE LabelID IN (?, ?)", 
                              (label_id_1, label_id_2))
                conn.commit()
                cursor.close()
                conn.close()
                print("Cleaned up test data")
    
    def test_2_create_note_rejects_empty_text(self):
        """
        Test 2: Verify create_note_with_labels rejects empty text.
        """
        print("\n" + "="*80)
        print("TEST 2: CREATE NOTE - REJECT EMPTY TEXT")
        print("="*80)
        
        # Create test label
        label_id = drawer_label_db.insert_label(f"label_{uuid.uuid4().hex[:8]}")
        
        try:
            print(f"✓ Created test label: {label_id}")
            
            # Try to create note with empty text
            with pytest.raises(ValueError, match="Note text cannot be empty"):
                drawer_note_service.create_note_with_labels(
                    note_text="   ",  # Only whitespace
                    label_ids=[label_id],
                    created_by_user_id=1,
                    created_by_name="test_user"
                )
            
            print(f"✓ Correctly rejected empty text")
            print("\n✅ PASS - empty text validation works")
            
        finally:
            # Clean up
            from api_v2.db_layer.drawer_label_db import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM dbo.APP_DrawerLabel WHERE LabelID = ?", (label_id,))
            conn.commit()
            cursor.close()
            conn.close()
            print("Cleaned up test data")
    
    def test_3_create_note_rejects_empty_label_list(self):
        """
        Test 3: Verify create_note_with_labels rejects empty label list.
        """
        print("\n" + "="*80)
        print("TEST 3: CREATE NOTE - REJECT EMPTY LABEL LIST")
        print("="*80)
        
        # Try to create note without labels
        with pytest.raises(ValueError, match="at least one label"):
            drawer_note_service.create_note_with_labels(
                note_text="Test note",
                label_ids=[],  # Empty list
                created_by_user_id=1,
                created_by_name="test_user"
            )
        
        print(f"✓ Correctly rejected empty label list")
        print("\n✅ PASS - label list validation works")
    
    def test_4_create_note_rejects_inactive_label(self):
        """
        Test 4: Verify create_note_with_labels rejects inactive labels.
        """
        print("\n" + "="*80)
        print("TEST 4: CREATE NOTE - REJECT INACTIVE LABEL")
        print("="*80)
        
        # Create and disable label
        label_id = drawer_label_db.insert_label(f"label_{uuid.uuid4().hex[:8]}")
        drawer_label_db.disable_label(label_id)
        
        try:
            print(f"✓ Created and disabled label: {label_id}")
            
            # Try to create note with disabled label
            with pytest.raises(ValueError, match="Invalid or inactive label IDs"):
                drawer_note_service.create_note_with_labels(
                    note_text="Test note",
                    label_ids=[label_id],
                    created_by_user_id=1,
                    created_by_name="test_user"
                )
            
            print(f"✓ Correctly rejected inactive label")
            print("\n✅ PASS - inactive label validation works")
            
        finally:
            # Clean up
            from api_v2.db_layer.drawer_label_db import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM dbo.APP_DrawerLabel WHERE LabelID = ?", (label_id,))
            conn.commit()
            cursor.close()
            conn.close()
            print("Cleaned up test data")
    
    def test_5_edit_note_text_success(self):
        """
        Test 5: Verify edit_note_text updates note content.
        """
        print("\n" + "="*80)
        print("TEST 5: EDIT NOTE TEXT - SUCCESS")
        print("="*80)
        
        # Create test label and note
        label_id = drawer_label_db.insert_label(f"label_{uuid.uuid4().hex[:8]}")
        original_text = f"Original text {uuid.uuid4().hex[:8]}"
        note_id = drawer_note_service.create_note_with_labels(
            note_text=original_text,
            label_ids=[label_id],
            created_by_user_id=1,
            created_by_name="test_user"
        )
        
        try:
            print(f"✓ Created note {note_id} with original text")
            
            # Edit text
            new_text = f"Updated text {uuid.uuid4().hex[:8]}"
            drawer_note_service.edit_note_text(note_id, new_text)
            print(f"✓ Edited note text")
            
            # Verify update
            note = drawer_note_db.get_note_by_id(note_id)
            assert note['note_text'] == new_text.strip(), "Text should be updated"
            assert note['note_text'] != original_text, "Text should differ from original"
            
            print(f"✓ Verified text updated in database")
            print("\n✅ PASS - edit_note_text success")
            
        finally:
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
    
    def test_6_edit_note_text_rejects_empty_text(self):
        """
        Test 6: Verify edit_note_text rejects empty text.
        """
        print("\n" + "="*80)
        print("TEST 6: EDIT NOTE TEXT - REJECT EMPTY TEXT")
        print("="*80)
        
        # Create test label and note
        label_id = drawer_label_db.insert_label(f"label_{uuid.uuid4().hex[:8]}")
        note_id = drawer_note_service.create_note_with_labels(
            note_text="Original text",
            label_ids=[label_id],
            created_by_user_id=1,
            created_by_name="test_user"
        )
        
        try:
            print(f"✓ Created note {note_id}")
            
            # Try to edit with empty text
            with pytest.raises(ValueError, match="Note text cannot be empty"):
                drawer_note_service.edit_note_text(note_id, "   ")
            
            print(f"✓ Correctly rejected empty text")
            print("\n✅ PASS - empty text validation works")
            
        finally:
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
    
    def test_7_edit_note_text_rejects_deleted_note(self):
        """
        Test 7: Verify edit_note_text rejects deleted notes.
        """
        print("\n" + "="*80)
        print("TEST 7: EDIT NOTE TEXT - REJECT DELETED NOTE")
        print("="*80)
        
        # Create test label and note
        label_id = drawer_label_db.insert_label(f"label_{uuid.uuid4().hex[:8]}")
        note_id = drawer_note_service.create_note_with_labels(
            note_text="Original text",
            label_ids=[label_id],
            created_by_user_id=1,
            created_by_name="test_user"
        )
        
        try:
            print(f"✓ Created note {note_id}")
            
            # Soft delete note
            drawer_note_service.soft_delete_note(note_id)
            print(f"✓ Soft deleted note {note_id}")
            
            # Try to edit deleted note
            with pytest.raises(ValueError, match="Cannot edit deleted note"):
                drawer_note_service.edit_note_text(note_id, "New text")
            
            print(f"✓ Correctly rejected editing deleted note")
            print("\n✅ PASS - deleted note validation works")
            
        finally:
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
    
    def test_8_edit_note_labels_success(self):
        """
        Test 8: Verify edit_note_labels replaces labels.
        """
        print("\n" + "="*80)
        print("TEST 8: EDIT NOTE LABELS - SUCCESS")
        print("="*80)
        
        # Create test labels
        label_id_1 = drawer_label_db.insert_label(f"label_{uuid.uuid4().hex[:8]}")
        label_id_2 = drawer_label_db.insert_label(f"label_{uuid.uuid4().hex[:8]}")
        label_id_3 = drawer_label_db.insert_label(f"label_{uuid.uuid4().hex[:8]}")
        
        # Create note with labels 1 and 2
        note_id = drawer_note_service.create_note_with_labels(
            note_text="Test note",
            label_ids=[label_id_1, label_id_2],
            created_by_user_id=1,
            created_by_name="test_user"
        )
        
        try:
            print(f"✓ Created note {note_id} with labels {label_id_1}, {label_id_2}")
            
            # Replace with labels 2 and 3
            drawer_note_service.edit_note_labels(note_id, [label_id_2, label_id_3])
            print(f"✓ Replaced labels with {label_id_2}, {label_id_3}")
            
            # Verify new label set
            label_ids = drawer_note_db.get_note_label_ids(note_id)
            assert len(label_ids) == 2, "Should have 2 labels"
            assert label_id_1 not in label_ids, "Label 1 should be removed"
            assert label_id_2 in label_ids, "Label 2 should still be present"
            assert label_id_3 in label_ids, "Label 3 should be added"
            
            print(f"✓ Verified labels replaced correctly")
            print("\n✅ PASS - edit_note_labels success")
            
        finally:
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
    
    def test_9_edit_note_labels_rejects_empty_list(self):
        """
        Test 9: Verify edit_note_labels rejects empty label list.
        """
        print("\n" + "="*80)
        print("TEST 9: EDIT NOTE LABELS - REJECT EMPTY LIST")
        print("="*80)
        
        # Create test label and note
        label_id = drawer_label_db.insert_label(f"label_{uuid.uuid4().hex[:8]}")
        note_id = drawer_note_service.create_note_with_labels(
            note_text="Test note",
            label_ids=[label_id],
            created_by_user_id=1,
            created_by_name="test_user"
        )
        
        try:
            print(f"✓ Created note {note_id}")
            
            # Try to set empty label list
            with pytest.raises(ValueError, match="at least one label"):
                drawer_note_service.edit_note_labels(note_id, [])
            
            print(f"✓ Correctly rejected empty label list")
            print("\n✅ PASS - empty label list validation works")
            
        finally:
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
    
    def test_10_soft_delete_note_success(self):
        """
        Test 10: Verify soft_delete_note sets IsDeleted = 1.
        """
        print("\n" + "="*80)
        print("TEST 10: SOFT DELETE NOTE - SUCCESS")
        print("="*80)
        
        # Create test label and note
        label_id = drawer_label_db.insert_label(f"label_{uuid.uuid4().hex[:8]}")
        note_id = drawer_note_service.create_note_with_labels(
            note_text="Test note",
            label_ids=[label_id],
            created_by_user_id=1,
            created_by_name="test_user"
        )
        
        try:
            print(f"✓ Created note {note_id}")
            
            # Soft delete
            drawer_note_service.soft_delete_note(note_id)
            print(f"✓ Soft deleted note {note_id}")
            
            # Verify deleted
            note = drawer_note_db.get_note_by_id(note_id)
            assert note['is_deleted'] == True, "Note should be marked as deleted"
            
            print(f"✓ Verified IsDeleted = 1 in database")
            print("\n✅ PASS - soft_delete_note success")
            
        finally:
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
    
    def test_11_list_notes_returns_non_deleted_only(self):
        """
        Test 11: Verify list_notes excludes deleted notes.
        """
        print("\n" + "="*80)
        print("TEST 11: LIST NOTES - EXCLUDE DELETED")
        print("="*80)
        
        # Create test label
        label_id = drawer_label_db.insert_label(f"label_{uuid.uuid4().hex[:8]}")
        
        # Create active note
        note_id_active = drawer_note_service.create_note_with_labels(
            note_text=f"Active note {uuid.uuid4().hex[:8]}",
            label_ids=[label_id],
            created_by_user_id=1,
            created_by_name="test_user"
        )
        
        # Create and delete note
        note_id_deleted = drawer_note_service.create_note_with_labels(
            note_text=f"Deleted note {uuid.uuid4().hex[:8]}",
            label_ids=[label_id],
            created_by_user_id=1,
            created_by_name="test_user"
        )
        drawer_note_service.soft_delete_note(note_id_deleted)
        
        try:
            print(f"✓ Created active note {note_id_active}")
            print(f"✓ Created and deleted note {note_id_deleted}")
            
            # List notes
            notes = drawer_note_service.list_notes(limit=100, offset=0)
            note_ids = [n['note_id'] for n in notes]
            
            print(f"✓ Listed {len(notes)} notes")
            
            # Verify active included, deleted excluded
            assert note_id_active in note_ids, "Active note should be in list"
            assert note_id_deleted not in note_ids, "Deleted note should NOT be in list"
            
            print(f"✓ Verified deleted note excluded")
            print("\n✅ PASS - list_notes excludes deleted notes")
            
        finally:
            # Clean up
            from api_v2.db_layer.drawer_note_db import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM dbo.APP_DrawerNoteLabelLink WHERE NoteID IN (?, ?)", 
                          (note_id_active, note_id_deleted))
            cursor.execute("DELETE FROM dbo.APP_DrawerNote WHERE NoteID IN (?, ?)", 
                          (note_id_active, note_id_deleted))
            cursor.execute("DELETE FROM dbo.APP_DrawerLabel WHERE LabelID = ?", (label_id,))
            conn.commit()
            cursor.close()
            conn.close()
            print("Cleaned up test data")
    
    def test_12_list_notes_with_label_filter(self):
        """
        Test 12: Verify list_notes with label filter returns correct subset.
        """
        print("\n" + "="*80)
        print("TEST 12: LIST NOTES - LABEL FILTER (AND LOGIC)")
        print("="*80)
        
        # Create test labels
        label_id_a = drawer_label_db.insert_label(f"label_A_{uuid.uuid4().hex[:8]}")
        label_id_b = drawer_label_db.insert_label(f"label_B_{uuid.uuid4().hex[:8]}")
        
        # Create note with both labels
        note_id_both = drawer_note_service.create_note_with_labels(
            note_text=f"Note with A+B {uuid.uuid4().hex[:8]}",
            label_ids=[label_id_a, label_id_b],
            created_by_user_id=1,
            created_by_name="test_user"
        )
        
        # Create note with only label A
        note_id_a_only = drawer_note_service.create_note_with_labels(
            note_text=f"Note with A only {uuid.uuid4().hex[:8]}",
            label_ids=[label_id_a],
            created_by_user_id=1,
            created_by_name="test_user"
        )
        
        try:
            print(f"✓ Created note {note_id_both} with labels A+B")
            print(f"✓ Created note {note_id_a_only} with label A only")
            
            # Filter by both labels (AND logic)
            filtered_notes = drawer_note_service.list_notes(
                label_ids=[label_id_a, label_id_b],
                limit=100,
                offset=0
            )
            filtered_note_ids = [n['note_id'] for n in filtered_notes]
            
            print(f"✓ Filtered by labels A+B")
            print(f"  Found {len(filtered_notes)} notes: {filtered_note_ids}")
            
            # Verify only note with both labels returned
            assert note_id_both in filtered_note_ids, "Note with both labels should be returned"
            assert note_id_a_only not in filtered_note_ids, "Note with only A should NOT be returned"
            
            print(f"✓ Verified AND filtering logic")
            print("\n✅ PASS - list_notes with label filter uses AND logic")
            
        finally:
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


def run_all_tests():
    """Run all tests in sequence."""
    print("\n" + "="*80)
    print("PHASE G-B5: DRAWER NOTE SERVICE LAYER TESTS")
    print("="*80)
    
    service_tests = TestDrawerNoteService()
    
    tests = [
        ("Create Note with Labels SUCCESS", service_tests.test_1_create_note_with_labels_success),
        ("Create Note REJECT Empty Text", service_tests.test_2_create_note_rejects_empty_text),
        ("Create Note REJECT Empty Labels", service_tests.test_3_create_note_rejects_empty_label_list),
        ("Create Note REJECT Inactive Label", service_tests.test_4_create_note_rejects_inactive_label),
        ("Edit Note Text SUCCESS", service_tests.test_5_edit_note_text_success),
        ("Edit Note Text REJECT Empty", service_tests.test_6_edit_note_text_rejects_empty_text),
        ("Edit Note Text REJECT Deleted", service_tests.test_7_edit_note_text_rejects_deleted_note),
        ("Edit Note Labels SUCCESS", service_tests.test_8_edit_note_labels_success),
        ("Edit Note Labels REJECT Empty", service_tests.test_9_edit_note_labels_rejects_empty_list),
        ("Soft Delete Note SUCCESS", service_tests.test_10_soft_delete_note_success),
        ("List Notes EXCLUDE Deleted", service_tests.test_11_list_notes_returns_non_deleted_only),
        ("List Notes LABEL Filter", service_tests.test_12_list_notes_with_label_filter),
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
