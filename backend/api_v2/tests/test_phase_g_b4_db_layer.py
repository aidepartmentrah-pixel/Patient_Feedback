"""
Test Suite: Phase G-B4 - Drawer Notes + Labels DB Layer
Tests all database access layer functions for drawer notes and labels.

Verifies:
- Note CRUD operations
- Label CRUD operations
- Label linking operations
- Multi-label filtering with AND logic
- Data integrity

Target: 
- backend/api_v2/db_layer/drawer_note_db.py
- backend/api_v2/db_layer/drawer_label_db.py

Test Coverage:
- All note DB functions
- All label DB functions
- Complex filtering scenarios
- Error conditions

Note: Uses real database connection (no mocks)
"""

import pytest
import sys
from pathlib import Path
import uuid

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

from api_v2.db_layer import drawer_note_db
from api_v2.db_layer import drawer_label_db


class TestDrawerNoteDBLayer:
    """Test suite for drawer note database layer functions."""
    
    def test_1_insert_note(self):
        """
        Test 1: Verify insert_note returns ID and row exists.
        """
        print("\n" + "="*80)
        print("TEST 1: INSERT NOTE")
        print("="*80)
        
        test_text = f"Test note {uuid.uuid4().hex[:8]}"
        test_user_id = 1
        test_user_name = "test_user"
        
        try:
            # Insert note
            note_id = drawer_note_db.insert_note(
                note_text=test_text,
                created_by_user_id=test_user_id,
                created_by_name=test_user_name
            )
            
            print(f"✓ Created note with ID: {note_id}")
            assert note_id is not None, "Should return note ID"
            assert note_id > 0, "Note ID should be positive"
            
            # Verify note exists
            note = drawer_note_db.get_note_by_id(note_id)
            assert note is not None, "Note should exist in database"
            assert note['note_text'] == test_text
            assert note['created_by_user_id'] == test_user_id
            assert note['created_by_name'] == test_user_name
            assert note['is_deleted'] == False
            
            print(f"✓ Note verified in database")
            print("\n✅ PASS - insert_note works correctly")
            
        finally:
            # Clean up
            if 'note_id' in locals():
                from api_v2.db_layer.drawer_note_db import get_db_connection
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("DELETE FROM dbo.APP_DrawerNote WHERE NoteID = ?", (note_id,))
                conn.commit()
                cursor.close()
                conn.close()
                print("Cleaned up test data")
    
    def test_2_update_note_text(self):
        """
        Test 2: Verify update_note_text changes note content.
        """
        print("\n" + "="*80)
        print("TEST 2: UPDATE NOTE TEXT")
        print("="*80)
        
        # Create test note
        original_text = f"Original text {uuid.uuid4().hex[:8]}"
        note_id = drawer_note_db.insert_note(original_text, 1, "test_user")
        
        try:
            print(f"✓ Created note {note_id}: {original_text[:30]}")
            
            # Update text
            new_text = f"Updated text {uuid.uuid4().hex[:8]}"
            drawer_note_db.update_note_text(note_id, new_text)
            print(f"✓ Updated note text")
            
            # Verify update
            note = drawer_note_db.get_note_by_id(note_id)
            assert note['note_text'] == new_text, "Text should be updated"
            assert note['note_text'] != original_text, "Text should differ from original"
            
            print(f"✓ Verified text changed: {new_text[:30]}")
            print("\n✅ PASS - update_note_text works correctly")
            
        finally:
            # Clean up
            from api_v2.db_layer.drawer_note_db import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM dbo.APP_DrawerNote WHERE NoteID = ?", (note_id,))
            conn.commit()
            cursor.close()
            conn.close()
            print("Cleaned up test data")
    
    def test_3_soft_delete_note(self):
        """
        Test 3: Verify soft_delete_note sets IsDeleted = 1.
        """
        print("\n" + "="*80)
        print("TEST 3: SOFT DELETE NOTE")
        print("="*80)
        
        # Create test note
        note_id = drawer_note_db.insert_note(
            f"Test note {uuid.uuid4().hex[:8]}", 1, "test_user"
        )
        
        try:
            # Verify initially not deleted
            note = drawer_note_db.get_note_by_id(note_id)
            assert note['is_deleted'] == False, "Note should not be deleted initially"
            print(f"✓ Note {note_id} initially not deleted")
            
            # Soft delete
            drawer_note_db.soft_delete_note(note_id)
            print(f"✓ Soft deleted note {note_id}")
            
            # Verify deleted
            note = drawer_note_db.get_note_by_id(note_id)
            assert note['is_deleted'] == True, "Note should be marked as deleted"
            
            print(f"✓ Verified IsDeleted = 1")
            print("\n✅ PASS - soft_delete_note works correctly")
            
        finally:
            # Clean up (hard delete)
            from api_v2.db_layer.drawer_note_db import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM dbo.APP_DrawerNote WHERE NoteID = ?", (note_id,))
            conn.commit()
            cursor.close()
            conn.close()
            print("Cleaned up test data")
    
    def test_4_list_notes_paged(self):
        """
        Test 4: Verify list_notes_paged excludes deleted notes.
        """
        print("\n" + "="*80)
        print("TEST 4: LIST NOTES PAGED")
        print("="*80)
        
        # Create test notes
        note_id_1 = drawer_note_db.insert_note(f"Active note 1 {uuid.uuid4().hex[:8]}", 1, "test")
        note_id_2 = drawer_note_db.insert_note(f"Active note 2 {uuid.uuid4().hex[:8]}", 1, "test")
        note_id_3 = drawer_note_db.insert_note(f"Deleted note {uuid.uuid4().hex[:8]}", 1, "test")
        
        try:
            print(f"✓ Created 3 test notes: {note_id_1}, {note_id_2}, {note_id_3}")
            
            # Soft delete one note
            drawer_note_db.soft_delete_note(note_id_3)
            print(f"✓ Soft deleted note {note_id_3}")
            
            # List notes (should exclude deleted)
            notes = drawer_note_db.list_notes_paged(limit=100, offset=0)
            note_ids = [n['note_id'] for n in notes]
            
            print(f"✓ Listed notes (found {len(notes)} total)")
            
            # Verify active notes included
            assert note_id_1 in note_ids, "Active note 1 should be in list"
            assert note_id_2 in note_ids, "Active note 2 should be in list"
            assert note_id_3 not in note_ids, "Deleted note should NOT be in list"
            
            print(f"✓ Verified deleted note excluded from list")
            print("\n✅ PASS - list_notes_paged excludes deleted notes")
            
        finally:
            # Clean up
            from api_v2.db_layer.drawer_note_db import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM dbo.APP_DrawerNote WHERE NoteID IN (?, ?, ?)", 
                          (note_id_1, note_id_2, note_id_3))
            conn.commit()
            cursor.close()
            conn.close()
            print("Cleaned up test data")
    
    def test_5_attach_labels_to_note(self):
        """
        Test 5: Verify attach_labels_to_note creates link rows.
        """
        print("\n" + "="*80)
        print("TEST 5: ATTACH LABELS TO NOTE")
        print("="*80)
        
        # Create test note and labels
        note_id = drawer_note_db.insert_note(f"Note {uuid.uuid4().hex[:8]}", 1, "test")
        label_id_1 = drawer_label_db.insert_label(f"label_{uuid.uuid4().hex[:8]}")
        label_id_2 = drawer_label_db.insert_label(f"label_{uuid.uuid4().hex[:8]}")
        
        try:
            print(f"✓ Created note {note_id} and labels {label_id_1}, {label_id_2}")
            
            # Attach labels
            drawer_note_db.attach_labels_to_note(note_id, [label_id_1, label_id_2])
            print(f"✓ Attached labels to note")
            
            # Verify links exist
            label_ids = drawer_note_db.get_note_label_ids(note_id)
            assert len(label_ids) == 2, "Should have 2 labels"
            assert label_id_1 in label_ids, "Label 1 should be attached"
            assert label_id_2 in label_ids, "Label 2 should be attached"
            
            print(f"✓ Verified both labels attached")
            print("\n✅ PASS - attach_labels_to_note works correctly")
            
        finally:
            # Clean up
            from api_v2.db_layer.drawer_note_db import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM dbo.APP_DrawerNoteLabelLink WHERE NoteID = ?", (note_id,))
            cursor.execute("DELETE FROM dbo.APP_DrawerLabel WHERE LabelID IN (?, ?)", 
                          (label_id_1, label_id_2))
            cursor.execute("DELETE FROM dbo.APP_DrawerNote WHERE NoteID = ?", (note_id,))
            conn.commit()
            cursor.close()
            conn.close()
            print("Cleaned up test data")
    
    def test_6_replace_note_labels(self):
        """
        Test 6: Verify replace_note_labels removes old and adds new labels.
        """
        print("\n" + "="*80)
        print("TEST 6: REPLACE NOTE LABELS")
        print("="*80)
        
        # Create test note and labels
        note_id = drawer_note_db.insert_note(f"Note {uuid.uuid4().hex[:8]}", 1, "test")
        label_id_1 = drawer_label_db.insert_label(f"label_{uuid.uuid4().hex[:8]}")
        label_id_2 = drawer_label_db.insert_label(f"label_{uuid.uuid4().hex[:8]}")
        label_id_3 = drawer_label_db.insert_label(f"label_{uuid.uuid4().hex[:8]}")
        
        try:
            print(f"✓ Created note {note_id} and 3 labels")
            
            # Attach initial labels
            drawer_note_db.attach_labels_to_note(note_id, [label_id_1, label_id_2])
            print(f"✓ Attached initial labels: {label_id_1}, {label_id_2}")
            
            # Replace with new set of labels
            drawer_note_db.replace_note_labels(note_id, [label_id_2, label_id_3])
            print(f"✓ Replaced with labels: {label_id_2}, {label_id_3}")
            
            # Verify new label set
            label_ids = drawer_note_db.get_note_label_ids(note_id)
            assert len(label_ids) == 2, "Should have 2 labels"
            assert label_id_1 not in label_ids, "Old label 1 should be removed"
            assert label_id_2 in label_ids, "Label 2 should still be present"
            assert label_id_3 in label_ids, "New label 3 should be added"
            
            print(f"✓ Verified old labels removed and new ones present")
            print("\n✅ PASS - replace_note_labels works correctly")
            
        finally:
            # Clean up
            from api_v2.db_layer.drawer_note_db import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM dbo.APP_DrawerNoteLabelLink WHERE NoteID = ?", (note_id,))
            cursor.execute("DELETE FROM dbo.APP_DrawerLabel WHERE LabelID IN (?, ?, ?)", 
                          (label_id_1, label_id_2, label_id_3))
            cursor.execute("DELETE FROM dbo.APP_DrawerNote WHERE NoteID = ?", (note_id,))
            conn.commit()
            cursor.close()
            conn.close()
            print("Cleaned up test data")
    
    def test_7_get_note_label_ids(self):
        """
        Test 7: Verify get_note_label_ids returns correct IDs.
        """
        print("\n" + "="*80)
        print("TEST 7: GET NOTE LABEL IDS")
        print("="*80)
        
        # Create test note and labels
        note_id = drawer_note_db.insert_note(f"Note {uuid.uuid4().hex[:8]}", 1, "test")
        label_id_1 = drawer_label_db.insert_label(f"label_{uuid.uuid4().hex[:8]}")
        label_id_2 = drawer_label_db.insert_label(f"label_{uuid.uuid4().hex[:8]}")
        
        try:
            # Attach labels
            drawer_note_db.attach_labels_to_note(note_id, [label_id_1, label_id_2])
            print(f"✓ Attached labels {label_id_1}, {label_id_2}")
            
            # Get label IDs
            label_ids = drawer_note_db.get_note_label_ids(note_id)
            
            print(f"✓ Retrieved label IDs: {label_ids}")
            assert len(label_ids) == 2, "Should return 2 label IDs"
            assert label_id_1 in label_ids, "Should include label 1"
            assert label_id_2 in label_ids, "Should include label 2"
            
            print("\n✅ PASS - get_note_label_ids returns correct IDs")
            
        finally:
            # Clean up
            from api_v2.db_layer.drawer_note_db import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM dbo.APP_DrawerNoteLabelLink WHERE NoteID = ?", (note_id,))
            cursor.execute("DELETE FROM dbo.APP_DrawerLabel WHERE LabelID IN (?, ?)", 
                          (label_id_1, label_id_2))
            cursor.execute("DELETE FROM dbo.APP_DrawerNote WHERE NoteID = ?", (note_id,))
            conn.commit()
            cursor.close()
            conn.close()
            print("Cleaned up test data")


class TestDrawerLabelDBLayer:
    """Test suite for drawer label database layer functions."""
    
    def test_8_insert_label(self):
        """
        Test 8: Verify insert_label returns ID.
        """
        print("\n" + "="*80)
        print("TEST 8: INSERT LABEL")
        print("="*80)
        
        test_label_name = f"test_label_{uuid.uuid4().hex[:8]}"
        
        try:
            # Insert label
            label_id = drawer_label_db.insert_label(test_label_name)
            
            print(f"✓ Created label '{test_label_name}' with ID: {label_id}")
            assert label_id is not None, "Should return label ID"
            assert label_id > 0, "Label ID should be positive"
            
            # Verify label exists in active list
            labels = drawer_label_db.list_active_labels()
            label_names = [l['label_name'] for l in labels]
            assert test_label_name in label_names, "Label should be in active list"
            
            print(f"✓ Verified label in active list")
            print("\n✅ PASS - insert_label works correctly")
            
        finally:
            # Clean up
            if 'label_id' in locals():
                from api_v2.db_layer.drawer_label_db import get_db_connection
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("DELETE FROM dbo.APP_DrawerLabel WHERE LabelID = ?", (label_id,))
                conn.commit()
                cursor.close()
                conn.close()
                print("Cleaned up test data")
    
    def test_9_list_active_labels(self):
        """
        Test 9: Verify list_active_labels includes new label.
        """
        print("\n" + "="*80)
        print("TEST 9: LIST ACTIVE LABELS")
        print("="*80)
        
        # Create test labels
        label_id_1 = drawer_label_db.insert_label(f"active_{uuid.uuid4().hex[:8]}")
        label_id_2 = drawer_label_db.insert_label(f"active_{uuid.uuid4().hex[:8]}")
        
        try:
            print(f"✓ Created labels {label_id_1}, {label_id_2}")
            
            # List active labels
            labels = drawer_label_db.list_active_labels()
            label_ids = [l['label_id'] for l in labels]
            
            print(f"✓ Listed {len(labels)} active labels")
            assert label_id_1 in label_ids, "Label 1 should be in active list"
            assert label_id_2 in label_ids, "Label 2 should be in active list"
            
            # Verify all are active
            for label in labels:
                assert label['is_active'] == True, "All listed labels should be active"
            
            print(f"✓ Verified both labels in active list")
            print("\n✅ PASS - list_active_labels works correctly")
            
        finally:
            # Clean up
            from api_v2.db_layer.drawer_label_db import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM dbo.APP_DrawerLabel WHERE LabelID IN (?, ?)", 
                          (label_id_1, label_id_2))
            conn.commit()
            cursor.close()
            conn.close()
            print("Cleaned up test data")
    
    def test_10_disable_label(self):
        """
        Test 10: Verify disable_label removes from active list.
        """
        print("\n" + "="*80)
        print("TEST 10: DISABLE LABEL")
        print("="*80)
        
        # Create test label
        label_id = drawer_label_db.insert_label(f"to_disable_{uuid.uuid4().hex[:8]}")
        
        try:
            print(f"✓ Created label {label_id}")
            
            # Verify initially in active list
            labels = drawer_label_db.list_active_labels()
            label_ids = [l['label_id'] for l in labels]
            assert label_id in label_ids, "Label should initially be active"
            print(f"✓ Label initially in active list")
            
            # Disable label
            drawer_label_db.disable_label(label_id)
            print(f"✓ Disabled label {label_id}")
            
            # Verify removed from active list
            labels = drawer_label_db.list_active_labels()
            label_ids = [l['label_id'] for l in labels]
            assert label_id not in label_ids, "Disabled label should not be in active list"
            
            print(f"✓ Verified label removed from active list")
            print("\n✅ PASS - disable_label works correctly")
            
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
    
    def test_11_get_label_ids_exist(self):
        """
        Test 11: Verify get_label_ids_exist returns only valid active IDs.
        """
        print("\n" + "="*80)
        print("TEST 11: GET LABEL IDS EXIST")
        print("="*80)
        
        # Create test labels
        label_id_1 = drawer_label_db.insert_label(f"valid_{uuid.uuid4().hex[:8]}")
        label_id_2 = drawer_label_db.insert_label(f"valid_{uuid.uuid4().hex[:8]}")
        label_id_3 = drawer_label_db.insert_label(f"disabled_{uuid.uuid4().hex[:8]}")
        
        try:
            print(f"✓ Created 3 labels")
            
            # Disable one label
            drawer_label_db.disable_label(label_id_3)
            print(f"✓ Disabled label {label_id_3}")
            
            # Check which IDs exist (include invalid ID and disabled ID)
            invalid_id = 999999
            ids_to_check = [label_id_1, label_id_2, label_id_3, invalid_id]
            
            valid_ids = drawer_label_db.get_label_ids_exist(ids_to_check)
            print(f"✓ Checked IDs: {ids_to_check}")
            print(f"✓ Valid active IDs: {valid_ids}")
            
            # Verify only active IDs returned
            assert label_id_1 in valid_ids, "Active label 1 should be valid"
            assert label_id_2 in valid_ids, "Active label 2 should be valid"
            assert label_id_3 not in valid_ids, "Disabled label should NOT be valid"
            assert invalid_id not in valid_ids, "Invalid ID should NOT be valid"
            assert len(valid_ids) == 2, "Should have exactly 2 valid IDs"
            
            print(f"✓ Verified only active labels returned")
            print("\n✅ PASS - get_label_ids_exist works correctly")
            
        finally:
            # Clean up
            from api_v2.db_layer.drawer_label_db import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM dbo.APP_DrawerLabel WHERE LabelID IN (?, ?, ?)", 
                          (label_id_1, label_id_2, label_id_3))
            conn.commit()
            cursor.close()
            conn.close()
            print("Cleaned up test data")


class TestDrawerNoteFiltering:
    """Test suite for complex filtering scenarios."""
    
    def test_12_filter_notes_by_label_ids_and_logic(self):
        """
        Test 12: Verify filter_notes_by_label_ids uses AND logic.
        Create note with labels A+B, note with label A only.
        Filter by A+B should return only first note.
        """
        print("\n" + "="*80)
        print("TEST 12: FILTER NOTES BY LABELS (AND LOGIC)")
        print("="*80)
        
        # Create test labels
        label_id_a = drawer_label_db.insert_label(f"label_A_{uuid.uuid4().hex[:8]}")
        label_id_b = drawer_label_db.insert_label(f"label_B_{uuid.uuid4().hex[:8]}")
        
        # Create note with both labels
        note_id_both = drawer_note_db.insert_note(
            f"Note with A+B {uuid.uuid4().hex[:8]}", 1, "test"
        )
        drawer_note_db.attach_labels_to_note(note_id_both, [label_id_a, label_id_b])
        
        # Create note with only label A
        note_id_a_only = drawer_note_db.insert_note(
            f"Note with A only {uuid.uuid4().hex[:8]}", 1, "test"
        )
        drawer_note_db.attach_labels_to_note(note_id_a_only, [label_id_a])
        
        try:
            print(f"✓ Created note {note_id_both} with labels A+B")
            print(f"✓ Created note {note_id_a_only} with label A only")
            
            # Filter by both labels (AND logic)
            filtered_notes = drawer_note_db.filter_notes_by_label_ids(
                [label_id_a, label_id_b], limit=100, offset=0
            )
            filtered_note_ids = [n['note_id'] for n in filtered_notes]
            
            print(f"✓ Filtered by labels A+B")
            print(f"  Found {len(filtered_notes)} notes: {filtered_note_ids}")
            
            # Verify only note with both labels returned
            assert note_id_both in filtered_note_ids, "Note with both labels should be returned"
            assert note_id_a_only not in filtered_note_ids, "Note with only A should NOT be returned"
            assert len(filtered_notes) >= 1, "Should return at least the test note"
            
            print(f"✓ Verified AND logic: only notes with ALL labels returned")
            print("\n✅ PASS - filter_notes_by_label_ids uses AND logic correctly")
            
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
    print("PHASE G-B4: DRAWER NOTES + LABELS DB LAYER TESTS")
    print("="*80)
    
    note_tests = TestDrawerNoteDBLayer()
    label_tests = TestDrawerLabelDBLayer()
    filter_tests = TestDrawerNoteFiltering()
    
    tests = [
        # Note DB tests
        ("Insert Note", note_tests.test_1_insert_note),
        ("Update Note Text", note_tests.test_2_update_note_text),
        ("Soft Delete Note", note_tests.test_3_soft_delete_note),
        ("List Notes Paged", note_tests.test_4_list_notes_paged),
        ("Attach Labels to Note", note_tests.test_5_attach_labels_to_note),
        ("Replace Note Labels", note_tests.test_6_replace_note_labels),
        ("Get Note Label IDs", note_tests.test_7_get_note_label_ids),
        
        # Label DB tests
        ("Insert Label", label_tests.test_8_insert_label),
        ("List Active Labels", label_tests.test_9_list_active_labels),
        ("Disable Label", label_tests.test_10_disable_label),
        ("Get Label IDs Exist", label_tests.test_11_get_label_ids_exist),
        
        # Filtering tests
        ("Filter Notes AND Logic", filter_tests.test_12_filter_notes_by_label_ids_and_logic),
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
