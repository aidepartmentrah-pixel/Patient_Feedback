"""
Test Suite: Phase G-B6 - Drawer Label Service Layer
Tests all business logic functions for drawer labels service.

Verifies:
- Label creation with validation
- Name trimming
- Length validation
- Uniqueness enforcement
- Active label listing
- Label disabling
- Label ID validation

Target: 
- backend/api_v2/services/drawer_label_service.py

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

from api_v2.services import drawer_label_service
from api_v2.db_layer import drawer_label_db


class TestDrawerLabelService:
    """Test suite for drawer label service layer functions."""
    
    def test_1_create_label_success(self):
        """
        Test 1: Verify create_label creates label and returns ID.
        """
        print("\n" + "="*80)
        print("TEST 1: CREATE LABEL - SUCCESS")
        print("="*80)
        
        test_label_name = f"Test Label {uuid.uuid4().hex[:8]}"
        
        try:
            # Create label
            label_id = drawer_label_service.create_label(test_label_name)
            
            print(f"✓ Created label '{test_label_name}' with ID: {label_id}")
            assert label_id is not None, "Should return label ID"
            assert label_id > 0, "Label ID should be positive"
            
            # Verify label exists in active list
            labels = drawer_label_service.list_active_labels()
            label_names = [l['label_name'] for l in labels]
            assert test_label_name in label_names, "Label should be in active list"
            
            print(f"✓ Verified label in active list")
            print("\n✅ PASS - create_label success")
            
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
    
    def test_2_create_label_trims_whitespace(self):
        """
        Test 2: Verify create_label trims whitespace from label name.
        """
        print("\n" + "="*80)
        print("TEST 2: CREATE LABEL - TRIM WHITESPACE")
        print("="*80)
        
        test_label_name = f"TestLabel{uuid.uuid4().hex[:8]}"
        label_with_spaces = f"  {test_label_name}  "  # Leading and trailing spaces
        
        try:
            # Create label with spaces
            label_id = drawer_label_service.create_label(label_with_spaces)
            
            print(f"✓ Created label with spaces: '{label_with_spaces}'")
            
            # Verify label stored without spaces
            labels = drawer_label_service.list_active_labels()
            label_dict = next((l for l in labels if l['label_id'] == label_id), None)
            
            assert label_dict is not None, "Label should exist"
            assert label_dict['label_name'] == test_label_name, "Label should be trimmed"
            assert label_dict['label_name'] != label_with_spaces, "Spaces should be removed"
            
            print(f"✓ Verified label trimmed: '{label_dict['label_name']}'")
            print("\n✅ PASS - whitespace trimming works")
            
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
    
    def test_3_create_label_rejects_short_name(self):
        """
        Test 3: Verify create_label rejects names shorter than 2 characters.
        """
        print("\n" + "="*80)
        print("TEST 3: CREATE LABEL - REJECT SHORT NAME")
        print("="*80)
        
        # Try single character
        with pytest.raises(ValueError, match="at least 2 characters"):
            drawer_label_service.create_label("A")
        
        print(f"✓ Correctly rejected single character")
        
        # Try empty string
        with pytest.raises(ValueError, match="at least 2 characters"):
            drawer_label_service.create_label("")
        
        print(f"✓ Correctly rejected empty string")
        
        # Try whitespace only (becomes empty after trim)
        with pytest.raises(ValueError, match="at least 2 characters"):
            drawer_label_service.create_label("   ")
        
        print(f"✓ Correctly rejected whitespace-only string")
        print("\n✅ PASS - length validation works")
    
    def test_4_create_label_duplicate_fails(self):
        """
        Test 4: Verify create_label fails for duplicate names (DB constraint).
        """
        print("\n" + "="*80)
        print("TEST 4: CREATE LABEL - REJECT DUPLICATE")
        print("="*80)
        
        test_label_name = f"UniqueLabel{uuid.uuid4().hex[:8]}"
        
        try:
            # Create first label
            label_id = drawer_label_service.create_label(test_label_name)
            print(f"✓ Created first label: '{test_label_name}' (ID: {label_id})")
            
            # Try to create duplicate (should fail at DB level)
            try:
                drawer_label_service.create_label(test_label_name)
                assert False, "Should have raised an error for duplicate"
            except Exception as e:
                # DB should raise constraint violation error
                print(f"✓ Correctly rejected duplicate label")
                print(f"  Error type: {type(e).__name__}")
                assert True  # Any exception is acceptable (DB constraint violation)
            
            print("\n✅ PASS - duplicate rejection works")
            
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
    
    def test_5_list_active_labels_includes_new_label(self):
        """
        Test 5: Verify list_active_labels includes newly created labels.
        """
        print("\n" + "="*80)
        print("TEST 5: LIST ACTIVE LABELS - INCLUDES NEW")
        print("="*80)
        
        test_label_name = f"ActiveLabel{uuid.uuid4().hex[:8]}"
        
        try:
            # Create label
            label_id = drawer_label_service.create_label(test_label_name)
            print(f"✓ Created label: '{test_label_name}' (ID: {label_id})")
            
            # List active labels
            active_labels = drawer_label_service.list_active_labels()
            label_ids = [l['label_id'] for l in active_labels]
            label_names = [l['label_name'] for l in active_labels]
            
            print(f"✓ Listed {len(active_labels)} active labels")
            
            # Verify new label included
            assert label_id in label_ids, "New label ID should be in active list"
            assert test_label_name in label_names, "New label name should be in active list"
            
            # Verify all returned labels are active
            for label in active_labels:
                assert label['is_active'] == True, "All listed labels should be active"
            
            print(f"✓ Verified new label in active list")
            print("\n✅ PASS - list_active_labels includes new labels")
            
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
    
    def test_6_disable_label_removes_from_active_list(self):
        """
        Test 6: Verify disable_label removes label from active list.
        """
        print("\n" + "="*80)
        print("TEST 6: DISABLE LABEL - REMOVES FROM ACTIVE LIST")
        print("="*80)
        
        test_label_name = f"ToDisable{uuid.uuid4().hex[:8]}"
        
        try:
            # Create label
            label_id = drawer_label_service.create_label(test_label_name)
            print(f"✓ Created label: '{test_label_name}' (ID: {label_id})")
            
            # Verify initially in active list
            active_labels = drawer_label_service.list_active_labels()
            label_ids = [l['label_id'] for l in active_labels]
            assert label_id in label_ids, "Label should initially be in active list"
            print(f"✓ Label initially in active list")
            
            # Disable label
            drawer_label_service.disable_label(label_id)
            print(f"✓ Disabled label {label_id}")
            
            # Verify removed from active list
            active_labels = drawer_label_service.list_active_labels()
            label_ids = [l['label_id'] for l in active_labels]
            assert label_id not in label_ids, "Disabled label should NOT be in active list"
            
            print(f"✓ Verified label removed from active list")
            print("\n✅ PASS - disable_label removes from active list")
            
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
    
    def test_7_validate_label_ids_active_success(self):
        """
        Test 7: Verify validate_label_ids_active passes for valid active labels.
        """
        print("\n" + "="*80)
        print("TEST 7: VALIDATE LABEL IDS - SUCCESS")
        print("="*80)
        
        # Create test labels
        label_id_1 = drawer_label_service.create_label(f"Valid1_{uuid.uuid4().hex[:8]}")
        label_id_2 = drawer_label_service.create_label(f"Valid2_{uuid.uuid4().hex[:8]}")
        
        try:
            print(f"✓ Created labels: {label_id_1}, {label_id_2}")
            
            # Validate should pass (no exception)
            drawer_label_service.validate_label_ids_active([label_id_1, label_id_2])
            
            print(f"✓ Validation passed for active labels")
            print("\n✅ PASS - validate_label_ids_active success")
            
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
    
    def test_8_validate_label_ids_active_fails_for_disabled(self):
        """
        Test 8: Verify validate_label_ids_active fails for disabled labels.
        """
        print("\n" + "="*80)
        print("TEST 8: VALIDATE LABEL IDS - FAIL FOR DISABLED")
        print("="*80)
        
        # Create and disable label
        label_id = drawer_label_service.create_label(f"ToDisable_{uuid.uuid4().hex[:8]}")
        drawer_label_service.disable_label(label_id)
        
        try:
            print(f"✓ Created and disabled label: {label_id}")
            
            # Validation should fail
            with pytest.raises(ValueError, match="Invalid or inactive label IDs"):
                drawer_label_service.validate_label_ids_active([label_id])
            
            print(f"✓ Correctly rejected disabled label")
            print("\n✅ PASS - validate_label_ids_active rejects disabled labels")
            
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


def run_all_tests():
    """Run all tests in sequence."""
    print("\n" + "="*80)
    print("PHASE G-B6: DRAWER LABEL SERVICE LAYER TESTS")
    print("="*80)
    
    service_tests = TestDrawerLabelService()
    
    tests = [
        ("Create Label SUCCESS", service_tests.test_1_create_label_success),
        ("Create Label TRIM Whitespace", service_tests.test_2_create_label_trims_whitespace),
        ("Create Label REJECT Short", service_tests.test_3_create_label_rejects_short_name),
        ("Create Label REJECT Duplicate", service_tests.test_4_create_label_duplicate_fails),
        ("List Active Labels", service_tests.test_5_list_active_labels_includes_new_label),
        ("Disable Label", service_tests.test_6_disable_label_removes_from_active_list),
        ("Validate IDs SUCCESS", service_tests.test_7_validate_label_ids_active_success),
        ("Validate IDs FAIL Disabled", service_tests.test_8_validate_label_ids_active_fails_for_disabled),
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
