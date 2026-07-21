#!/usr/bin/env python
"""
Integration Verification Test
Checks that all ML components are properly connected
"""

import sys
import os

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all required imports work"""
    print("=" * 70)
    print("INTEGRATION VERIFICATION TEST")
    print("=" * 70)
    print()
    
    # Test 1: Import mapper
    try:
        from backend.config.ml_encoding_mapper import load_mapping, map_db_to_ml, get_all_mapped_fields
        print("✓ Mapper imported successfully")
        print("  - load_mapping()")
        print("  - map_db_to_ml()")
        print("  - get_all_mapped_fields()")
    except Exception as e:
        print(f"✗ Mapper import failed: {e}")
        return False
    
    # Test 2: Import adapter
    try:
        from backend.ml_mapping.ml_insert_adapter import add_to_ml_database
        print("✓ ML adapter imported successfully")
        print("  - add_to_ml_database()")
    except Exception as e:
        print(f"✗ Adapter import failed: {e}")
        return False
    
    # Test 3: Load mapping
    try:
        mapping = load_mapping()
        fields = get_all_mapped_fields()
        print(f"✓ Mapping loaded successfully")
        print(f"  - Available entity types: {fields}")
    except Exception as e:
        print(f"✗ Mapping load failed: {e}")
        return False
    
    # Test 4: Test mappings
    try:
        test_mappings = [
            ('domain', 1),
            ('category', 1),
            ('subcategory', 1),
            ('severity_level', 1),
        ]
        
        print("✓ Testing sample mappings:")
        for entity, db_id in test_mappings:
            result = map_db_to_ml(entity, db_id)
            print(f"  - map_db_to_ml('{entity}', {db_id}) = {result}")
    except Exception as e:
        print(f"✗ Mapping test failed: {e}")
        return False
    
    # Test 5: Verify insert service has hook
    try:
        with open('backend/api/services/insert_service.py', 'r') as f:
            content = f.read()
            if 'add_to_ml_database' in content and 'ML INSERT HOOK' in content:
                print("✓ ML hook found in insert_service.py")
            else:
                print("✗ ML hook not found in insert_service.py")
                return False
    except Exception as e:
        print(f"✗ Could not verify insert_service.py: {e}")
        return False
    
    print()
    print("=" * 70)
    print("✓ ALL INTEGRATION CHECKS PASSED")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Run backend/test_ml_hook_integration.py for full tests")
    print("2. Monitor console for [ML INSERT WARNING] messages during operation")
    print("3. Begin training pipeline with accumulated ML data")
    print()
    return True


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
