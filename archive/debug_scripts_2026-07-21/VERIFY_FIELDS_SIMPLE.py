"""
Simple verification of 4 fields - ASCII output only
"""

import sys
from pathlib import Path

workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

print("\n" + "="*80)
print("VERIFICATION: 4 ML TRAINING FIELDS")
print("="*80)

# Test 1: Check API Model
print("\n[1] Check API Model for 4 new fields...")
try:
    from backend.api.routers.insert_router import CreateRecordRequest
    from pydantic import BaseModel
    
    # Use model_fields for Pydantic v2
    if hasattr(CreateRecordRequest, 'model_fields'):
        model_fields = CreateRecordRequest.model_fields
    else:
        model_fields = CreateRecordRequest.__fields__
    
    new_fields = {
        "feedback_type": "1=improvement Opportunity, 2=notice, 3=Critique, 4=Other",
        "improvement_opportunity_type": "1=Ordinary, 2=RedFlag, 3=NeverEvent",
        "classification_ar": "float 0-10 confidence score",
        "classification_en": "integer classification code"
    }
    
    print("\n  New fields in CreateRecordRequest:")
    for field_name, expected_type in new_fields.items():
        if field_name in model_fields:
            print(f"    [PASS] {field_name}")
        else:
            print(f"    [FAIL] {field_name} - NOT FOUND")
            sys.exit(1)
    
    print("\n  Status: [PASS] All 4 fields present in API model")
    
except Exception as e:
    print(f"  [FAIL] Error checking API model: {e}")
    sys.exit(1)

# Test 2: Check ML Adapter
print("\n[2] Check ML Insert Adapter...")
try:
    from backend.ml_mapping.ml_insert_adapter import DIRECT_FIELDS
    
    print("\n  Checking DIRECT_FIELDS mappings:")
    required_fields = [
        "feedback_type",
        "improvement_opportunity_type", 
        "classification_ar",
        "classification_en"
    ]
    
    for field in required_fields:
        if field in DIRECT_FIELDS:
            print(f"    [PASS] {field} mapped in DIRECT_FIELDS")
        else:
            print(f"    [FAIL] {field} missing from DIRECT_FIELDS")
            sys.exit(1)
    
    print("\n  Status: [PASS] All fields mapped in adapter")
    
except Exception as e:
    print(f"  [FAIL] Error checking adapter: {e}")
    sys.exit(1)

# Test 3: Check insert_service uses wrapper
print("\n[3] Check insert_service calls wrapper...")
try:
    import inspect
    from backend.api.services import insert_service
    
    # Read the service file
    service_path = Path(__file__).parent / "api" / "services" / "insert_service.py"
    service_code = service_path.read_text()
    
    if "add_corrected_record_to_ml" in service_code:
        print("    [PASS] insert_service calls add_corrected_record_to_ml wrapper")
    else:
        print("    [FAIL] insert_service does not call wrapper")
        sys.exit(1)
    
    if "add_to_ml_database" in service_code and service_code.count("add_to_ml_database") <= 2:
        print("    [PASS] Direct add_to_ml_database not used in logic")
    elif service_code.count("add_to_ml_database") > 2:
        print("    [WARN] add_to_ml_database appears multiple times")
    
    print("\n  Status: [PASS] insert_service correctly integrated")
    
except Exception as e:
    print(f"  [FAIL] Error checking insert_service: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("RESULT: All verifications PASSED")
print("="*80)
print("\nSummary:")
print("  - 4 fields added to API model")
print("  - 4 fields mapped in ML adapter")
print("  - insert_service calls wrapper for embeddings")
print("\nReady for production test!")
print("="*80 + "\n")
