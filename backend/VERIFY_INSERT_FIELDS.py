"""
Verify Insert Flow: Check that all fields reach ML and APP databases
"""

import sys
from pathlib import Path

workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

print("\n" + "="*80)
print("VERIFY INSERT FLOW: Missing Fields Check")
print("="*80)

# Check 1: API Model has all 4 ML fields
print("\n[CHECK 1] API Model - 4 ML Fields")
try:
    from backend.api.routers.insert_router import CreateRecordRequest
    
    if hasattr(CreateRecordRequest, 'model_fields'):
        fields = CreateRecordRequest.model_fields
    else:
        fields = CreateRecordRequest.__fields__
    
    ml_fields = [
        'feedback_type',
        'improvement_opportunity_type',
        'classification_ar',
        'classification_en'
    ]
    
    for field in ml_fields:
        if field in fields:
            print(f"  [OK] {field}")
        else:
            print(f"  [FAIL] {field}")
    
    # Check for APP database fields
    app_fields = [
        'patient_name',
        'building_id',
        'explanation_status_id'
    ]
    
    for field in app_fields:
        if field in fields:
            print(f"  [OK] {field} (APP DB)")
        else:
            print(f"  [FAIL] {field} (APP DB)")
    
except Exception as e:
    print(f"  [ERROR] {e}")

# Check 2: ML Adapter has DIRECT_FIELDS mapping
print("\n[CHECK 2] ML Adapter - DIRECT_FIELDS Mapping")
try:
    from backend.ml_mapping.ml_insert_adapter import DIRECT_FIELDS
    
    ml_fields = [
        'feedback_type',
        'improvement_opportunity_type',
        'classification_ar',
        'classification_en'
    ]
    
    for field in ml_fields:
        if field in DIRECT_FIELDS:
            print(f"  [OK] {field} -> {DIRECT_FIELDS[field]}")
        else:
            print(f"  [FAIL] {field} NOT IN DIRECT_FIELDS")
    
except Exception as e:
    print(f"  [ERROR] {e}")

# Check 3: ML Database KNOWN_COLUMNS
print("\n[CHECK 3] ML Database - KNOWN_COLUMNS")
try:
    from backend.ml_mapping.ml_insert_adapter import KNOWN_COLUMNS
    
    ml_columns = [
        'feedback_type',
        'improvement_opportunity_type',
        'classification_ar',
        'classification_en'
    ]
    
    for col in ml_columns:
        if col in KNOWN_COLUMNS:
            print(f"  [OK] {col}")
        else:
            print(f"  [FAIL] {col} NOT IN KNOWN_COLUMNS")
    
except Exception as e:
    print(f"  [ERROR] {e}")

# Check 4: insert_service passes data to ML wrapper
print("\n[CHECK 4] insert_service - ML Wrapper Call")
try:
    service_path = Path(__file__).parent / "backend" / "api" / "services" / "insert_service.py"
    service_code = service_path.read_text()
    
    if "add_corrected_record_to_ml(data)" in service_code:
        print(f"  [OK] Wrapper called with full data dict")
    else:
        print(f"  [FAIL] Wrapper not called correctly")
    
except Exception as e:
    print(f"  [ERROR] {e}")

# Check 5: APP Database payload includes patient_name, building_id, explanation_status_id
print("\n[CHECK 5] insert_service - APP Database Payload")
try:
    service_path = Path(__file__).parent / "backend" / "api" / "services" / "insert_service.py"
    service_code = service_path.read_text()
    
    checks = [
        ('PatientName', 'patient_name'),
        ('BuildingID', 'building_id'),
        ('ExplanationStatusID', 'explanation_status_id')
    ]
    
    for db_field, api_field in checks:
        if f'"{db_field}": data.get(\'{api_field}\')' in service_code or \
           f'"{db_field}\": data.get(\"{api_field}\")' in service_code or \
           f'"{db_field}":' in service_code:
            print(f"  [OK] {db_field} in payload")
        else:
            print(f"  [WARN] {db_field} might not be in payload")
    
except Exception as e:
    print(f"  [ERROR] {e}")

print("\n" + "="*80)
print("INTEGRATION TEST: Complete Request Flow")
print("="*80)

# Simulate a complete request
print("\n[TEST] Create request with all fields...")

try:
    from datetime import date
    
    test_request = {
        # Required APP DB fields
        'complaint_text': 'Test complaint',
        'feedback_received_date': date(2026, 1, 5),
        'domain_id': 1,
        'category_id': 1,
        'severity_id': 1,
        'issuing_department_id': 1,
        
        # Optional APP DB fields
        'patient_name': 'Ahmed Ali',
        'building_id': 2,
        'explanation_status_id': 1,
        
        # ML Training fields
        'feedback_type': 1,
        'improvement_opportunity_type': 2,
        'classification_ar': 8.5,
        'classification_en': 5,
        
        # Other fields
        'subcategory_id': 10,
        'classification_id': 100,
        'stage_id': 1,
        'harm_id': 1,
    }
    
    # Check if all fields would be accepted
    from backend.api.routers.insert_router import CreateRecordRequest
    
    request_obj = CreateRecordRequest(**test_request)
    request_dict = request_obj.model_dump()
    
    print(f"  [OK] Request created successfully")
    print(f"\n  Checking data dict has all fields:")
    
    check_fields = [
        'feedback_type',
        'improvement_opportunity_type',
        'classification_ar',
        'classification_en',
        'patient_name',
        'building_id',
        'explanation_status_id'
    ]
    
    for field in check_fields:
        if field in request_dict:
            value = request_dict[field]
            print(f"    [OK] {field}: {value}")
        else:
            print(f"    [FAIL] {field}: MISSING")
    
except Exception as e:
    print(f"  [ERROR] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("STATUS: Check complete")
print("="*80 + "\n")
