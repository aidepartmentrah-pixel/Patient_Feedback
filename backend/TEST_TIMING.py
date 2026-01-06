import sys
from pathlib import Path
from datetime import date
import time
sys.path.insert(0, str(Path(__file__).parent.parent))

print("\n" + "="*60)
print("TIMING TEST: Which step is slow?")
print("="*60)

test_data = {
    'complaint_text': 'Quick test',
    'feedback_received_date': date(2026, 1, 6),
    'issuing_department_id': 1,
    'domain_id': 1,
    'category_id': 6,
    'subcategory_id': 19,
    'classification_id': 132,
    'severity_id': 1,
    'stage_id': 1,
    'harm_id': 1,
    'patient_name': 'Test Patient',
    'building_id': 1,
    'explanation_status_id': 1,
    'feedback_type': 1,
    'improvement_opportunity_type': 2,
    'classification_ar': 8.5,
    'classification_en': 5,
}

start = time.time()
print(f"\n[{time.time()-start:.2f}s] Importing insert service...")
from backend.api.services.insert_service import create_record

print(f"[{time.time()-start:.2f}s] Starting insert...")
result = create_record(test_data)

print(f"[{time.time()-start:.2f}s] INSERT COMPLETE")
print(f"  Success: {result['success']}")
print(f"  ID: {result.get('id')}")

print("\n" + "="*60 + "\n")
