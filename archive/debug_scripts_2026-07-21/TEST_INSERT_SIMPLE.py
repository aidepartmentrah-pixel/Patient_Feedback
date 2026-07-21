import sys
from pathlib import Path
from datetime import date
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.database import get_connection

# First get valid IDs
conn = get_connection()
cursor = conn.cursor()

cursor.execute('SELECT TOP 1 ClassificationID FROM APP_LOOKUP_CLASSIFICATION')
classification_id = cursor.fetchone()[0]

cursor.execute('SELECT TOP 1 DomainID FROM APP_LOOKUP_DOMAIN')
domain_id = cursor.fetchone()[0]

cursor.execute('SELECT TOP 1 CategoryID FROM APP_LOOKUP_CATEGORY')
category_id = cursor.fetchone()[0]

cursor.execute('SELECT TOP 1 SubCategoryID FROM APP_LOOKUP_SUBCATEGORY')
subcategory_id = cursor.fetchone()[0]

cursor.execute('SELECT TOP 1 SeverityID FROM APP_LOOKUP_SEVERITY')
severity_id = cursor.fetchone()[0]

cursor.execute('SELECT TOP 1 StageID FROM APP_LOOKUP_CASE_STAGE')
stage_id = cursor.fetchone()[0]

cursor.execute('SELECT TOP 1 HarmLevelID FROM APP_LOOKUP_HARM_LEVEL')
harm_id = cursor.fetchone()[0]

cursor.execute('SELECT TOP 1 DepartmentID FROM APP_LOOKUP_DOMAIN')
department_id = cursor.fetchone()[0]

cursor.close()
conn.close()

print("\n" + "="*80)
print("TEST: Insert Record with All 7 Missing Fields")
print("="*80)

# Prepare test data with valid IDs
test_data = {
    # APP Database required fields
    'complaint_text': 'Test complaint for insert with all fields',
    'feedback_received_date': date(2026, 1, 5),
    'issuing_department_id': department_id,
    'domain_id': domain_id,
    'category_id': category_id,
    'subcategory_id': subcategory_id,
    'classification_id': classification_id,
    'severity_id': severity_id,
    'stage_id': stage_id,
    'harm_id': harm_id,
    
    # APP Database optional - MISSING FIELDS
    'patient_name': 'Test Patient Name XYZ',
    'building_id': 5,
    'explanation_status_id': 1,
    
    # ML Database fields - MISSING FIELDS
    'feedback_type': 1,
    'improvement_opportunity_type': 2,
    'classification_ar': 8.5,
    'classification_en': 5,
    
    # Other optional
    'immediate_action': 'Immediate action taken',
    'taken_action': 'Follow-up action taken',
}

print("\n[STEP 1] Calling insert service...")
try:
    from backend.api.services.insert_service import create_record
    
    result = create_record(test_data)
    
    print(f"\n[RESULT] Success: {result['success']}")
    if result['success']:
        print(f"  Record ID: {result.get('record_id')}")
        print(f"  Database ID: {result.get('id')}")
    else:
        print(f"  Error: {result.get('message')}")
        if result.get('field'):
            print(f"  Field: {result.get('field')}")
    
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()

print("\n[STEP 2] Check APP database for record with 'XYZ' name...")
try:
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get the record we just inserted
    cursor.execute("""
        SELECT TOP 1
            IncidentRequestCaseID, PatientName, BuildingID, ExplanationStatusID, FeedbackRecievedDate
        FROM APP_IncidentCase
        WHERE PatientName LIKE '%XYZ%'
        ORDER BY IncidentRequestCaseID DESC
    """)
    
    row = cursor.fetchone()
    if row:
        print(f"  Found record in APP DB:")
        print(f"    IncidentRequestCaseID: {row[0]}")
        print(f"    PatientName: {row[1]}")
        print(f"    BuildingID: {row[2]}")
        print(f"    ExplanationStatusID: {row[3]}")
        print(f"    Date: {row[4]}")
    else:
        print("  No record with 'XYZ' found")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80 + "\n")
