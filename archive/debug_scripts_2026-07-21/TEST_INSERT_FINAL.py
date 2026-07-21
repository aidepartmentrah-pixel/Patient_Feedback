import sys
from pathlib import Path
from datetime import date
import os
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.database import get_connection

print("\n" + "="*80)
print("TEST: Insert Record with All 7 Missing Fields")
print("="*80)

# Use simple valid IDs that definitely exist
test_data = {
    # APP Database required fields
    'complaint_text': 'Test complaint for insert with all fields',
    'feedback_received_date': date(2026, 1, 5),
    'issuing_department_id': 1,  # Should exist
    'domain_id': 1,  # Clinical
    'category_id': 6,  # Quality of Care
    'subcategory_id': 19,  # Examination & Monitoring
    'classification_id': 132,  # Daily Doctor Visits
    'severity_id': 1,
    'stage_id': 1,
    'harm_id': 1,
    
    # APP Database optional - MISSING FIELDS
    'patient_name': 'Test Patient Name XYZ 123',
    'building_id': 1,  # RAH
    'explanation_status_id': 1,  # Waiting
    
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

print("\n[STEP 2] Check APP database for record with 'XYZ 123' name...")
try:
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get the record we just inserted
    cursor.execute("""
        SELECT TOP 1
            IncidentRequestCaseID, PatientName, BuildingID, ExplanationStatusID, FeedbackRecievedDate
        FROM APP_IncidentCase
        WHERE PatientName LIKE '%XYZ 123%'
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
        print("\n  >>> APP DB INSERT SUCCESS - All 3 optional fields present!")
    else:
        print("  >>> No record with 'XYZ 123' found - INSERT FAILED")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()

print("\n[STEP 3] Check ML database for record...")
try:
    import sqlite3
    
    ml_db_path = r"C:\Users\IT\Documents\GitHub Repository\Patient_Feedback\models_directory\patient_feedback_ml.db"
    
    if os.path.exists(ml_db_path):
        conn = sqlite3.connect(ml_db_path)
        cursor = conn.cursor()
        
        # Get the last inserted record
        cursor.execute("""
            SELECT 
                id, feedback_type, improvement_opportunity_type, classification_ar, classification_en,
                complaint_text
            FROM patient_feedback_encoded
            ORDER BY id DESC
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        if row:
            print(f"  Found record in ML DB:")
            print(f"    ID: {row[0]}")
            print(f"    feedback_type: {row[1]}")
            print(f"    improvement_opportunity_type: {row[2]}")
            print(f"    classification_ar: {row[3]}")
            print(f"    classification_en: {row[4]}")
            print(f"    complaint_text: {row[5][:50]}...")
            print("\n  >>> ML DB INSERT SUCCESS - All 4 ML type fields present!")
        else:
            print("  No records found in ML database")
        
        cursor.close()
        conn.close()
    else:
        print(f"  ML database not found at: {ml_db_path}")
    
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80 + "\n")
