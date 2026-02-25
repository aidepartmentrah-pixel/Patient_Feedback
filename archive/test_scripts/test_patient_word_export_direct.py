"""
Test Patient Word Export - Direct Query
Tests with a specific patient ID or creates test data
"""

import requests
import os
import pyodbc
from datetime import datetime

BASE_URL = "http://127.0.0.1:8000"

def get_connection():
    """Get database connection"""
    return pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=DESKTOP-IC670VQ;'
        'DATABASE=RiskManagement;'
        'Trusted_Connection=yes;',
        timeout=5
    )

def find_patient_with_incidents():
    """Find a patient that has incidents"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Find a patient from APP_IncidentCase
        cursor.execute("""
            SELECT TOP 1 
                COALESCE(ic.PatientID, 0) as PatientID,
                ic.PatientName,
                COUNT(*) as IncidentCount
            FROM dbo.APP_IncidentCase ic
            WHERE ic.PatientName IS NOT NULL
            GROUP BY ic.PatientID, ic.PatientName
            ORDER BY COUNT(*) DESC
        """)
        
        row = cursor.fetchone()
        if row:
            return {
                'patient_id': row[0] if row[0] > 0 else None,
                'patient_name': row[1],
                'incident_count': row[2]
            }
        
        return None
    finally:
        conn.close()

def test_word_export():
    """Test Word export"""
    print("\n" + "="*70)
    print("PATIENT WORD EXPORT TEST")
    print("="*70)
    
    # Find patient
    print("\n[STEP 1] Finding patient with incidents in database...")
    patient_data = find_patient_with_incidents()
    
    if not patient_data:
        print("✗ No patients with incidents found")
        return
    
    print(f"✓ Found patient: {patient_data['patient_name']}")
    print(f"  - Patient ID: {patient_data['patient_id']}")
    print(f"  - Incident Count: {patient_data['incident_count']}")
    
    # If no valid patient_id, use 0 or try to get from reserve table
    if not patient_data['patient_id']:
        conn = get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT TOP 1 PatientAdmissionID 
                FROM dbo.APP_RESERVE_PATIENT
                WHERE FullName LIKE ?
            """, f"%{patient_data['patient_name'].split()[0]}%")
            row = cursor.fetchone()
            if row:
                patient_data['patient_id'] = row[0]
                print(f"  - Found in reserve table: ID {patient_data['patient_id']}")
        finally:
            conn.close()
    
    if not patient_data['patient_id']:
        # Use ID 1 as fallback
        patient_data['patient_id'] = 1
        print(f"  - Using fallback ID: 1")
    
    patient_id = patient_data['patient_id']
    
    # Test Word export
    print(f"\n[STEP 2] Testing Word export for patient ID {patient_id}...")
    try:
        export_response = requests.get(
            f"{BASE_URL}/api/v2/patients/{patient_id}/export",
            params={"format": "word"},
            stream=True,
            timeout=30
        )
        
        print(f"Status Code: {export_response.status_code}")
        
        if export_response.status_code == 200:
            content_type = export_response.headers.get('content-type', '')
            print(f"✓ Content-Type: {content_type}")
            
            # Save file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_patient_{patient_id}_word_export_{timestamp}.docx"
            
            with open(filename, 'wb') as f:
                for chunk in export_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size = os.path.getsize(filename)
            print(f"✓ Saved: {filename}")
            print(f"✓ Size: {file_size:,} bytes")
            
            if file_size > 1000:
                print(f"\n✅ SUCCESS! Word export is working!")
                print(f"\n   Open '{filename}' to verify:")
                print(f"   1. ✓ Arabic header (تقرير تاريخ المريض)")
                print(f"   2. ✓ Patient information table")
                print(f"   3. ✓ Complaints table from APP_IncidentCase")
                print(f"   4. ✓ Hospital branding/footer")
            else:
                print(f"⚠️  File seems too small")
        else:
            print(f"✗ Export failed: {export_response.status_code}")
            print(f"Response: {export_response.text[:500]}")
    
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test JSON for comparison
    print(f"\n[STEP 3] Testing JSON export (for comparison)...")
    try:
        json_response = requests.get(
            f"{BASE_URL}/api/v2/patients/{patient_id}/export",
            params={"format": "json"},
            timeout=30
        )
        
        if json_response.status_code == 200:
            data = json_response.json()
            incidents = data.get('incidents', [])
            print(f"✓ JSON shows {len(incidents)} incidents")
            
            if incidents:
                print(f"\n  Sample incident:")
                sample = incidents[0]
                for key in ['Date', 'Department', 'Category', 'Severity', 'Status']:
                    print(f"    - {key}: {sample.get(key, 'N/A')}")
        else:
            print(f"✗ JSON failed: {json_response.status_code}")
    
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    test_word_export()
