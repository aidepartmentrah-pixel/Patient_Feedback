"""
Test Patient Word Export
Tests the new Word export functionality for patient history
"""

import requests
import os
from datetime import datetime

BASE_URL = "http://127.0.0.1:8000"

def test_patient_word_export():
    """Test the Word export endpoint for patients"""
    print("\n" + "="*70)
    print("TESTING PATIENT WORD EXPORT")
    print("="*70)
    
    # Create session for authentication
    session = requests.Session()
    
    # Login first
    print("\n[AUTH] Logging in...")
    login_response = session.post(
        f"{BASE_URL}/api/auth/login",
        json={"username": "software_admin", "password": "admin123"}
    )
    
    if login_response.status_code != 200:
        print(f"✗ Login failed: {login_response.status_code}")
        return
    
    print("✓ Login successful")
    
    # Step 1: Use a known patient ID
    print("\n[STEP 1] Finding a patient with incidents...")
    patient_id = 100022
    patient_name = "Test Patient"
    print(f"✓ Using known patient ID: {patient_id}")
    
    # Step 2: Test Word export
    print(f"\n[STEP 2] Testing Word export for patient {patient_id}...")
    try:
        export_response = session.get(
            f"{BASE_URL}/api/v2/patients/{patient_id}/export",
            params={"format": "word"},
            stream=True
        )
        
        print(f"Status Code: {export_response.status_code}")
        
        if export_response.status_code == 200:
            # Check content type
            content_type = export_response.headers.get('content-type', '')
            print(f"Content-Type: {content_type}")
            
            # Check if it's a Word document
            if 'wordprocessingml' in content_type or 'application/octet-stream' in content_type:
                print("✓ Received Word document")
                
                # Save the file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"test_patient_{patient_id}_export_{timestamp}.docx"
                
                with open(filename, 'wb') as f:
                    for chunk in export_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                file_size = os.path.getsize(filename)
                print(f"✓ Saved Word document: {filename}")
                print(f"✓ File size: {file_size:,} bytes")
                
                if file_size > 0:
                    print(f"\n✓ SUCCESS! Word export is working!")
                    print(f"  Open the file to verify:")
                    print(f"  - Hospital template (Arabic)")
                    print(f"  - Patient information")
                    print(f"  - Complaints table from APP_IncidentCase")
                else:
                    print(f"✗ FAILED: File is empty")
            else:
                print(f"✗ Wrong content type: {content_type}")
                print(f"Response preview: {export_response.text[:200]}")
        else:
            print(f"✗ Export failed")
            print(f"Response: {export_response.text}")
    
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Step 3: Compare with JSON export
    print(f"\n[STEP 3] Comparing with JSON export (to verify data)...")
    try:
        json_response = requests.get(
            f"{BASE_URL}/api/v2/patients/{patient_id}/export",
            params={"format": "json"}
        )
        
        if json_response.status_code == 200:
            json_data = json_response.json()
            incident_count = len(json_data.get('incidents', []))
            print(f"✓ JSON export shows {incident_count} incidents")
            
            if incident_count > 0:
                print(f"  Sample incident:")
                sample = json_data['incidents'][0]
                print(f"  - Date: {sample.get('Date')}")
                print(f"  - Department: {sample.get('Department')}")
                print(f"  - Category: {sample.get('Category')}")
                print(f"  - Complaint: {sample.get('ComplaintText', '')[:50]}...")
        else:
            print(f"✗ JSON export failed: {json_response.status_code}")
    
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
    
    # Step 4: Test CSV export (ensure it still works)
    print(f"\n[STEP 4] Testing CSV export (ensure old functionality works)...")
    try:
        csv_response = requests.get(
            f"{BASE_URL}/api/v2/patients/{patient_id}/export",
            params={"format": "csv"}
        )
        
        if csv_response.status_code == 200:
            print(f"✓ CSV export still works")
        else:
            print(f"✗ CSV export broken: {csv_response.status_code}")
    
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

if __name__ == "__main__":
    test_patient_word_export()
