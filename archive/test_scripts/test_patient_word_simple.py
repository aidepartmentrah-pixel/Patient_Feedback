"""
Test Patient Word Export - Simple Test
Tests Word export with a known patient ID
"""

import requests
import os
from datetime import datetime

BASE_URL = "http://127.0.0.1:8000"

def test_word_export_simple():
    """Simple test with patient ID 100022 from earlier search"""
    print("\n" + "="*70)
    print("PATIENT WORD EXPORT - SIMPLE TEST")
    print("="*70)
    
    # Use patient ID we found earlier: 100022 (محمد from reserve table)
    patient_id = 100022
    
    print(f"\n[TEST] Exporting Word document for Patient ID: {patient_id}")
    print("-" * 70)
    
    try:
        # Test Word export
        print("\n1. Testing Word export...")
        export_response = requests.get(
            f"{BASE_URL}/api/v2/patients/{patient_id}/export",
            params={"format": "word"},
            stream=True,
            timeout=30
        )
        
        print(f"   Status: {export_response.status_code}")
        
        if export_response.status_code == 200:
            content_type = export_response.headers.get('content-type', '')
            content_disposition = export_response.headers.get('content-disposition', '')
            
            print(f"   ✓ Content-Type: {content_type}")
            print(f"   ✓ Content-Disposition: {content_disposition}")
            
            # Save file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"patient_{patient_id}_report_{timestamp}.docx"
            
            with open(filename, 'wb') as f:
                for chunk in export_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size = os.path.getsize(filename)
            print(f"   ✓ Saved: {filename}")
            print(f"   ✓ Size: {file_size:,} bytes")
            
            if file_size > 1000:
                print(f"\n   ✅ SUCCESS! Word document generated!")
                print(f"\n   📄 Open the file to verify it contains:")
                print(f"      • Arabic title: 'تقرير تاريخ المريض'")
                print(f"      • Patient info: ID, MRN, Name")
                print(f"      • Complaints table (table from APP_IncidentCase)")
                print(f"      • Hospital footer in Arabic")
                print(f"\n   File path: {os.path.abspath(filename)}")
            else:
                print(f"\n   ⚠️  Warning: File size is small ({file_size} bytes)")
                
        elif export_response.status_code == 404:
            print(f"   ✗ Patient {patient_id} not found")
            print("   Trying with a different ID...")
            
            # Try ID 1
            test_word_export_with_id(1)
            
        else:
            print(f"   ✗ Failed: {export_response.status_code}")
            print(f"   Response: {export_response.text[:300]}")
    
    except Exception as e:
        print(f"   ✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test JSON for data verification
    print("\n2. Testing JSON export (verify data)...")
    try:
        json_response = requests.get(
            f"{BASE_URL}/api/v2/patients/{patient_id}/export",
            params={"format": "json"},
            timeout=15
        )
        
        if json_response.status_code == 200:
            data = json_response.json()
            patient = data.get('patient', {})
            incidents = data.get('incidents', [])
            
            print(f"   ✓ Patient: {patient.get('full_name', 'N/A')}")
            print(f"   ✓ MRN: {patient.get('mrn', 'N/A')}")
            print(f"   ✓ Incidents found: {len(incidents)}")
            
            if incidents:
                print(f"\n   Sample incident:")
                sample = incidents[0]
                print(f"     - Date: {sample.get('Date', 'N/A')}")
                print(f"     - Dept: {sample.get('Department', 'N/A')}")
                print(f"     - Category: {sample.get('Category', 'N/A')}")
        else:
            print(f"   Status: {json_response.status_code}")
    
    except Exception as e:
        print(f"   ✗ ERROR: {str(e)}")
    
    # Test CSV (ensure we didn't break it)
    print("\n3. Testing CSV export (backward compatibility)...")
    try:
        csv_response = requests.get(
            f"{BASE_URL}/api/v2/patients/{patient_id}/export",
            params={"format": "csv"},
            timeout=15
        )
        
        if csv_response.status_code == 200:
            print(f"   ✓ CSV export still works (Status: 200)")
        else:
            print(f"   ✗ CSV failed: {csv_response.status_code}")
    
    except Exception as e:
        print(f"   ✗ ERROR: {str(e)}")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


def test_word_export_with_id(patient_id):
    """Test with specific ID"""
    print(f"\n   Testing with Patient ID: {patient_id}...")
    try:
        export_response = requests.get(
            f"{BASE_URL}/api/v2/patients/{patient_id}/export",
            params={"format": "word"},
            stream=True,
            timeout=30
        )
        
        if export_response.status_code == 200:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"patient_{patient_id}_report_{timestamp}.docx"
            
            with open(filename, 'wb') as f:
                for chunk in export_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size = os.path.getsize(filename)
            print(f"   ✓ SUCCESS with ID {patient_id}!")
            print(f"   ✓ File: {filename} ({file_size:,} bytes)")
        else:
            print(f"   ✗ Failed with ID {patient_id}: {export_response.status_code}")
    
    except Exception as e:
        print(f"   ✗ ERROR: {str(e)}")


if __name__ == "__main__":
    test_word_export_simple()
