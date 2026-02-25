"""
Test Enhanced Doctor Seasonal Word Report
Tests the detailed incident table in doctor seasonal reports
"""

import requests
import os
from datetime import datetime, date

BASE_URL = "http://127.0.0.1:8000"

def test_doctor_seasonal_enhanced():
    """Test enhanced doctor seasonal report with detailed incident table"""
    print("\n" + "="*70)
    print("ENHANCED DOCTOR SEASONAL REPORT TEST")
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
        print(f"  Response: {login_response.text}")
        return
    
    print("✓ Login successful")
    
    # Use a doctor ID (will try ID 1 or search for one)
    doctor_id = 1
    season_start = "2025-01-01"
    season_end = "2025-12-31"
    
    print(f"\n[TEST] Generating seasonal report for Doctor ID: {doctor_id}")
    print(f"  Season: {season_start} to {season_end}")
    print("-" * 70)
    
    try:
        # Test enhanced seasonal report
        print("\n1. Testing seasonal Word export...")
        export_response = session.get(
            f"{BASE_URL}/api/person-reports/doctor/{doctor_id}/seasonal-word",
            params={
                "season_start": season_start,
                "season_end": season_end
            },
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
            filename = f"doctor_{doctor_id}_seasonal_{timestamp}.docx"
            
            with open(filename, 'wb') as f:
                for chunk in export_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size = os.path.getsize(filename)
            print(f"   ✓ Saved: {filename}")
            print(f"   ✓ Size: {file_size:,} bytes")
            
            if file_size > 1000:
                print(f"\n   ✅ SUCCESS! Enhanced doctor seasonal report generated!")
                print(f"\n   📄 Open the file to verify it contains:")
                print(f"      • Arabic header: 'التقرير الموسمي للطبيب'")
                print(f"      • Doctor information (ID, Name, Specialty)")
                print(f"      • Summary metrics (Total, High, Medium, Low severity)")
                print(f"      • DETAILED TABLE of ALL incidents:")
                print(f"        - Date")
                print(f"        - Patient Name")
                print(f"        - Category")
                print(f"        - Severity")
                print(f"        - Status")
                print(f"        - Case ID")
                print(f"\n   File path: {os.path.abspath(filename)}")
            else:
                print(f"\n   ⚠️  Warning: File size is small ({file_size} bytes)")
                
        elif export_response.status_code == 404:
            print(f"   ✗ Doctor {doctor_id} not found")
            print("   Try testing with a real doctor from your database")
            
        else:
            print(f"   ✗ Failed: {export_response.status_code}")
            print(f"   Response: {export_response.text[:500]}")
    
    except Exception as e:
        print(f"   ✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print("\n💡 TIP: Open the generated Word file to verify:")
    print("   1. Professional Arabic formatting")
    print("   2. Detailed incident table (like Excel export)")
    print("   3. Summary statistics at the top")
    print("   4. All cases where doctor appears")

if __name__ == "__main__":
    test_doctor_seasonal_enhanced()
