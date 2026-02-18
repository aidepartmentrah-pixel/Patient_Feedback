"""
Test Aggregate Doctors Seasonal Word Report
Tests the modified all-doctors comparison report with filtering
"""

import requests
import os
from datetime import datetime

BASE_URL = "http://127.0.0.1:8000"

def test_aggregate_doctors_word():
    """Test aggregate doctors seasonal comparison report"""
    print("\n" + "="*70)
    print("AGGREGATE DOCTORS COMPARISON REPORT TEST")
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
    
    # Test parameters - use same dates that worked for single doctor  
    season_start = "2025-01-01"
    season_end = "2025-12-31"
    
    print(f"\n[TEST] Generating aggregate report")
    print(f"  Season: {season_start} to {season_end}")
    print("-" * 70)
    
    try:
        # Test aggregate report
        print("\n1. Testing aggregate doctors Word export...")
        export_response = session.get(
            f"{BASE_URL}/api/person-reports/doctors/all-seasonal-word",
            params={
                "season_start": season_start,
                "season_end": season_end
            },
            stream=True,
            timeout=60  # Longer timeout for aggregate report
        )
        
        print(f"   Status: {export_response.status_code}")
        
        if export_response.status_code == 200:
            # Check content type
            content_type = export_response.headers.get('content-type', '')
            print(f"   ✓ Content-Type: {content_type}")
            
            # Check filename
            content_disp = export_response.headers.get('content-disposition', '')
            print(f"   ✓ Content-Disposition: {content_disp}")
            
            # Save the file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"aggregate_doctors_{timestamp}.docx"
            
            with open(filename, 'wb') as f:
                for chunk in export_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Get file size
            file_size = os.path.getsize(filename)
            print(f"   ✓ Saved: {filename}")
            print(f"   ✓ Size: {file_size:,} bytes")
            
            if file_size > 1000:
                print(f"\n   ✅ SUCCESS! Aggregate doctors comparison report generated!")
                print(f"\n   📄 Open the file to verify it contains:")
                print(f"      • Arabic title: 'تقرير مقارنة الأطباء الموسمي'")
                print(f"      • Summary statistics (total cases, averages)")
                print(f"      • COMPARISON TABLE with ALL doctors who have cases:")
                print(f"        - Sorted by total cases (most first)")
                print(f"        - Columns: #, Name, Specialty, Total, High, Medium, Low")
                print(f"        - NO doctors with zero cases")
                print(f"      • Professional Arabic formatting")
                print(f"\n   File path: {os.path.abspath(filename)}")
            else:
                print(f"\n   ⚠ Warning: File seems small ({file_size} bytes)")
        
        elif export_response.status_code == 404:
            print(f"   ✗ Failed: 404 - Endpoint not found")
            print(f"   Response: {export_response.text}")
        elif export_response.status_code == 400:
            print(f"   ✗ Failed: 400 - Bad request")
            print(f"   Response: {export_response.text}")
        else:
            print(f"   ✗ Failed: {export_response.status_code}")
            print(f"   Response: {export_response.text[:500]}")
        
    except requests.exceptions.Timeout:
        print(f"   ✗ ERROR: Request timed out after 60 seconds")
    except Exception as e:
        print(f"   ✗ ERROR: {str(e)}")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print("\n💡 TIP: Open the generated Word file to verify:")
    print("   1. Only doctors WITH cases appear (filtered)")
    print("   2. Table sorted by total cases descending")
    print("   3. Professional Arabic headers and formatting")
    print("   4. Summary statistics at the top")

if __name__ == "__main__":
    test_aggregate_doctors_word()
