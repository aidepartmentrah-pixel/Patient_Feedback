"""
Phase 5: Manual API Test - POST /api/doctors

Run this after starting the server manually:
cd backend
uvicorn main:app --reload

Then run this script:
python test_phase5_manual.py
"""

import requests
import json
from datetime import datetime


BASE_URL = "http://localhost:8000"


def main():
    print("\n" + "="*70)
    print(" PHASE 5: POST /api/doctors - Manual Test")
    print("="*70)
    
    # Test 1: Create a doctor
    print("\n1️⃣  Creating a new doctor...")
    print("-" * 70)
    
    payload = {
        "doctor_name": f"Dr. Manual Test {datetime.now().strftime('%H%M%S')}",
        "specialty": "Interventional Cardiology",
        "is_active": True,
        "source_system": "MANUAL"
    }
    
    print(f"\n📤 REQUEST:")
    print(f"POST {BASE_URL}/api/doctors")
    print(f"Content-Type: application/json\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    
    try:
        response = requests.post(f"{BASE_URL}/api/doctors", json=payload, timeout=5)
        
        print(f"\n📥 RESPONSE:")
        print(f"Status: {response.status_code} {response.reason}")
        print(f"Content-Type: {response.headers.get('content-type')}\n")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        
        if response.status_code == 201:
            print("\n✅ SUCCESS: Doctor created!")
            doctor_id = response.json().get('doctor', {}).get('id')
            
            # Test 2: Search for the created doctor
            print("\n\n2️⃣  Searching for the created doctor...")
            print("-" * 70)
            
            search_response = requests.get(
                f"{BASE_URL}/api/doctors",
                params={"query": "Manual Test", "limit": 5}
            )
            
            print(f"\n📤 REQUEST:")
            print(f"GET {BASE_URL}/api/doctors?query=Manual+Test&limit=5")
            
            print(f"\n📥 RESPONSE:")
            print(f"Status: {search_response.status_code}")
            search_data = search_response.json()
            print(f"Total found: {search_data.get('total', 0)}")
            
            if search_data.get('doctors'):
                print(f"\nFirst result:")
                first = search_data['doctors'][0]
                print(f"  • ID: {first.get('id')}")
                print(f"  • Name: {first.get('name_en')}")
                print(f"  • Source: {first.get('source')}")
                print("\n✅ Doctor is searchable!")
        else:
            print(f"\n❌ ERROR: Expected 201, got {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to server!")
        print("   Make sure the server is running:")
        print("   cd backend")
        print("   uvicorn main:app --reload")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
    
    print("\n" + "="*70)
    print(" Test Complete")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
