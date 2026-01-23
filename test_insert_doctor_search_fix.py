"""
Test to verify that reserve doctors appear in Insert Page doctor search.

This test creates a doctor in the reserve table, then searches for it
using the Insert Page search endpoint.
"""

import requests
from datetime import datetime

BASE_URL = "http://localhost:8000"


def test_reserve_doctor_in_insert_search():
    """
    Test that doctors created in reserve table appear in Insert Page search.
    """
    print("="*70)
    print("TEST: Reserve Doctors in Insert Page Search")
    print("="*70)
    
    # Step 1: Create a unique reserve doctor
    unique_name = f"Dr. InsertTest {datetime.now().strftime('%H%M%S')}"
    print(f"\n1. Creating reserve doctor: {unique_name}")
    
    create_payload = {
        "doctor_name": unique_name,
        "specialty": "Test Specialty for Insert Page"
    }
    
    response = requests.post(f"{BASE_URL}/api/doctors", json=create_payload)
    
    if response.status_code != 201:
        print(f"✗ FAIL: Failed to create doctor")
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.text}")
        return False
    
    doctor_data = response.json()
    doctor_id = doctor_data.get('doctor', {}).get('id')
    print(f"✓ Doctor created successfully")
    print(f"  ID: {doctor_id}")
    print(f"  Name: {unique_name}")
    
    # Step 2: Search for the doctor using Insert Page endpoint
    print(f"\n2. Searching via Insert Page endpoint...")
    search_term = unique_name.split()[1]  # Use "InsertTest" part
    
    search_response = requests.get(
        f"{BASE_URL}/api/records/search/doctors",
        params={"q": search_term, "limit": 20}
    )
    
    if search_response.status_code != 200:
        print(f"✗ FAIL: Search request failed")
        print(f"  Status: {search_response.status_code}")
        print(f"  Response: {search_response.text}")
        return False
    
    search_data = search_response.json()
    doctors = search_data.get('doctors', [])
    
    print(f"✓ Search completed")
    print(f"  Total results: {len(doctors)}")
    
    # Step 3: Verify our doctor is in the results
    found = False
    for doc in doctors:
        if doc.get('doctor_id') == doctor_id:
            found = True
            print(f"\n✓ SUCCESS: Reserve doctor found in search results!")
            print(f"  Doctor ID: {doc.get('doctor_id')}")
            print(f"  Name: {doc.get('name')}")
            print(f"  Specialty: {doc.get('speciality_name')}")
            print(f"  Source: {doc.get('source', 'N/A')}")
            break
    
    if not found:
        print(f"\n✗ FAIL: Reserve doctor NOT found in search results")
        print(f"  Expected doctor_id: {doctor_id}")
        print(f"  Expected name: {unique_name}")
        print(f"\nReturned doctors:")
        for doc in doctors[:5]:  # Show first 5
            print(f"  - ID: {doc.get('doctor_id')}, Name: {doc.get('name')}, Source: {doc.get('source', 'N/A')}")
        return False
    
    # Step 4: Test get doctor by ID via Insert Page endpoint
    print(f"\n3. Getting doctor details via Insert Page endpoint...")
    
    detail_response = requests.get(f"{BASE_URL}/api/records/doctor/{doctor_id}")
    
    if detail_response.status_code != 200:
        print(f"✗ FAIL: Get doctor detail failed")
        print(f"  Status: {detail_response.status_code}")
        return False
    
    detail_data = detail_response.json()
    doctor_detail = detail_data.get('doctor', {})
    
    if doctor_detail.get('doctor_id') == doctor_id:
        print(f"✓ Doctor detail retrieved successfully")
        print(f"  Name: {doctor_detail.get('name')}")
        print(f"  Specialty: {doctor_detail.get('speciality_name')}")
        print(f"  Source: {doctor_detail.get('source', 'N/A')}")
    else:
        print(f"✗ FAIL: Doctor detail doesn't match")
        return False
    
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED!")
    print("Reserve doctors now appear in Insert Page search!")
    print("="*70)
    return True


if __name__ == "__main__":
    try:
        success = test_reserve_doctor_in_insert_search()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ TEST ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
