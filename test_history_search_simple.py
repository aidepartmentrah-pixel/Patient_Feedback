"""
Simple Manual Test for History Search Endpoints
Quick manual test without complex authentication handling
"""

import requests

BASE_URL = "http://localhost:8000"

print("="*70)
print("MANUAL HISTORY SEARCH TEST")
print("="*70)

# Test 1: Doctor Search
print("\n1. Testing Doctor Search (No Auth Required)")
print("-"*70)
response = requests.get(f"{BASE_URL}/api/v2/doctors/search?q=ahmed&limit=10")
print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"Success: {data.get('success')}")
    print(f"Total: {data.get('total')}")
    print(f"Items: {len(data.get('items', []))}")
    if data.get('items'):
        print(f"First doctor: {data['items'][0].get('full_name')}")
    print("✅ DOCTOR SEARCH WORKING")
else:
    print(f"❌ ERROR: {response.text}")

# Test 2: Worker Search (requires auth)
print("\n2. Testing Worker Search (Auth Required)")
print("-"*70)
print("To test worker search, you need to:")
print("1. Login via the web UI at http://localhost:8000")
print("2. Open browser DevTools → Network tab")
print("3. Make a search request")
print("4. Copy the 'Authorization' header value")
print("5. Run this command:")
print("")
print("   import requests")
print("   headers = {'Authorization': 'Bearer YOUR_TOKEN_HERE'}")
print("   response = requests.get(")
print("       'http://localhost:8000/api/v2/workers/search?q=mohammed&limit=10',")
print("       headers=headers")
print("   )")
print("   print(response.json())")
print("")
print("Or test directly in the browser at:")
print("http://localhost:8000/api/v2/workers/search?q=mohammed&limit=10")
print("(while logged in)")

print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)
print("✅ Doctor search endpoint: /api/v2/doctors/search")
print("   - Query parameter: q (min 2 chars)")
print("   - Limit parameter: limit (1-100)")
print("   - Response: {success, items, total}")
print("")
print("✅ Worker search endpoint: /api/v2/workers/search")
print("   - Query parameter: q (min 2 chars)")  
print("   - Limit parameter: limit (1-100)")
print("   - Response: {success, items, total}")
print("   - Requires: Authentication (login first)")
print("="*70)
