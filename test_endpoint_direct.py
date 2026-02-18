"""
Direct test of the grouped inbox endpoint
"""
import sys
sys.path.insert(0, 'backend')
import requests

BASE_URL = "http://localhost:8000"

# Try to login with common credentials
test_users = [
    ("administration_admin", "password"),
    ("administration_admin", "123456"),
    ("section_admin", "password"),
    ("dept_admin", "password"),
    ("worker", "password"),
]

print("="*70)
print("TESTING GROUPED INBOX ENDPOINT")
print("="*70)

session = None
for username, password in test_users:
    print(f"\nTrying: {username} / {password}")
    try:
        response = requests.post(
            f"{BASE_URL}/api/auth/login",
            json={"username": username, "password": password},
            timeout=5
        )
        if response.status_code == 200:
            print(f"✅ SUCCESS! Logged in as {username}")
            session = requests.Session()
            # Copy cookies or token
            session.cookies.update(response.cookies)
            break
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   Error: {str(e)[:50]}")

if not session:
    print("\n" + "="*70)
    print("❌ Could not login with any test credentials")
    print("="*70)
    print("\nPlease provide valid credentials:")
    print("Run this manually from Swagger UI at: http://localhost:8000/docs")
    print("Navigate to: GET /api/v2/insight/grouped-inbox")
    sys.exit(1)

# Test the endpoint
print("\n" + "="*70)
print("CALLING GROUPED INBOX ENDPOINT")
print("="*70)

try:
    response = session.get(f"{BASE_URL}/api/v2/insight/grouped-inbox", timeout=10)
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        print(f"\n✅ ENDPOINT WORKS!")
        print(f"✅ Found {len(data)} groups\n")
        
        for i, group in enumerate(data[:3], 1):  # Show first 3 groups
            print(f"Group {i}:")
            print(f"  Section: {group.get('section_name')}")
            print(f"  Supervisor: {group.get('supervisor_name')}")
            print(f"  Pending: {group.get('pending_count')} subcases")
            print(f"  Org Unit ID: {group.get('section_id')}")
            
            if group.get('subcases'):
                print(f"  Sample subcase:")
                sc = group['subcases'][0]
                print(f"    - ID: {sc.get('subcase_id')}")
                print(f"    - Type: {sc.get('case_type')}")
                print(f"    - Status: {sc.get('status')}")
                print(f"    - Waiting: {sc.get('waiting_days')} days")
                if sc.get('severity'):
                    print(f"    - Severity: {sc.get('severity')}")
            print()
        
        if len(data) > 3:
            print(f"... and {len(data) - 3} more groups")
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        
    else:
        print(f"\n❌ Endpoint failed: {response.status_code}")
        print(f"Response: {response.text[:500]}")
        
except Exception as e:
    print(f"\n❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
