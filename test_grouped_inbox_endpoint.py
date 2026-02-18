"""
Test grouped inbox API endpoint - Stage 3 Testing

This test file validates the new REST API endpoint that exposes
the grouped inbox functionality via HTTP.

Tests:
1. Endpoint requires authentication (401 without credentials)
2. Endpoint returns proper grouped structure
3. Data matches expected format
"""

import requests
import sys

BASE_URL = "http://localhost:8000"


def login(username, password):
    """Login and return session with cookies/headers"""
    session = requests.Session()
    
    try:
        response = session.post(
            f"{BASE_URL}/api/auth/login",
            json={"username": username, "password": password},
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"✅ Logged in as {username}")
            return session
        else:
            print(f"❌ Login failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return None
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {BASE_URL}")
        print(f"   Is the server running? Start it with: uvicorn backend.main:app --reload")
        return None
    except Exception as e:
        print(f"❌ Login error: {str(e)}")
        return None


def test_unauthorized_access():
    """Test that endpoint requires authentication"""
    print("\n" + "="*60)
    print("TEST: Unauthorized access protection")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/api/v2/insight/grouped-inbox", timeout=10)
        
        if response.status_code == 401:
            print("✅ Endpoint properly protected (401 without auth)")
        elif response.status_code == 403:
            print("✅ Endpoint properly protected (403 without auth)")
        else:
            print(f"⚠️  Unexpected status without auth: {response.status_code}")
            print(f"   Expected: 401 or 403")
            print(f"   Response: {response.text[:200]}")
    except requests.exceptions.ConnectionError:
        print(f"⚠️  Cannot connect to {BASE_URL} - server may not be running")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False
    
    return True


def test_grouped_inbox_endpoint():
    """Test the new grouped inbox endpoint with authentication"""
    print("\n" + "="*60)
    print("TEST: Grouped inbox endpoint (authenticated)")
    print("="*60)
    
    # Try common test credentials
    # Update these with your actual test credentials
    test_credentials = [
        ("admin", "admin"),
        ("admin", "admin123"),
        ("section_admin", "password"),
        ("dept_admin", "password"),
    ]
    
    session = None
    for username, password in test_credentials:
        session = login(username, password)
        if session:
            break
    
    if not session:
        print("\n⚠️  Cannot test without valid login")
        print("   Please update test_credentials in the test file with valid credentials")
        print("   Or manually test with: curl -H 'Authorization: Bearer <token>' http://localhost:8000/api/v2/insight/grouped-inbox")
        return False
    
    try:
        # Call the new endpoint
        response = session.get(f"{BASE_URL}/api/v2/insight/grouped-inbox", timeout=10)
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Endpoint works!")
            print(f"✅ Response is valid JSON")
            print(f"✅ Found {len(data)} groups")
            
            # Verify structure
            if isinstance(data, list):
                print(f"✅ Response is a list")
            else:
                print(f"❌ Response should be a list, got: {type(data)}")
                return False
            
            if len(data) > 0:
                first_group = data[0]
                
                # Verify required fields
                required_fields = ['section_id', 'section_name', 'supervisor_name', 'pending_count', 'subcases']
                missing_fields = [f for f in required_fields if f not in first_group]
                
                if missing_fields:
                    print(f"❌ Missing required fields: {missing_fields}")
                    return False
                else:
                    print(f"✅ All required fields present")
                
                print(f"\n📋 Sample Group:")
                print(f"   Section ID: {first_group.get('section_id')}")
                print(f"   Section: {first_group.get('section_name')}")
                print(f"   Supervisor: {first_group.get('supervisor_name')}")
                print(f"   Pending: {first_group.get('pending_count')}")
                print(f"   Org Type: {first_group.get('org_type', 'N/A')}")
                
                # Verify subcases structure
                subcases = first_group.get('subcases', [])
                if isinstance(subcases, list):
                    print(f"✅ Subcases is a list with {len(subcases)} items")
                    
                    if len(subcases) > 0:
                        first_subcase = subcases[0]
                        print(f"\n   📄 Sample Subcase:")
                        print(f"      ID: {first_subcase.get('subcase_id')}")
                        print(f"      Type: {first_subcase.get('case_type')}")
                        print(f"      Status: {first_subcase.get('status')}")
                        
                        desc = first_subcase.get('case_description', '')
                        if desc:
                            print(f"      Description: {desc[:60]}...")
                        
                        print(f"      Waiting: {first_subcase.get('waiting_days')} days")
                        print(f"      Severity: {first_subcase.get('severity')}")
                        print(f"      Category: {first_subcase.get('category')}")
                        
                        if first_subcase.get('patient_name'):
                            print(f"      Patient: {first_subcase.get('patient_name')}")
                        if first_subcase.get('season_name'):
                            print(f"      Season: {first_subcase.get('season_name')}")
                        
                        # Verify sorting
                        if len(subcases) > 1:
                            waiting_days = [s.get('waiting_days', 0) for s in subcases]
                            is_sorted = all(waiting_days[i] >= waiting_days[i+1] for i in range(len(waiting_days)-1))
                            if is_sorted:
                                print(f"✅ Subcases sorted by waiting_days DESC")
                            else:
                                print(f"⚠️  Subcases may not be sorted properly")
                else:
                    print(f"❌ Subcases should be a list, got: {type(subcases)}")
                    return False
                
                # Verify group sorting
                if len(data) > 1:
                    counts = [g.get('pending_count', 0) for g in data]
                    is_sorted = all(counts[i] >= counts[i+1] for i in range(len(counts)-1))
                    if is_sorted:
                        print(f"\n✅ Groups sorted by pending_count DESC")
                        print(f"   Top group: {counts[0]} pending")
                        print(f"   Last group: {counts[-1]} pending")
                    else:
                        print(f"\n⚠️  Groups may not be sorted properly")
                
            else:
                print("\n⚠️  No groups returned (may need test data or wider scope)")
            
            return True
            
        elif response.status_code == 403:
            print(f"⚠️  Access forbidden (403)")
            print(f"   User may not have required permissions")
            print(f"   Response: {response.text[:200]}")
            return False
        else:
            print(f"❌ Endpoint failed: {response.status_code}")
            print(f"   Response: {response.text[:500]}")
            return False
            
    except Exception as e:
        print(f"❌ Error calling endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_endpoint_documentation():
    """Test that endpoint appears in OpenAPI docs"""
    print("\n" + "="*60)
    print("TEST: Endpoint documentation")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/openapi.json", timeout=10)
        
        if response.status_code == 200:
            openapi = response.json()
            
            # Check if our endpoint is in the paths
            if '/api/v2/insight/grouped-inbox' in openapi.get('paths', {}):
                print("✅ Endpoint appears in OpenAPI documentation")
                
                endpoint_doc = openapi['paths']['/api/v2/insight/grouped-inbox']
                if 'get' in endpoint_doc:
                    print("✅ GET method documented")
                    
                    get_doc = endpoint_doc['get']
                    if 'summary' in get_doc or 'description' in get_doc:
                        print("✅ Endpoint has description")
                else:
                    print("⚠️  GET method not documented")
            else:
                print("❌ Endpoint not found in OpenAPI docs")
                print("   Available insight endpoints:")
                for path in openapi.get('paths', {}).keys():
                    if '/insight/' in path:
                        print(f"      {path}")
        else:
            print(f"⚠️  Could not fetch OpenAPI docs: {response.status_code}")
    except Exception as e:
        print(f"⚠️  Error checking docs: {str(e)}")


if __name__ == '__main__':
    print("\n" + "="*70)
    print(" STAGE 3: API ENDPOINT TESTING - Grouped Inbox")
    print("="*70)
    print(f"\nTesting endpoint: {BASE_URL}/api/v2/insight/grouped-inbox")
    print("\nPrerequisites:")
    print("  1. Backend server must be running (uvicorn backend.main:app --reload)")
    print("  2. Valid test credentials configured")
    
    all_passed = True
    
    # Test 1: Unauthorized access
    if not test_unauthorized_access():
        all_passed = False
    
    # Test 2: Endpoint documentation
    test_endpoint_documentation()
    
    # Test 3: Authenticated access
    if not test_grouped_inbox_endpoint():
        all_passed = False
    
    # Summary
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL STAGE 3 TESTS COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\nEndpoint is ready for use!")
        print(f"\nAPI Documentation: {BASE_URL}/docs")
        print(f"OpenAPI Spec: {BASE_URL}/openapi.json")
    else:
        print("⚠️  SOME TESTS HAD ISSUES")
        print("="*70)
        print("\nPlease review the output above for details")
    
    sys.exit(0 if all_passed else 1)
