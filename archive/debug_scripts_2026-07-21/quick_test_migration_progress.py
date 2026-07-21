"""
QUICK API TEST - Migration Progress Endpoint

This script tests the migration progress endpoint through the actual API
to verify it's working end-to-end.

RUN:
    python quick_test_migration_progress.py
"""

import sys
from pathlib import Path

backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from fastapi.testclient import TestClient
from main import app
from api.dependencies.user_context import get_current_user
from api.schemas.auth_models import CurrentUser, UserScope
from core.constants.roles import SOFTWARE_ADMIN, WORKER

client = TestClient(app)


def create_mock_user(role: str) -> CurrentUser:
    """Create a mock user with specified role."""
    return CurrentUser(
        user_id=1,
        username="test_user",
        is_active=True,
        scopes=[
            UserScope(
                role_code=role,
                org_unit_id=1,
                org_unit_type="HOSPITAL"
            )
        ],
        allowed_unit_ids={1},
        roles=[role],
        primary_unit_id=1,
        primary_unit_type="HOSPITAL"
    )


def test_endpoint():
    """Quick test of the migration progress endpoint"""
    print("\n" + "=" * 80)
    print("  MIGRATION PROGRESS ENDPOINT - QUICK TEST")
    print("=" * 80)
    
    # Override auth with SOFTWARE_ADMIN
    mock_user = create_mock_user(SOFTWARE_ADMIN)
    app.dependency_overrides[get_current_user] = lambda: mock_user
    
    try:
        # Call the endpoint
        print("\n📡 Calling GET /api/migration/progress...")
        response = client.get("/api/migration/progress")
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            print("\n✅ SUCCESS - Response received:")
            print(f"   total_legacy: {data.get('total_legacy')}")
            print(f"   migrated_total: {data.get('migrated_total')}")
            print(f"   percent: {data.get('percent')}")
            
            # Verify response structure
            has_all_fields = all(key in data for key in ['total_legacy', 'migrated_total', 'percent'])
            
            if has_all_fields:
                print("\n✅ All required fields present")
                
                # Check data types
                types_correct = (
                    isinstance(data['total_legacy'], int) and
                    isinstance(data['migrated_total'], int) and
                    isinstance(data['percent'], (int, float))
                )
                
                if types_correct:
                    print("✅ Data types correct")
                    
                    # Check percent precision
                    percent_str = str(data['percent'])
                    decimal_places = 0
                    if '.' in percent_str:
                        decimal_places = len(percent_str.split('.')[1])
                    
                    if decimal_places <= 1:
                        print(f"✅ Percent precision correct (1 decimal place)")
                    else:
                        print(f"⚠️  Percent has {decimal_places} decimal places (expected 1)")
                    
                    print("\n" + "=" * 80)
                    print("  🎉 ENDPOINT TEST PASSED")
                    print("=" * 80)
                    return True
                else:
                    print("❌ Data types incorrect")
            else:
                print("❌ Missing required fields")
                print(f"   Received fields: {list(data.keys())}")
        else:
            print(f"❌ FAILED - Status: {response.status_code}")
            print(f"   Response: {response.json()}")
        
        return False
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        app.dependency_overrides.clear()


if __name__ == "__main__":
    try:
        success = test_endpoint()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
