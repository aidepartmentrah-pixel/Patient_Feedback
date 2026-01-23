"""
Phase 3 API Integration Test
Tests the GET /api/settings/training/grouped-status endpoint
"""

import requests
import json
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"
API_PREFIX = "/api/settings/training"

print("=" * 80)
print("PHASE 3: API INTEGRATION TEST - Model Grouping & Aggregation")
print("=" * 80)
print()

def test_grouped_status_endpoint():
    """Test the grouped status endpoint."""
    print("[TEST 1] GET /api/settings/training/grouped-status")
    print("-" * 80)
    
    try:
        url = f"{BASE_URL}{API_PREFIX}/grouped-status"
        response = requests.get(url, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ FAILED: Expected status 200, got {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        data = response.json()
        
        # Validate structure
        required_fields = ["last_run", "status", "model_families", "alerts", "summary"]
        for field in required_fields:
            if field not in data:
                print(f"❌ FAILED: Missing field '{field}' in response")
                return False
        
        print(f"✓ Response has all required fields")
        
        # Check model families
        families = data["model_families"]
        print(f"✓ Found {len(families)} model families")
        
        if len(families) > 0:
            first_family = families[0]
            print(f"  - First family: {first_family['family_name']} ({first_family['model_count']} models)")
            print(f"    Avg F1: {first_family['avg_f1']:.4f}")
            print(f"    Total Records: {first_family['total_records']}")
        
        # Check alerts
        alerts = data["alerts"]
        print(f"✓ Found {len(alerts)} alerts")
        
        if len(alerts) > 0:
            critical_alerts = [a for a in alerts if a["severity"] == "critical"]
            warning_alerts = [a for a in alerts if a["severity"] == "warning"]
            info_alerts = [a for a in alerts if a["severity"] == "info"]
            
            print(f"  - Critical: {len(critical_alerts)}")
            print(f"  - Warning: {len(warning_alerts)}")
            print(f"  - Info: {len(info_alerts)}")
            
            if len(alerts) > 0:
                print(f"  - First alert: {alerts[0]['severity']} - {alerts[0]['message']}")
        
        # Check summary
        summary = data["summary"]
        print(f"✓ Summary statistics:")
        print(f"  - Total Models: {summary['total_models']}")
        print(f"  - Total Families: {summary['total_families']}")
        print(f"  - Overall Avg F1: {summary['overall_avg_f1']:.4f}")
        print(f"  - Critical Alerts: {summary['critical_alerts']}")
        print(f"  - Warning Alerts: {summary['warning_alerts']}")
        
        # Validate bilingual support
        if len(families) > 0:
            first_family = families[0]
            if "family_name_ar" in first_family:
                print(f"✓ Bilingual support confirmed (Arabic names present)")
        
        if len(alerts) > 0:
            first_alert = alerts[0]
            if "message_ar" in first_alert and "recommendation_ar" in first_alert:
                print(f"✓ Bilingual support confirmed (Arabic messages present)")
        
        print()
        print("✅ PASSED: Grouped status endpoint works correctly")
        print()
        
        # Pretty print the full response for review
        print("Full Response:")
        print("-" * 80)
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ FAILED: Could not connect to server")
        print("   Make sure the FastAPI server is running on http://localhost:8000")
        return False
    except Exception as e:
        print(f"❌ FAILED: Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Prerequisites:")
    print("  1. FastAPI server must be running on port 8000")
    print("  2. Training must have been run at least once")
    print()
    
    success = test_grouped_status_endpoint()
    
    print()
    print("=" * 80)
    if success:
        print("🎉 ALL API INTEGRATION TESTS PASSED!")
    else:
        print("⚠️ Some tests failed. Please review the output above.")
    print("=" * 80)
