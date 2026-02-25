"""
Quick Test: Explanation Endpoints with Action Items
Direct database verification after API calls.
"""

import requests
import json
import pyodbc
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000"

def get_db_connection():
    return pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=SOCIALMEDIA;"
        "DATABASE=IncidentManager;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )

def test_red_flag_with_actions():
    """Test RED FLAG explanation with 2 action items"""
    print("\n" + "="*60)
    print("TEST 1: Red Flag Explanation with Action Items")
    print("="*60)
    
    case_id = 92
    payload = {
        "explanation_text": "Root cause analysis completed. Staff training needed urgently.",
        "causes_staff": {
            "training": True,
            "competency": True,
            "understaffed": False,
            "non_compliance": False,
            "no_coordination": False,
            "other": False,
            "incentives": False
        },
        "causes_process": {"not_comprehensive": False, "unclear": True, "missing_protocol": False, "other": False},
        "causes_equipment": {"not_available": False, "system_incomplete": False, "hard_to_apply": False, "other": False},
        "causes_environment": {"place_nature": False, "surroundings": False, "work_conditions": False, "other": False},
        "preventive_actions": {"monthly_meetings": True, "training_programs": True, "increase_staff": False, "mm_committee_actions": False, "other": False},
        "action_items": [
            {
                "action_title": "Emergency Protocol Training",
                "action_description": "2-day training for ER staff",
                "due_date": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
            },
            {
                "action_title": "Update Equipment Checklist",
                "action_description": "Verify all equipment availability",
                "due_date": (datetime.now() + timedelta(days=15)).strftime("%Y-%m-%d")
            }
        ],
        "user_id": 1
    }
    
    try:
        url = f"{BASE_URL}/api/explanations/red-flag/{case_id}"
        print(f"\nPOST {url}")
        response = requests.post(url, json=payload, timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result.get('message')}")
            print(f"   Action items created: {result.get('action_items_count')}")
            
            # Verify in database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM dbo.APP_ActionItem WHERE IncidentRequestCaseID = ?", (case_id,))
            count = cursor.fetchone()[0]
            print(f"   Database verification: {count} action items found for case {case_id}")
            conn.close()
            
            return result.get('action_items_created', [])
        else:
            print(f"❌ Failed: {response.text}")
            return []
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return []

def test_ordinary_with_actions():
    """Test ORDINARY explanation with 1 action item"""
    print("\n" + "="*60)
    print("TEST 2: Ordinary Case Explanation with Action Items")
    print("="*60)
    
    case_id = 94
    payload = {
        "explanation_text": "Communication issue between departments resolved. Will improve coordination.",
        "action_items": [
            {
                "action_title": "Improve Interdepartmental Communication",
                "action_description": "Daily handoff meetings",
                "due_date": (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d")
            }
        ],
        "user_id": 1
    }
    
    try:
        url = f"{BASE_URL}/api/explanations/ordinary/{case_id}"
        print(f"\nPOST {url}")
        response = requests.post(url, json=payload, timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result.get('message')}")
            print(f"   Action items created: {result.get('action_items_count')}")
            
            # Verify in database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM dbo.APP_ActionItem WHERE IncidentRequestCaseID = ?", (case_id,))
            count = cursor.fetchone()[0]
            print(f"   Database verification: {count} action items found for case {case_id}")
            conn.close()
            
            return result.get('action_items_created', [])
        else:
            print(f"❌ Failed: {response.text}")
            return []
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return []

def test_seasonal_with_actions():
    """Test SEASONAL explanation with 3 action items"""
    print("\n" + "="*60)
    print("TEST 3: Seasonal Report Explanation with Action Items")
    print("="*60)
    
    report_id = 5
    payload = {
        "explanation_text": "Elevated violations due to staff shortage in Q1 2026. 25% staff turnover impacted case management. Corrective actions planned.",
        "action_items": [
            {
                "action_title": "Hire 5 Additional Case Managers",
                "action_description": "Recruitment drive to address shortage",
                "due_date": (datetime.now() + timedelta(days=60)).strftime("%Y-%m-%d")
            },
            {
                "action_title": "Implement Staff Retention Program",
                "action_description": "Develop incentives and career plans",
                "due_date": (datetime.now() + timedelta(days=45)).strftime("%Y-%m-%d")
            },
            {
                "action_title": "Review Workload Distribution Policy",
                "action_description": "Fair case assignment across departments",
                "due_date": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
            }
        ],
        "user_id": 1
    }
    
    try:
        url = f"{BASE_URL}/api/explanations/seasonal/{report_id}"
        print(f"\nPOST {url}")
        response = requests.post(url, json=payload, timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result.get('message')}")
            print(f"   Action items created: {result.get('action_items_count')}")
            
            # Verify in database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM dbo.APP_ActionItem WHERE SeasonalReportID = ?", (report_id,))
            count = cursor.fetchone()[0]
            print(f"   Database verification: {count} action items found for report {report_id}")
            conn.close()
            
            return result.get('action_items_created', [])
        else:
            print(f"❌ Failed: {response.text}")
            return []
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return []

def verify_in_followup():
    """Verify all action items appear in follow-up API"""
    print("\n" + "="*60)
    print("TEST 4: Verify Action Items in Follow-Up API")
    print("="*60)
    
    try:
        url = f"{BASE_URL}/api/follow-up/actions?include_completed=false"
        print(f"\nGET {url}")
        response = requests.get(url, timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            total = result.get('total', 0)
            print(f"✅ Follow-up API accessible")
            print(f"   Total pending actions: {total}")
            
            # Show recent actions
            actions = result.get('actions', [])
            if actions:
                print(f"\n   Recent actions (first 5):")
                for action in actions[:5]:
                    print(f"   - ID {action['id']}: {action['actionTitle'][:50]}")
                    print(f"     Source: {action['sourceType']}, Due: {action['dueDate']}")
            
            return True
        else:
            print(f"❌ Failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    print("\n" + "="*70)
    print(" EXPLANATION WITH ACTION ITEMS - QUICK TEST SUITE")
    print("="*70)
    print(f" Base URL: {BASE_URL}")
    print(f" Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    red_flag_items = test_red_flag_with_actions()
    ordinary_items = test_ordinary_with_actions()
    seasonal_items = test_seasonal_with_actions()
    verify_in_followup()
    
    # Summary
    total = len(red_flag_items) + len(ordinary_items) + len(seasonal_items)
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    print(f" Total action items created: {total}")
    print(f"   - Red Flag: {len(red_flag_items)}")
    print(f"   - Ordinary: {len(ordinary_items)}")
    print(f"   - Seasonal: {len(seasonal_items)}")
    
    if total > 0:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n⚠️ No action items created - check for errors above")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
