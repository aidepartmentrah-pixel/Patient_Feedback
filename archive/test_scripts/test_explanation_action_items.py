"""
Test Script: Explanation Endpoints with Action Items
Tests Red Flag, Ordinary, and Seasonal explanation submission with action item creation.
"""

import requests
import json
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000"
USER_ID = 1

def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def print_result(response, description):
    print(f"\n{description}")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return data
    else:
        print(f"Error: {response.text}")
        return None

# =============================================================================
# TEST 1: RED FLAG EXPLANATION WITH ACTION ITEMS
# =============================================================================
def test_red_flag_explanation_with_actions():
    print_section("TEST 1: Red Flag Explanation with Action Items")
    
    case_id = 92  # Red Flag case from your database
    
    payload = {
        "explanation_text": "تحليل شامل للسبب الجذري: حدث الحادث نتيجة نقص في التدريب وعدم وضوح الإجراءات. سيتم اتخاذ إجراءات وقائية فورية.",
        "causes_staff": {
            "training": True,
            "incentives": False,
            "competency": True,
            "understaffed": False,
            "non_compliance": False,
            "no_coordination": True,
            "other": False,
            "other_text": None
        },
        "causes_process": {
            "not_comprehensive": False,
            "unclear": True,
            "missing_protocol": False,
            "other": False,
            "other_text": None
        },
        "causes_equipment": {
            "not_available": False,
            "system_incomplete": False,
            "hard_to_apply": False,
            "other": False,
            "other_text": None
        },
        "causes_environment": {
            "place_nature": False,
            "surroundings": False,
            "work_conditions": False,
            "other": False,
            "other_text": None
        },
        "preventive_actions": {
            "monthly_meetings": True,
            "training_programs": True,
            "increase_staff": False,
            "mm_committee_actions": True,
            "other": False,
            "other_text": None
        },
        "action_items": [
            {
                "action_title": "تدريب الطاقم على بروتوكول الطوارئ",
                "action_description": "تنفيذ دورة تدريبية مدتها يومين لجميع طاقم الطوارئ",
                "due_date": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
            },
            {
                "action_title": "مراجعة قائمة المعدات وتحديثها",
                "action_description": "التأكد من توفر جميع المعدات وصلاحيتها",
                "due_date": (datetime.now() + timedelta(days=15)).strftime("%Y-%m-%d")
            },
            {
                "action_title": "تحديث دليل الإجراءات",
                "action_description": "توضيح الخطوات في حالات الطوارئ الحرجة",
                "due_date": (datetime.now() + timedelta(days=20)).strftime("%Y-%m-%d")
            }
        ],
        "user_id": USER_ID
    }
    
    url = f"{BASE_URL}/api/explanations/red-flag/{case_id}"
    response = requests.post(url, json=payload)
    result = print_result(response, f"POST {url}")
    
    if result and result.get("success"):
        print(f"\n✅ Red Flag explanation submitted successfully!")
        print(f"   Action items created: {result.get('action_items_count')}")
        for item in result.get('action_items_created', []):
            print(f"   - ID {item['action_item_id']}: {item['title']}")
        return result.get('action_items_created', [])
    else:
        print(f"\n❌ Red Flag explanation failed!")
        return []

# =============================================================================
# TEST 2: ORDINARY CASE EXPLANATION WITH ACTION ITEMS
# =============================================================================
def test_ordinary_explanation_with_actions():
    print_section("TEST 2: Ordinary Case Explanation with Action Items")
    
    case_id = 94  # Ordinary case requiring explanation
    
    payload = {
        "explanation_text": "تم التأخير بسبب سوء التواصل بين الأقسام. تم التعامل مع المشكلة وسيتم تحسين التنسيق.",
        "action_items": [
            {
                "action_title": "تحسين بروتوكول التواصل بين الأقسام",
                "action_description": "تطبيق اجتماعات يومية لتسليم المهام",
                "due_date": (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d")
            }
        ],
        "user_id": USER_ID
    }
    
    url = f"{BASE_URL}/api/explanations/ordinary/{case_id}"
    response = requests.post(url, json=payload)
    result = print_result(response, f"POST {url}")
    
    if result and result.get("success"):
        print(f"\n✅ Ordinary explanation submitted successfully!")
        print(f"   Action items created: {result.get('action_items_count')}")
        for item in result.get('action_items_created', []):
            print(f"   - ID {item['action_item_id']}: {item['title']}")
        return result.get('action_items_created', [])
    else:
        print(f"\n❌ Ordinary explanation failed!")
        return []

# =============================================================================
# TEST 3: SEASONAL REPORT EXPLANATION WITH ACTION ITEMS
# =============================================================================
def test_seasonal_explanation_with_actions():
    print_section("TEST 3: Seasonal Report Explanation with Action Items")
    
    report_id = 5  # Non-compliant seasonal report
    
    payload = {
        "explanation_text": "ارتفاع معدل انتهاكات المجال الإداري نتج عن نقص الموظفين خلال الربع الأول من 2026. شهدنا دوران موظفين بنسبة 25٪ مما أثر على قدرتنا على إدارة الحالات بفعالية. سيتم اتخاذ إجراءات تصحيحية شاملة.",
        "action_items": [
            {
                "action_title": "توظيف 5 مدراء حالات إضافيين",
                "action_description": "حملة توظيف لمعالجة نقص الموظفين",
                "due_date": (datetime.now() + timedelta(days=60)).strftime("%Y-%m-%d")
            },
            {
                "action_title": "تطبيق برنامج الاحتفاظ بالموظفين",
                "action_description": "تطوير حوافز وخطط التطوير الوظيفي",
                "due_date": (datetime.now() + timedelta(days=45)).strftime("%Y-%m-%d")
            },
            {
                "action_title": "مراجعة سياسة توزيع أعباء العمل",
                "action_description": "ضمان توزيع عادل للحالات عبر الأقسام",
                "due_date": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
            }
        ],
        "user_id": USER_ID
    }
    
    url = f"{BASE_URL}/api/explanations/seasonal/{report_id}"
    response = requests.post(url, json=payload)
    result = print_result(response, f"POST {url}")
    
    if result and result.get("success"):
        print(f"\n✅ Seasonal explanation submitted successfully!")
        print(f"   Action items created: {result.get('action_items_count')}")
        for item in result.get('action_items_created', []):
            print(f"   - ID {item['action_item_id']}: {item['title']}")
        return result.get('action_items_created', [])
    else:
        print(f"\n❌ Seasonal explanation failed!")
        return []

# =============================================================================
# TEST 4: VERIFY ACTION ITEMS IN FOLLOW-UP PAGE
# =============================================================================
def test_verify_action_items_in_followup(action_item_ids):
    print_section("TEST 4: Verify Action Items Appear in Follow-Up Page")
    
    url = f"{BASE_URL}/api/follow-up/actions?include_completed=false"
    response = requests.get(url)
    result = print_result(response, f"GET {url}")
    
    if result and result.get("actions"):
        print(f"\n✅ Follow-up page accessible!")
        print(f"   Total actions: {result.get('total')}")
        print(f"   Actions returned: {len(result.get('actions'))}")
        
        # Find our created action items
        created_ids = [item['action_item_id'] for item in action_item_ids]
        found_items = [a for a in result['actions'] if a['id'] in created_ids]
        
        print(f"\n   Looking for our created action items ({len(created_ids)} items):")
        for action in found_items:
            print(f"   ✅ Found ID {action['id']}: {action['actionTitle']}")
            print(f"      Source: {action['sourceType']}, Status: {action['status']}, Due: {action['dueDate']}")
        
        if len(found_items) != len(created_ids):
            print(f"\n   ⚠️ Warning: Expected {len(created_ids)} items, found {len(found_items)}")
        
        return found_items
    else:
        print(f"\n❌ Failed to retrieve follow-up actions!")
        return []

# =============================================================================
# TEST 5: CALENDAR VIEW TEST
# =============================================================================
def test_calendar_view():
    print_section("TEST 5: Calendar View")
    
    now = datetime.now()
    url = f"{BASE_URL}/api/follow-up/calendar?year={now.year}&month={now.month}&status=all"
    response = requests.get(url)
    result = print_result(response, f"GET {url}")
    
    if result and result.get("calendar"):
        print(f"\n✅ Calendar view accessible!")
        print(f"   Year: {result.get('year')}, Month: {result.get('month')}")
        
        calendar = result.get('calendar', {})
        print(f"   Days with actions: {len(calendar)}")
        
        # Show first 3 days with actions
        for i, (date, actions) in enumerate(list(calendar.items())[:3]):
            print(f"\n   📅 {date}: {len(actions)} actions")
            for action in actions[:2]:  # Show first 2 actions per day
                print(f"      - [{action['priority']}] {action['actionTitle']}")
                print(f"        Source: {action['sourceType']}, Overdue: {action['isOverdue']}")
        
        return True
    else:
        print(f"\n❌ Calendar view failed!")
        return False

# =============================================================================
# TEST 6: COMPLETE AN ACTION ITEM
# =============================================================================
def test_complete_action(action_item_id):
    print_section(f"TEST 6: Complete Action Item #{action_item_id}")
    
    payload = {
        "completionNotes": "تم إنجاز المهمة بنجاح في الوقت المحدد",
        "completedDate": datetime.now().strftime("%Y-%m-%d")
    }
    
    url = f"{BASE_URL}/api/follow-up/actions/{action_item_id}/complete"
    response = requests.post(url, json=payload)
    result = print_result(response, f"POST {url}")
    
    if result and result.get("status") == "completed":
        print(f"\n✅ Action item completed successfully!")
        print(f"   Status: {result.get('status')}")
        print(f"   Completed Date: {result.get('completedDate')}")
        return True
    else:
        print(f"\n❌ Failed to complete action item!")
        return False

# =============================================================================
# TEST 7: DELAY AN ACTION ITEM
# =============================================================================
def test_delay_action(action_item_id):
    print_section(f"TEST 7: Delay Action Item #{action_item_id}")
    
    payload = {
        "delayDays": 7,
        "reason": "تأجيل لمدة أسبوع لانتظار موافقة الإدارة"
    }
    
    url = f"{BASE_URL}/api/follow-up/actions/{action_item_id}/delay"
    response = requests.post(url, json=payload)
    result = print_result(response, f"POST {url}")
    
    if result:
        print(f"\n✅ Action item delayed successfully!")
        print(f"   New Due Date: {result.get('dueDate')}")
        print(f"   Status: {result.get('status')}")
        return True
    else:
        print(f"\n❌ Failed to delay action item!")
        return False

# =============================================================================
# MAIN TEST RUNNER
# =============================================================================
def main():
    print("\n" + "="*80)
    print("  EXPLANATION WITH ACTION ITEMS - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print(f"  Base URL: {BASE_URL}")
    print(f"  Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    all_action_items = []
    
    # Test 1: Red Flag with actions
    red_flag_items = test_red_flag_explanation_with_actions()
    all_action_items.extend(red_flag_items)
    
    # Test 2: Ordinary with actions
    ordinary_items = test_ordinary_explanation_with_actions()
    all_action_items.extend(ordinary_items)
    
    # Test 3: Seasonal with actions
    seasonal_items = test_seasonal_explanation_with_actions()
    all_action_items.extend(seasonal_items)
    
    # Test 4: Verify in follow-up page
    if all_action_items:
        found_items = test_verify_action_items_in_followup(all_action_items)
        
        # Test 5: Calendar view
        test_calendar_view()
        
        # Test 6: Complete first action
        if found_items and len(found_items) > 0:
            test_complete_action(found_items[0]['id'])
        
        # Test 7: Delay second action
        if found_items and len(found_items) > 1:
            test_delay_action(found_items[1]['id'])
    
    # Final Summary
    print_section("TEST SUMMARY")
    print(f"Total action items created: {len(all_action_items)}")
    print(f"  - From Red Flag: {len(red_flag_items)}")
    print(f"  - From Ordinary: {len(ordinary_items)}")
    print(f"  - From Seasonal: {len(seasonal_items)}")
    
    if len(all_action_items) > 0:
        print("\n✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("\nAction Items Created:")
        for item in all_action_items:
            print(f"  - ID {item['action_item_id']}: {item['title']}")
    else:
        print("\n⚠️ No action items were created. Check for errors above.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
