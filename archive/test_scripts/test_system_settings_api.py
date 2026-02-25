"""
Test System Settings API endpoints.
"""

import requests
import json

BASE_URL = "http://localhost:8000"

print("=" * 70)
print("SYSTEM SETTINGS API TEST")
print("=" * 70)

# Test 1: Get all settings
print("\n1️⃣  GET ALL SETTINGS")
print("-" * 70)
try:
    response = requests.get(f"{BASE_URL}/api/settings/system-settings")
    print(f"Status: {response.status_code}")
    if response.ok:
        data = response.json()
        print(f"Total Settings: {data['total']}")
        print("\nSettings:")
        for setting in data['settings']:
            print(f"  • {setting['setting_key']:30} = {setting['setting_value']:15} ({setting['setting_type']})")
            print(f"    {setting['label']} / {setting['label_ar']}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: Get specific setting
print("\n2️⃣  GET SPECIFIC SETTING")
print("-" * 70)
try:
    response = requests.get(f"{BASE_URL}/api/settings/system-settings/max_file_size_mb")
    print(f"Status: {response.status_code}")
    if response.ok:
        setting = response.json()
        print(f"Key: {setting['setting_key']}")
        print(f"Value: {setting['setting_value']}")
        print(f"Label: {setting['label']} / {setting['label_ar']}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 3: Update setting
print("\n3️⃣  UPDATE SETTING")
print("-" * 70)
try:
    update_data = {
        "setting_value": "25"
    }
    response = requests.put(
        f"{BASE_URL}/api/settings/system-settings/max_file_size_mb",
        json=update_data
    )
    print(f"Status: {response.status_code}")
    if response.ok:
        result = response.json()
        print(f"✅ {result['message']}")
        print(f"New Value: {result['setting']['setting_value']}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 4: Create new setting
print("\n4️⃣  CREATE NEW SETTING")
print("-" * 70)
try:
    new_setting = {
        "setting_key": "my_custom_parameter",
        "setting_value": "test_value",
        "label": "My Custom Parameter",
        "label_ar": "المتغير المخصص",
        "setting_type": "text",
        "description": "This is a custom parameter for testing",
        "description_ar": "هذا متغير مخصص للاختبار"
    }
    response = requests.post(
        f"{BASE_URL}/api/settings/system-settings",
        json=new_setting
    )
    print(f"Status: {response.status_code}")
    if response.ok:
        result = response.json()
        print(f"✅ {result['message']}")
        print(f"Created Setting ID: {result['setting']['id']}")
        print(f"Key: {result['setting']['setting_key']}")
        print(f"Value: {result['setting']['setting_value']}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 5: Verify new setting appears in list
print("\n5️⃣  VERIFY NEW SETTING IN LIST")
print("-" * 70)
try:
    response = requests.get(f"{BASE_URL}/api/settings/system-settings")
    if response.ok:
        data = response.json()
        print(f"Total Settings: {data['total']}")
        custom = [s for s in data['settings'] if s['setting_key'] == 'my_custom_parameter']
        if custom:
            print(f"✅ Custom parameter found!")
            print(f"   Value: {custom[0]['setting_value']}")
        else:
            print("❌ Custom parameter not found")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
