"""
Test the pending seasonal endpoint
"""
import requests
import json

url = "http://localhost:8000/api/explanations/pending/seasonal"

print("\n" + "="*100)
print("Testing GET /api/explanations/pending/seasonal")
print("="*100 + "\n")

try:
    response = requests.get(url)
    print(f"Status Code: {response.status_code}\n")
    
    if response.status_code == 200:
        data = response.json()
        print("✅ SUCCESS - Response received")
        print("\nResponse Structure:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(f"❌ ERROR - Status {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
