"""
MANUAL API TEST - Migration Progress Endpoint

Use this to manually test the endpoint with curl or browser.

INSTRUCTIONS:
1. Start the backend server:
   cd backend
   uvicorn main:app --reload

2. Get an authentication token (login as SOFTWARE_ADMIN or WORKER)

3. Test the endpoint using one of the methods below
"""

# ============================================================
# METHOD 1: PowerShell (Windows)
# ============================================================

# Replace YOUR_TOKEN_HERE with actual token
$token = "YOUR_TOKEN_HERE"

$headers = @{
    "Authorization" = "Bearer $token"
}

$response = Invoke-RestMethod -Uri "http://localhost:8000/api/migration/progress" -Headers $headers -Method GET

Write-Host "Response:"
Write-Host "  total_legacy: $($response.total_legacy)"
Write-Host "  migrated_total: $($response.migrated_total)"
Write-Host "  percent: $($response.percent)"


# ============================================================
# METHOD 2: curl (Cross-platform)
# ============================================================

# Replace YOUR_TOKEN_HERE with actual token
curl -X GET "http://localhost:8000/api/migration/progress" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "Content-Type: application/json"


# ============================================================
# METHOD 3: Python Requests
# ============================================================

"""
import requests

# Replace with actual token
token = "YOUR_TOKEN_HERE"

response = requests.get(
    "http://localhost:8000/api/migration/progress",
    headers={"Authorization": f"Bearer {token}"}
)

print("Status:", response.status_code)
print("Response:", response.json())

if response.status_code == 200:
    data = response.json()
    print(f"Migration Progress: {data['percent']}%")
    print(f"{data['migrated_total']} of {data['total_legacy']} cases migrated")
"""


# ============================================================
# METHOD 4: Browser Testing (using browser's developer console)
# ============================================================

"""
// 1. Login to the application
// 2. Open browser DevTools (F12)
// 3. Go to Console tab
// 4. Run this code:

fetch('/api/migration/progress', {
    headers: {
        'Authorization': 'Bearer ' + localStorage.getItem('token') // or wherever token is stored
    }
})
.then(response => response.json())
.then(data => {
    console.log('Migration Progress:', data);
    console.log(`${data.percent}% complete`);
    console.log(`${data.migrated_total} of ${data.total_legacy} cases migrated`);
});
"""


# ============================================================
# EXPECTED RESPONSE (Example)
# ============================================================

"""
{
  "total_legacy": 79,
  "migrated_total": 1,
  "percent": 1.3
}
"""


# ============================================================
# ERROR RESPONSES
# ============================================================

# 401 Unauthorized (no token or invalid token)
"""
{
  "detail": "Not authenticated"
}
"""

# 403 Forbidden (wrong role)
"""
{
  "detail": {
    "error": "FORBIDDEN",
    "message": "Access denied. Required roles: SOFTWARE_ADMIN, WORKER",
    "message_ar": "ممنوع الوصول. الأدوار المطلوبة: مسؤول البرنامج، عامل"
  }
}
"""

# 500 Internal Server Error
"""
{
  "detail": {
    "error": "PROGRESS_FAILED",
    "message": "Failed to retrieve migration progress: [error details]",
    "message_ar": "فشل في استرجاع تقدم الترحيل"
  }
}
"""
