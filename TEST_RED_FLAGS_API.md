# Red Flags API Testing Guide

This document provides comprehensive testing instructions for the **Red Flags (Critical Issues)** API endpoints.

## Table of Contents
1. [Overview](#overview)
2. [Base URL](#base-url)
3. [Endpoint Testing](#endpoint-testing)
4. [Sample Requests](#sample-requests)
5. [Expected Responses](#expected-responses)
6. [Common Error Codes](#common-error-codes)

---

## Overview

The Red Flags API provides access to high-risk incidents requiring immediate attention and governance follow-up. These are incidents where `ClinicalRiskTypeID = 2` (REDFLAG).

**Key Features:**
- Filter and search red flags
- View statistics and KPIs
- Track trends over time
- Access detailed incident information
- Cross-reference with Never Events

---

## Base URL

```
http://127.0.0.1:8000
```

**API Documentation (Swagger UI):**
```
http://127.0.0.1:8000/docs#/Red%20Flags
```

---

## Endpoint Testing

### 1. Test Endpoint (Verify Service is Running)

**Request:**
```bash
curl -X GET "http://127.0.0.1:8000/api/red-flags/test"
```

**Expected Response:**
```json
{
  "status": "operational",
  "service": "red-flags",
  "message": "Red Flags API is operational",
  "endpoints": [
    "GET /api/red-flags",
    "GET /api/red-flags/statistics",
    "GET /api/red-flags/trends",
    "GET /api/red-flags/{id}",
    "POST /api/red-flags/{id}/export-pdf (not implemented)",
    "POST /api/red-flags/export-batch (not implemented)"
  ]
}
```

---

### 2. Get Red Flags List (with Filters)

**Basic Request (All Red Flags):**
```bash
curl -X GET "http://127.0.0.1:8000/api/red-flags?limit=50&offset=0"
```

**Filter by Status:**
```bash
# Open cases only
curl -X GET "http://127.0.0.1:8000/api/red-flags?status=OPEN&limit=50"

# Finished cases
curl -X GET "http://127.0.0.1:8000/api/red-flags?status=FINISHED&limit=50"

# Under review
curl -X GET "http://127.0.0.1:8000/api/red-flags?status=UNDER_REVIEW&limit=50"
```

**Filter by Date Range:**
```bash
curl -X GET "http://127.0.0.1:8000/api/red-flags?from_date=2024-01-01&to_date=2024-12-31&limit=100"
```

**Filter by Severity:**
```bash
# Critical severity only
curl -X GET "http://127.0.0.1:8000/api/red-flags?severity=CRITICAL&limit=50"

# High severity
curl -X GET "http://127.0.0.1:8000/api/red-flags?severity=HIGH&limit=50"
```

**Search by Patient Name or Record ID:**
```bash
curl -X GET "http://127.0.0.1:8000/api/red-flags?search=RF-2024-001"
```

**Filter Red Flags that are Also Never Events:**
```bash
curl -X GET "http://127.0.0.1:8000/api/red-flags?is_never_event=true&limit=50"
```

**Combined Filters:**
```bash
curl -X GET "http://127.0.0.1:8000/api/red-flags?status=OPEN&severity=CRITICAL&from_date=2024-01-01&department=الطوارئ&limit=50"
```

**Expected Response Structure:**
```json
{
  "red_flags": [
    {
      "red_flag_id": 1,
      "recordID": "RF-2024-001",
      "patient_name": "محمد أحمد",
      "date_received": "2024-01-15",
      "department": "الطوارئ",
      "category": "Patient Safety",
      "severity": "CRITICAL",
      "status": "UNDER_REVIEW",
      "isNeverEvent": true,
      "complaint_summary": "حدث تأخير في تشخيص حالة حرجة..."
    }
  ],
  "total": 156,
  "limit": 50,
  "offset": 0
}
```

---

### 3. Get Red Flags Statistics

**Basic Request:**
```bash
curl -X GET "http://127.0.0.1:8000/api/red-flags/statistics"
```

**With Date Range:**
```bash
curl -X GET "http://127.0.0.1:8000/api/red-flags/statistics?from_date=2024-01-01&to_date=2024-12-31"
```

**Expected Response:**
```json
{
  "total_red_flags": 245,
  "unfinished": 87,
  "finished": 158,
  "by_status": {
    "OPEN": 32,
    "UNDER_REVIEW": 55,
    "FINISHED": 158
  },
  "by_category": {
    "Patient Safety": 98,
    "Medical Errors": 67,
    "Medication Issues": 45,
    "Surgical Complications": 35
  },
  "by_severity": {
    "CRITICAL": 89,
    "HIGH": 156
  },
  "current_month": {
    "count": 23,
    "month": "2024-12"
  },
  "previous_month": {
    "count": 19,
    "month": "2024-11"
  },
  "never_event_overlap": {
    "total_never_events": 45,
    "red_flags_also_never_events": 34,
    "never_events_only": 11,
    "red_flags_only": 211
  },
  "date_range": {
    "from": "2024-01-01",
    "to": "2024-12-31"
  }
}
```

---

### 4. Get Red Flags Trends

**Monthly Trends (Default):**
```bash
curl -X GET "http://127.0.0.1:8000/api/red-flags/trends?granularity=monthly"
```

**Quarterly Trends:**
```bash
curl -X GET "http://127.0.0.1:8000/api/red-flags/trends?granularity=quarterly"
```

**Weekly Trends:**
```bash
curl -X GET "http://127.0.0.1:8000/api/red-flags/trends?granularity=weekly"
```

**Group by Category:**
```bash
curl -X GET "http://127.0.0.1:8000/api/red-flags/trends?granularity=monthly&group_by=category"
```

**Group by Severity:**
```bash
curl -X GET "http://127.0.0.1:8000/api/red-flags/trends?granularity=monthly&group_by=severity"
```

**Group by Department:**
```bash
curl -X GET "http://127.0.0.1:8000/api/red-flags/trends?granularity=monthly&group_by=department"
```

**Custom Date Range:**
```bash
curl -X GET "http://127.0.0.1:8000/api/red-flags/trends?from_date=2023-01-01&to_date=2024-12-31&granularity=quarterly"
```

**Expected Response (No Grouping):**
```json
{
  "trends": [
    {
      "period": "يناير 2024",
      "count": 18
    },
    {
      "period": "فبراير 2024",
      "count": 22
    },
    {
      "period": "مارس 2024",
      "count": 19
    }
  ],
  "date_range": {
    "from": "2024-01-01",
    "to": "2024-12-31"
  },
  "granularity": "monthly"
}
```

**Expected Response (With Grouping):**
```json
{
  "trends": [
    {
      "period": "يناير 2024",
      "breakdown": {
        "Patient Safety": 8,
        "Medical Errors": 6,
        "Medication Issues": 4
      },
      "total": 18
    },
    {
      "period": "فبراير 2024",
      "breakdown": {
        "Patient Safety": 10,
        "Medical Errors": 7,
        "Medication Issues": 5
      },
      "total": 22
    }
  ],
  "date_range": {
    "from": "2024-01-01",
    "to": "2024-12-31"
  },
  "granularity": "monthly",
  "grouped_by": "category"
}
```

---

### 5. Get Single Red Flag Details

**Request:**
```bash
curl -X GET "http://127.0.0.1:8000/api/red-flags/1"
```

**Expected Response:**
```json
{
  "red_flag_id": 1,
  "recordID": "RF-2024-001",
  "patient_name": "محمد أحمد",
  "date_received": "2024-01-15",
  "department": "الطوارئ",
  "category": "Patient Safety",
  "subcategory": "Delayed Diagnosis",
  "severity": "CRITICAL",
  "status": "UNDER_REVIEW",
  "isNeverEvent": true,
  "incident_details": {
    "complaint_text": "حدث تأخير في تشخيص حالة حرجة مما أدى إلى تدهور حالة المريض...",
    "immediate_action": "تم نقل المريض للعناية المركزة فورًا",
    "actions_taken": "تم عقد اجتماع لفريق الطوارئ لمراجعة البروتوكولات...",
    "root_cause": "نقص في عدد الأطباء المتخصصين خلال الفترة المسائية",
    "harm_level": "Moderate Harm",
    "stage": "Occurrence Stage"
  },
  "timeline": [
    {
      "date": "2024-01-15",
      "event": "تلقي البلاغ",
      "details": "تم استلام البلاغ عن الحادث"
    },
    {
      "date": "2024-01-16",
      "event": "بدء التحقيق",
      "details": "تم تشكيل لجنة التحقيق"
    },
    {
      "date": "2024-01-20",
      "event": "اجتماع اللجنة",
      "details": "اجتماع أول للجنة التحقيق"
    }
  ],
  "related_actions": [
    {
      "action": "تحديث بروتوكول التشخيص",
      "responsible": "رئيس قسم الطوارئ",
      "deadline": "2024-02-15",
      "status": "In Progress"
    },
    {
      "action": "تدريب الفريق الطبي",
      "responsible": "قسم التدريب",
      "deadline": "2024-03-01",
      "status": "Planned"
    }
  ]
}
```

**Error Response (Not Found):**
```json
{
  "error": "RED_FLAG_NOT_FOUND",
  "message": "Red flag with ID 999 not found",
  "message_ar": "لم يتم العثور على العلم الأحمر ذو المعرف 999"
}
```

---

### 6. Export PDF (Not Implemented Yet)

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/api/red-flags/1/export-pdf"
```

**Expected Response (501 Not Implemented):**
```json
{
  "error": "NOT_IMPLEMENTED",
  "message": "PDF export functionality is not yet implemented",
  "message_ar": "وظيفة تصدير PDF غير مطبقة بعد"
}
```

---

### 7. Batch Export (Not Implemented Yet)

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/api/red-flags/export-batch" \
  -H "Content-Type: application/json" \
  -d '{
    "filters": {
      "status": "FINISHED",
      "from_date": "2024-01-01",
      "to_date": "2024-12-31"
    },
    "format": "pdf"
  }'
```

**Expected Response (501 Not Implemented):**
```json
{
  "error": "NOT_IMPLEMENTED",
  "message": "Batch export functionality is not yet implemented",
  "message_ar": "وظيفة التصدير الجماعي غير مطبقة بعد"
}
```

---

## Sample Requests (Python)

### Using `requests` Library

```python
import requests

BASE_URL = "http://127.0.0.1:8000"

# 1. Get red flags list
response = requests.get(
    f"{BASE_URL}/api/red-flags",
    params={
        "status": "OPEN",
        "severity": "CRITICAL",
        "limit": 50
    }
)
data = response.json()
print(f"Found {data['total']} red flags")

# 2. Get statistics
response = requests.get(
    f"{BASE_URL}/api/red-flags/statistics",
    params={
        "from_date": "2024-01-01",
        "to_date": "2024-12-31"
    }
)
stats = response.json()
print(f"Total red flags: {stats['total_red_flags']}")
print(f"Unfinished: {stats['unfinished']}")

# 3. Get trends
response = requests.get(
    f"{BASE_URL}/api/red-flags/trends",
    params={
        "granularity": "monthly",
        "group_by": "category"
    }
)
trends = response.json()
for trend in trends['trends']:
    print(f"{trend['period']}: {trend['total']} incidents")

# 4. Get single red flag details
red_flag_id = 1
response = requests.get(f"{BASE_URL}/api/red-flags/{red_flag_id}")
if response.status_code == 200:
    details = response.json()
    print(f"Record ID: {details['recordID']}")
    print(f"Status: {details['status']}")
else:
    print(f"Error: {response.json()}")
```

---

## Sample Requests (JavaScript/React)

### Using Axios

```javascript
import axios from 'axios';

const BASE_URL = 'http://127.0.0.1:8000';

// 1. Get red flags list with filters
async function fetchRedFlags() {
  try {
    const response = await axios.get(`${BASE_URL}/api/red-flags`, {
      params: {
        status: 'OPEN',
        severity: 'CRITICAL',
        from_date: '2024-01-01',
        limit: 50,
        offset: 0
      }
    });
    
    console.log(`Total: ${response.data.total}`);
    console.log(`Red flags:`, response.data.red_flags);
    return response.data;
  } catch (error) {
    console.error('Error fetching red flags:', error.response?.data);
  }
}

// 2. Get statistics
async function fetchStatistics() {
  try {
    const response = await axios.get(`${BASE_URL}/api/red-flags/statistics`, {
      params: {
        from_date: '2024-01-01',
        to_date: '2024-12-31'
      }
    });
    
    console.log('Statistics:', response.data);
    return response.data;
  } catch (error) {
    console.error('Error fetching statistics:', error.response?.data);
  }
}

// 3. Get trends
async function fetchTrends(granularity = 'monthly', groupBy = 'none') {
  try {
    const response = await axios.get(`${BASE_URL}/api/red-flags/trends`, {
      params: {
        granularity,
        group_by: groupBy
      }
    });
    
    console.log('Trends:', response.data.trends);
    return response.data;
  } catch (error) {
    console.error('Error fetching trends:', error.response?.data);
  }
}

// 4. Get single red flag
async function fetchRedFlagDetails(id) {
  try {
    const response = await axios.get(`${BASE_URL}/api/red-flags/${id}`);
    console.log('Red flag details:', response.data);
    return response.data;
  } catch (error) {
    if (error.response?.status === 404) {
      console.error('Red flag not found');
    } else {
      console.error('Error:', error.response?.data);
    }
  }
}

// Usage in React component
function RedFlagsPage() {
  const [redFlags, setRedFlags] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    async function loadData() {
      setLoading(true);
      const [flagsData, statsData] = await Promise.all([
        fetchRedFlags(),
        fetchStatistics()
      ]);
      
      setRedFlags(flagsData?.red_flags || []);
      setStats(statsData);
      setLoading(false);
    }
    
    loadData();
  }, []);
  
  // ... render UI
}
```

---

## Common Error Codes

| Status Code | Error | Description |
|-------------|-------|-------------|
| **200** | Success | Request completed successfully |
| **404** | RED_FLAG_NOT_FOUND | Red flag with specified ID does not exist |
| **500** | QUERY_FAILED | Database query error |
| **500** | STATISTICS_FAILED | Error calculating statistics |
| **500** | TRENDS_FAILED | Error generating trend data |
| **501** | NOT_IMPLEMENTED | Feature not yet available (PDF export) |

---

## Testing Checklist

Use this checklist to verify all endpoints work correctly:

- [ ] **Test endpoint** works (`/api/red-flags/test`)
- [ ] **List endpoint** returns data without filters
- [ ] **List endpoint** filters by status (OPEN, UNDER_REVIEW, FINISHED)
- [ ] **List endpoint** filters by severity (HIGH, CRITICAL)
- [ ] **List endpoint** filters by date range
- [ ] **List endpoint** searches by patient name/record ID
- [ ] **List endpoint** filters Never Events overlap
- [ ] **List endpoint** pagination works (limit/offset)
- [ ] **Statistics endpoint** returns all KPIs
- [ ] **Statistics endpoint** respects date range
- [ ] **Trends endpoint** works with monthly granularity
- [ ] **Trends endpoint** works with quarterly granularity
- [ ] **Trends endpoint** works with weekly granularity
- [ ] **Trends endpoint** groups by category
- [ ] **Trends endpoint** groups by severity
- [ ] **Trends endpoint** groups by department
- [ ] **Details endpoint** returns full red flag info
- [ ] **Details endpoint** returns 404 for invalid ID
- [ ] **Export PDF** returns 501 (not implemented)
- [ ] **Batch export** returns 501 (not implemented)

---

## Next Steps

1. **Test all endpoints** using the examples above
2. **Verify data accuracy** against database
3. **Check Arabic text rendering** in responses
4. **Test pagination** with different limits/offsets
5. **Validate date range filtering**
6. **Report any bugs or issues** to backend team

---

**Last Updated:** 2024-12-XX  
**API Version:** 1.0  
**Contact:** Backend Development Team
