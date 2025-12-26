# TableView API - Browser Test URLs

**Base URL:** `http://127.0.0.1:8000`

---

## ✅ Endpoint 1: GET /api/complaints (Paginated Complaints List)

### Test 1: Basic Request (All Complaints, Page 1)
```
http://127.0.0.1:8000/api/complaints
```
**Expected:** Returns first 50 complaints with default sorting (received_date DESC)

---

### Test 2: With Pagination
```
http://127.0.0.1:8000/api/complaints?page=1&page_size=10
```
**Expected:** Returns 10 complaints per page

---

### Test 3: With Search (Complaint Number)
```
http://127.0.0.1:8000/api/complaints?search=C-2024
```
**Expected:** Returns complaints with "C-2024" in complaint_number, patient_name, or complaint_text

---

### Test 4: Filter by Status
```
http://127.0.0.1:8000/api/complaints?status=open
```
**Expected:** Returns only open complaints

---

### Test 5: Filter by Issuing Department
```
http://127.0.0.1:8000/api/complaints?issuing_dept_id=1
```
**Expected:** Returns complaints issued by department ID 1

---

### Test 6: Filter by Target Department
```
http://127.0.0.1:8000/api/complaints?target_dept_id=1
```
**Expected:** Returns complaints targeting department ID 1

---

### Test 7: Filter by Source
```
http://127.0.0.1:8000/api/complaints?source=patient
```
**Expected:** Returns complaints reported by patients

---

### Test 8: Filter by Severity
```
http://127.0.0.1:8000/api/complaints?severity_id=3
```
**Expected:** Returns complaints with severity level 3 (High)

---

### Test 9: Filter by Domain
```
http://127.0.0.1:8000/api/complaints?domain_id=1
```
**Expected:** Returns complaints in domain ID 1

---

### Test 10: Filter by Category
```
http://127.0.0.1:8000/api/complaints?category_id=5
```
**Expected:** Returns complaints in category ID 5

---

### Test 11: Filter by Red Flag
```
http://127.0.0.1:8000/api/complaints?is_red_flag=true
```
**Expected:** Returns only red-flagged complaints

---

### Test 12: Filter by Never Event
```
http://127.0.0.1:8000/api/complaints?is_never_event=true
```
**Expected:** Returns only Never Event complaints

---

### Test 13: Filter by Year
```
http://127.0.0.1:8000/api/complaints?year=2024
```
**Expected:** Returns complaints received in 2024

---

### Test 14: Filter by Year and Month
```
http://127.0.0.1:8000/api/complaints?year=2024&month=3
```
**Expected:** Returns complaints received in March 2024

---

### Test 15: Filter by Date Range
```
http://127.0.0.1:8000/api/complaints?start_date=2024-03-01&end_date=2024-03-31
```
**Expected:** Returns complaints received between March 1-31, 2024

---

### Test 16: Combined Filters (Status + Department + Date Range)
```
http://127.0.0.1:8000/api/complaints?status=open&issuing_dept_id=1&start_date=2024-01-01&end_date=2024-12-31
```
**Expected:** Returns open complaints from department 1 in 2024

---

### Test 17: Sort by Severity (Ascending)
```
http://127.0.0.1:8000/api/complaints?sort_by=severity_id&sort_order=asc
```
**Expected:** Returns complaints sorted by severity (Low to High)

---

### Test 18: Sort by Updated Date (Descending)
```
http://127.0.0.1:8000/api/complaints?sort_by=updated_at&sort_order=desc
```
**Expected:** Returns complaints sorted by most recently updated first

---

### Test 19: Custom Page Size (500 max)
```
http://127.0.0.1:8000/api/complaints?page_size=500
```
**Expected:** Returns up to 500 complaints in one page

---

### Test 20: Simplified View
```
http://127.0.0.1:8000/api/complaints?view=simplified
```
**Expected:** Returns complaints with view="simplified" metadata

---

## ✅ Endpoint 2: GET /api/complaints/filter-options (Dropdown Options)

### Test 1: Basic Filter Options
```
http://127.0.0.1:8000/api/complaints/filter-options
```
**Expected:** Returns all available filter options (departments, sources, statuses, severities, domains, categories)

---

### Test 2: Filter Options with Counts
```
http://127.0.0.1:8000/api/complaints/filter-options?include_counts=true
```
**Expected:** Returns filter options with record counts for each option

---

## ✅ Endpoint 3: GET /api/complaints/{id} (Single Complaint Details)

### Test 1: Get Complaint by ID
```
http://127.0.0.1:8000/api/complaints/1
```
**Expected:** Returns full details of complaint with CaseID=1 (UNMASKED patient data)

---

### Test 2: Get Non-Existent Complaint (404)
```
http://127.0.0.1:8000/api/complaints/999999
```
**Expected:** Returns 404 error with message "Complaint not found"

---

## ✅ Endpoint 4: GET /api/complaints/count (Count Matching Filters)

### Test 1: Count All Complaints
```
http://127.0.0.1:8000/api/complaints/count
```
**Expected:** Returns total count of all complaints

---

### Test 2: Count Open Complaints
```
http://127.0.0.1:8000/api/complaints/count?status=open
```
**Expected:** Returns count of open complaints

---

### Test 3: Count Red Flags
```
http://127.0.0.1:8000/api/complaints/count?is_red_flag=true
```
**Expected:** Returns count of red-flagged complaints

---

### Test 4: Count with Date Range
```
http://127.0.0.1:8000/api/complaints/count?start_date=2024-01-01&end_date=2024-12-31
```
**Expected:** Returns count of complaints in 2024

---

### Test 5: Count with Combined Filters
```
http://127.0.0.1:8000/api/complaints/count?status=open&issuing_dept_id=1&severity_id=3
```
**Expected:** Returns count of open, high-severity complaints from department 1

---

## ✅ Endpoint 5: POST /api/complaints/export (Export Complaints)

**NOTE:** This endpoint requires POST request with JSON body. Cannot test directly in browser.
**Use Postman, curl, or frontend fetch() to test.**

### Test with curl (Windows PowerShell):
```powershell
curl -X POST "http://127.0.0.1:8000/api/complaints/export" `
  -H "Content-Type: application/json" `
  -d '{
    "format": "csv",
    "filters": {
      "status": "open",
      "start_date": "2024-01-01",
      "end_date": "2024-12-31"
    },
    "columns": [
      "complaint_number",
      "received_date",
      "issuing_dept_name",
      "domain_name",
      "severity_name",
      "status"
    ],
    "include_patient_identifiers": false,
    "language": "en"
  }'
```

**Expected:** Returns export metadata with `export_id`, `file_name`, `download_url`, `record_count`

---

## ✅ Endpoint 6: GET /api/complaints/views (Table View Configurations)

### Test 1: Get Available Views
```
http://127.0.0.1:8000/api/complaints/views
```
**Expected:** Returns array of view configurations (complete, simplified, red_flags_only) with column lists

---

## 🔧 Quick Testing Checklist

### ✅ Step 1: Start Backend Server
```powershell
cd "c:\Users\IT\Documents\GitHub Repository\Patient_Feedback\backend"
uvicorn main:app --reload
```

### ✅ Step 2: Test in Browser
Open each URL above in your browser. URLs should return JSON responses.

### ✅ Step 3: Check Automatic API Docs
```
http://127.0.0.1:8000/docs
```
FastAPI automatically generates interactive API documentation where you can test all endpoints (including POST).

### ✅ Step 4: Test POST Export Endpoint
Use the `/docs` interface to test POST /api/complaints/export:
1. Go to http://127.0.0.1:8000/docs
2. Find "POST /api/complaints/export"
3. Click "Try it out"
4. Enter request body (JSON example above)
5. Click "Execute"

---

## 📊 Expected Response Structure Examples

### GET /api/complaints Response:
```json
{
  "complaints": [
    {
      "id": 1,
      "complaint_number": "C-2024-001234",
      "complaint_summary": "Patient reported delayed response...",
      "received_date": "2024-03-15",
      "patient_mrn": "MRN-***789",
      "patient_name": "Ahmed H.",
      "issuing_dept_name": "Emergency Department",
      "target_dept_name": "Nursing Services",
      "domain_name": "Communication",
      "severity_name": "Medium",
      "status": "in_progress",
      "is_red_flag": false,
      "days_open": 33
    }
  ],
  "pagination": {
    "page": 1,
    "page_size": 50,
    "total_records": 2847,
    "total_pages": 57
  },
  "filters_applied": {
    "status": null,
    "issuing_dept_id": null
  },
  "view": "complete"
}
```

### GET /api/complaints/filter-options Response:
```json
{
  "issuing_departments": [
    {
      "id": 12,
      "name": "Emergency Department",
      "name_ar": "قسم الطوارئ",
      "code": "ER"
    }
  ],
  "sources": [
    {
      "value": "patient",
      "label": "Patient",
      "label_ar": "مريض"
    }
  ],
  "statuses": [
    {
      "value": "open",
      "label": "Open",
      "label_ar": "مفتوح"
    }
  ]
}
```

### GET /api/complaints/count Response:
```json
{
  "total_count": 287,
  "filters_applied": {
    "status": "open",
    "start_date": "2024-01-01"
  }
}
```

---

## ⚠️ Common Issues & Troubleshooting

### Issue 1: Empty complaints array
**Cause:** All incidents have NULL FeedbackRecievedDate (as discovered in investigation testing)
**Solution:** Run UPDATE statement to populate dates, or filter by other fields

### Issue 2: 500 Internal Server Error
**Cause:** Database connection issue or SQL syntax error
**Solution:** Check terminal logs for stack trace

### Issue 3: 400 Bad Request
**Cause:** Invalid parameter values (e.g., page_size > 500, invalid date format)
**Solution:** Check error message in response body

### Issue 4: Masked Patient Data Shows "undefined"
**Cause:** Patient data fields are NULL in database
**Solution:** Use get_complaint_by_id (unmasked) or ensure data exists

---

## 🎯 Priority Testing Order

1. ✅ GET /api/complaints (basic, no filters)
2. ✅ GET /api/complaints?page_size=10 (pagination)
3. ✅ GET /api/complaints/filter-options (dropdown data)
4. ✅ GET /api/complaints/count (total count)
5. ✅ GET /api/complaints/{id} (single record)
6. ✅ GET /api/complaints?status=open (filtering)
7. ✅ GET /api/complaints/views (view configurations)
8. ✅ POST /api/complaints/export (via /docs)

---

**All endpoints are now registered and ready to test!**
