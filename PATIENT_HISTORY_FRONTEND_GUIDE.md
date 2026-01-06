# 📱 Patient History Page - Frontend Integration Guide

## Base URL
```
http://0.0.0.0:8000/api/patients
```

---

## 🚀 6 API Endpoints

### 1️⃣ GET `/search` - Search Patients

**Search for patients**

```bash
curl "http://0.0.0.0:8000/api/patients/search?query=أحمد&limit=20"
curl "http://0.0.0.0:8000/api/patients/search?mrn=MRN-123456"
curl "http://0.0.0.0:8000/api/patients/search?phone=966"
```

**Query Parameters:**
| Param | Type | Required | Example |
|-------|------|----------|---------|
| `query` | string | No | `"أحمد"` |
| `mrn` | string | No | `"MRN-123456"` |
| `phone` | string | No | `"+966500"` |
| `date_of_birth` | string | No | `"1985-05-15"` |
| `limit` | integer | No | `50` (max: 100) |

**Response (200):**
```json
{
  "patients": [
    {
      "patient_id": "12345",
      "mrn": "MRN-123456",
      "full_name": "أحمد محمد علي",
      "date_of_birth": "1985-05-15",
      "age": 39,
      "gender": "Male",
      "phone": "+966XXXXXXXXX"
    }
  ],
  "total": 1
}
```

---

### 2️⃣ GET `/{patient_id}/profile` - Patient Profile

**Get patient details**

```bash
curl "http://0.0.0.0:8000/api/patients/12345/profile"
```

**Response (200):**
```json
{
  "patient_id": "12345",
  "mrn": "MRN-123456",
  "full_name": "أحمد محمد علي",
  "full_name_en": "Ahmed Mohamed Ali",
  "date_of_birth": "1985-05-15",
  "age": 39,
  "gender": "Male",
  "nationality": "Saudi Arabia",
  "phone": "+966XXXXXXXXX",
  "email": "ahmed@example.com",
  "address": "الرياض، السعودية",
  "emergency_contact": "فاطمة علي",
  "emergency_phone": "+966YYYYYYYYY",
  "total_incidents": 5,
  "last_visit_date": "2024-11-15",
  "registration_date": "2020-03-10"
}
```

**Errors:**
- 404: Patient not found
- 500: Server error

---

### 3️⃣ GET `/{patient_id}/incidents` - Patient Incidents

**Get patient's incidents/feedback records**

```bash
curl "http://0.0.0.0:8000/api/patients/12345/incidents"
curl "http://0.0.0.0:8000/api/patients/12345/incidents?severity=High&limit=50&offset=0"
curl "http://0.0.0.0:8000/api/patients/12345/incidents?from_date=2024-01-01&to_date=2024-12-31"
```

**Query Parameters:**
| Param | Type | Required | Example |
|-------|------|----------|---------|
| `from_date` | string | No | `"2024-01-01"` |
| `to_date` | string | No | `"2024-12-31"` |
| `department` | string | No | `"Emergency"` |
| `severity` | string | No | `"High"` |
| `status` | string | No | `"Closed"` |
| `limit` | integer | No | `100` (max: 100) |
| `offset` | integer | No | `0` |

**Response (200):**
```json
{
  "patient_id": "12345",
  "patient_name": "أحمد محمد علي",
  "incidents": [
    {
      "incident_id": 1,
      "record_id": "C-2024-0015",
      "date": "2024-11-15",
      "feedback_received_date": "2024-11-15",
      "department": "Emergency Department",
      "department_ar": "قسم الطوارئ",
      "category": "Delayed Diagnosis",
      "category_ar": "تأخر في التشخيص",
      "severity": "High",
      "doctor_name": "د. خالد حسن",
      "status": "Closed",
      "description": "تأخر كبير في تشخيص الحالة...",
      "is_red_flag": false,
      "is_never_event": false
    }
  ],
  "total": 5,
  "limit": 100,
  "offset": 0
}
```

---

### 4️⃣ GET `/{patient_id}/incidents/{incident_id}` - Incident Details

**Get full incident details**

```bash
curl "http://0.0.0.0:8000/api/patients/12345/incidents/1"
```

**Response (200):**
```json
{
  "incident_id": 1,
  "record_id": "C-2024-0015",
  "date": "2024-11-15",
  "feedback_received_date": "2024-11-15",
  "patient_id": "12345",
  "patient_name": "أحمد محمد علي",
  "department": "Emergency Department",
  "target_department": "Emergency Department",
  "category": "Delayed Diagnosis",
  "category_ar": "تأخر في التشخيص",
  "classification": "Clinical > Delayed Diagnosis > Emergency",
  "severity": "High",
  "harm_level": "Minor",
  "stage": "Admission",
  "doctor_name": "د. خالد حسن",
  "status": "Closed",
  "complaint_text": "تأخر كبير في تشخيص الحالة الطارئة مما أدى إلى تفاقم الحالة",
  "immediate_action": "تم توفير الرعاية الفورية",
  "taken_action": "تم متابعة الحالة",
  "is_red_flag": false,
  "is_never_event": false,
  "created_at": "2024-11-15T10:30:00",
  "last_updated_at": "2024-11-20T14:00:00"
}
```

**Errors:**
- 404: Incident not found
- 500: Server error

---

### 5️⃣ GET `/{patient_id}/full-history` - Combined Profile + Incidents

**Get profile and incidents in single call (for efficiency)**

```bash
curl "http://0.0.0.0:8000/api/patients/12345/full-history"
curl "http://0.0.0.0:8000/api/patients/12345/full-history?severity=High"
```

**Response (200):**
```json
{
  "profile": {
    "patient_id": "12345",
    "mrn": "MRN-123456",
    "full_name": "أحمد محمد علي",
    "age": 39,
    "total_incidents": 5,
    ...
  },
  "incidents": {
    "patient_id": "12345",
    "patient_name": "أحمد محمد علي",
    "incidents": [...],
    "total": 5,
    "limit": 100,
    "offset": 0
  }
}
```

---

### 6️⃣ GET `/{patient_id}/export` - Export History

**Export patient history as CSV or JSON**

```bash
curl "http://0.0.0.0:8000/api/patients/12345/export?format=csv"
curl "http://0.0.0.0:8000/api/patients/12345/export?format=json"
curl "http://0.0.0.0:8000/api/patients/12345/export?format=csv&from_date=2024-01-01&to_date=2024-12-31"
```

**Query Parameters:**
| Param | Type | Required | Example |
|-------|------|----------|---------|
| `format` | string | Yes | `"csv"` or `"json"` |
| `from_date` | string | No | `"2024-01-01"` |
| `to_date` | string | No | `"2024-12-31"` |
| `include_profile` | boolean | No | `true` |

**Response (CSV - 200):**
```
Content-Type: text/csv
Content-Disposition: attachment; filename="patient_12345_history_2024-12-17.csv"

PATIENT PROFILE
Patient ID,12345
MRN,MRN-123456
Full Name,أحمد محمد علي
Total Incidents,5

INCIDENT HISTORY
Record ID,Date,Department,Category,Severity,Doctor,Status,Complaint
C-2024-0015,2024-11-15,Emergency,Delayed Diagnosis,High,د. خالد,Closed,تأخر كبير...
```

**Response (JSON - 200):**
```json
{
  "export_date": "2024-12-17T15:30:00",
  "format": "json",
  "patient": {
    "patient_id": "12345",
    "mrn": "MRN-123456",
    "full_name": "أحمد محمد علي",
    "total_incidents": 5
  },
  "incidents": [...]
}
```

**Errors:**
- 400: Invalid format (must be 'csv' or 'json')
- 404: Patient not found
- 500: Server error

---

## ⚡ Frontend Implementation Checklist

### Page Load
- [ ] Call `GET /search` with empty/null params OR show search input
- [ ] When user selects patient from search → Call `GET /{id}/full-history`
- [ ] Display patient profile card from response
- [ ] Display incidents table from response

### Search
- [ ] Input field for search query
- [ ] Input for MRN (optional)
- [ ] Call `GET /search?query=...&limit=50`
- [ ] Display results in dropdown/table
- [ ] Click patient → Load full-history

### Patient Profile Card
- [ ] Display: name, MRN, age, gender, phone
- [ ] Display: nationality, email, address
- [ ] Display: emergency contact info
- [ ] Display: total incidents count, last visit date
- [ ] "Refresh" button → Re-call full-history

### Incidents Table
- [ ] Columns: Date, Department (Arabic), Category, Severity, Status, Doctor
- [ ] Sorting: By date descending (most recent first)
- [ ] Pagination: limit=100, offset controlled by UI
- [ ] Click row → Call `GET /{patient_id}/incidents/{incident_id}`
- [ ] Show red flag/never event indicators (is_red_flag, is_never_event)

### Incident Detail Modal
- [ ] Show full complaint text
- [ ] Show immediate action & taken action
- [ ] Show classification hierarchy
- [ ] Show harm level, stage, target department
- [ ] Close button to return to table

### Filters
- [ ] Date range filter (from_date, to_date)
- [ ] Department filter
- [ ] Severity filter (High, Medium, Low)
- [ ] Status filter
- [ ] Re-call incidents endpoint when filters change

### Export
- [ ] "Export CSV" button → `GET /export?format=csv`
- [ ] "Export JSON" button → `GET /export?format=json`
- [ ] CSV downloads as file
- [ ] JSON displays or downloads
- [ ] Allow date range selection before export

### Error Handling
- [ ] 404: Show "Patient not found" message
- [ ] 400: Show "Invalid export format"
- [ ] 500: Show "Server error, try again"
- [ ] Show loading spinner while fetching
- [ ] Show empty state if no incidents

---

## 🎨 Quick JavaScript Examples

### Search & Load Profile
```javascript
// Search
const response = await fetch(`/api/patients/search?query=${query}`);
const data = await response.json();
const patients = data.patients;

// Select patient
const patientId = patients[0].patient_id;
const fullResponse = await fetch(`/api/patients/${patientId}/full-history`);
const fullData = await fullResponse.json();

displayProfile(fullData.profile);
displayIncidents(fullData.incidents.incidents);
```

### Filter Incidents
```javascript
const params = new URLSearchParams({
  severity: 'High',
  from_date: '2024-01-01',
  to_date: '2024-12-31',
  limit: 50,
  offset: 0
});

const response = await fetch(`/api/patients/${patientId}/incidents?${params}`);
const data = await response.json();
displayIncidents(data.incidents);
```

### Export CSV
```javascript
const response = await fetch(`/api/patients/${patientId}/export?format=csv`);
const blob = await response.blob();
const url = window.URL.createObjectURL(blob);
const a = document.createElement('a');
a.href = url;
a.download = `patient_history.csv`;
a.click();
```

---

## 📊 HTTP Status Codes

| Code | Scenario | Handle |
|------|----------|--------|
| **200** | Success | Display data |
| **400** | Bad format | Show error message |
| **404** | Not found | Show "Not found" |
| **500** | Server error | Show "Try again" |

---

## ✅ That's It!

6 endpoints, all documented, ready to integrate. Copy-paste and go! 🚀
