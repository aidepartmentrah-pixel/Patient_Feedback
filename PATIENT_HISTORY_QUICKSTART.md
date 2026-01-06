# 📱 Patient History - Frontend Dev Quickstart

## Copy This

### Base URL
```
http://0.0.0.0:8000/api/patients
```

### 6 Endpoints (That's All You Need)

```javascript
// 1. SEARCH patients
GET /search?query=أحمد&limit=50

// 2. GET patient profile
GET /{patient_id}/profile

// 3. GET patient incidents
GET /{patient_id}/incidents?severity=High&limit=100&offset=0

// 4. GET incident details
GET /{patient_id}/incidents/{incident_id}

// 5. GET profile + incidents together (USE THIS!)
GET /{patient_id}/full-history

// 6. EXPORT as CSV or JSON
GET /{patient_id}/export?format=csv
GET /{patient_id}/export?format=json
```

---

## Response Examples

### Search (GET /search)
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
      "phone": "+966500000000"
    }
  ],
  "total": 1
}
```

### Profile (GET /{id}/profile)
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
  "phone": "+966500000000",
  "email": "ahmed@example.com",
  "address": "الرياض، السعودية",
  "emergency_contact": "فاطمة علي",
  "emergency_phone": "+966511111111",
  "total_incidents": 5,
  "last_visit_date": "2024-11-15",
  "registration_date": "2020-03-10"
}
```

### Incidents (GET /{id}/incidents)
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
      "description": "تأخر كبير في تشخيص...",
      "is_red_flag": false,
      "is_never_event": false
    }
  ],
  "total": 5,
  "limit": 100,
  "offset": 0
}
```

### Incident Detail (GET /{id}/incidents/{incident_id})
```json
{
  "incident_id": 1,
  "record_id": "C-2024-0015",
  "date": "2024-11-15",
  "complaint_text": "تأخر كبير في تشخيص الحالة الطارئة",
  "immediate_action": "تم توفير الرعاية الفورية",
  "taken_action": "تم متابعة الحالة",
  "classification": "Clinical > Delayed Diagnosis > Emergency",
  "severity": "High",
  "harm_level": "Minor",
  "stage": "Admission",
  "is_red_flag": false,
  "is_never_event": false,
  "created_at": "2024-11-15T10:30:00",
  "last_updated_at": "2024-11-20T14:00:00"
}
```

### Full History (GET /{id}/full-history) - BEST
```json
{
  "profile": { /* profile object above */ },
  "incidents": { /* incidents object above */ }
}
```

### Export JSON (GET /{id}/export?format=json)
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

### Export CSV (GET /{id}/export?format=csv)
```
Returns: file download with headers + patient profile + incidents table
```

---

## Quick JavaScript

### Search
```javascript
const search = async (query) => {
  const res = await fetch(`/api/patients/search?query=${query}`);
  return await res.json();
};
```

### Load Patient (Most Efficient)
```javascript
const loadPatient = async (patientId) => {
  const res = await fetch(`/api/patients/${patientId}/full-history`);
  const data = await res.json();
  
  displayProfile(data.profile);
  displayIncidents(data.incidents.incidents);
  return data;
};
```

### Filter Incidents
```javascript
const filterIncidents = async (patientId, severity, fromDate, toDate) => {
  const params = new URLSearchParams({
    severity,
    from_date: fromDate,
    to_date: toDate,
    limit: 50,
    offset: 0
  });
  
  const res = await fetch(`/api/patients/${patientId}/incidents?${params}`);
  return await res.json();
};
```

### Get Incident Details
```javascript
const getDetails = async (patientId, incidentId) => {
  const res = await fetch(`/api/patients/${patientId}/incidents/${incidentId}`);
  return await res.json();
};
```

### Export
```javascript
const exportCSV = async (patientId) => {
  const res = await fetch(`/api/patients/${patientId}/export?format=csv`);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'patient_history.csv';
  a.click();
};
```

---

## UI Components Needed

1. **Search Component**
   - Input field
   - Search button
   - Results dropdown
   - Click → load patient

2. **Patient Card**
   - Name, MRN, Age, Gender
   - Phone, Email
   - Emergency contact
   - Total incidents, Last visit
   - Refresh button

3. **Incidents Table**
   - Date, Department, Category, Severity, Status
   - Sortable by date
   - Paginated (100 per page)
   - Click row → detail modal
   - Filter buttons

4. **Incident Modal**
   - Full complaint text
   - Actions taken
   - Classification hierarchy
   - Harm level, Stage
   - Red flag/Never event badges
   - Close button

5. **Filters**
   - Date from/to
   - Department select
   - Severity select
   - Status select
   - Apply/Reset buttons

6. **Export**
   - CSV button
   - JSON button
   - Date range selector (optional)

---

## Error Handling

```javascript
.catch(err => {
  if (err.status === 404) showError("Not found");
  else if (err.status === 400) showError("Invalid request");
  else showError("Server error");
});
```

---

## Query Parameters Reference

| Endpoint | Param | Type | Example |
|----------|-------|------|---------|
| /search | query | string | "أحمد" |
| /search | mrn | string | "MRN-123456" |
| /search | phone | string | "966500" |
| /search | limit | int | 50 |
| /incidents | from_date | date | "2024-01-01" |
| /incidents | to_date | date | "2024-12-31" |
| /incidents | department | string | "Emergency" |
| /incidents | severity | string | "High" |
| /incidents | status | string | "Closed" |
| /incidents | limit | int | 100 |
| /incidents | offset | int | 0 |
| /export | format | string | "csv" or "json" |

---

Done! 🚀 Start building!
