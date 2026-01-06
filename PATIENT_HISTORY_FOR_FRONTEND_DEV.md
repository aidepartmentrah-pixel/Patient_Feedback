# 🔌 Patient History - For Frontend Dev (Copy This)

## Minimal Text to Give Frontend Coder

---

Hi, here are the 6 API endpoints for the patient history page.

### Base URL
```
http://0.0.0.0:8000/api/patients
```

### Endpoints

**1. Search patients**
```
GET /search?query=أحمد&limit=50
```
Returns: `{patients: [...], total: N}`

**2. Get patient profile**
```
GET /{patient_id}/profile
```
Returns: Full patient details (name, age, contact, etc.)

**3. Get patient incidents**
```
GET /{patient_id}/incidents?severity=High&from_date=2024-01-01
```
Returns: `{patient_id, incidents: [...], total: N}`

**4. Get incident details**
```
GET /{patient_id}/incidents/{incident_id}
```
Returns: Full incident with complaint text, actions, etc.

**5. Get profile + incidents (MOST EFFICIENT)**
```
GET /{patient_id}/full-history
```
Returns: `{profile: {...}, incidents: {...}}`

**6. Export history**
```
GET /{patient_id}/export?format=csv
GET /{patient_id}/export?format=json
```
Returns: CSV file or JSON data

### Response Example
```json
// GET /search?query=أحمد
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

### Use This Endpoint Most
```
GET /{patient_id}/full-history
```
Returns both profile AND incidents in one call. Most efficient.

### JavaScript Example
```javascript
// Search
const res = await fetch('/api/patients/search?query=أحمد');
const data = await res.json();

// Load patient (most efficient)
const res2 = await fetch(`/api/patients/${data.patients[0].patient_id}/full-history`);
const fullData = await res2.json();
displayProfile(fullData.profile);
displayIncidents(fullData.incidents.incidents);

// Export
const res3 = await fetch(`/api/patients/${patientId}/export?format=csv`);
const blob = await res3.blob();
// download file...
```

### Query Parameters
- `query` - search by name
- `mrn` - search by MRN
- `phone` - search by phone
- `from_date` - filter incidents from date (YYYY-MM-DD)
- `to_date` - filter incidents to date (YYYY-MM-DD)
- `severity` - filter by severity (High, Medium, Low)
- `department` - filter by department
- `status` - filter by status
- `limit` - max results (default 100)
- `offset` - pagination (default 0)

### Error Codes
- 200 = Success
- 400 = Bad request
- 404 = Not found
- 500 = Server error

---

That's it. 6 endpoints. All JSON responses. Ready to use.

For more details, see:
- **PATIENT_HISTORY_QUICKSTART.md** (quick reference)
- **PATIENT_HISTORY_FRONTEND_GUIDE.md** (detailed guide)
