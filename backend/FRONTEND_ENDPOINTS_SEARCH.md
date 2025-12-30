# Search Endpoints for Insert Page - Frontend Integration

## Overview
These endpoints allow searching and selecting existing patients, doctors, and employees from the database. Users search by typing, then select from the results.

---

## BASE URL
```
http://127.0.0.1:8000
```

---

## ENDPOINTS

### 1. Search Patients (Single Selection)
**Endpoint:** `GET /api/records/search/patients`

**Query Parameters:**
- `q` (required) - Search text (minimum 1 character)
- `limit` (optional) - Max results (default: 20, max: 100)

**Example Request:**
```
GET /api/records/search/patients?q=محمد&limit=10
```

**Response:**
```json
{
  "success": true,
  "patients": [
    {
      "patient_admission_id": 12345,
      "full_name": "أحمد محمد علي",
      "first_name": "أحمد",
      "last_name": "علي",
      "document_number": "123456789",
      "phone_number": "0501234567",
      "birth_date": "1990-05-15",
      "sex": "M",
      "medical_file_number": "MF-2024-001",
      "admission_date": "2024-12-15T10:30:00"
    }
  ],
  "count": 1
}
```

**UI Behavior:**
- User types in search box → calls this endpoint
- Display results showing: `full_name`, `document_number`, `medical_file_number`
- User selects ONE patient only
- Store: `patient_admission_id`

---

### 2. Search Doctors (Multiple Selection)
**Endpoint:** `GET /api/records/search/doctors`

**Query Parameters:**
- `q` (required) - Search text (minimum 1 character)
- `limit` (optional) - Max results (default: 20, max: 100)

**Example Request:**
```
GET /api/records/search/doctors?q=خالد&limit=10
```

**Response:**
```json
{
  "success": true,
  "doctors": [
    {
      "doctor_id": 45,
      "name": "د. خالد حسن",
      "speciality_id": 3,
      "speciality_name": "طب الطوارئ",
      "is_active": true,
      "is_admitted": true,
      "is_clinic": false
    }
  ],
  "count": 1
}
```

**UI Behavior:**
- User types in search box → calls this endpoint
- Display results showing: `name` + " - " + `speciality_name`
  - Example: "د. خالد حسن - طب الطوارئ"
- User can select MULTIPLE doctors
- Store: Array of `doctor_id` values

---

### 3. Search Employees (Multiple Selection)
**Endpoint:** `GET /api/records/search/employees`

**Query Parameters:**
- `q` (required) - Search text (minimum 1 character)
- `limit` (optional) - Max results (default: 20, max: 100)

**Example Request:**
```
GET /api/records/search/employees?q=محمد&limit=10
```

**Response:**
```json
{
  "success": true,
  "employees": [
    {
      "employee_id": 789,
      "full_name": "محمد أحمد السعيد",
      "job_title": "ممرض",
      "job_id": 12,
      "department_id": 5,
      "section_id": 8,
      "administration_id": 2,
      "is_manager": false,
      "is_active": true
    }
  ],
  "count": 1
}
```

**UI Behavior:**
- User types in search box → calls this endpoint
- Display results showing: `full_name` + " - " + `job_title`
  - Example: "محمد أحمد السعيد - ممرض"
- User can select MULTIPLE employees
- Store: Array of `employee_id` values

---

### 4. Verify Patient Selection (Optional)
**Endpoint:** `GET /api/records/patient/{patient_admission_id}`

**Example Request:**
```
GET /api/records/patient/12345
```

**Response:**
```json
{
  "success": true,
  "patient": {
    "patient_admission_id": 12345,
    "full_name": "أحمد محمد علي",
    "document_number": "123456789",
    ...
  }
}
```

---

### 5. Verify Doctor Selection (Optional)
**Endpoint:** `GET /api/records/doctor/{doctor_id}`

**Example Request:**
```
GET /api/records/doctor/45
```

**Response:**
```json
{
  "success": true,
  "doctor": {
    "doctor_id": 45,
    "name": "د. خالد حسن",
    "speciality_name": "طب الطوارئ",
    ...
  }
}
```

---

### 6. Verify Employee Selection (Optional)
**Endpoint:** `GET /api/records/employee/{employee_id}`

**Example Request:**
```
GET /api/records/employee/789
```

**Response:**
```json
{
  "success": true,
  "employee": {
    "employee_id": 789,
    "full_name": "محمد أحمد السعيد",
    "job_title": "ممرض",
    ...
  }
}
```

---

## UI COMPONENT REQUIREMENTS

### Patient Field (Single Select Autocomplete)
```
[🔍 Search Patient...                    ]
     ↓ (user types)
┌────────────────────────────────────────┐
│ أحمد محمد علي                         │
│ Document: 123456789                   │
│ File: MF-2024-001                     │
├────────────────────────────────────────┤
│ محمد علي حسن                          │
│ Document: 987654321                   │
│ File: MF-2024-002                     │
└────────────────────────────────────────┘
```

### Doctor Field (Multi-Select Autocomplete)
```
[🔍 Search Doctors...                    ]
     ↓ (user types)
┌────────────────────────────────────────┐
│ د. خالد حسن - طب الطوارئ             │
├────────────────────────────────────────┤
│ د. أحمد محمود - جراحة عامة           │
├────────────────────────────────────────┤
│ د. فاطمة علي - طب الأطفال            │
└────────────────────────────────────────┘

Selected: [د. خالد حسن - طب الطوارئ] [د. أحمد محمود - جراحة] [×]
```

### Employee Field (Multi-Select Autocomplete)
```
[🔍 Search Employees...                  ]
     ↓ (user types)
┌────────────────────────────────────────┐
│ محمد أحمد السعيد - ممرض               │
├────────────────────────────────────────┤
│ علي حسن محمد - فني أشعة              │
├────────────────────────────────────────┤
│ سارة محمود - صيدلانية                │
└────────────────────────────────────────┘

Selected: [محمد أحمد - ممرض] [علي حسن - فني أشعة] [×]
```

---

## IMPLEMENTATION NOTES

### Search Behavior
- Trigger search when user types (debounce 300-500ms)
- Minimum 1 character to search
- Show loading indicator while searching
- Display "No results found" when empty

### Display Format
- **Patient:** Show name, document number, medical file
- **Doctor:** Show name AND speciality together
- **Employee:** Show name AND job title together

### Selection Storage
- **Patient:** Store single `patient_admission_id` (integer)
- **Doctors:** Store array of `doctor_id` values (e.g., `[45, 67, 89]`)
- **Employees:** Store array of `employee_id` values (e.g., `[789, 456, 123]`)

### Validation
- Patient: Required, must be selected from list
- Doctors: Optional, can select multiple
- Employees: Optional, can select multiple

---

## ERROR HANDLING

### No Results
```json
{
  "success": true,
  "patients": [],
  "count": 0
}
```
**UI:** Display "No patients found" message

### Search Failed
```json
{
  "detail": {
    "error": "SEARCH_FAILED",
    "message": "Failed to search patients"
  }
}
```
**UI:** Display error message to user

### Invalid Query (empty string)
```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["query", "q"],
      "msg": "String should have at least 1 character"
    }
  ]
}
```
**UI:** Require minimum 1 character before searching

---

## EXAMPLE JAVASCRIPT/TYPESCRIPT CODE

### Search Function
```javascript
const searchPatients = async (query) => {
  if (!query || query.length < 1) return [];
  
  const response = await fetch(
    `http://127.0.0.1:8000/api/records/search/patients?q=${encodeURIComponent(query)}&limit=20`
  );
  
  if (!response.ok) {
    throw new Error('Search failed');
  }
  
  const data = await response.json();
  return data.patients;
};

const searchDoctors = async (query) => {
  if (!query || query.length < 1) return [];
  
  const response = await fetch(
    `http://127.0.0.1:8000/api/records/search/doctors?q=${encodeURIComponent(query)}&limit=20`
  );
  
  const data = await response.json();
  return data.doctors;
};

const searchEmployees = async (query) => {
  if (!query || query.length < 1) return [];
  
  const response = await fetch(
    `http://127.0.0.1:8000/api/records/search/employees?q=${encodeURIComponent(query)}&limit=20`
  );
  
  const data = await response.json();
  return data.employees;
};
```

### Display Format Function
```javascript
// Format doctor for display
const formatDoctor = (doctor) => {
  return `${doctor.name} - ${doctor.speciality_name}`;
};

// Format employee for display
const formatEmployee = (employee) => {
  return `${employee.full_name} - ${employee.job_title}`;
};

// Format patient for display
const formatPatient = (patient) => {
  return `${patient.full_name} (${patient.document_number})`;
};
```

---

## TESTING

### Test in Browser Console
```javascript
// Test patient search
fetch('http://127.0.0.1:8000/api/records/search/patients?q=test&limit=5')
  .then(r => r.json())
  .then(console.log);

// Test doctor search
fetch('http://127.0.0.1:8000/api/records/search/doctors?q=د&limit=5')
  .then(r => r.json())
  .then(console.log);

// Test employee search
fetch('http://127.0.0.1:8000/api/records/search/employees?q=م&limit=5')
  .then(r => r.json())
  .then(console.log);
```

### Interactive API Documentation
```
http://127.0.0.1:8000/docs
```
Navigate to "Records" section to test all endpoints interactively.

---

## SUMMARY

**Changed/New Endpoints:**
1. ✅ `GET /api/records/search/patients` - Search patients
2. ✅ `GET /api/records/search/doctors` - Search doctors with speciality
3. ✅ `GET /api/records/search/employees` - Search employees with job title
4. ✅ `GET /api/records/patient/{id}` - Get patient by ID
5. ✅ `GET /api/records/doctor/{id}` - Get doctor by ID
6. ✅ `GET /api/records/employee/{id}` - Get employee by ID

**No changes to existing insert endpoint** (`POST /api/records/add`)
