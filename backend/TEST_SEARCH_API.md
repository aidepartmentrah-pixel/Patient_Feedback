# Test Search API Endpoints
# Test the search functionality for patients, doctors, and employees

## Base URL
http://127.0.0.1:8000

## Test Endpoints

### 1. Search Patients
Search for patients by name, document number, or medical file number.

**Endpoint:** `GET /api/records/search/patients`

**Test Cases:**

#### Test 1.1: Search by Arabic name
```
GET http://127.0.0.1:8000/api/records/search/patients?q=محمد&limit=10
```

#### Test 1.2: Search by partial name
```
GET http://127.0.0.1:8000/api/records/search/patients?q=أحم&limit=5
```

#### Test 1.3: Search by document number
```
GET http://127.0.0.1:8000/api/records/search/patients?q=123&limit=20
```

**Expected Response:**
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

---

### 2. Search Doctors
Search for active doctors by name with speciality information.

**Endpoint:** `GET /api/records/search/doctors`

**Test Cases:**

#### Test 2.1: Search by Arabic name
```
GET http://127.0.0.1:8000/api/records/search/doctors?q=خالد&limit=10
```

#### Test 2.2: Search by partial name
```
GET http://127.0.0.1:8000/api/records/search/doctors?q=أحم&limit=5
```

#### Test 2.3: Get all active doctors (using common letter)
```
GET http://127.0.0.1:8000/api/records/search/doctors?q=د&limit=50
```

**Expected Response:**
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

**Note:** The speciality_name field shows the doctor's specialization.

---

### 3. Search Employees
Search for active employees by name with job title information.

**Endpoint:** `GET /api/records/search/employees`

**Test Cases:**

#### Test 3.1: Search by Arabic name
```
GET http://127.0.0.1:8000/api/records/search/employees?q=محمد&limit=10
```

#### Test 3.2: Search by partial name
```
GET http://127.0.0.1:8000/api/records/search/employees?q=أحم&limit=5
```

#### Test 3.3: Get multiple results
```
GET http://127.0.0.1:8000/api/records/search/employees?q=ع&limit=20
```

**Expected Response:**
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

**Note:** The job_title field shows the employee's job position/speciality.

---

### 4. Get Specific Patient by ID
Verify a patient selection by ID.

**Endpoint:** `GET /api/records/patient/{patient_admission_id}`

**Test Case:**
```
GET http://127.0.0.1:8000/api/records/patient/12345
```

**Expected Response:**
```json
{
  "success": true,
  "patient": {
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
}
```

---

### 5. Get Specific Doctor by ID
Verify a doctor selection by ID.

**Endpoint:** `GET /api/records/doctor/{doctor_id}`

**Test Case:**
```
GET http://127.0.0.1:8000/api/records/doctor/45
```

**Expected Response:**
```json
{
  "success": true,
  "doctor": {
    "doctor_id": 45,
    "name": "د. خالد حسن",
    "speciality_id": 3,
    "speciality_name": "طب الطوارئ",
    "is_active": true,
    "is_admitted": true,
    "is_clinic": false
  }
}
```

---

### 6. Get Specific Employee by ID
Verify an employee selection by ID.

**Endpoint:** `GET /api/records/employee/{employee_id}`

**Test Case:**
```
GET http://127.0.0.1:8000/api/records/employee/789
```

**Expected Response:**
```json
{
  "success": true,
  "employee": {
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
}
```

---

## Testing with curl

### Search Patients
```bash
curl "http://127.0.0.1:8000/api/records/search/patients?q=محمد&limit=10"
```

### Search Doctors
```bash
curl "http://127.0.0.1:8000/api/records/search/doctors?q=أحمد&limit=10"
```

### Search Employees
```bash
curl "http://127.0.0.1:8000/api/records/search/employees?q=محمد&limit=10"
```

---

## Testing with Python

```python
import requests

base_url = "http://127.0.0.1:8000"

# Search patients
response = requests.get(f"{base_url}/api/records/search/patients", params={"q": "محمد", "limit": 10})
print("Patients:", response.json())

# Search doctors
response = requests.get(f"{base_url}/api/records/search/doctors", params={"q": "أحمد", "limit": 10})
print("Doctors:", response.json())

# Search employees
response = requests.get(f"{base_url}/api/records/search/employees", params={"q": "محمد", "limit": 10})
print("Employees:", response.json())

# Get specific patient
response = requests.get(f"{base_url}/api/records/patient/12345")
print("Patient Details:", response.json())
```

---

## Error Responses

### Invalid Query Parameter (empty search)
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

### Not Found (404)
```json
{
  "detail": {
    "error": "NOT_FOUND",
    "message": "Patient not found"
  }
}
```

### Server Error (500)
```json
{
  "detail": {
    "error": "SEARCH_FAILED",
    "message": "Failed to search patients"
  }
}
```

---

## Notes for Frontend Integration

### Patient Selection
- **ONE patient only** can be selected
- Must search and select from existing patients
- Display: Full Name, Document Number, Medical File Number
- Store: patient_admission_id

### Doctor Selection
- **MULTIPLE doctors** can be selected
- Must search and select from existing active doctors
- Display: Name, **Speciality Name** (important!)
- Store: doctor_id (array of IDs)

### Employee Selection
- **MULTIPLE employees** can be selected
- Must search and select from existing active employees
- Display: Full Name, **Job Title** (this is the employee's "speciality")
- Store: employee_id (array of IDs)

### Autocomplete Implementation
1. User types minimum 1 character
2. Call search endpoint with query
3. Display results with speciality/job title
4. User selects from dropdown
5. Validate selection by ID before submission

### Search Features
- Partial matching (LIKE %query%)
- Case-insensitive
- Searches multiple fields (names, IDs, document numbers)
- Limited results (default 20, max 100)
- Only active records returned

---

## API Documentation
FastAPI automatically generates interactive documentation:
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc
