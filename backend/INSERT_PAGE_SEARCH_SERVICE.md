# Insert Page Search Service - Implementation Summary

## Overview
Created a comprehensive search service for the insert page that allows searching and selecting patients, doctors, and employees from the database. This ensures data integrity by only allowing selection of existing entities.

## Files Created/Modified

### 1. **backend/api/services/search_service.py** (NEW)
Complete search service with the following functions:

#### Patient Search Functions
- `search_patients(search_text, limit=20)` - Search for patients by name or document number
- `get_patient_by_id(patient_admission_id)` - Get specific patient details

**Searches in:**
- FullName
- FirstName
- LastName
- DocumentNumber
- MedicalFileNumber

**Returns:** PatientAdmissionID, full patient details, admission date, etc.

#### Doctor Search Functions
- `search_doctors(search_text, limit=20)` - Search for active doctors by name
- `get_doctor_by_id(doctor_id)` - Get specific doctor details

**Searches in:**
- Name (doctor name)

**Returns:** DoctorID, Name, **SpecialityID, SpecialityName**, IsActive, IsAdmitted, IsClinic

**Key Feature:** Includes speciality information for each doctor!

#### Employee Search Functions
- `search_employees(search_text, limit=20)` - Search for active employees by name
- `get_employee_by_id(employee_id)` - Get specific employee details

**Searches in:**
- FullName (employee name)

**Returns:** EmployeeID, FullName, **JobTitle** (this is the employee's "speciality"), JobID, Department info, IsManager, IsActive

**Key Feature:** JobTitle field serves as the employee's speciality/role!

### 2. **backend/api/routers/insert_router.py** (MODIFIED)
Added 6 new endpoints:

#### Search Endpoints
1. `GET /api/records/search/patients?q={query}&limit={limit}`
2. `GET /api/records/search/doctors?q={query}&limit={limit}`
3. `GET /api/records/search/employees?q={query}&limit={limit}`

#### Verification Endpoints (Get by ID)
4. `GET /api/records/patient/{patient_admission_id}`
5. `GET /api/records/doctor/{doctor_id}`
6. `GET /api/records/employee/{employee_id}`

### 3. **backend/TEST_SEARCH_API.md** (NEW)
Complete API testing documentation with:
- Endpoint descriptions
- Test cases
- Example requests/responses
- curl commands
- Python test examples
- Frontend integration notes

### 4. **backend/test_search_endpoints.py** (NEW)
Python test script to verify all endpoints work correctly.

## Database Tables Used

### APP_VIEWTABLE_PATIENT_ADMISSION
- **Primary Key:** PatientAdmissionID
- **Key Fields:** FullName, FirstName, LastName, DocumentNumber, MedicalFileNumber
- **Additional:** BirthDate, SEX, PhoneNumber, AdmissionDate

### APP_VIEWTABLE_VW_DOCTORS
- **Primary Key:** DoctorID
- **Key Fields:** Name, **SpecialityID, SpecialityName** ✨
- **Filter:** IsActive = 1 (only active doctors)

### APP_VIEWTABLE_HR_EMPLOYEES
- **Primary Key:** EmployeeID
- **Key Fields:** FullName, **JobTitle** ✨ (employee's speciality)
- **Additional:** JobID, DepartmentID, SectionID, IsManager
- **Filter:** IsActive = 1 (only active employees)

## API Endpoints Summary

### Search Patients
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
      "document_number": "123456789",
      "medical_file_number": "MF-2024-001",
      ...
    }
  ],
  "count": 1
}
```

### Search Doctors (with Speciality!)
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
      "speciality_name": "طب الطوارئ",  ← Doctor's speciality!
      "is_active": true
    }
  ],
  "count": 1
}
```

### Search Employees (with Job Title!)
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
      "job_title": "ممرض",  ← Employee's job title (speciality)!
      "is_manager": false,
      "is_active": true
    }
  ],
  "count": 1
}
```

## Key Features

### ✅ Search Functionality
- **Partial matching:** Searches with LIKE %query%
- **Case-insensitive:** Works with Arabic and English
- **Multiple fields:** Searches across multiple columns
- **Limited results:** Default 20, max 100 to prevent overload
- **Active only:** Only returns active doctors and employees

### ✅ Speciality Information
- **Doctors:** Include `speciality_name` field showing their medical specialty
- **Employees:** Include `job_title` field showing their role/position

### ✅ Data Validation
- Only existing entities can be selected
- Can verify selections using "get by ID" endpoints
- All foreign key references are valid

### ✅ Selection Rules
- **Patients:** ONE patient only (single selection)
- **Doctors:** MULTIPLE doctors (array of IDs)
- **Employees:** MULTIPLE employees (array of IDs)

## Frontend Integration Guide

### Patient Selection (Single)
```javascript
// 1. User types in search box
const searchPatients = async (query) => {
  const response = await fetch(
    `http://localhost:8000/api/records/search/patients?q=${query}&limit=20`
  );
  const data = await response.json();
  return data.patients;
};

// 2. Display results in dropdown
// Show: full_name, document_number, medical_file_number

// 3. User selects ONE patient
const selectedPatient = { patient_admission_id: 12345 };

// 4. Verify selection before submission
const verifyPatient = async (id) => {
  const response = await fetch(
    `http://localhost:8000/api/records/patient/${id}`
  );
  return await response.json();
};
```

### Doctor Selection (Multiple)
```javascript
// 1. User types in search box
const searchDoctors = async (query) => {
  const response = await fetch(
    `http://localhost:8000/api/records/search/doctors?q=${query}&limit=20`
  );
  const data = await response.json();
  return data.doctors;
};

// 2. Display results with SPECIALITY
// Show: name + " - " + speciality_name
// Example: "د. خالد حسن - طب الطوارئ"

// 3. User selects MULTIPLE doctors
const selectedDoctors = [
  { doctor_id: 45, name: "د. خالد", speciality_name: "طب الطوارئ" },
  { doctor_id: 67, name: "د. أحمد", speciality_name: "جراحة" }
];
```

### Employee Selection (Multiple)
```javascript
// 1. User types in search box
const searchEmployees = async (query) => {
  const response = await fetch(
    `http://localhost:8000/api/records/search/employees?q=${query}&limit=20`
  );
  const data = await response.json();
  return data.employees;
};

// 2. Display results with JOB TITLE
// Show: full_name + " - " + job_title
// Example: "محمد أحمد - ممرض"

// 3. User selects MULTIPLE employees
const selectedEmployees = [
  { employee_id: 789, full_name: "محمد أحمد", job_title: "ممرض" },
  { employee_id: 456, full_name: "علي حسن", job_title: "فني أشعة" }
];
```

## Testing Instructions

### 1. Start the Server
```bash
cd backend
python -m uvicorn main:app --reload
```

### 2. Test with curl
```bash
# Search patients
curl "http://127.0.0.1:8000/api/records/search/patients?q=test&limit=5"

# Search doctors
curl "http://127.0.0.1:8000/api/records/search/doctors?q=د&limit=5"

# Search employees
curl "http://127.0.0.1:8000/api/records/search/employees?q=م&limit=5"
```

### 3. Run Test Script
```bash
python test_search_endpoints.py
```

### 4. Use Swagger UI
Open browser: http://127.0.0.1:8000/docs

Navigate to the "Records" section and test the search endpoints interactively.

## Error Handling

All endpoints return consistent error responses:

### Search Failed (500)
```json
{
  "detail": {
    "error": "SEARCH_FAILED",
    "message": "Failed to search patients"
  }
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

### Invalid Query (422)
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

## Database Connection

Uses the existing connection configuration from `backend/core/database.py`:
```python
Server: SOCIALMEDIA
Database: IncidentManager
Authentication: Windows Authentication (Trusted_Connection)
```

## Security Considerations

1. **SQL Injection Protection:** Uses parameterized queries
2. **Active Records Only:** Only returns IsActive=1 for doctors/employees
3. **Limit Protection:** Max 100 results per query to prevent abuse
4. **Input Validation:** FastAPI validates query parameters

## Next Steps for Frontend

1. **Implement Autocomplete Components:**
   - Create reusable autocomplete/dropdown component
   - Debounce search input (300-500ms)
   - Show loading state while searching

2. **Display Speciality Information:**
   - For doctors: Show name and speciality_name together
   - For employees: Show full_name and job_title together

3. **Handle Multiple Selections:**
   - Doctors: Multi-select dropdown or tag input
   - Employees: Multi-select dropdown or tag input
   - Patients: Single-select dropdown only

4. **Validation:**
   - Ensure patient is selected before submission
   - Verify all doctor/employee IDs exist
   - Store IDs in the incident case tables

5. **User Experience:**
   - Minimum 1 character to start search
   - Show "No results" message appropriately
   - Display count of results
   - Clear search on selection

## Related Tables for Insert

After selecting entities, store them in:
- **APP_IncidentCaseDoctor** - For selected doctors
- **APP_IncidentCaseEmployee** - For selected employees
- Patient info stored directly in the main incident case

## Status
✅ **Complete and Ready for Integration**

All endpoints created, tested for syntax errors, and documented. The frontend team can now integrate these search features into the insert page.
