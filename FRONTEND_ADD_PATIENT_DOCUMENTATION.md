# Frontend Documentation: Add Patient Feature

## 📋 Overview

This document provides all the information needed to implement the **Add Patient** frontend UI, including API endpoints, request/response structures, validation rules, and testing URLs.

**Feature**: Create and manage patients in the reserve table (user-created patients separate from hospital database)

**Base URL**: `http://localhost:8000`

---

## 🎯 API Endpoints

### 1. Create New Patient

**Endpoint**: `POST /api/patients/create`

**Purpose**: Create a new patient in the reserve table

#### Request Structure

```typescript
interface CreatePatientRequest {
  first_name: string;              // REQUIRED (2-150 chars)
  middle_name?: string | null;     // Optional (max 150 chars)
  last_name?: string | null;       // Optional (max 150 chars)
  mother_name?: string | null;     // Optional (max 150 chars)
  phone_number?: string | null;    // Optional (min 7 digits, max 50 chars)
  phone_number2?: string | null;   // Optional (min 7 digits, max 50 chars)
  birth_date?: string | null;      // Optional (YYYY-MM-DD format)
  sex?: string | null;             // Optional (M, F, Male, or Female)
  document_number?: string | null; // Optional (max 100 chars)
  medical_file_number?: string | null; // Optional (max 100 chars)
  spouse?: string | null;          // Optional (max 150 chars)
  address_line1?: string | null;   // Optional (max 300 chars)
  address_line2?: string | null;   // Optional (max 300 chars)
}
```

#### Example Request

```javascript
// JavaScript/TypeScript Example
const createPatient = async (patientData) => {
  const response = await fetch('http://localhost:8000/api/patients/create', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      first_name: "Ahmed",
      middle_name: "Mohammed",
      last_name: "Al-Rashid",
      mother_name: "Fatima",
      phone_number: "0501234567",
      phone_number2: "0509876543",
      birth_date: "1985-05-15",
      sex: "M",
      document_number: "1234567890",
      medical_file_number: "MRN-2026-001234",
      spouse: "Sara Al-Ahmad",
      address_line1: "123 King Fahd Road, Riyadh",
      address_line2: "Building 5, Apartment 201"
    })
  });
  
  return await response.json();
};
```

#### Success Response (201 Created)

```json
{
  "success": true,
  "message": "Patient 'Ahmed Mohammed Al-Rashid' created successfully with ID 100001",
  "message_ar": "تم إنشاء المريض 'Ahmed Mohammed Al-Rashid' بنجاح بالرقم 100001",
  "patient": {
    "patient_admission_id": 100001,
    "full_name": "Ahmed Mohammed Al-Rashid",
    "first_name": "Ahmed",
    "middle_name": "Mohammed",
    "last_name": "Al-Rashid",
    "mother_name": "Fatima",
    "phone_number": "0501234567",
    "phone_number2": "0509876543",
    "birth_date": "1985-05-15",
    "sex": "M",
    "document_number": "1234567890",
    "medical_file_number": "MRN-2026-001234",
    "spouse": "Sara Al-Ahmad",
    "address_line1": "123 King Fahd Road, Riyadh",
    "address_line2": "Building 5, Apartment 201",
    "source": "reserve"
  }
}
```

#### Error Responses

**400 Bad Request - Validation Error**
```json
{
  "detail": {
    "error": "VALIDATION_ERROR",
    "message": "FirstName is required and must be at least 2 characters",
    "message_ar": "خطأ في التحقق من الصحة"
  }
}
```

**409 Conflict - Duplicate Patient**
```json
{
  "detail": {
    "error": "DUPLICATE_PATIENT",
    "message": "Patient with FullName 'Ahmed Mohammed Al-Rashid' already exists in reserve (ID: 100001). Cannot create duplicate.",
    "message_ar": "المريض موجود بالفعل"
  }
}
```

**422 Unprocessable Entity - Invalid Field Type**
```json
{
  "detail": [
    {
      "type": "string_type",
      "loc": ["body", "first_name"],
      "msg": "Input should be a valid string",
      "input": 123
    }
  ]
}
```

**500 Internal Server Error**
```json
{
  "detail": {
    "error": "INTERNAL_ERROR",
    "message": "Failed to create patient: Database connection error",
    "message_ar": "خطأ داخلي في النظام"
  }
}
```

---

### 2. Get All Reserve Patients

**Endpoint**: `GET /api/patients/reserve`

**Purpose**: Retrieve all user-created patients (reserve table only)

#### Query Parameters

```typescript
interface GetReservePatientsParams {
  limit?: number;    // Max records per page (default: 100, max: 200)
  offset?: number;   // Records to skip (default: 0)
  order_by?: string; // 'created_at' (newest first) or 'name' (alphabetical)
}
```

#### Example Requests

```javascript
// Get first 100 patients (newest first)
const getReservePatients = async () => {
  const response = await fetch('http://localhost:8000/api/patients/reserve');
  return await response.json();
};

// Get patients with pagination
const getPatientsPage = async (page = 0, pageSize = 50) => {
  const offset = page * pageSize;
  const response = await fetch(
    `http://localhost:8000/api/patients/reserve?limit=${pageSize}&offset=${offset}`
  );
  return await response.json();
};

// Get patients alphabetically
const getPatientsAlphabetically = async () => {
  const response = await fetch(
    'http://localhost:8000/api/patients/reserve?order_by=name&limit=100'
  );
  return await response.json();
};
```

#### Response Structure

```typescript
interface GetReservePatientsResponse {
  patients: Patient[];
  total: number;    // Total number of reserve patients
  count: number;    // Number of patients in this response
  limit: number;    // Applied limit
  offset: number;   // Applied offset
}

interface Patient {
  patient_admission_id: number;
  full_name: string;
  first_name: string;
  middle_name: string | null;
  last_name: string | null;
  mother_name: string | null;
  phone_number: string | null;
  phone_number2: string | null;
  birth_date: string | null;      // YYYY-MM-DD format
  sex: string | null;             // M or F
  document_number: string | null;
  medical_file_number: string | null;
  spouse: string | null;
  address_line1: string | null;
  address_line2: string | null;
  created_at: string;             // ISO datetime
  source: "reserve";              // Always "reserve"
}
```

#### Example Response

```json
{
  "patients": [
    {
      "patient_admission_id": 100002,
      "full_name": "Sara Ali Hassan",
      "first_name": "Sara",
      "middle_name": "Ali",
      "last_name": "Hassan",
      "mother_name": "Maryam",
      "phone_number": "0507654321",
      "phone_number2": null,
      "birth_date": "1992-08-20",
      "sex": "F",
      "document_number": "9876543210",
      "medical_file_number": "MRN-2026-000456",
      "spouse": "Omar Ahmed",
      "address_line1": "456 Prince Sultan Road",
      "address_line2": null,
      "created_at": "2026-01-21 14:30:00",
      "source": "reserve"
    },
    {
      "patient_admission_id": 100001,
      "full_name": "Ahmed Mohammed Al-Rashid",
      "first_name": "Ahmed",
      "middle_name": "Mohammed",
      "last_name": "Al-Rashid",
      "mother_name": "Fatima",
      "phone_number": "0501234567",
      "phone_number2": "0509876543",
      "birth_date": "1985-05-15",
      "sex": "M",
      "document_number": "1234567890",
      "medical_file_number": "MRN-2026-001234",
      "spouse": "Sara Al-Ahmad",
      "address_line1": "123 King Fahd Road, Riyadh",
      "address_line2": "Building 5, Apartment 201",
      "created_at": "2026-01-21 10:30:00",
      "source": "reserve"
    }
  ],
  "total": 42,
  "count": 2,
  "limit": 100,
  "offset": 0
}
```

---

## ✅ Validation Rules

### Required Fields
- **first_name**: REQUIRED, minimum 2 characters, maximum 150 characters

### Optional Fields with Validation

| Field | Max Length | Validation Rules |
|-------|-----------|------------------|
| middle_name | 150 chars | Letters, spaces, Arabic, hyphens, apostrophes |
| last_name | 150 chars | Letters, spaces, Arabic, hyphens, apostrophes |
| mother_name | 150 chars | Letters, spaces, Arabic, hyphens, apostrophes |
| phone_number | 50 chars | Min 7 digits, can include +, -, (, ), spaces |
| phone_number2 | 50 chars | Min 7 digits, can include +, -, (, ), spaces |
| birth_date | - | YYYY-MM-DD format, not in future, age < 150 years |
| sex | - | M, F, Male, or Female (case-insensitive, normalized to M/F) |
| document_number | 100 chars | Alphanumeric, hyphens allowed |
| medical_file_number | 100 chars | Alphanumeric, hyphens allowed |
| spouse | 150 chars | Letters, spaces, Arabic, hyphens, apostrophes |
| address_line1 | 300 chars | Any characters |
| address_line2 | 300 chars | Any characters |

### Important Notes

1. **Duplicate Detection**: System checks for duplicates by:
   - Full Name (combination of first, middle, last name)
   - Document Number (if provided)
   - Medical File Number (if provided)

2. **Automatic Processing**:
   - All whitespace is automatically trimmed
   - Full Name is auto-generated from first_name + middle_name + last_name
   - Sex is normalized: "Male" → "M", "Female" → "F"
   - Source is always set to "reserve"

3. **Patient ID Range**:
   - Reserve patient IDs start at **100000** (to distinguish from hospital patients)
   - Hospital patient IDs are typically < 100000

---

## 🧪 Testing URLs (Manual Browser Testing)

### Test Create Patient (POST)

Since browsers can't easily send POST requests, use these tools:

**Option 1: Using cURL**
```bash
curl -X POST http://localhost:8000/api/patients/create \
  -H "Content-Type: application/json" \
  -d '{
    "first_name": "TestPatient",
    "phone_number": "0501234567",
    "document_number": "TEST-001"
  }'
```

**Option 2: Using PowerShell**
```powershell
$body = @{
    first_name = "TestPatient"
    phone_number = "0501234567"
    document_number = "TEST-001"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/patients/create" `
  -Method POST `
  -Body $body `
  -ContentType "application/json"
```

**Option 3: Using Browser Console (JavaScript)**
```javascript
// Open browser console (F12) and run:
fetch('http://localhost:8000/api/patients/create', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    first_name: "TestPatient",
    phone_number: "0501234567",
    document_number: "TEST-001"
  })
})
.then(r => r.json())
.then(data => console.log(data));
```

### Test Get Reserve Patients (GET - Can use browser directly)

Open these URLs in your browser:

```
# Get all reserve patients (default)
http://localhost:8000/api/patients/reserve

# Get first 10 patients
http://localhost:8000/api/patients/reserve?limit=10

# Get patients alphabetically
http://localhost:8000/api/patients/reserve?order_by=name

# Get second page (patients 11-20)
http://localhost:8000/api/patients/reserve?limit=10&offset=10

# Get maximum allowed (200 patients)
http://localhost:8000/api/patients/reserve?limit=200
```

### API Documentation (Interactive Testing)

FastAPI provides interactive API documentation:

```
# Swagger UI (recommended for testing)
http://localhost:8000/docs

# ReDoc (alternative documentation)
http://localhost:8000/redoc
```

In Swagger UI, you can:
1. Click on an endpoint
2. Click "Try it out"
3. Fill in the parameters
4. Click "Execute"
5. See the response

---

## 💡 Frontend Implementation Examples

### React Example

```typescript
import React, { useState } from 'react';

interface CreatePatientForm {
  first_name: string;
  middle_name?: string;
  last_name?: string;
  phone_number?: string;
  // ... other fields
}

const AddPatientPage: React.FC = () => {
  const [formData, setFormData] = useState<CreatePatientForm>({
    first_name: '',
    middle_name: '',
    last_name: '',
    phone_number: ''
  });
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const response = await fetch('http://localhost:8000/api/patients/create', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
      });

      const data = await response.json();

      if (response.ok) {
        setSuccess(data.message);
        // Reset form or redirect
      } else {
        setError(data.detail?.message || 'Failed to create patient');
      }
    } catch (err) {
      setError('Network error. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="text"
        placeholder="First Name *"
        value={formData.first_name}
        onChange={(e) => setFormData({...formData, first_name: e.target.value})}
        required
      />
      {/* Add more fields */}
      
      {error && <div className="error">{error}</div>}
      {success && <div className="success">{success}</div>}
      
      <button type="submit" disabled={loading}>
        {loading ? 'Creating...' : 'Create Patient'}
      </button>
    </form>
  );
};
```

### Vue.js Example

```vue
<template>
  <div class="add-patient-page">
    <form @submit.prevent="createPatient">
      <div class="form-group">
        <label>First Name *</label>
        <input 
          v-model="formData.first_name" 
          type="text" 
          required 
          minlength="2"
          maxlength="150"
        />
      </div>
      
      <div class="form-group">
        <label>Phone Number</label>
        <input 
          v-model="formData.phone_number" 
          type="tel"
          maxlength="50"
        />
      </div>
      
      <!-- More fields -->
      
      <div v-if="error" class="alert alert-error">{{ error }}</div>
      <div v-if="success" class="alert alert-success">{{ success }}</div>
      
      <button type="submit" :disabled="loading">
        {{ loading ? 'Creating...' : 'Create Patient' }}
      </button>
    </form>
  </div>
</template>

<script>
export default {
  data() {
    return {
      formData: {
        first_name: '',
        middle_name: '',
        last_name: '',
        phone_number: '',
        // ... other fields
      },
      loading: false,
      error: null,
      success: null
    };
  },
  methods: {
    async createPatient() {
      this.loading = true;
      this.error = null;
      this.success = null;

      try {
        const response = await fetch('http://localhost:8000/api/patients/create', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(this.formData)
        });

        const data = await response.json();

        if (response.ok) {
          this.success = data.message;
          this.resetForm();
        } else {
          this.error = data.detail?.message || 'Failed to create patient';
        }
      } catch (err) {
        this.error = 'Network error. Please try again.';
      } finally {
        this.loading = false;
      }
    },
    resetForm() {
      this.formData = {
        first_name: '',
        middle_name: '',
        last_name: '',
        phone_number: ''
      };
    }
  }
};
</script>
```

### Vanilla JavaScript Example

```javascript
// HTML Form
/*
<form id="addPatientForm">
  <input type="text" id="first_name" required placeholder="First Name *">
  <input type="text" id="middle_name" placeholder="Middle Name">
  <input type="text" id="last_name" placeholder="Last Name">
  <input type="tel" id="phone_number" placeholder="Phone Number">
  <button type="submit">Create Patient</button>
</form>
<div id="message"></div>
*/

document.getElementById('addPatientForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  
  const formData = {
    first_name: document.getElementById('first_name').value,
    middle_name: document.getElementById('middle_name').value || null,
    last_name: document.getElementById('last_name').value || null,
    phone_number: document.getElementById('phone_number').value || null
  };
  
  try {
    const response = await fetch('http://localhost:8000/api/patients/create', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(formData)
    });
    
    const data = await response.json();
    const messageDiv = document.getElementById('message');
    
    if (response.ok) {
      messageDiv.className = 'success';
      messageDiv.textContent = data.message;
      e.target.reset();
    } else {
      messageDiv.className = 'error';
      messageDiv.textContent = data.detail?.message || 'Failed to create patient';
    }
  } catch (error) {
    document.getElementById('message').textContent = 'Network error';
  }
});
```

---

## 📊 Pagination Implementation

### Example: Patient List with Pagination

```typescript
interface PaginationState {
  currentPage: number;
  pageSize: number;
  totalPatients: number;
  patients: Patient[];
}

const PatientListPage: React.FC = () => {
  const [state, setState] = useState<PaginationState>({
    currentPage: 0,
    pageSize: 50,
    totalPatients: 0,
    patients: []
  });
  
  const [loading, setLoading] = useState(false);

  const loadPatients = async (page: number = 0) => {
    setLoading(true);
    try {
      const offset = page * state.pageSize;
      const response = await fetch(
        `http://localhost:8000/api/patients/reserve?limit=${state.pageSize}&offset=${offset}`
      );
      const data = await response.json();
      
      setState({
        ...state,
        currentPage: page,
        totalPatients: data.total,
        patients: data.patients
      });
    } catch (error) {
      console.error('Failed to load patients:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadPatients();
  }, []);

  const totalPages = Math.ceil(state.totalPatients / state.pageSize);

  return (
    <div>
      <h1>Reserve Patients ({state.totalPatients})</h1>
      
      {loading ? (
        <div>Loading...</div>
      ) : (
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Name</th>
              <th>Phone</th>
              <th>Created</th>
            </tr>
          </thead>
          <tbody>
            {state.patients.map(patient => (
              <tr key={patient.patient_admission_id}>
                <td>{patient.patient_admission_id}</td>
                <td>{patient.full_name}</td>
                <td>{patient.phone_number}</td>
                <td>{patient.created_at}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      
      <div className="pagination">
        <button 
          onClick={() => loadPatients(state.currentPage - 1)}
          disabled={state.currentPage === 0}
        >
          Previous
        </button>
        <span>Page {state.currentPage + 1} of {totalPages}</span>
        <button 
          onClick={() => loadPatients(state.currentPage + 1)}
          disabled={state.currentPage >= totalPages - 1}
        >
          Next
        </button>
      </div>
    </div>
  );
};
```

---

## 🔍 Error Handling Best Practices

```typescript
const handleCreatePatient = async (data: CreatePatientRequest) => {
  try {
    const response = await fetch('http://localhost:8000/api/patients/create', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(data)
    });
    
    const result = await response.json();
    
    switch (response.status) {
      case 201:
        // Success
        return {
          success: true,
          message: result.message,
          patient: result.patient
        };
        
      case 400:
        // Validation error
        return {
          success: false,
          error: 'VALIDATION_ERROR',
          message: result.detail?.message || 'Invalid data'
        };
        
      case 409:
        // Duplicate patient
        return {
          success: false,
          error: 'DUPLICATE',
          message: result.detail?.message || 'Patient already exists'
        };
        
      case 422:
        // Pydantic validation error
        const fieldErrors = result.detail.map(err => 
          `${err.loc.join('.')}: ${err.msg}`
        ).join(', ');
        return {
          success: false,
          error: 'INVALID_FIELDS',
          message: fieldErrors
        };
        
      case 500:
        // Server error
        return {
          success: false,
          error: 'SERVER_ERROR',
          message: 'Internal server error. Please try again.'
        };
        
      default:
        return {
          success: false,
          error: 'UNKNOWN',
          message: 'Unexpected error occurred'
        };
    }
  } catch (error) {
    return {
      success: false,
      error: 'NETWORK_ERROR',
      message: 'Unable to connect to server'
    };
  }
};
```

---

## 🎨 UI/UX Recommendations

### Form Layout

1. **Required Fields**
   - Mark with asterisk (*)
   - first_name is the only required field

2. **Field Grouping**
   - Personal Information: first_name, middle_name, last_name, mother_name
   - Contact Information: phone_number, phone_number2, address_line1, address_line2
   - Identification: document_number, medical_file_number
   - Other Details: birth_date, sex, spouse

3. **Input Validation**
   - Show character count for fields with limits
   - Validate phone format before submission
   - Show date picker for birth_date
   - Use radio buttons or dropdown for sex field

4. **Feedback Messages**
   - Success: Show green banner with patient name and ID
   - Error: Show red banner with specific error message
   - Loading: Disable form and show spinner during submission

### Patient List Layout

1. **Table Columns**
   - ID, Full Name, Phone Number, Document Number, Created Date
   - Add action buttons (View, Edit if implemented later)

2. **Sorting Options**
   - Toggle between newest first / alphabetical
   - Show current sort order

3. **Pagination**
   - Show total count
   - Previous/Next buttons
   - Page number indicator
   - Page size selector (25, 50, 100, 200)

4. **Empty State**
   - Show friendly message if no patients exist
   - Add "Create First Patient" button

---

## 📝 Summary Checklist

- [ ] Implement create patient form with validation
- [ ] Handle all error types (400, 409, 422, 500)
- [ ] Display success/error messages in both English and Arabic
- [ ] Implement patient list with GET /api/patients/reserve
- [ ] Add pagination controls
- [ ] Add sorting options (by name / by date)
- [ ] Test with browser console / Postman / Swagger UI
- [ ] Handle network errors gracefully
- [ ] Show loading states during API calls
- [ ] Implement form reset after successful creation

---

## ❓ Questions or Issues?

If you encounter any issues during implementation:

1. Check the API documentation at `http://localhost:8000/docs`
2. Test endpoints manually using Swagger UI
3. Check browser console for errors
4. Verify server is running on port 8000
5. Check database connection

**Server Start Command**:
```bash
cd backend
uvicorn main:app --reload
```

**Test Suite**:
```bash
python test_reserve_patients_endpoint.py
```

---

## 📚 Additional Resources

- **API Swagger Docs**: `http://localhost:8000/docs`
- **API ReDoc**: `http://localhost:8000/redoc`
- **Backend Code**: 
  - Router: `backend/api/routers/patients_router.py`
  - Service: `backend/api/services/patients_service.py`
  - DB Layer: `backend/api/db_layer/patients_db.py`

Good luck with your frontend implementation! 🚀
