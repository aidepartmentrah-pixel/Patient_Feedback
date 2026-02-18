# Offline/Online Database Switching Guide

## Current Test Results (All Passing)

| Test Suite | Count | Status |
|---|---|---|
| Phase 1 | 20/20 | PASS |
| Phase 2 | 28/28 | PASS |
| Phase 3 | 49/49 | PASS |
| Phase 4 | 32/32 | PASS |
| Phase 5 | 38/38 | PASS |
| Phase 6 | 37/37 | PASS |
| Endpoint Tests | 49/49 | PASS |
| Table Config Tests | 31/31 | PASS |
| **TOTAL** | **284/284** | **ALL PASS** |

---

## What You Need to Change (Only 2 Files)

When moving from **offline (development)** to **online (production hospital)**, you only touch 2 files:

```
backend/
  core/
    db_config.py       ← File 1: Database server & name
    table_config.py    ← File 2: External system table/view names
```

Nothing else needs to change. No code, no routes, no services.

---

## File 1: `backend/core/db_config.py` — Database Connection

This file controls **which SQL Server** and **which database** the app connects to.

### Current (Offline/Development)

```python
DB_SERVER   = "SOCIALMEDIA"
DB_DATABASE = "IncidentManager"
DB_DRIVER   = "ODBC Driver 17 for SQL Server"
USE_WINDOWS_AUTH = True
TRUST_SERVER_CERTIFICATE = True
```

### What to Change for Online (Production)

```python
DB_SERVER   = "YOUR_HOSPITAL_SERVER"     # ← Change to production server name/IP
DB_DATABASE = "IncidentManager"          # ← Change if database name is different
DB_DRIVER   = "ODBC Driver 17 for SQL Server"  # Usually keep the same
USE_WINDOWS_AUTH = True                  # Keep True for Windows domain auth
TRUST_SERVER_CERTIFICATE = True          # Keep True unless you have proper SSL certs
```

| Setting | Offline Value | Online Value | Notes |
|---|---|---|---|
| `DB_SERVER` | `"SOCIALMEDIA"` | Your hospital server name or IP | Ask IT for the server name |
| `DB_DATABASE` | `"IncidentManager"` | Same or new name | Usually the same |
| `USE_WINDOWS_AUTH` | `True` | `True` | Change to `False` only if using SQL username/password |
| `TRUST_SERVER_CERTIFICATE` | `True` | `True` | Keep unless proper SSL is set up |

---

## File 2: `backend/core/table_config.py` — External System Views

This file controls the **3 external system table/view names** — these are the views that come from the hospital's HR and HIS (Hospital Information System).

In offline mode, our app reads from local **copy tables** (`APP_VIEWTABLE_*`).  
In online mode, it reads from **real hospital views** that IT provides.

### Current (Offline/Development)

```python
HR_EMPLOYEES_TABLE       = "APP_VIEWTABLE_HR_EMPLOYEES"
PATIENT_ADMISSION_TABLE  = "APP_VIEWTABLE_PATIENT_ADMISSION"
DOCTORS_TABLE            = "APP_VIEWTABLE_VW_DOCTORS"
```

### What to Change for Online (Production)

```python
HR_EMPLOYEES_TABLE       = "VW_HR_EMPLOYEES"          # ← Real HR view name from IT
PATIENT_ADMISSION_TABLE  = "VW_PATIENT_ADMISSION"     # ← Real HIS patient view from IT
DOCTORS_TABLE            = "VW_DOCTORS"                # ← Real HIS doctor view from IT
```

> **Important:** The real view names depend on what your hospital IT team has set up.
> Ask them: "What are the SQL view names for HR employees, patient admissions, and doctors?"

| Variable | Offline (Copy Table) | Online (Real View) | What It Provides |
|---|---|---|---|
| `HR_EMPLOYEES_TABLE` | `APP_VIEWTABLE_HR_EMPLOYEES` | Ask IT | Employee names, job titles, departments |
| `PATIENT_ADMISSION_TABLE` | `APP_VIEWTABLE_PATIENT_ADMISSION` | Ask IT | Patient names, MRN, admission data |
| `DOCTORS_TABLE` | `APP_VIEWTABLE_VW_DOCTORS` | Ask IT | Doctor names, specialities |

### Required Columns

The real views **must** have these columns (same names as the offline tables):

**HR Employees View:**
- `EmployeeID`, `FullName`, `JobTitle`, `JobID`
- `DepartmentID`, `SectionID`, `AdministrationID`
- `IsManager`, `IsActive`

**Patient Admission View:**
- `PatientAdmissionID`, `FullName`, `FirstName`, `LastName`
- `DocumentNumber`, `PhoneNumber1`, `BirthDate`, `SEX`
- `MedicalFileNumber`, `AdmissionDate`, `SystemTime`

**Doctors View:**
- `DoctorID`, `Name`, `SpecialityID`, `SpecialityName`
- `IsActive`, `IsAdmitted`, `IsClinic`

---

## Step-by-Step Switching Procedure

### Going Online (Development → Production)

```
Step 1: Stop the backend server
Step 2: Edit backend/core/db_config.py
        → Change DB_SERVER to production server
        → Change DB_DATABASE if needed
Step 3: Edit backend/core/table_config.py
        → Change 3 table names to real view names from IT
Step 4: Start the backend server
Step 5: Run test_table_config.py to verify
        → All 31 tests should pass
        → If "Live Query Test" fails, the view names are wrong
```

### Going Back Offline (Production → Development)

```
Step 1: Stop the backend server
Step 2: Edit backend/core/db_config.py
        → Set DB_SERVER = "SOCIALMEDIA"
        → Set DB_DATABASE = "IncidentManager"
Step 3: Edit backend/core/table_config.py
        → Set HR_EMPLOYEES_TABLE = "APP_VIEWTABLE_HR_EMPLOYEES"
        → Set PATIENT_ADMISSION_TABLE = "APP_VIEWTABLE_PATIENT_ADMISSION"
        → Set DOCTORS_TABLE = "APP_VIEWTABLE_VW_DOCTORS"
Step 4: Start the backend server
Step 5: Run full test suite to verify (should be 284/284)
```

---

## How to Verify After Switching

Run this single test — it checks everything:

```powershell
cd "c:\Users\IT\Documents\GitHub Repository\Patient_Feedback"
python test_table_config.py
```

What it verifies:
- Config imports correctly
- All 3 table/view names resolve to real tables in the database
- Patient search works
- Doctor search works  
- Employee search works
- Worker profile works

If it says **31/31 PASSED**, you're good.

---

## Architecture Diagram

```
┌────────────────────────────────────────────────────┐
│                    Your App                         │
│                                                     │
│  search_service.py ──┐                              │
│  patients_db.py ─────┤                              │
│  worker_reporting_db.py ─┤   import from            │
│  aggregate_seasonal_report_service.py ──┘           │
│                          │                          │
│                    ┌─────▼──────────┐               │
│                    │ table_config.py│  ← CHANGE     │
│                    │ (3 variables)  │    THESE       │
│                    └─────┬─────────┘               │
│                          │                          │
│                    ┌─────▼─────┐                    │
│                    │database.py│                     │
│                    └─────┬─────┘                    │
│                          │                          │
│                    ┌─────▼──────┐                   │
│                    │db_config.py│  ← AND THIS       │
│                    │(server/db) │                    │
│                    └─────┬──────┘                   │
└──────────────────────────┼─────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │     SQL Server          │
              │                         │
              │  OFFLINE:               │
              │  APP_VIEWTABLE_* tables │
              │  (local copies)         │
              │                         │
              │  ONLINE:                │
              │  VW_* views             │
              │  (real hospital data)   │
              └─────────────────────────┘
```

---

## Questions to Ask Hospital IT Before Going Online

1. **Server name:** What is the SQL Server hostname or IP for production?
2. **Database name:** Is the database called `IncidentManager` or something else?
3. **Authentication:** Windows domain auth or SQL Server login?
4. **HR view name:** What is the SQL view for HR employees? (needs: EmployeeID, FullName, JobTitle, etc.)
5. **Patient view name:** What is the SQL view for patient admissions? (needs: PatientAdmissionID, FullName, MedicalFileNumber, etc.)
6. **Doctor view name:** What is the SQL view for doctors? (needs: DoctorID, Name, SpecialityName, etc.)
7. **Network access:** Can the app server reach the database server on port 1433?
8. **ODBC driver:** Is "ODBC Driver 17 for SQL Server" installed on the production machine?
