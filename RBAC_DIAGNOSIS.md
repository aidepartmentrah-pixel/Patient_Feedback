# 🔐 RBAC System Diagnostic Report

**Date:** January 29, 2026  
**Database:** IncidentManager (SERVER: SOCIALMEDIA)  
**Purpose:** Comprehensive analysis of existing RBAC implementation and user data

---

## 📊 Executive Summary

### Current Status: ✅ RBAC Infrastructure Complete, Test Users Only

- **RBAC Tables:** ✅ Fully implemented (APP_Users, APP_Roles, APP_UserRoleScope)
- **Authentication System:** ✅ Complete (bcrypt-ready, session-based)
- **Authorization Guards:** ✅ Implemented (role-based guards available)
- **User Database:** ⚠️ **TEST USERS ONLY** (6 test accounts, not hospital staff)
- **Org Unit Integration:** ⚠️ **PARTIAL** (OrgUnitID/Type stored but no org tables)

---

## 1. 👤 Current User Accounts in Database

The database contains **6 test user accounts** created on **January 27, 2026** at 3:02 PM.

| User ID | Username | Password | Status | Purpose |
|---------|----------|----------|--------|---------|
| 1 | `software_admin` | `admin123` | ✅ Active | System administrator (testing) |
| 2 | `worker` | `worker123` | ✅ Active | Basic worker role (testing) |
| 3 | `complaint_supervisor` | `sup123` | ✅ Active | Supervisor role (testing) |
| 4 | `section_admin` | `section123` | ✅ Active | Section admin (testing) |
| 5 | `department_admin` | `dept123` | ✅ Active | Department admin (testing) |
| 6 | `administration_admin` | `adminis123` | ✅ Active | Administration admin (testing) |

### 🔑 Working Login Credentials

```
Username: software_admin           | Password: admin123
Username: worker                   | Password: worker123
Username: complaint_supervisor     | Password: sup123
Username: section_admin            | Password: section123
Username: department_admin         | Password: dept123
Username: administration_admin     | Password: adminis123
```

### Password Storage Details

All 6 users currently have **TEMPORARY PASSWORD HASHES** in format:
```
TEMP_HASH_<password>
```

Example: `TEMP_HASH_admin123`

The auth system (auth_db.py) recognizes these temporary hashes and performs **plain-text comparison** for testing. These should be replaced with proper **bcrypt hashes** before production use.

---

## 2. 🎭 User Roles & Assignments

The system defines **6 distinct roles**:

| Role ID | Role Code | English Name | Arabic Name |
|---------|-----------|--------------|-------------|
| 1 | `SOFTWARE_ADMIN` | Software Administrator | مسؤول النظام |
| 2 | `WORKER` | Worker | موظف |
| 3 | `COMPLAINT_SUPERVISOR` | Complaint Supervisor | مشرف الشكاوى |
| 4 | `SECTION_ADMIN` | Section Administrator | مسؤول القسم |
| 5 | `DEPARTMENT_ADMIN` | Department Administrator | مسؤول الإدارة |
| 6 | `ADMINISTRATION_ADMIN` | Administration Administrator | مسؤول الإدارة العامة |

### User-Role Mappings

Each test user has **ONE role** assigned:

| Username | Role Code | Org Unit ID | Org Unit Type | Meaning |
|----------|-----------|-------------|---------------|---------|
| `software_admin` | SOFTWARE_ADMIN | 0 | ADMINISTRATION | System-wide admin |
| `worker` | WORKER | 10 | COMPLAINT | Worker for complaint ID 10 |
| `complaint_supervisor` | COMPLAINT_SUPERVISOR | 10 | COMPLAINT | Supervisor for complaint ID 10 |
| `section_admin` | SECTION_ADMIN | 10 | SECTION | Admin for section ID 10 |
| `department_admin` | DEPARTMENT_ADMIN | 5 | DEPARTMENT | Admin for department ID 5 |
| `administration_admin` | ADMINISTRATION_ADMIN | 1 | ADMINISTRATION | Admin for administration ID 1 |

---

## 3. 🏢 Organizational Unit Linkage

### What Exists in APP_UserRoleScope

The `APP_UserRoleScope` table stores:
- `OrgUnitID` (integer)
- `OrgUnitType` (string: "SECTION", "DEPARTMENT", "ADMINISTRATION", "COMPLAINT")

### What's Missing

**NO dedicated organizational unit tables exist yet!**

The database has:
- ❌ No `APP_Sections` table
- ❌ No `APP_Departments` table  
- ❌ No `APP_Administrations` table

However, there is:
- ✅ `VW_AdminstrationUnit` view (from legacy IncidentRequest system)
- ✅ `AdminsrationUnit` table (typo in name, from old system)
- ✅ References to SectionID/DepartmentID in old `IncidentRequest` tables

### Current User Distribution

**Q: Do we have one user per section/department/administration?**

**A: NO.** We have:
- ❌ **Not one user per section** → Only 1 test user for "section 10"
- ❌ **Not one user per department** → Only 1 test user for "department 5"
- ❌ **Not one user per administration** → Only 1 test user for "administration 1"
- ✅ **Test coverage only** → These are placeholder accounts for testing role logic

---

## 4. 🔐 Authentication System Analysis

### How Login Works (Step by Step)

#### 📥 **Step 1: User POSTs to `/api/auth/login`**

**Request:**
```json
{
  "username": "software_admin",
  "password": "admin123"
}
```

**Location:** `backend/api/routers/auth_router.py` → `@router.post("/login")`

---

#### 🔍 **Step 2: Router calls Auth Service**

**File:** `backend/api/services/auth_service.py`

Function: `login(username, password, request)`

Actions:
1. Calls `validate_user_credentials(username, password)` from DB layer
2. If valid, stores `user_id` in `request.session["user_id"]`
3. Returns success message with username

---

#### 💾 **Step 3: DB Layer validates credentials**

**File:** `backend/api/db_layer/auth_db.py`

Function: `validate_user_credentials(username, password)`

Logic:
```python
1. Query APP_Users WHERE Username = ?
2. If not found → return None (invalid username)
3. If user.IsActive = False → return None (inactive user)
4. Check password:
   - If PasswordHash starts with "TEMP_HASH_":
     → Extract plain password from hash
     → Compare plain text: password == expected_password
   - Else:
     → Use bcrypt.checkpw() to verify hash
5. If password invalid → return None
6. If valid → call get_user_with_scopes(user_id)
7. Return full user dict with roles
```

---

#### 🎟️ **Step 4: Session created**

**Session Storage:** Starlette SessionMiddleware (server-side)

**Session Data:**
```python
request.session["user_id"] = 1  # Example for software_admin
```

**Cookie:** `incident_manager_session` (signed cookie, sent to client)

---

#### ✅ **Step 5: Response returned**

**Response:**
```json
{
  "success": true,
  "message": "Login successful",
  "user": {
    "user_id": 1,
    "username": "software_admin",
    "is_active": true,
    "scopes": [
      {
        "role_code": "SOFTWARE_ADMIN",
        "org_unit_id": 0,
        "org_unit_type": "ADMINISTRATION"
      }
    ]
  }
}
```

---

### Valid Username/Password Combinations

✅ **All 6 test users work with these credentials:**

| Username | Password | Works? |
|----------|----------|--------|
| `software_admin` | `admin123` | ✅ YES |
| `worker` | `worker123` | ✅ YES |
| `complaint_supervisor` | `sup123` | ✅ YES |
| `section_admin` | `section123` | ✅ YES |
| `department_admin` | `dept123` | ✅ YES |
| `administration_admin` | `adminis123` | ✅ YES |

---

### Why Login Attempts Might Fail

#### ❌ **1. Invalid Username**
**Error:** HTTP 401
```json
{
  "detail": {
    "error": "INVALID_CREDENTIALS",
    "message": "Invalid username or password",
    "message_ar": "اسم المستخدم أو كلمة المرور غير صحيحة"
  }
}
```
**Cause:** User doesn't exist in `APP_Users` table

---

#### ❌ **2. Invalid Password**
**Error:** HTTP 401 (same response as above)

**Cause:** 
- Password doesn't match stored hash
- For temp hashes: Plain text comparison fails
- For bcrypt hashes: `bcrypt.checkpw()` returns False

---

#### ❌ **3. Inactive User Account**
**Error:** HTTP 401 (same response)

**Cause:** `APP_Users.IsActive = 0` (user deactivated)

---

#### ❌ **4. Missing Session Middleware**
**Error:** Server error (session not available)

**Cause:** SessionMiddleware not configured in `main.py`

**Check:** Ensure this exists in FastAPI app setup:
```python
from starlette.middleware.sessions import SessionMiddleware
app.add_middleware(SessionMiddleware, secret_key="...")
```

---

#### ❌ **5. Database Connection Failure**
**Error:** HTTP 500
```json
{
  "detail": {
    "error": "LOGIN_ERROR",
    "message": "Login failed: <connection error>",
    "message_ar": "فشل تسجيل الدخول"
  }
}
```
**Cause:** Cannot connect to SQL Server (SOCIALMEDIA/IncidentManager)

---

#### ❌ **6. Wrong Password Format (Case Sensitivity)**
**Passwords are case-sensitive!**

Example:
- ✅ `admin123` → Works
- ❌ `Admin123` → FAILS
- ❌ `ADMIN123` → FAILS

---

## 5. 🔒 Authorization System (Guards)

After successful login, protected endpoints use **guards** for authorization.

### Guard Functions Available

**File:** `backend/api/utils/guards.py`

```python
# Base guard
require_role(current_user, [SOFTWARE_ADMIN, WORKER])

# Role-specific shortcuts
require_software_admin(current_user)
require_worker(current_user)
require_complaint_supervisor(current_user)
require_section_admin(current_user)
require_department_admin(current_user)
require_administration_admin(current_user)

# Utility functions
has_role(current_user, SOFTWARE_ADMIN)  # Returns True/False
require_logged_in(current_user)  # Check if authenticated
```

### How Guards Work

1. Endpoint receives `current_user` from dependency:
   ```python
   @router.get("/admin-only")
   def endpoint(current_user: CurrentUser = Depends(get_current_user)):
   ```

2. Guard checks user's roles:
   ```python
   require_software_admin(current_user)  # Raises 403 if not admin
   ```

3. If user lacks role → HTTP 403:
   ```json
   {
     "detail": {
       "error": "FORBIDDEN",
       "message": "Access denied. Required role: SOFTWARE_ADMIN",
       "required_roles": ["SOFTWARE_ADMIN"],
       "user_roles": ["WORKER"]
     }
   }
   ```

---

## 6. 🏥 Real Hospital Users vs Test Users

### Current State: TEST USERS ONLY ⚠️

**The database contains ONLY 6 artificial test accounts created for development.**

These are **NOT real hospital users**. They are:
- Created by seed data in `phase2_create_rbac_tables.sql`
- Have generic names like "software_admin", "worker"
- Use simple passwords (admin123, worker123)
- Linked to arbitrary org unit IDs (5, 10, 1)

---

### What's Missing for Real Hospital Users

#### ❌ **1. Actual Hospital Staff Data**

Need to populate `APP_Users` with:
- Real employee usernames (from HR system?)
- Proper bcrypt password hashes
- Linkage to actual hospital organizational structure
- Multiple users per section/department/administration

**Example:**
```sql
-- Real users instead of test users
INSERT INTO APP_Users (Username, PasswordHash, IsActive)
VALUES 
  ('dr.mohammed.ahmed', '$2b$12$...', 1),  -- Real bcrypt hash
  ('nurse.fatima.hassan', '$2b$12$...', 1),
  ('admin.sara.abdullah', '$2b$12$...', 1);
```

---

#### ❌ **2. Organizational Unit Master Tables**

Currently using arbitrary IDs (5, 10, 1). Need real tables:

```sql
-- Not implemented yet
CREATE TABLE APP_Sections (
  SectionID INT PRIMARY KEY,
  SectionNameEn NVARCHAR(100),
  SectionNameAr NVARCHAR(100),
  DepartmentID INT,
  IsActive BIT
);

CREATE TABLE APP_Departments (
  DepartmentID INT PRIMARY KEY,
  DepartmentNameEn NVARCHAR(100),
  DepartmentNameAr NVARCHAR(100),
  AdministrationID INT,
  IsActive BIT
);

CREATE TABLE APP_Administrations (
  AdministrationID INT PRIMARY KEY,
  AdministrationNameEn NVARCHAR(100),
  AdministrationNameAr NVARCHAR(100),
  IsActive BIT
);
```

**Note:** Legacy tables exist (`AdminsrationUnit`, `VW_AdminstrationUnit`) but:
- Have typo in name ("Adminsration" instead of "Administration")
- Use different structure than new RBAC system
- May need migration/integration

---

#### ❌ **3. Realistic Role Assignments**

Current: 1 user per role type (testing only)

Real hospital needs:
- **Multiple workers per section** (e.g., 10 workers in Emergency section)
- **Multiple supervisors across departments**
- **One section admin per section**
- **One department admin per department**  
- **One administration admin per administration**
- **Users with MULTIPLE roles** (e.g., worker + supervisor)

---

#### ❌ **4. Integration with HR System**

Need to decide:
- **User provisioning:** Manual vs automated from HR database?
- **Password management:** Self-service reset? Admin-managed?
- **User lifecycle:** Automatic deactivation when employee leaves?
- **Role synchronization:** Roles tied to job titles/positions?

---

#### ❌ **5. Proper Password Hashing**

Current temp hashes (`TEMP_HASH_admin123`) must be replaced:

```python
# Use the provided utility function
from api.db_layer.auth_db import hash_password

# Generate real bcrypt hash
real_hash = hash_password("SecurePassword123!")
# Returns: '$2b$12$...' (bcrypt hash)

# Update user
update_user_password(user_id, real_hash)
```

---

#### ❌ **6. Audit Trail**

Missing:
- Login history (who logged in when)
- Failed login attempts
- Password change history
- Role assignment change logs

---

#### ❌ **7. User Management Interface**

Need admin tools to:
- Create new users
- Reset passwords
- Assign/revoke roles
- Activate/deactivate accounts
- View user activity

---

## 7. 📋 Recommendations for Production Readiness

### Phase 1: User Data Migration

1. **Map existing hospital org structure** to new RBAC model
2. **Create org unit master tables** (Sections, Departments, Administrations)
3. **Extract employee list** from HR system or manual input
4. **Generate usernames** (policy: first.last? employee.id?)
5. **Create initial passwords** (temporary → force change on first login)
6. **Assign roles** based on job positions
7. **Test with pilot group** before full rollout

---

### Phase 2: Security Hardening

1. **Replace all TEMP_HASH passwords** with bcrypt hashes
2. **Implement password policy:**
   - Minimum 8 characters
   - Complexity requirements (uppercase, number, special char)
   - Password expiration (90 days?)
3. **Add account lockout** after X failed login attempts
4. **Enable audit logging** for all authentication events
5. **Setup session expiration** (currently indefinite?)

---

### Phase 3: User Management Features

1. **Admin portal** for user CRUD operations
2. **Self-service password reset** (email verification)
3. **Bulk user import** (CSV/Excel upload)
4. **Role assignment wizard** (select user → assign org units)
5. **User activity reports** (login history, access patterns)

---

### Phase 4: Org Unit Integration

1. **Migrate/integrate** legacy `AdminsrationUnit` data
2. **Link complaints/incidents** to org unit structure via RBAC
3. **Implement data filtering** by user's org unit scope
4. **Dashboard widgets** showing org-specific metrics
5. **Reports scoped** to user's organizational access

---

## 8. 🔍 Database Schema Summary

### RBAC Core Tables (✅ Complete)

```
APP_Users
├── UserID (PK)
├── Username (UNIQUE)
├── PasswordHash
├── IsActive
└── CreatedAt

APP_Roles
├── RoleID (PK)
├── RoleCode (UNIQUE)
├── RoleNameEn
└── RoleNameAr

APP_UserRoleScope
├── UserRoleScopeID (PK)
├── UserID (FK → APP_Users)
├── RoleID (FK → APP_Roles)
├── OrgUnitID
└── OrgUnitType
```

### Missing Org Unit Tables (❌ Not Implemented)

```
APP_Sections [MISSING]
APP_Departments [MISSING]
APP_Administrations [MISSING]
```

### Legacy Tables (⚠️ Exists but needs integration)

```
AdminsrationUnit (typo in name)
VW_AdminstrationUnit (view)
IncidentRequest (has SectionID, DepartmentID)
```

---

## 9. ✅ Conclusion

### What's Working

✅ **RBAC infrastructure is complete and functional**
✅ **6 test users can login successfully**
✅ **Session-based authentication works**
✅ **Role-based authorization guards implemented**
✅ **Password hashing system ready (bcrypt)**

### What's Test-Only

⚠️ **ALL 6 user accounts are artificial test data**
⚠️ **Org unit IDs are arbitrary placeholders**
⚠️ **Passwords use temporary plain-text format**
⚠️ **No real hospital staff or org structure**

### What's Needed for Production

❌ **Real hospital employee data import**
❌ **Organizational unit master tables**
❌ **Proper bcrypt password hashes**
❌ **User management admin interface**
❌ **Audit logging and security features**
❌ **Integration with HR systems**

---

## 📞 Next Steps

1. **Decide:** Manual vs automated user provisioning?
2. **Design:** Org unit table schema (reuse legacy or new?)
3. **Implement:** User management API endpoints
4. **Build:** Admin frontend for user CRUD
5. **Test:** Pilot with small group (5-10 real users)
6. **Rollout:** Migrate all hospital staff

---

**Report End** | Generated: January 29, 2026 | Status: Test Environment
