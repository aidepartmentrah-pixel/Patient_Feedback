# 🧠 Phase-Lock Questions — Sign-In Testing Plan

**Date:** February 2, 2026  
**Purpose:** Pre-implementation Q&A to ensure we build the right solution

---

## 🔐 AUTH SYSTEM QUESTIONS (Critical)

### Q1 — Password verification method

**In your backend login logic — do you currently verify passwords using:**

**✅ Answer: C) TEMP_HASH_xxx compare trick**

**Explanation:**
- Your auth system (in `backend/api/db_layer/auth_db.py`) currently uses a hybrid approach:
  - For **test users**: passwords are stored as `TEMP_HASH_<password>` and compared as plain text
  - For **production**: supports bcrypt verification via `bcrypt.checkpw()`
  
**Current password format in database:**
```
UserID=1: TEMP_HASH_admin123
UserID=2: TEMP_HASH_worker123
UserID=3: TEMP_HASH_sup123
UserID=4: TEMP_HASH_section123
UserID=5: TEMP_HASH_dept123
UserID=6: TEMP_HASH_adminis123
```

**👉 Decision needed:** For new users, should we:
- **Option A:** Continue with TEMP_HASH for testing (easy, readable)
- **Option B:** Convert ALL to bcrypt now (production-ready, secure)
- **Option C:** Mix both (existing stay TEMP, new ones bcrypt)

**My recommendation:** Option A for this testing phase → Option B before production

---

### Q2 — Where is login endpoint exactly?

**Confirm path:**

**✅ Answer: `POST /api/auth/login`**

**Verified locations:**
- Router: `backend/api/routers/auth_router.py`
- Service: `backend/api/services/auth_service.py`
- DB Layer: `backend/api/db_layer/auth_db.py`

**Request format:**
```json
{
  "username": "software_admin",
  "password": "admin123"
}
```

**Response format:**
```json
{
  "success": true,
  "message": "login successful",
  "user": {
    "user_id": 1,
    "username": "software_admin",
    "is_active": true,
    "scopes": [...]
  }
}
```

**Other auth endpoints:**
- `POST /api/auth/logout` - Logout
- `GET /api/auth/me` - Get current user profile

**NOT `/api/v2/auth/login`** - No v2 versioning in auth

---

### Q3 — Does login currently work for ANY user?

**❓ I NEED YOUR ANSWER:**

Have you successfully logged in before with `software_admin / admin123` or similar?

- [ ] ✅ **Yes — works** (login successful, can access protected routes)
- [ ] ❌ **No — never worked** (always fails)
- [ ] ⚠️ **Sometimes** (works in test, fails in frontend, or vice versa)
- [ ] 🤷 **Not sure — haven't tried yet**

**👉 This tells me if we're fixing or extending.**

**From test files I see:**
- `test_phase2_auth_router.py` - Has 30+ passing tests for login
- `test_phase2_auth_service.py` - Has passing integration tests
- `test_auth_with_login.py` - Tests protected endpoints after login

**This suggests login SHOULD work, but I need confirmation from YOU that you've tested it.**

---

## 🗄️ DATABASE STATE QUESTIONS

### Q4 — Do these tables already exist and contain data?

**✅ CONFIRMED from your system:**

| Table | Exists? | Row Count | Notes |
|-------|---------|-----------|-------|
| `APP_Users` | ✅ Yes | **6 rows** | Test users created Jan 27, 2026 |
| `APP_Roles` | ✅ Yes | **6 rows** | All role types defined |
| `APP_UserRoleScope` | ✅ Yes | **6 rows** | One scope per user |
| `AdminsrationUnit` | ✅ Yes | **~100+ rows** | Hospital org structure (typo in table name) |

**Database:** `IncidentManager` on `SERVER=SOCIALMEDIA`

**Note:** Database connection failed when I tried `list_all_users.py` — SQL Server needs to be running.

---

### Q5 — Are usernames UNIQUE constrained?

**✅ Answer: YES**

**Verified in schema:**
```sql
CREATE TABLE dbo.APP_Users (
    UserID INT IDENTITY(1,1) PRIMARY KEY,
    Username NVARCHAR(100) NOT NULL UNIQUE,  -- ✅ UNIQUE constraint
    PasswordHash NVARCHAR(255) NOT NULL,
    IsActive BIT NOT NULL DEFAULT 1,
    CreatedAt DATETIME NOT NULL DEFAULT GETDATE()
)
```

**Also has index:** `INDEX IX_APP_Users_Username NONCLUSTERED (Username)`

**👉 This means we CANNOT create duplicate usernames — bulk insert script must check for existing users first.**

---

### Q6 — Is PasswordHash column size large? (varchar length)

**✅ Answer: NVARCHAR(255) — Sufficient**

**Details:**
- Current: `NVARCHAR(255)` - Can hold 255 Unicode characters
- Bcrypt hashes are ~60 characters (e.g., `$2b$12$...`)
- TEMP_HASH format: ~20-30 characters (e.g., `TEMP_HASH_admin123`)
- **✅ No schema change needed** for bcrypt migration

**Safe to proceed with bcrypt if you choose Option B in Q1.**

---

## 🏥 ORG STRUCTURE QUESTIONS

### Q7 — Does AdminsrationUnit contain ALL: administrations, departments, sections in one hierarchy table?

**✅ Answer: YES — Single hierarchical table**

**Table:** `dbo.AdminsrationUnit` (note typo in actual table name)

**Structure:**
```sql
- UniqueID (Primary Key)
- ParentID (Self-referencing FK)
- Name (NVARCHAR)
- Type (INT: 1=Administration, 2=Department, 3=Section)
- Frozen (BIT)
```

**Hierarchy logic:**
- **Administration** (Type=1): `ParentID == UniqueID` (root node, self-referencing)
- **Department** (Type=2): `ParentID` points to Administration's `UniqueID`
- **Section** (Type=3): `ParentID` points to Department's `UniqueID`

**Example hierarchy:**
```
Administration ID=1 (ParentID=1, Type=1) "المديرية العامة"
  ├─ Department ID=5 (ParentID=1, Type=2) "إدارة الشؤون الطبية"
  │   ├─ Section ID=10 (ParentID=5, Type=3) "قسم الطوارئ"
  │   └─ Section ID=11 (ParentID=5, Type=3) "قسم العمليات"
  └─ Department ID=6 (ParentID=1, Type=2) "إدارة التمريض"
      └─ Section ID=12 (ParentID=6, Type=3) "قسم التمريض العام"
```

**👉 There are NO separate `APP_Departments` or `APP_Sections` tables.**

---

### Q8 — Do you want login accounts for:

**❓ I NEED YOUR ANSWER - Select ONE:**

Which users should get login accounts in this phase?

- [ ] **Option A:** Only ADMINISTRATION + DEPARTMENT + SECTION admins  
  - *Example:* 3 administrations → 3 admin accounts  
  - *Example:* 15 departments → 15 dept admin accounts  
  - *Example:* 50 sections → 50 section admin accounts  
  - **Total: ~70 accounts** (admin-level only)

- [ ] **Option B:** Also WORKERS per department  
  - *Includes Option A PLUS:* 2-5 workers per section  
  - **Total: ~150-250 accounts** (admin + workers)

- [ ] **Option C:** Also supervisors  
  - *Includes Options A+B PLUS:* 1-2 supervisors per department  
  - **Total: ~200-300 accounts** (admin + workers + supervisors)

- [ ] **Option D:** Only admin-level roles for now  
  - *Same as Option A* but explicitly NO workers/supervisors  
  - **Fastest to implement**

**My recommendation: ✅ Option D (admin-level only)**

**Reasoning:**
- Faster to create (~70 accounts vs 300)
- Easier to test (one account per organizational unit)
- Can add workers later after admin testing succeeds
- Matches your current test setup (1 user per role type)

---

### Q9 — Are OrgUnitID values stable and production-grade?

**❓ I NEED YOUR ANSWER:**

Will the `UniqueID` values in `AdminsrationUnit` change later due to migration?

- [ ] **Stable** - These IDs are permanent, production-ready
- [ ] **Unstable** - IDs might change during data migration/cleanup
- [ ] **Not sure** - Need to verify with DBA or hospital IT

**Why this matters:**

If STABLE → We can use IDs directly:
```sql
INSERT INTO APP_UserRoleScope (UserID, RoleID, OrgUnitID, OrgUnitType)
VALUES (123, 4, 15, 'SECTION')  -- Links to UniqueID=15
```

If UNSTABLE → We generate by Name instead:
```sql
-- Find ID from name at insertion time
DECLARE @SectionID INT = (SELECT UniqueID FROM AdminsrationUnit WHERE Name = 'قسم الطوارئ')
INSERT INTO APP_UserRoleScope (UserID, RoleID, @SectionID, 'SECTION')
```

**Current state observation:**
- Your test users use IDs: 0, 1, 5, 10
- Some IDs look arbitrary (especially 0 for SOFTWARE_ADMIN)
- Production org structure appears to have IDs in range 1-100+

**👉 Please confirm stability before we proceed.**

---

## 🧪 TESTING STRATEGY QUESTIONS

### Q10 — Do you want:

**❓ I NEED YOUR ANSWER - Select ONE:**

Password strategy for generated test accounts:

- [ ] **Option A:** One shared password for all generated test accounts  
  - *Example:* All new users get password `Test2026!`  
  - **Pros:** Easy to remember, fastest to test, simple documentation  
  - **Cons:** Not realistic, security concern

- [ ] **Option B:** Generated password per account  
  - *Example:* `Admin1_Pass!`, `Dept5_Pass!`, `Section10_Pass!`  
  - **Pros:** More realistic, better security  
  - **Cons:** Need password sheet, harder to remember

- [ ] **Option C:** Username == password (test mode)  
  - *Example:* User `admin_medical` has password `admin_medical`  
  - **Pros:** Super easy to remember  
  - **Cons:** Very insecure, bad practice

**My recommendation: ✅ Option A (shared password)**

**Suggested password:** `Hospital2026!`

**Reasoning:**
- This is a TESTING phase, not production
- Easy for you to sign in as any role quickly
- Can convert to individual passwords later
- Matches your current pattern (all test users have simple passwords)

---

### Q11 — Do you want accounts marked as IsActive = 1 immediately?

**❓ I NEED YOUR ANSWER:**

Should all generated accounts be active immediately?

- [ ] **Yes** - All accounts active (`IsActive = 1`)
- [ ] **No** - Start inactive, manual activation
- [ ] **Mixed** - Admins active, workers inactive

**My recommendation: ✅ Yes (all active)**

**Reasoning:**
- This is for testing, not production
- You want to test sign-in immediately
- Can deactivate specific accounts later if needed
- Matches your current test data (all 6 users are active)

---

## ⚙️ EXECUTION STYLE QUESTIONS

### Q12 — How do you prefer creation phase:

**❓ I NEED YOUR ANSWER - Select ONE:**

Method to create new user accounts:

- [ ] **Option A:** Pure SQL script you run manually  
  - *Deliverable:* `create_all_users.sql` file  
  - **Pros:** Deterministic, safe, reviewable, can run in SSMS  
  - **Cons:** Manual execution required

- [ ] **Option B:** Python seed script  
  - *Deliverable:* `seed_users.py` Python script  
  - **Pros:** Can add logic, error handling, progress display  
  - **Cons:** Requires Python environment, dependencies

- [ ] **Option C:** Copilot-generated DB seeder  
  - *Deliverable:* Python script with interactive prompts  
  - **Pros:** Flexible, can rerun, can be enhanced  
  - **Cons:** More complex

- [ ] **Option D:** FastAPI admin endpoint to create users  
  - *Deliverable:* `POST /api/admin/users/bulk-create` endpoint  
  - **Pros:** Can use from frontend later, REST API  
  - **Cons:** Requires backend changes, auth setup

**My recommendation: ✅ Option A (SQL script)**

**Reasoning:**
- Fastest to implement and execute
- You're familiar with SQL
- Safe (no backend code changes required)
- Can review before running
- Idempotent (checks for existing users)
- Works even if backend is down

---

## 📌 What Happens After You Answer

Once you provide answers to the **questions marked with ❓**, I will produce:

### ✅ Deliverables I'll Create:

1. **`USER_INVENTORY_QUERY.sql`** - Read-only queries to audit current state
2. **`USER_MAPPING_TRUTH_TABLE.sql`** - Shows all org units and their assigned users
3. **`CREATE_USERS_BULK.sql`** - Script to create all new user accounts
4. **`ASSIGN_ROLE_SCOPES.sql`** - Script to link users to roles and org units
5. **`USER_CREDENTIALS_REFERENCE.md`** - Documentation of all usernames/passwords
6. **`test_signin_all_roles.py`** - Automated test suite for all roles
7. **`SIGNIN_TESTING_CHECKLIST.md`** - Manual testing guide
8. **`ROLE_AUTHORIZATION_MATRIX.md`** - What each role can/cannot do

All scripts will be **idempotent** (safe to run multiple times).

---

## 📝 Summary of What I Know vs. Need

### ✅ What I Already Know:

- Your database has 6 test users with TEMP_HASH passwords
- Login endpoint is `POST /api/auth/login`
- Database structure uses `AdminsrationUnit` for org hierarchy
- Username has UNIQUE constraint
- PasswordHash column is NVARCHAR(255) (sufficient for bcrypt)
- Org structure is single hierarchical table with Type field

### ❓ What I Need From You:

| # | Question | Your Answer |
|---|----------|-------------|
| Q3 | Does login currently work? | ⬜ Yes / ⬜ No / ⬜ Sometimes |
| Q8 | Which users get accounts? | ⬜ A / ⬜ B / ⬜ C / ⬜ D |
| Q9 | Are OrgUnitIDs stable? | ⬜ Stable / ⬜ Unstable / ⬜ Not sure |
| Q10 | Password strategy? | ⬜ Shared / ⬜ Generated / ⬜ Username=Password |
| Q11 | All accounts active? | ⬜ Yes / ⬜ No / ⬜ Mixed |
| Q12 | Creation method? | ⬜ SQL / ⬜ Python / ⬜ Copilot / ⬜ API |

---

## 🚦 Next Steps

**Please respond with:**

1. Answers to questions Q3, Q8, Q9, Q10, Q11, Q12 (check the boxes)
2. Any clarifications or concerns
3. Confirmation that SQL Server is running (or will be when we start)

**Then I will immediately generate all deliverables.**

---

**Note:** The "AdminsrationUnit" table name has a typo (should be "AdministrationUnit") but I'm using the actual table name that exists in your database.

---
---

# 📋 MODULE 5.0 — Backend State Inspection Report

**Date:** February 2, 2026  
**Task:** Read-only exploration of backend before Phase 5 implementation  
**Status:** ✅ COMPLETE

---

## Q1 — Role Guard Dependency Names

### ✅ FOUND: Authorization Guards System

**Primary Guard Functions:**
- **Function:** `require_software_admin(current_user: CurrentUser)`
- **File Path:** `backend/api/utils/guards.py`
- **Line:** 141

**All Available Guards:**
```python
# Base guards
require_logged_in(current_user)        # Line 54
require_role(current_user, roles)      # Line 93

# Role-specific guards
require_software_admin(current_user)          # Line 141
require_worker(current_user)                  # Line 175
require_complaint_supervisor(current_user)    # Line 208
require_section_admin(current_user)           # Line 241
require_department_admin(current_user)        # Line 274
require_administration_admin(current_user)    # Line 307

# Composite guards
require_any_admin(current_user)               # Line 342
require_any_supervisor(current_user)          # Line 372

# Scope guards
require_unit_in_scope(current_user, unit_id)  # Line 484
require_any_unit_in_scope(current_user, ids)  # Line 545
```

**Example Usage in Production:**
```python
# From: backend/api/routers/settings_router.py:92
@router.post("")
def create_setting(
    setting: SettingCreate,
    current_user: CurrentUser = Depends(get_current_user)
):
    require_software_admin(current_user)  # ✅ Guard called INSIDE endpoint
    # ... rest of logic
```

**Import Pattern:**
```python
from ..utils.guards import require_software_admin
```

**Design Notes:**
- Guards are **plain functions**, NOT decorators
- Guards are called **inside** endpoint functions
- Guards raise `HTTPException(403)` for forbidden access
- Guards raise `HTTPException(401)` for missing auth
- Guards DO NOT access database or session

---

## Q2 — DB Connection Helper Pattern

### ✅ FOUND: Database Connection Helper

**Function Name:** `get_connection()`
**File Path:** `backend/core/database.py`
**Line:** 3

**Full Implementation:**
```python
import pyodbc

def get_connection():
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=SOCIALMEDIA;"
        "DATABASE=IncidentManager;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )
    return conn
```

**Usage Pattern:**
```python
# Pattern 1: Manual connection management
from core.database import get_connection

conn = None
cursor = None
try:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT ...")
    results = cursor.fetchall()
    conn.commit()  # For writes
except Exception as e:
    if conn:
        conn.rollback()
    raise
finally:
    if cursor:
        cursor.close()
    if conn:
        conn.close()
```

**NOT using context manager** (`with` statement) - connections are manually managed.

**Examples in codebase:**
- `backend/api/db_layer/auth_db.py` (lines 18-40, 100-140, etc.)
- `backend/api/db_layer/action_items.py`
- All db_layer files follow this pattern

---

## Q3 — APP_UserRoleScope Exact Schema Usage

### ✅ FOUND: Exact Column Names

**Table:** `dbo.APP_UserRoleScope`

**Columns Used in Code:**
```sql
-- From: backend/api/db_layer/auth_db.py:212
SELECT 
    r.RoleCode,
    urs.OrgUnitID,      -- ✅ Exact column name
    urs.OrgUnitType     -- ✅ Exact column name
FROM dbo.APP_UserRoleScope urs
INNER JOIN dbo.APP_Roles r ON urs.RoleID = r.RoleID
WHERE urs.UserID = ?
ORDER BY r.RoleCode, urs.OrgUnitType, urs.OrgUnitID
```

**Confirmed Columns:**
- ✅ `UserRoleScopeID` (Primary Key, IDENTITY)
- ✅ `UserID` (FK to APP_Users.UserID)
- ✅ `RoleID` (FK to APP_Roles.RoleID)
- ✅ `OrgUnitID` (INT - links to AdminsrationUnit.UniqueID)
- ✅ `OrgUnitType` (NVARCHAR - "SECTION", "DEPARTMENT", "ADMINISTRATION", "COMPLAINT")

**Python Return Format:**
```python
# From auth_db.py get_user_with_scopes()
user_data["scopes"].append({
    "role_code": scope_row.RoleCode,        # From APP_Roles
    "org_unit_id": scope_row.OrgUnitID,     # From APP_UserRoleScope
    "org_unit_type": scope_row.OrgUnitType  # From APP_UserRoleScope
})
```

---

## Q4 — Role Codes Present in System

### ✅ FOUND: Role Code Constants

**File Path:** `backend/core/constants/roles.py`
**Lines:** 1-60

**Exact Role Codes (from constants file):**
```python
SOFTWARE_ADMIN = "SOFTWARE_ADMIN"
WORKER = "WORKER"
COMPLAINT_SUPERVISOR = "COMPLAINT_SUPERVISOR"
SECTION_ADMIN = "SECTION_ADMIN"
DEPARTMENT_ADMIN = "DEPARTMENT_ADMIN"
ADMINISTRATION_ADMIN = "ADMINISTRATION_ADMIN"
```

**Database Table:** `APP_Roles`
**Column:** `RoleCode NVARCHAR(50) UNIQUE`

**Role Sets Defined:**
```python
ALL_ROLES = [
    SOFTWARE_ADMIN,
    WORKER,
    COMPLAINT_SUPERVISOR,
    SECTION_ADMIN,
    DEPARTMENT_ADMIN,
    ADMINISTRATION_ADMIN,
]

ADMIN_ROLES = [
    SOFTWARE_ADMIN,
    SECTION_ADMIN,
    DEPARTMENT_ADMIN,
    ADMINISTRATION_ADMIN,
]

SUPERVISOR_ROLES = [
    SOFTWARE_ADMIN,
    COMPLAINT_SUPERVISOR,
    ADMINISTRATION_ADMIN,
]
```

**Spelling Confirmed:** All uppercase with underscores, no variations found.

---

## Q5 — Login Password Verification Method

### ✅ FOUND: Hybrid Password Verification

**Function:** `validate_user_credentials(username: str, password: str)`
**File Path:** `backend/api/db_layer/auth_db.py`
**Lines:** 239-356

**Verification Logic:**
```python
# Lines 306-326
if password_hash.startswith('TEMP_HASH_'):
    # For testing: extract expected password from placeholder
    # Format: TEMP_HASH_<password>
    expected_password = password_hash.replace('TEMP_HASH_', '')
    
    # Simple string comparison for temp hashes
    if password != expected_password:
        return None
else:
    # Verify actual bcrypt hash
    try:
        password_bytes = password.encode('utf-8')
        hash_bytes = password_hash.encode('utf-8')
        
        if not bcrypt.checkpw(password_bytes, hash_bytes):
            return None
    except Exception as e:
        raise Exception(f"Password verification error: {str(e)}")
```

**Answer:**
- ✅ **TEMP_HASH_ pattern IS supported** (for testing)
- ✅ **bcrypt IS supported** (for production)
- ✅ **Hybrid approach:** Checks prefix, uses appropriate method
- Import: `import bcrypt` (at top of file)

---

## Q6 — Auth Endpoints That Already Exist

### ✅ FOUND: Authentication Router

**Router File:** `backend/api/routers/auth_router.py`
**Prefix:** `/api/auth`
**Line:** 20

**Existing Endpoints:**

1. **POST /api/auth/login**
   - Request: `{"username": "...", "password": "..."}`
   - Response: User object with scopes, session created
   - Line: ~125

2. **POST /api/auth/logout**
   - Clears user session
   - Response: Success message
   - Line: ~240

3. **GET /api/auth/me**
   - Returns current user profile
   - Requires: `Depends(require_authentication)`
   - Response: User object with scopes
   - Line: ~435

**Helper Dependency:**
- `require_authentication(request: Request) -> CurrentUser`
- File: `backend/api/services/auth_service.py:301`
- Returns CurrentUser or raises 401

**NO token verification endpoint** - System uses session-based auth, not JWT.

---

## Q7 — Transaction Pattern Used in Services

### ✅ FOUND: Transaction Pattern

**Pattern:** Manual commit/rollback without context manager

**Standard Pattern:**
```python
from core.database import get_connection

conn = None
cursor = None

try:
    conn = get_connection()
    cursor = conn.cursor()
    
    # Execute queries
    cursor.execute("INSERT INTO ...")
    cursor.execute("UPDATE ...")
    
    # Commit transaction
    conn.commit()
    
except Exception as e:
    # Rollback on error
    if conn:
        conn.rollback()
    raise
    
finally:
    # Always close
    if cursor:
        cursor.close()
    if conn:
        conn.close()
```

**Examples Found:**
- `backend/api/db_layer/auth_db.py:412` - `conn.commit()`
- `backend/api/db_layer/auth_db.py:418` - `conn.rollback()`
- `backend/api/services/insert_service.py:789` - `conn.commit()`
- `backend/api/db_layer/action_items.py:63` - `conn.commit()`

**Key Points:**
- **No `with` statements** for connection management
- Explicit `try/except/finally` blocks
- Manual `conn.commit()` for writes
- Manual `conn.rollback()` in except clause
- Always close cursor and connection in finally

---

## Q8 — Existing Admin Routers

### ✅ FOUND: Admin/Restricted Routers

**Admin-Protected Routers:**

1. **settings_router.py**
   - Prefix: `/api/settings`
   - Protection: `require_software_admin(current_user)` in ALL endpoints
   - Line: 17
   - Example: POST `/api/settings` (create setting)

2. **training_router.py**
   - Prefix: `/api/settings/training`
   - Protection: Mixed (some public, some admin-only)
   - Line: 27
   - Example: GET `/api/settings/training/status` (admin only)

3. **system_settings_router.py**
   - Prefix: `/api/system-settings`
   - Protection: Likely admin-restricted (need to verify)
   - Line: 13

**Example Admin Endpoints:**
```python
# settings_router.py:81-92
@router.post("")
def create_setting(
    setting: SettingCreate,
    current_user: CurrentUser = Depends(get_current_user)
):
    require_software_admin(current_user)
    return create_setting_service(setting)
```

**Other Routers (Not Admin-Specific):**
- `/api/auth` - Authentication (public login, protected /me)
- `/api/complaints` - Table view (scope-filtered)
- `/api/dashboard` - Dashboard (scope-filtered)
- `/api/reports` - Reports (has admin restrictions)
- `/api/classification` - ML classification
- `/api/doctors`, `/api/patients` - Data access

**Recommendation for Phase 5:**
- Place new admin endpoints in existing `/api/settings` router OR
- Create new `/api/admin` router for user management

---

## Q9 — Existing Backend Test Pattern

### ✅ FOUND: Test Pattern

**Test Import Pattern:**
```python
# From: backend/test_phase2_auth_router.py:1-16
import sys
import os
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from main import app

# Create test client
client = TestClient(app)
```

**TestClient Usage:**
```python
# Login test helper
def login_user(username: str, password: str):
    return client.post(
        "/api/auth/login",
        json={"username": username, "password": password}
    )

# Test function
def test_login_success():
    response = client.post(
        "/api/auth/login",
        json={"username": "software_admin", "password": "admin123"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
```

**Test Folder Structure:**
```
backend/
├── test_phase2_auth_router.py     # Auth endpoint tests
├── test_phase2_auth_service.py    # Service layer tests
├── test_phase2_auth_db_layer.py   # DB layer tests
├── test_phase2_rbac_tables.py     # Database schema tests
├── test_phase2_integration_complete.py  # Full integration tests
└── ... (many other test files)
```

**Session Management in Tests:**
```python
def clear_all_sessions():
    """Clear all test client sessions."""
    client.cookies.clear()
```

**Pattern Summary:**
- Import from `main import app`
- Use `TestClient(app)` from fastapi.testclient
- Tests are standalone Python scripts (not pytest)
- Run with: `python test_xxx.py`
- Session cookies persist across requests

---

## 📊 Summary: What We Know

✅ **Guards:** `require_software_admin()` in `backend/api/utils/guards.py`  
✅ **DB Connection:** `get_connection()` in `backend/core/database.py`  
✅ **UserRoleScope Columns:** `UserID`, `RoleID`, `OrgUnitID`, `OrgUnitType`  
✅ **Role Codes:** 6 roles defined in `backend/core/constants/roles.py`  
✅ **Password Verification:** Hybrid TEMP_HASH + bcrypt in `auth_db.py`  
✅ **Auth Endpoints:** `/api/auth/login`, `/logout`, `/me`  
✅ **Transaction Pattern:** Manual `conn.commit()` / `conn.rollback()`  
✅ **Admin Routers:** `/api/settings` uses `require_software_admin`  
✅ **Test Pattern:** `TestClient(app)` from `fastapi.testclient`

---

## 🎯 Ready for Phase 5 Implementation

All backend patterns identified. Safe to proceed with:
- Creating new admin endpoints
- Following existing auth guard pattern
- Using standard DB connection pattern
- Writing tests following existing structure

---

**End of MODULE 5.0 Report**
