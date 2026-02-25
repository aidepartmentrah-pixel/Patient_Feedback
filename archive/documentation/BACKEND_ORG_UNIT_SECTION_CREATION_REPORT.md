# 📋 Backend Org Unit & Section Creation - Documentation Report

**Generated:** February 5, 2026  
**Purpose:** Document REAL current endpoints and services for org unit creation, section admin creation, and org hierarchy

---

## 1. SECTION CREATION ENDPOINTS

### 1.1 Create Section with Admin User

**Router File:** [admin_section_router.py](backend/api/routers/admin_section_router.py)

**Full Route Path:** `/api/admin/create-section-with-admin`

**HTTP Method:** `POST`

**Request Model:** `CreateSectionRequest`
```python
class CreateSectionRequest(BaseModel):
    section_name: str
    parent_department_id: int
```

**Response Model:** `CreateSectionResponse`
```python
class CreateSectionResponse(BaseModel):
    section_id: int
    username: str
    password: str
```

**Permission Guards:**
- `get_current_user` (authentication required)
- `require_software_admin(current_user)` (SOFTWARE_ADMIN role required)

**Service Function Called:** `create_section_with_admin(section_name, parent_department_id)`

**Request Example:**
```json
POST /api/admin/create-section-with-admin
{
    "section_name": "Emergency Department",
    "parent_department_id": 5
}
```

**Response Example:**
```json
{
    "section_id": 101,
    "username": "sec_101_admin",
    "password": "Hospital2026!"
}
```

---

### 1.2 Recreate Section Admin User

**Router File:** [admin_section_admin_recreate_router.py](backend/api/routers/admin_section_admin_recreate_router.py)

**Full Route Path:** `/api/admin/sections/{section_id}/recreate-admin`

**HTTP Method:** `POST`

**Request Parameters:**
- `section_id` (path parameter, int)

**Response Model:** Dictionary
```python
{
    "section_id": int,
    "username": str,
    "password": str
}
```

**Permission Guards:**
- `get_current_user` (authentication required)
- `require_software_admin(current_user)` (SOFTWARE_ADMIN role required)

**Service Function Called:** `recreate_section_admin_service(section_id)`

**Request Example:**
```json
POST /api/admin/sections/10/recreate-admin
```

**Response Example:**
```json
{
    "section_id": 10,
    "username": "sec_10_admin_v2",
    "password": "Hospital2026!"
}
```

**Note:** Creates ADDITIONAL admin user with versioned username (v2, v3, etc.). Does NOT delete existing section admins.

---

## 2. SECTION CREATION SERVICE LOGIC

### 2.1 Create Section with Admin Service

**Service File:** [section_admin_creator_service.py](backend/api/services/section_admin_creator_service.py)

**Function Name:** `create_section_with_admin`

**Function Signature:**
```python
def create_section_with_admin(section_name: str, parent_department_id: int) -> Dict[str, Any]
```

**Input Parameters:**
- `section_name: str` - Name of the new section
- `parent_department_id: int` - Parent department's UniqueID

**What It Creates:**

1. **Org Unit Row** (AdminsrationUnit table)
   - Name: `section_name`
   - ParentID: `parent_department_id`
   - Type: `324` (SECTION)
   - Frozen: `0`
   - CreateDate: `GETDATE()`

2. **User** (APP_Users table)
   - Username: `sec_{section_id}_admin`
   - PasswordHash: `TEMP_HASH_Hospital2026!`
   - IsActive: `1`
   - CreatedAt: `GETDATE()`
   - DisplayName: `username` (fallback)
   - DepartmentDisplayName: `NULL`

3. **User Scope** (APP_UserRoleScope table)
   - UserID: (newly created user)
   - RoleID: (resolved from 'SECTION_ADMIN' role code)
   - OrgUnitID: `section_id`
   - OrgUnitType: `'SECTION'` (string)

**Transaction Behavior:**
- Opens connection with `get_connection()`
- Performs all inserts within single transaction
- Commits on success
- Rolls back on error
- Always closes connection in `finally` block

**Return Value:**
```python
{
    "section_id": int,
    "username": str,
    "temp_password": "Hospital2026!"
}
```

---

### 2.2 Recreate Section Admin Service

**Service File:** [section_admin_recreate_service.py](backend/api/services/section_admin_recreate_service.py)

**Function Name:** `recreate_section_admin_service`

**Function Signature:**
```python
def recreate_section_admin_service(section_id: int) -> Dict[str, Any]
```

**Input Parameters:**
- `section_id: int` - ID of existing section

**Process:**
1. **Verify Section Exists** - Query AdminsrationUnit for section_id
2. **Validate Type = 324** - Ensure org unit is a SECTION (not Administration/Department)
3. **Generate Unique Username** - Try `sec_{id}_admin`, then `sec_{id}_admin_v2`, `v3`, etc.
4. **Create User** - Insert into APP_Users with TEMP_HASH password
5. **Assign SECTION_ADMIN Role** - Insert into APP_UserRoleScope
6. **Commit Transaction**

**What It Creates:**

1. **User** (APP_Users table)
   - Username: `sec_{section_id}_admin` or versioned (v2, v3)
   - PasswordHash: `TEMP_HASH_Hospital2026!`
   - IsActive: `1`
   - CreatedAt: `GETDATE()`
   - DisplayName: `username`
   - DepartmentDisplayName: `NULL`

2. **User Scope** (APP_UserRoleScope table)
   - UserID: (newly created user)
   - RoleID: (resolved from 'SECTION_ADMIN' role code)
   - OrgUnitID: `section_id`
   - OrgUnitType: `'SECTION'`

**Transaction Behavior:**
- Same pattern as create service
- Transaction-safe with commit/rollback

**Return Value:**
```python
{
    "section_id": int,
    "username": str,
    "temp_password": "Hospital2026!"
}
```

---

## 3. DB LAYER INSERT FUNCTIONS

### 3.1 Insert Section

**DB File:** [section_admin_creator_db.py](backend/api/db_layer/section_admin_creator_db.py)

**Function Signature:**
```python
def insert_section(conn, name: str, parent_department_id: int) -> int
```

**Required Fields:**
- `Name`: str
- `ParentID`: int (parent department UniqueID)
- `Type`: 324 (hardcoded)
- `Frozen`: 0 (hardcoded)
- `CreateDate`: GETDATE() (auto)

**SQL:**
```sql
INSERT INTO dbo.AdminsrationUnit (Name, ParentID, Type, Frozen, CreateDate)
OUTPUT INSERTED.UniqueID
VALUES (?, ?, 324, 0, GETDATE())
```

**Returns:** `int` - New section's UniqueID

**Transaction:** Does NOT commit - caller controls transaction

---

### 3.2 Insert User

**DB File:** [section_admin_creator_db.py](backend/api/db_layer/section_admin_creator_db.py)

**Function Signature:**
```python
def insert_user(conn, username: str, display_name: str = None, department_display_name: str = None) -> int
```

**Required Fields:**
- `Username`: str (must be unique)
- `PasswordHash`: 'TEMP_HASH_Hospital2026!' (hardcoded)
- `IsActive`: 1 (hardcoded)
- `CreatedAt`: GETDATE() (auto)
- `DisplayName`: str or username fallback
- `DepartmentDisplayName`: str or NULL

**SQL:**
```sql
INSERT INTO dbo.APP_Users (Username, PasswordHash, IsActive, CreatedAt, DisplayName, DepartmentDisplayName)
OUTPUT INSERTED.UserID
VALUES (?, 'TEMP_HASH_Hospital2026!', 1, GETDATE(), ?, ?)
```

**Returns:** `int` - New user's UserID

**Transaction:** Does NOT commit - caller controls transaction

**Validation:** Checks for duplicate username before insert

---

### 3.3 Insert User Scope

**DB File:** [section_admin_creator_db.py](backend/api/db_layer/section_admin_creator_db.py)

**Function Signature:**
```python
def insert_user_scope(conn, user_id: int, role_code: str, org_unit_id: int) -> None
```

**Required Fields:**
- `UserID`: int
- `RoleID`: int (resolved from role_code)
- `OrgUnitID`: int
- `OrgUnitType`: 'SECTION' (hardcoded string)

**SQL:**
```sql
-- Step 1: Resolve RoleID
SELECT RoleID FROM dbo.APP_Roles WHERE RoleCode = ?

-- Step 2: Insert scope
INSERT INTO dbo.APP_UserRoleScope (UserID, RoleID, OrgUnitID, OrgUnitType)
VALUES (?, ?, ?, 'SECTION')
```

**Returns:** None

**Transaction:** Does NOT commit - caller controls transaction

---

### 3.4 Additional DB Functions (Recreate Flow)

**DB File:** [section_admin_recreate_db.py](backend/api/db_layer/section_admin_recreate_db.py)

**Function: get_section**
```python
def get_section(conn, section_id: int) -> Optional[Any]
```
- Queries: `AdminsrationUnit` table
- Returns: Row with `UniqueID`, `Name`, `Type` or `None`

**Function: username_exists**
```python
def username_exists(conn, username: str) -> bool
```
- Queries: `APP_Users` table
- Returns: `True` if username exists, `False` otherwise

---

## 4. ORG UNIT TYPE SYSTEM

### 4.1 Storage Format

**Database Column:** `AdminsrationUnit.Type`

**Data Type:** `INT` (integer)

### 4.2 Type Values

Based on code inspection and SQL queries:

| Type Value | Org Unit Level    | Description                    |
|------------|-------------------|--------------------------------|
| **323**    | ADMINISTRATION    | Top-level organizational unit  |
| **324**    | SECTION           | Leaf-level unit (Qism)         |
| **325**    | DEPARTMENT        | Mid-level unit (Dayra)         |

**Evidence Sources:**
- [section_admin_creator_db.py](backend/api/db_layer/section_admin_creator_db.py): `Type = 324` for sections
- [admin_units.py](backend/api/db_layer/admin_units.py): `get_units_by_type(323|324|325)` function
- [DIAGNOSE_TARGET_DEPT_TYPES.py](backend/DIAGNOSE_TARGET_DEPT_TYPES.py): Type mapping documentation
- [verify_bulk_users.py](backend/verify_bulk_users.py): Type-to-string conversion CASE statements

### 4.3 String Representation

**In APP_UserRoleScope:** `OrgUnitType` is stored as **VARCHAR/STRING**, not integer

**Values:**
- `'ADMINISTRATION'`
- `'SECTION'`
- `'DEPARTMENT'`

**Example from code:**
```python
# section_admin_creator_db.py line 134-137
scope_query = """
    INSERT INTO dbo.APP_UserRoleScope (UserID, RoleID, OrgUnitID, OrgUnitType)
    VALUES (?, ?, ?, 'SECTION')
"""
```

### 4.4 Type Constants (No Enum File)

**Finding:** No centralized constants file or enum class exists for org unit types.

**Implementation Pattern:** Type values are **hardcoded directly in SQL queries and code**

**Observed Pattern:**
```python
# Hardcoded in insert functions
Type = 324  # Section

# Hardcoded in queries
WHERE Type = 323  # Administration
WHERE Type = 325  # Department
```

---

## 5. ORG TREE READ ENDPOINTS

### 5.1 Investigation Hierarchy Endpoint

**Router File:** [investigation_router.py](backend/api/routers/investigation_router.py)

**Route Path:** `/api/investigation/hierarchy`

**HTTP Method:** `GET`

**Response Structure:**
```json
{
    "administrations": [
        {"id": int, "name": str, "name_ar": str}
    ],
    "departments": [
        {"id": int, "name": str, "name_ar": str, "parent_administration_id": int}
    ],
    "sections": [
        {"id": int, "name": str, "name_ar": str, "parent_department_id": int}
    ]
}
```

**Nesting Format:** Flat arrays with parent ID references (not nested tree)

**Service Function:** `get_organizational_hierarchy()`

---

### 5.2 Dashboard Hierarchy Endpoint

**Router File:** [dashboard_router.py](backend/api/routers/dashboard_router.py)

**Route Path:** `/api/dashboard/hierarchy`

**HTTP Method:** `GET`

**Authentication:** Requires `get_current_user` dependency

**Response Structure:** Filtered by user's scope

**Service Function:** `get_dashboard_hierarchy(current_user)`

---

### 5.3 Investigation Tree Endpoint

**Router File:** [investigation_router.py](backend/api/routers/investigation_router.py)

**Route Path:** `/api/investigation/tree`

**HTTP Method:** `GET`

**Query Parameters:**
- `season: str` (required)
- `tree_type: Literal[...]` (required)
- `administration_id: int | None`
- `department_id: int | None`
- `section_id: int | None`

**Response Structure:** Nested tree with aggregated incident metrics

**Nesting Format:** Hierarchical tree structure with children arrays

**Service Function:** `get_investigation_tree(...)`

---

### 5.4 Org Tree Service (In-Memory)

**Service File:** [org_tree_service.py](backend/api/services/org_tree_service.py)

**Purpose:** Central org tree traversal service - loads full tree into memory

**Public API Functions:**

**1. get_full_tree()**
```python
def get_full_tree() -> list[dict]
```
Returns raw org unit tree as list of dicts:
```python
[
    {
        "UniqueID": int,
        "ParentID": int,
        "Type": int,
        "Name": str
    },
    ...
]
```

**2. get_descendants()**
```python
def get_descendants(root_id: int) -> set[int]
```
Returns set of root_id and all descendant org unit IDs (DFS traversal)

**3. get_ancestors()**
```python
def get_ancestors(node_id: int) -> set[int]
```
Returns set of ancestors up to root

**4. is_ancestor()**
```python
def is_ancestor(parent_id: int, child_id: int) -> bool
```
Returns True if parent_id is ancestor of child_id

---

## 6. ROUTER PREFIX REGISTRATION

**Main App File:** [main.py](backend/main.py)

### 6.1 Admin Section Router

**Import Statement:**
```python
from api.routers.admin_section_router import router as admin_section_router
```

**Include Router:**
```python
app.include_router(admin_section_router)
```

**Router Prefix (defined in router file):**
```python
router = APIRouter(
    prefix="/api/admin",
    tags=["admin-sections"]
)
```

**Mounted Prefix:** `/api/admin`

**Full Endpoint Path:** `/api/admin/create-section-with-admin`

---

### 6.2 Admin Section Admin Recreate Router

**Import Statement:**
```python
from api.routers.admin_section_admin_recreate_router import router as admin_section_admin_recreate_router
```

**Include Router:**
```python
app.include_router(admin_section_admin_recreate_router)
```

**Router Prefix (defined in router file):**
```python
router = APIRouter(
    prefix="/api/admin/sections",
    tags=["admin-sections"]
)
```

**Mounted Prefix:** `/api/admin/sections`

**Full Endpoint Path:** `/api/admin/sections/{section_id}/recreate-admin`

---

### 6.3 Settings Router

**Import Statement:**
```python
from api.routers.settings_router import router as settings_router
```

**Include Router:**
```python
app.include_router(settings_router)
```

**Router Prefix (defined in router file):**
```python
router = APIRouter(
    prefix="/api/settings",
    tags=["Settings"]
)
```

**Mounted Prefix:** `/api/settings`

**Full Endpoint Paths:**
- `/api/settings/departments` (GET, POST)
- `/api/settings/departments/{department_id}` (PUT, DELETE)
- `/api/settings/attributes` (GET)

**Note:** Settings router contains department CRUD but NOT section creation endpoints

---

## 7. GENERIC ORG UNIT CREATION

### Confirmation: NO Generic Org Unit Creation Endpoint Exists

**Finding:** There is **NO** generic org unit creation endpoint that accepts Type as a parameter.

**Evidence:**
- Searched all routers for POST endpoints with "unit" or "org" patterns
- No routes found matching `/api/units`, `/api/org-units`, or similar
- Settings router only handles departments (Type 325) via separate CRUD endpoints
- Section creation only available via specialized `/api/admin/create-section-with-admin` endpoint

**Current Implementation Pattern:**
- **Sections (Type 324):** Created via specialized endpoint `/api/admin/create-section-with-admin`
- **Departments (Type 325):** Created via `/api/settings/departments` (POST)
- **Administrations (Type 323):** NO creation endpoint found

**Endpoint Specialization:**
| Org Unit Type     | Creation Endpoint                        | Exists |
|-------------------|------------------------------------------|--------|
| Administration    | None found                               | ❌     |
| Department        | POST `/api/settings/departments`         | ✅     |
| Section           | POST `/api/admin/create-section-with-admin` | ✅     |
| Generic (any type)| None                                     | ❌     |

---

## 8. SETTINGS ROUTER - DEPARTMENT MANAGEMENT

**Router File:** [settings_router.py](backend/api/routers/settings_router.py)

### 8.1 Create Department Endpoint

**Route Path:** `/api/settings/departments`

**HTTP Method:** `POST`

**Request Model:** `DepartmentCreateRequest`
```python
class DepartmentCreateRequest(BaseModel):
    name: str
    name_ar: str
    code: str
    parent_id: Optional[int] = None
    mapping_mode: str = "internal"  # "internal" or "external"
    is_active: bool = True
    display_order: int = 0
```

**Permission Guards:**
- `require_logged_in(current_user)`
- `require_software_admin(current_user)`

**Service Function:** `SettingsService.create_department(...)`

**Note:** This creates departments in a separate settings-managed table, NOT AdminsrationUnit

---

### 8.2 Get Departments Endpoint

**Route Path:** `/api/settings/departments`

**HTTP Method:** `GET`

**Query Parameters:**
- `mapping_mode: Optional[str]` (filter "internal" or "external")
- `is_active: Optional[bool]` (default: True)
- `include_children: bool` (default: True)
- `flat: bool` (default: False - return tree structure)

**Response:** Hierarchical tree or flat array of departments

---

### 8.3 Update Department Endpoint

**Route Path:** `/api/settings/departments/{department_id}`

**HTTP Method:** `PUT`

---

### 8.4 Delete Department Endpoint

**Route Path:** `/api/settings/departments/{department_id}`

**HTTP Method:** `DELETE`

**Query Parameters:**
- `force: bool` (default: False)

---

## 9. ADMIN UNITS DB LAYER FUNCTIONS

**DB File:** [admin_units.py](backend/api/db_layer/admin_units.py)

### 9.1 Query Functions

**get_admin_unit_by_id(admin_unit_id: int)**
- Returns single unit by UniqueID

**get_admin_unit_type(admin_unit_id: int) -> int | None**
- Returns Type value only

**get_admin_unit_children(parent_id: int)**
- Returns direct children of a unit

**get_admin_unit_parent(admin_unit_id: int)**
- Returns parent unit via JOIN

**get_admin_unit_tree()**
- Returns ALL units (full tree)

**get_admin_unit_leaves()**
- Returns units with no children

**get_active_admin_units()**
- Returns units where `Frozen = 0` and `Type IS NOT NULL`
- Returns: `[{"UniqueID": int, "Name": str}, ...]`

**get_units_by_type(unit_type: int)**
- Filters by specific Type value (323, 324, 325)
- Excludes frozen and NULL type units
- Returns: `[{"id": int, "name": str, "parent_id": int}, ...]`

**Function Signature:**
```python
def get_units_by_type(unit_type: int):
    """
    Get all active organizational units of a specific type.
    Excludes frozen units and units with NULL type.
    
    Args:
        unit_type: 323=Administration, 324=Section, 325=Department
        
    Returns:
        List of dicts with UniqueID and Name
    """
```

**SQL:**
```sql
SELECT UniqueID, Name, ParentID
FROM AdminsrationUnit
WHERE Frozen = 0 AND Type = ? AND Type IS NOT NULL
ORDER BY Name
```

---

## 10. SUMMARY FINDINGS

### 10.1 Section Creation Flow

**Endpoint:** POST `/api/admin/create-section-with-admin`

**Complete Flow:**
1. Router validates SOFTWARE_ADMIN permission
2. Service opens transaction
3. DB layer inserts section (Type=324) into AdminsrationUnit
4. Service generates username `sec_{id}_admin`
5. DB layer inserts user into APP_Users
6. DB layer inserts scope into APP_UserRoleScope (OrgUnitType='SECTION')
7. Service commits transaction
8. Router returns credentials

**Transaction Safety:** ✅ Full transaction with rollback on error

**Atomicity:** ✅ All 3 inserts succeed or all fail

---

### 10.2 Section Admin Recreation Flow

**Endpoint:** POST `/api/admin/sections/{section_id}/recreate-admin`

**Complete Flow:**
1. Router validates SOFTWARE_ADMIN permission
2. Service opens transaction
3. DB layer verifies section exists and Type=324
4. Service generates unique versioned username
5. DB layer inserts new user into APP_Users
6. DB layer inserts scope into APP_UserRoleScope
7. Service commits transaction
8. Router returns credentials

**Key Difference:** Does NOT create new section, only adds additional admin user

---

### 10.3 Org Unit Type System

**Storage:**
- AdminsrationUnit.Type: INTEGER (323, 324, 325)
- APP_UserRoleScope.OrgUnitType: VARCHAR ('ADMINISTRATION', 'SECTION', 'DEPARTMENT')

**No Central Constants:** Type values hardcoded throughout codebase

---

### 10.4 Generic Org Unit Creation

**Status:** ❌ Does NOT exist

**Current Pattern:**
- Specialized endpoints per org unit type
- Section creation includes automatic admin user creation
- Department creation separate in settings router
- No administration creation endpoint found

---

### 10.5 Org Hierarchy Read Endpoints

**Available Endpoints:**
1. `/api/investigation/hierarchy` - Flat arrays with parent references
2. `/api/dashboard/hierarchy` - User-scoped flat arrays
3. `/api/investigation/tree` - Nested tree with metrics

**Shared Service:** `org_tree_service.py` provides in-memory tree traversal

---

## 11. ARCHITECTURAL NOTES

### 11.1 Transaction Pattern

All section creation functions follow same pattern:
```python
conn = None
try:
    conn = get_connection()
    # ... perform inserts ...
    conn.commit()
except Exception as e:
    if conn:
        conn.rollback()
    raise Exception(...)
finally:
    if conn:
        conn.close()
```

### 11.2 Username Generation

**Base Pattern:** `sec_{section_id}_admin`

**Collision Handling:** Append version suffix `_v2`, `_v3`, etc.

**Max Attempts:** 100 versions (safety limit)

### 11.3 Password System

**Current:** `TEMP_HASH_Hospital2026!` hardcoded test password

**Future:** Production should use secure password generation and hashing

### 11.4 Role Resolution

Roles stored in `APP_Roles` table with `RoleCode` (e.g., 'SECTION_ADMIN')

Service resolves RoleID dynamically via:
```sql
SELECT RoleID FROM dbo.APP_Roles WHERE RoleCode = ?
```

### 11.5 Org Tree Caching

`org_tree_service.py` loads entire tree into memory on first access

Cache invalidation: Manual via `clear_cache()` function

---

## 12. MISSING FUNCTIONALITY

**Based on this audit, the following are NOT implemented:**

1. ❌ Generic org unit creation endpoint accepting Type parameter
2. ❌ Administration (Type 323) creation endpoint
3. ❌ Section creation WITHOUT automatic admin user
4. ❌ Centralized org unit type constants/enum
5. ❌ Section update/delete endpoints
6. ❌ Bulk section creation
7. ❌ Section admin user list endpoint (per section)

---

## END OF REPORT

**Report Status:** ✅ Complete - No code changes made (read-only analysis)

**Next Steps:** Use this documentation to design Settings tool endpoints for section creation
