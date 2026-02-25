# SCOPING CURRENT STATE ANALYSIS

**Analysis Date:** January 29, 2026  
**Purpose:** Analyze RBAC and organizational unit scoping implementation in FastAPI backend  
**Status:** ⚠️ Analysis Only - No Code Changes Made

---

## Executive Summary

The backend has **authentication** fully implemented (session-based, NO JWT), **role-based authorization guards** in place, but **organizational unit scoping is NOT enforced** at the application level. Data filtering by organizational units happens through explicit query parameters or hardcoded logic, not through automatic user-context-based scoping.

### Key Findings:
- ✅ **Authentication**: Fully implemented (session-based)
- ✅ **Role-Based Access Control (RBAC)**: Guards check roles
- ❌ **Org Unit Scoping**: NOT enforced based on user context
- ⚠️ **AdministrationUnit**: Used for queries but NO traversal helpers
- ⚠️ **Scoping in Dashboards/Reports**: Relies on manual parameters, not user scopes

---

## 1. How AdministrationUnit is Used

### Database Table: `AdminsrationUnit` (Note: typo in table name)

**Location:** `IncidentManager.dbo.AdminsrationUnit`

**Structure:**
```sql
- UniqueID (Primary Key)
- ParentID (Self-referencing FK)
- Name
- Type (323=Administration, 324=Section, 325=Department)
- Frozen (Boolean)
```

**Hierarchy Logic:**
- **Administration**: `ParentID == UniqueID` (self-referencing root)
- **Department**: `ParentID == Administration.UniqueID AND UniqueID != ParentID`
- **Section**: `ParentID == Department.UniqueID`

### Where is it Accessed?

#### 1.1 Database Layer (`backend/api/db_layer/admin_units.py`)

**Functions:**
- `get_admin_unit_tree()` - Returns ALL org units (no filtering)
- `get_admin_unit_by_id(admin_unit_id)` - Single unit lookup
- `get_admin_unit_type(admin_unit_id)` - Get Type field
- `get_admin_unit_children(parent_id)` - Get direct children
- `get_admin_unit_parent(admin_unit_id)` - Get parent unit
- `get_admin_unit_leaves()` - Get leaf nodes (no children)
- `get_active_admin_units()` - Get non-frozen units with valid type
- `get_units_by_type(unit_type)` - Filter by Type (323/324/325)

**Critical Finding:** No functions for **tree traversal** or **descendant collection** at DB layer.

#### 1.2 Service Layer Usage

**Dashboard Service** (`api/services/dashboard_service.py`):
```python
def _resolve_scope(scope, admin_id, dept_id, section_id):
    raw_units = admin_units.get_admin_unit_tree()
    units = [_row_to_dict(u) for u in raw_units]
    
    if scope == "hospital":
        return [u["UniqueID"] for u in units], True
    
    if scope == "administration":
        descendants = _collect_descendants(units, admin_id)
        return descendants, True
    
    if scope == "department":
        descendants = _collect_descendants(units, dept_id)
        return descendants, True
    
    if scope == "section":
        return [section_id], False
```

**Tree Traversal Helper:**
```python
def _collect_descendants(units, root_id):
    """
    Collect root_id and all its descendants using iterative traversal.
    """
    result = set()
    stack = [root_id]
    visited = set()
    
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        result.add(current)
        
        for u in units:
            child_id = u["UniqueID"]
            parent_id = u["ParentID"]
            if parent_id == current and child_id != current and child_id not in visited:
                stack.append(child_id)
    
    return list(result)
```

**Status:** ✅ Dashboard service has tree traversal logic  
**Location:** Service layer only (NOT reusable across services)

#### 1.3 Reports Service (`api/services/reports_service.py`, `monthly_report_service.py`)

**NO tree traversal logic** - Reports rely on:
1. Direct unit IDs passed as query parameters
2. UNION logic in SQL queries (multiple IDs combined with OR)
3. Hardcoded organizational filters

**Example from Monthly Reports:**
```python
if administration_ids:
    admin_id_list = [int(x.strip()) for x in administration_ids.split(",")]
    filters["idara_id"] = admin_id_list

if department_ids:
    dept_id_list = [int(x.strip()) for x in department_ids.split(",")]
    filters["dayra_id"] = dept_id_list
```

**Status:** ❌ No tree-aware scoping, relies on explicit parameter lists

#### 1.4 Seasonal Reports (`api/db_layer/seasonal_report_aggregation.py`)

**Has partial tree-aware filtering:**
```python
if orgunit_type == 1:  # Administration
    idara_id = orgunit_id
    org_filter = """
        WHERE td.IdaraID IN (
            SELECT UniqueID FROM IncidentManager.dbo.AdminsrationUnit
            WHERE UniqueID = ? OR ParentID = ?
        )
    """
```

**Status:** ⚠️ Partial tree expansion, inconsistent across services

### Summary: AdministrationUnit Access

| Component | Tree Traversal | Reusable | Scoped by User Context |
|-----------|---------------|----------|------------------------|
| DB Layer | ❌ No | N/A | ❌ No |
| Dashboard Service | ✅ Yes (`_collect_descendants`) | ❌ No (local function) | ❌ No |
| Reports Service | ❌ No | N/A | ❌ No |
| Seasonal Reports | ⚠️ Partial (SQL-based) | ❌ No | ❌ No |
| Trend Service | ✅ Yes (calls dashboard's `_resolve_scope`) | ⚠️ Indirect | ❌ No |

---

## 2. How APP_UserRoleScope is Used

### Database Table: `APP_UserRoleScope`

**Location:** `IncidentManager.dbo.APP_UserRoleScope`

**Structure:**
```sql
- UserID (FK to APP_Users)
- RoleID (FK to APP_Roles)
- OrgUnitID (FK to AdminsrationUnit.UniqueID)
- OrgUnitType (String: "HOSPITAL", "ADMINISTRATION", "DEPARTMENT", "SECTION")
```

**Purpose:** Maps users to roles within specific organizational units.

### Where is it Read?

#### 2.1 Database Layer (`api/db_layer/auth_db.py`)

**Function:** `get_user_with_scopes(user_id)`

```python
def get_user_with_scopes(user_id: int) -> Optional[Dict[str, Any]]:
    # Load user basic info
    cursor.execute("SELECT UserID, Username, IsActive FROM dbo.APP_Users WHERE UserID = ?", (user_id,))
    user_row = cursor.fetchone()
    
    # Load all role scopes
    cursor.execute("""
        SELECT 
            r.RoleCode,
            urs.OrgUnitID,
            urs.OrgUnitType
        FROM dbo.APP_UserRoleScope urs
        INNER JOIN dbo.APP_Roles r ON urs.RoleID = r.RoleID
        WHERE urs.UserID = ?
        ORDER BY r.RoleCode, urs.OrgUnitType, urs.OrgUnitID
    """, (user_id,))
    
    scope_rows = cursor.fetchall()
    
    # Build scopes array
    for scope_row in scope_rows:
        user_data["scopes"].append({
            "role_code": scope_row.RoleCode,
            "org_unit_id": scope_row.OrgUnitID,
            "org_unit_type": scope_row.OrgUnitType
        })
    
    return user_data
```

**Status:** ✅ Loaded during login/session validation

#### 2.2 Service Layer (`api/services/auth_service.py`)

**Function:** `get_current_user_from_session(request)`

```python
def get_current_user_from_session(request: Request) -> CurrentUser:
    user_id = request.session.get("user_id")
    
    if user_id is None:
        raise HTTPException(401, detail="NOT_AUTHENTICATED")
    
    # Load user with scopes from database
    user_data = get_user_with_scopes(user_id)
    
    # Convert to CurrentUser model
    current_user = CurrentUser(
        user_id=user_data["user_id"],
        username=user_data["username"],
        is_active=user_data["is_active"],
        scopes=[
            UserScope(
                role_code=scope["role_code"],
                org_unit_id=scope["org_unit_id"],
                org_unit_type=scope["org_unit_type"]
            )
            for scope in user_data["scopes"]
        ]
    )
    
    return current_user
```

**Status:** ✅ Scopes loaded on every authenticated request

### Is it Used Only in Login or Also in Guards/Services?

#### Used in Login Flow:
1. `POST /api/auth/login` → `validate_user_credentials()` → loads scopes
2. Creates session with `user_id`
3. `CurrentUser` object populated with scopes

#### Used in Guards:
Guards check **role_code** ONLY:

```python
def require_role(current_user: CurrentUser, allowed_roles: list[str]) -> None:
    user_roles = [scope.role_code for scope in current_user.scopes]
    has_required_role = any(role in allowed_roles for role in user_roles)
    
    if not has_required_role:
        raise HTTPException(403, detail="FORBIDDEN")
```

**Critical:** Guards check if user has the **role**, but do NOT enforce **org_unit_id** filtering.

#### Used in Services:
**❌ NOT USED** - Services do NOT check `current_user.scopes[].org_unit_id` for filtering.

**Example from Reports Router:**
```python
@router.post("/seasonal/view")
def view_seasonal_report(request: SeasonalViewRequestV2):
    # request contains: year, trimester, orgunit_id, orgunit_type
    # NO current_user parameter!
    # NO scoping validation!
    
    report = get_or_generate_seasonal_report(
        season_id=season_id,
        orgunit_id=request.orgunit_id,  # ← Directly from request, not validated against user scopes
        orgunit_type=request.orgunit_type,
        user_id=request.user_id
    )
```

**Status:** ⚠️ Scopes loaded but **NOT enforced** in data filtering.

### Summary: APP_UserRoleScope Usage

| Usage Location | Read Scopes? | Check role_code? | Check org_unit_id? | Enforce Data Filtering? |
|----------------|--------------|------------------|-------------------|------------------------|
| Login (`auth_service.py`) | ✅ Yes | N/A | N/A | N/A |
| `get_current_user()` Dependency | ✅ Yes | N/A | N/A | N/A |
| Authorization Guards | ✅ Yes | ✅ Yes | ❌ No | ❌ No |
| Dashboard Service | ❌ No | ❌ No | ❌ No | ❌ No |
| Reports Service | ❌ No | ❌ No | ❌ No | ❌ No |
| Trend Service | ❌ No | ❌ No | ❌ No | ❌ No |

---

## 3. What get_current_user() Returns

### Function Location
- **Dependency:** `api/dependencies/user_context.py`
- **Service:** `api/services/auth_service.py`

### Exact Shape and Fields

```python
class UserScope(BaseModel):
    role_code: str         # e.g., "SOFTWARE_ADMIN", "SECTION_ADMIN", "WORKER"
    org_unit_id: int       # FK to AdminsrationUnit.UniqueID
    org_unit_type: str     # "HOSPITAL", "ADMINISTRATION", "DEPARTMENT", "SECTION"

class CurrentUser(BaseModel):
    user_id: int
    username: str
    is_active: bool
    scopes: List[UserScope]
```

### Example Return Value:

```json
{
  "user_id": 5,
  "username": "section_admin",
  "is_active": true,
  "scopes": [
    {
      "role_code": "SECTION_ADMIN",
      "org_unit_id": 10,
      "org_unit_type": "SECTION"
    },
    {
      "role_code": "WORKER",
      "org_unit_id": 10,
      "org_unit_type": "SECTION"
    }
  ]
}
```

### When is it Called?

```python
from api.dependencies.user_context import get_current_user

@router.get("/protected-endpoint")
def my_endpoint(current_user: CurrentUser = Depends(get_current_user)):
    # current_user is automatically injected by FastAPI
    # Contains user_id, username, is_active, scopes[]
    return {"user": current_user.username}
```

### Authentication Flow:

1. Request arrives with session cookie
2. FastAPI executes `get_current_user()` dependency
3. Reads `user_id` from `request.session`
4. Calls `get_user_with_scopes(user_id)` from DB
5. Returns `CurrentUser` object with all fields populated
6. Endpoint receives `current_user` parameter

### Error Handling:

| Condition | HTTP Status | Error Code |
|-----------|-------------|------------|
| No session cookie | 401 | `NOT_AUTHENTICATED` |
| User not found in DB | 401 | `USER_NOT_FOUND` |
| User inactive (`is_active=False`) | 401 | `USER_INACTIVE` |
| Database error | 500 | `SESSION_ERROR` |

---

## 4. How Scoping is Currently Applied

### 4.1 Dashboard (`api/routers/dashboard_router.py`)

**Endpoint:** `GET /api/dashboard/stats`

**Parameters:**
```python
@router.get("/stats")
def dashboard_stats(
    scope: str,                      # "hospital" | "administration" | "department" | "section"
    administration_id: int | None,
    department_id: int | None,
    section_id: int | None,
    start_date: date | None,
    end_date: date | None,
    # ... chart type parameters
):
```

**Current Behavior:**
1. ❌ **NO** `current_user` parameter
2. Scope is determined by **query parameters only**
3. Any client can request any scope (no validation)

**Service Layer:** `dashboard_service.get_dashboard_stats()`

```python
def get_dashboard_stats(
    scope: str,
    administration_id: int | None,
    department_id: int | None,
    section_id: int | None,
    start_date: date,
    end_date: date,
    # ...
) -> dict:
    scope_unit_ids, include_issuing_dept = _resolve_scope(
        scope, administration_id, department_id, section_id
    )
    
    incidents = _fetch_incidents_in_scope(scope_unit_ids, start_date, end_date)
    # ...
```

**Scoping Logic:**
- Uses `_resolve_scope()` to calculate org unit IDs
- Collects descendants using `_collect_descendants()` (tree traversal)
- Filters incidents by `IssuingOrgUnitID IN (scope_unit_ids)`

**Critical Gap:**
- ❌ No validation that user has access to requested scope
- ❌ User with SECTION role can query HOSPITAL-wide data
- ✅ Tree traversal works correctly (descendants included)

### 4.2 Trend Monitoring (`api/routers/trend_router.py`)

**Endpoints:**
- `GET /api/trends/domains`
- `GET /api/trends/categories`

**Parameters:**
```python
@router.get("/domains")
def fetch_domain_trends(
    start_date: str | None,
    end_date: str | None,
    include_zero_months: bool = True,
    # ...
):
```

**Current Behavior:**
1. ❌ **NO** `current_user` parameter
2. ❌ **NO** scope filtering parameters at all
3. Returns **hospital-wide** trends for ALL domains/categories
4. Any authenticated user can see all data

**Service Layer:** `trend_service.get_domain_trends()`

```python
def get_domain_trends(
    start_date: date | None = None,
    end_date: date | None = None,
    # ... (NO org unit parameters)
) -> dict:
    # Fetch ALL incidents in date range (no org filtering)
    raw_data = _fetch_incidents_by_domain_and_month(start_date, end_date)
    # ...
```

**Critical Gap:**
- ❌ Completely unscoped - returns hospital-wide data
- ❌ No org unit filtering available
- ❌ User scopes ignored

**Alternative Endpoint:** `GET /api/trends/analysis`

This endpoint DOES support scoping:
```python
@router.get("/analysis")
def fetch_trends_analysis(
    scope: str,                     # "hospital" | "administration" | "department" | "section"
    administration_id: int | None,
    department_id: int | None,
    section_id: int | None,
    # ...
):
```

But still:
- ❌ **NO** `current_user` parameter
- ❌ No validation against user scopes

### 4.3 Reporting (`api/routers/reports_router.py`)

**Endpoint:** `POST /api/reports/seasonal/view`

**Request Body:**
```python
class SeasonalViewRequestV2(BaseModel):
    year: int
    trimester: str                  # "Q1", "Q2", "Q3", "Q4", "Trim1", "Trim2", "Trim3"
    orgunit_id: int                 # ← Client controls this
    orgunit_type: int               # ← Client controls this
    user_id: Optional[int] = 1
```

**Current Behavior:**
1. ❌ **NO** `current_user` parameter (despite having `user_id` in request body!)
2. Client specifies `orgunit_id` and `orgunit_type` freely
3. No validation that user has access to requested org unit

**Service Layer:** `get_or_generate_seasonal_report()`

```python
def get_or_generate_seasonal_report(
    season_id: int,
    orgunit_id: int,      # ← From request, not validated
    orgunit_type: int,
    user_id: int
):
    # Query filters by orgunit_id directly (no scope validation)
    report = seasonal_report_db.get_seasonal_report(
        season_id=season_id,
        orgunit_id=orgunit_id,
        orgunit_type=orgunit_type
    )
    # ...
```

**Critical Gap:**
- ❌ User can request ANY org unit's report
- ❌ No cross-check with `current_user.scopes[].org_unit_id`

**Monthly Reports:** `POST /api/reports/monthly/view`

```python
class MonthlyViewRequest(BaseModel):
    year: int
    month: Optional[int] = None
    scope: Optional[str] = None
    administration_ids: Optional[str] = None  # ← CSV of IDs
    department_ids: Optional[str] = None
    section_ids: Optional[str] = None
```

**Current Behavior:**
1. ❌ **NO** `current_user` parameter
2. Client specifies org units via CSV strings
3. UNION logic (OR) - combines multiple units
4. No validation

**Critical Gap:**
- ❌ User can request ANY combination of org units
- ❌ No scope enforcement

---

## 5. Summary: Where Scoping is Missing or Hardcoded

### Missing Scoping (User Context Not Enforced)

| Feature | Endpoint | current_user? | Scope Parameter? | Validated Against User Scopes? | Risk Level |
|---------|----------|---------------|------------------|--------------------------------|------------|
| **Dashboard** | `GET /api/dashboard/stats` | ❌ No | ✅ Yes (manual) | ❌ No | 🔴 HIGH |
| **Trends (Domains)** | `GET /api/trends/domains` | ❌ No | ❌ No | ❌ No | 🔴 CRITICAL |
| **Trends (Categories)** | `GET /api/trends/categories` | ❌ No | ❌ No | ❌ No | 🔴 CRITICAL |
| **Trends (Analysis)** | `GET /api/trends/analysis` | ❌ No | ✅ Yes (manual) | ❌ No | 🔴 HIGH |
| **Seasonal Reports** | `POST /api/reports/seasonal/view` | ❌ No | ✅ Yes (in body) | ❌ No | 🔴 CRITICAL |
| **Monthly Reports** | `POST /api/reports/monthly/view` | ❌ No | ✅ Yes (CSV) | ❌ No | 🔴 HIGH |
| **Report Export** | `POST /api/reports/export` | ❌ No | ✅ Yes | ❌ No | 🔴 HIGH |

### Hardcoded Scoping Logic

| Component | Hardcoded Behavior | Location |
|-----------|-------------------|----------|
| **Dashboard Hierarchy** | Frontend must know hierarchy structure | `dashboard_service.get_dashboard_hierarchy()` |
| **Org Unit IDs** | Frontend passes explicit IDs | All report/dashboard routers |
| **Tree Traversal** | Each service implements own logic | `dashboard_service._collect_descendants()` |
| **Seasonal Aggregation** | SQL-based tree expansion (partial) | `seasonal_report_aggregation.py` |

### Working Authorization (Role-Based)

| Feature | Endpoint | current_user? | Role Guards? | Status |
|---------|----------|---------------|--------------|--------|
| **Settings** | `GET /api/settings/*` | ✅ Yes | ✅ Yes (require_logged_in) | ✅ PROTECTED |
| **Training** | `POST /api/training/*` | ✅ Yes | ✅ Yes (require_logged_in) | ✅ PROTECTED |
| **Insert/Update** | `POST /api/insert/*` | ✅ Yes | ✅ Yes (require_logged_in) | ✅ PROTECTED |

---

## 6. Architectural Gaps

### 6.1 No Centralized Scoping Service

**Problem:**
- Tree traversal logic exists in `dashboard_service._collect_descendants()`
- NOT reusable across other services
- Trend service calls dashboard's `_resolve_scope()` indirectly
- Seasonal reports use SQL-based tree expansion

**Missing:**
- Centralized `org_unit_service` or `scoping_service`
- Reusable functions like:
  - `get_descendants(org_unit_id)`
  - `get_user_accessible_units(current_user)`
  - `validate_access(current_user, target_org_unit_id)`

### 6.2 No Scoping Middleware

**Problem:**
- Each endpoint must manually extract `current_user`
- Each service must manually validate scope
- Easy to forget validation

**Missing:**
- Automatic scope filtering based on `current_user.scopes`
- Middleware to inject `accessible_org_units` into request context
- Decorator like `@require_scope_access` for endpoints

### 6.3 No Database-Level Helper Functions

**Problem:**
- `admin_units.py` only has basic CRUD functions
- No functions for:
  - `get_descendants(root_id)` (recursive)
  - `get_ancestors(leaf_id)` (upward traversal)
  - `is_ancestor_of(parent_id, child_id)` (validation)

### 6.4 Inconsistent Scope Parameters

**Problem:**
- Dashboard uses: `scope`, `administration_id`, `department_id`, `section_id`
- Reports use: `orgunit_id`, `orgunit_type`
- Trends use: nothing (hospital-wide only)

**Missing:**
- Standardized scope parameter model across all endpoints

---

## 7. Recommendations (Analysis Only)

### Priority 1: Critical Security Gaps

1. **Add `current_user` to all data-access endpoints**
   - Dashboard, trends, reports must inject `current_user` dependency
   - Prevents unauthorized data access

2. **Validate org unit access against user scopes**
   - Before querying data, check if `requested_org_unit_id` is within user's accessible units
   - Use tree traversal to include descendants

3. **Remove client-controlled org unit parameters where possible**
   - Derive accessible units from `current_user.scopes`
   - Allow optional filtering within accessible scope

### Priority 2: Architectural Improvements

4. **Create centralized scoping service**
   - Module: `api/services/org_unit_scoping_service.py`
   - Functions:
     - `get_user_accessible_units(current_user) -> List[int]`
     - `get_descendants(root_id) -> List[int]`
     - `validate_user_access(current_user, target_org_unit_id) -> bool`

5. **Add database-level tree helpers**
   - Module: `api/db_layer/org_unit_tree.py`
   - Functions:
     - `get_subtree(root_id) -> List[dict]` (recursive)
     - `get_ancestors(leaf_id) -> List[dict]` (upward)

6. **Standardize scope parameters**
   - Use consistent naming across all endpoints
   - Single source of truth for scope filtering logic

### Priority 3: Code Cleanup

7. **Consolidate tree traversal logic**
   - Move `_collect_descendants()` from dashboard_service to shared module
   - Reuse in trend_service, reports_service, etc.

8. **Add scope validation guards**
   - New guard: `require_org_unit_access(current_user, org_unit_id)`
   - Use in routers before calling services

---

## 8. Conclusion

### What Works:
- ✅ Session-based authentication
- ✅ Role-based authorization guards (`require_software_admin`, etc.)
- ✅ User scopes loaded and available in `current_user.scopes`
- ✅ Tree traversal logic exists (but not reusable)

### What Doesn't Work:
- ❌ Org unit scoping NOT enforced based on user context
- ❌ Endpoints accept arbitrary org unit IDs without validation
- ❌ Trends are hospital-wide (no org filtering)
- ❌ Reports can be requested for any org unit

### Security Impact:
- 🔴 **HIGH RISK:** Any authenticated user can access any org unit's data
- Users with SECTION-level roles can view hospital-wide dashboards
- No separation of data between departments

### Next Steps:
1. Prioritize adding `current_user` dependency to critical endpoints
2. Implement scope validation before data queries
3. Create reusable scoping service for tree traversal
4. Standardize scope parameters across all features

---

**End of Analysis**
