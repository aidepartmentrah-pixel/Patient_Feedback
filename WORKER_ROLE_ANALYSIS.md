# WORKER ROLE ANALYSIS

**Analysis Date:** January 29, 2026  
**Purpose:** Investigate the meaning, usage, and security implications of the WORKER role  
**Status:** ⚠️ Analysis Only - No Code Changes Made

---

## Executive Summary

The **WORKER** role is defined as a "basic complaint handler" but is **NOT actively enforced** in most endpoints. Despite being the lowest privilege role in the hierarchy, WORKER can access nearly all system features due to:
1. Missing role checks in critical endpoints (dashboard, trends, reports)
2. Using `require_logged_in()` instead of role-specific guards
3. No organizational unit scoping enforcement

### Critical Finding:
🔴 **WORKER is treated as "any authenticated user" rather than a restricted workflow role.**

---

## 1. Where is WORKER Defined?

### 1.1 Code Constants (`backend/core/constants/roles.py`)

```python
# Worker - Basic complaint handling role
# Can view and update complaints within their assigned organizational unit
WORKER = "WORKER"
```

**Role Hierarchy:**
```python
ROLE_HIERARCHY = [
    SOFTWARE_ADMIN,         # Highest privilege
    ADMINISTRATION_ADMIN,
    DEPARTMENT_ADMIN,
    SECTION_ADMIN,
    COMPLAINT_SUPERVISOR,
    WORKER,                 # Lowest privilege ⚠️
]
```

**Role Sets:**
```python
# Worker-level roles (can handle complaints)
WORKER_ROLES = [
    WORKER,  # Only WORKER in this set
]

# WORKER is NOT in ADMIN_ROLES
ADMIN_ROLES = [SOFTWARE_ADMIN, SECTION_ADMIN, DEPARTMENT_ADMIN, ADMINISTRATION_ADMIN]

# WORKER is NOT in SUPERVISOR_ROLES
SUPERVISOR_ROLES = [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR, SECTION_ADMIN, DEPARTMENT_ADMIN, ADMINISTRATION_ADMIN]
```

**Display Name:**
- English: "Worker"
- Arabic: "موظف"

### 1.2 Database Seeding (`backend/database_migrations/phase2_create_rbac_tables.sql`)

**APP_Roles Table:**
```sql
IF NOT EXISTS (SELECT 1 FROM dbo.APP_Roles WHERE RoleCode = 'WORKER')
    INSERT INTO dbo.APP_Roles (RoleCode, RoleNameEn, RoleNameAr)
    VALUES ('WORKER', 'Worker', 'موظف');
```

**Test User:**
```sql
-- Username: worker
-- Password: worker123
-- Assigned to: OrgUnitID=10, OrgUnitType='COMPLAINT'
INSERT INTO dbo.APP_Users (Username, PasswordHash, IsActive)
VALUES ('worker', 'TEMP_HASH_worker123', 1);

INSERT INTO dbo.APP_UserRoleScope (UserID, RoleID, OrgUnitID, OrgUnitType)
VALUES (@UserID_worker, @RoleID_WORKER, 10, 'COMPLAINT');
```

**Status:** ✅ Properly seeded in database

### 1.3 Documentation Comments

**From `guards.py`:**
```python
def require_worker(current_user: CurrentUser) -> None:
    """
    Require WORKER role.
    
    Workers are basic complaint handlers with limited permissions within
    their assigned organizational unit.
    
    Permissions:
    - View complaints in assigned org unit
    - Update complaint status
    - Add notes and actions
    """
```

**Intended Design:**
- ✅ Limited to assigned org unit
- ✅ Can view/update complaints
- ✅ Can add notes and actions
- ❌ Should NOT have admin privileges
- ❌ Should NOT access hospital-wide data

---

## 2. Where is WORKER Checked in the Codebase?

### 2.1 Authorization Guard (`backend/api/utils/guards.py`)

**Function:**
```python
def require_worker(current_user: CurrentUser) -> None:
    """Require WORKER role."""
    require_role(current_user, [WORKER])
```

**Status:** ✅ Guard exists and is properly implemented

### 2.2 Router Usage

#### Example Router (`backend/api/routers/example_guarded_router.py`)

```python
@router.get("/worker-only")
async def worker_only_endpoint(current_user: CurrentUser = Depends(get_current_user)):
    """
    Worker-only endpoint - Requires WORKER role.
    Only users with WORKER role can access this endpoint.
    """
    require_worker(current_user)
    
    return {
        "message": f"Worker {current_user.username} authorized",
        "authorization": "WORKER only",
        "action": "worker operation performed"
    }
```

**Status:** ✅ Example endpoint demonstrates correct usage (but this is just an example, not real functionality)

#### Real Routers

**Search Across All Routers:**
- `dashboard_router.py` - ❌ NO `require_worker` or even `get_current_user`
- `trend_router.py` - ❌ NO `require_worker` or `get_current_user`
- `reports_router.py` - ❌ NO `require_worker` (uses `require_logged_in` only)
- `insert_router.py` - ❌ NO `require_worker` (uses `require_logged_in` only)
- `settings_router.py` - ❌ NO `require_worker` (uses `require_logged_in` only)
- `training_router.py` - ❌ NO `require_worker` (uses `require_logged_in` only)
- `explanation_routes.py` - ❌ NO `require_worker` (uses `require_logged_in` only)
- `follow_up_router.py` - ❌ NO `require_worker` (uses `require_logged_in` only)
- `action_items.py` - ❌ NO `require_worker` (uses `require_logged_in` only)

**Critical Finding:**
🔴 **`require_worker()` is NEVER used in production routers** - only in example code and tests!

### 2.3 What is Actually Used? `require_logged_in()`

**From `guards.py`:**
```python
def require_logged_in(current_user: Optional[CurrentUser]) -> None:
    """
    Verify that user is authenticated.
    
    This is a basic guard that ensures a user is logged in.
    """
    if current_user is None:
        raise HTTPException(401, detail="NOT_AUTHENTICATED")
```

**Usage Count:**
- `require_logged_in`: **74 occurrences** across routers
- `require_worker`: **2 occurrences** (only in example router)
- `require_software_admin`: **Multiple occurrences** (in settings/training)

**What This Means:**
- Most endpoints check **authentication only** (not authorization)
- Any logged-in user (including WORKER) can access these endpoints
- WORKER effectively becomes "any authenticated user"

---

## 3. What Permissions Does WORKER Currently Have?

### 3.1 Endpoints Using `require_logged_in()` (Accessible to WORKER)

| Feature | Router | Endpoint | Guard | WORKER Access |
|---------|--------|----------|-------|---------------|
| **Insert Complaint** | `insert_router.py` | `POST /api/insert` | `require_logged_in` | ✅ YES |
| **Search Complaints** | `insert_router.py` | `GET /api/insert/search` | `require_logged_in` | ✅ YES |
| **Get Complaint Details** | `insert_router.py` | `GET /api/insert/{id}` | `require_logged_in` | ✅ YES |
| **Update Complaint** | `insert_router.py` | `PUT /api/insert/{id}` | `require_logged_in` | ✅ YES |
| **Delete Complaint** | `insert_router.py` | `DELETE /api/insert/{id}` | `require_logged_in` | ✅ YES |
| **Add Doctor** | `insert_router.py` | `POST /api/insert/doctor` | `require_logged_in` | ✅ YES |
| **Update Doctor** | `insert_router.py` | `PUT /api/insert/doctor/{id}` | `require_logged_in` | ✅ YES |
| **Add Patient** | `insert_router.py` | `POST /api/insert/patient` | `require_logged_in` | ✅ YES |
| **Update Patient** | `insert_router.py` | `PUT /api/insert/patient/{id}` | `require_logged_in` | ✅ YES |
| **Explanation Workflow** | `explanation_routes.py` | `POST /api/explanation/*` | `require_logged_in` | ✅ YES |
| **Action Items** | `action_items.py` | `GET/POST /api/action-items/*` | `require_logged_in` | ✅ YES |
| **Follow-Up** | `follow_up_router.py` | `GET/POST /api/follow-up/*` | `require_logged_in` | ✅ YES |
| **Reports (Seasonal)** | `reports_router.py` | `POST /api/reports/seasonal/*` | `require_logged_in` | ✅ YES |
| **Reports (Monthly)** | `reports_router.py` | `POST /api/reports/monthly/*` | `require_logged_in` | ✅ YES |
| **Settings** | `settings_router.py` | `GET /api/settings/*` | `require_logged_in` | ✅ YES |
| **Training** | `training_router.py` | `POST /api/training/*` | `require_logged_in` | ✅ YES |

### 3.2 Endpoints with NO Authentication (Accessible to WORKER)

| Feature | Router | Endpoint | Guard | WORKER Access |
|---------|--------|----------|-------|---------------|
| **Dashboard Stats** | `dashboard_router.py` | `GET /api/dashboard/stats` | ❌ None | ✅ YES (public!) |
| **Dashboard Hierarchy** | `dashboard_router.py` | `GET /api/dashboard/hierarchy` | ❌ None | ✅ YES (public!) |
| **Domain Trends** | `trend_router.py` | `GET /api/trends/domains` | ❌ None | ✅ YES (public!) |
| **Category Trends** | `trend_router.py` | `GET /api/trends/categories` | ❌ None | ✅ YES (public!) |
| **Trends Analysis** | `trend_router.py` | `GET /api/trends/analysis` | ❌ None | ✅ YES (public!) |
| **Seasonal View** | `reports_router.py` | `POST /api/reports/seasonal/view` | ❌ None | ✅ YES (public!) |
| **Monthly View** | `reports_router.py` | `POST /api/reports/monthly/view` | ❌ None | ✅ YES (public!) |

### 3.3 Endpoints Requiring Admin (NOT Accessible to WORKER)

| Feature | Router | Endpoint | Guard | WORKER Access |
|---------|--------|----------|-------|---------------|
| **Create System Setting** | `settings_router.py` | `POST /api/settings` | `require_software_admin` | ❌ NO |
| **Training Status** | `training_router.py` | `GET /api/training/status` | `require_software_admin` | ❌ NO |

**Summary:**
- ✅ WORKER can access **~90% of endpoints**
- ❌ WORKER blocked from **~5 admin-only endpoints**
- ⚠️ WORKER can access **public dashboards/trends** (no auth required)

---

## 4. Is WORKER Treated As...

### 4.1 A Complaint Workflow Worker? (Intended Design)

**Evidence FOR this interpretation:**
- ✅ Role name suggests workflow focus
- ✅ Documentation says "basic complaint handler"
- ✅ Seeded with OrgUnitType='COMPLAINT'
- ✅ Guard documentation mentions "assigned org unit"

**Evidence AGAINST this interpretation:**
- ❌ No org unit scoping enforcement anywhere
- ❌ Can access hospital-wide dashboards/trends
- ❌ Can view/edit ANY complaint (not just in their org unit)
- ❌ Can access training, settings, reports globally

**Conclusion:** 🔴 **NOT treated as workflow-specific** in practice

### 4.2 A Generic Authenticated User? (Current Reality)

**Evidence FOR this interpretation:**
- ✅ Most endpoints use `require_logged_in()` (not `require_worker()`)
- ✅ Effectively grants same access as any authenticated user
- ✅ No distinction between WORKER and other roles in most routers
- ✅ Can access features unrelated to complaint handling

**Evidence AGAINST this interpretation:**
- ❌ Excluded from ADMIN_ROLES and SUPERVISOR_ROLES
- ❌ Documentation suggests limited permissions
- ❌ Should not have access to system-wide data

**Conclusion:** ✅ **YES - Currently treated as generic authenticated user**

---

## 5. Can WORKER Access...

### 5.1 Dashboard?

**Endpoint:** `GET /api/dashboard/stats`

**Authentication Check:**
```python
@router.get("/stats")
def dashboard_stats(
    scope: str,
    administration_id: int | None,
    department_id: int | None,
    section_id: int | None,
    # ...
):
    # NO current_user parameter!
    # NO authentication required!
    return get_dashboard_stats(...)
```

**Answer:** ✅ **YES** - WORKER can access dashboard with ANY scope (hospital, administration, department, section)

**Security Impact:**
- WORKER with section-level role can view **hospital-wide** statistics
- No validation that requested scope matches user's org unit scope
- Client controls the scope parameter

### 5.2 Trends?

**Endpoint:** `GET /api/trends/domains`

**Authentication Check:**
```python
@router.get("/domains")
def fetch_domain_trends(
    start_date: str | None,
    end_date: str | None,
    # ...
):
    # NO current_user parameter!
    # NO authentication required!
    # NO scope filtering!
    return get_domain_trends(...)  # Returns hospital-wide data
```

**Answer:** ✅ **YES** - WORKER can access **hospital-wide** trend data

**Security Impact:**
- Trends endpoints return ALL incidents across entire hospital
- No org unit filtering available
- Any user (even unauthenticated) can access

### 5.3 Reports?

**Endpoint:** `POST /api/reports/seasonal/view`

**Authentication Check:**
```python
def view_seasonal_report(request: SeasonalViewRequestV2):
    # NO current_user parameter!
    # Client specifies orgunit_id and orgunit_type
    report = get_or_generate_seasonal_report(
        season_id=season_id,
        orgunit_id=request.orgunit_id,  # ← Client controls this!
        orgunit_type=request.orgunit_type,
        user_id=request.user_id
    )
```

**Answer:** ✅ **YES** - WORKER can request reports for **ANY org unit**

**Security Impact:**
- WORKER can specify any orgunit_id in request body
- No validation against user's assigned org unit
- Can access reports for units they shouldn't see

### 5.4 Admin Endpoints?

**Endpoints Requiring `require_software_admin`:**
- `POST /api/settings` - Create system settings
- `PUT /api/settings/{id}` - Update system settings  
- `DELETE /api/settings/{id}` - Delete system settings
- `GET /api/training/status` - Get training status

**Answer:** ❌ **NO** - WORKER **cannot** access SOFTWARE_ADMIN endpoints

**Status:** ✅ This is **correctly** enforced

---

## 6. Is WORKER Restricted by Org-Unit Scope?

### Database Scope Assignment

**Test User Scope:**
```sql
INSERT INTO dbo.APP_UserRoleScope (UserID, RoleID, OrgUnitID, OrgUnitType)
VALUES (@UserID_worker, @RoleID_WORKER, 10, 'COMPLAINT');
```

**In Memory (CurrentUser model):**
```python
{
  "user_id": 2,
  "username": "worker",
  "scopes": [
    {
      "role_code": "WORKER",
      "org_unit_id": 10,
      "org_unit_type": "COMPLAINT"
    }
  ]
}
```

**Status:** ✅ Scope is **loaded** and available in `current_user.scopes`

### Enforcement in Endpoints

**Dashboard:**
```python
def dashboard_stats(scope, administration_id, department_id, section_id):
    # ❌ Does NOT check current_user.scopes
    # ❌ Does NOT validate requested scope against user's org_unit_id
    # ✅ Client can request ANY scope
```

**Reports:**
```python
def view_seasonal_report(request):
    # ❌ Does NOT check current_user.scopes
    # ❌ Does NOT validate request.orgunit_id against user's org_unit_id
    # ✅ Client can request ANY orgunit_id
```

**Insert/Update:**
```python
@router.post("/api/insert")
def create_complaint(complaint: ComplaintRequest, current_user = Depends(get_current_user)):
    require_logged_in(current_user)
    # ❌ Does NOT filter by current_user.scopes[].org_unit_id
    # ✅ Can insert complaint for ANY org unit
```

**Answer:** ❌ **NO** - WORKER is **NOT restricted** by org unit scope

**Security Impact:**
- WORKER's `org_unit_id=10` is ignored
- Can view/edit data from any org unit
- Scope field exists but is not enforced

---

## 7. Summary: What WORKER Means Today

### Intended Design (From Documentation)

| Aspect | Intended Behavior |
|--------|-------------------|
| **Purpose** | Basic complaint handler |
| **Scope** | Limited to assigned org unit |
| **Permissions** | View/update complaints, add notes |
| **Access Level** | Lowest privilege role |
| **Data Visibility** | Only their org unit |

### Current Reality (From Code Analysis)

| Aspect | Actual Behavior |
|--------|-----------------|
| **Purpose** | Generic authenticated user |
| **Scope** | Hospital-wide (no restrictions) |
| **Permissions** | Access to ~90% of endpoints |
| **Access Level** | Same as most other roles |
| **Data Visibility** | All data across entire hospital |

### Gap Analysis

| Feature | Intended | Actual | Gap |
|---------|----------|--------|-----|
| **Org Unit Scoping** | Enforced | ❌ Not enforced | 🔴 CRITICAL |
| **Dashboard Access** | Section only | ✅ Hospital-wide | 🔴 CRITICAL |
| **Trends Access** | Section only | ✅ Hospital-wide | 🔴 CRITICAL |
| **Reports Access** | Section only | ✅ Any org unit | 🔴 CRITICAL |
| **Complaint Editing** | Own org unit | ✅ Any org unit | 🔴 CRITICAL |
| **Role Guard Usage** | `require_worker()` | ❌ `require_logged_in()` | 🔴 HIGH |
| **Admin Access** | Blocked | ✅ Correctly blocked | ✅ OK |

---

## 8. What is Dangerous or Incorrect

### 8.1 Security Vulnerabilities

#### Critical: No Org Unit Scoping

**Problem:**
- WORKER is assigned to `org_unit_id=10`
- This assignment is **never checked** when accessing data
- WORKER can view/edit data from **any org unit**

**Example Attack:**
```python
# WORKER user is assigned to Section 10
# But can do this:
GET /api/dashboard/stats?scope=hospital&...
# Returns hospital-wide data (all departments, all sections)

POST /api/reports/seasonal/view
{
  "orgunit_id": 999,  # Some other department
  "orgunit_type": 1,
  "year": 2026,
  "trimester": "Q1"
}
# Returns data for department 999 (not their assigned unit!)
```

**Impact:**
- Data breach: WORKER can see sensitive data from other departments
- Compliance violation: Breaks separation of duties
- Audit failure: No enforcement of least privilege

#### Critical: Hospital-Wide Dashboard/Trends

**Problem:**
- Dashboard and trends endpoints have **no authentication**
- Any user (including WORKER) can access
- No scope filtering applied

**Example Attack:**
```python
# WORKER assigned to small section
# Can view hospital-wide incident trends:
GET /api/trends/domains  # Returns ALL incidents across entire hospital
GET /api/dashboard/stats?scope=hospital  # Returns full hospital dashboard
```

**Impact:**
- WORKER sees data they shouldn't (patient privacy concerns)
- Can analyze incident patterns across entire organization
- May reveal sensitive operational data

#### High: Role Guard Not Used

**Problem:**
- `require_worker()` exists but is **never used**
- Most endpoints use `require_logged_in()` instead
- No distinction between WORKER and higher-privilege roles

**Example Issue:**
```python
# These endpoints should probably require ADMIN or SUPERVISOR:
POST /api/training/train  # Train ML model - uses require_logged_in
POST /api/settings/create-season  # Create season - uses require_logged_in
GET /api/reports/export  # Export reports - uses require_logged_in

# WORKER can access all of these!
```

**Impact:**
- WORKER can perform administrative actions
- No separation between operational and administrative functions
- Difficult to audit who performed sensitive operations

### 8.2 Design Issues

#### Issue 1: Ambiguous Role Purpose

**Problem:**
- Code says "complaint handler"
- Reality: "any authenticated user"
- No clear distinction from other roles

**Impact:**
- Developers don't know when to use `require_worker()` vs `require_logged_in()`
- Inconsistent security enforcement
- Role hierarchy becomes meaningless

#### Issue 2: Unused Scope Data

**Problem:**
- `current_user.scopes[].org_unit_id` is loaded but never used
- Database field serves no purpose
- Creates false sense of security

**Impact:**
- Wasted database queries
- Misleading documentation
- Developers think scoping exists when it doesn't

#### Issue 3: Public Endpoints

**Problem:**
- Critical endpoints have no authentication requirement
- Dashboard, trends, some reports are public
- Not even `require_logged_in()`

**Impact:**
- Unauthenticated users can access data
- No audit trail
- Potential data exposure

### 8.3 Compliance Risks

#### Risk 1: No Separation of Duties

**Requirement:** Users should only access data within their responsibilities

**Reality:**
- WORKER can access any department's data
- No enforcement of assigned scope
- Violates principle of least privilege

#### Risk 2: No Audit Trail for Scope Violations

**Requirement:** Log when users access data outside their scope

**Reality:**
- No detection of scope violations
- Can't identify when WORKER accesses unauthorized data
- Audit logs don't capture scope context

#### Risk 3: Patient Privacy (HIPAA/GDPR)

**Requirement:** Limit access to patient data based on need-to-know

**Reality:**
- WORKER can view patient data from any department
- No mechanism to restrict by org unit
- May violate healthcare privacy regulations

---

## 9. Recommendations (Analysis Only)

### Priority 1: Critical Security Fixes

1. **Enforce Org Unit Scoping**
   - Validate all data access against `current_user.scopes[].org_unit_id`
   - Filter dashboard/trends/reports by user's accessible org units
   - Reject requests for org units outside user's scope

2. **Add Authentication to Public Endpoints**
   - Add `current_user: CurrentUser = Depends(get_current_user)` to:
     - `/api/dashboard/*`
     - `/api/trends/*`
     - `/api/reports/*/view`

3. **Use Role-Specific Guards**
   - Replace `require_logged_in()` with appropriate role guards
   - Use `require_worker()` for complaint handling operations
   - Use `require_any_supervisor()` for oversight features
   - Use `require_any_admin()` for administrative features

### Priority 2: Design Clarification

4. **Define WORKER Role Clearly**
   - Document: Is WORKER a specialized role or generic user?
   - If specialized: Limit to complaint handling only
   - If generic: Rename to "USER" or "AUTHENTICATED_USER"

5. **Create Org Unit Scoping Service**
   - Centralized validation: `validate_user_access(current_user, target_org_unit)`
   - Tree traversal: `get_user_accessible_units(current_user)`
   - Apply consistently across all endpoints

6. **Standardize Guard Usage**
   - Create decision matrix: Which endpoints need which guards
   - Document guard selection guidelines
   - Enforce in code reviews

### Priority 3: Audit & Monitoring

7. **Log Scope Violations**
   - When user requests data outside their scope
   - When user attempts unauthorized role actions
   - Include in security audit logs

8. **Add Scope Context to Audit Logs**
   - Record user's `org_unit_id` on every request
   - Record requested `scope` or `orgunit_id` parameters
   - Enable post-incident analysis

---

## 10. Conclusion

### What WORKER Means Today:
- **In name:** "Basic complaint handler with limited permissions"
- **In code:** "Generic authenticated user with hospital-wide access"

### What is Dangerous:
1. 🔴 **No org unit scoping enforcement** - WORKER can access any department's data
2. 🔴 **Public critical endpoints** - Dashboard/trends have no authentication
3. 🔴 **Role guards not used** - Most endpoints treat all authenticated users equally
4. 🔴 **Compliance risk** - Violates least privilege and separation of duties

### What Needs to Change:
- Enforce `current_user.scopes[].org_unit_id` filtering
- Add authentication to public endpoints
- Use role-specific guards instead of generic `require_logged_in()`
- Clarify whether WORKER is specialized or generic role
- Implement centralized org unit scoping service

### Security Impact:
**WORKER role currently provides nearly unrestricted access to the system, defeating the purpose of RBAC.**

---

**End of Analysis**
