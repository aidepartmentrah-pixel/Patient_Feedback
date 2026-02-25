# 🔐 USER CREDENTIALS FOR TESTING

Based on the Phase 2 RBAC database migration and test files, here are all the user accounts:

================================================================================
## 📋 ALL USER ACCOUNTS WITH CREDENTIALS
================================================================================

### 1. SOFTWARE_ADMIN (Super Admin - Full System Access)
   **Username:** `software_admin`
   **Password:** `admin123`
   **Role:** SOFTWARE_ADMIN
   **Permissions:** Full system access, can access all protected endpoints including admin-only features
   **Can Access:**
   - All Settings endpoints (departments, attributes, policies, system settings)
   - All Training endpoints
   - Admin force close cases
   - All other protected endpoints

---

### 2. WORKER (Basic User)
   **Username:** `worker`
   **Password:** `worker123`
   **Role:** WORKER
   **Permissions:** Basic access, cannot access admin endpoints
   **Can Access:**
   - Regular endpoints (insert, follow-up, action items)
   - Explanation endpoints (except admin force close)
   - Report viewing/exporting
   - Patient management

---

### 3. COMPLAINT_SUPERVISOR (Supervisor Level)
   **Username:** `complaint_supervisor`
   **Password:** `sup123`
   **Role:** COMPLAINT_SUPERVISOR
   **Permissions:** Supervisor-level access, cannot access admin endpoints
   **Can Access:**
   - All worker permissions
   - Supervisor-specific features
   - Cannot access admin-only features

---

### 4. SECTION_ADMIN (Section Administrator)
   **Username:** `section_admin`
   **Password:** `section123`
   **Role:** SECTION_ADMIN
   **Permissions:** Admin access for specific sections, cannot access global admin features
   **Can Access:**
   - Section-level administration
   - All worker permissions
   - Section-specific reports
   - Cannot access global system settings

---

### 5. DEPARTMENT_ADMIN (Department Administrator)
   **Username:** `department_admin`
   **Password:** `dept123`
   **Role:** DEPARTMENT_ADMIN
   **Permissions:** Admin access for specific departments
   **Can Access:**
   - Department-level administration
   - All worker permissions
   - Department-specific reports
   - Cannot access global system settings

---

### 6. ADMINISTRATION_ADMIN (Administration Administrator)
   **Username:** `administration_admin`
   **Password:** `adminis123`
   **Role:** ADMINISTRATION_ADMIN
   **Permissions:** Admin access for entire administration
   **Can Access:**
   - Administration-level management
   - All worker permissions
   - Administration-wide reports
   - Cannot access global system settings

================================================================================

## 🎯 QUICK REFERENCE TABLE

| Username              | Password      | Role                    | Admin Access |
|-----------------------|---------------|-------------------------|--------------|
| software_admin        | admin123      | SOFTWARE_ADMIN          | ✅ YES       |
| worker                | worker123     | WORKER                  | ❌ NO        |
| complaint_supervisor  | sup123        | COMPLAINT_SUPERVISOR    | ❌ NO        |
| section_admin         | section123    | SECTION_ADMIN           | ❌ NO        |
| department_admin      | dept123       | DEPARTMENT_ADMIN        | ❌ NO        |
| administration_admin  | adminis123    | ADMINISTRATION_ADMIN    | ❌ NO        |

================================================================================

## 🧪 TESTING SCENARIOS

### Test Authentication (All Users Should Login Successfully)
```bash
# Software Admin
Username: software_admin
Password: admin123

# Worker
Username: worker
Password: worker123

# Complaint Supervisor
Username: complaint_supervisor
Password: sup123

# Section Admin
Username: section_admin
Password: section123

# Department Admin
Username: department_admin
Password: dept123

# Administration Admin
Username: administration_admin
Password: adminis123
```

---

### Test Admin-Only Endpoints (Only SOFTWARE_ADMIN Should Access)

**Settings Router Endpoints (Require SOFTWARE_ADMIN):**
- GET /api/settings/departments
- POST /api/settings/departments
- PUT /api/settings/departments/{id}
- DELETE /api/settings/departments/{id}
- GET /api/settings/attributes
- PUT /api/settings/attributes
- GET /api/settings/policies
- PUT /api/settings/policies
- GET /api/settings/export
- POST /api/settings/save-snapshot
- GET /api/settings/snapshots
- GET /api/settings/system-settings
- GET /api/settings/system-settings/{key}
- PUT /api/settings/system-settings/{key}
- POST /api/settings/system-settings

**Training Router Endpoints (Require SOFTWARE_ADMIN):**
- GET /api/settings/training/status
- GET /api/settings/training/progress
- GET /api/settings/training/grouped-status
- GET /api/settings/training/history
- GET /api/settings/training/db-size
- POST /api/settings/training/run
- GET /api/settings/training/charts/db-growth
- GET /api/settings/training/charts/performance-trends
- GET /api/settings/training/charts/training-timeline
- GET /api/settings/training/charts/family-comparison

**Explanation Router Admin Endpoint:**
- POST /api/explanations/{case_id}/force-close (Requires SOFTWARE_ADMIN)

---

### Test Protected Endpoints (All Logged-in Users Should Access)

**Explanation Endpoints (Require Authentication):**
- GET /api/explanations/pending
- GET /api/explanations/statistics
- GET /api/explanations/{case_id}
- GET /api/explanations/{case_id}/completion-status
- POST /api/explanations/{case_id}
- PUT /api/explanations/{case_id}/requires-explanation
- POST /api/explanations/{case_id}/check-closure
- POST /api/explanations/{case_id}/mark-action-complete
- POST /api/explanations/{case_id}/validate

**Reports Router Protected Endpoints:**
- POST /api/reports/seasonal/{report_id}/explanation
- PUT /api/reports/seasonal/{report_id}/explanation
- POST /api/reports/export
- POST /api/reports/seasonal/export
- POST /api/reports/monthly/export

---

### Test Public Endpoints (No Authentication Required)

**Reports Router Public Endpoints:**
- POST /api/reports/seasonal/view
- POST /api/reports/monthly/view
- GET /api/reports/download/{export_id}

================================================================================

## 🔒 EXPECTED BEHAVIOR

### For Non-Admin Users (worker, complaint_supervisor, section_admin, department_admin, administration_admin):
- ✅ Can login successfully
- ✅ Can access explanation endpoints
- ✅ Can access report export endpoints
- ❌ Get 403 Forbidden when accessing Settings endpoints
- ❌ Get 403 Forbidden when accessing Training endpoints
- ❌ Get 403 Forbidden when accessing admin force close

### For SOFTWARE_ADMIN:
- ✅ Can login successfully
- ✅ Can access ALL endpoints (no restrictions)
- ✅ Can access Settings and Training endpoints
- ✅ Can force close cases

### For Not Logged In:
- ❌ Get 401 Unauthorized on all protected endpoints
- ✅ Can access public report viewing endpoints

================================================================================

## 📝 NOTES

1. **Password Storage**: The passwords are stored with temporary hashes in the format 
   `TEMP_HASH_<password>`. The authentication system handles these automatically.

2. **Case Sensitivity**: Usernames are NOT case-sensitive in SQL Server by default.

3. **Session Management**: After login, a session cookie named `incident_manager_session` 
   is created. This cookie is used for authentication on subsequent requests.

4. **Role-Based Access**: Only SOFTWARE_ADMIN role has access to admin-only endpoints. 
   All other roles will receive 403 Forbidden when attempting to access these endpoints.

5. **Testing Order**:
   - Test login with each user
   - Test protected endpoints with each user
   - Test admin endpoints with non-admin users (should fail with 403)
   - Test admin endpoints with software_admin (should succeed)
   - Test public endpoints without login (should succeed)

================================================================================

## 🚀 RECOMMENDED TESTING WORKFLOW

1. **Login as software_admin** → Test all admin endpoints (should work)
2. **Login as worker** → Test admin endpoints (should get 403)
3. **Login as worker** → Test regular endpoints (should work)
4. **Logout** → Test protected endpoints (should get 401)
5. **No login** → Test public endpoints (should work)

================================================================================

Created: January 28, 2026
Source: Phase 2 RBAC Database Migration (phase2_create_rbac_tables.sql)
