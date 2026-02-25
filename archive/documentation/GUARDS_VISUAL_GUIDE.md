# Visual Guide: Authorization Guards in Action

This guide shows exactly how the authorization system works with real examples.

---

## 🎬 Scenario 1: Worker Tries to Delete (DENIED)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. CLIENT: Worker User                                      │
│    Request: DELETE /api/guarded/admin-only                  │
│    Cookie: incident_manager_session=abc123                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. SESSION MIDDLEWARE                                       │
│    - Reads cookie: incident_manager_session=abc123          │
│    - Loads session data: {user_id: 5, username: "worker"}   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. DEPENDENCY: get_current_user()                           │
│    - Calls auth_service.get_current_user_from_session()     │
│    - Returns: CurrentUser(user_id=5, username="worker",     │
│               scopes=[{role="WORKER", ...}])                │
│    ✅ AUTHENTICATION PASSED                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼ current_user
┌─────────────────────────────────────────────────────────────┐
│ 4. ENDPOINT FUNCTION                                        │
│    async def admin_only_endpoint(                           │
│        current_user: CurrentUser = Depends(get_current_user)│
│    ):                                                       │
│        require_software_admin(current_user)  # GUARD        │
│        return {"message": "success"}                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼ current_user.scopes = [{role: "WORKER"}]
┌─────────────────────────────────────────────────────────────┐
│ 5. GUARD: require_software_admin(current_user)              │
│    - Checks: Does user have SOFTWARE_ADMIN role?            │
│    - User roles: ["WORKER"]                                 │
│    - Required: ["SOFTWARE_ADMIN"]                           │
│    - Result: ❌ NO MATCH                                     │
│    - Action: raise HTTPException(403)                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼ EXCEPTION RAISED
┌─────────────────────────────────────────────────────────────┐
│ 6. FASTAPI ERROR HANDLER                                    │
│    - Catches HTTPException(403)                             │
│    - Returns error response                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. CLIENT RECEIVES: HTTP 403 Forbidden                      │
│    {                                                        │
│      "detail": {                                            │
│        "error": "FORBIDDEN",                                │
│        "message": "Access denied. Required: SOFTWARE_ADMIN",│
│        "message_ar": "تم رفض الوصول",                       │
│        "required_roles": ["SOFTWARE_ADMIN"],                │
│        "user_roles": ["WORKER"]                             │
│      }                                                      │
│    }                                                        │
└─────────────────────────────────────────────────────────────┘
```

**Result:** ❌ **REQUEST DENIED** - Worker lacks required role

---

## 🎬 Scenario 2: Software Admin Deletes (ALLOWED)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. CLIENT: Software Admin                                   │
│    Request: DELETE /api/guarded/admin-only                  │
│    Cookie: incident_manager_session=xyz789                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. SESSION MIDDLEWARE                                       │
│    - Reads cookie: incident_manager_session=xyz789          │
│    - Loads session: {user_id: 1, username: "software_admin"}│
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. DEPENDENCY: get_current_user()                           │
│    - Returns: CurrentUser(user_id=1,                        │
│               username="software_admin",                    │
│               scopes=[{role="SOFTWARE_ADMIN", ...}])        │
│    ✅ AUTHENTICATION PASSED                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼ current_user
┌─────────────────────────────────────────────────────────────┐
│ 4. ENDPOINT FUNCTION                                        │
│    async def admin_only_endpoint(                           │
│        current_user: CurrentUser = Depends(get_current_user)│
│    ):                                                       │
│        require_software_admin(current_user)  # GUARD        │
│        return {"message": "success"}                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼ current_user.scopes = [{role: "SOFTWARE_ADMIN"}]
┌─────────────────────────────────────────────────────────────┐
│ 5. GUARD: require_software_admin(current_user)              │
│    - Checks: Does user have SOFTWARE_ADMIN role?            │
│    - User roles: ["SOFTWARE_ADMIN"]                         │
│    - Required: ["SOFTWARE_ADMIN"]                           │
│    - Result: ✅ MATCH FOUND                                  │
│    - Action: return (no exception)                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼ GUARD PASSED
┌─────────────────────────────────────────────────────────────┐
│ 6. BUSINESS LOGIC EXECUTES                                  │
│    - Process deletion                                       │
│    - return {"message": "success"}                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. CLIENT RECEIVES: HTTP 200 OK                             │
│    {                                                        │
│      "message": "Welcome, Software Administrator!",         │
│      "authentication": "required",                          │
│      "authorization": "SOFTWARE_ADMIN only",                │
│      "action": "sensitive admin operation performed"        │
│    }                                                        │
└─────────────────────────────────────────────────────────────┘
```

**Result:** ✅ **REQUEST ALLOWED** - Admin has required role

---

## 🎬 Scenario 3: Unauthenticated User (DENIED)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. CLIENT: No Login                                         │
│    Request: GET /api/guarded/authenticated-only             │
│    Cookie: (none)                                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. SESSION MIDDLEWARE                                       │
│    - Reads cookie: (none)                                   │
│    - Session data: (none)                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. DEPENDENCY: get_current_user()                           │
│    - Calls auth_service.get_current_user_from_session()     │
│    - Session: None                                          │
│    - Result: ❌ NOT AUTHENTICATED                            │
│    - Action: raise HTTPException(401)                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼ EXCEPTION RAISED
┌─────────────────────────────────────────────────────────────┐
│ 4. FASTAPI ERROR HANDLER                                    │
│    - Catches HTTPException(401)                             │
│    - Returns error response                                 │
│    - Endpoint function NEVER CALLED                         │
│    - Guard function NEVER CALLED                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. CLIENT RECEIVES: HTTP 401 Unauthorized                   │
│    {                                                        │
│      "detail": {                                            │
│        "error": "NOT_AUTHENTICATED",                        │
│        "message": "Authentication required. Please log in.",│
│        "message_ar": "المصادقة مطلوبة. الرجاء تسجيل الدخول"  │
│      }                                                      │
│    }                                                        │
└─────────────────────────────────────────────────────────────┘
```

**Result:** ❌ **REQUEST DENIED** - No authentication

---

## 🎬 Scenario 4: Conditional Access (BOTH ALLOWED)

```python
@router.post("/conditional-access")
async def conditional_access(current_user: CurrentUser = Depends(get_current_user)):
    if has_role(current_user, SOFTWARE_ADMIN):
        return {"data": "full", "access_level": "admin"}
    else:
        return {"data": "limited", "access_level": "basic"}
```

### Worker Request:
```
┌──────────────────────────────────────────┐
│ 1. Worker logged in                     │
│    ✅ Authentication: PASS               │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│ 2. Check: has_role(user, SOFTWARE_ADMIN) │
│    User roles: ["WORKER"]                │
│    Result: False (no exception)          │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│ 3. Return limited data                   │
│    {"data": "limited",                   │
│     "access_level": "basic"}             │
└──────────────────────────────────────────┘
```

### Admin Request:
```
┌──────────────────────────────────────────┐
│ 1. Software Admin logged in             │
│    ✅ Authentication: PASS               │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│ 2. Check: has_role(user, SOFTWARE_ADMIN) │
│    User roles: ["SOFTWARE_ADMIN"]        │
│    Result: True (no exception)           │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│ 3. Return full data                      │
│    {"data": "full",                      │
│     "access_level": "admin"}             │
└──────────────────────────────────────────┘
```

**Result:** ✅ **BOTH ALLOWED** - Different responses based on role

---

## 🎬 Scenario 5: Multi-Level Authorization

```python
@router.delete("/dangerous-operation/{resource_id}")
async def dangerous_operation(
    resource_id: int,
    current_user: CurrentUser = Depends(get_current_user)
):
    require_any_admin(current_user)  # First check
    
    if resource_id <= 10:  # Protected resources
        require_software_admin(current_user)  # Second check
    
    return {"message": "Deleted"}
```

### Section Admin Deleting Resource #5:
```
┌─────────────────────────────────────────────┐
│ 1. Section Admin logged in                 │
│    User roles: ["SECTION_ADMIN"]            │
│    Request: DELETE /dangerous-operation/5   │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 2. Guard: require_any_admin(current_user)   │
│    Check: Is user an admin?                 │
│    User roles: ["SECTION_ADMIN"]            │
│    Allowed: [SOFTWARE_ADMIN, SECTION_ADMIN, │
│              DEPARTMENT_ADMIN, ...]         │
│    Result: ✅ PASS                           │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 3. Check: resource_id <= 10                 │
│    resource_id: 5                           │
│    Result: True (protected resource)        │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 4. Guard: require_software_admin(user)      │
│    Check: Is user SOFTWARE_ADMIN?           │
│    User roles: ["SECTION_ADMIN"]            │
│    Required: ["SOFTWARE_ADMIN"]             │
│    Result: ❌ FAIL                           │
│    Action: raise HTTPException(403)         │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 5. CLIENT RECEIVES: HTTP 403 Forbidden      │
│    {                                        │
│      "detail": {                            │
│        "error": "FORBIDDEN",                │
│        "message": "Access denied..."        │
│      }                                      │
│    }                                        │
└─────────────────────────────────────────────┘
```

**Result:** ❌ **DENIED** - Protected resource requires SOFTWARE_ADMIN

### Section Admin Deleting Resource #15:
```
┌─────────────────────────────────────────────┐
│ 1. Section Admin logged in                 │
│    User roles: ["SECTION_ADMIN"]            │
│    Request: DELETE /dangerous-operation/15  │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 2. Guard: require_any_admin(current_user)   │
│    Result: ✅ PASS                           │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 3. Check: resource_id <= 10                 │
│    resource_id: 15                          │
│    Result: False (not protected)            │
│    Action: Skip second guard                │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 4. Business logic executes                  │
│    return {"message": "Deleted"}            │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 5. CLIENT RECEIVES: HTTP 200 OK             │
│    {"message": "Deleted"}                   │
└─────────────────────────────────────────────┘
```

**Result:** ✅ **ALLOWED** - Non-protected resource, any admin can delete

---

## 📋 Quick Reference Table

| Scenario | Authentication | Authorization | Result | HTTP Code |
|----------|----------------|---------------|--------|-----------|
| No login → Protected endpoint | ❌ | N/A | DENIED | 401 |
| Worker → Admin endpoint | ✅ | ❌ | DENIED | 403 |
| Admin → Admin endpoint | ✅ | ✅ | ALLOWED | 200 |
| Worker → Worker endpoint | ✅ | ✅ | ALLOWED | 200 |
| Admin → Worker endpoint | ✅ | ❌ | DENIED | 403 |
| Worker → Any-user endpoint | ✅ | ✅ | ALLOWED | 200 |
| Admin → Any-user endpoint | ✅ | ✅ | ALLOWED | 200 |

---

## 🔑 Key Takeaways

1. **Authentication (401) happens FIRST** - via `get_current_user` dependency
2. **Authorization (403) happens SECOND** - via guard functions
3. **Guards don't throw for checking** - use `has_role()` for conditionals
4. **Guards throw for enforcement** - use `require_role()` to block
5. **Multiple guards can be used** - for complex authorization logic
6. **Errors are consistent** - bilingual, structured, informative

---

## 💡 Code Patterns at a Glance

### Pattern: Strict Role Check
```python
@router.delete("/admin-only")
async def endpoint(current_user: CurrentUser = Depends(get_current_user)):
    require_software_admin(current_user)  # Throws 403 if not admin
    # Only SOFTWARE_ADMIN reaches here
```

### Pattern: Multiple Allowed Roles
```python
@router.post("/approve")
async def endpoint(current_user: CurrentUser = Depends(get_current_user)):
    require_any_supervisor(current_user)  # Supervisors OR admins
    # Supervisors and admins reach here
```

### Pattern: Conditional Logic
```python
@router.get("/data")
async def endpoint(current_user: CurrentUser = Depends(get_current_user)):
    if has_role(current_user, SOFTWARE_ADMIN):  # No exception
        return full_data
    else:
        return limited_data
```

### Pattern: Multi-Level Check
```python
@router.delete("/resource/{id}")
async def endpoint(id: int, current_user: CurrentUser = Depends(get_current_user)):
    require_any_admin(current_user)  # First gate
    if id <= 10:
        require_software_admin(current_user)  # Second gate
    # Execute delete
```

---

This visual guide shows exactly how guards work in practice. Use these patterns when implementing your own protected endpoints!
