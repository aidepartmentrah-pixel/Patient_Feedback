# Authorization Guards Quick Reference

## Overview

Authorization guards are functions that check if a user has the required role(s) to access an endpoint. They work alongside the authentication system to provide complete access control.

**Key Principles:**
- Guards are **plain functions**, not decorators
- Guards receive `CurrentUser` as argument
- Guards raise `HTTPException(403)` for unauthorized access
- Guards do NOT access request, session, or database directly
- Guards only check roles (no org-unit scoping at this level)

---

## Basic Usage Pattern

```python
from fastapi import APIRouter, Depends
from ..dependencies.user_context import get_current_user
from ..schemas.auth_models import CurrentUser
from ..utils.guards import require_software_admin, require_any_admin

router = APIRouter()

@router.post("/admin-operation")
async def admin_operation(current_user: CurrentUser = Depends(get_current_user)):
    # Check authorization
    require_software_admin(current_user)
    
    # Proceed with operation
    return {"message": "Admin operation performed"}
```

---

## Available Guard Functions

### Core Guards

#### `require_logged_in(current_user)`
Verifies user is authenticated. Useful with optional dependencies.

```python
@router.get("/optional-auth")
def endpoint(current_user: Optional[CurrentUser] = Depends(get_current_user_optional)):
    if some_condition:
        require_logged_in(current_user)
    # Continue
```

#### `require_role(current_user, allowed_roles)`
Check if user has at least one of the specified roles.

```python
# Single role
require_role(current_user, [SOFTWARE_ADMIN])

# Multiple roles
require_role(current_user, [SOFTWARE_ADMIN, SECTION_ADMIN, DEPARTMENT_ADMIN])
```

### Role-Specific Guards

Convenience functions for specific roles:

```python
require_software_admin(current_user)        # SOFTWARE_ADMIN only
require_worker(current_user)                # WORKER only
require_complaint_supervisor(current_user)  # COMPLAINT_SUPERVISOR only
require_section_admin(current_user)         # SECTION_ADMIN only
require_department_admin(current_user)      # DEPARTMENT_ADMIN only
require_administration_admin(current_user)  # ADMINISTRATION_ADMIN only
```

### Combined Guards

```python
require_any_admin(current_user)       # Any admin role
require_any_supervisor(current_user)  # Supervisor or admin roles
```

---

## Helper Functions (Non-Throwing)

For conditional logic without raising exceptions:

### `has_role(current_user, role_code) -> bool`
Check if user has a specific role.

```python
if has_role(current_user, SOFTWARE_ADMIN):
    # Show admin features
else:
    # Show limited features
```

### `has_any_role(current_user, role_codes) -> bool`
Check if user has any of the specified roles.

```python
if has_any_role(current_user, [SOFTWARE_ADMIN, SECTION_ADMIN]):
    # User is an admin
```

### `get_user_roles(current_user) -> list[str]`
Get list of all roles for the user.

```python
roles = get_user_roles(current_user)
# Returns: ["SOFTWARE_ADMIN", "SECTION_ADMIN"]
```

---

## Common Patterns

### Pattern 1: Simple Role Check

```python
@router.delete("/delete-resource/{id}")
async def delete_resource(
    id: int,
    current_user: CurrentUser = Depends(get_current_user)
):
    require_software_admin(current_user)
    # Delete logic here
    return {"message": "Deleted"}
```

### Pattern 2: Multiple Allowed Roles

```python
@router.post("/approve-complaint")
async def approve_complaint(
    current_user: CurrentUser = Depends(get_current_user)
):
    require_any_supervisor(current_user)
    # Approval logic here
```

### Pattern 3: Conditional Access

```python
@router.get("/reports")
async def get_reports(current_user: CurrentUser = Depends(get_current_user)):
    if has_role(current_user, SOFTWARE_ADMIN):
        # Return all reports
        return {"reports": all_reports}
    else:
        # Return filtered reports
        return {"reports": filtered_reports}
```

### Pattern 4: Multi-Level Authorization

```python
@router.delete("/critical-resource/{id}")
async def delete_critical(
    id: int,
    current_user: CurrentUser = Depends(get_current_user)
):
    # First check: must be admin
    require_any_admin(current_user)
    
    # Second check: extra critical resources need SOFTWARE_ADMIN
    if id <= 100:  # Critical IDs
        require_software_admin(current_user)
    
    # Delete logic
    return {"message": "Deleted"}
```

### Pattern 5: Permission Metadata

```python
@router.get("/my-permissions")
async def get_permissions(current_user: CurrentUser = Depends(get_current_user)):
    return {
        "user": current_user.username,
        "roles": get_user_roles(current_user),
        "can_delete": has_role(current_user, SOFTWARE_ADMIN),
        "can_approve": has_any_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR]),
        "can_view_reports": True  # All authenticated users
    }
```

---

## Error Responses

### 401 Unauthorized (Not Authenticated)

```json
{
  "detail": {
    "error": "NOT_AUTHENTICATED",
    "message": "Authentication required. Please log in.",
    "message_ar": "المصادقة مطلوبة. الرجاء تسجيل الدخول"
  }
}
```

### 403 Forbidden (Not Authorized)

```json
{
  "detail": {
    "error": "FORBIDDEN",
    "message": "Access denied. Required role: SOFTWARE_ADMIN",
    "message_ar": "تم رفض الوصول. الدور المطلوب: SOFTWARE_ADMIN",
    "required_roles": ["SOFTWARE_ADMIN"],
    "user_roles": ["WORKER"]
  }
}
```

---

## Role Constants

Import from `core.constants.roles`:

```python
from core.constants.roles import (
    SOFTWARE_ADMIN,
    WORKER,
    COMPLAINT_SUPERVISOR,
    SECTION_ADMIN,
    DEPARTMENT_ADMIN,
    ADMINISTRATION_ADMIN,
    ADMIN_ROLES,      # All admin roles
    SUPERVISOR_ROLES, # Supervisor + admin roles
    ALL_ROLES        # All roles
)
```

---

## Complete Example

```python
"""
Example router with various authorization patterns.
"""
from fastapi import APIRouter, Depends, HTTPException
from ..dependencies.user_context import get_current_user
from ..schemas.auth_models import CurrentUser
from ..utils.guards import (
    require_software_admin,
    require_any_admin,
    has_role,
    get_user_roles
)
from core.constants.roles import SOFTWARE_ADMIN

router = APIRouter(prefix="/api/complaints", tags=["Complaints"])

@router.get("/")
async def list_complaints(current_user: CurrentUser = Depends(get_current_user)):
    """Any authenticated user can list complaints."""
    # No guard needed - authentication via get_current_user is sufficient
    return {"complaints": [...]}

@router.post("/")
async def create_complaint(current_user: CurrentUser = Depends(get_current_user)):
    """Any authenticated user can create complaints."""
    return {"message": "Complaint created"}

@router.put("/{id}/approve")
async def approve_complaint(
    id: int,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Only supervisors and admins can approve."""
    from ..utils.guards import require_any_supervisor
    require_any_supervisor(current_user)
    
    return {"message": "Complaint approved"}

@router.delete("/{id}")
async def delete_complaint(
    id: int,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Only SOFTWARE_ADMIN can delete complaints."""
    require_software_admin(current_user)
    
    return {"message": "Complaint deleted"}

@router.get("/{id}/details")
async def get_complaint_details(
    id: int,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Return different detail levels based on role."""
    complaint = get_complaint_from_db(id)
    
    if has_role(current_user, SOFTWARE_ADMIN):
        # Admin sees everything
        return {
            "complaint": complaint,
            "internal_notes": complaint["internal_notes"],
            "audit_log": complaint["audit_log"]
        }
    else:
        # Regular users see limited info
        return {
            "complaint": {
                "id": complaint["id"],
                "description": complaint["description"],
                "status": complaint["status"]
            }
        }
```

---

## Testing Guards

```python
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

# Login as admin
client.post("/api/auth/login", json={
    "username": "software_admin",
    "password": "admin123"
})

# Should succeed
response = client.delete("/api/complaints/123")
assert response.status_code == 200

# Login as worker
client.post("/api/auth/login", json={
    "username": "worker",
    "password": "worker123"
})

# Should fail with 403
response = client.delete("/api/complaints/123")
assert response.status_code == 403
assert "FORBIDDEN" in response.json()["detail"]["error"]
```

---

## Best Practices

### ✅ DO:
- Call guards early in the endpoint function
- Use role-specific guards for clarity (`require_software_admin` vs `require_role`)
- Use helper functions for conditional logic
- Return consistent error responses
- Document required roles in endpoint docstrings
- Test both authorized and unauthorized access

### ❌ DON'T:
- Don't catch guard exceptions (let them propagate to FastAPI)
- Don't modify CurrentUser in guards
- Don't access database in guards
- Don't implement business logic in guards
- Don't use guards for org-unit scoping (that's for later phases)

---

## Next Steps

**Current Scope:** Role-based authorization (guards check roles only)

**Future Scope (Phase 3):** Org-unit scoping will add another layer:
```python
# Future: Guards + Scoping
require_software_admin(current_user)
require_scope(current_user, org_unit_id=123, org_unit_type="SECTION")
```

For now, guards only verify roles. Org-unit restrictions will be added in a future phase.

---

## Troubleshooting

### Guard raises 403 but user should have access

**Check:**
1. User actually has the required role in database
2. Session is valid and user is logged in
3. Role codes match exactly (case-sensitive)

```python
# Debug: Print user roles
print(get_user_roles(current_user))
```

### Guard not being called

**Check:**
1. Guard is called AFTER `current_user = Depends(get_current_user)`
2. Guard is called in the function body, not in decorator
3. No early returns before guard

### Wrong error code

**Guards return:**
- 401: Not authenticated (from `get_current_user` dependency)
- 403: Not authorized (from guard functions)

If you're getting 401 when you expect 403, the user isn't logged in.

---

## Summary

Guards provide a clean, testable way to implement role-based authorization:

1. **Import guard functions** from `api.utils.guards`
2. **Ensure authentication** via `Depends(get_current_user)`
3. **Call guard early** in endpoint function
4. **Let exceptions propagate** - FastAPI handles them
5. **Test thoroughly** - both success and failure cases

Guards are the foundation of your authorization system. Use them consistently across all protected endpoints!
