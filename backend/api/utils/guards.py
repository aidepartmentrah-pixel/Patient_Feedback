"""
Authorization Guards
Role-Based Access Control (RBAC) authorization functions.

This module provides guard functions for protecting API endpoints based on user roles.
Guards are plain functions that check if a user has the required role(s) to access
a resource.

Phase 2 RBAC: Authorization layer for role-based access control.

Design Principles:
- Guards are plain functions, not decorators
- Guards receive CurrentUser as argument
- Guards raise HTTPException(403) for forbidden access
- Guards raise HTTPException(401) for authentication failures
- Guards do NOT access request, session, or database
- Guards do NOT implement org-unit scoping (future phases)
- Guards ONLY check roles

Usage Example:
    ```python
    from fastapi import APIRouter, Depends
    from api.dependencies.user_context import get_current_user
    from api.schemas.auth_models import CurrentUser
    from api.utils.guards import require_software_admin
    
    router = APIRouter()
    
    @router.post("/admin-only-endpoint")
    def admin_endpoint(current_user: CurrentUser = Depends(get_current_user)):
        # Check authorization
        require_software_admin(current_user)
        
        # If we reach here, user is authorized
        return {"message": "Admin action performed"}
    ```
"""

from fastapi import HTTPException, status, Depends
from typing import Optional
from core.constants.roles import (
    SOFTWARE_ADMIN,
    WORKER,
    COMPLAINT_SUPERVISOR,
    SECTION_ADMIN,
    DEPARTMENT_ADMIN,
    ADMINISTRATION_ADMIN,
)
from ..schemas.auth_models import CurrentUser, UserScope


# ==================== BASE GUARD FUNCTIONS ====================

def require_logged_in(current_user: Optional[CurrentUser]) -> None:
    """
    Verify that user is authenticated.
    
    This is a basic guard that ensures a user is logged in. It's useful for
    endpoints that accept the current_user as optional but still need to verify
    authentication for certain operations.
    
    Args:
        current_user: CurrentUser object or None
    
    Raises:
        HTTPException(401): If current_user is None (not authenticated)
    
    Example:
        ```python
        @router.get("/optional-auth")
        def endpoint(current_user: Optional[CurrentUser] = Depends(get_current_user_optional)):
            if some_condition:
                require_logged_in(current_user)
            # Continue with logic
        ```
    
    Note:
        If you're using Depends(get_current_user), this guard is redundant since
        the dependency already ensures authentication. This is mainly useful with
        optional dependencies.
    """
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "NOT_AUTHENTICATED",
                "message": "Authentication required. Please log in.",
                "message_ar": "المصادقة مطلوبة. الرجاء تسجيل الدخول"
            }
        )


def require_role(current_user: CurrentUser, allowed_roles: list[str]) -> None:
    """
    Verify that user has one of the allowed roles.
    
    This is the core authorization function. It checks if the user has any role
    that matches the allowed_roles list. If the user has multiple role scopes,
    ANY matching role will grant access.
    
    Args:
        current_user: Authenticated user with role scopes
        allowed_roles: List of role codes that are allowed access
    
    Raises:
        HTTPException(403): If user doesn't have any of the required roles
    
    Example:
        ```python
        # Allow both SOFTWARE_ADMIN and SECTION_ADMIN
        require_role(current_user, [SOFTWARE_ADMIN, SECTION_ADMIN])
        ```
    
    Note:
        - Checks ALL of user's scopes, not just the first one
        - User may have multiple roles across different org units
        - Any matching role grants access
        - Case-sensitive role comparison
    """
    # Extract all role codes from user's scopes
    user_roles = [scope.role_code for scope in current_user.scopes]
    
    # Check if user has any of the allowed roles
    has_required_role = any(role in allowed_roles for role in user_roles)
    
    if not has_required_role:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "FORBIDDEN",
                "message": f"Access denied. Required role: {', '.join(allowed_roles)}",
                "message_ar": "تم رفض الوصول. ليس لديك الصلاحيات المطلوبة",
                "required_roles": allowed_roles,
                "user_roles": user_roles
            }
        )


# ==================== ROLE-SPECIFIC GUARD SHORTCUTS ====================

def require_software_admin(current_user: CurrentUser) -> None:
    """
    Require SOFTWARE_ADMIN role.
    
    Software Administrators have full system access. This is the highest
    privilege role in the system.
    
    Permissions:
    - Full system access
    - User management
    - System configuration
    - All data access
    
    Args:
        current_user: Authenticated user
    
    Raises:
        HTTPException(403): If user is not a SOFTWARE_ADMIN
    
    Example:
        ```python
        @router.delete("/users/{user_id}")
        def delete_user(
            user_id: int,
            current_user: CurrentUser = Depends(get_current_user)
        ):
            require_software_admin(current_user)
            # Only SOFTWARE_ADMIN can delete users
            return delete_user_service(user_id)
        ```
    """
    require_role(current_user, [SOFTWARE_ADMIN])


def require_worker(current_user: CurrentUser) -> None:
    """
    Require WORKER role.
    
    Workers are basic complaint handlers with limited permissions within
    their assigned organizational unit.
    
    Permissions:
    - View complaints in assigned org unit
    - Update complaint status
    - Add notes and actions
    
    Args:
        current_user: Authenticated user
    
    Raises:
        HTTPException(403): If user is not a WORKER
    
    Example:
        ```python
        @router.post("/complaints/{id}/update")
        def update_complaint(
            id: int,
            current_user: CurrentUser = Depends(get_current_user)
        ):
            require_worker(current_user)
            # Workers can update complaints
            return update_complaint_service(id)
        ```
    """
    require_role(current_user, [WORKER])


def require_complaint_supervisor(current_user: CurrentUser) -> None:
    """
    Require COMPLAINT_SUPERVISOR role.
    
    Complaint Supervisors oversee complaint handling and can approve actions.
    
    Permissions:
    - All WORKER permissions
    - Approve/reject complaint resolutions
    - Access supervisor reports
    - Reassign complaints
    
    Args:
        current_user: Authenticated user
    
    Raises:
        HTTPException(403): If user is not a COMPLAINT_SUPERVISOR
    
    Example:
        ```python
        @router.post("/complaints/{id}/approve")
        def approve_complaint(
            id: int,
            current_user: CurrentUser = Depends(get_current_user)
        ):
            require_complaint_supervisor(current_user)
            # Supervisors can approve complaints
            return approve_complaint_service(id)
        ```
    """
    require_role(current_user, [COMPLAINT_SUPERVISOR])


def require_section_admin(current_user: CurrentUser) -> None:
    """
    Require SECTION_ADMIN role.
    
    Section Administrators manage a section within a department.
    
    Permissions:
    - Manage users in their section
    - Access section-level reports
    - Configure section settings
    - All supervisor permissions
    
    Args:
        current_user: Authenticated user
    
    Raises:
        HTTPException(403): If user is not a SECTION_ADMIN
    
    Example:
        ```python
        @router.get("/sections/{id}/reports")
        def get_section_reports(
            id: int,
            current_user: CurrentUser = Depends(get_current_user)
        ):
            require_section_admin(current_user)
            # Section admins can view section reports
            return get_reports_service(id)
        ```
    """
    require_role(current_user, [SECTION_ADMIN])


def require_department_admin(current_user: CurrentUser) -> None:
    """
    Require DEPARTMENT_ADMIN role.
    
    Department Administrators manage an entire department.
    
    Permissions:
    - Manage users in their department
    - Access department-level reports
    - Configure department settings
    - All section admin permissions
    
    Args:
        current_user: Authenticated user
    
    Raises:
        HTTPException(403): If user is not a DEPARTMENT_ADMIN
    
    Example:
        ```python
        @router.post("/departments/{id}/settings")
        def update_department_settings(
            id: int,
            current_user: CurrentUser = Depends(get_current_user)
        ):
            require_department_admin(current_user)
            # Department admins can update settings
            return update_settings_service(id)
        ```
    """
    require_role(current_user, [DEPARTMENT_ADMIN])


def require_administration_admin(current_user: CurrentUser) -> None:
    """
    Require ADMINISTRATION_ADMIN role.
    
    Administration Administrators have broad administrative access across
    the entire organization.
    
    Permissions:
    - Manage users across organization
    - Access organization-wide reports
    - Configure high-level settings
    - All department admin permissions
    
    Args:
        current_user: Authenticated user
    
    Raises:
        HTTPException(403): If user is not an ADMINISTRATION_ADMIN
    
    Example:
        ```python
        @router.get("/organization/statistics")
        def get_org_statistics(
            current_user: CurrentUser = Depends(get_current_user)
        ):
            require_administration_admin(current_user)
            # Administration admins can view org stats
            return get_statistics_service()
        ```
    """
    require_role(current_user, [ADMINISTRATION_ADMIN])


# ==================== COMBINED GUARD FUNCTIONS ====================

def require_any_admin(current_user: CurrentUser) -> None:
    """
    Require any administrative role.
    
    Allows access to users with any admin-level role: SOFTWARE_ADMIN,
    SECTION_ADMIN, DEPARTMENT_ADMIN, or ADMINISTRATION_ADMIN.
    
    Args:
        current_user: Authenticated user
    
    Raises:
        HTTPException(403): If user doesn't have an admin role
    
    Example:
        ```python
        @router.get("/admin/dashboard")
        def admin_dashboard(current_user: CurrentUser = Depends(get_current_user)):
            require_any_admin(current_user)
            # Any admin can access dashboard
            return get_dashboard_data()
        ```
    """
    require_role(current_user, [
        SOFTWARE_ADMIN,
        SECTION_ADMIN,
        DEPARTMENT_ADMIN,
        ADMINISTRATION_ADMIN
    ])


def require_any_supervisor(current_user: CurrentUser) -> None:
    """
    Require any supervisory role.
    
    Allows access to users with supervisory or administrative roles:
    COMPLAINT_SUPERVISOR or any admin role.
    
    Args:
        current_user: Authenticated user
    
    Raises:
        HTTPException(403): If user doesn't have a supervisor role
    
    Example:
        ```python
        @router.get("/complaints/pending-approval")
        def get_pending_complaints(
            current_user: CurrentUser = Depends(get_current_user)
        ):
            require_any_supervisor(current_user)
            # Supervisors and admins can view pending complaints
            return get_pending_service()
        ```
    """
    require_role(current_user, [
        SOFTWARE_ADMIN,
        COMPLAINT_SUPERVISOR,
        SECTION_ADMIN,
        DEPARTMENT_ADMIN,
        ADMINISTRATION_ADMIN
    ])


# ==================== HELPER FUNCTIONS ====================

def has_role(current_user: CurrentUser, role_code: str) -> bool:
    """
    Check if user has a specific role (non-throwing).
    
    This is a helper function that returns a boolean instead of raising
    an exception. Useful for conditional logic within endpoints.
    
    Args:
        current_user: Authenticated user
        role_code: Role code to check
    
    Returns:
        True if user has the role, False otherwise
    
    Example:
        ```python
        @router.get("/data")
        def get_data(current_user: CurrentUser = Depends(get_current_user)):
            if has_role(current_user, SOFTWARE_ADMIN):
                # Return all data for admin
                return get_all_data()
            else:
                # Return limited data for others
                return get_limited_data(current_user)
        ```
    """
    user_roles = [scope.role_code for scope in current_user.scopes]
    return role_code in user_roles


def has_any_role(current_user: CurrentUser, role_codes: list[str]) -> bool:
    """
    Check if user has any of the specified roles (non-throwing).
    
    Args:
        current_user: Authenticated user
        role_codes: List of role codes to check
    
    Returns:
        True if user has any of the roles, False otherwise
    
    Example:
        ```python
        @router.get("/reports")
        def get_reports(current_user: CurrentUser = Depends(get_current_user)):
            if has_any_role(current_user, [SOFTWARE_ADMIN, SECTION_ADMIN]):
                return get_detailed_reports()
            else:
                return get_basic_reports()
        ```
    """
    user_roles = [scope.role_code for scope in current_user.scopes]
    return any(role in user_roles for role in role_codes)


def get_user_roles(current_user: CurrentUser) -> list[str]:
    """
    Get list of all roles assigned to user.
    
    Args:
        current_user: Authenticated user
    
    Returns:
        List of role codes
    
    Example:
        ```python
        roles = get_user_roles(current_user)
        print(f"User has roles: {roles}")
        # Output: User has roles: ['SOFTWARE_ADMIN']
        ```
    """
    return [scope.role_code for scope in current_user.scopes]


# ==================== ORGANIZATIONAL SCOPE GUARDS (PHASE 2.5) ====================

def require_unit_in_scope(current_user: CurrentUser, unit_id: int) -> None:
    """
    Verify that user has access to a specific organizational unit.
    
    This guard enforces organizational scoping by checking if the requested
    unit_id is within the user's allowed organizational scope. The scope is
    pre-computed and stored in current_user.allowed_unit_ids.
    
    Phase 2.5: Organizational Scoping Engine
    
    Args:
        current_user: Authenticated user with computed scope
        unit_id: The organizational unit ID to check access for
    
    Raises:
        HTTPException(500): If scope context is not initialized
        HTTPException(403): If user doesn't have access to the unit
    
    Example:
        ```python
        @router.get("/department/{dept_id}/data")
        def get_dept_data(
            dept_id: int,
            current_user: CurrentUser = Depends(get_current_user)
        ):
            # Ensure user has access to this department
            require_unit_in_scope(current_user, dept_id)
            
            # If we reach here, user is authorized for this unit
            return get_department_data(dept_id)
        ```
    
    Note:
        - This guard does NOT compute scope, only enforces it
        - Scope must be pre-computed in current_user.allowed_unit_ids
        - This is a security boundary - violations raise 403
    """
    # Verify scope context is initialized
    if not hasattr(current_user, 'allowed_unit_ids') or current_user.allowed_unit_ids is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "SCOPE_NOT_INITIALIZED",
                "message": "Scope context is not initialized",
                "message_ar": "سياق النطاق غير مهيأ"
            }
        )
    
    # Check if unit is in user's allowed scope
    if unit_id not in current_user.allowed_unit_ids:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "FORBIDDEN",
                "message": "Access to this organizational unit is forbidden",
                "message_ar": "تم رفض الوصول إلى هذه الوحدة التنظيمية",
                "requested_unit_id": unit_id
            }
        )


def require_any_unit_in_scope(
    current_user: CurrentUser,
    unit_ids: list[int] | set[int]
) -> None:
    """
    Verify that user has access to at least one of the specified organizational units.
    
    This guard is useful when querying data that may span multiple organizational
    units, and the user needs access to at least one of them.
    
    Phase 2.5: Organizational Scoping Engine
    
    Args:
        current_user: Authenticated user with computed scope
        unit_ids: List or set of organizational unit IDs to check
    
    Raises:
        HTTPException(500): If scope context is not initialized
        HTTPException(403): If user doesn't have access to any of the units
    
    Example:
        ```python
        @router.get("/multi-unit-report")
        def get_report(
            unit_ids: list[int],
            current_user: CurrentUser = Depends(get_current_user)
        ):
            # Ensure user has access to at least one requested unit
            require_any_unit_in_scope(current_user, unit_ids)
            
            # Filter to only units the user can access
            accessible_units = [
                uid for uid in unit_ids
                if uid in current_user.allowed_unit_ids
            ]
            
            return generate_report(accessible_units)
        ```
    
    Note:
        - Empty unit_ids list is treated as forbidden (raises 403)
        - User only needs access to ONE unit to pass the guard
        - Does not compute scope, only checks pre-computed allowed_unit_ids
    """
    # Verify scope context is initialized
    if not hasattr(current_user, 'allowed_unit_ids') or current_user.allowed_unit_ids is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "SCOPE_NOT_INITIALIZED",
                "message": "Scope context is not initialized",
                "message_ar": "سياق النطاق غير مهيأ"
            }
        )
    
    # Empty list is forbidden
    if not unit_ids:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "FORBIDDEN",
                "message": "Access to this organizational unit is forbidden",
                "message_ar": "تم رفض الوصول إلى هذه الوحدة التنظيمية",
                "requested_unit_ids": []
            }
        )
    
    # Check if user has access to at least one unit
    has_access = any(unit_id in current_user.allowed_unit_ids for unit_id in unit_ids)
    
    if not has_access:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "FORBIDDEN",
                "message": "Access to this organizational unit is forbidden",
                "message_ar": "تم رفض الوصول إلى هذه الوحدة التنظيمية",
                "requested_unit_ids": list(unit_ids) if isinstance(unit_ids, set) else unit_ids
            }
        )


# ==================== FASTAPI DEPENDENCY GUARDS ====================
# Phase C - B-C6: Fast API dependency-style guards for route-level protection

# Import auth service here for dependency guards (after main imports to avoid circular deps)
from ..services.auth_service import get_current_user as _get_current_user


def get_current_software_admin(
    current_user: CurrentUser = Depends(_get_current_user)
) -> CurrentUser:
    """
    FastAPI dependency that ensures the current user is a SOFTWARE_ADMIN.
    
    This combines authentication (get_current_user) and authorization
    (require_software_admin) into a single dependency that can be used
    directly in route definitions.
    
    Args:
        current_user: Authenticated user from get_current_user dependency
        
    Returns:
        CurrentUser: The authenticated and authorized user
        
    Raises:
        HTTPException(401): If user is not authenticated
        HTTPException(403): If user is not a SOFTWARE_ADMIN
        
    Usage:
        ```python
        @router.post("/admin-endpoint")
        def admin_only(
            admin: CurrentUser = Depends(get_current_software_admin)
        ):
            # admin is guaranteed to be a SOFTWARE_ADMIN
            return {"message": "Success"}
        ```
        
    Note:
        This is the preferred pattern for route-level authorization in FastAPI.
        It explicitly declares the authorization requirement in the route signature.
    """
    # Check SOFTWARE_ADMIN permission
    require_software_admin(current_user)
    
    # Return the authorized user
    return current_user


# ==================== PHASE D: SEASONAL REPORT ACCESS GUARDS ====================

def require_doctor_report_access(current_user: CurrentUser) -> None:
    """
    Require permission to access doctor seasonal reports.
    
    Doctor seasonal reports contain sensitive performance data and should only
    be accessible to users with administrative or supervisory roles.
    
    Allowed Roles:
    - SOFTWARE_ADMIN: Full system access
    - ADMINISTRATION_ADMIN: Cross-organizational access
    - DEPARTMENT_ADMIN: Department-level access
    - SECTION_ADMIN: Section-level access
    - COMPLAINT_SUPERVISOR: Supervisory access
    
    Phase D-B10: Permission Guards for Seasonal Export Endpoints
    
    Args:
        current_user: Authenticated user
    
    Raises:
        HTTPException(403): If user doesn't have required role
    
    Example:
        ```python
        @router.get("/doctors/{doctor_id}/seasonal-report")
        def export_doctor_report(
            doctor_id: int,
            current_user: CurrentUser = Depends(get_current_user)
        ):
            require_doctor_report_access(current_user)
            # Generate and return report
            return seasonal_report
        ```
    
    Note:
        - Workers cannot access doctor reports (insufficient privileges)
        - Organizational scoping (if needed) should be applied separately
        - This guard only checks role permissions, not resource ownership
    """
    require_role(current_user, [
        SOFTWARE_ADMIN,
        ADMINISTRATION_ADMIN,
        DEPARTMENT_ADMIN,
        SECTION_ADMIN,
        COMPLAINT_SUPERVISOR
    ])


def require_worker_report_access(current_user: CurrentUser) -> None:
    """
    Require permission to access worker seasonal reports.
    
    Worker seasonal reports contain employee performance data and should only
    be accessible to users with administrative or supervisory roles.
    
    Allowed Roles:
    - SOFTWARE_ADMIN: Full system access
    - ADMINISTRATION_ADMIN: Cross-organizational access
    - DEPARTMENT_ADMIN: Department-level access
    - SECTION_ADMIN: Section-level access
    - COMPLAINT_SUPERVISOR: Supervisory access
    
    Phase D-B10: Permission Guards for Seasonal Export Endpoints
    
    Args:
        current_user: Authenticated user
    
    Raises:
        HTTPException(403): If user doesn't have required role
    
    Example:
        ```python
        @router.get("/workers/{employee_id}/seasonal-report")
        def export_worker_report(
            employee_id: int,
            current_user: CurrentUser = Depends(get_current_user)
        ):
            require_worker_report_access(current_user)
            # Generate and return report
            return seasonal_report
        ```
    
    Note:
        - Workers cannot access other workers' reports (insufficient privileges)
        - Same permission level as doctor reports (consistent policy)
        - Organizational scoping (if needed) should be applied separately
        - This guard only checks role permissions, not resource ownership
    """
    require_role(current_user, [
        SOFTWARE_ADMIN,
        ADMINISTRATION_ADMIN,
        DEPARTMENT_ADMIN,
        SECTION_ADMIN,
        COMPLAINT_SUPERVISOR
    ])
