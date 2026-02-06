"""
Authentication Service Layer
Business logic for session-based authentication.

This module implements SESSION-BASED authentication (NOT JWT, NOT tokens).
All authentication state is stored server-side in sessions.

Key Features:
- Username/password login with session creation
- Session-based user context retrieval
- Logout with session cleanup
- No tokens, no JWT, no Authorization headers

Session Storage:
- request.session["user_id"] - Stores authenticated user ID
- Session managed by Starlette SessionMiddleware
"""

from fastapi import HTTPException, Request, status
from typing import Dict, Any, List, Optional

from ..db_layer.auth_db import (
    validate_user_credentials,
    get_user_with_scopes
)
from ..schemas.auth_models import CurrentUser, UserScope
from .scope_resolver import resolve_user_scope


# ==================== HELPER FUNCTIONS ====================

def _select_primary_scope(scopes: List[UserScope]) -> Optional[UserScope]:
    """
    Select primary organizational unit from user scopes (Phase 4).
    
    Deterministic selection rules:
    1. If exactly one scope has non-null org_unit_id → use it
    2. If multiple scopes → prefer highest level: ADMINISTRATION > DEPARTMENT > SECTION
    3. If all scopes have null org_unit_id (e.g., SOFTWARE_ADMIN) → return None
    
    Args:
        scopes: List of user scopes
    
    Returns:
        Primary scope or None if no org_unit_id is present
    """
    # Filter scopes with non-null org_unit_id
    org_scopes = [s for s in scopes if s.org_unit_id is not None]
    
    if not org_scopes:
        # All scopes have null org_unit_id (e.g., SOFTWARE_ADMIN)
        return None
    
    if len(org_scopes) == 1:
        # Exactly one scope with org_unit_id
        return org_scopes[0]
    
    # Multiple scopes - select by priority
    priority = {
        "ADMINISTRATION": 1,
        "DEPARTMENT": 2,
        "SECTION": 3
    }
    
    # Sort by priority (lower number = higher priority)
    sorted_scopes = sorted(
        org_scopes,
        key=lambda s: priority.get(s.org_unit_type, 999)  # Unknown types get lowest priority
    )
    
    return sorted_scopes[0]


# ==================== LOGIN ====================

def login(username: str, password: str, request: Request) -> Dict[str, str]:
    """
    Authenticate user and create session.
    
    This function:
    1. Validates username and password against database
    2. Creates server-side session
    3. Stores user_id in session
    4. Returns success message
    
    Args:
        username: Username to authenticate
        password: Plain text password
        request: FastAPI Request object (for session access)
    
    Returns:
        Success message dict: {"message": "login successful", "username": "..."}
    
    Raises:
        HTTPException(401): If credentials are invalid or user is inactive
        HTTPException(500): If unexpected error occurs
    
    Example:
        >>> login("section_admin", "section123", request)
        {"message": "login successful", "username": "section_admin"}
    """
    try:
        # Validate credentials using DB layer
        user_data = validate_user_credentials(username, password)
        
        if user_data is None:
            # Invalid credentials or inactive user
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error": "INVALID_CREDENTIALS",
                    "message": "Invalid username or password",
                    "message_ar": "اسم المستخدم أو كلمة المرور غير صحيحة"
                }
            )
        
        # Credentials valid! Create session
        request.session["user_id"] = user_data["user_id"]
        
        return {
            "message": "login successful",
            "username": user_data["username"]
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    
    except Exception as e:
        # Unexpected error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "LOGIN_ERROR",
                "message": f"Login failed: {str(e)}",
                "message_ar": "فشل تسجيل الدخول"
            }
        )


# ==================== LOGOUT ====================

def logout(request: Request) -> None:
    """
    Clear user session and log out.
    
    This function:
    1. Clears all session data
    2. Destroys the session
    
    Args:
        request: FastAPI Request object (for session access)
    
    Returns:
        None
    
    Example:
        >>> logout(request)
        # Session cleared
    """
    try:
        # Clear entire session
        request.session.clear()
    
    except Exception as e:
        # Log error but don't fail - logout should always succeed
        # In production, this would go to a logging service
        print(f"Warning: Error during logout: {str(e)}")


# ==================== GET CURRENT USER ====================

def get_current_user_from_session(request: Request) -> CurrentUser:
    """
    Get currently authenticated user from session.
    
    This function:
    1. Reads user_id from session
    2. Loads full user data with scopes from database
    3. Converts to CurrentUser model
    4. Returns authenticated user context
    
    This is the PRIMARY way to get the current user in the application.
    Use as a dependency in protected routes.
    
    Args:
        request: FastAPI Request object (for session access)
    
    Returns:
        CurrentUser object with all scopes
    
    Raises:
        HTTPException(401): If no session exists or user not found
        HTTPException(500): If unexpected error occurs
    
    Example:
        >>> user = get_current_user_from_session(request)
        >>> print(user.username, user.scopes)
        section_admin [UserScope(role_code='SECTION_ADMIN', ...)]
    """
    try:
        # Check if session exists and has user_id
        user_id = request.session.get("user_id")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error": "NOT_AUTHENTICATED",
                    "message": "No active session. Please log in.",
                    "message_ar": "لا توجد جلسة نشطة. الرجاء تسجيل الدخول"
                }
            )
        
        # Load user with scopes from database
        user_data = get_user_with_scopes(user_id)
        
        if user_data is None:
            # User not found (deleted or invalid session)
            # Clear invalid session
            request.session.clear()
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error": "USER_NOT_FOUND",
                    "message": "User account not found. Session cleared.",
                    "message_ar": "لم يتم العثور على حساب المستخدم"
                }
            )
        
        # Check if user is active
        if not user_data["is_active"]:
            # User account deactivated
            # Clear session
            request.session.clear()
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error": "USER_INACTIVE",
                    "message": "User account is inactive",
                    "message_ar": "حساب المستخدم غير نشط"
                }
            )
        
        # Convert to CurrentUser model
        scopes_list = [
            UserScope(
                role_code=scope["role_code"],
                org_unit_id=scope["org_unit_id"],
                org_unit_type=scope["org_unit_type"]
            )
            for scope in user_data["scopes"]
        ]
        
        current_user = CurrentUser(
            user_id=user_data["user_id"],
            username=user_data["username"],
            display_name=user_data.get("display_name") or user_data["username"],
            department_display_name=user_data.get("department_display_name"),
            is_active=user_data["is_active"],
            scopes=scopes_list
        )
        
        # Compute effective organizational scope for this user
        # This happens once per request and fails fast on misconfiguration
        current_user.allowed_unit_ids = resolve_user_scope(current_user)
        
        # Phase 4: Compute derived fields for frontend consumption
        # Extract unique role codes from scopes
        current_user.roles = list(set(scope.role_code for scope in scopes_list))
        
        # Determine primary unit (deterministic selection)
        primary_scope = _select_primary_scope(scopes_list)
        if primary_scope:
            current_user.primary_unit_id = primary_scope.org_unit_id
            current_user.primary_unit_type = primary_scope.org_unit_type
        else:
            current_user.primary_unit_id = None
            current_user.primary_unit_type = None
        
        return current_user
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    
    except Exception as e:
        # Unexpected error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "SESSION_ERROR",
                "message": f"Failed to load user from session: {str(e)}",
                "message_ar": "فشل في تحميل بيانات المستخدم"
            }
        )


# ==================== HELPER FUNCTIONS ====================

def require_authentication(request: Request) -> CurrentUser:
    """
    Dependency function to require authentication.
    
    Use this as a FastAPI dependency for protected routes:
    
    Example:
        @router.get("/protected")
        async def protected_route(
            user: CurrentUser = Depends(require_authentication)
        ):
            return {"message": f"Hello {user.username}"}
    
    Args:
        request: FastAPI Request object
    
    Returns:
        CurrentUser if authenticated
    
    Raises:
        HTTPException(401): If not authenticated
    """
    return get_current_user_from_session(request)


def get_current_user_optional(request: Request) -> CurrentUser | None:
    """
    Get current user from session, or None if not authenticated.
    
    Use this for routes that optionally support authentication.
    
    Args:
        request: FastAPI Request object
    
    Returns:
        CurrentUser if authenticated, None otherwise
    """
    try:
        return get_current_user_from_session(request)
    except HTTPException:
        return None


# Alias for convenience - most routers use this name
get_current_user = get_current_user_from_session
