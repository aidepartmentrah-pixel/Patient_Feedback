"""
User Context Dependency
FastAPI dependency for authentication and user context injection.

This module provides the get_current_user() dependency that:
- Validates user sessions
- Loads user data from database
- Injects CurrentUser into protected endpoints

Phase 2 RBAC: Session-based authentication (NO JWT, NO tokens)
"""

from fastapi import Request, HTTPException, status
from ..schemas.auth_models import CurrentUser
from ..services.auth_service import get_current_user_from_session


def get_current_user(request: Request) -> CurrentUser:
    """
    FastAPI dependency to get currently authenticated user from session.
    
    This function is designed to be used with FastAPI's Depends() injection:
    
        @router.get("/protected")
        def protected_endpoint(current_user: CurrentUser = Depends(get_current_user)):
            # current_user is automatically populated from session
            return {"user": current_user.username}
    
    Authentication Flow:
    1. Check if request.session contains "user_id"
    2. If no user_id: raise HTTP 401 Unauthorized
    3. Load user from database using auth_service
    4. Validate user is active
    5. Return CurrentUser with all role scopes
    
    Session Management:
    - Session cookie: "incident_manager_session"
    - Session created by POST /api/auth/login
    - Session cleared by POST /api/auth/logout
    - Session managed by Starlette SessionMiddleware
    
    Args:
        request: FastAPI Request object (automatically injected by FastAPI)
    
    Returns:
        CurrentUser: Authenticated user with user_id, username, is_active, scopes[]
    
    Raises:
        HTTPException(401): If not authenticated or session invalid
            - NOT_AUTHENTICATED: No active session
            - USER_NOT_FOUND: User doesn't exist in database
            - USER_INACTIVE: User account is deactivated
    
    Example Usage:
        ```python
        from fastapi import APIRouter, Depends
        from ..dependencies.user_context import get_current_user
        from ..schemas.auth_models import CurrentUser
        
        router = APIRouter()
        
        @router.get("/api/protected-endpoint")
        def my_endpoint(current_user: CurrentUser = Depends(get_current_user)):
            # Only authenticated users can access this endpoint
            return {
                "message": f"Hello {current_user.username}",
                "roles": [scope.role_code for scope in current_user.scopes]
            }
        ```
    
    Session Validation:
    - Checks session exists and contains user_id
    - Loads fresh user data from database on every request
    - Validates user is still active
    - Returns 401 if any validation fails
    
    Security Notes:
    - NO JWT tokens used
    - NO Authorization headers required
    - Session-only authentication
    - Server-side session storage
    - Automatic session validation on every request
    - User data refreshed from DB (no stale cache)
    
    Error Responses:
    - 401 NOT_AUTHENTICATED: No active session (user not logged in)
    - 401 USER_NOT_FOUND: User deleted or session corrupted
    - 401 USER_INACTIVE: User account deactivated
    """
    try:
        # Delegate to auth_service which handles all validation logic
        # This function already:
        # - Checks session for user_id
        # - Loads user from database
        # - Validates user is active
        # - Returns CurrentUser model
        # - Raises appropriate HTTPExceptions
        current_user = get_current_user_from_session(request)
        return current_user
    
    except HTTPException:
        # Re-raise HTTP exceptions from auth service
        raise
    
    except Exception as e:
        # Unexpected errors - wrap in 500
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "AUTHENTICATION_ERROR",
                "message": f"Failed to authenticate user: {str(e)}",
                "message_ar": "فشل في المصادقة على المستخدم"
            }
        )
