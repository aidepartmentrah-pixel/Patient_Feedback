"""
Auth Router
API endpoints for authentication and session management.
Phase 2 RBAC Core - Session-based authentication (NO JWT).
"""

from fastapi import APIRouter, HTTPException, Request, status, Depends
from pydantic import BaseModel, Field
from typing import Optional

from ..services.auth_service import (
    login as login_service,
    logout as logout_service,
    get_current_user_from_session,
    require_authentication
)
from ..schemas.auth_models import CurrentUser, UserScope


router = APIRouter(prefix="/api/auth", tags=["Authentication"])


# ==================== REQUEST/RESPONSE MODELS ====================

class LoginRequest(BaseModel):
    """Request model for user login."""
    username: str = Field(
        ...,
        min_length=3,
        max_length=100,
        description="Username for authentication",
        example="software_admin"
    )
    password: str = Field(
        ...,
        min_length=1,
        description="Password for authentication",
        example="AdminPass123!"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "username": "software_admin",
                "password": "admin123"
            }
        }


class LoginResponse(BaseModel):
    """Response model for successful login."""
    success: bool = Field(
        ...,
        description="Indicates successful authentication",
        example=True
    )
    message: str = Field(
        ...,
        description="Success message",
        example="Login successful"
    )
    user: CurrentUser = Field(
        ...,
        description="Authenticated user information with scopes"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Login successful",
                "user": {
                    "user_id": 1,
                    "username": "software_admin",
                    "is_active": True,
                    "scopes": [
                        {
                            "role_code": "SOFTWARE_ADMIN",
                            "org_unit_id": None,
                            "org_unit_type": None
                        }
                    ]
                }
            }
        }


class LogoutResponse(BaseModel):
    """Response model for logout."""
    success: bool = Field(
        ...,
        description="Indicates successful logout",
        example=True
    )
    message: str = Field(
        ...,
        description="Success message",
        example="Logout successful"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Logout successful"
            }
        }


class UserProfileResponse(BaseModel):
    """Response model for current user profile."""
    user: CurrentUser = Field(
        ...,
        description="Current user information with scopes"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "user": {
                    "user_id": 1,
                    "username": "software_admin",
                    "display_name": "Software Administrator",
                    "department_display_name": "IT Department",
                    "is_active": True,
                    "scopes": [
                        {
                            "role_code": "SOFTWARE_ADMIN",
                            "org_unit_id": None,
                            "org_unit_type": None
                        }
                    ]
                }
            }
        }


# ==================== ENDPOINTS ====================

@router.post(
    "/login",
    response_model=LoginResponse,
    status_code=status.HTTP_200_OK,
    summary="User Login",
    responses={
        200: {
            "description": "Login successful - session created",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "message": "Login successful",
                        "user": {
                            "user_id": 1,
                            "username": "software_admin",
                            "is_active": True,
                            "scopes": [
                                {
                                    "role_code": "SOFTWARE_ADMIN",
                                    "org_unit_id": None,
                                    "org_unit_type": None
                                }
                            ]
                        }
                    }
                }
            }
        },
        401: {
            "description": "Authentication failed - invalid credentials",
            "content": {
                "application/json": {
                    "example": {
                        "detail": {
                            "error": "AUTHENTICATION_FAILED",
                            "message": "Invalid username or password",
                            "message_ar": "اسم المستخدم أو كلمة المرور غير صحيحة"
                        }
                    }
                }
            }
        },
        500: {
            "description": "Internal server error"
        }
    }
)
async def login(request: Request, credentials: LoginRequest):
    """
    Authenticate user and create session.
    
    This endpoint validates user credentials using bcrypt password hashing
    and creates a server-side session. Session data is stored in a signed
    cookie named 'incident_manager_session'.
    
    **Authentication Flow:**
    1. Validate username and password against APP_Users table
    2. Load user's role scopes from APP_UserRoleScope
    3. Create session and store user_id
    4. Return user profile with all scopes
    
    **Request Body:**
    - `username`: User's login username (3-100 characters)
    - `password`: User's password (plaintext - hashed by server)
    
    **Session Details:**
    - Session cookie: `incident_manager_session`
    - Max age: 24 hours (86400 seconds)
    - HTTP only: Yes
    - Secure: Recommended for production
    - SameSite: Lax
    
    **Response:**
    - Returns authenticated user with all role scopes
    - Each scope includes: role_code, org_unit_id, org_unit_type
    - Session automatically included in response cookies
    
    **Example Request:**
    ```json
    {
      "username": "software_admin",
      "password": "admin123"
    }
    ```
    
    **Example Success Response (200):**
    ```json
    {
      "success": true,
      "message": "Login successful",
      "user": {
        "user_id": 1,
        "username": "software_admin",
        "is_active": true,
        "scopes": [
          {
            "role_code": "SOFTWARE_ADMIN",
            "org_unit_id": null,
            "org_unit_type": null
          }
        ]
      }
    }
    ```
    
    **Error Response (401 - Invalid Credentials):**
    ```json
    {
      "detail": {
        "error": "AUTHENTICATION_FAILED",
        "message": "Invalid username or password",
        "message_ar": "اسم المستخدم أو كلمة المرور غير صحيحة"
      }
    }
    ```
    
    **Test Users:**
    - `software_admin` / `admin123` - SOFTWARE_ADMIN (global access)
    - `worker` / `worker123` - WORKER (department scoped)
    - `complaint_supervisor` / `sup123` - COMPLAINT_SUPERVISOR
    - `section_admin` / `section123` - SECTION_ADMIN
    - `department_admin` / `dept123` - DEPARTMENT_ADMIN
    - `administration_admin` / `adminis123` - ADMINISTRATION_ADMIN
    
    **Security Notes:**
    - Passwords are hashed with bcrypt (cost factor 12)
    - Sessions are signed with secret key
    - Failed login attempts return generic error message
    - No user enumeration possible
    """
    try:
        # Call service layer to authenticate and create session
        # Service returns: {"message": "login successful", "username": "..."}
        # Session is automatically created with user_id stored
        result = login_service(
            username=credentials.username,
            password=credentials.password,
            request=request
        )
        
        # Login successful - now load full user profile from session
        current_user = get_current_user_from_session(request)
        
        return LoginResponse(
            success=True,
            message="Login successful",
            user=current_user
        )
    
    except HTTPException:
        raise
    except Exception as e:
        # Unexpected errors (500)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": f"Login failed: {str(e)}",
                "message_ar": "خطأ داخلي في النظام"
            }
        )


@router.post(
    "/logout",
    response_model=LogoutResponse,
    status_code=status.HTTP_200_OK,
    summary="User Logout",
    responses={
        200: {
            "description": "Logout successful - session cleared",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "message": "Logout successful"
                    }
                }
            }
        },
        500: {
            "description": "Internal server error"
        }
    }
)
async def logout(request: Request):
    """
    Logout user and clear session.
    
    This endpoint clears the server-side session, effectively logging out
    the user. The session cookie will be invalidated and any subsequent
    requests will require re-authentication.
    
    **Logout Flow:**
    1. Clear all session data (request.session.clear())
    2. Session cookie automatically invalidated
    3. Return success message
    
    **Session Behavior:**
    - Clears all session data including user_id
    - Session cookie remains but is empty
    - User must login again to access protected routes
    
    **Response:**
    - Always returns success (even if no active session)
    - Safe to call multiple times
    
    **Example Request:**
    ```
    POST /api/auth/logout
    Cookie: incident_manager_session=<session-token>
    ```
    
    **Example Success Response (200):**
    ```json
    {
      "success": true,
      "message": "Logout successful"
    }
    ```
    
    **Notes:**
    - No authentication required (can logout even if not logged in)
    - Idempotent operation
    - Does not return user information
    - Session cookie automatically handled by middleware
    """
    try:
        # Call service layer to clear session
        logout_service(request)
        
        return LogoutResponse(
            success=True,
            message="Logout successful"
        )
    
    except Exception as e:
        # Unexpected errors (500)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": f"Logout failed: {str(e)}",
                "message_ar": "خطأ داخلي في النظام"
            }
        )


@router.get(
    "/me",
    response_model=UserProfileResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Current User",
    responses={
        200: {
            "description": "Current user profile retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "user": {
                            "user_id": 1,
                            "username": "software_admin",
                            "display_name": "Software Administrator",
                            "department_display_name": "IT Department",
                            "is_active": True,
                            "scopes": [
                                {
                                    "role_code": "SOFTWARE_ADMIN",
                                    "org_unit_id": None,
                                    "org_unit_type": None
                                }
                            ]
                        }
                    }
                }
            }
        },
        401: {
            "description": "Not authenticated - no active session",
            "content": {
                "application/json": {
                    "example": {
                        "detail": {
                            "error": "NOT_AUTHENTICATED",
                            "message": "Authentication required",
                            "message_ar": "يجب تسجيل الدخول"
                        }
                    }
                }
            }
        },
        500: {
            "description": "Internal server error"
        }
    }
)
async def get_current_user(
    request: Request,
    current_user: CurrentUser = Depends(require_authentication)
):
    """
    Get currently authenticated user profile.
    
    This endpoint returns the profile of the currently authenticated user
    by reading the session cookie and loading user data from the database.
    
    **Authentication Required:**
    - Must have valid session cookie
    - Session must contain valid user_id
    - User must exist in APP_Users table
    - User must be active (IsActive = 1)
    
    **Response:**
    - Returns full user profile with all role scopes
    - Includes: user_id, username, is_active, scopes[]
    - Each scope includes: role_code, org_unit_id, org_unit_type
    
    **Use Cases:**
    - Check if user is logged in
    - Load user profile on page load
    - Verify session validity
    - Display user info in UI
    - Authorization checks in frontend
    
    **Example Request:**
    ```
    GET /api/auth/me
    Cookie: incident_manager_session=<session-token>
    ```
    
    **Example Success Response (200):**
    ```json
    {
      "user": {
        "user_id": 2,
        "username": "worker",
        "display_name": "Dr. John Smith",
        "department_display_name": "Emergency Medicine",
        "is_active": true,
        "scopes": [
          {
            "role_code": "WORKER",
            "org_unit_id": 5,
            "org_unit_type": "DEPARTMENT"
          }
        ]
      }
    }
    ```
    
    **Error Response (401 - Not Authenticated):**
    ```json
    {
      "detail": {
        "error": "NOT_AUTHENTICATED",
        "message": "Authentication required",
        "message_ar": "يجب تسجيل الدخول"
      }
    }
    ```
    
    **Session Validation:**
    - Checks for 'user_id' in session
    - Loads user from database with scopes
    - Validates user is active
    - Returns 401 if any validation fails
    
    **Security Notes:**
    - Session automatically validated by middleware
    - User data refreshed from database on each request
    - Inactive users automatically rejected
    - No caching - always fresh data
    """
    try:
        # Dependency already loaded user via require_authentication
        # Just return the user profile
        return UserProfileResponse(user=current_user)
    
    except HTTPException:
        raise
    except Exception as e:
        # Unexpected errors (500)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": f"Failed to get user profile: {str(e)}",
                "message_ar": "خطأ داخلي في النظام"
            }
        )
