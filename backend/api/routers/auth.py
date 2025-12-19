"""
AUTO-GENERATED API CONTRACT

Source: Obsidian – Login / Authentication Page
Iteration: 1
Status: API skeleton only – no implementation
"""

from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, Header
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


# =====================================================
# Request Models
# =====================================================

class LoginRequest(BaseModel):
    """
    Login credentials.
    Identifier may represent username, employee_id, or email.
    """
    identifier: str = Field(..., description="Username, employee ID, or email")
    password: str = Field(..., description="User password")


# =====================================================
# User / Auth Models
# =====================================================

class AuthenticatedUser(BaseModel):
    user_id: int | str
    username: str
    full_name: str
    full_name_ar: Optional[str] = None
    email: Optional[str] = None
    department: Optional[str] = None
    role: Optional[str | List[str]] = None


class LoginSuccessResponse(BaseModel):
    success: bool = True
    token: str
    user: AuthenticatedUser
    expires_at: Optional[datetime] = None


class LoginErrorResponse(BaseModel):
    success: bool = False
    error: str
    message: str
    fields: Optional[List[str]] = None


# =====================================================
# Token Validation Models
# =====================================================

class TokenValidationResponse(BaseModel):
    valid: bool
    user: Optional[AuthenticatedUser] = None
    expires_at: Optional[datetime] = None
    error: Optional[str] = None
    message: Optional[str] = None


# =====================================================
# Logout Models
# =====================================================

class LogoutResponse(BaseModel):
    success: bool
    message: str


# =====================================================
# Routes
# =====================================================

@router.post(
    "/login",
    response_model=LoginSuccessResponse,
)
def login(request: LoginRequest):
    """
    Authenticate user credentials and issue an authentication token.
    """
    raise NotImplementedError


@router.get(
    "/validate",
    response_model=TokenValidationResponse,
)
def validate_token(
    authorization: Optional[str] = Header(None, description="Bearer authentication token"),
):
    """
    Validate an existing authentication token.
    """
    raise NotImplementedError


@router.post(
    "/logout",
    response_model=LogoutResponse,
)
def logout(
    authorization: Optional[str] = Header(None, description="Bearer authentication token"),
):
    """
    Invalidate authentication token (if applicable).
    """
    raise NotImplementedError
