"""
📋 PHASE F — TEST F-B8 — ACTION LOG ROLE GUARD

Tests for Action Log authorization guard.
Tests role-based access control ONLY.

No scope checking - only role validation.

Tests verify:
- SOFTWARE_ADMIN allowed
- WORKER allowed
- Other roles rejected (SECTION_ADMIN, DEPARTMENT_ADMIN, etc.)
- Empty roles rejected
"""

import pytest
from fastapi import HTTPException
from backend.api_v2.guards.action_log_guards import require_action_log_role


# ============================================================================
# FAKE USER CLASS
# ============================================================================

class FakeUser:
    """
    Minimal fake user for testing guards.
    Only needs roles property.
    """
    def __init__(self, roles=None):
        self.roles = roles or []


# ============================================================================
# TEST 1 — SOFTWARE_ADMIN ALLOWED
# ============================================================================

def test_software_admin_allowed():
    """
    Test that SOFTWARE_ADMIN role is allowed.
    """
    user = FakeUser(roles=["SOFTWARE_ADMIN"])
    
    # Should not raise exception
    result = require_action_log_role(user)
    
    # Should return same user
    assert result == user
    
    print("✅ SOFTWARE_ADMIN allowed")


# ============================================================================
# TEST 2 — WORKER ALLOWED
# ============================================================================

def test_worker_allowed():
    """
    Test that WORKER role is allowed.
    """
    user = FakeUser(roles=["WORKER"])
    
    # Should not raise exception
    result = require_action_log_role(user)
    
    # Should return same user
    assert result == user
    
    print("✅ WORKER allowed")


# ============================================================================
# TEST 3 — BOTH ALLOWED (MULTIPLE ROLES)
# ============================================================================

def test_worker_with_section_admin_allowed():
    """
    Test that user with WORKER + other roles is allowed.
    """
    user = FakeUser(roles=["WORKER", "SECTION_ADMIN"])
    
    # Should not raise exception (has WORKER)
    result = require_action_log_role(user)
    
    # Should return same user
    assert result == user
    
    print("✅ WORKER + SECTION_ADMIN allowed (has WORKER)")


# ============================================================================
# TEST 4 — SECTION_ADMIN REJECTED
# ============================================================================

def test_section_admin_rejected():
    """
    Test that SECTION_ADMIN alone is rejected.
    """
    user = FakeUser(roles=["SECTION_ADMIN"])
    
    # Should raise HTTPException 403
    with pytest.raises(HTTPException) as exc_info:
        require_action_log_role(user)
    
    assert exc_info.value.status_code == 403
    assert "Not authorized" in exc_info.value.detail
    
    print("✅ SECTION_ADMIN rejected")


# ============================================================================
# TEST 5 — DEPARTMENT_ADMIN REJECTED
# ============================================================================

def test_department_admin_rejected():
    """
    Test that DEPARTMENT_ADMIN is rejected.
    """
    user = FakeUser(roles=["DEPARTMENT_ADMIN"])
    
    # Should raise HTTPException 403
    with pytest.raises(HTTPException) as exc_info:
        require_action_log_role(user)
    
    assert exc_info.value.status_code == 403
    assert "Not authorized" in exc_info.value.detail
    
    print("✅ DEPARTMENT_ADMIN rejected")


# ============================================================================
# TEST 6 — EMPTY ROLES REJECTED
# ============================================================================

def test_empty_roles_rejected():
    """
    Test that user with no roles is rejected.
    """
    user = FakeUser(roles=[])
    
    # Should raise HTTPException 403
    with pytest.raises(HTTPException) as exc_info:
        require_action_log_role(user)
    
    assert exc_info.value.status_code == 403
    assert "Not authorized" in exc_info.value.detail
    
    print("✅ Empty roles rejected")


# ============================================================================
# TEST 7 — ADMINISTRATION_ADMIN REJECTED
# ============================================================================

def test_administration_admin_rejected():
    """
    Test that ADMINISTRATION_ADMIN is rejected.
    """
    user = FakeUser(roles=["ADMINISTRATION_ADMIN"])
    
    # Should raise HTTPException 403
    with pytest.raises(HTTPException) as exc_info:
        require_action_log_role(user)
    
    assert exc_info.value.status_code == 403
    assert "Not authorized" in exc_info.value.detail
    
    print("✅ ADMINISTRATION_ADMIN rejected")


# ============================================================================
# TEST 8 — SOFTWARE_ADMIN WITH OTHER ROLES ALLOWED
# ============================================================================

def test_software_admin_with_multiple_roles_allowed():
    """
    Test that SOFTWARE_ADMIN with other roles is allowed.
    """
    user = FakeUser(roles=["SOFTWARE_ADMIN", "DEPARTMENT_ADMIN", "SECTION_ADMIN"])
    
    # Should not raise exception (has SOFTWARE_ADMIN)
    result = require_action_log_role(user)
    
    # Should return same user
    assert result == user
    
    print("✅ SOFTWARE_ADMIN + multiple roles allowed")


# ============================================================================
# TEST 9 — NULL ROLES REJECTED
# ============================================================================

def test_null_roles_rejected():
    """
    Test that user with None roles is rejected.
    """
    user = FakeUser(roles=None)
    
    # Should raise HTTPException 403
    with pytest.raises(HTTPException) as exc_info:
        require_action_log_role(user)
    
    assert exc_info.value.status_code == 403
    assert "Not authorized" in exc_info.value.detail
    
    print("✅ NULL roles rejected")


# ============================================================================
# TEST 10 — CASE SENSITIVITY CHECK
# ============================================================================

def test_role_case_sensitivity():
    """
    Test that role check is case-sensitive (lowercase worker should fail).
    """
    user = FakeUser(roles=["worker"])  # lowercase
    
    # Should raise HTTPException 403 (role must be uppercase WORKER)
    with pytest.raises(HTTPException) as exc_info:
        require_action_log_role(user)
    
    assert exc_info.value.status_code == 403
    
    print("✅ Role check is case-sensitive")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
