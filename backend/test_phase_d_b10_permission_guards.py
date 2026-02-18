"""
Phase D - Task B10: Permission Guards for Seasonal Export Endpoints

Tests for role-based authorization guards that protect doctor and worker seasonal report endpoints.
Verifies that only users with admin/supervisor roles can access seasonal reports.

Test Coverage:
- Guard function behavior for different roles
- Access granted for authorized roles
- Access denied for unauthorized roles (403)
- Guard integration in router endpoints
"""

import pytest
from fastapi import HTTPException

from backend.api.utils.guards import (
    require_doctor_report_access,
    require_worker_report_access
)
from backend.api.schemas.auth_models import CurrentUser, UserScope
from core.constants.roles import (
    SOFTWARE_ADMIN,
    WORKER,
    COMPLAINT_SUPERVISOR,
    SECTION_ADMIN,
    DEPARTMENT_ADMIN,
    ADMINISTRATION_ADMIN,
)


# ==================== HELPER FUNCTIONS ====================

def create_user_with_role(role_code: str) -> CurrentUser:
    """Create a mock CurrentUser with specified role."""
    return CurrentUser(
        user_id=1,
        username="test_user",
        is_active=True,
        scopes=[
            UserScope(
                role_code=role_code,
                role_name=f"Role {role_code}",
                orgunit_id=1,
                orgunit_name="Test Unit",
                orgunit_type="SECTION"
            )
        ]
    )


# ==================== DOCTOR REPORT GUARD TESTS ====================

class TestDoctorReportAccessGuard:
    """Tests for require_doctor_report_access() guard function."""
    
    def test_software_admin_granted(self):
        """SOFTWARE_ADMIN can access doctor reports."""
        user = create_user_with_role(SOFTWARE_ADMIN)
        
        # Should not raise exception
        require_doctor_report_access(user)
    
    def test_administration_admin_granted(self):
        """ADMINISTRATION_ADMIN can access doctor reports."""
        user = create_user_with_role(ADMINISTRATION_ADMIN)
        
        # Should not raise exception
        require_doctor_report_access(user)
    
    def test_department_admin_granted(self):
        """DEPARTMENT_ADMIN can access doctor reports."""
        user = create_user_with_role(DEPARTMENT_ADMIN)
        
        # Should not raise exception
        require_doctor_report_access(user)
    
    def test_section_admin_granted(self):
        """SECTION_ADMIN can access doctor reports."""
        user = create_user_with_role(SECTION_ADMIN)
        
        # Should not raise exception
        require_doctor_report_access(user)
    
    def test_complaint_supervisor_granted(self):
        """COMPLAINT_SUPERVISOR can access doctor reports."""
        user = create_user_with_role(COMPLAINT_SUPERVISOR)
        
        # Should not raise exception
        require_doctor_report_access(user)
    
    def test_worker_denied(self):
        """WORKER cannot access doctor reports (403)."""
        user = create_user_with_role(WORKER)
        
        with pytest.raises(HTTPException) as exc_info:
            require_doctor_report_access(user)
        
        assert exc_info.value.status_code == 403
        assert "FORBIDDEN" in exc_info.value.detail["error"]
    
    def test_multiple_roles_at_least_one_allowed(self):
        """User with multiple roles passes if at least one is allowed."""
        user = CurrentUser(
            user_id=2,
            username="mixed_user",
            is_active=True,
            scopes=[
                UserScope(
                    role_code=WORKER,
                    role_name="Worker",
                    orgunit_id=1,
                    orgunit_name="Section A",
                    orgunit_type="SECTION"
                ),
                UserScope(
                    role_code=SECTION_ADMIN,
                    role_name="Section Admin",
                    orgunit_id=2,
                    orgunit_name="Section B",
                    orgunit_type="SECTION"
                )
            ]
        )
        
        # Should not raise (has SECTION_ADMIN)
        require_doctor_report_access(user)
    
    def test_no_roles_denied(self):
        """User with no roles cannot access doctor reports."""
        user = CurrentUser(
            user_id=3,
            username="no_roles",
            is_active=True,
            scopes=[]
        )
        
        with pytest.raises(HTTPException) as exc_info:
            require_doctor_report_access(user)
        
        assert exc_info.value.status_code == 403


# ==================== WORKER REPORT GUARD TESTS ====================

class TestWorkerReportAccessGuard:
    """Tests for require_worker_report_access() guard function."""
    
    def test_software_admin_granted(self):
        """SOFTWARE_ADMIN can access worker reports."""
        user = create_user_with_role(SOFTWARE_ADMIN)
        
        # Should not raise exception
        require_worker_report_access(user)
    
    def test_administration_admin_granted(self):
        """ADMINISTRATION_ADMIN can access worker reports."""
        user = create_user_with_role(ADMINISTRATION_ADMIN)
        
        # Should not raise exception
        require_worker_report_access(user)
    
    def test_department_admin_granted(self):
        """DEPARTMENT_ADMIN can access worker reports."""
        user = create_user_with_role(DEPARTMENT_ADMIN)
        
        # Should not raise exception
        require_worker_report_access(user)
    
    def test_section_admin_granted(self):
        """SECTION_ADMIN can access worker reports."""
        user = create_user_with_role(SECTION_ADMIN)
        
        # Should not raise exception
        require_worker_report_access(user)
    
    def test_complaint_supervisor_granted(self):
        """COMPLAINT_SUPERVISOR can access worker reports."""
        user = create_user_with_role(COMPLAINT_SUPERVISOR)
        
        # Should not raise exception
        require_worker_report_access(user)
    
    def test_worker_denied(self):
        """WORKER cannot access worker reports (403)."""
        user = create_user_with_role(WORKER)
        
        with pytest.raises(HTTPException) as exc_info:
            require_worker_report_access(user)
        
        assert exc_info.value.status_code == 403
        assert "FORBIDDEN" in exc_info.value.detail["error"]
    
    def test_multiple_roles_at_least_one_allowed(self):
        """User with multiple roles passes if at least one is allowed."""
        user = CurrentUser(
            user_id=4,
            username="multi_role",
            is_active=True,
            scopes=[
                UserScope(
                    role_code=WORKER,
                    role_name="Worker",
                    orgunit_id=1,
                    orgunit_name="Section A",
                    orgunit_type="SECTION"
                ),
                UserScope(
                    role_code=COMPLAINT_SUPERVISOR,
                    role_name="Supervisor",
                    orgunit_id=1,
                    orgunit_name="Section A",
                    orgunit_type="SECTION"
                )
            ]
        )
        
        # Should not raise (has COMPLAINT_SUPERVISOR)
        require_worker_report_access(user)
    
    def test_no_roles_denied(self):
        """User with no roles cannot access worker reports."""
        user = CurrentUser(
            user_id=5,
            username="empty_scopes",
            is_active=True,
            scopes=[]
        )
        
        with pytest.raises(HTTPException) as exc_info:
            require_worker_report_access(user)
        
        assert exc_info.value.status_code == 403


# ==================== GUARD CONSISTENCY TESTS ====================

class TestGuardConsistency:
    """Tests verifying both guards have consistent behavior."""
    
    def test_same_allowed_roles(self):
        """Doctor and worker guards should have identical allowed roles."""
        allowed_roles = [
            SOFTWARE_ADMIN,
            ADMINISTRATION_ADMIN,
            DEPARTMENT_ADMIN,
            SECTION_ADMIN,
            COMPLAINT_SUPERVISOR
        ]
        
        for role in allowed_roles:
            user = create_user_with_role(role)
            
            # Both should allow access (no exception)
            require_doctor_report_access(user)
            require_worker_report_access(user)
    
    def test_same_denied_roles(self):
        """Doctor and worker guards should deny same roles."""
        denied_roles = [WORKER]
        
        for role in denied_roles:
            user = create_user_with_role(role)
            
            # Both should deny access (403)
            with pytest.raises(HTTPException) as exc1:
                require_doctor_report_access(user)
            
            with pytest.raises(HTTPException) as exc2:
                require_worker_report_access(user)
            
            assert exc1.value.status_code == 403
            assert exc2.value.status_code == 403


# ==================== ROUTER INTEGRATION TESTS ====================

class TestRouterGuardIntegration:
    """Tests verifying guards are properly integrated in router."""
    
    def test_router_imports_guards(self):
        """seasonal_export_router.py imports guard functions."""
        import backend.api.routers.seasonal_export_router as router_module
        
        # Check imports exist
        assert hasattr(router_module, 'require_doctor_report_access')
        assert hasattr(router_module, 'require_worker_report_access')
    
    def test_router_imports_current_user_schema(self):
        """Router correctly imports CurrentUser schema."""
        import backend.api.routers.seasonal_export_router as router_module
        
        assert hasattr(router_module, 'CurrentUser')
    
    def test_guards_imported_from_correct_module(self):
        """Guards are imported from api.utils.guards."""
        from backend.api.utils.guards import (
            require_doctor_report_access as guard1,
            require_worker_report_access as guard2
        )
        
        # These should be callable functions
        assert callable(guard1)
        assert callable(guard2)


# ==================== ERROR MESSAGE TESTS ====================

class TestGuardErrorMessages:
    """Tests for guard error message structure."""
    
    def test_doctor_guard_error_includes_required_roles(self):
        """Doctor guard error detail includes required roles list."""
        user = create_user_with_role(WORKER)
        
        with pytest.raises(HTTPException) as exc_info:
            require_doctor_report_access(user)
        
        detail = exc_info.value.detail
        assert "required_roles" in detail
        assert SOFTWARE_ADMIN in detail["required_roles"]
        assert SECTION_ADMIN in detail["required_roles"]
    
    def test_worker_guard_error_includes_required_roles(self):
        """Worker guard error detail includes required roles list."""
        user = create_user_with_role(WORKER)
        
        with pytest.raises(HTTPException) as exc_info:
            require_worker_report_access(user)
        
        detail = exc_info.value.detail
        assert "required_roles" in detail
        assert SOFTWARE_ADMIN in detail["required_roles"]
        assert COMPLAINT_SUPERVISOR in detail["required_roles"]
    
    def test_error_includes_user_roles(self):
        """Error detail includes user's actual roles for debugging."""
        user = create_user_with_role(WORKER)
        
        with pytest.raises(HTTPException) as exc_info:
            require_doctor_report_access(user)
        
        detail = exc_info.value.detail
        assert "user_roles" in detail
        assert WORKER in detail["user_roles"]


# ==================== RUN TESTS ====================

if __name__ == "__main__":
    print("=" * 70)
    print("Phase D - Task B10: Permission Guards Tests")
    print("=" * 70)
    print()
    
    # Run with verbose output
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--color=yes"
    ])
    
    print()
    print("=" * 70)
    if exit_code == 0:
        print("✅ ALL TESTS PASSED - D-B10 Permission Guards Complete")
    else:
        print("❌ SOME TESTS FAILED - Review output above")
    print("=" * 70)
    
    exit(exit_code)
