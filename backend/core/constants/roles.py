"""
Role Constants
Defines all system roles for RBAC (Role-Based Access Control).

These constants correspond to the RoleCode column in the APP_Roles table.
They are used throughout the application for role checking and authorization.

Phase 2 RBAC: Role definitions matching database schema.
"""

# ==================== SYSTEM ROLES ====================

# Software Administrator - Full system access
# Can manage all system settings, users, roles, and data
SOFTWARE_ADMIN = "SOFTWARE_ADMIN"

# Worker - Basic complaint handling role
# Can view and update complaints within their assigned organizational unit
WORKER = "WORKER"

# Complaint Supervisor - Oversees complaint handling
# Can supervise complaint processing and access reports
COMPLAINT_SUPERVISOR = "COMPLAINT_SUPERVISOR"

# Section Administrator - Manages a section
# Has administrative access to section-level data and users
SECTION_ADMIN = "SECTION_ADMIN"

# Department Administrator - Manages a department
# Has administrative access to department-level data and users
DEPARTMENT_ADMIN = "DEPARTMENT_ADMIN"

# Administration Administrator - High-level admin
# Has broad administrative access across the organization
ADMINISTRATION_ADMIN = "ADMINISTRATION_ADMIN"


# ==================== ROLE SETS ====================

# All valid roles in the system
ALL_ROLES = [
    SOFTWARE_ADMIN,
    WORKER,
    COMPLAINT_SUPERVISOR,
    SECTION_ADMIN,
    DEPARTMENT_ADMIN,
    ADMINISTRATION_ADMIN,
]

# Administrative roles (can manage users and settings)
ADMIN_ROLES = [
    SOFTWARE_ADMIN,
    SECTION_ADMIN,
    DEPARTMENT_ADMIN,
    ADMINISTRATION_ADMIN,
]

# Supervisor roles (can oversee and approve)
SUPERVISOR_ROLES = [
    SOFTWARE_ADMIN,
    COMPLAINT_SUPERVISOR,
    SECTION_ADMIN,
    DEPARTMENT_ADMIN,
    ADMINISTRATION_ADMIN,
]

# Worker-level roles (can handle complaints)
WORKER_ROLES = [
    WORKER,
]


# ==================== ROLE HIERARCHY ====================

# Role hierarchy for permission inheritance
# Higher roles in the list have more privileges
ROLE_HIERARCHY = [
    SOFTWARE_ADMIN,         # Highest privilege
    ADMINISTRATION_ADMIN,
    DEPARTMENT_ADMIN,
    SECTION_ADMIN,
    COMPLAINT_SUPERVISOR,
    WORKER,                 # Lowest privilege
]


# ==================== HELPER FUNCTIONS ====================

def is_valid_role(role_code: str) -> bool:
    """
    Check if a role code is valid.
    
    Args:
        role_code: Role code to validate
    
    Returns:
        True if valid, False otherwise
    
    Example:
        >>> is_valid_role("SOFTWARE_ADMIN")
        True
        >>> is_valid_role("INVALID_ROLE")
        False
    """
    return role_code in ALL_ROLES


def is_admin_role(role_code: str) -> bool:
    """
    Check if a role is an administrative role.
    
    Args:
        role_code: Role code to check
    
    Returns:
        True if admin role, False otherwise
    
    Example:
        >>> is_admin_role("SOFTWARE_ADMIN")
        True
        >>> is_admin_role("WORKER")
        False
    """
    return role_code in ADMIN_ROLES


def is_supervisor_role(role_code: str) -> bool:
    """
    Check if a role is a supervisory role.
    
    Args:
        role_code: Role code to check
    
    Returns:
        True if supervisor role, False otherwise
    
    Example:
        >>> is_supervisor_role("COMPLAINT_SUPERVISOR")
        True
        >>> is_supervisor_role("WORKER")
        False
    """
    return role_code in SUPERVISOR_ROLES


def get_role_display_name(role_code: str) -> str:
    """
    Get display name for a role.
    
    Args:
        role_code: Role code
    
    Returns:
        Human-readable role name
    
    Example:
        >>> get_role_display_name("SOFTWARE_ADMIN")
        "Software Administrator"
    """
    role_names = {
        SOFTWARE_ADMIN: "Software Administrator",
        WORKER: "Worker",
        COMPLAINT_SUPERVISOR: "Complaint Supervisor",
        SECTION_ADMIN: "Section Administrator",
        DEPARTMENT_ADMIN: "Department Administrator",
        ADMINISTRATION_ADMIN: "Administration Administrator",
    }
    return role_names.get(role_code, role_code)
