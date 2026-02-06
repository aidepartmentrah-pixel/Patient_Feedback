"""
User Inventory Service Layer
Business logic for organizational unit and user mapping queries.

Phase 5 - Module 5.1: Inventory & Mapping Engine
Provides read-only access to user-org unit mappings.
"""

from typing import List, Dict, Any
from ..db_layer.user_inventory_db import (
    get_org_unit_user_inventory,
    get_org_units_without_users,
    get_inventory_summary
)


def get_org_unit_user_inventory_service() -> List[Dict[str, Any]]:
    """
    Get comprehensive inventory of all organizational units and their assigned users.
    
    Service layer wrapper around database query.
    No additional business logic - pure pass-through.
    
    Returns:
        List of dictionaries with org unit and user information.
        See db_layer.user_inventory_db.get_org_unit_user_inventory() for format.
    """
    return get_org_unit_user_inventory()


def get_org_units_without_users_service() -> List[Dict[str, Any]]:
    """
    Get organizational units that have NO users assigned.
    
    Useful for identifying which units need user accounts created.
    
    Returns:
        List of org units without any user assignments.
        See db_layer.user_inventory_db.get_org_units_without_users() for format.
    """
    return get_org_units_without_users()


def get_inventory_summary_service() -> Dict[str, Any]:
    """
    Get summary statistics about user inventory.
    
    Returns:
        Dictionary with summary counts (total units, total users, etc).
        See db_layer.user_inventory_db.get_inventory_summary() for format.
    """
    return get_inventory_summary()
