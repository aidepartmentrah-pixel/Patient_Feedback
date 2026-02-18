"""
PHASE K — Legacy Case Service

Service layer for legacy case listing and detail views.

This service provides read-only access to legacy case data
for the Migration Page dashboard.

NO WRITES - Read-only legacy data access only.
"""

from api.db_layer.legacy_case_db import list_legacy_cases_paged as db_list_legacy_cases_paged
from api.db_layer.legacy_case_db import get_legacy_case_by_id


def list_legacy_cases_paged(page: int = 1, page_size: int = 50) -> dict:
    """
    Get paginated list of legacy cases.
    
    Args:
        page: Page number (1-indexed)
        page_size: Number of records per page
        
    Returns:
        dict: {
            "cases": [...],
            "total": int
        }
    """
    return db_list_legacy_cases_paged(page, page_size)


def get_legacy_case_detail(legacy_case_id: int) -> dict:
    """
    Get detailed legacy case record.
    
    Args:
        legacy_case_id: Legacy case ID
        
    Returns:
        dict: Full case record
        None: If case not found
    """
    return get_legacy_case_by_id(legacy_case_id)
