"""
PHASE K — Migration Progress Service

Service layer for migration progress reporting.

This service calculates and returns migration statistics:
- Total legacy case count
- Migrated case count
- Remaining case count
- Migration percent complete

Used by Migration Page dashboard.

NO WRITES - Read-only reporting only.
"""

from api.db_layer.migration_progress_db import get_migration_progress_counts


def get_migration_progress() -> dict:
    """
    Get migration progress statistics.
    
    Returns comprehensive progress report including:
    - Total cases in legacy system
    - Cases already migrated
    - Cases remaining to migrate
    - Percent complete (0.00 to 100.00)
    
    Returns:
        dict: {
            "success": True,
            "total_cases": int,
            "migrated_cases": int,
            "remaining_cases": int,
            "percent_complete": float
        }
        
    Example:
        {
            "success": True,
            "total_cases": 1000,
            "migrated_cases": 350,
            "remaining_cases": 650,
            "percent_complete": 35.00
        }
    """
    
    # Get raw counts from database
    counts = get_migration_progress_counts()
    
    total = counts["total_cases"]
    migrated = counts["migrated_cases"]
    
    # Calculate remaining
    remaining = total - migrated
    
    # Ensure remaining never goes negative
    if remaining < 0:
        remaining = 0
    
    # Calculate percent complete with zero-division safety
    # Round to 1 decimal place as per API contract
    if total == 0:
        percent = 0.0
    else:
        percent = round((migrated * 100.0) / total, 1)
    
    return {
        "success": True,
        "total_cases": total,
        "migrated_cases": migrated,
        "remaining_cases": remaining,
        "percent_complete": percent
    }
