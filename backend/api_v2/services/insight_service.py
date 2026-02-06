"""
Insight Service Layer (Phase 4B - B-I7)

Service layer for analytics/insight endpoints.
Orchestrates DB layer calls with scope enforcement via allowed_unit_ids.

Architecture:
  Router → Service → DB Layer

Functions read current_user.allowed_unit_ids and pass to DB layer.
No SQL queries here - all SQL in insight_db.py.
"""

from typing import Dict, Any, List
import pyodbc
from ..db_layer import insight_db
from backend.api.schemas.auth_models import CurrentUser


def get_db_connection():
    """Get database connection using project standard."""
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=SOCIALMEDIA;"
        "DATABASE=IncidentManager;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )
    return conn


def get_kpi_summary(current_user: CurrentUser) -> Dict[str, Any]:
    """
    Get KPI summary for current user's scope.
    
    Returns aggregate metrics:
    - Subcase status distribution
    - Action item counts (total, open, completed, overdue)
    - Stuck case count
    
    Args:
        current_user: Authenticated user with allowed_unit_ids
        
    Returns:
        Dictionary with KPI metrics
    """
    conn = get_db_connection()
    try:
        # Convert set to list for DB layer
        allowed_unit_ids = list(current_user.allowed_unit_ids)
        
        # Get subcase status counts
        status_counts = insight_db.get_subcase_status_counts(conn, allowed_unit_ids)
        
        # Get action item counts
        action_items = insight_db.get_action_item_counts(conn, allowed_unit_ids)
        
        # Compute total subcases from status counts
        total_subcases = sum(item['count'] for item in status_counts)
        
        return {
            "total_subcases": total_subcases,
            "by_status": status_counts,
            "action_items": action_items
        }
    finally:
        conn.close()


def get_distribution(current_user: CurrentUser, dimension: str) -> List[Dict[str, Any]]:
    """
    Get subcase distribution by specified dimension.
    
    Currently supports:
    - dimension="status": Distribution by subcase status
    - dimension="org_unit": Distribution by target org unit
    
    Args:
        current_user: Authenticated user with allowed_unit_ids
        dimension: Dimension to group by ("status" or "org_unit")
        
    Returns:
        List of distribution items with counts
        Shape: [{"key": value, "count": int}]
        
    Raises:
        ValueError: If dimension is not supported
    """
    # Validate dimension
    valid_dimensions = ["status", "org_unit"]
    if dimension not in valid_dimensions:
        raise ValueError(f"Invalid dimension: '{dimension}'. Must be one of: {valid_dimensions}")
    
    conn = get_db_connection()
    try:
        # Convert set to list for DB layer
        allowed_unit_ids = list(current_user.allowed_unit_ids)
        
        if dimension == "status":
            # Get status distribution
            raw_results = insight_db.get_subcase_status_counts(conn, allowed_unit_ids)
            # Transform to standard shape: {"key": status_value, "count": count}
            return [
                {"key": item["status"], "count": item["count"]}
                for item in raw_results
            ]
        
        elif dimension == "org_unit":
            # Get org unit distribution
            raw_results = insight_db.get_subcase_org_unit_counts(conn, allowed_unit_ids)
            # Transform to standard shape: {"key": org_unit_id, "count": count}
            return [
                {"key": item["target_org_unit_id"], "count": item["count"]}
                for item in raw_results
            ]
    finally:
        conn.close()


def get_trend(current_user: CurrentUser, bucket: str) -> List[Dict[str, Any]]:
    """
    Get time-series trend of subcase creation.
    
    Buckets:
    - "day": Daily trend (YYYY-MM-DD)
    - "month": Monthly trend (YYYY-MM)
    - "year": Yearly trend (YYYY)
    
    Args:
        current_user: Authenticated user with allowed_unit_ids
        bucket: Time bucket granularity ("day", "month", or "year")
        
    Returns:
        List of time buckets with counts, sorted ascending
        Shape: [{"bucket": str, "count": int}]
        
    Raises:
        ValueError: If bucket is not supported
    """
    # Validate bucket
    valid_buckets = ["day", "month", "year"]
    if bucket not in valid_buckets:
        raise ValueError(f"Invalid bucket: '{bucket}'. Must be one of: {valid_buckets}")
    
    conn = get_db_connection()
    try:
        # Convert set to list for DB layer
        allowed_unit_ids = list(current_user.allowed_unit_ids)
        
        # Get time buckets from DB
        raw_results = insight_db.get_subcase_created_time_buckets(conn, allowed_unit_ids, bucket)
        
        # Transform: rename bucket_label → bucket
        return [
            {"bucket": item["bucket_label"], "count": item["count"]}
            for item in raw_results
        ]
    finally:
        conn.close()


def get_stuck_cases(current_user: CurrentUser, days_threshold: int = 30) -> List[Dict[str, Any]]:
    """
    Get list of subcases stuck in non-terminal status.
    
    Stuck = UpdatedAt older than threshold AND not in terminal statuses.
    
    A subcase is "stuck" if it has been in current status for >= days_threshold days
    and is not in a terminal status (ADMIN_APPROVED, SECTION_DENIED, FORCE_CLOSED).
    
    Args:
        current_user: Authenticated user with allowed_unit_ids
        days_threshold: Minimum days in status to be considered "stuck" (default: 30)
        
    Returns:
        List of stuck subcases with subcase_id, status, target_org_unit_id,
        updated_at, days_in_stage
    """
    conn = get_db_connection()
    try:
        # Convert set to list for DB layer
        allowed_unit_ids = list(current_user.allowed_unit_ids)
        
        # Get stuck subcases from DB (returns rows unchanged)
        return insight_db.get_stuck_subcases(conn, allowed_unit_ids, days_threshold)
    finally:
        conn.close()
