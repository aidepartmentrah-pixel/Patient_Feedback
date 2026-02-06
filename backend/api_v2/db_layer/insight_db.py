"""
Insight Database Layer (API V2)
Handles read-only aggregation queries for Insight analytics endpoints.

This is part of Phase 4B Insight implementation.
NO business logic. NO authorization. ONLY SELECT aggregation queries.
NO workflow mutations. NO table modifications.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import pyodbc


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


# ============================================================
# SUBCASE STATUS AGGREGATION
# ============================================================

def get_subcase_status_counts(conn, allowed_unit_ids: List[int]) -> List[Dict[str, Any]]:
    """
    Aggregate subcase counts by Status field.
    
    Table: APP_AdministrativeSubcase
    Grouping: Status
    Filter: TargetOrgUnitID IN allowed_unit_ids
    
    Args:
        conn: Database connection
        allowed_unit_ids: List of org unit IDs for scope filtering
    
    Returns:
        List of status count dicts:
        [
            {"status": "SUBMITTED_TO_SECTION", "count": 10},
            {"status": "SECTION_ACCEPTED_PENDING_DEPT", "count": 5},
            {"status": "DEPT_ACCEPTED_PENDING_ADMIN", "count": 3},
            {"status": "ADMIN_APPROVED", "count": 100},
            ...
        ]
    """
    cursor = conn.cursor()
    
    try:
        # Handle empty allowed_unit_ids
        if not allowed_unit_ids:
            return []
        
        # Build parameterized IN clause
        placeholders = ','.join('?' * len(allowed_unit_ids))
        
        query = f"""
            SELECT 
                Status,
                COUNT(*) AS Count
            FROM dbo.APP_AdministrativeSubcase
            WHERE TargetOrgUnitID IN ({placeholders})
            GROUP BY Status
            ORDER BY Status
        """
        
        cursor.execute(query, allowed_unit_ids)
        rows = cursor.fetchall()
        
        return [
            {
                "status": row.Status,
                "count": row.Count
            }
            for row in rows
        ]
    
    finally:
        cursor.close()


# ============================================================
# ACTION ITEM AGGREGATION
# ============================================================

def get_action_item_counts(conn, allowed_unit_ids: List[int]) -> Dict[str, int]:
    """
    Aggregate action item metrics joined through subcase scope.
    
    Tables: 
    - APP_SubcaseActionItem (alias a)
    - APP_AdministrativeSubcase (alias s)
    
    Join: a.SubcaseID = s.SubcaseID
    Filter: s.TargetOrgUnitID IN allowed_unit_ids
    
    Args:
        conn: Database connection
        allowed_unit_ids: List of org unit IDs for scope filtering
    
    Returns:
        Dict with action item metrics:
        {
            "total": 100,      # All action items
            "open": 45,        # CompletedAt IS NULL
            "completed": 55,   # CompletedAt IS NOT NULL
            "overdue": 12      # DueDate < TODAY AND CompletedAt IS NULL
        }
    """
    cursor = conn.cursor()
    
    try:
        # Handle empty allowed_unit_ids
        if not allowed_unit_ids:
            return {
                "total": 0,
                "open": 0,
                "completed": 0,
                "overdue": 0
            }
        
        # Build parameterized IN clause
        placeholders = ','.join('?' * len(allowed_unit_ids))
        
        query = f"""
            SELECT 
                COUNT(*) AS Total,
                SUM(CASE WHEN a.CompletedAt IS NULL THEN 1 ELSE 0 END) AS OpenCount,
                SUM(CASE WHEN a.CompletedAt IS NOT NULL THEN 1 ELSE 0 END) AS CompletedCount,
                SUM(CASE 
                    WHEN a.DueDate < CAST(GETDATE() AS DATE) 
                        AND a.CompletedAt IS NULL 
                    THEN 1 
                    ELSE 0 
                END) AS OverdueCount
            FROM dbo.APP_SubcaseActionItem a
            INNER JOIN dbo.APP_AdministrativeSubcase s 
                ON a.SubcaseID = s.SubcaseID
            WHERE s.TargetOrgUnitID IN ({placeholders})
        """
        
        cursor.execute(query, allowed_unit_ids)
        row = cursor.fetchone()
        
        if not row or row.Total is None:
            return {
                "total": 0,
                "open": 0,
                "completed": 0,
                "overdue": 0
            }
        
        return {
            "total": row.Total or 0,
            "open": row.OpenCount or 0,
            "completed": row.CompletedCount or 0,
            "overdue": row.OverdueCount or 0
        }
    
    finally:
        cursor.close()


# ============================================================
# STUCK SUBCASE DETECTION
# ============================================================

def get_stuck_subcases(conn, allowed_unit_ids: List[int], days_threshold: int) -> List[Dict[str, Any]]:
    """
    Find subcases that have not been updated beyond a time threshold.
    
    Table: APP_AdministrativeSubcase
    Logic: UpdatedAt < (NOW - days_threshold) AND Status NOT IN terminal_statuses
    Filter: TargetOrgUnitID IN allowed_unit_ids
    Terminal statuses: ADMIN_APPROVED, SECTION_DENIED, FORCE_CLOSED
    
    Args:
        conn: Database connection
        allowed_unit_ids: List of org unit IDs for scope filtering
        days_threshold: Number of days to consider a subcase "stuck"
    
    Returns:
        List of subcase dicts with fields:
        [
            {
                "subcase_id": 123,
                "status": "SUBMITTED_TO_SECTION",
                "target_org_unit_id": 5,
                "updated_at": datetime,
                "days_in_stage": 14
            },
            ...
        ]
    """
    cursor = conn.cursor()
    
    try:
        # Handle empty allowed_unit_ids
        if not allowed_unit_ids:
            return []
        
        # Build parameterized IN clause
        placeholders = ','.join('?' * len(allowed_unit_ids))
        
        query = f"""
            SELECT 
                SubcaseID,
                Status,
                TargetOrgUnitID,
                UpdatedAt,
                DATEDIFF(day, UpdatedAt, GETDATE()) AS DaysInStage
            FROM dbo.APP_AdministrativeSubcase
            WHERE TargetOrgUnitID IN ({placeholders})
              AND Status NOT IN ('ADMIN_APPROVED', 'SECTION_DENIED', 'FORCE_CLOSED')
              AND DATEDIFF(day, UpdatedAt, GETDATE()) >= ?
            ORDER BY DaysInStage DESC
        """
        
        # Parameters: allowed_unit_ids + days_threshold
        params = allowed_unit_ids + [days_threshold]
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        return [
            {
                "subcase_id": row.SubcaseID,
                "status": row.Status,
                "target_org_unit_id": row.TargetOrgUnitID,
                "updated_at": row.UpdatedAt,
                "days_in_stage": row.DaysInStage
            }
            for row in rows
        ]
    
    finally:
        cursor.close()


# ============================================================
# TIME-BUCKET TREND AGGREGATION
# ============================================================

def get_subcase_created_time_buckets(conn, allowed_unit_ids: List[int], bucket: str) -> List[Dict[str, Any]]:
    """
    Aggregate subcase creation counts over time buckets.
    
    Table: APP_AdministrativeSubcase
    Grouping: CreatedAt truncated to bucket (day, month, year)
    Filter: TargetOrgUnitID IN allowed_unit_ids
    
    Args:
        conn: Database connection
        allowed_unit_ids: List of org unit IDs for scope filtering
        bucket: Time bucket granularity ("day", "month", "year")
    
    Returns:
        List of time bucket dicts:
        [
            {
                "bucket_label": "2026-02",  # Format varies by bucket type
                "count": 25
            },
            ...
        ]
        Sorted ascending by time.
    
    Raises:
        ValueError: If bucket is not one of: "day", "month", "year"
    """
    cursor = conn.cursor()
    
    try:
        # Validate bucket parameter
        valid_buckets = ["day", "month", "year"]
        if bucket not in valid_buckets:
            raise ValueError(f"Invalid bucket: '{bucket}'. Must be one of: {valid_buckets}")
        
        # Handle empty allowed_unit_ids
        if not allowed_unit_ids:
            return []
        
        # Build parameterized IN clause
        placeholders = ','.join('?' * len(allowed_unit_ids))
        
        # Build query based on bucket type
        if bucket == "day":
            # Group by day: YYYY-MM-DD
            query = f"""
                SELECT 
                    CONVERT(varchar(10), CAST(CreatedAt AS DATE), 23) AS BucketLabel,
                    COUNT(*) AS Count
                FROM dbo.APP_AdministrativeSubcase
                WHERE TargetOrgUnitID IN ({placeholders})
                GROUP BY CAST(CreatedAt AS DATE)
                ORDER BY CAST(CreatedAt AS DATE)
            """
        elif bucket == "month":
            # Group by month: YYYY-MM
            query = f"""
                SELECT 
                    CONCAT(YEAR(CreatedAt), '-', RIGHT('0'+CAST(MONTH(CreatedAt) AS varchar),2)) AS BucketLabel,
                    COUNT(*) AS Count
                FROM dbo.APP_AdministrativeSubcase
                WHERE TargetOrgUnitID IN ({placeholders})
                GROUP BY YEAR(CreatedAt), MONTH(CreatedAt)
                ORDER BY YEAR(CreatedAt), MONTH(CreatedAt)
            """
        else:  # bucket == "year"
            # Group by year: YYYY
            query = f"""
                SELECT 
                    CAST(YEAR(CreatedAt) AS varchar) AS BucketLabel,
                    COUNT(*) AS Count
                FROM dbo.APP_AdministrativeSubcase
                WHERE TargetOrgUnitID IN ({placeholders})
                GROUP BY YEAR(CreatedAt)
                ORDER BY YEAR(CreatedAt)
            """
        
        cursor.execute(query, allowed_unit_ids)
        rows = cursor.fetchall()
        
        return [
            {
                "bucket_label": row.BucketLabel,
                "count": row.Count
            }
            for row in rows
        ]
    
    finally:
        cursor.close()


def get_subcase_org_unit_counts(conn, allowed_unit_ids: List[int]) -> List[Dict[str, Any]]:
    """
    Get subcase counts grouped by target organizational unit.
    
    Returns distribution of subcases across organizational units.
    No org unit name lookups - just IDs and counts.
    
    Args:
        conn: Database connection
        allowed_unit_ids: List of org unit IDs to filter by
        
    Returns:
        List of dicts with target_org_unit_id (int) and count (int)
        No guaranteed sorting order
    """
    if not allowed_unit_ids:
        return []
    
    placeholders = ','.join('?' * len(allowed_unit_ids))
    
    query = f"""
        SELECT 
            TargetOrgUnitID,
            COUNT(*) AS count
        FROM dbo.APP_AdministrativeSubcase
        WHERE TargetOrgUnitID IN ({placeholders})
        GROUP BY TargetOrgUnitID
    """
    
    cursor = conn.cursor()
    cursor.execute(query, allowed_unit_ids)
    
    results = []
    for row in cursor.fetchall():
        results.append({
            'target_org_unit_id': row[0],
            'count': row[1]
        })
    
    cursor.close()
    return results
