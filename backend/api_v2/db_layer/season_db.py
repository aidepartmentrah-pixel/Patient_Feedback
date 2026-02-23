"""
Season Database Layer (API V2)
Handles SQL operations for dbo.Season table.

NO business logic. NO authorization. ONLY SQL operations.
"""

from typing import Dict, Any, Optional
from datetime import date, datetime
import pyodbc
from core.database import get_connection


# ============================================================
# QUARTER DATE RANGES (Fixed Pattern)
# ============================================================
QUARTER_DATE_RANGES = {
    1: {"start_month": 1, "start_day": 1, "end_month": 3, "end_day": 31},    # Q1: Jan 1 - Mar 31
    2: {"start_month": 4, "start_day": 1, "end_month": 6, "end_day": 30},    # Q2: Apr 1 - Jun 30
    3: {"start_month": 7, "start_day": 1, "end_month": 9, "end_day": 30},    # Q3: Jul 1 - Sep 30
    4: {"start_month": 10, "start_day": 1, "end_month": 12, "end_day": 31},  # Q4: Oct 1 - Dec 31
}


# ============================================================
# AUTO-GENERATION FUNCTIONS
# ============================================================

def ensure_seasons_exist(conn: pyodbc.Connection, years_ahead: int = 2, reference_date: date = None) -> int:
    """
    Auto-create missing seasons for current year and future years.
    
    This ensures the software works autonomously without manual intervention.
    Call this before fetching seasons for dropdowns.
    
    Args:
        conn: Database connection
        years_ahead: How many years into the future to create (default: 2)
        reference_date: Date to use as "today" (for testing). If None, uses current date.
        
    Returns:
        Number of seasons created
    """
    if reference_date is None:
        reference_date = date.today()
    
    current_year = reference_date.year
    created_count = 0
    
    cursor = conn.cursor()
    
    # Get the current max UniqueID
    cursor.execute("SELECT ISNULL(MAX(UniqueID), 0) FROM dbo.Season")
    next_id = cursor.fetchone()[0] + 1
    
    # Create seasons for current year through years_ahead
    for year in range(current_year, current_year + years_ahead + 1):
        for quarter in range(1, 5):
            season_name = f"Q{quarter}-{year}"
            
            # Check if this season already exists
            cursor.execute(
                "SELECT UniqueID FROM dbo.Season WHERE SeasonName = ?",
                (season_name,)
            )
            
            if cursor.fetchone() is not None:
                # Season already exists, skip
                continue
            
            # Calculate dates
            q = QUARTER_DATE_RANGES[quarter]
            start_date = date(year, q["start_month"], q["start_day"])
            end_date = date(year, q["end_month"], q["end_day"])
            
            # Insert new season with explicit UniqueID
            cursor.execute(
                """
                INSERT INTO dbo.Season (UniqueID, SeasonName, StartDate, EndDate, IsDone, Frozen)
                VALUES (?, ?, ?, ?, 0, 0)
                """,
                (next_id, season_name, start_date, end_date)
            )
            created_count += 1
            next_id += 1
            print(f"[Season Auto-Gen] Created: {season_name} ({start_date} to {end_date})")
    
    if created_count > 0:
        conn.commit()
        print(f"[Season Auto-Gen] Total created: {created_count} seasons")
    
    cursor.close()
    return created_count


def get_current_season_by_date(conn: pyodbc.Connection, reference_date: date = None) -> Optional[Dict[str, Any]]:
    """
    Get the current season based on date ranges, not IsDone flag.
    
    This is the correct way to determine "current" season.
    
    Args:
        conn: Database connection
        reference_date: Date to check (for testing). If None, uses current date.
        
    Returns:
        Season dict if found, None otherwise
    """
    if reference_date is None:
        reference_date = date.today()
    
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT 
            UniqueID,
            SeasonName,
            StartDate,
            EndDate,
            IsDone,
            Frozen
        FROM dbo.Season
        WHERE StartDate <= ? AND EndDate >= ?
        """,
        (reference_date, reference_date)
    )
    
    row = cursor.fetchone()
    cursor.close()
    
    if not row:
        return None
    
    return {
        "season_id": row.UniqueID,
        "season_name": row.SeasonName,
        "start_date": row.StartDate,
        "end_date": row.EndDate,
        "is_done": bool(row.IsDone) if row.IsDone is not None else False,
        "frozen": bool(row.Frozen) if row.Frozen is not None else False
    }


# ============================================================
# SEASON QUERIES
# ============================================================

def get_season_by_id(conn: pyodbc.Connection, season_id: int) -> Optional[Dict[str, Any]]:
    """
    Get a season record by its UniqueID.
    
    Args:
        conn: Database connection
        season_id: The season UniqueID (primary key)
        
    Returns:
        Dict with season data or None if not found
        {
            "season_id": int,
            "season_name": str | None,
            "start_date": date,
            "end_date": date,
            "is_done": bool,
            "frozen": bool
        }
    """
    cursor = conn.cursor()
    
    query = """
        SELECT 
            UniqueID,
            SeasonName,
            StartDate,
            EndDate,
            IsDone,
            Frozen
        FROM dbo.Season
        WHERE UniqueID = ?
    """
    
    cursor.execute(query, (season_id,))
    row = cursor.fetchone()
    cursor.close()
    
    if not row:
        return None
    
    return {
        "season_id": row.UniqueID,
        "season_name": row.SeasonName,
        "start_date": row.StartDate,
        "end_date": row.EndDate,
        "is_done": bool(row.IsDone) if row.IsDone is not None else False,
        "frozen": bool(row.Frozen) if row.Frozen is not None else False
    }


def get_all_seasons(conn: pyodbc.Connection) -> list[Dict[str, Any]]:
    """
    Get all season records, ordered by start date descending (newest first).
    
    Args:
        conn: Database connection
        
    Returns:
        List of season dicts
    """
    cursor = conn.cursor()
    
    query = """
        SELECT 
            UniqueID,
            SeasonName,
            StartDate,
            EndDate,
            IsDone,
            Frozen
        FROM dbo.Season
        ORDER BY StartDate DESC
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    
    seasons = []
    for row in rows:
        seasons.append({
            "season_id": row.UniqueID,
            "season_name": row.SeasonName,
            "start_date": row.StartDate,
            "end_date": row.EndDate,
            "is_done": bool(row.IsDone) if row.IsDone is not None else False,
            "frozen": bool(row.Frozen) if row.Frozen is not None else False
        })
    
    return seasons


def get_seasons_by_year(conn: pyodbc.Connection, year: int) -> list[Dict[str, Any]]:
    """
    Get all seasons for a specific year.
    
    Args:
        conn: Database connection
        year: Year to filter by (YEAR(StartDate))
        
    Returns:
        List of season dicts
    """
    cursor = conn.cursor()
    
    query = """
        SELECT 
            UniqueID,
            SeasonName,
            StartDate,
            EndDate,
            IsDone,
            Frozen
        FROM dbo.Season
        WHERE YEAR(StartDate) = ?
        ORDER BY StartDate
    """
    
    cursor.execute(query, (year,))
    rows = cursor.fetchall()
    cursor.close()
    
    seasons = []
    for row in rows:
        seasons.append({
            "season_id": row.UniqueID,
            "season_name": row.SeasonName,
            "start_date": row.StartDate,
            "end_date": row.EndDate,
            "is_done": bool(row.IsDone) if row.IsDone is not None else False,
            "frozen": bool(row.Frozen) if row.Frozen is not None else False
        })
    
    return seasons
