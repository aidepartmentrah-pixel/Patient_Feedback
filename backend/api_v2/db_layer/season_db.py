"""
Season Database Layer (API V2)
Handles SQL operations for dbo.Season table.

NO business logic. NO authorization. ONLY SQL operations.
"""

from typing import Dict, Any, Optional
from datetime import date
import pyodbc
from core.database import get_connection


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
