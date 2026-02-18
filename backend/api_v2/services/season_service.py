"""
Season Service Layer (API V2)
Business logic for Season operations.

Used by Action Log Report and other seasonal features.
"""

from typing import Dict, Any
from datetime import date

from core.database import get_connection
from backend.api_v2.db_layer import season_db


class SeasonNotFoundError(ValueError):
    """Raised when a season record is not found."""
    pass


class InvalidSeasonDataError(ValueError):
    """Raised when season data is invalid (e.g., null dates)."""
    pass


# ============================================================
# SEASON RESOLVER
# ============================================================

def resolve_season_date_range(season_id: int) -> Dict[str, Any]:
    """
    Resolve season_id to date range and metadata.
    
    Used by Action Log Report and other time-bounded queries.
    
    Args:
        season_id: The Season UniqueID
        
    Returns:
        Dict with:
        {
            "season_id": int,
            "season_name": str | None,
            "start_date": date,
            "end_date": date
        }
        
    Raises:
        SeasonNotFoundError: If season_id does not exist
        InvalidSeasonDataError: If StartDate or EndDate is NULL
    """
    conn = get_connection()
    
    try:
        season_data = season_db.get_season_by_id(conn, season_id)
        
        if not season_data:
            raise SeasonNotFoundError(f"Season with ID {season_id} not found")
        
        # Validate required date fields
        start_date = season_data.get("start_date")
        end_date = season_data.get("end_date")
        
        if start_date is None:
            raise InvalidSeasonDataError(f"Season {season_id} has NULL StartDate")
        
        if end_date is None:
            raise InvalidSeasonDataError(f"Season {season_id} has NULL EndDate")
        
        # Validate date logic
        if start_date > end_date:
            raise InvalidSeasonDataError(
                f"Season {season_id} has invalid date range: "
                f"StartDate ({start_date}) > EndDate ({end_date})"
            )
        
        return {
            "season_id": season_data["season_id"],
            "season_name": season_data["season_name"],
            "start_date": start_date,
            "end_date": end_date
        }
    
    finally:
        conn.close()


def get_all_seasons() -> list[Dict[str, Any]]:
    """
    Get all seasons (for UI dropdowns).
    
    Returns:
        List of season dicts with basic info
    """
    conn = get_connection()
    
    try:
        return season_db.get_all_seasons(conn)
    finally:
        conn.close()


def get_seasons_by_year(year: int) -> list[Dict[str, Any]]:
    """
    Get all seasons for a specific year (for UI year-based filtering).
    
    Args:
        year: Year to filter by
        
    Returns:
        List of season dicts for that year
    """
    conn = get_connection()
    
    try:
        return season_db.get_seasons_by_year(conn, year)
    finally:
        conn.close()
