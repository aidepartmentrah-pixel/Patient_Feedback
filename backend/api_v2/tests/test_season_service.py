"""
📋 PHASE F — TEST F-B2 — SEASON DATE RANGE RESOLVER TESTS (API V2)

Integration tests for Season service layer.
These tests require a real database connection.
"""

import pytest
from datetime import date
from backend.api_v2.services.season_service import (
    resolve_season_date_range,
    get_all_seasons,
    get_seasons_by_year,
    SeasonNotFoundError,
    InvalidSeasonDataError
)
from backend.api_v2.db_layer import season_db


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def db_connection():
    """Provide a database connection for tests."""
    try:
        conn = season_db.get_db_connection()
        yield conn
        conn.close()
    except Exception as e:
        pytest.skip(f"Database connection not available: {e}")


@pytest.fixture(scope="module")
def valid_season_id(db_connection):
    """
    Get a valid season ID from the database for testing.
    Assumes at least one season exists in the DB.
    """
    cursor = db_connection.cursor()
    cursor.execute("SELECT TOP 1 UniqueID FROM dbo.Season WHERE StartDate IS NOT NULL AND EndDate IS NOT NULL ORDER BY StartDate DESC")
    row = cursor.fetchone()
    cursor.close()
    
    if not row:
        pytest.skip("No valid seasons found in database for testing")
    
    return row.UniqueID


@pytest.fixture(scope="module")
def season_with_known_year(db_connection):
    """
    Get a season ID and its year for year-based filtering tests.
    """
    cursor = db_connection.cursor()
    cursor.execute("""
        SELECT TOP 1 UniqueID, YEAR(StartDate) as StartYear 
        FROM dbo.Season 
        WHERE StartDate IS NOT NULL 
        ORDER BY StartDate DESC
    """)
    row = cursor.fetchone()
    cursor.close()
    
    if not row:
        pytest.skip("No valid seasons found in database for testing")
    
    return {"season_id": row.UniqueID, "year": row.StartYear}


# ============================================================================
# TEST 1 — VALID SEASON ID RETURNS DATES
# ============================================================================

def test_resolve_season_date_range_valid_id(valid_season_id):
    """
    Test that resolve_season_date_range returns valid data for a real season.
    """
    result = resolve_season_date_range(valid_season_id)
    
    # Assert structure
    assert "season_id" in result
    assert "season_name" in result
    assert "start_date" in result
    assert "end_date" in result
    
    # Assert types
    assert isinstance(result["season_id"], int)
    assert result["season_id"] == valid_season_id
    assert isinstance(result["start_date"], date)
    assert isinstance(result["end_date"], date)
    
    # Assert date logic
    assert result["start_date"] <= result["end_date"], \
        f"StartDate ({result['start_date']}) should be <= EndDate ({result['end_date']})"
    
    print(f"✅ Season {valid_season_id}: {result['season_name']}")
    print(f"   Date range: {result['start_date']} to {result['end_date']}")


# ============================================================================
# TEST 2 — MISSING SEASON ID RAISES
# ============================================================================

def test_resolve_season_date_range_missing_id():
    """
    Test that resolve_season_date_range raises SeasonNotFoundError for invalid ID.
    """
    invalid_season_id = 99999999
    
    with pytest.raises(SeasonNotFoundError) as exc_info:
        resolve_season_date_range(invalid_season_id)
    
    assert "not found" in str(exc_info.value).lower()
    assert str(invalid_season_id) in str(exc_info.value)
    
    print(f"✅ Correctly raised SeasonNotFoundError for ID {invalid_season_id}")


# ============================================================================
# TEST 3 — GET ALL SEASONS
# ============================================================================

def test_get_all_seasons():
    """
    Test that get_all_seasons returns a list of seasons.
    """
    seasons = get_all_seasons()
    
    # Should return a list (even if empty, though unlikely)
    assert isinstance(seasons, list)
    
    if len(seasons) > 0:
        # Check structure of first season
        first_season = seasons[0]
        assert "season_id" in first_season
        assert "season_name" in first_season
        assert "start_date" in first_season
        assert "end_date" in first_season
        
        # Check types
        assert isinstance(first_season["season_id"], int)
        assert isinstance(first_season["start_date"], date)
        assert isinstance(first_season["end_date"], date)
        
        print(f"✅ Found {len(seasons)} seasons in database")
        print(f"   First season: {first_season['season_name']} ({first_season['start_date']} to {first_season['end_date']})")
    else:
        pytest.skip("No seasons found in database")


# ============================================================================
# TEST 4 — GET SEASONS BY YEAR
# ============================================================================

def test_get_seasons_by_year(season_with_known_year):
    """
    Test that get_seasons_by_year returns seasons for a specific year.
    """
    year = season_with_known_year["year"]
    season_id = season_with_known_year["season_id"]
    
    seasons = get_seasons_by_year(year)
    
    # Should return a list
    assert isinstance(seasons, list)
    assert len(seasons) > 0, f"Expected at least one season for year {year}"
    
    # Check that our known season is in the results
    season_ids = [s["season_id"] for s in seasons]
    assert season_id in season_ids, f"Season {season_id} should be in year {year} results"
    
    # Check that all seasons are from the correct year
    for season in seasons:
        assert season["start_date"].year == year, \
            f"Season {season['season_id']} has StartDate year {season['start_date'].year}, expected {year}"
    
    print(f"✅ Found {len(seasons)} seasons for year {year}")


# ============================================================================
# TEST 5 — DB LAYER DIRECT TEST
# ============================================================================

def test_season_db_get_season_by_id(db_connection, valid_season_id):
    """
    Test the DB layer function directly.
    """
    season_data = season_db.get_season_by_id(db_connection, valid_season_id)
    
    assert season_data is not None
    assert season_data["season_id"] == valid_season_id
    assert isinstance(season_data["start_date"], date)
    assert isinstance(season_data["end_date"], date)
    assert "is_done" in season_data
    assert "frozen" in season_data
    
    print(f"✅ DB layer correctly retrieved season {valid_season_id}")


def test_season_db_get_season_by_id_not_found(db_connection):
    """
    Test that DB layer returns None for non-existent season.
    """
    season_data = season_db.get_season_by_id(db_connection, 99999999)
    
    assert season_data is None
    
    print("✅ DB layer correctly returned None for non-existent season")


# ============================================================================
# TEST 6 — SEASON NAME CAN BE NULL
# ============================================================================

def test_resolve_season_allows_null_name(valid_season_id):
    """
    Test that season_name being NULL doesn't cause errors.
    
    This is a data integrity test - season_name is optional.
    """
    result = resolve_season_date_range(valid_season_id)
    
    # season_name can be None or a string
    assert result["season_name"] is None or isinstance(result["season_name"], str)
    
    print(f"✅ Season name handling: {result['season_name']}")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
