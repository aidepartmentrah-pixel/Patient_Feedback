"""
Test Suite: Distribution DB Layer

Tests for SQL generation, time window conversion, and query execution.
Uses mocks to avoid requiring actual database connections.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from datetime import date
from decimal import Decimal

from api.db_layer.operators.distribution_db import (
    DistributionDB,
    TimeWindowConverter
)
from api.schemas.operators.base import (
    TimeWindowYear,
    TimeWindowSeason,
    TimeWindowMonth,
    TimeWindowRange,
    OperatorFilters
)


# ============================================================================
# TEST TIME WINDOW CONVERTER
# ============================================================================

class TestTimeWindowConverter:
    """Test TimeWindowConverter date range conversions"""
    
    def test_year_conversion(self):
        """Test year to date range conversion"""
        window = TimeWindowYear(value=2025)
        from_date, to_date = TimeWindowConverter.to_date_range(window)
        
        assert from_date == date(2025, 1, 1)
        assert to_date == date(2025, 12, 31)
    
    def test_year_conversion_edge_years(self):
        """Test year conversion for edge years"""
        # Year 2000
        window = TimeWindowYear(value=2000)
        from_date, to_date = TimeWindowConverter.to_date_range(window)
        assert from_date == date(2000, 1, 1)
        assert to_date == date(2000, 12, 31)
        
        # Year 2100
        window = TimeWindowYear(value=2100)
        from_date, to_date = TimeWindowConverter.to_date_range(window)
        assert from_date == date(2100, 1, 1)
        assert to_date == date(2100, 12, 31)
    
    def test_quarter_conversions(self):
        """Test all quarters convert correctly"""
        test_cases = [
            ("2025-Q1", date(2025, 1, 1), date(2025, 3, 31)),
            ("2025-Q2", date(2025, 4, 1), date(2025, 6, 30)),
            ("2025-Q3", date(2025, 7, 1), date(2025, 9, 30)),
            ("2025-Q4", date(2025, 10, 1), date(2025, 12, 31)),
        ]
        
        for season_str, expected_from, expected_to in test_cases:
            window = TimeWindowSeason(value=season_str)
            from_date, to_date = TimeWindowConverter.to_date_range(window)
            assert from_date == expected_from, f"Failed for {season_str}: expected {expected_from}, got {from_date}"
            assert to_date == expected_to, f"Failed for {season_str}: expected {expected_to}, got {to_date}"
    
    def test_trimester_conversions(self):
        """Test all trimesters convert correctly"""
        test_cases = [
            ("2025-T1", date(2025, 1, 1), date(2025, 4, 30)),
            ("2025-T2", date(2025, 5, 1), date(2025, 8, 31)),
            ("2025-T3", date(2025, 9, 1), date(2025, 12, 31)),
        ]
        
        for season_str, expected_from, expected_to in test_cases:
            window = TimeWindowSeason(value=season_str)
            from_date, to_date = TimeWindowConverter.to_date_range(window)
            assert from_date == expected_from, f"Failed for {season_str}"
            assert to_date == expected_to, f"Failed for {season_str}"
    
    def test_month_conversions(self):
        """Test month conversions for various months"""
        test_cases = [
            ("2025-01", date(2025, 1, 1), date(2025, 1, 31)),   # 31 days
            ("2025-02", date(2025, 2, 1), date(2025, 2, 28)),   # 28 days (non-leap)
            ("2024-02", date(2024, 2, 1), date(2024, 2, 29)),   # 29 days (leap year)
            ("2025-04", date(2025, 4, 1), date(2025, 4, 30)),   # 30 days
            ("2025-12", date(2025, 12, 1), date(2025, 12, 31)), # December
        ]
        
        for month_str, expected_from, expected_to in test_cases:
            window = TimeWindowMonth(value=month_str)
            from_date, to_date = TimeWindowConverter.to_date_range(window)
            assert from_date == expected_from, f"Failed for {month_str}"
            assert to_date == expected_to, f"Failed for {month_str}"
    
    def test_range_conversion(self):
        """Test custom range (pass-through)"""
        window = TimeWindowRange(
            from_date=date(2025, 3, 15),
            to_date=date(2025, 6, 20)
        )
        from_date, to_date = TimeWindowConverter.to_date_range(window)
        
        assert from_date == date(2025, 3, 15)
        assert to_date == date(2025, 6, 20)
    
    def test_invalid_season_format_raises_error(self):
        """Test that invalid season format raises ValueError"""
        # This should not be reached due to Pydantic validation, but test defensive code
        window = Mock()
        window.value = "2025-X1"  # Invalid season type
        
        with pytest.raises(ValueError, match="Invalid season format"):
            TimeWindowConverter._season_to_range(window)


# ============================================================================
# TEST DISTRIBUTION DB
# ============================================================================

class TestDistributionDB:
    """Test DistributionDB SQL generation and query execution"""
    
    def test_dimension_column_mapping(self):
        """Test that all dimensions map to correct column names"""
        db = DistributionDB()
        
        assert db.DIMENSION_COLUMNS["domain"] == "Domain"
        assert db.DIMENSION_COLUMNS["category"] == "Category"
        assert db.DIMENSION_COLUMNS["subcategory"] == "SubCategory"
        assert db.DIMENSION_COLUMNS["classification"] == "Classification"
        assert db.DIMENSION_COLUMNS["stage"] == "Stage"
        assert db.DIMENSION_COLUMNS["severity"] == "Severity"
        assert db.DIMENSION_COLUMNS["harm"] == "Harm"
        assert len(db.DIMENSION_COLUMNS) == 7
    
    def test_invalid_dimension_raises_error(self):
        """Test that invalid dimension name raises ValueError"""
        db = DistributionDB()
        
        with pytest.raises(ValueError, match="Invalid dimension"):
            db._execute_distribution_query(
                dimension="invalid_dimension",
                from_date=date(2025, 1, 1),
                to_date=date(2025, 12, 31),
                filters=OperatorFilters()
            )
    
    @patch('backend.api.db_layer.operators.distribution_db.get_connection')
    def test_query_single_window_with_year(self, mock_get_conn):
        """Test query_single_window with year time window"""
        # Setup mock
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn
        
        # Mock query results
        mock_cursor.fetchall.return_value = [
            ("High", 150),
            ("Medium", 300),
            ("Low", 50)
        ]
        mock_cursor.description = [("dimension_value",), ("count",)]
        
        # Execute
        db = DistributionDB()
        results = db.query_single_window(
            dimension="severity",
            time_window=TimeWindowYear(value=2025),
            filters=OperatorFilters()
        )
        
        # Verify results
        assert len(results) == 3
        assert results[0] == {"dimension_value": "High", "count": 150}
        assert results[1] == {"dimension_value": "Medium", "count": 300}
        assert results[2] == {"dimension_value": "Low", "count": 50}
        
        # Verify SQL was executed with correct date range
        executed_sql = mock_cursor.execute.call_args[0][0]
        executed_params = mock_cursor.execute.call_args[0][1]
        
        assert "ic.Severity" in executed_sql
        assert "ic.IncidentDate >= ?" in executed_sql
        assert "ic.IncidentDate <= ?" in executed_sql
        assert executed_params[0] == date(2025, 1, 1)
        assert executed_params[1] == date(2025, 12, 31)
    
    @patch('backend.api.db_layer.operators.distribution_db.get_connection')
    def test_query_with_organizational_filters(self, mock_get_conn):
        """Test query with organizational filters joins target department table"""
        # Setup mock
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn
        
        mock_cursor.fetchall.return_value = []
        mock_cursor.description = [("dimension_value",), ("count",)]
        
        # Execute with department filter
        db = DistributionDB()
        filters = OperatorFilters(department_id=42)
        
        db.query_single_window(
            dimension="domain",
            time_window=TimeWindowYear(value=2025),
            filters=filters
        )
        
        # Verify SQL includes join and filter
        executed_sql = mock_cursor.execute.call_args[0][0]
        executed_params = mock_cursor.execute.call_args[0][1]
        
        assert "APP_IncidentCaseTargetDepartment" in executed_sql
        assert "INNER JOIN" in executed_sql
        assert "td.DepartmentID = ?" in executed_sql
        assert 42 in executed_params
    
    @patch('backend.api.db_layer.operators.distribution_db.get_connection')
    def test_query_with_dimension_filters(self, mock_get_conn):
        """Test query with dimension filters (e.g., severity=High)"""
        # Setup mock
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn
        
        mock_cursor.fetchall.return_value = []
        mock_cursor.description = [("dimension_value",), ("count",)]
        
        # Execute with severity filter
        db = DistributionDB()
        filters = OperatorFilters(severity="High")
        
        db.query_single_window(
            dimension="domain",
            time_window=TimeWindowYear(value=2025),
            filters=filters
        )
        
        # Verify SQL includes filter
        executed_sql = mock_cursor.execute.call_args[0][0]
        executed_params = mock_cursor.execute.call_args[0][1]
        
        assert "ic.Severity = ?" in executed_sql
        assert "High" in executed_params
    
    @patch('backend.api.db_layer.operators.distribution_db.get_connection')
    def test_query_with_combined_filters(self, mock_get_conn):
        """Test query with both organizational and dimension filters"""
        # Setup mock
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn
        
        mock_cursor.fetchall.return_value = []
        mock_cursor.description = [("dimension_value",), ("count",)]
        
        # Execute with multiple filters
        db = DistributionDB()
        filters = OperatorFilters(
            department_id=42,
            severity="High",
            domain="Clinical"
        )
        
        db.query_single_window(
            dimension="stage",
            time_window=TimeWindowYear(value=2025),
            filters=filters
        )
        
        # Verify SQL includes all filters
        executed_sql = mock_cursor.execute.call_args[0][0]
        executed_params = mock_cursor.execute.call_args[0][1]
        
        assert "APP_IncidentCaseTargetDepartment" in executed_sql
        assert "td.DepartmentID = ?" in executed_sql
        assert "ic.Severity = ?" in executed_sql
        assert "ic.Domain = ?" in executed_sql
        assert 42 in executed_params
        assert "High" in executed_params
        assert "Clinical" in executed_params
    
    @patch('backend.api.db_layer.operators.distribution_db.get_connection')
    def test_query_date_range_with_none_dates(self, mock_get_conn):
        """Test query_date_range with None dates (no bounds)"""
        # Setup mock
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn
        
        mock_cursor.fetchall.return_value = []
        mock_cursor.description = [("dimension_value",), ("count",)]
        
        # Execute with no date bounds
        db = DistributionDB()
        db.query_date_range(
            dimension="severity",
            from_date=None,
            to_date=None,
            filters=OperatorFilters()
        )
        
        # Verify SQL does NOT include date filters
        executed_sql = mock_cursor.execute.call_args[0][0]
        
        assert "IncidentDate >=" not in executed_sql
        assert "IncidentDate <=" not in executed_sql
    
    @patch('backend.api.db_layer.operators.distribution_db.get_connection')
    def test_query_date_range_with_from_date_only(self, mock_get_conn):
        """Test query_date_range with only from_date"""
        # Setup mock
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn
        
        mock_cursor.fetchall.return_value = []
        mock_cursor.description = [("dimension_value",), ("count",)]
        
        # Execute with only from_date
        db = DistributionDB()
        db.query_date_range(
            dimension="severity",
            from_date=date(2023, 1, 1),
            to_date=None,
            filters=OperatorFilters()
        )
        
        # Verify SQL includes only >= filter
        executed_sql = mock_cursor.execute.call_args[0][0]
        executed_params = mock_cursor.execute.call_args[0][1]
        
        assert "ic.IncidentDate >= ?" in executed_sql
        assert "ic.IncidentDate <=" not in executed_sql
        assert date(2023, 1, 1) in executed_params
    
    @patch('backend.api.db_layer.operators.distribution_db.get_connection')
    def test_query_converts_decimal_to_int(self, mock_get_conn):
        """Test that Decimal counts are converted to int"""
        # Setup mock
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn
        
        # Mock query returns Decimal (as SQL Server does)
        mock_cursor.fetchall.return_value = [
            ("High", Decimal("150")),
            ("Medium", Decimal("300"))
        ]
        mock_cursor.description = [("dimension_value",), ("count",)]
        
        # Execute
        db = DistributionDB()
        results = db.query_single_window(
            dimension="severity",
            time_window=TimeWindowYear(value=2025),
            filters=OperatorFilters()
        )
        
        # Verify counts are integers
        assert isinstance(results[0]["count"], int)
        assert isinstance(results[1]["count"], int)
        assert results[0]["count"] == 150
        assert results[1]["count"] == 300
    
    @patch('backend.api.db_layer.operators.distribution_db.get_connection')
    def test_connection_cleanup(self, mock_get_conn):
        """Test that connection is properly closed after query"""
        # Setup mock
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn
        
        mock_cursor.fetchall.return_value = []
        mock_cursor.description = [("dimension_value",), ("count",)]
        
        # Execute
        db = DistributionDB()
        db.query_single_window(
            dimension="severity",
            time_window=TimeWindowYear(value=2025),
            filters=OperatorFilters()
        )
        
        # Verify connection was closed
        mock_conn.close.assert_called_once()
        mock_cursor.close.assert_called_once()
    
    def test_reused_connection_not_closed(self):
        """Test that provided connection is NOT closed"""
        # Create a mock connection to reuse
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        
        mock_cursor.fetchall.return_value = []
        mock_cursor.description = [("dimension_value",), ("count",)]
        
        # Create DB with reusable connection
        db = DistributionDB(connection=mock_conn)
        db.query_single_window(
            dimension="severity",
            time_window=TimeWindowYear(value=2025),
            filters=OperatorFilters()
        )
        
        # Verify connection was NOT closed
        mock_conn.close.assert_not_called()
        # But cursor should still be closed
        mock_cursor.close.assert_called_once()
    
    @patch('backend.api.db_layer.operators.distribution_db.get_connection')
    def test_sql_includes_group_by_and_order(self, mock_get_conn):
        """Test that SQL includes GROUP BY and ORDER BY clauses"""
        # Setup mock
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn
        
        mock_cursor.fetchall.return_value = []
        mock_cursor.description = [("dimension_value",), ("count",)]
        
        # Execute
        db = DistributionDB()
        db.query_single_window(
            dimension="severity",
            time_window=TimeWindowYear(value=2025),
            filters=OperatorFilters()
        )
        
        # Verify SQL structure
        executed_sql = mock_cursor.execute.call_args[0][0]
        
        assert "GROUP BY ic.Severity" in executed_sql
        assert "ORDER BY count DESC" in executed_sql
    
    @patch('backend.api.db_layer.operators.distribution_db.get_connection')
    def test_all_dimensions_work(self, mock_get_conn):
        """Test that all 7 dimensions generate valid SQL"""
        # Setup mock
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn
        
        mock_cursor.fetchall.return_value = []
        mock_cursor.description = [("dimension_value",), ("count",)]
        
        db = DistributionDB()
        dimensions = ["domain", "category", "subcategory", "classification", "stage", "severity", "harm"]
        
        for dimension in dimensions:
            mock_cursor.reset_mock()
            
            db.query_single_window(
                dimension=dimension,
                time_window=TimeWindowYear(value=2025),
                filters=OperatorFilters()
            )
            
            # Verify query was executed
            assert mock_cursor.execute.called, f"Query not executed for dimension: {dimension}"
            
            # Verify correct column in SQL
            executed_sql = mock_cursor.execute.call_args[0][0]
            expected_col = db.DIMENSION_COLUMNS[dimension]
            assert f"ic.{expected_col}" in executed_sql, f"Missing column for dimension: {dimension}"
