"""
DB Layer: Distribution Operator

Handles SQL generation and query execution for the Distribution Operator (DIST_1D_TIME_PARTITIONED).

This module provides database access for computing univariate categorical distributions
across time partitions. It translates time windows into SQL date filters and executes
aggregation queries on the incident data.

Key Functions:
- Time window to date range conversion (year, season, month, range)
- SQL generation for dimension aggregation
- Filter application (organizational and dimensional)
- Support for SINGLE, MULTI, and BINARY_SPLIT time modes
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import date, datetime
from decimal import Decimal
from core.database import get_connection

from ...schemas.operators.base import (
    TimeWindowYear,
    TimeWindowSeason,
    TimeWindowMonth,
    TimeWindowRange,
    TimeWindow,
    OperatorFilters
)


# ============================================================================
# TIME WINDOW CONVERSION
# ============================================================================

class TimeWindowConverter:
    """
    Converts TimeWindow objects to SQL date ranges.
    
    Handles year, season (quarters/trimesters), month, and custom range conversions.
    """
    
    @staticmethod
    def to_date_range(window: TimeWindow) -> Tuple[date, date]:
        """
        Convert a TimeWindow to a (from_date, to_date) tuple.
        
        Args:
            window: TimeWindow object (Year, Season, Month, or Range)
            
        Returns:
            Tuple of (from_date, to_date) representing the inclusive date range
            
        Examples:
            Year(2025) -> (2025-01-01, 2025-12-31)
            Season("2025-Q1") -> (2025-01-01, 2025-03-31)
            Month("2025-03") -> (2025-03-01, 2025-03-31)
            Range(2025-01-01, 2025-06-30) -> (2025-01-01, 2025-06-30)
        """
        if isinstance(window, TimeWindowYear):
            return TimeWindowConverter._year_to_range(window)
        elif isinstance(window, TimeWindowSeason):
            return TimeWindowConverter._season_to_range(window)
        elif isinstance(window, TimeWindowMonth):
            return TimeWindowConverter._month_to_range(window)
        elif isinstance(window, TimeWindowRange):
            return TimeWindowConverter._range_to_range(window)
        else:
            raise ValueError(f"Unknown time window type: {type(window)}")
    
    @staticmethod
    def _year_to_range(window: TimeWindowYear) -> Tuple[date, date]:
        """Convert year to date range"""
        year = window.value
        return (
            date(year, 1, 1),
            date(year, 12, 31)
        )
    
    @staticmethod
    def _season_to_range(window: TimeWindowSeason) -> Tuple[date, date]:
        """
        Convert season to date range.
        
        Supports:
        - Quarters (Q1-Q4): 3-month periods
        - Trimesters (T1-T3): 4-month periods
        """
        value = window.value  # Format: "2025-Q1" or "2025-T1"
        year_str, season_str = value.split("-")
        year = int(year_str)
        
        if season_str.startswith("Q"):
            # Quarterly: Q1-Q4
            quarter = int(season_str[1])
            quarter_starts = {
                1: (1, 1),   # Jan 1
                2: (4, 1),   # Apr 1
                3: (7, 1),   # Jul 1
                4: (10, 1)   # Oct 1
            }
            quarter_ends = {
                1: (3, 31),  # Mar 31
                2: (6, 30),  # Jun 30
                3: (9, 30),  # Sep 30
                4: (12, 31)  # Dec 31
            }
            start_month, start_day = quarter_starts[quarter]
            end_month, end_day = quarter_ends[quarter]
            
        elif season_str.startswith("T"):
            # Trimester: T1-T3
            trimester = int(season_str[1])
            trimester_starts = {
                1: (1, 1),   # Jan 1
                2: (5, 1),   # May 1
                3: (9, 1)    # Sep 1
            }
            trimester_ends = {
                1: (4, 30),  # Apr 30
                2: (8, 31),  # Aug 31
                3: (12, 31)  # Dec 31
            }
            start_month, start_day = trimester_starts[trimester]
            end_month, end_day = trimester_ends[trimester]
        else:
            raise ValueError(f"Invalid season format: {value}")
        
        return (
            date(year, start_month, start_day),
            date(year, end_month, end_day)
        )
    
    @staticmethod
    def _month_to_range(window: TimeWindowMonth) -> Tuple[date, date]:
        """Convert month to date range"""
        value = window.value  # Format: "2025-03"
        year_str, month_str = value.split("-")
        year = int(year_str)
        month = int(month_str)
        
        # Get last day of month
        if month == 12:
            next_month_start = date(year + 1, 1, 1)
        else:
            next_month_start = date(year, month + 1, 1)
        
        from datetime import timedelta
        last_day = next_month_start - timedelta(days=1)
        
        return (
            date(year, month, 1),
            last_day
        )
    
    @staticmethod
    def _range_to_range(window: TimeWindowRange) -> Tuple[date, date]:
        """Convert custom range (already a range)"""
        return (window.from_date, window.to_date)


# ============================================================================
# DISTRIBUTION DB CLASS
# ============================================================================

class DistributionDB:
    """
    Database layer for Distribution Operator.
    
    Provides methods to query incident data and compute categorical distributions
    across different time partitions.
    """
    
    # Map dimension names to database column names (ID columns)
    DIMENSION_COLUMNS = {
        "domain": "DomainID",
        "category": "CategoryID",
        "subcategory": "SubCategoryID",
        "classification": "ClassificationID",
        "stage": "StageID",
        "severity": "SeverityID",
        "harm": "HarmLevelID"
    }
    
    # Map dimension names to their lookup tables and name columns
    DIMENSION_LOOKUP_TABLES = {
        "domain": ("APP_LOOKUP_DOMAIN", "DomainID", "DomainName"),
        "category": ("APP_LOOKUP_CATEGORY", "CategoryID", "CategoryName"),
        "subcategory": ("APP_LOOKUP_SUBCATEGORY", "SubCategoryID", "SubCategoryName"),
        "classification": ("APP_LOOKUP_CLASSIFICATION", "ClassificationID", "Classification_EN"),
        "stage": ("APP_LOOKUP_CASE_STAGE", "StageID", "StageName"),
        "severity": ("APP_LOOKUP_SEVERITY", "SeverityID", "SeverityName"),
        "harm": ("APP_LOOKUP_HARM_LEVEL", "HarmID", "HarmLevel")
    }
    
    # Map filter names to database column names (ID columns)
    FILTER_COLUMNS = {
        "org_unit_id": "OrgUnitID",
        "administration_id": "AdministrationID",
        "department_id": "DepartmentID",
        "section_id": "SectionID",
        "domain": "DomainID",
        "category": "CategoryID",
        "subcategory": "SubCategoryID",
        "classification": "ClassificationID",
        "stage": "StageID",
        "severity": "SeverityID",
        "harm": "HarmLevelID"
    }
    
    def __init__(self, connection=None):
        """
        Initialize DistributionDB.
        
        Args:
            connection: Optional database connection. If None, creates new connections per query.
        """
        self.connection = connection
    
    def _get_conn(self):
        """Get database connection (reuse or create new)"""
        if self.connection:
            return self.connection
        return get_connection()
    
    def _close_conn(self, conn):
        """Close connection if it was created by this instance"""
        if not self.connection:
            conn.close()
    
    def query_single_window(
        self,
        dimension: str,
        time_window: TimeWindow,
        filters: OperatorFilters
    ) -> List[Dict[str, Any]]:
        """
        Query distribution for a single time window.
        
        Args:
            dimension: Dimension name (domain, category, etc.)
            time_window: Time window specification
            filters: Optional filters to apply
            
        Returns:
            List of dicts with keys: dimension_value, count
            
        Example:
            [
                {"dimension_value": "Low", "count": 123},
                {"dimension_value": "Medium", "count": 456},
                {"dimension_value": "High", "count": 78}
            ]
        """
        # Convert time window to date range
        from_date, to_date = TimeWindowConverter.to_date_range(time_window)
        
        # Build and execute query
        return self._execute_distribution_query(
            dimension=dimension,
            from_date=from_date,
            to_date=to_date,
            filters=filters
        )
    
    def query_date_range(
        self,
        dimension: str,
        from_date: Optional[date],
        to_date: Optional[date],
        filters: OperatorFilters
    ) -> List[Dict[str, Any]]:
        """
        Query distribution for an explicit date range.
        
        Args:
            dimension: Dimension name
            from_date: Start date (None = no lower bound)
            to_date: End date (None = no upper bound)
            filters: Optional filters
            
        Returns:
            List of dicts with keys: dimension_value, count
        """
        return self._execute_distribution_query(
            dimension=dimension,
            from_date=from_date,
            to_date=to_date,
            filters=filters
        )
    
    def _execute_distribution_query(
        self,
        dimension: str,
        from_date: Optional[date],
        to_date: Optional[date],
        filters: OperatorFilters
    ) -> List[Dict[str, Any]]:
        """
        Execute the core distribution query.
        
        Generates and executes SQL to compute P(dimension=v) for all values v.
        """
        # Get column name for dimension and lookup table info
        dimension_col = self.DIMENSION_COLUMNS.get(dimension)
        if not dimension_col:
            raise ValueError(f"Invalid dimension: {dimension}")
        
        # Get lookup table information
        lookup_info = self.DIMENSION_LOOKUP_TABLES.get(dimension)
        if not lookup_info:
            raise ValueError(f"No lookup table defined for dimension: {dimension}")
        
        lookup_table, lookup_id_col, lookup_name_col = lookup_info
        
        # Build SQL query with JOIN to lookup table
        sql = f"""
            SELECT 
                COALESCE(lkp.{lookup_name_col}, 'Unknown') AS dimension_value,
                COUNT(*) AS count
            FROM dbo.APP_IncidentCase ic
            LEFT JOIN dbo.{lookup_table} lkp ON ic.{dimension_col} = lkp.{lookup_id_col}
            WHERE 1=1
        """
        
        params = []
        
        # Add date filters (use FeedbackRecievedDate, not IncidentDate)
        if from_date is not None:
            sql += " AND ic.FeedbackRecievedDate >= ?"
            params.append(from_date)
        
        if to_date is not None:
            sql += " AND ic.FeedbackRecievedDate <= ?"
            params.append(to_date)
        
        # Add organizational filters (need to join with target department table)
        org_filters = []
        if filters.org_unit_id is not None:
            org_filters.append("org_unit_id")
        if filters.administration_id is not None:
            org_filters.append("administration_id")
        if filters.department_id is not None:
            org_filters.append("department_id")
        if filters.section_id is not None:
            org_filters.append("section_id")
        
        if org_filters:
            # Join with target department table for organizational filters
            # Need to add this JOIN after the lookup table JOIN
            sql = sql.replace(
                "WHERE 1=1",
                """INNER JOIN dbo.APP_IncidentCaseTargetDepartment td 
                    ON ic.IncidentCaseID = td.IncidentCaseID
            WHERE 1=1"""
            )
            
            if filters.org_unit_id is not None:
                sql += " AND td.OrgUnitID = ?"
                params.append(filters.org_unit_id)
            
            if filters.administration_id is not None:
                sql += " AND td.AdministrationID = ?"
                params.append(filters.administration_id)
            
            if filters.department_id is not None:
                sql += " AND td.DepartmentID = ?"
                params.append(filters.department_id)
            
            if filters.section_id is not None:
                sql += " AND td.SectionID = ?"
                params.append(filters.section_id)
        
        # Add dimension filters (filters on the incident case itself)
        # These are text values that need to be converted to IDs via JOIN
        dimension_filters = {
            "domain": (filters.domain, "APP_LOOKUP_DOMAIN", "DomainID", "DomainName"),
            "category": (filters.category, "APP_LOOKUP_CATEGORY", "CategoryID", "CategoryName"),
            "subcategory": (filters.subcategory, "APP_LOOKUP_SUBCATEGORY", "SubCategoryID", "SubCategoryName"),
            "classification": (filters.classification, "APP_LOOKUP_CLASSIFICATION", "ClassificationID", "Classification_EN"),
            "stage": (filters.stage, "APP_LOOKUP_CASE_STAGE", "StageID", "StageName"),
            "severity": (filters.severity, "APP_LOOKUP_SEVERITY", "SeverityID", "SeverityName"),
            "harm": (filters.harm, "APP_LOOKUP_HARM_LEVEL", "HarmID", "HarmLevel")
        }
        
        for dim_name, (dim_value, table_name, id_col, name_col) in dimension_filters.items():
            if dim_value is not None:
                # Use subquery to find ID from name
                col_name = self.FILTER_COLUMNS[dim_name]
                sql += f" AND ic.{col_name} IN (SELECT {id_col} FROM dbo.{table_name} WHERE {name_col} = ?)"
                params.append(dim_value)
        
        # Group by dimension name and order by count descending
        sql += f"""
            GROUP BY lkp.{lookup_name_col}
            ORDER BY count DESC, lkp.{lookup_name_col}
        """
        
        # Execute query
        conn = self._get_conn()
        cursor = conn.cursor()
        
        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description]
            
            # Convert to list of dicts
            results = []
            for row in rows:
                row_dict = dict(zip(columns, row))
                # Convert Decimal to int for count
                if isinstance(row_dict.get('count'), Decimal):
                    row_dict['count'] = int(row_dict['count'])
                results.append(row_dict)
            
            return results
        finally:
            cursor.close()
            self._close_conn(conn)
