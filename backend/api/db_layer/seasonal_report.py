"""
DB Layer for Seasonal Report Persistence
Handles CRUD operations for materialized seasonal reports and related tables.
"""

from typing import Dict, Any, List, Optional
from core.database import get_connection


def resolve_season_id_from_year_trimester(year: int, trimester: str) -> Optional[int]:
    """
    Resolve season_id (UniqueID) from year and period identifier.
    
    Supports both Quarter and Trimester formats:
    
    Quarter mapping (3-month periods):
    - Q1: Jan-Mar (months 1-3)
    - Q2: Apr-Jun (months 4-6)
    - Q3: Jul-Sep (months 7-9)
    - Q4: Oct-Dec (months 10-12)
    
    Trimester mapping (4-month periods):
    - Trim1: Jan-Apr (months 1-4)
    - Trim2: May-Aug (months 5-8)
    - Trim3: Sep-Dec (months 9-12)
    
    Args:
        year: Calendar year (e.g., 2025)
        trimester: Period identifier (Q1, Q2, Q3, Q4, Trim1, Trim2, Trim3)
    
    Returns:
        Season UniqueID if found uniquely, None if not found
    
    Raises:
        ValueError: If multiple seasons match or period format invalid
    """
    conn = None
    cursor = None
    
    # Quarter mapping (3-month periods) - standard business quarters
    quarter_ranges = {
        "Q1": (1, 3),      # Jan-Mar
        "Q2": (4, 6),      # Apr-Jun
        "Q3": (7, 9),      # Jul-Sep
        "Q4": (10, 12)     # Oct-Dec
    }
    
    # Trimester mapping (4-month periods) - legacy support
    trimester_ranges = {
        "Trim1": (1, 4),   # Jan-Apr
        "Trim2": (5, 8),   # May-Aug
        "Trim3": (9, 12)   # Sep-Dec
    }
    
    # Determine which format and get month range
    if trimester in quarter_ranges:
        start_month, end_month = quarter_ranges[trimester]
    elif trimester in trimester_ranges:
        start_month, end_month = trimester_ranges[trimester]
    else:
        valid_formats = list(quarter_ranges.keys()) + list(trimester_ranges.keys())
        raise ValueError(
            f"Invalid period: {trimester}. Must be one of {valid_formats}"
        )
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Query seasons that overlap with the year-trimester range
        cursor.execute(
            """
            SELECT UniqueID
            FROM dbo.Season
            WHERE YEAR(StartDate) = ?
              AND MONTH(StartDate) >= ?
              AND MONTH(StartDate) <= ?
            """,
            (year, start_month, end_month)
        )
        
        rows = cursor.fetchall()
        
        if not rows:
            # Season doesn't exist - auto-create for quarters only
            if trimester in quarter_ranges:
                # Auto-create the season
                season_id = create_season_if_not_exists(year, trimester)
                return season_id
            else:
                # For trimesters, don't auto-create (legacy format)
                return None
        
        if len(rows) > 1:
            raise ValueError(f"Ambiguous season: Multiple seasons found for year={year}, trimester={trimester}")
        
        return rows[0].UniqueID
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def create_season_if_not_exists(year: int, period: str) -> int:
    """
    Create a Season record if it doesn't exist for the given year and period.
    
    Uses the month/day pattern from an existing season with the same period identifier.
    For example, if creating Q1-2026, it will find any existing Q1-* season (e.g., Q1-2025)
    and extract the consistent start/end month/day, then apply them to the new year.
    
    Args:
        year: Calendar year (e.g., 2026)
        period: Period identifier (Q1, Q2, Q3, Q4)
    
    Returns:
        Season UniqueID (existing or newly created)
    
    Raises:
        ValueError: If period format is invalid or no template season found
    """
    if not period.startswith("Q") or period not in ["Q1", "Q2", "Q3", "Q4"]:
        raise ValueError(f"Cannot auto-create season for period: {period}. Only Q1-Q4 supported.")
    
    season_name = f"{period}-{year}"
    
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Check if season already exists
        cursor.execute(
            """
            SELECT UniqueID 
            FROM dbo.Season 
            WHERE SeasonName = ?
            """,
            (season_name,)
        )
        
        existing = cursor.fetchone()
        if existing:
            return existing.UniqueID
        
        # Find an existing season with the same period to extract month/day pattern
        cursor.execute(
            """
            SELECT TOP 1 StartDate, EndDate
            FROM dbo.Season
            WHERE SeasonName LIKE ?
            ORDER BY SeasonName DESC
            """,
            (f"{period}-%",)
        )
        
        template = cursor.fetchone()
        if not template:
            raise ValueError(
                f"Cannot auto-create {period}-{year}: No existing {period} season found to use as template. "
                f"Please create the first {period} season manually in the database."
            )
        
        # Extract month and day from template, apply to new year
        template_start = template.StartDate
        template_end = template.EndDate
        
        start_date = template_start.replace(year=year)
        end_date = template_end.replace(year=year)
        
        # Get the next UniqueID (UniqueID is NOT an IDENTITY column)
        cursor.execute("SELECT ISNULL(MAX(UniqueID), 0) + 1 AS NextID FROM dbo.Season")
        next_id = cursor.fetchone().NextID
        
        # Create new season
        cursor.execute(
            """
            INSERT INTO dbo.Season (UniqueID, SeasonName, StartDate, EndDate, IsDone, Frozen, CreateDate, CreateID)
            VALUES (?, ?, ?, ?, 0, 0, GETDATE(), 1)
            """,
            (next_id, season_name, start_date, end_date)
        )
        
        conn.commit()
        
        return int(next_id)
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_previous_season(season_id: int) -> Optional[int]:
    """
    Get the previous season ID for a given season.
    
    Logic:
    - Q1-2025 → Q4-2024
    - Q2-2025 → Q1-2025
    - Q3-2025 → Q2-2025
    - Q4-2025 → Q3-2025
    - Trim1-2025 → Trim3-2024
    - Trim2-2025 → Trim1-2025
    - Trim3-2025 → Trim2-2025
    
    Args:
        season_id: Current season UniqueID
    
    Returns:
        Previous season UniqueID if exists, None otherwise
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get current season info
        cursor.execute(
            """
            SELECT SeasonName, StartDate, EndDate
            FROM dbo.Season
            WHERE UniqueID = ?
            """,
            (season_id,)
        )
        
        current_season = cursor.fetchone()
        if not current_season:
            return None
        
        season_name = current_season.SeasonName
        
        # Parse season name (format: Q1-2025 or Trim1-2025)
        if '-' not in season_name:
            return None
        
        period, year_str = season_name.split('-')
        year = int(year_str)
        
        # Determine previous season based on period type
        previous_period = None
        previous_year = year
        
        if period.startswith('Q'):
            # Quarter logic
            quarter_num = int(period[1])
            if quarter_num == 1:
                previous_period = 'Q4'
                previous_year = year - 1
            else:
                previous_period = f'Q{quarter_num - 1}'
        elif period.startswith('Trim'):
            # Trimester logic
            trim_num = int(period[4])
            if trim_num == 1:
                previous_period = 'Trim3'
                previous_year = year - 1
            else:
                previous_period = f'Trim{trim_num - 1}'
        else:
            # Unknown format
            return None
        
        # Look up previous season
        previous_season_name = f'{previous_period}-{previous_year}'
        cursor.execute(
            """
            SELECT UniqueID
            FROM dbo.Season
            WHERE SeasonName = ?
            """,
            (previous_season_name,)
        )
        
        previous_row = cursor.fetchone()
        return previous_row.UniqueID if previous_row else None
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def validate_season_exists(season_id: int) -> bool:
    """
    Check if a season exists in the Season table.
    
    Args:
        season_id: Season identifier (UniqueID)
    
    Returns:
        True if season exists, False otherwise
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM dbo.Season
            WHERE UniqueID = ?
            """,
            (season_id,)
        )
        
        row = cursor.fetchone()
        return row[0] > 0 if row else False
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_existing_seasonal_report_id(
    season_id: int,
    orgunit_id: int,
    orgunit_type: int
) -> Optional[int]:
    """
    Get existing seasonal report ID for the given key.
    
    Args:
        season_id: Season identifier
        orgunit_id: Organizational unit identifier
        orgunit_type: Type of organizational unit
    
    Returns:
        SeasonalReportID if exists, None otherwise
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT SeasonalReportID
            FROM dbo.APP_SeasonalOrgUnitReport
            WHERE SeasonID = ? AND OrgUnitID = ? AND OrgUnitType = ?
            """,
            (season_id, orgunit_id, orgunit_type)
        )
        
        row = cursor.fetchone()
        return row.SeasonalReportID if row else None
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def update_seasonal_report_header(
    seasonal_report_id: int,
    total_cases: int,
    low_severity_count: int,
    medium_severity_count: int,
    high_severity_count: int,
    clinical_domain_count: int,
    management_domain_count: int,
    relational_domain_count: int,
    is_compliant: bool,
    violated_rules: Optional[str],
    explanation_status_id: int,
    updated_by_user_id: int
) -> None:
    """
    Update existing seasonal report header record.
    Preserves the existing SeasonalReportID and linked action items.
    
    Args:
        seasonal_report_id: Existing report ID to update
        total_cases: Total case count
        low_severity_count: Low severity case count
        medium_severity_count: Medium severity case count
        high_severity_count: High severity case count
        clinical_domain_count: Clinical domain case count
        management_domain_count: Management domain case count
        relational_domain_count: Relational domain case count
        is_compliant: Compliance status (1=compliant, 0=violated)
        violated_rules: JSON string of violated rules or None
        explanation_status_id: Explanation status FK
        updated_by_user_id: User who regenerated the report
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            UPDATE dbo.APP_SeasonalOrgUnitReport
            SET
                TotalCases = ?,
                LowSeverityCount = ?,
                MediumSeverityCount = ?,
                HighSeverityCount = ?,
                ClinicalDomainCount = ?,
                ManagementDomainCount = ?,
                RelationalDomainCount = ?,
                IsCompliant = ?,
                ViolatedRules = ?,
                ExplanationStatusID = ?,
                EvaluatedAt = GETDATE(),
                CreatedByUserID = ?
            WHERE SeasonalReportID = ?
            """,
            (
                total_cases,
                low_severity_count,
                medium_severity_count,
                high_severity_count,
                clinical_domain_count,
                management_domain_count,
                relational_domain_count,
                is_compliant,
                violated_rules,
                explanation_status_id,
                updated_by_user_id,
                seasonal_report_id
            )
        )
        
        conn.commit()
    
    except Exception as e:
        if conn:
            conn.rollback()
        raise Exception(f"Failed to update seasonal report header: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def delete_seasonal_report_by_key(
    season_id: int,
    orgunit_id: int,
    orgunit_type: int
) -> None:
    """
    Delete existing seasonal report by unique key.
    Cascade deletes classification stats, policy snapshot, and action items.
    
    Args:
        season_id: Season identifier
        orgunit_id: Organizational unit identifier
        orgunit_type: Type of organizational unit (1=Dept, 2=Building, 3=Org)
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Find existing report ID
        cursor.execute(
            """
            SELECT SeasonalReportID
            FROM dbo.APP_SeasonalOrgUnitReport
            WHERE SeasonID = ? AND OrgUnitID = ? AND OrgUnitType = ?
            """,
            (season_id, orgunit_id, orgunit_type)
        )
        
        row = cursor.fetchone()
        if row:
            seasonal_report_id = row.SeasonalReportID
            
            # Delete report (cascade handles children)
            cursor.execute(
                """
                DELETE FROM dbo.APP_SeasonalOrgUnitReport
                WHERE SeasonalReportID = ?
                """,
                (seasonal_report_id,)
            )
        
        conn.commit()
    
    except Exception as e:
        if conn:
            conn.rollback()
        raise Exception(f"Failed to delete seasonal report: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def insert_seasonal_report_header(
    season_id: int,
    orgunit_id: int,
    orgunit_type: int,
    total_cases: int,
    low_severity_count: int,
    medium_severity_count: int,
    high_severity_count: int,
    clinical_domain_count: int,
    management_domain_count: int,
    relational_domain_count: int,
    is_compliant: bool,
    violated_rules: Optional[str],
    explanation_status_id: int,
    created_by_user_id: int
) -> int:
    """
    Insert seasonal report header record.
    
    Args:
        season_id: Season identifier
        orgunit_id: Organizational unit identifier
        orgunit_type: Type of organizational unit
        total_cases: Total case count
        low_severity_count: Low severity case count
        medium_severity_count: Medium severity case count
        high_severity_count: High severity case count
        clinical_domain_count: Clinical domain case count
        management_domain_count: Management domain case count
        relational_domain_count: Relational domain case count
        is_compliant: Compliance status (1=compliant, 0=violated)
        violated_rules: JSON string of violated rules or None
        explanation_status_id: Explanation status FK
        created_by_user_id: User who created the report
    
    Returns:
        int: New SeasonalReportID
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            INSERT INTO dbo.APP_SeasonalOrgUnitReport (
                SeasonID,
                OrgUnitID,
                OrgUnitType,
                TotalCases,
                LowSeverityCount,
                MediumSeverityCount,
                HighSeverityCount,
                ClinicalDomainCount,
                ManagementDomainCount,
                RelationalDomainCount,
                IsCompliant,
                ViolatedRules,
                ExplanationStatusID,
                CreatedByUserID,
                CreatedAt
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, GETDATE())
            """,
            (
                season_id,
                orgunit_id,
                orgunit_type,
                total_cases,
                low_severity_count,
                medium_severity_count,
                high_severity_count,
                clinical_domain_count,
                management_domain_count,
                relational_domain_count,
                1 if is_compliant else 0,
                violated_rules,
                explanation_status_id,
                created_by_user_id
            )
        )
        
        # Get the new ID
        cursor.execute("SELECT @@IDENTITY AS ID")
        new_id = cursor.fetchone().ID
        
        conn.commit()
        return int(new_id)
    
    except Exception as e:
        if conn:
            conn.rollback()
        raise Exception(f"Failed to insert seasonal report header: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def delete_seasonal_report_classification_stats(seasonal_report_id: int) -> None:
    """
    Delete all classification stats for a seasonal report.
    
    Args:
        seasonal_report_id: Parent report ID
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            DELETE FROM dbo.APP_SeasonalOrgUnitReport_ClassificationStats
            WHERE SeasonalReportID = ?
            """,
            (seasonal_report_id,)
        )
        
        conn.commit()
    
    except Exception as e:
        if conn:
            conn.rollback()
        raise Exception(f"Failed to delete classification stats: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def delete_seasonal_report_policy_snapshot(seasonal_report_id: int) -> None:
    """
    Delete policy snapshot for a seasonal report.
    
    Args:
        seasonal_report_id: Parent report ID
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            DELETE FROM dbo.APP_SeasonalOrgUnitReport_PolicySnapshot
            WHERE SeasonalReportID = ?
            """,
            (seasonal_report_id,)
        )
        
        conn.commit()
    
    except Exception as e:
        if conn:
            conn.rollback()
        raise Exception(f"Failed to delete policy snapshot: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def insert_seasonal_report_classification_stats(
    seasonal_report_id: int,
    stats: List[Dict[str, Any]]
) -> None:
    """
    Bulk insert classification statistics for a seasonal report.
    
    Args:
        seasonal_report_id: Parent report ID
        stats: List of stat dictionaries, each containing:
            - classification_id: int
            - total_count: int
            - low_count: int
            - medium_count: int
            - high_count: int
            - preventive_yes_count: int
            - preventive_no_count: int
    """
    if not stats:
        return
    
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        for stat in stats:
            cursor.execute(
                """
                INSERT INTO dbo.APP_SeasonalOrgUnitReport_ClassificationStats (
                    SeasonalReportID,
                    ClassificationID,
                    DomainID,
                    CategoryID,
                    SubCategoryID,
                    TotalCount,
                    LowCount,
                    MediumCount,
                    HighCount,
                    PreventiveYesCount,
                    PreventiveNoCount
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    seasonal_report_id,
                    stat['classification_id'],
                    stat['domain_id'],
                    stat['category_id'],
                    stat['subcategory_id'],
                    stat['total_count'],
                    stat['low_count'],
                    stat['medium_count'],
                    stat['high_count'],
                    stat['preventive_yes_count'],
                    stat['preventive_no_count']
                )
            )
        
        conn.commit()
    
    except Exception as e:
        if conn:
            conn.rollback()
        raise Exception(f"Failed to insert classification stats: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def insert_seasonal_report_policy_snapshot(
    seasonal_report_id: int,
    policy_row: Dict[str, Any]
) -> None:
    """
    Insert policy snapshot for a seasonal report.
    Copies the current policy state at report generation time.
    
    Args:
        seasonal_report_id: Parent report ID
        policy_row: Dictionary containing all policy fields from APP_OrgUnitPolicy
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            INSERT INTO dbo.APP_SeasonalOrgUnitReport_PolicySnapshot (
                SeasonalReportID,
                LowSeverityLimit,
                MediumSeverityLimit,
                HighSeverityLimit,
                ClinicalDomainLimit,
                ManagementDomainLimit,
                RelationalDomainLimit,
                EnableLowSeverityRepetitionRule,
                EnableMediumSeverityRepetitionRule,
                EnableHighSeverityPercentageRule,
                EnableHighSeverityPercentageByDomainRule
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                seasonal_report_id,
                policy_row.get('LowSeverityLimit'),
                policy_row.get('MediumSeverityLimit'),
                policy_row.get('HighSeverityLimit'),
                policy_row.get('ClinicalDomainLimit'),
                policy_row.get('ManagementDomainLimit'),
                policy_row.get('RelationalDomainLimit'),
                policy_row.get('EnableLowSeverityRepetitionRule'),
                policy_row.get('EnableMediumSeverityRepetitionRule'),
                policy_row.get('EnableHighSeverityPercentageRule'),
                policy_row.get('EnableHighSeverityPercentageByDomainRule')
            )
        )
        
        conn.commit()
    
    except Exception as e:
        if conn:
            conn.rollback()
        raise Exception(f"Failed to insert policy snapshot: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_seasonal_report_keys_by_id(seasonal_report_id: int) -> Optional[Dict[str, int]]:
    """
    Get the unique keys (SeasonID, OrgUnitID, OrgUnitType) for a seasonal report.
    
    Args:
        seasonal_report_id: SeasonalReportID
    
    Returns:
        Dictionary with keys: season_id, orgunit_id, orgunit_type
        Returns None if report not found
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT SeasonID, OrgUnitID, OrgUnitType
            FROM dbo.APP_SeasonalOrgUnitReport
            WHERE SeasonalReportID = ?
            """,
            (seasonal_report_id,)
        )
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return {
            'season_id': row.SeasonID,
            'orgunit_id': row.OrgUnitID,
            'orgunit_type': row.OrgUnitType
        }
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_full_seasonal_report(
    season_id: int,
    orgunit_id: int,
    orgunit_type: int
) -> Optional[Dict[str, Any]]:
    """
    Fetch complete seasonal report with all related data.
    
    Args:
        season_id: Season identifier
        orgunit_id: Organizational unit identifier
        orgunit_type: Type of organizational unit
    
    Returns:
        Dictionary with keys:
            - header: Report header data
            - classification_stats: List of classification statistics
            - policy_snapshot: Policy snapshot data
        Returns None if report not found.
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Fetch header with season and orgunit names
        cursor.execute(
            """
            SELECT
                sr.SeasonalReportID,
                sr.SeasonID,
                sr.OrgUnitID,
                sr.OrgUnitType,
                sr.TotalCases,
                sr.LowSeverityCount,
                sr.MediumSeverityCount,
                sr.HighSeverityCount,
                sr.ClinicalDomainCount,
                sr.ManagementDomainCount,
                sr.RelationalDomainCount,
                sr.IsCompliant,
                sr.ViolatedRules,
                sr.ExplanationStatusID,
                sr.CreatedByUserID,
                sr.CreatedAt,
                s.SeasonName as SeasonPeriod,
                ou.Name as OrgUnitName
            FROM dbo.APP_SeasonalOrgUnitReport sr
            LEFT JOIN dbo.Season s ON sr.SeasonID = s.UniqueID
            LEFT JOIN dbo.AdminsrationUnit ou ON sr.OrgUnitID = ou.UniqueID
            WHERE sr.SeasonID = ? AND sr.OrgUnitID = ? AND sr.OrgUnitType = ?
            """,
            (season_id, orgunit_id, orgunit_type)
        )
        
        header_row = cursor.fetchone()
        if not header_row:
            return None
        
        header = {
            'seasonal_report_id': header_row.SeasonalReportID,
            'season_id': header_row.SeasonID,
            'orgunit_id': header_row.OrgUnitID,
            'orgunit_type': header_row.OrgUnitType,
            'total_cases': header_row.TotalCases,
            'low_severity_count': header_row.LowSeverityCount,
            'medium_severity_count': header_row.MediumSeverityCount,
            'high_severity_count': header_row.HighSeverityCount,
            'clinical_domain_count': header_row.ClinicalDomainCount,
            'management_domain_count': header_row.ManagementDomainCount,
            'relational_domain_count': header_row.RelationalDomainCount,
            'is_compliant': bool(header_row.IsCompliant),
            'violated_rules': header_row.ViolatedRules,
            'explanation_status_id': header_row.ExplanationStatusID,
            'created_by_user_id': header_row.CreatedByUserID,
            'created_at': header_row.CreatedAt.isoformat() if header_row.CreatedAt else None,
            'period': header_row.SeasonPeriod,
            'orgunit_name': header_row.OrgUnitName
        }
        
        seasonal_report_id = header_row.SeasonalReportID
        
        # Fetch classification stats with names from lookup tables
        cursor.execute(
            """
            SELECT
                cs.ClassificationID,
                cs.TotalCount,
                cs.LowCount,
                cs.MediumCount,
                cs.HighCount,
                cs.PreventiveYesCount,
                cs.PreventiveNoCount,
                d.DomainName,
                c.CategoryName,
                sc.SubCategoryName,
                cl.Classification_AR as ClassificationName,
                cl.Classification_EN as ClassificationNameEN
            FROM dbo.APP_SeasonalOrgUnitReport_ClassificationStats cs
            LEFT JOIN dbo.APP_LOOKUP_CLASSIFICATION cl ON cs.ClassificationID = cl.ClassificationID
            LEFT JOIN dbo.APP_LOOKUP_SUBCATEGORY sc ON cl.SubCategoryID = sc.SubCategoryID
            LEFT JOIN dbo.APP_LOOKUP_CATEGORY c ON sc.CategoryID = c.CategoryID
            LEFT JOIN dbo.APP_LOOKUP_DOMAIN d ON c.DomainID = d.DomainID
            WHERE cs.SeasonalReportID = ?
            ORDER BY d.DomainID, c.CategoryID, sc.SubCategoryID, cl.ClassificationID
            """,
            (seasonal_report_id,)
        )
        
        classification_stats = []
        for row in cursor.fetchall():
            classification_stats.append({
                'classification_id': row.ClassificationID,
                'total_count': row.TotalCount,
                'low_count': row.LowCount,
                'medium_count': row.MediumCount,
                'high_count': row.HighCount,
                'preventive_yes_count': row.PreventiveYesCount,
                'preventive_no_count': row.PreventiveNoCount,
                'domain_name': row.DomainName,
                'category_name': row.CategoryName,
                'subcategory_name': row.SubCategoryName,
                'classification_name': row.ClassificationName,
                'classification_name_en': row.ClassificationNameEN
            })
        
        # Fetch policy snapshot
        cursor.execute(
            """
            SELECT
                LowSeverityLimit,
                MediumSeverityLimit,
                HighSeverityLimit,
                ClinicalDomainLimit,
                ManagementDomainLimit,
                RelationalDomainLimit,
                EnableLowSeverityRepetitionRule,
                EnableMediumSeverityRepetitionRule,
                EnableHighSeverityPercentageRule,
                EnableHighSeverityPercentageByDomainRule
            FROM dbo.APP_SeasonalOrgUnitReport_PolicySnapshot
            WHERE SeasonalReportID = ?
            """,
            (seasonal_report_id,)
        )
        
        policy_row = cursor.fetchone()
        policy_snapshot = None
        if policy_row:
            policy_snapshot = {
                'low_severity_limit': policy_row.LowSeverityLimit,
                'medium_severity_limit': policy_row.MediumSeverityLimit,
                'high_severity_limit': policy_row.HighSeverityLimit,
                'clinical_domain_limit': policy_row.ClinicalDomainLimit,
                'management_domain_limit': policy_row.ManagementDomainLimit,
                'relational_domain_limit': policy_row.RelationalDomainLimit,
                'enable_low_severity_repetition_rule': bool(policy_row.EnableLowSeverityRepetitionRule),
                'enable_medium_severity_repetition_rule': bool(policy_row.EnableMediumSeverityRepetitionRule),
                'enable_high_severity_percentage_rule': bool(policy_row.EnableHighSeverityPercentageRule),
                'enable_high_severity_percentage_by_domain_rule': bool(policy_row.EnableHighSeverityPercentageByDomainRule)
            }
        
        return {
            'header': header,
            'classification_stats': classification_stats,
            'policy_snapshot': policy_snapshot
        }
    
    except Exception as e:
        raise Exception(f"Failed to fetch seasonal report: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
