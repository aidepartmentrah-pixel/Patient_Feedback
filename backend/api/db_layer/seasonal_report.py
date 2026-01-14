"""
DB Layer for Seasonal Report Persistence
Handles CRUD operations for materialized seasonal reports and related tables.
"""

from typing import Dict, Any, List, Optional
from core.database import get_connection


def resolve_season_id_from_year_trimester(year: int, trimester: str) -> Optional[int]:
    """
    Resolve season_id (UniqueID) from year and trimester.
    
    Trimester mapping:
    - Trim1: Jan-Apr (months 1-4)
    - Trim2: May-Aug (months 5-8)
    - Trim3: Sep-Dec (months 9-12)
    
    Args:
        year: Calendar year (e.g., 2025)
        trimester: Trimester identifier (Trim1, Trim2, Trim3)
    
    Returns:
        Season UniqueID if found uniquely, None if not found
    
    Raises:
        ValueError: If multiple seasons match or trimester format invalid
    """
    conn = None
    cursor = None
    
    # Map trimester to month ranges
    trimester_ranges = {
        "Trim1": (1, 4),    # Jan-Apr
        "Trim2": (5, 8),    # May-Aug
        "Trim3": (9, 12)    # Sep-Dec
    }
    
    if trimester not in trimester_ranges:
        raise ValueError(f"Invalid trimester: {trimester}. Must be Trim1, Trim2, or Trim3")
    
    start_month, end_month = trimester_ranges[trimester]
    
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
            return None
        
        if len(rows) > 1:
            raise ValueError(f"Ambiguous season: Multiple seasons found for year={year}, trimester={trimester}")
        
        return rows[0].UniqueID
    
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
                    TotalCount,
                    LowCount,
                    MediumCount,
                    HighCount,
                    PreventiveYesCount,
                    PreventiveNoCount
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    seasonal_report_id,
                    stat['classification_id'],
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
                OrgUnitID,
                OrgUnitType,
                MaxAllowedCases,
                MaxClinicalDomain,
                MaxManagementDomain,
                MaxRelationalDomain,
                RequireExplanationAboveThreshold,
                EscalationEnabled,
                EscalationThresholdPercentage
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                seasonal_report_id,
                policy_row.get('OrgUnitID'),
                policy_row.get('OrgUnitType'),
                policy_row.get('MaxAllowedCases'),
                policy_row.get('MaxClinicalDomain'),
                policy_row.get('MaxManagementDomain'),
                policy_row.get('MaxRelationalDomain'),
                policy_row.get('RequireExplanationAboveThreshold'),
                policy_row.get('EscalationEnabled'),
                policy_row.get('EscalationThresholdPercentage')
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
        
        # Fetch header
        cursor.execute(
            """
            SELECT
                SeasonalReportID,
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
            FROM dbo.APP_SeasonalOrgUnitReport
            WHERE SeasonID = ? AND OrgUnitID = ? AND OrgUnitType = ?
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
            'created_at': header_row.CreatedAt.isoformat() if header_row.CreatedAt else None
        }
        
        seasonal_report_id = header_row.SeasonalReportID
        
        # Fetch classification stats
        cursor.execute(
            """
            SELECT
                ClassificationID,
                TotalCount,
                LowCount,
                MediumCount,
                HighCount,
                PreventiveYesCount,
                PreventiveNoCount
            FROM dbo.APP_SeasonalOrgUnitReport_ClassificationStats
            WHERE SeasonalReportID = ?
            ORDER BY ClassificationID
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
                'preventive_no_count': row.PreventiveNoCount
            })
        
        # Fetch policy snapshot
        cursor.execute(
            """
            SELECT
                OrgUnitID,
                OrgUnitType,
                MaxAllowedCases,
                MaxClinicalDomain,
                MaxManagementDomain,
                MaxRelationalDomain,
                RequireExplanationAboveThreshold,
                EscalationEnabled,
                EscalationThresholdPercentage
            FROM dbo.APP_SeasonalOrgUnitReport_PolicySnapshot
            WHERE SeasonalReportID = ?
            """,
            (seasonal_report_id,)
        )
        
        policy_row = cursor.fetchone()
        policy_snapshot = None
        if policy_row:
            policy_snapshot = {
                'orgunit_id': policy_row.OrgUnitID,
                'orgunit_type': policy_row.OrgUnitType,
                'max_allowed_cases': policy_row.MaxAllowedCases,
                'max_clinical_domain': policy_row.MaxClinicalDomain,
                'max_management_domain': policy_row.MaxManagementDomain,
                'max_relational_domain': policy_row.MaxRelationalDomain,
                'require_explanation_above_threshold': bool(policy_row.RequireExplanationAboveThreshold),
                'escalation_enabled': bool(policy_row.EscalationEnabled),
                'escalation_threshold_percentage': policy_row.EscalationThresholdPercentage
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
