"""
DB Layer for Seasonal Report Aggregation
Read-only queries for computing seasonal report statistics from raw case data.
"""

from typing import Dict, Any, List, Optional
from core.database import get_connection


def get_seasonal_classification_stats(
    season_id: int,
    orgunit_id: int,
    orgunit_type: int
) -> List[Dict[str, Any]]:
    """
    Aggregate per-classification statistics for a seasonal report.
    
    Args:
        season_id: Season identifier
        orgunit_id: Organizational unit identifier
        orgunit_type: Type of organizational unit
    
    Returns:
        List of dictionaries with classification-level aggregations:
        [
            {
                'classification_id': int,
                'total_count': int,
                'low_count': int,
                'medium_count': int,
                'high_count': int,
                'preventive_yes_count': int,
                'preventive_no_count': int
            },
            ...
        ]
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get season date range
        cursor.execute(
            """
            SELECT StartDate, EndDate
            FROM dbo.APP_LOOKUP_SEASON
            WHERE SeasonID = ?
            """,
            (season_id,)
        )
        
        season_row = cursor.fetchone()
        if not season_row:
            return []
        
        start_date = season_row.StartDate
        end_date = season_row.EndDate
        
        # TODO: OrgUnit filtering logic is ambiguous
        # Currently filtering by IssuingOrgUnitID only
        # May need to consider target departments or hierarchical org structure
        
        # Aggregate classification stats with severity breakdown
        cursor.execute(
            """
            SELECT
                ic.ClassificationID,
                COUNT(*) AS TotalCount,
                SUM(CASE WHEN ic.SeverityID = 1 THEN 1 ELSE 0 END) AS LowCount,
                SUM(CASE WHEN ic.SeverityID = 2 THEN 1 ELSE 0 END) AS MediumCount,
                SUM(CASE WHEN ic.SeverityID = 3 THEN 1 ELSE 0 END) AS HighCount,
                SUM(CASE 
                    WHEN ic.ClinicalRiskTypeID IN (2, 3) AND icf.IsPreventive = 1 
                    THEN 1 
                    ELSE 0 
                END) AS PreventiveYesCount,
                SUM(CASE 
                    WHEN ic.ClinicalRiskTypeID IN (2, 3) AND icf.IsPreventive = 0 
                    THEN 1 
                    ELSE 0 
                END) AS PreventiveNoCount
            FROM dbo.APP_IncidentCase ic
            LEFT JOIN dbo.APP_IncidentCaseFeedback icf
                ON ic.IncidentRequestCaseID = icf.IncidentRequestCaseID
            WHERE ic.FeedbackRecievedDate >= ?
              AND ic.FeedbackRecievedDate <= ?
              AND ic.IssuingOrgUnitID = ?
            GROUP BY ic.ClassificationID
            ORDER BY ic.ClassificationID
            """,
            (start_date, end_date, orgunit_id)
        )
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'classification_id': row.ClassificationID,
                'total_count': row.TotalCount,
                'low_count': row.LowCount,
                'medium_count': row.MediumCount,
                'high_count': row.HighCount,
                'preventive_yes_count': row.PreventiveYesCount,
                'preventive_no_count': row.PreventiveNoCount
            })
        
        return results
    
    except Exception as e:
        raise Exception(f"Failed to get classification stats: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_seasonal_domain_totals(
    season_id: int,
    orgunit_id: int,
    orgunit_type: int
) -> Dict[str, int]:
    """
    Compute domain-level totals for a seasonal report.
    
    Args:
        season_id: Season identifier
        orgunit_id: Organizational unit identifier
        orgunit_type: Type of organizational unit
    
    Returns:
        Dictionary with domain and severity aggregations:
        {
            'total_cases': int,
            'clinical_domain_count': int,
            'management_domain_count': int,
            'relational_domain_count': int,
            'low_severity_count': int,
            'medium_severity_count': int,
            'high_severity_count': int
        }
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get season date range
        cursor.execute(
            """
            SELECT StartDate, EndDate
            FROM dbo.APP_LOOKUP_SEASON
            WHERE SeasonID = ?
            """,
            (season_id,)
        )
        
        season_row = cursor.fetchone()
        if not season_row:
            return {
                'total_cases': 0,
                'clinical_domain_count': 0,
                'management_domain_count': 0,
                'relational_domain_count': 0,
                'low_severity_count': 0,
                'medium_severity_count': 0,
                'high_severity_count': 0
            }
        
        start_date = season_row.StartDate
        end_date = season_row.EndDate
        
        # TODO: Domain ID mapping is assumed based on context
        # DomainID 1 = Clinical, 2 = Management, 3 = Relational
        # Verify with actual APP_LOOKUP_DOMAIN table
        
        # TODO: Same orgunit filtering ambiguity as classification stats
        
        cursor.execute(
            """
            SELECT
                COUNT(*) AS TotalCases,
                SUM(CASE WHEN ic.DomainID = 1 THEN 1 ELSE 0 END) AS ClinicalDomainCount,
                SUM(CASE WHEN ic.DomainID = 2 THEN 1 ELSE 0 END) AS ManagementDomainCount,
                SUM(CASE WHEN ic.DomainID = 3 THEN 1 ELSE 0 END) AS RelationalDomainCount,
                SUM(CASE WHEN ic.SeverityID = 1 THEN 1 ELSE 0 END) AS LowSeverityCount,
                SUM(CASE WHEN ic.SeverityID = 2 THEN 1 ELSE 0 END) AS MediumSeverityCount,
                SUM(CASE WHEN ic.SeverityID = 3 THEN 1 ELSE 0 END) AS HighSeverityCount
            FROM dbo.APP_IncidentCase ic
            WHERE ic.FeedbackRecievedDate >= ?
              AND ic.FeedbackRecievedDate <= ?
              AND ic.IssuingOrgUnitID = ?
            """,
            (start_date, end_date, orgunit_id)
        )
        
        row = cursor.fetchone()
        if not row:
            return {
                'total_cases': 0,
                'clinical_domain_count': 0,
                'management_domain_count': 0,
                'relational_domain_count': 0,
                'low_severity_count': 0,
                'medium_severity_count': 0,
                'high_severity_count': 0
            }
        
        return {
            'total_cases': row.TotalCases or 0,
            'clinical_domain_count': row.ClinicalDomainCount or 0,
            'management_domain_count': row.ManagementDomainCount or 0,
            'relational_domain_count': row.RelationalDomainCount or 0,
            'low_severity_count': row.LowSeverityCount or 0,
            'medium_severity_count': row.MediumSeverityCount or 0,
            'high_severity_count': row.HighSeverityCount or 0
        }
    
    except Exception as e:
        raise Exception(f"Failed to get domain totals: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_orgunit_policy_row(
    orgunit_id: int,
    orgunit_type: int
) -> Optional[Dict[str, Any]]:
    """
    Fetch active policy configuration for an organizational unit.
    Used to create policy snapshot at report generation time.
    
    Args:
        orgunit_id: Organizational unit identifier
        orgunit_type: Type of organizational unit
    
    Returns:
        Dictionary with policy fields from APP_OrgUnitPolicy, or None if not found:
        {
            'OrgUnitID': int,
            'OrgUnitType': int,
            'MaxAllowedCases': int,
            'MaxClinicalDomain': int,
            'MaxManagementDomain': int,
            'MaxRelationalDomain': int,
            'RequireExplanationAboveThreshold': bool,
            'EscalationEnabled': bool,
            'EscalationThresholdPercentage': int
        }
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # TODO: Policy table structure assumed based on context
        # May have additional fields or different column names
        # May have IsActive flag or multiple policies per orgunit
        
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
            FROM dbo.APP_OrgUnitPolicy
            WHERE OrgUnitID = ?
              AND OrgUnitType = ?
            """,
            (orgunit_id, orgunit_type)
        )
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return {
            'OrgUnitID': row.OrgUnitID,
            'OrgUnitType': row.OrgUnitType,
            'MaxAllowedCases': row.MaxAllowedCases,
            'MaxClinicalDomain': row.MaxClinicalDomain,
            'MaxManagementDomain': row.MaxManagementDomain,
            'MaxRelationalDomain': row.MaxRelationalDomain,
            'RequireExplanationAboveThreshold': row.RequireExplanationAboveThreshold,
            'EscalationEnabled': row.EscalationEnabled,
            'EscalationThresholdPercentage': row.EscalationThresholdPercentage
        }
    
    except Exception as e:
        raise Exception(f"Failed to get policy row: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
