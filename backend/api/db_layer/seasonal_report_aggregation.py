"""
DB Layer for Seasonal Report Aggregation
Read-only queries for computing seasonal report statistics from raw case data.
"""

from typing import Dict, Any, List, Optional
from core.database import get_connection
from backend.api.db_layer.reports_db import build_org_filter_condition


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
            FROM dbo.Season
            WHERE UniqueID = ?
            """,
            (season_id,)
        )
        
        season_row = cursor.fetchone()
        if not season_row:
            return []
        
        start_date = season_row.StartDate
        end_date = season_row.EndDate
        
        # Hospital-level (orgunit_id=1, orgunit_type=0): Aggregate ALL incidents
        # Specific unit: Filter by TARGET DEPARTMENTS with tree expansion (same as monthly reporting)
        # This ensures Administration reports include ALL complaints where target dept belongs to that Administration
        if orgunit_id == 1 and orgunit_type == 0:
            # Hospital-level: No orgunit filter
            where_clause = f"""
            WHERE ic.FeedbackRecievedDate >= ?
              AND ic.FeedbackRecievedDate <= ?
            """
            params = (start_date, end_date)
        else:
            # Specific unit: Apply tree-aware target department filter
            # Map orgunit_type to filter parameters:
            # orgunit_type 1 (Administration) -> idara_id
            # orgunit_type 2 (Department) -> dayra_id  
            # orgunit_type 3 (Section) -> qism_id
            if orgunit_type == 1:  # Administration
                idara_id = orgunit_id
                dayra_id = None
                qism_id = None
            elif orgunit_type == 2:  # Department
                idara_id = None
                dayra_id = orgunit_id
                qism_id = None
            elif orgunit_type == 3:  # Section
                idara_id = None
                dayra_id = None
                qism_id = orgunit_id
            else:
                idara_id = None
                dayra_id = None
                qism_id = None
            
            # Build tree-aware org filter (same mechanism as monthly reporting)
            org_filter = build_org_filter_condition(None, idara_id, dayra_id, qism_id)
            
            where_clause = f"""
            WHERE ic.FeedbackRecievedDate >= ?
              AND ic.FeedbackRecievedDate <= ?
              AND {org_filter}
            """
            params = (start_date, end_date)
        
        # Aggregate classification stats with severity breakdown
        # Note: IsPreventive calculated based on presence of preventive measures
        # Group by ClassificationID, DomainID, CategoryID, AND SubCategoryID (required by table schema)
        query = f"""
            SELECT
                ic.ClassificationID,
                ic.DomainID,
                ic.CategoryID,
                ic.SubCategoryID,
                COUNT(*) AS TotalCount,
                SUM(CASE WHEN ic.SeverityID = 1 THEN 1 ELSE 0 END) AS LowCount,
                SUM(CASE WHEN ic.SeverityID = 2 THEN 1 ELSE 0 END) AS MediumCount,
                SUM(CASE WHEN ic.SeverityID = 3 THEN 1 ELSE 0 END) AS HighCount,
                SUM(CASE 
                    WHEN ic.ClinicalRiskTypeID IN (2, 3) AND (
                        icf.Preventive_MonthlyMeetings = 1 OR
                        icf.Preventive_TrainingPrograms = 1 OR
                        icf.Preventive_IncreaseStaff = 1 OR
                        icf.Preventive_MMCommitteeActions = 1 OR
                        icf.Preventive_Other = 1
                    )
                    THEN 1 
                    ELSE 0 
                END) AS PreventiveYesCount,
                SUM(CASE 
                    WHEN ic.ClinicalRiskTypeID IN (2, 3) AND icf.IncidentRequestCaseID IS NOT NULL AND (
                        ISNULL(icf.Preventive_MonthlyMeetings, 0) = 0 AND
                        ISNULL(icf.Preventive_TrainingPrograms, 0) = 0 AND
                        ISNULL(icf.Preventive_IncreaseStaff, 0) = 0 AND
                        ISNULL(icf.Preventive_MMCommitteeActions, 0) = 0 AND
                        ISNULL(icf.Preventive_Other, 0) = 0
                    )
                    THEN 1 
                    ELSE 0 
                END) AS PreventiveNoCount
            FROM dbo.APP_IncidentCase ic
            LEFT JOIN dbo.APP_IncidentCaseFeedback icf
                ON ic.IncidentRequestCaseID = icf.IncidentRequestCaseID
            INNER JOIN dbo.APP_IncidentCaseTargetDepartment td
                ON ic.IncidentRequestCaseID = td.IncidentRequestCaseID
            {where_clause}
            GROUP BY ic.ClassificationID, ic.DomainID, ic.CategoryID, ic.SubCategoryID
            ORDER BY ic.ClassificationID, ic.DomainID, ic.CategoryID, ic.SubCategoryID
        """
        
        cursor.execute(query, params)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'classification_id': row.ClassificationID,
                'domain_id': row.DomainID,
                'category_id': row.CategoryID,
                'subcategory_id': row.SubCategoryID,
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
            FROM dbo.Season
            WHERE UniqueID = ?
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
        
        # Hospital-level (orgunit_id=1, orgunit_type=0): Aggregate ALL incidents
        # Specific unit: Filter by TARGET DEPARTMENTS with tree expansion (same as monthly reporting)
        if orgunit_id == 1 and orgunit_type == 0:
            # Hospital-level: No orgunit filter
            where_clause = f"""
            WHERE ic.FeedbackRecievedDate >= ?
              AND ic.FeedbackRecievedDate <= ?
            """
            params = (start_date, end_date)
        else:
            # Specific unit: Apply tree-aware target department filter
            # Map orgunit_type to filter parameters
            if orgunit_type == 1:  # Administration
                idara_id = orgunit_id
                dayra_id = None
                qism_id = None
            elif orgunit_type == 2:  # Department
                idara_id = None
                dayra_id = orgunit_id
                qism_id = None
            elif orgunit_type == 3:  # Section
                idara_id = None
                dayra_id = None
                qism_id = orgunit_id
            else:
                idara_id = None
                dayra_id = None
                qism_id = None
            
            # Build tree-aware org filter (same mechanism as monthly reporting)
            org_filter = build_org_filter_condition(None, idara_id, dayra_id, qism_id)
            
            where_clause = f"""
            WHERE ic.FeedbackRecievedDate >= ?
              AND ic.FeedbackRecievedDate <= ?
              AND {org_filter}
            """
            params = (start_date, end_date)
        
        # Count DISTINCT complaints to avoid counting same complaint multiple times
        # (a complaint may have multiple target departments)
        # FIXED: Use COUNT(DISTINCT ...) for ALL aggregations to ensure consistency
        # Total cases = sum of domain counts (one case belongs to exactly one domain)
        query = f"""
            SELECT
                COUNT(DISTINCT ic.IncidentRequestCaseID) AS TotalCases,
                COUNT(DISTINCT CASE WHEN ic.DomainID = 1 THEN ic.IncidentRequestCaseID ELSE NULL END) AS ClinicalDomainCount,
                COUNT(DISTINCT CASE WHEN ic.DomainID = 2 THEN ic.IncidentRequestCaseID ELSE NULL END) AS ManagementDomainCount,
                COUNT(DISTINCT CASE WHEN ic.DomainID = 3 THEN ic.IncidentRequestCaseID ELSE NULL END) AS RelationalDomainCount,
                COUNT(DISTINCT CASE WHEN ic.SeverityID = 1 THEN ic.IncidentRequestCaseID ELSE NULL END) AS LowSeverityCount,
                COUNT(DISTINCT CASE WHEN ic.SeverityID = 2 THEN ic.IncidentRequestCaseID ELSE NULL END) AS MediumSeverityCount,
                COUNT(DISTINCT CASE WHEN ic.SeverityID = 3 THEN ic.IncidentRequestCaseID ELSE NULL END) AS HighSeverityCount
            FROM dbo.APP_IncidentCase ic
            INNER JOIN dbo.APP_IncidentCaseTargetDepartment td
                ON ic.IncidentRequestCaseID = td.IncidentRequestCaseID
            {where_clause}
        """
        
        cursor.execute(query, params)
        
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
        
        # Extract values
        total_cases = row.TotalCases or 0
        clinical_count = row.ClinicalDomainCount or 0
        management_count = row.ManagementDomainCount or 0
        relational_count = row.RelationalDomainCount or 0
        low_severity = row.LowSeverityCount or 0
        medium_severity = row.MediumSeverityCount or 0
        high_severity = row.HighSeverityCount or 0
        
        # SANITY CHECK: Verify domain counts sum equals total
        domain_sum = clinical_count + management_count + relational_count
        severity_sum = low_severity + medium_severity + high_severity
        
        print(f"\n{'='*80}")
        print(f"[SANITY CHECK] Domain Totals Aggregation")
        print(f"  Total Cases: {total_cases}")
        print(f"  Domain Sum: {domain_sum} (Clinical={clinical_count}, Management={management_count}, Relational={relational_count})")
        print(f"  Severity Sum: {severity_sum} (Low={low_severity}, Medium={medium_severity}, High={high_severity})")
        
        if domain_sum != total_cases:
            print(f"  [WARNING] Domain sum ({domain_sum}) != Total cases ({total_cases})")
        else:
            print(f"  [PASS] Domain sum matches total cases")

        if severity_sum != total_cases:
            print(f"  [WARNING] Severity sum ({severity_sum}) != Total cases ({total_cases})")
        else:
            print(f"  [PASS] Severity sum matches total cases")
        print(f"{'='*80}\n")
        
        return {
            'total_cases': total_cases,
            'clinical_domain_count': clinical_count,
            'management_domain_count': management_count,
            'relational_domain_count': relational_count,
            'low_severity_count': low_severity,
            'medium_severity_count': medium_severity,
            'high_severity_count': high_severity
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
            FROM dbo.APP_OrgUnitPolicy
            WHERE OrgUnitID = ?
              AND OrgUnitType = ?
              AND IsActive = 1
            """,
            (orgunit_id, orgunit_type)
        )
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return {
            'OrgUnitID': row.OrgUnitID,
            'OrgUnitType': row.OrgUnitType,
            'LowSeverityLimit': row.LowSeverityLimit,
            'MediumSeverityLimit': row.MediumSeverityLimit,
            'HighSeverityLimit': row.HighSeverityLimit,
            'ClinicalDomainLimit': row.ClinicalDomainLimit,
            'ManagementDomainLimit': row.ManagementDomainLimit,
            'RelationalDomainLimit': row.RelationalDomainLimit,
            'EnableLowSeverityRepetitionRule': row.EnableLowSeverityRepetitionRule,
            'EnableMediumSeverityRepetitionRule': row.EnableMediumSeverityRepetitionRule,
            'EnableHighSeverityPercentageRule': row.EnableHighSeverityPercentageRule,
            'EnableHighSeverityPercentageByDomainRule': row.EnableHighSeverityPercentageByDomainRule
        }
    
    except Exception as e:
        raise Exception(f"Failed to get policy row: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
