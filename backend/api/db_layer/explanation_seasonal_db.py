"""
Database Layer: Seasonal Report Explanations
=============================================
Handles updating ExplanationText field in APP_SeasonalOrgUnitReport table.

For seasonal reports, explanations are added to the ExplanationText field
with status tracking via ExplanationStatusID.
"""

import json
from typing import Dict, Any, Optional
from core.database import get_connection
from datetime import datetime


def update_seasonal_explanation(
    seasonal_report_id: int,
    explanation_text: str,
    user_id: int
) -> Dict[str, Any]:
    """
    Update ExplanationText field in seasonal report.
    
    Args:
        seasonal_report_id: SeasonalReportID from APP_SeasonalOrgUnitReport
        explanation_text: Explanation text for the seasonal report
        user_id: ID of user submitting explanation
    
    Returns:
        Success/error dictionary
    
    Note: Seasonal reports don't follow the same FSM as case explanations.
    They have their own ExplanationStatusID field.
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Validate seasonal report exists
        cursor.execute(
            """
            SELECT 
                SeasonalReportID,
                ExplanationStatusID,
                ExplanationText,
                OrgUnitID,
                SeasonID
            FROM dbo.APP_SeasonalOrgUnitReport
            WHERE SeasonalReportID = ?
            """,
            (seasonal_report_id,)
        )
        row = cursor.fetchone()
        
        if not row:
            return {
                "success": False,
                "error": "REPORT_NOT_FOUND",
                "message": f"Seasonal report ID {seasonal_report_id} not found"
            }
        
        current_explanation_status = row.ExplanationStatusID
        existing_explanation = row.ExplanationText
        org_unit_id = row.OrgUnitID
        season_id = row.SeasonID
        
        # Check if already explained (optional - can allow updates)
        if current_explanation_status == 2:
            print(f"[WARNING] Seasonal report {seasonal_report_id} already has explanation. Overwriting.")
        
        # Update explanation
        cursor.execute(
            """
            UPDATE dbo.APP_SeasonalOrgUnitReport
            SET ExplanationText = ?,
                ExplanationStatusID = 2,  -- Responded
                ExplanationSubmittedAt = GETDATE()
            WHERE SeasonalReportID = ?
            """,
            (explanation_text, seasonal_report_id)
        )
        
        conn.commit()
        
        return {
            "success": True,
            "message": "Seasonal report explanation submitted successfully",
            "seasonal_report_id": seasonal_report_id,
            "org_unit_id": org_unit_id,
            "season_id": season_id,
            "updated_field": "ExplanationText",
            "previous_status": current_explanation_status,
            "new_status": 2
        }
    
    except Exception as e:
        if conn:
            conn.rollback()
        return {
            "success": False,
            "error": "DATABASE_ERROR",
            "message": f"Failed to update seasonal explanation: {str(e)}"
        }
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_seasonal_explanation(seasonal_report_id: int) -> Dict[str, Any]:
    """
    Retrieve explanation details for a seasonal report.
    
    Returns:
        Dictionary with report details or error
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT 
                sr.SeasonalReportID,
                sr.SeasonID,
                sr.OrgUnitID,
                sr.OrgUnitType,
                sr.ExplanationText,
                sr.ExplanationStatusID,
                sr.ExplanationSubmittedAt,
                sr.TotalCases,
                sr.LowSeverityCount,
                sr.MediumSeverityCount,
                sr.HighSeverityCount,
                sr.IsCompliant,
                sr.ViolatedRules,
                s.SeasonName,
                s.StartDate,
                s.EndDate
            FROM dbo.APP_SeasonalOrgUnitReport sr
            LEFT JOIN dbo.Season s ON sr.SeasonID = s.UniqueID
            WHERE sr.SeasonalReportID = ?
            """,
            (seasonal_report_id,)
        )
        
        row = cursor.fetchone()
        
        if not row:
            return {
                "success": False,
                "error": "REPORT_NOT_FOUND",
                "message": f"Seasonal report ID {seasonal_report_id} not found"
            }
        
        # Parse violated_rules from JSON string to list
        violated_rules = []
        if row.ViolatedRules:
            try:
                violated_rules = json.loads(row.ViolatedRules)
            except (json.JSONDecodeError, TypeError):
                # If parsing fails, keep as empty list
                violated_rules = []
        
        return {
            "success": True,
            "seasonal_report_id": row.SeasonalReportID,
            "season_id": row.SeasonID,
            "season_name": row.SeasonName,
            "season_start_date": row.StartDate.isoformat() if row.StartDate else None,
            "season_end_date": row.EndDate.isoformat() if row.EndDate else None,
            "org_unit_id": row.OrgUnitID,
            "org_unit_type": row.OrgUnitType,
            "explanation_text": row.ExplanationText,
            "explanation_status_id": row.ExplanationStatusID,
            "explanation_submitted_at": row.ExplanationSubmittedAt.isoformat() if row.ExplanationSubmittedAt else None,
            "total_cases": row.TotalCases,
            "low_severity_count": row.LowSeverityCount,
            "medium_severity_count": row.MediumSeverityCount,
            "high_severity_count": row.HighSeverityCount,
            "is_compliant": row.IsCompliant,
            "violated_rules": violated_rules
        }
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_seasonal_reports_needing_explanation(
    org_unit_id: Optional[int] = None,
    season_id: Optional[int] = None,
    non_compliant_only: bool = False
) -> Dict[str, Any]:
    """
    Get all seasonal reports that need explanations.
    
    Args:
        org_unit_id: Filter by organization unit
        season_id: Filter by season
        non_compliant_only: Only return non-compliant reports
    
    Returns:
        Dictionary with success status and list of reports
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Build dynamic query
        where_clauses = ["(sr.ExplanationStatusID IS NULL OR sr.ExplanationStatusID = 1)"]
        params = []
        
        if org_unit_id:
            where_clauses.append("sr.OrgUnitID = ?")
            params.append(org_unit_id)
        
        if season_id:
            where_clauses.append("sr.SeasonID = ?")
            params.append(season_id)
        
        if non_compliant_only:
            where_clauses.append("sr.IsCompliant = 0")
        
        where_clause = " AND ".join(where_clauses)
        
        query = f"""
            SELECT 
                sr.SeasonalReportID,
                sr.SeasonID,
                sr.OrgUnitID,
                sr.OrgUnitType,
                sr.TotalCases,
                sr.LowSeverityCount,
                sr.MediumSeverityCount,
                sr.HighSeverityCount,
                sr.IsCompliant,
                sr.ViolatedRules,
                sr.ExplanationStatusID,
                s.SeasonName,
                s.StartDate,
                s.EndDate,
                au.Name as OrgUnitName,
                au.NameAr as OrgUnitNameAr
            FROM dbo.APP_SeasonalOrgUnitReport sr
            LEFT JOIN dbo.Season s ON sr.SeasonID = s.UniqueID
            LEFT JOIN dbo.AdminsrationUnit au ON sr.OrgUnitID = au.UniqueID
            WHERE {where_clause}
            ORDER BY s.StartDate DESC, sr.OrgUnitID
        """
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        reports = []
        for row in rows:
            # Parse violated_rules from JSON string to list
            violated_rules = []
            if row.ViolatedRules:
                try:
                    violated_rules = json.loads(row.ViolatedRules)
                except (json.JSONDecodeError, TypeError):
                    # If parsing fails, keep as empty list
                    violated_rules = []
            
            # Safely access org unit names (might be NULL if unit not found)
            org_unit_name_en = getattr(row, 'OrgUnitName', None)
            org_unit_name_ar = getattr(row, 'OrgUnitNameAr', None)
            
            # Build display string for org unit type and name
            org_unit_type_display = row.OrgUnitType or "Unit"
            
            reports.append({
                "seasonal_report_id": row.SeasonalReportID,
                "season_id": row.SeasonID,
                "season_name": row.SeasonName,
                "season_start_date": row.StartDate.isoformat() if row.StartDate else None,
                "season_end_date": row.EndDate.isoformat() if row.EndDate else None,
                "org_unit_id": row.OrgUnitID,
                "org_unit_type": row.OrgUnitType,
                "org_unit_name": org_unit_name_en,  # Keep for backward compatibility
                "org_unit_name_en": org_unit_name_en,
                "org_unit_name_ar": org_unit_name_ar,
                "total_cases": row.TotalCases,
                "low_severity_count": row.LowSeverityCount,
                "medium_severity_count": row.MediumSeverityCount,
                "high_severity_count": row.HighSeverityCount,
                "is_compliant": bool(row.IsCompliant),
                "violated_rules": violated_rules,
                "explanation_status_id": row.ExplanationStatusID or 1
            })
        
        return {
            "success": True,
            "data": reports,
            "statistics": {
                "total_count": len(reports),
                "non_compliant_count": sum(1 for r in reports if not r["is_compliant"])
            }
        }
    
    except Exception as e:
        import traceback
        print(f"\n{'='*80}")
        print("[ERROR] get_seasonal_reports_needing_explanation failed")
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        print(f"{'='*80}\n")
        return {
            "success": False,
            "error": str(e),
            "data": [],
            "statistics": {"total_count": 0, "non_compliant_count": 0}
        }
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
