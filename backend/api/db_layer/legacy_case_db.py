"""
PHASE K — Legacy Case DB Layer

Database layer functions for reading legacy case data.

This module provides read-only operations for the APP_IncidentCase table
to support migration page listing and detail views.

NO WRITES - Read-only legacy data access only.
"""

from core.database import get_connection


def list_legacy_cases_paged(page: int = 1, page_size: int = 50) -> dict:
    """
    Retrieve paginated list of legacy cases.
    
    Args:
        page: Page number (1-indexed)
        page_size: Number of records per page
        
    Returns:
        dict: {
            "cases": [
                {
                    "legacy_case_id": int,
                    "complaint_text": str,
                    "patient_name": str,
                    "feedback_received_date": datetime,
                    "case_status_id": int,
                    "created_at": datetime,
                    "migrated": bool
                },
                ...
            ],
            "total": int
        }
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Calculate offset
        offset = (page - 1) * page_size
        
        # Query with migration status check
        cursor.execute("""
            SELECT 
                ic.IncidentRequestCaseID,
                ic.ComplaintText,
                ic.PatientName,
                ic.FeedbackRecievedDate,
                ic.CaseStatusID,
                ic.CreatedAt,
                CASE WHEN dm.MapID IS NOT NULL THEN 1 ELSE 0 END AS Migrated
            FROM dbo.APP_IncidentCase ic
            LEFT JOIN dbo.APP_DataMigration_Map dm ON ic.IncidentRequestCaseID = dm.legacy_case_id
            ORDER BY ic.IncidentRequestCaseID DESC
            OFFSET ? ROWS
            FETCH NEXT ? ROWS ONLY
        """, offset, page_size)
        
        rows = cursor.fetchall()
        
        # Query total count
        cursor.execute("SELECT COUNT(*) FROM dbo.APP_IncidentCase")
        total = cursor.fetchone()[0]
        
        cases = []
        for row in rows:
            cases.append({
                "legacy_case_id": row[0],
                "complaint_text": row[1],
                "patient_name": row[2],
                "feedback_received_date": row[3],
                "case_status_id": row[4],
                "created_at": row[5],
                "migrated": bool(row[6])
            })
        
        return {
            "cases": cases,
            "total": total
        }
        
    except Exception as e:
        raise Exception("Failed to list legacy cases: " + str(e))
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_legacy_case_by_id(legacy_case_id: int) -> dict:
    """
    Retrieve detailed legacy case record.
    
    Args:
        legacy_case_id: Legacy case ID from APP_IncidentCase
        
    Returns:
        dict: Full case record with all fields
        None: If case not found
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                ic.IncidentRequestCaseID,
                ic.ComplaintText,
                ic.ImmediateAction,
                ic.TakenAction,
                ic.FeedbackRecievedDate,
                ic.PatientName,
                ic.IssuingOrgUnitID,
                ic.CreatedAt,
                ic.CreatedByUserID,
                ic.isINPatient,
                ic.ClinicalRiskTypeID,
                ic.FeedbackIntentTypeID,
                ic.BuildingID,
                ic.DomainID,
                ic.CategoryID,
                ic.SubCategoryID,
                ic.ClassificationID,
                ic.SeverityID,
                ic.StageID,
                ic.HarmLevelID,
                ic.CaseStatusID,
                ic.SourceID,
                ic.ExplanationStatusID,
                ic.RequiresExplanation,
                CASE WHEN dm.MapID IS NOT NULL THEN 1 ELSE 0 END AS Migrated,
                dm.new_case_id AS MigratedCaseID
            FROM dbo.APP_IncidentCase ic
            LEFT JOIN dbo.APP_DataMigration_Map dm ON ic.IncidentRequestCaseID = dm.legacy_case_id
            WHERE ic.IncidentRequestCaseID = ?
        """, legacy_case_id)
        
        row = cursor.fetchone()
        
        if not row:
            return None
        
        return {
            "legacy_case_id": row[0],
            "complaint_text": row[1],
            "immediate_action": row[2],
            "taken_action": row[3],
            "feedback_received_date": row[4],
            "patient_name": row[5],
            "issuing_org_unit_id": row[6],
            "created_at": row[7],
            "created_by_user_id": row[8],
            "is_inpatient": bool(row[9]) if row[9] is not None else None,
            "clinical_risk_type_id": row[10],
            "feedback_intent_type_id": row[11],
            "building_id": row[12],
            "domain_id": row[13],
            "category_id": row[14],
            "subcategory_id": row[15],
            "classification_id": row[16],
            "severity_id": row[17],
            "stage_id": row[18],
            "harm_id": row[19],
            "case_status_id": row[20],
            "source_id": row[21],
            "explanation_status_id": row[22],
            "requires_explanation": row[23],
            "migrated": bool(row[24]),
            "migrated_case_id": row[25]
        }
        
    except Exception as e:
        raise Exception("Failed to get legacy case: " + str(e))
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
