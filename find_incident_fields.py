"""
Find required fields for incident creation
"""
import sys
sys.path.insert(0, 'backend')

from core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

try:
    # Get one existing incident as template
    cursor.execute("""
        SELECT TOP 1 
            ComplaintText, ImmediateAction, TakenAction,
            FeedbackRecievedDate, PatientName, IssuingOrgUnitID,
            CreatedByUserID, isINPatient, ClinicalRiskTypeID,
            FeedbackIntentTypeID, BuildingID, DomainID,
            CategoryID, SubCategoryID, ClassificationID,
            SeverityID, StageID, HarmLevelID, CaseStatusID,
            SourceID, ExplanationStatusID, RequiresExplanation
        FROM APP_IncidentCase
        WHERE IncidentRequestCaseID IS NOT NULL
        ORDER BY IncidentRequestCaseID DESC
    """)
    
    row = cursor.fetchone()
    if row:
        print("Sample incident fields:")
        print(f"  ClinicalRiskTypeID: {row.ClinicalRiskTypeID}")
        print(f"  FeedbackIntentTypeID: {row.FeedbackIntentTypeID}")
        print(f"  BuildingID: {row.BuildingID}")
        print(f"  DomainID: {row.DomainID}")
        print(f"  CategoryID: {row.CategoryID}")
        print(f"  SubCategoryID: {row.SubCategoryID}")
        print(f"  ClassificationID: {row.ClassificationID}")
        print(f"  SeverityID: {row.SeverityID}")
        print(f"  StageID: {row.StageID}")
        print(f"  HarmLevelID: {row.HarmLevelID}")
        print(f"  SourceID: {row.SourceID}")
    
    # Check constraints
    cursor.execute("""
        SELECT COLUMN_NAME, IS_NULLABLE, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = 'APP_IncidentCase'
        AND IS_NULLABLE = 'NO'
        AND COLUMN_NAME NOT IN ('IncidentRequestCaseID')
        ORDER BY ORDINAL_POSITION
    """)
    
    print("\nRequired columns (NOT NULL):")
    for col in cursor.fetchall():
        print(f"  {col.COLUMN_NAME} ({col.DATA_TYPE})")

finally:
    cursor.close()
    conn.close()
