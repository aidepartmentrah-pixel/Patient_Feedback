import sys
from pathlib import Path
from datetime import date
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.database import get_connection

print("\nDirect APP DB Insert Test")
print("="*60)

conn = get_connection()
cursor = conn.cursor()

# Insert directly
cursor.execute("""
    INSERT INTO dbo.APP_IncidentCase (
        ComplaintText, FeedbackRecievedDate, IssuingOrgUnitID,
        DomainID, CategoryID, SubCategoryID, ClassificationID,
        SeverityID, StageID, HarmLevelID, CreatedByUserID, CaseStatusID,
        PatientName, BuildingID, ExplanationStatusID
    )
    OUTPUT INSERTED.IncidentRequestCaseID
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""",
    "Direct test complaint",
    date(2026, 1, 6),
    1,
    1, 6, 19, 132,
    1, 1, 1, 1, 3,  # Added CaseStatusID
    "Direct Test Patient",
    1, 1
)

record_id = cursor.fetchone()[0]
conn.commit()

print(f"✓ Inserted record ID: {record_id}")

# Verify
cursor.execute("""
    SELECT PatientName, BuildingID, ExplanationStatusID
    FROM APP_IncidentCase WHERE IncidentRequestCaseID = ?
""", record_id)

row = cursor.fetchone()
print(f"✓ PatientName: {row[0]}")
print(f"✓ BuildingID: {row[1]}")
print(f"✓ ExplanationStatusID: {row[2]}")

conn.close()
print("\n✓✓✓ DIRECT INSERT TEST PASSED ✓✓✓\n")
