"""Debug why seasonal report is empty"""
import sys
sys.path.insert(0, "c:\\Users\\IT\\Documents\\GitHub Repository\\Patient_Feedback\\backend")

from core.database import get_connection
from api.db_layer.seasonal_report import resolve_season_id_from_year_trimester

print("=" * 80)
print("DEBUG: Why Seasonal Report is Empty")
print("=" * 80)

conn = get_connection()
cursor = conn.cursor()

# Get season info for Q1 2026
season_id = resolve_season_id_from_year_trimester(2026, "Q1")
print(f"\n✅ Q1-2026 Season ID: {season_id}")

# Get season date range
cursor.execute("SELECT SeasonName, StartDate, EndDate FROM Season WHERE UniqueID = ?", (season_id,))
season = cursor.fetchone()
print(f"   Season: {season.SeasonName}")
print(f"   Dates: {season.StartDate} to {season.EndDate}")

# Check incident cases in this date range
print(f"\n🔍 Checking incident cases in date range...")

cursor.execute("""
    SELECT 
        COUNT(*) as TotalCases,
        MIN(FeedbackRecievedDate) as EarliestDate,
        MAX(FeedbackRecievedDate) as LatestDate,
        COUNT(DISTINCT IssuingOrgUnitID) as UniqueOrgUnits
    FROM APP_IncidentCase
    WHERE FeedbackRecievedDate >= ? AND FeedbackRecievedDate <= ?
""", (season.StartDate, season.EndDate))

result = cursor.fetchone()
print(f"   Total cases in date range: {result.TotalCases}")
print(f"   Date range: {result.EarliestDate} to {result.LatestDate}")
print(f"   Unique org units: {result.UniqueOrgUnits}")

# Check what org units exist
print(f"\n🔍 Checking org units in the data...")
cursor.execute("""
    SELECT DISTINCT IssuingOrgUnitID, COUNT(*) as CaseCount
    FROM APP_IncidentCase
    WHERE FeedbackRecievedDate >= ? AND FeedbackRecievedDate <= ?
    GROUP BY IssuingOrgUnitID
    ORDER BY CaseCount DESC
""", (season.StartDate, season.EndDate))

print("   Org Unit ID | Case Count")
print("   " + "-" * 30)
for row in cursor.fetchall():
    print(f"   {row.IssuingOrgUnitID:11} | {row.CaseCount}")

# Check the specific query used in aggregation
print(f"\n🔍 Testing aggregation query for OrgUnit 1...")
cursor.execute("""
    SELECT COUNT(*) AS TotalCases
    FROM APP_IncidentCase
    WHERE FeedbackRecievedDate >= ?
      AND FeedbackRecievedDate <= ?
      AND IssuingOrgUnitID = 1
""", (season.StartDate, season.EndDate))

result = cursor.fetchone()
print(f"   Cases for OrgUnit 1: {result.TotalCases}")

# Check ALL incident cases
print(f"\n🔍 Checking ALL incident cases (no filters)...")
cursor.execute("SELECT COUNT(*) as Total, MIN(FeedbackRecievedDate) as Min, MAX(FeedbackRecievedDate) as Max FROM APP_IncidentCase")
result = cursor.fetchone()
print(f"   Total: {result.Total}")
print(f"   Date range: {result.Min} to {result.Max}")

# Show sample cases
print(f"\n🔍 Sample incident cases...")
cursor.execute("""
    SELECT TOP 5
        IncidentRequestCaseID,
        FeedbackRecievedDate,
        IssuingOrgUnitID,
        ClassificationID,
        SeverityID,
        DomainID
    FROM APP_IncidentCase
    ORDER BY FeedbackRecievedDate DESC
""")

print("   ID    | Date       | OrgUnit | Class | Severity | Domain")
print("   " + "-" * 60)
for row in cursor.fetchall():
    print(f"   {row.IncidentRequestCaseID:5} | {row.FeedbackRecievedDate} | {row.IssuingOrgUnitID:7} | {row.ClassificationID:5} | {row.SeverityID:8} | {row.DomainID:6}")

conn.close()
print("=" * 80)
