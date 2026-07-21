from core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

# Check subcase 525 full state
cursor.execute(
    "SELECT SubcaseID, CaseType, Status, ForceClosedAt, ForceClosedByUserID, "
    "ForceCloseReason, UpdatedAt, UpdatedByUserID, IncidentRequestCaseID "
    "FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = 525"
)
cols = [d[0] for d in cursor.description]
r = cursor.fetchone()
if r:
    print("Subcase 525:", dict(zip(cols, r)))
else:
    print("Subcase 525 NOT FOUND")

# Check if ForceCloseReason column exists
cursor.execute(
    "SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE "
    "FROM INFORMATION_SCHEMA.COLUMNS "
    "WHERE TABLE_NAME = 'APP_AdministrativeSubcase' "
    "AND COLUMN_NAME LIKE '%Force%' "
    "ORDER BY ORDINAL_POSITION"
)
print("\nForce-close columns:")
for r in cursor.fetchall():
    print(f"  {r[0]} ({r[1]}, nullable={r[2]})")

# Also check what incident links to subcase 525
cursor.execute(
    "SELECT SubcaseID, IncidentRequestCaseID, SeasonalReportID, CaseType "
    "FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = 525"
)
r = cursor.fetchone()
if r:
    print(f"\nSubcase 525: IncidentID={r[1]}, SeasonalReportID={r[2]}, CaseType={r[3]}")

# Try seasonal report 924 to debug 500 error
print("\n--- Seasonal Report 924 Debug ---")
cursor.execute(
    "SELECT SeasonalReportID, SeasonID, OrgUnitID, OrgUnitType "
    "FROM dbo.APP_SeasonalOrgUnitReport WHERE SeasonalReportID = 924"
)
r = cursor.fetchone()
if r:
    print(f"Report 924: SeasonID={r[0]}, OrgUnitID={r[1]}, OrgUnitType={r[2]}")
    # Try the full query that the endpoint uses
    season_id, orgunit_id, orgunit_type = r[1], r[2], r[3]
    
    # Check if the JOIN table exists for this report
    cursor.execute(
        "SELECT sr.SeasonalReportID, s.SeasonName, ou.Name "
        "FROM dbo.APP_SeasonalOrgUnitReport sr "
        "LEFT JOIN dbo.Season s ON sr.SeasonID = s.UniqueID "
        "LEFT JOIN dbo.AdminsrationUnit ou ON sr.OrgUnitID = ou.UniqueID "
        "WHERE sr.SeasonID = ? AND sr.OrgUnitID = ? AND sr.OrgUnitType = ?",
        (season_id, orgunit_id, orgunit_type)
    )
    r2 = cursor.fetchone()
    if r2:
        print(f"Full query result: ReportID={r2[0]}, Season={r2[1]}, OrgUnit={r2[2]}")
    else:
        print("Full query returned NO RESULTS")
        
        # Debug: check if AdminsrationUnit has this ID
        cursor.execute("SELECT UniqueID, Name FROM dbo.AdminsrationUnit WHERE UniqueID = ?", (orgunit_id,))
        r3 = cursor.fetchone()
        print(f"AdminsrationUnit lookup for {orgunit_id}: {r3}")
        
        # Check APP_OrganizationUnits instead
        cursor.execute("SELECT OUnitID, OUnitName FROM dbo.APP_OrganizationUnits WHERE OUnitID = ?", (orgunit_id,))
        r4 = cursor.fetchone()
        print(f"APP_OrganizationUnits lookup for {orgunit_id}: {r4}")
else:
    print("Report 924 NOT FOUND")

cursor.close()
conn.close()
