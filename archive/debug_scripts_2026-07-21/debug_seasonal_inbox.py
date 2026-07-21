from core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

# Find ALL seasonal subcases
cursor.execute(
    "SELECT SubcaseID, CaseType, SeasonalReportID, TargetOrgUnitID, Status, CreatedAt "
    "FROM dbo.APP_AdministrativeSubcase "
    "WHERE CaseType LIKE '%SEASONAL%' ORDER BY SubcaseID"
)
cols = [d[0] for d in cursor.description]
rows = cursor.fetchall()
for r in rows:
    print(dict(zip(cols, r)))

print()

# Check reports for unit 95
cursor.execute(
    "SELECT SeasonalReportID, SeasonID, OrgUnitID, OrgUnitType, TotalCases, IsCompliant "
    "FROM dbo.APP_SeasonalOrgUnitReport WHERE OrgUnitID = 95"
)
cols2 = [d[0] for d in cursor.description]
for r in cursor.fetchall():
    print("Unit95:", dict(zip(cols2, r)))

cursor.close()
conn.close()
