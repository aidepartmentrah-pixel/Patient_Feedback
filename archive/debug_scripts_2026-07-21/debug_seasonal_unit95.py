from core.database import get_connection
conn = get_connection()
cursor = conn.cursor()

cursor.execute(
    "SELECT SubcaseID, SeasonalReportID, TargetOrgUnitID, Status "
    "FROM dbo.APP_AdministrativeSubcase "
    "WHERE TargetOrgUnitID = 95 AND CaseType = 'SEASONAL_REPORT_RESPONSE'"
)
for r in cursor.fetchall():
    print(r)

print()

# Get the seasonal report details for those
cursor.execute(
    "SELECT r.SeasonalReportID, r.SeasonID, r.OrgUnitID, r.TotalCases, "
    "r.LowSeverityCount, r.MediumSeverityCount, r.HighSeverityCount, "
    "r.ClinicalDomainCount, r.ManagementDomainCount, r.RelationalDomainCount, "
    "r.IsCompliant, r.ViolatedRules, s.SeasonName "
    "FROM dbo.APP_SeasonalOrgUnitReport r "
    "JOIN dbo.Season s ON s.UniqueID = r.SeasonID "
    "WHERE r.SeasonalReportID IN ("
    "  SELECT SeasonalReportID FROM dbo.APP_AdministrativeSubcase "
    "  WHERE TargetOrgUnitID = 95 AND CaseType = 'SEASONAL_REPORT_RESPONSE'"
    ")"
)
cols = [d[0] for d in cursor.description]
for r in cursor.fetchall():
    print(dict(zip(cols, r)))

cursor.close()
conn.close()
