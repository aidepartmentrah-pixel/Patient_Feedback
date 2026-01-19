import pyodbc

conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=SOCIALMEDIA;'
    'DATABASE=IncidentManager;'
    'Trusted_Connection=yes;'
    'TrustServerCertificate=yes;'
)
c = conn.cursor()

# Check existing valid case IDs
c.execute("SELECT TOP 1 * FROM dbo.APP_IncidentCase")
if c.description:
    cols = [col[0] for col in c.description]
    row = c.fetchone()
    if row:
        case_dict = dict(zip(cols, row))
        print("Valid IDs from existing case:")
        print(f"  DomainID: {case_dict.get('DomainID')}")
        print(f"  CategoryID: {case_dict.get('CategoryID')}")
        print(f"  SubCategoryID: {case_dict.get('SubCategoryID')}")
        print(f"  ClassificationID: {case_dict.get('ClassificationID')}")
        print(f"  SeverityID: {case_dict.get('SeverityID')}")
        print(f"  StageID: {case_dict.get('StageID')}")
        print(f"  HarmLevelID: {case_dict.get('HarmLevelID')}")
        print(f"  IssuingOrgUnitID: {case_dict.get('IssuingOrgUnitID')}")
        print(f"  ClinicalRiskTypeID: {case_dict.get('ClinicalRiskTypeID')}")
conn.close()
