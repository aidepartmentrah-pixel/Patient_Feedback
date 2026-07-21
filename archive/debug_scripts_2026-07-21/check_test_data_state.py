"""Check current state of test data for closed-loop reporting test."""
from core.database import get_connection
from collections import defaultdict

conn = get_connection()
cursor = conn.cursor()

# 1. Find season/semester table
cursor.execute("""
SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_NAME LIKE '%eason%' OR TABLE_NAME LIKE '%emester%' OR TABLE_NAME LIKE '%period%'
""")
season_tables = [r[0] for r in cursor.fetchall()]
print(f"Season/Semester tables found: {season_tables}")

# Check each one
for tbl in season_tables:
    cursor.execute(f"SELECT TOP 5 * FROM dbo.[{tbl}]")
    cols = [d[0] for d in cursor.description]
    print(f"\n=== {tbl} ===")
    print(f"  Columns: {cols}")
    for r in cursor.fetchall():
        print(f"  {list(r)}")

# 2. Target departments per case
print("\n=== TARGET DEPARTMENTS PER CASE ===")
cursor.execute("""
SELECT IncidentRequestCaseID, DepartmentID
FROM dbo.APP_IncidentCaseTargetDepartment
WHERE IncidentRequestCaseID IN (492,493,494,495,496,497,498,499,500,501)
ORDER BY IncidentRequestCaseID, DepartmentID
""")
case_depts = defaultdict(list)
for r in cursor.fetchall():
    case_depts[r[0]].append(r[1])
for cid, depts in sorted(case_depts.items()):
    print(f"  CaseID={cid}: TargetDepts={depts}")

# 3. Case dates and details
print("\n=== CASE DATES ===")
cursor.execute("""
SELECT IncidentRequestCaseID, 
    CONVERT(varchar, FeedbackRecievedDate, 23) as DateStr,
    IssuingOrgUnitID
FROM dbo.APP_IncidentCase
WHERE IncidentRequestCaseID IN (492,493,494,495,496,497,498,499,500,501)
ORDER BY IncidentRequestCaseID
""")
for r in cursor.fetchall():
    print(f"  CaseID={r[0]}, Date={r[1]}, IssuingOrg={r[2]}")

# 4. Subcases
print("\n=== SUBCASES ===")
cursor.execute("""
SELECT IncidentRequestCaseID, SubcaseID, TargetOrgUnitID
FROM dbo.APP_AdministrativeSubcase
WHERE IncidentRequestCaseID IN (492,493,494,495,496,497,498,499,500,501)
ORDER BY IncidentRequestCaseID, SubcaseID
""")
for r in cursor.fetchall():
    print(f"  CaseID={r[0]}, SubcaseID={r[1]}, TargetOrg={r[2]}")

# 5. Section ID mapping
print("\n=== SECTION IDS FOR OUR 8 SECTIONS ===")
cursor.execute("""
SELECT OrgUnitID, OrgUnitName FROM dbo.APP_OrgUnit
WHERE OrgUnitID IN (43, 95, 60, 72, 98, 42, 309, 93)
ORDER BY OrgUnitID
""")
for r in cursor.fetchall():
    print(f"  ID={r[0]}: {r[1]}")

conn.close()
