"""Map the full org hierarchy for our 8 test sections."""
from core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

# Our 8 sections
section_ids = [43, 95, 60, 72, 98, 42, 309, 93]

print("=" * 100)
print("FULL ORG HIERARCHY: Section → Department → Administration")
print("=" * 100)

for sid in section_ids:
    # Walk up the tree: Section → Parent (Dept) → GrandParent (Admin)
    cursor.execute("""
        SELECT s.UniqueID as SecID, s.Name as SecName, s.Type as SecType, s.ParentID,
               d.UniqueID as DeptID, d.Name as DeptName, d.Type as DeptType, d.ParentID as DeptParentID,
               a.UniqueID as AdminID, a.Name as AdminName, a.Type as AdminType
        FROM dbo.AdminsrationUnit s
        LEFT JOIN dbo.AdminsrationUnit d ON s.ParentID = d.UniqueID
        LEFT JOIN dbo.AdminsrationUnit a ON d.ParentID = a.UniqueID
        WHERE s.UniqueID = ?
    """, sid)
    r = cursor.fetchone()
    if r:
        print(f"\n  Section:        ID={r[0]:>4}  Type={r[2]}  Name=[{r[1]}]")
        print(f"  Department:     ID={r[4]:>4}  Type={r[6]}  Name=[{r[5]}]")
        print(f"  Administration: ID={r[8]:>4}  Type={r[10]}  Name=[{r[9]}]")
    else:
        print(f"\n  Section ID={sid}: NOT FOUND")

# Also check org unit type values
print("\n" + "=" * 100)
print("ORG UNIT TYPE DEFINITIONS")
print("=" * 100)
cursor.execute("""
    SELECT DISTINCT Type, COUNT(*) as cnt
    FROM dbo.AdminsrationUnit
    GROUP BY Type
    ORDER BY Type
""")
for r in cursor.fetchall():
    label = {323: "Administration", 324: "Section", 325: "Department"}.get(r[0], "Unknown")
    print(f"  Type={r[0]}: {r[1]} units  ({label})")

# Print department grouping
print("\n" + "=" * 100)
print("DEPARTMENT-LEVEL GROUPING (sections rolled up to departments)")
print("=" * 100)
dept_sections = {}
for sid in section_ids:
    cursor.execute("SELECT ParentID FROM dbo.AdminsrationUnit WHERE UniqueID = ?", sid)
    parent = cursor.fetchone()[0]
    if parent not in dept_sections:
        cursor.execute("SELECT Name FROM dbo.AdminsrationUnit WHERE UniqueID = ?", parent)
        dept_sections[parent] = {"name": cursor.fetchone()[0], "sections": []}
    dept_sections[parent]["sections"].append(sid)

for did, info in sorted(dept_sections.items()):
    print(f"  Dept ID={did} [{info['name']}]: sections={info['sections']}")

# Print administration grouping
print("\n" + "=" * 100)
print("ADMIN-LEVEL GROUPING (sections rolled up to administrations)")
print("=" * 100)
admin_sections = {}
for sid in section_ids:
    cursor.execute("""
        SELECT a.UniqueID, a.Name
        FROM dbo.AdminsrationUnit s
        JOIN dbo.AdminsrationUnit d ON s.ParentID = d.UniqueID
        JOIN dbo.AdminsrationUnit a ON d.ParentID = a.UniqueID
        WHERE s.UniqueID = ?
    """, sid)
    r = cursor.fetchone()
    if r:
        aid, aname = r
        if aid not in admin_sections:
            admin_sections[aid] = {"name": aname, "sections": []}
        admin_sections[aid]["sections"].append(sid)

for aid, info in sorted(admin_sections.items()):
    print(f"  Admin ID={aid} [{info['name']}]: sections={info['sections']}")

conn.close()
