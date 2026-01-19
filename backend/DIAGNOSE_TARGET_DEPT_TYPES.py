"""
Database Diagnostic: Check Target Department Assignments
Identifies type mismatches and null values in target department assignments
"""
import sys
sys.path.insert(0, r"C:\Users\IT\Documents\GitHub Repository\Patient_Feedback\backend")

from api.db_layer.reports_db import get_connection

conn = get_connection()
cursor = conn.cursor()

print("=" * 100)
print("DATABASE DIAGNOSTIC: Target Department Assignment Check")
print("=" * 100)
print()

# ========== CHECK 1: AdminsrationUnit Types Overview ==========
print("📊 STEP 1: AdminsrationUnit Types Overview")
print("-" * 100)

type_query = """
SELECT 
    Type,
    CASE 
        WHEN Type = 323 THEN 'Administration'
        WHEN Type = 324 THEN 'Section'
        WHEN Type = 325 THEN 'Department'
        ELSE 'Unknown'
    END as TypeName,
    COUNT(*) as Count
FROM dbo.AdminsrationUnit
GROUP BY Type
ORDER BY Type
"""

cursor.execute(type_query)
type_rows = cursor.fetchall()

print("Unit Types in AdminsrationUnit table:")
for row in type_rows:
    type_id = row.Type if row.Type is not None else 'NULL'
    type_name = row.TypeName
    count = row.Count
    print(f"  Type {type_id} ({type_name}): {count} units")
print()

# ========== CHECK 2: Target Departments with Type Info ==========
print("📊 STEP 2: Check Target Department Assignments")
print("-" * 100)

target_dept_query = """
SELECT 
    td.IncidentRequestCaseID,
    td.DepartmentID,
    au.Name as DepartmentName,
    au.Type as DepartmentType,
    CASE 
        WHEN au.Type = 323 THEN 'Administration (WRONG!)'
        WHEN au.Type = 324 THEN 'Section (CORRECT)'
        WHEN au.Type = 325 THEN 'Department (WRONG!)'
        WHEN au.Type IS NULL THEN 'NULL TYPE (ERROR!)'
        ELSE 'Unknown Type'
    END as TypeStatus,
    au.ParentID,
    parent.Name as ParentName,
    parent.Type as ParentType
FROM dbo.APP_IncidentCaseTargetDepartment td
LEFT JOIN dbo.AdminsrationUnit au ON td.DepartmentID = au.UniqueID
LEFT JOIN dbo.AdminsrationUnit parent ON au.ParentID = parent.UniqueID
ORDER BY td.IncidentRequestCaseID, td.DepartmentID
"""

cursor.execute(target_dept_query)
target_rows = cursor.fetchall()

print(f"Total Target Department Assignments: {len(target_rows)}")
print()

# Categorize by type
correct_sections = []
wrong_administrations = []
wrong_departments = []
null_types = []
orphaned = []

for row in target_rows:
    if row.DepartmentType is None:
        null_types.append(row)
    elif row.DepartmentType == 324:
        correct_sections.append(row)
    elif row.DepartmentType == 323:
        wrong_administrations.append(row)
    elif row.DepartmentType == 325:
        wrong_departments.append(row)
    
    # Check if orphaned (no match in AdminsrationUnit)
    if row.DepartmentName is None:
        orphaned.append(row)

print("Categorization by Type:")
print(f"  ✓ Correct (Section, Type 324):         {len(correct_sections)}")
print(f"  ❌ Wrong (Administration, Type 323):    {len(wrong_administrations)}")
print(f"  ❌ Wrong (Department, Type 325):        {len(wrong_departments)}")
print(f"  ❌ NULL Type:                           {len(null_types)}")
print(f"  ❌ Orphaned (No match in AdminUnit):    {len(orphaned)}")
print()

# ========== CHECK 3: Show Examples of Wrong Assignments ==========
if wrong_administrations:
    print("❌ WRONG ASSIGNMENTS - Administrations (should be Sections):")
    print("-" * 100)
    for row in wrong_administrations[:5]:
        print(f"  Complaint #{row.IncidentRequestCaseID}")
        print(f"    Assigned To: {row.DepartmentName} (ID: {row.DepartmentID})")
        print(f"    Type: {row.DepartmentType} (Administration) ← SHOULD BE 324 (Section)")
        print(f"    Parent: {row.ParentName} (ID: {row.ParentID}, Type: {row.ParentType})")
        print()

if wrong_departments:
    print("❌ WRONG ASSIGNMENTS - Departments (should be Sections):")
    print("-" * 100)
    for row in wrong_departments[:5]:
        print(f"  Complaint #{row.IncidentRequestCaseID}")
        print(f"    Assigned To: {row.DepartmentName} (ID: {row.DepartmentID})")
        print(f"    Type: {row.DepartmentType} (Department) ← SHOULD BE 324 (Section)")
        print(f"    Parent: {row.ParentName} (ID: {row.ParentID}, Type: {row.ParentType})")
        print()

if null_types:
    print("❌ NULL TYPE ASSIGNMENTS:")
    print("-" * 100)
    for row in null_types[:5]:
        print(f"  Complaint #{row.IncidentRequestCaseID}")
        print(f"    Assigned To: {row.DepartmentName or 'UNKNOWN'} (ID: {row.DepartmentID})")
        print(f"    Type: NULL ← MISSING TYPE!")
        print()

if orphaned:
    print("❌ ORPHANED ASSIGNMENTS (DepartmentID not in AdminsrationUnit):")
    print("-" * 100)
    for row in orphaned[:5]:
        print(f"  Complaint #{row.IncidentRequestCaseID}")
        print(f"    DepartmentID: {row.DepartmentID} ← NOT FOUND in AdminsrationUnit table!")
        print()

# ========== CHECK 4: Find Valid Sections for Corrections ==========
print("📊 STEP 3: Valid Sections Available for Assignment")
print("-" * 100)

sections_query = """
SELECT 
    au.UniqueID,
    au.Name,
    au.Type,
    parent_dept.Name as DepartmentName,
    parent_admin.Name as AdministrationName
FROM dbo.AdminsrationUnit au
LEFT JOIN dbo.AdminsrationUnit parent_dept ON au.ParentID = parent_dept.UniqueID
LEFT JOIN dbo.AdminsrationUnit parent_admin ON parent_dept.ParentID = parent_admin.UniqueID
WHERE au.Type = 324
ORDER BY parent_admin.Name, parent_dept.Name, au.Name
"""

cursor.execute(sections_query)
section_rows = cursor.fetchall()

print(f"Total Valid Sections (Type 324) available: {len(section_rows)}")
print("\nSample sections (first 10):")
for row in section_rows[:10]:
    print(f"  ID {row.UniqueID}: {row.Name}")
    print(f"    → Department: {row.DepartmentName}")
    print(f"    → Administration: {row.AdministrationName}")
print()

# ========== SUMMARY ==========
print("=" * 100)
print("🔍 SUMMARY & RECOMMENDATIONS")
print("=" * 100)
print()

total_wrong = len(wrong_administrations) + len(wrong_departments) + len(null_types) + len(orphaned)
total_assignments = len(target_rows)
percent_wrong = (total_wrong / total_assignments * 100) if total_assignments > 0 else 0

print(f"Total Target Department Assignments: {total_assignments}")
print(f"Correct Assignments (Section Type 324): {len(correct_sections)} ({len(correct_sections)/total_assignments*100:.1f}%)")
print(f"Incorrect Assignments: {total_wrong} ({percent_wrong:.1f}%)")
print()

if total_wrong > 0:
    print("❌ DATA QUALITY ISSUES FOUND:")
    if wrong_administrations:
        print(f"  1. {len(wrong_administrations)} assignments point to Administrations (Type 323) instead of Sections")
    if wrong_departments:
        print(f"  2. {len(wrong_departments)} assignments point to Departments (Type 325) instead of Sections")
    if null_types:
        print(f"  3. {len(null_types)} assignments have NULL type")
    if orphaned:
        print(f"  4. {len(orphaned)} assignments reference non-existent IDs")
    print()
    print("RECOMMENDATION:")
    print("  - Target departments should ONLY reference Sections (Type 324)")
    print("  - Update APP_IncidentCaseTargetDepartment.DepartmentID to point to valid Section IDs")
    print("  - Add database constraint to prevent future incorrect assignments")
else:
    print("✅ All target department assignments are correct!")

print("=" * 100)

conn.close()
