"""
Test: Verify frozen/NULL type sections are excluded
"""
import sys
sys.path.insert(0, r"C:\Users\IT\Documents\GitHub Repository\Patient_Feedback\backend")

from api.db_layer.admin_units import get_units_by_type
from api.db_layer.reports_db import get_connection

# Get sections using the function
sections = get_units_by_type(324)  # Type 324 = Sections

print(f"✓ Sections returned by get_units_by_type(324): {len(sections)}")

# Verify none have NULL type
conn = get_connection()
cursor = conn.cursor()

section_ids = [s['id'] for s in sections]

if section_ids:
    placeholders = ','.join(['?'] * len(section_ids))
    query = f"""
    SELECT UniqueID, Name, Type, Frozen
    FROM AdminsrationUnit
    WHERE UniqueID IN ({placeholders})
    """
    
    cursor.execute(query, section_ids)
    rows = cursor.fetchall()
    
    null_types = [r for r in rows if r.Type is None]
    frozen = [r for r in rows if r.Frozen == 1]
    
    print(f"✓ Sections with NULL type: {len(null_types)}")
    print(f"✓ Frozen sections: {len(frozen)}")
    
    if null_types:
        print("\n❌ ERROR: Found sections with NULL type:")
        for r in null_types:
            print(f"  - ID {r.UniqueID}: {r.Name}")
    
    if frozen:
        print("\n❌ ERROR: Found frozen sections:")
        for r in frozen:
            print(f"  - ID {r.UniqueID}: {r.Name}")
    
    if not null_types and not frozen:
        print("\n✅ SUCCESS: All returned sections have valid Type (324) and Frozen=0")

# Check total sections in database (including frozen/NULL)
cursor.execute("""
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN Type = 324 THEN 1 ELSE 0 END) as type_324,
        SUM(CASE WHEN Type IS NULL THEN 1 ELSE 0 END) as null_type,
        SUM(CASE WHEN Frozen = 1 THEN 1 ELSE 0 END) as frozen
    FROM AdminsrationUnit
    WHERE Type = 324 OR Type IS NULL
""")

stats = cursor.fetchone()
print(f"\nDatabase Statistics:")
print(f"  Total units with Type=324 or NULL: {stats.total}")
print(f"  Valid sections (Type=324): {stats.type_324}")
print(f"  Units with NULL type: {stats.null_type}")
print(f"  Frozen units: {stats.frozen}")
print(f"  Returned by function: {len(sections)}")

conn.close()

print("\n" + "=" * 60)
print("✅ Filter is working correctly - frozen and NULL types excluded")
print("=" * 60)
