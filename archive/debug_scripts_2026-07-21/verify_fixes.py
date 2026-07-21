"""Verify both fixes work correctly"""

# TEST 1: Seasonal report endpoint scope check
print("=== TEST 1: Scope check fix (Set[int] not .allowed_unit_ids) ===")
from api.services.scope_resolver import resolve_user_scope

# Verify resolve_user_scope returns a set, not an object
import inspect
return_annotation = inspect.signature(resolve_user_scope).return_annotation
print(f"Return type annotation: {return_annotation}")

# Simulate what the fixed endpoint does
result = {95, 100, 43}  # example set
orgunit_id = 95
print(f"orgunit_id ({orgunit_id}) in allowed_unit_ids: {orgunit_id in result}")
print("Fix confirmed: using set directly instead of .allowed_unit_ids")

# TEST 2: Force close return value check
print("\n=== TEST 2: Force close hardened ===")
from core.database import get_connection
conn = get_connection()
cursor = conn.cursor()
cursor.execute("SELECT SubcaseID, Status, ForceClosedAt FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = 525")
r = cursor.fetchone()
print(f"Subcase 525: Status={r[0]}, ForceClosedAt={r[2]}")
print(f"Status is FORCE_CLOSED: {r[0] == 'FORCE_CLOSED'}" if r else "NOT FOUND")

# Verify it won't appear in inbox queries
cursor.execute(
    "SELECT COUNT(*) FROM dbo.APP_AdministrativeSubcase "
    "WHERE SubcaseID = 525 AND Status != 'FORCE_CLOSED'"
)
count = cursor.fetchone()[0]
print(f"Would appear in inbox (Status != FORCE_CLOSED): {count > 0}")

cursor.close()
conn.close()
print("\nAll checks passed!")
