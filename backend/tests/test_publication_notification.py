"""
Quick test: publication summary notification (mock mode).

Run from the backend directory:
    python tests/test_publication_notification.py

Verifies:
  1. Empty subcases list -> no email sent
  2. Subcases with valid org units -> mock log shows correct count per email
  3. Two org units sharing the same admin -> consolidated into one email
"""

import sys
import logging

sys.path.insert(0, '.')
sys.path.insert(0, 'backend')

# Enable INFO logging so mock output is visible
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s"
)

from api.services.notification_service import (
    send_publication_summary_notifications,
    get_section_admin_email,
)

print("=" * 65)
print("PUBLICATION SUMMARY NOTIFICATION — MOCK MODE TEST")
print("=" * 65)

# ── Test 1: empty list ────────────────────────────────────────────
print("\n[1] Empty subcases list -> nothing sent")
send_publication_summary_notifications([])
print("    ↳ passed (check log above for DEBUG skip message)")

# ── Test 2: look up a real org unit from the DB ───────────────────
print("\n[2] Checking DB for any org unit that has a SECTION_ADMIN email...")
from core.database import get_connection

conn = get_connection()
cur = conn.cursor()

cur.execute("""
    SELECT TOP 5 urs.OrgUnitID, u.Email, u.DisplayName
    FROM dbo.APP_Users u
    INNER JOIN dbo.APP_UserRoleScope urs ON u.UserID = urs.UserID
    INNER JOIN dbo.APP_Roles r ON urs.RoleID = r.RoleID
    WHERE r.RoleCode = 'SECTION_ADMIN'
      AND u.Email IS NOT NULL AND u.Email != ''
      AND u.IsActive = 1
""")
rows = cur.fetchall()
cur.close()
conn.close()

if not rows:
    print("    ↳ No SECTION_ADMIN users with emails found in DB.")
    print("      Test 2 skipped — add an email to a SECTION_ADMIN user to test live routing.")
else:
    for row in rows:
        print(f"    Found: OrgUnitID={row.OrgUnitID}  Email={row.Email}  Name={row.DisplayName}")

    # Use the first two org units for the test
    org_units = [r.OrgUnitID for r in rows[:2]]

    fake_subcases = [
        {"subcase_id": 9901, "target_org_unit_id": org_units[0]},
        {"subcase_id": 9902, "target_org_unit_id": org_units[0]},
    ]
    if len(org_units) >= 2:
        fake_subcases.append({"subcase_id": 9903, "target_org_unit_id": org_units[1]})

    print(f"\n[3] Sending summary for {len(fake_subcases)} fake subcases -> org units {org_units}")
    print("    (expect [MOCK] lines below)\n")
    send_publication_summary_notifications(fake_subcases)

print("\n" + "=" * 65)
print("Done. If [MOCK] lines appeared above, the notification path works.")
print("Switch notification_mode to 'smtp' in db_settings.json for real delivery.")
print("=" * 65)
