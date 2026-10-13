"""
Regression test for "Create Section + Admin User" failing with
"Cannot insert the value NULL into column 'UniqueID', table AdminsrationUnit".

AdminsrationUnit.UniqueID is a plain int PK (no IDENTITY/default), so
insert_section must compute the ID itself.

Tests:
1. A section is created with UniqueID = previous MAX + 1, Type = SECTION,
   correct ParentID, plus a SECTION_ADMIN user scoped to it.
2. Two concurrent creates both succeed with distinct IDs (UPDLOCK/HOLDLOCK).

Every row created here is deleted at the end.

Run: python backend/tests/test_section_creation.py
"""

import sys
import os
import threading

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from core.database import get_connection
from api.constants.org_unit_types import ORG_TYPE_SECTION, ORG_TYPE_ADMINISTRATION, ORG_TYPE_DEPARTMENT
from api.services.section_admin_creator_service import create_section_with_admin


def scalar(sql, params=()):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        row = cursor.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def get_parent_unit_id():
    return scalar(
        f"SELECT TOP 1 UniqueID FROM dbo.AdminsrationUnit "
        f"WHERE Type IN ({ORG_TYPE_ADMINISTRATION}, {ORG_TYPE_DEPARTMENT}) ORDER BY UniqueID"
    )


def cleanup(section_ids):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        for section_id in section_ids:
            username = f"sec_{section_id}_admin"
            cursor.execute(
                "DELETE FROM dbo.APP_UserRoleScope WHERE UserID IN "
                "(SELECT UserID FROM dbo.APP_Users WHERE Username = ?)", (username,))
            cursor.execute("DELETE FROM dbo.APP_Users WHERE Username = ?", (username,))
            cursor.execute("DELETE FROM dbo.AdminsrationUnit WHERE UniqueID = ?", (section_id,))
        conn.commit()
    finally:
        conn.close()


def test_single_create(created):
    parent_id = get_parent_unit_id()
    assert parent_id, "No ADMINISTRATION/DEPARTMENT unit to use as parent"
    max_before = scalar("SELECT ISNULL(MAX(UniqueID), 0) FROM dbo.AdminsrationUnit")

    result = create_section_with_admin("TEST_SECTION_CREATION", parent_id)
    created.append(result["section_id"])

    assert result["section_id"] == max_before + 1, (result["section_id"], max_before)
    assert scalar("SELECT Type FROM dbo.AdminsrationUnit WHERE UniqueID = ?", (result["section_id"],)) == ORG_TYPE_SECTION
    assert scalar("SELECT ParentID FROM dbo.AdminsrationUnit WHERE UniqueID = ?", (result["section_id"],)) == parent_id
    assert result["username"] == f"sec_{result['section_id']}_admin"
    assert scalar(
        "SELECT COUNT(*) FROM dbo.APP_UserRoleScope s JOIN dbo.APP_Users u ON u.UserID = s.UserID "
        "WHERE u.Username = ? AND s.OrgUnitID = ? AND s.OrgUnitType = 'SECTION'",
        (result["username"], result["section_id"])) == 1
    print(f"  OK single create -> UniqueID {result['section_id']} (previous max {max_before})")


def test_concurrent_create(created):
    parent_id = get_parent_unit_id()
    results, errors = [], []

    def worker(i):
        try:
            results.append(create_section_with_admin(f"TEST_SECTION_CONCURRENT_{i}", parent_id))
        except Exception as e:
            errors.append(str(e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    created.extend(r["section_id"] for r in results)
    assert not errors, errors
    ids = [r["section_id"] for r in results]
    assert len(set(ids)) == 2, ids
    print(f"  OK concurrent create -> distinct UniqueIDs {sorted(ids)}")


if __name__ == "__main__":
    created = []
    try:
        test_single_create(created)
        test_concurrent_create(created)
        print("ALL SECTION CREATION TESTS PASSED")
    finally:
        cleanup(created)
        print(f"  cleaned up sections {created}")
