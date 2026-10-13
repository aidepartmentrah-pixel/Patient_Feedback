"""
Smoke tests for the credentials export: every place a password is set keeps
a decryptable copy in dbo.APP_UserPasswordExport, and Export Credentials
prints the real, current password for every account.

Tests:
1. create (production mode) -> decryptable export row
2. admin reset (production mode) -> row upserted in place, new password
3. edit user with new password (production mode) -> login hash is bcrypt
   (not TEMP_HASH_), export row updated to the new password
4. no PASSWORD_EXPORT_ENCRYPTION_KEY -> key file generated and used; a row
   saved under the key file still decrypts once the env key is set again
5. test mode (TEMP_HASH_) create -> export row written too
6. import from a .docx: correct password stored, wrong password / unknown
   user / placeholder rows not stored
7. the Word export itself shows every test user's real password
8. delete_user works for a user that has an export row (FK)

Forces password mode via HCAT_PASSWORD_MODE and redirects the key file to a
temp dir, so nothing outside the test users is touched. The env key is
restored afterwards.

Run: python backend/tests/test_password_export_smoke.py
"""

import sys
import os
import tempfile
import traceback
from io import BytesIO
from pathlib import Path

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from cryptography.fernet import Fernet
from docx import Document

import core.password_export_encryption as pe
from core.database import get_connection
from core.password_export_encryption import decrypt_password
from api.db_layer.user_management_db import delete_user_scopes, delete_user
from api.services.user_management_service import (
    create_user_with_role_scope, admin_reset_user_password_service, update_user_service,
)
from api.services.password_import_service import import_passwords_from_docx
from api.routers.admin_user_markdown_router import export_user_credentials_as_word
from api.schemas.auth_models import CurrentUser, UserScope

_KEY_ENV = "PASSWORD_EXPORT_ENCRYPTION_KEY"
_MODE_ENV = "HCAT_PASSWORD_MODE"
_PID = os.getpid()


def query_one(sql, params=()):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        return cursor.fetchone()
    finally:
        conn.close()


def role_and_unit():
    role_id = query_one("SELECT TOP 1 RoleID FROM dbo.APP_Roles WHERE RoleCode = 'WORKER'").RoleID
    unit_id = query_one("SELECT TOP 1 UniqueID FROM dbo.AdminsrationUnit ORDER BY UniqueID").UniqueID
    return role_id, unit_id


def new_user(suffix, password):
    role_id, unit_id = role_and_unit()
    return create_user_with_role_scope(
        username=f"pwexport_{suffix}_{_PID}", password_plain=password,
        display_name="Password Export Smoke Test", department_display_name="Test",
        role_id=role_id, org_unit_id=unit_id,
    )


def exported_password(user_id):
    row = query_one("SELECT EncryptedPassword FROM dbo.APP_UserPasswordExport WHERE UserID = ?", (user_id,))
    return decrypt_password(row.EncryptedPassword) if row else None


def export_row_count(user_id):
    return query_one("SELECT COUNT(*) FROM dbo.APP_UserPasswordExport WHERE UserID = ?", (user_id,))[0]


def password_hash(user_id):
    return query_one("SELECT PasswordHash FROM dbo.APP_Users WHERE UserID = ?", (user_id,)).PasswordHash


def username_of(user_id):
    return query_one("SELECT Username FROM dbo.APP_Users WHERE UserID = ?", (user_id,)).Username


def delete_test_user(user_id):
    conn = get_connection()
    try:
        delete_user_scopes(conn, user_id)
        delete_user(conn, user_id)  # must also remove the export row (FK)
        conn.commit()
    finally:
        conn.close()


def credentials_docx(rows):
    doc = Document()
    table = doc.add_table(rows=1, cols=4)
    for i, h in enumerate(["#", "Username", "Password", "Role"]):
        table.rows[0].cells[i].text = h
    for n, (username, password) in enumerate(rows, start=1):
        cells = table.add_row().cells
        cells[0].text, cells[1].text, cells[2].text, cells[3].text = str(n), username, password, "WORKER"
    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()


def word_export_passwords():
    """username -> password as printed by the real Word export endpoint."""
    supervisor = CurrentUser(
        user_id=1, username="smoke_supervisor", is_active=True, roles=["COMPLAINT_SUPERVISOR"],
        scopes=[UserScope(role_code="COMPLAINT_SUPERVISOR", org_unit_id=1, org_unit_type="ADMINISTRATION")],
    )
    doc = Document(BytesIO(export_user_credentials_as_word(supervisor).body))
    table = doc.tables[1]
    header = [c.text.strip() for c in table.rows[0].cells]
    ui, pi = header.index("Username"), header.index("Password")
    return {r.cells[ui].text.strip(): r.cells[pi].text.strip() for r in table.rows[1:]}


def run_all_tests():
    original_key = os.environ.get(_KEY_ENV)
    original_mode = os.environ.get(_MODE_ENV)
    original_key_file = pe._KEY_FILE
    tmpdir = tempfile.mkdtemp()
    pe._KEY_FILE = Path(tmpdir) / "password_export.key"
    os.environ[_KEY_ENV] = Fernet.generate_key().decode()
    os.environ[_MODE_ENV] = "production"

    created, results = [], []

    def test(name, fn):
        print(f"\n--- {name}")
        try:
            fn()
            print("    PASSED")
            results.append((name, True))
        except Exception as e:
            print(f"    FAILED: {e}")
            traceback.print_exc()
            results.append((name, False))

    state = {}

    def t1():
        uid = new_user("create", "SmokeCreate123!")
        created.append(uid)
        state["uid"] = uid
        assert exported_password(uid) == "SmokeCreate123!"
        assert not password_hash(uid).startswith("TEMP_HASH_")

    def t2():
        uid = state["uid"]
        admin_reset_user_password_service(user_id=uid, new_password="SmokeReset456!")
        assert export_row_count(uid) == 1
        assert exported_password(uid) == "SmokeReset456!"

    def t3():
        uid = state["uid"]
        update_user_service(user_id=uid, password="SmokeEdit789!")
        assert password_hash(uid).startswith("$2"), "production-mode edit must store bcrypt, not TEMP_HASH_"
        assert export_row_count(uid) == 1
        assert exported_password(uid) == "SmokeEdit789!"

    def t4():
        env_key = os.environ.pop(_KEY_ENV)
        try:
            uid = new_user("nokey", "SmokeNoKey012!")
            created.append(uid)
            assert pe._KEY_FILE.exists(), "key file should be generated when no env key is set"
            assert exported_password(uid) == "SmokeNoKey012!"
            state["nokey_uid"] = uid
        finally:
            os.environ[_KEY_ENV] = env_key
        # env key back: the row saved under the key file must still decrypt
        assert exported_password(state["nokey_uid"]) == "SmokeNoKey012!"

    def t5():
        os.environ[_MODE_ENV] = "test"
        try:
            uid = new_user("testmode", "SmokeTempHash345!")
            created.append(uid)
            assert password_hash(uid) == "TEMP_HASH_SmokeTempHash345!"
            assert exported_password(uid) == "SmokeTempHash345!"
        finally:
            os.environ[_MODE_ENV] = "production"

    def t6():
        uid = new_user("import", "ImportRight678!")
        created.append(uid)
        conn = get_connection()
        try:
            conn.cursor().execute("DELETE FROM dbo.APP_UserPasswordExport WHERE UserID = ?", (uid,))
            conn.commit()
        finally:
            conn.close()
        assert exported_password(uid) is None
        name = username_of(uid)
        other = username_of(state["uid"])
        result = import_passwords_from_docx(credentials_docx([
            (name, "ImportRight678!"),
            (other, "NotTheRightPassword!"),
            (f"no_such_user_{_PID}", "Whatever123!"),
            (name + "_x", ""),
            (name, "[secured]"),
        ]))
        assert result["stored"] == [name], result
        assert result["wrong_password"] == [other], result
        assert result["unknown_user"] == [f"no_such_user_{_PID}"], result
        assert result["skipped_count"] == 2, result
        assert exported_password(uid) == "ImportRight678!"
        assert exported_password(state["uid"]) == "SmokeEdit789!", "wrong password must not overwrite"
        try:
            import_passwords_from_docx(b"not a docx")
            raise AssertionError("expected ValueError for a non-docx upload")
        except ValueError:
            pass

    def t7():
        printed = word_export_passwords()
        expected = {
            username_of(state["uid"]): "SmokeEdit789!",
            username_of(state["nokey_uid"]): "SmokeNoKey012!",
        }
        for username, password in expected.items():
            assert printed.get(username) == password, (username, printed.get(username))
        assert "[secured]" not in printed.values()

    def t8():
        for uid in list(created):
            delete_test_user(uid)
            assert query_one("SELECT COUNT(*) FROM dbo.APP_Users WHERE UserID = ?", (uid,))[0] == 0
            created.remove(uid)

    try:
        test("1 create writes decryptable export row", t1)
        test("2 admin reset upserts export row", t2)
        test("3 edit user: bcrypt in production + export row updated", t3)
        test("4 no env key -> key file generated and used", t4)
        test("5 test mode also writes export row", t5)
        test("6 import from .docx stores only verified passwords", t6)
        test("7 Word export prints real passwords", t7)
        test("8 delete_user removes users that have export rows", t8)
    finally:
        for uid in created:
            try:
                delete_test_user(uid)
            except Exception as e:
                print(f"  [Cleanup Warning] user {uid}: {e}")
        pe._KEY_FILE = original_key_file
        if original_key is not None:
            os.environ[_KEY_ENV] = original_key
        else:
            os.environ.pop(_KEY_ENV, None)
        if original_mode is not None:
            os.environ[_MODE_ENV] = original_mode
        else:
            os.environ.pop(_MODE_ENV, None)

    print("\n" + "=" * 70)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"{'PASSED' if ok else 'FAILED'}: {name}")
    print(f"\nTotal: {passed}/{len(results)} passed")
    return passed == len(results)


if __name__ == "__main__":
    sys.exit(0 if run_all_tests() else 1)
