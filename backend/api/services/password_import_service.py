"""
Import known passwords into dbo.APP_UserPasswordExport from a credentials
document (the "System Access Credentials" Word export, or any .docx whose
table has Username and Password columns).

Accounts provisioned with a bcrypt hash have no recoverable password, so
Export Credentials can't print them until their real password is supplied
once. Every imported password is checked against the account's actual login
hash first -- only passwords that really log in are stored, so the export
never prints a wrong password.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import bcrypt
from docx import Document

from core.database import get_connection
from .password_export_service import save_exportable_password

logger = logging.getLogger(__name__)

# Placeholder values earlier exports printed instead of a password.
_NOT_A_PASSWORD = {"", "[secured]", "not available - reset", "—", "-"}


def _read_credential_rows(docx_bytes: bytes) -> List[Tuple[str, str]]:
    """(username, password) pairs from every table with Username + Password header cells."""
    try:
        doc = Document(BytesIO(docx_bytes))
    except Exception as e:
        raise ValueError(f"Could not open the file as a Word document ({type(e).__name__}).")
    pairs = []
    for table in doc.tables:
        user_col = pass_col = None
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if user_col is None:
                lowered = [c.lower() for c in cells]
                if "username" in lowered and "password" in lowered:
                    user_col, pass_col = lowered.index("username"), lowered.index("password")
                continue
            if max(user_col, pass_col) < len(cells):
                pairs.append((cells[user_col], cells[pass_col]))
    return pairs


def password_matches(stored_hash: Optional[str], password: str) -> bool:
    """Same check login uses (auth_db.authenticate_user): TEMP_HASH_ compare or bcrypt."""
    if not stored_hash:
        return False
    if stored_hash.startswith("TEMP_HASH_"):
        return stored_hash[len("TEMP_HASH_"):] == password
    try:
        return bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
    except ValueError:
        return False


def import_passwords_from_docx(docx_bytes: bytes, admin_user_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Verify and store every password in the document.

    Returns counts plus the usernames in each outcome:
      stored          -- verified against the login hash and saved
      wrong_password  -- does not log in to that account (not saved)
      unknown_user    -- no such username in APP_Users
      skipped         -- rows with no password (placeholder text)
      failed          -- verified, but saving failed (see backend log)
    Raises ValueError if the document has no Username/Password table.
    """
    rows = _read_credential_rows(docx_bytes)
    if not rows:
        raise ValueError("No table with 'Username' and 'Password' columns found in the document.")

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT UserID, Username, PasswordHash FROM dbo.APP_Users")
        users = {r.Username.lower(): (r.UserID, r.PasswordHash) for r in cursor.fetchall()}
        cursor.close()

        result = {"stored": [], "wrong_password": [], "unknown_user": [], "skipped": [], "failed": []}
        to_check = []
        for username, password in rows:
            if not username:
                continue
            if password in _NOT_A_PASSWORD:
                result["skipped"].append(username)
            elif username.lower() not in users:
                result["unknown_user"].append(username)
            else:
                to_check.append((username, password))

        # bcrypt is deliberately slow (~0.25s per check); it releases the
        # GIL, so a small pool keeps a 150+ user import to a few seconds.
        with ThreadPoolExecutor(max_workers=8) as pool:
            verdicts = list(pool.map(lambda up: password_matches(users[up[0].lower()][1], up[1]), to_check))

        for (username, password), ok in zip(to_check, verdicts):
            if not ok:
                result["wrong_password"].append(username)
            elif save_exportable_password(conn, users[username.lower()][0], password, updated_by_user_id=admin_user_id):
                result["stored"].append(username)
            else:
                result["failed"].append(username)

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    logger.info(
        "Password import: %d stored, %d wrong password, %d unknown user, %d skipped, %d failed",
        len(result["stored"]), len(result["wrong_password"]), len(result["unknown_user"]),
        len(result["skipped"]), len(result["failed"]),
    )
    return {f"{k}_count": len(v) for k, v in result.items()} | result
