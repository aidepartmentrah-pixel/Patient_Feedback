"""
DB layer for dbo.APP_UserPasswordExport -- stores the Fernet-encrypted,
reversible copy of a real (production-mode) password so it can be shown in
the "System Access Credentials" export. Never touches APP_Users.PasswordHash.
"""

from typing import Optional


def upsert_password_export(conn, user_id: int, encrypted_password: str, updated_by_user_id: Optional[int] = None) -> None:
    """
    Insert or update the exportable (encrypted) password for a user.

    Uses the caller's connection and does NOT commit -- meant to run inside
    the same transaction as the corresponding APP_Users.PasswordHash write,
    so both succeed or roll back together.
    """
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            MERGE dbo.APP_UserPasswordExport AS tgt
            USING (SELECT ? AS UserID) AS src
            ON tgt.UserID = src.UserID
            WHEN MATCHED THEN
                UPDATE SET EncryptedPassword = ?,
                           UpdatedAt         = GETDATE(),
                           UpdatedByUserID   = ?
            WHEN NOT MATCHED THEN
                INSERT (UserID, EncryptedPassword, UpdatedAt, UpdatedByUserID)
                VALUES (src.UserID, ?, GETDATE(), ?);
            """,
            (user_id, encrypted_password, updated_by_user_id, encrypted_password, updated_by_user_id)
        )
    finally:
        cursor.close()
