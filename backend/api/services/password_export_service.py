"""
Service layer for capturing a real, working password into
dbo.APP_UserPasswordExport every time one is set (create user, admin reset,
edit user), whatever the password mode -- so Export Credentials always has
the current password, and never a stale one after a change.
"""

import logging
from typing import Optional
from core.password_export_encryption import encrypt_password, has_encryption_key
from ..db_layer.user_password_export_db import upsert_password_export

logger = logging.getLogger(__name__)


def save_exportable_password(conn, user_id: int, plaintext_password: str, updated_by_user_id: Optional[int] = None) -> bool:
    """
    Best-effort: encrypt and upsert `plaintext_password` for `user_id` into
    APP_UserPasswordExport, using the caller's existing transaction (`conn`).

    Never raises -- a failure here (e.g. no usable encryption key) must not
    block account creation or password reset, which are the actually-critical
    operations. Returns whether the password was saved.
    """
    if not has_encryption_key():
        logger.error("No password export key available -- skipping exportable password capture for user_id=%s", user_id)
        return False
    try:
        encrypted = encrypt_password(plaintext_password)
        upsert_password_export(conn, user_id, encrypted, updated_by_user_id)
        return True
    except Exception:
        logger.exception("Failed to save exportable password for user_id=%s (non-fatal)", user_id)
        return False
