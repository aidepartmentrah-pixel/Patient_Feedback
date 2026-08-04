"""
Symmetric encryption for the "export user credentials" admin feature
(dbo.APP_UserPasswordExport).

Deliberately separate from PasswordHash (bcrypt, one-way, used for login --
never touched here) and from SETTINGS_ENCRYPTION_KEY (a different secret
domain entirely -- third-party integration keys). A dedicated key here means
rotating or, worst case, leaking one key doesn't expose the other secret
domain.

The encryption key itself is supplied ONLY via the
PASSWORD_EXPORT_ENCRYPTION_KEY process environment variable -- it is never
written to db_settings.json, the database, or any log.

Usage:
    from core.password_export_encryption import encrypt_password, decrypt_password

    ciphertext = encrypt_password("plain-password")   # -> str, safe to store
    plaintext = decrypt_password(ciphertext)           # -> str, decrypted in memory only
"""

import os
import logging

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)

_ENV_VAR_NAME = "PASSWORD_EXPORT_ENCRYPTION_KEY"


def _get_fernet() -> Fernet:
    """
    Build a Fernet cipher from the PASSWORD_EXPORT_ENCRYPTION_KEY env var.

    Looked up lazily (not at import time) so the app doesn't crash on
    startup just because nobody has configured this feature yet -- the
    error only surfaces when someone actually tries to encrypt or decrypt
    an exportable password.
    """
    key = os.environ.get(_ENV_VAR_NAME)
    if not key:
        raise RuntimeError(
            f"{_ENV_VAR_NAME} environment variable is not set. Generate one with "
            f"`python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"` "
            f"and set it as a service environment variable before saving or reading "
            f"any exportable password."
        )
    try:
        return Fernet(key.encode("utf-8"))
    except Exception as e:
        # Never log the key value itself, even on a malformed-key error.
        raise RuntimeError(f"{_ENV_VAR_NAME} is not a valid Fernet key: {type(e).__name__}") from e


def encrypt_password(plaintext: str) -> str:
    """Encrypt a plaintext password for storage in APP_UserPasswordExport."""
    if not plaintext:
        return ""
    token = _get_fernet().encrypt(plaintext.encode("utf-8"))
    return token.decode("utf-8")


def decrypt_password(ciphertext: str) -> str:
    """
    Decrypt a stored exportable password. Empty input returns empty output.

    Raises ValueError (not the raw cryptography exception) if the stored
    value can't be decrypted with the current key -- most commonly because
    PASSWORD_EXPORT_ENCRYPTION_KEY was rotated/changed without re-saving.
    """
    if not ciphertext:
        return ""
    try:
        return _get_fernet().decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        logger.error("Failed to decrypt a stored exportable password -- PASSWORD_EXPORT_ENCRYPTION_KEY may have changed.")
        raise ValueError(
            "Stored password could not be decrypted with the current PASSWORD_EXPORT_ENCRYPTION_KEY. "
            "It may have been encrypted with a different key -- re-save it again."
        )


def has_encryption_key() -> bool:
    """Non-raising check for whether PASSWORD_EXPORT_ENCRYPTION_KEY is configured at all."""
    return bool(os.environ.get(_ENV_VAR_NAME))
