"""
Symmetric encryption for runtime integration secrets (e.g. the Hospital
Directory API key) stored in SQL Server.

The encryption key itself is supplied ONLY via the SETTINGS_ENCRYPTION_KEY
process environment variable — it is never written to db_settings.json, the
database, or any log. This is deliberately separate from the bootstrap
database connection settings (which live in db_settings.json); see
core/config_loader.py and core/deployment_port.py for that.

Usage:
    from core.settings_encryption import encrypt_value, decrypt_value

    ciphertext = encrypt_value("plain-secret")   # -> str, safe to store
    plaintext = decrypt_value(ciphertext)        # -> str, decrypted in memory only
"""

import os
import logging

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)

_ENV_VAR_NAME = "SETTINGS_ENCRYPTION_KEY"


def _get_fernet() -> Fernet:
    """
    Build a Fernet cipher from the SETTINGS_ENCRYPTION_KEY env var.

    Looked up lazily (not at import time) so the app doesn't crash on
    startup just because nobody has configured an external integration yet
    — the error only surfaces when someone actually tries to encrypt or
    decrypt a secret.
    """
    key = os.environ.get(_ENV_VAR_NAME)
    if not key:
        raise RuntimeError(
            f"{_ENV_VAR_NAME} environment variable is not set. Generate one with "
            f"`python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"` "
            f"and set it as a service environment variable before saving or reading "
            f"any encrypted integration settings."
        )
    try:
        return Fernet(key.encode("utf-8"))
    except Exception as e:
        # Never log the key value itself, even on a malformed-key error.
        raise RuntimeError(f"{_ENV_VAR_NAME} is not a valid Fernet key: {type(e).__name__}") from e


def encrypt_value(plaintext: str) -> str:
    """Encrypt a plaintext secret for storage. Empty input returns empty output."""
    if not plaintext:
        return ""
    token = _get_fernet().encrypt(plaintext.encode("utf-8"))
    return token.decode("utf-8")


def decrypt_value(ciphertext: str) -> str:
    """
    Decrypt a stored secret. Empty input returns empty output.

    Raises ValueError (not the raw cryptography exception) if the stored
    value can't be decrypted with the current key — most commonly because
    SETTINGS_ENCRYPTION_KEY was rotated/changed without re-saving the secret.
    """
    if not ciphertext:
        return ""
    try:
        return _get_fernet().decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        logger.error("Failed to decrypt a stored integration secret — SETTINGS_ENCRYPTION_KEY may have changed.")
        raise ValueError(
            "Stored value could not be decrypted with the current SETTINGS_ENCRYPTION_KEY. "
            "It may have been encrypted with a different key — re-enter and save it again."
        )


def has_encryption_key() -> bool:
    """Non-raising check for whether SETTINGS_ENCRYPTION_KEY is configured at all."""
    return bool(os.environ.get(_ENV_VAR_NAME))
