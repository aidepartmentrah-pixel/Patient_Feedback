"""
Symmetric encryption for the "export user credentials" admin feature
(dbo.APP_UserPasswordExport).

Deliberately separate from PasswordHash (bcrypt, one-way, used for login --
never touched here) and from SETTINGS_ENCRYPTION_KEY (a different secret
domain entirely -- third-party integration keys). A dedicated key here means
rotating or, worst case, leaking one key doesn't expose the other secret
domain.

Where the key comes from:
  1. PASSWORD_EXPORT_ENCRYPTION_KEY environment variable, when set.
  2. Otherwise config/password_export.key, generated automatically on first
     use. config/ is the persistent backend_config volume in Docker, so the
     key survives image rebuilds -- the export works without anyone having
     to configure anything.
Both keys are tried when decrypting, so rows saved under either one stay
readable if the env var is later added or removed. The key is never written
to db_settings.json, the database, or any log.

Usage:
    from core.password_export_encryption import encrypt_password, decrypt_password

    ciphertext = encrypt_password("plain-password")   # -> str, safe to store
    plaintext = decrypt_password(ciphertext)           # -> str, decrypted in memory only
"""

import os
import logging
from pathlib import Path
from typing import List, Optional

from cryptography.fernet import Fernet, InvalidToken, MultiFernet

logger = logging.getLogger(__name__)

_ENV_VAR_NAME = "PASSWORD_EXPORT_ENCRYPTION_KEY"
_KEY_FILE = Path(__file__).resolve().parent.parent / "config" / "password_export.key"


def _load_or_create_key_file() -> Optional[bytes]:
    """Read config/password_export.key, creating it (owner-only) on first use."""
    try:
        if not _KEY_FILE.exists():
            _KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
            try:
                fd = os.open(str(_KEY_FILE), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                pass  # another worker created it first -- read theirs below
            else:
                with os.fdopen(fd, "wb") as f:
                    f.write(Fernet.generate_key())
                logger.info("Generated password export key file %s", _KEY_FILE)
        key = _KEY_FILE.read_bytes().strip()
        Fernet(key)  # validate
        return key
    except Exception as e:
        logger.error("Password export key file %s unusable: %s", _KEY_FILE, type(e).__name__)
        return None


def _keys() -> List[bytes]:
    """Keys in priority order: env var (if set and valid), then the key file."""
    keys = []
    env_key = os.environ.get(_ENV_VAR_NAME, "").strip()
    if env_key:
        try:
            Fernet(env_key.encode("utf-8"))
            keys.append(env_key.encode("utf-8"))
        except Exception as e:
            # Never log the key value itself, even on a malformed-key error.
            logger.error("%s is not a valid Fernet key (%s) -- using the key file instead", _ENV_VAR_NAME, type(e).__name__)
    file_key = _load_or_create_key_file()
    if file_key and file_key not in keys:
        keys.append(file_key)
    return keys


def _get_fernet() -> MultiFernet:
    """
    Encrypts with the first available key, decrypts with any of them.
    Looked up lazily (not at import time) so the app never fails to start
    because of this feature.
    """
    keys = _keys()
    if not keys:
        raise RuntimeError(
            f"No usable password export key: {_ENV_VAR_NAME} is unset/invalid and "
            f"{_KEY_FILE} could not be created or read."
        )
    return MultiFernet([Fernet(k) for k in keys])


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
    value can't be decrypted with any available key -- e.g. it was saved
    under a key that no longer exists. Re-importing or resetting that
    user's password stores it again under the current key.
    """
    if not ciphertext:
        return ""
    try:
        return _get_fernet().decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except (InvalidToken, RuntimeError):
        logger.error("Failed to decrypt a stored exportable password -- it was saved under a key that is no longer available.")
        raise ValueError(
            "Stored password could not be decrypted with the available keys. "
            "Re-import or reset this user's password to store it again."
        )


def has_encryption_key() -> bool:
    """Non-raising check for whether any usable key (env var or key file) is available."""
    return bool(_keys())
