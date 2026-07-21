"""
DB layer for APP_ExternalApiSettings — runtime configuration for external
REST API integrations (currently just the Hospital Directory API), stored in
SQL Server and read fresh on every request (no process-lifetime caching, no
restart required to apply changes — unlike the bootstrap DB connection
settings in db_settings.json, see core/database.py for that distinction).

Single row per integration, identified by IntegrationName. This module never
encrypts/decrypts — that's core/settings_encryption.py's job — it only
stores and retrieves whatever ciphertext it's given.
"""

from typing import Optional

from core.database import get_connection

HOSPITAL_DIRECTORY = "hospital_directory"

_COLUMNS = (
    "BaseUrl", "ApiKeyEncrypted", "TimeoutSeconds", "VerifyTls", "Enabled",
    "LastTestStatus", "LastTestMessage", "LastTestAt", "UpdatedAt",
)


def get_settings_row(integration_name: str = HOSPITAL_DIRECTORY) -> Optional[dict]:
    """Return the settings row as a dict, or None if it hasn't been seeded (migration not run)."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM APP_ExternalApiSettings WHERE IntegrationName = ?",
            integration_name,
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "base_url": row.BaseUrl,
            "api_key_encrypted": row.ApiKeyEncrypted,
            "timeout_seconds": row.TimeoutSeconds,
            "verify_tls": bool(row.VerifyTls),
            "enabled": bool(row.Enabled),
            "last_test_status": row.LastTestStatus,
            "last_test_message": row.LastTestMessage,
            "last_test_at": row.LastTestAt,
            "updated_at": row.UpdatedAt,
        }
    finally:
        conn.close()


def save_settings(integration_name: str = HOSPITAL_DIRECTORY, **fields) -> None:
    """
    Update only the provided fields (dynamic UPDATE). Accepts any of:
    base_url, api_key_encrypted, timeout_seconds, verify_tls, enabled.

    Passing no fields is a no-op (no UPDATE executed).
    """
    column_map = {
        "base_url": "BaseUrl",
        "api_key_encrypted": "ApiKeyEncrypted",
        "timeout_seconds": "TimeoutSeconds",
        "verify_tls": "VerifyTls",
        "enabled": "Enabled",
    }
    set_clauses = []
    values = []
    for key, value in fields.items():
        if key not in column_map:
            raise ValueError(f"Unknown external API settings field: {key}")
        set_clauses.append(f"{column_map[key]} = ?")
        values.append(value)

    if not set_clauses:
        return

    set_clauses.append("UpdatedAt = GETDATE()")
    values.append(integration_name)

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            f"UPDATE APP_ExternalApiSettings SET {', '.join(set_clauses)} WHERE IntegrationName = ?",
            *values,
        )
        conn.commit()
    finally:
        conn.close()


def record_test_result(status: str, message: str, integration_name: str = HOSPITAL_DIRECTORY) -> None:
    """Persist the outcome of the most recent connectivity test (SUCCESS/FAILED + message + server timestamp)."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE APP_ExternalApiSettings "
            "SET LastTestStatus = ?, LastTestMessage = ?, LastTestAt = GETDATE() "
            "WHERE IntegrationName = ?",
            status, message[:1000] if message else message, integration_name,
        )
        conn.commit()
    finally:
        conn.close()
