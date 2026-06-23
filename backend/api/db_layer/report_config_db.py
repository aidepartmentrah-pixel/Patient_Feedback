"""
DB layer for APP_ReportConfig — institutional report header/footer metadata.
"""

from typing import Dict, Optional
from datetime import datetime
from core.database import get_connection


_VALID_KEYS = {
    "header_title", "header_subtitle", "footer_text", "report_code",
    "seasonal_header_title", "seasonal_header_subtitle",
    "seasonal_footer_text", "seasonal_report_code",
    "monthly_report_format",   # "classical" (default) | "stylish" — controls DOCX formatter for Monthly Detailed Report
}


def get_report_config() -> Dict[str, str]:
    """Return all config keys as a plain dict. Missing keys fall back to empty string."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT ConfigKey, ConfigValue FROM dbo.APP_ReportConfig")
        rows = cursor.fetchall()
        config = {row[0]: row[1] for row in rows}
        # Ensure all expected keys are present
        for key in _VALID_KEYS:
            config.setdefault(key, "")
        return config
    finally:
        cursor.close()
        conn.close()


def set_report_config(updates: Dict[str, str], user_id: Optional[int] = None) -> None:
    """
    Update one or more config keys.
    Only keys in _VALID_KEYS are accepted; others are silently ignored.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        for key, value in updates.items():
            if key not in _VALID_KEYS:
                continue
            cursor.execute(
                """
                MERGE dbo.APP_ReportConfig AS tgt
                USING (SELECT ? AS ConfigKey, ? AS ConfigValue) AS src
                ON tgt.ConfigKey = src.ConfigKey
                WHEN MATCHED THEN
                    UPDATE SET ConfigValue = src.ConfigValue,
                               UpdatedAt   = GETDATE(),
                               UpdatedBy   = ?
                WHEN NOT MATCHED THEN
                    INSERT (ConfigKey, ConfigValue, UpdatedAt, UpdatedBy)
                    VALUES (src.ConfigKey, src.ConfigValue, GETDATE(), ?);
                """,
                (key, value, user_id, user_id)
            )
        conn.commit()
    finally:
        cursor.close()
        conn.close()
