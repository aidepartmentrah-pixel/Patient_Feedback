"""
Seasonal Report Explanation Service

Handles submission and update of explanations for seasonal reports.
"""

from datetime import datetime
from core.database import get_connection


class SeasonalReportExplanationService:
    def __init__(self):
        pass

    # ============================================================
    # Submit explanation (first time only)
    # ============================================================
    def submit_explanation(self, seasonal_report_id: int, explanation_text: str, submitted_by_user_id: int):
        conn = get_connection()
        cursor = conn.cursor()

        try:
            # 1. Check report exists
            cursor.execute(
                """
                SELECT SeasonalReportID, ExplanationText
                FROM dbo.APP_SeasonalReport
                WHERE SeasonalReportID = ?
                """,
                seasonal_report_id
            )
            row = cursor.fetchone()

            if not row:
                raise ValueError("Seasonal report not found")

            # 2. Check explanation does not already exist
            existing_text = row[1]
            if existing_text is not None and str(existing_text).strip() != "":
                raise ValueError("Explanation already exists")

            # 3. Insert explanation
            cursor.execute(
                """
                UPDATE dbo.APP_SeasonalReport
                SET
                    ExplanationText = ?,
                    ExplanationSubmittedAt = ?,
                    ExplanationStatusID = ?,   -- 2 = Responded (adjust if needed)
                    ExplanationSubmittedByUserID = ?
                WHERE SeasonalReportID = ?
                """,
                explanation_text,
                datetime.now(),
                2,  # 2 = Responded (CHANGE if your lookup is different)
                submitted_by_user_id,
                seasonal_report_id
            )

            conn.commit()

        finally:
            conn.close()

    # ============================================================
    # Update explanation (already exists)
    # ============================================================
    def update_explanation(self, seasonal_report_id: int, explanation_text: str, submitted_by_user_id: int):
        conn = get_connection()
        cursor = conn.cursor()

        try:
            # 1. Check report exists
            cursor.execute(
                """
                SELECT SeasonalReportID, ExplanationText
                FROM dbo.APP_SeasonalReport
                WHERE SeasonalReportID = ?
                """,
                seasonal_report_id
            )
            row = cursor.fetchone()

            if not row:
                raise ValueError("Seasonal report not found")

            # 2. Check explanation exists
            existing_text = row[1]
            if existing_text is None or str(existing_text).strip() == "":
                raise ValueError("Explanation does not exist yet")

            # 3. Update explanation
            cursor.execute(
                """
                UPDATE dbo.APP_SeasonalReport
                SET
                    ExplanationText = ?,
                    ExplanationSubmittedAt = ?,
                    ExplanationStatusID = ?,   -- still Responded
                    ExplanationSubmittedByUserID = ?
                WHERE SeasonalReportID = ?
                """,
                explanation_text,
                datetime.now(),
                2,  # Responded
                submitted_by_user_id,
                seasonal_report_id
            )

            conn.commit()

        finally:
            conn.close()
