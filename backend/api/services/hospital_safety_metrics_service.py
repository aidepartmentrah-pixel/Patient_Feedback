"""
Hospital Safety Metrics Service — HCAT Iteration 4, Target Analysis.

Two CONSTANT hospital-wide safety thresholds, defined directly in code —
NOT stored in APP_OrgUnitPolicy, NOT configurable via the Settings tab.
These are fixed business-policy constants, layered on top of (not
replacing) the existing database-driven domain % targets.

  1. High severity cases must not exceed 5% of total hospital cases.
  2. Cases that are BOTH High severity AND Clinical domain must not
     exceed 3% of total hospital cases.

Hospital scope only — these are whole-hospital safety signals, not
scoped to any single org unit.
"""

from datetime import date
from typing import Any, Dict

from core.database import get_connection

HIGH_SEVERITY_TARGET_PCT = 5.0
HIGH_SEVERITY_CLINICAL_TARGET_PCT = 3.0

CLINICAL_DOMAIN_ID = 1
HIGH_SEVERITY_ID = 3


def get_hospital_safety_metrics(date_from: date, date_to: date) -> Dict[str, Any]:
    """
    Compute hospital-wide (no org unit filtering) safety metrics for the
    given date range and compare against the two constant thresholds above.
    """
    sql = """
        SELECT
            COUNT(DISTINCT ic.IncidentRequestCaseID) AS total_count,
            COUNT(DISTINCT CASE WHEN ic.SeverityID = ?
                                THEN ic.IncidentRequestCaseID END) AS high_count,
            COUNT(DISTINCT CASE WHEN ic.SeverityID = ? AND ic.DomainID = ?
                                THEN ic.IncidentRequestCaseID END) AS high_clinical_count
        FROM dbo.APP_IncidentCase ic
        WHERE ic.FeedbackRecievedDate >= ?
          AND ic.FeedbackRecievedDate <= ?
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            sql,
            (HIGH_SEVERITY_ID, HIGH_SEVERITY_ID, CLINICAL_DOMAIN_ID, date_from, date_to),
        )
        row = cursor.fetchone()
        total_count         = row[0] or 0
        high_count          = row[1] or 0
        high_clinical_count = row[2] or 0
    finally:
        cursor.close()
        conn.close()

    high_pct = round((high_count / total_count) * 100, 2) if total_count > 0 else 0.0
    high_clinical_pct = round((high_clinical_count / total_count) * 100, 2) if total_count > 0 else 0.0

    return {
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "total_count": total_count,
        "high_severity": {
            "actual_count": high_count,
            "actual_pct":   high_pct,
            "target_pct":   HIGH_SEVERITY_TARGET_PCT,
            "exceeded":     high_pct > HIGH_SEVERITY_TARGET_PCT,
        },
        "high_severity_clinical": {
            "actual_count": high_clinical_count,
            "actual_pct":   high_clinical_pct,
            "target_pct":   HIGH_SEVERITY_CLINICAL_TARGET_PCT,
            "exceeded":     high_clinical_pct > HIGH_SEVERITY_CLINICAL_TARGET_PCT,
        },
    }
