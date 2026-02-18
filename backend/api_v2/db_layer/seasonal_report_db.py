"""
Seasonal Report Database Helper for API V2
Helper functions for querying seasonal report data needed by case creation service.
"""

from typing import List
from core.database import get_connection


def get_target_orgunits_for_seasonal_report(seasonal_report_id: int) -> List[int]:
    """
    Get target org unit IDs for a seasonal report.
    
    Args:
        seasonal_report_id: The seasonal report ID
        
    Returns:
        List of OrgUnitID values
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            SELECT OrgUnitID
            FROM dbo.APP_SeasonalOrgUnitReport
            WHERE SeasonalReportID = ?
        """
        
        cursor.execute(query, (seasonal_report_id,))
        rows = cursor.fetchall()
        
        return [row.OrgUnitID for row in rows]
    
    finally:
        cursor.close()
        conn.close()
