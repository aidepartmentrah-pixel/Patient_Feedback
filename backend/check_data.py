"""
Check which seasons have incident data.
"""

import sys
sys.path.insert(0, ".")

from api.db_layer.database import get_connection

def check_incident_data():
    """Check incident data distribution."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get all seasons
    print("\n" + "="*80)
    print("SEASONS IN DATABASE")
    print("="*80)
    
    cursor.execute("""
        SELECT UniqueID, SeasonName, StartDate, EndDate, IsDone
        FROM dbo.Season
        WHERE Frozen = 0
        ORDER BY StartDate DESC
    """)
    
    seasons = cursor.fetchall()
    for season in seasons:
        print(f"Season {season.UniqueID}: {season.SeasonName}")
        print(f"  Dates: {season.StartDate} to {season.EndDate}")
        print(f"  Is Done: {season.IsDone}")
    
    # Get incident count by date range
    print("\n" + "="*80)
    print("INCIDENTS BY DATE RANGE")
    print("="*80)
    
    cursor.execute("""
        SELECT 
            COUNT(*) as TotalIncidents,
            MIN(FeedbackRecievedDate) as EarliestDate,
            MAX(FeedbackRecievedDate) as LatestDate
        FROM dbo.APP_IncidentCase
    """)
    
    result = cursor.fetchone()
    print(f"Total Incidents: {result.TotalIncidents}")
    print(f"Date Range: {result.EarliestDate} to {result.LatestDate}")
    
    # Count incidents per season
    print("\n" + "="*80)
    print("INCIDENTS PER SEASON")
    print("="*80)
    
    for season in seasons:
        cursor.execute("""
            SELECT COUNT(*) as IncidentCount
            FROM dbo.APP_IncidentCase
            WHERE FeedbackRecievedDate >= ?
              AND FeedbackRecievedDate <= ?
        """, season.StartDate, season.EndDate)
        
        count = cursor.fetchone().IncidentCount
        print(f"Season {season.UniqueID} ({season.SeasonName}): {count} incidents")
    
    # Get recent incidents with org units
    print("\n" + "="*80)
    print("SAMPLE INCIDENTS (Last 10)")
    print("="*80)
    
    cursor.execute("""
        SELECT TOP 10
            IncidentRequestCaseID,
            FeedbackRecievedDate,
            IssuingOrgUnitID,
            DomainID,
            SeverityID
        FROM dbo.APP_IncidentCase
        ORDER BY FeedbackRecievedDate DESC
    """)
    
    incidents = cursor.fetchall()
    for inc in incidents:
        print(f"  Incident {inc.IncidentRequestCaseID}: {inc.FeedbackRecievedDate}")
        print(f"    Org Unit: {inc.IssuingOrgUnitID}, Domain: {inc.DomainID}, Severity: {inc.SeverityID}")
    
    # Check organizational unit coverage
    print("\n" + "="*80)
    print("ORGANIZATIONAL UNIT COVERAGE")
    print("="*80)
    
    cursor.execute("""
        SELECT COUNT(DISTINCT IssuingOrgUnitID) as UniqueUnits
        FROM dbo.APP_IncidentCase
        WHERE IssuingOrgUnitID IS NOT NULL
    """)
    
    unique_units = cursor.fetchone().UniqueUnits
    print(f"Incidents span {unique_units} unique organizational units")
    
    cursor.execute("""
        SELECT COUNT(*) as TotalUnits
        FROM dbo.AdminsrationUnit
        WHERE Frozen = 0
    """)
    
    total_units = cursor.fetchone().TotalUnits
    print(f"Total organizational units in database: {total_units}")
    
    conn.close()


if __name__ == "__main__":
    check_incident_data()
