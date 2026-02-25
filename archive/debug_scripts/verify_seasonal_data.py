"""
Verify Seasonal Report Data
Check Q4 2025 and Q1 2026 incident counts against database
"""

import pyodbc
from datetime import datetime

# Database connection
conn_str = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=SOCIALMEDIA;"
    "DATABASE=IncidentManager;"
    "Trusted_Connection=yes;"
)

def verify_season_data(season_id, season_name):
    """Verify incident counts for a season."""
    print(f"\n{'='*80}")
    print(f"VERIFYING: {season_name} (Season ID: {season_id})")
    print('='*80)
    
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    # Get season date range
    cursor.execute("""
        SELECT UniqueID, SeasonName, StartDate, EndDate
        FROM dbo.Season
        WHERE UniqueID = ?
    """, (season_id,))
    
    season = cursor.fetchone()
    if not season:
        print(f"❌ Season {season_id} not found!")
        return
    
    print(f"Season: {season.SeasonName}")
    print(f"Date Range: {season.StartDate} to {season.EndDate}")
    
    # Count total incidents in date range
    cursor.execute("""
        SELECT 
            COUNT(*) as TotalCount,
            SUM(CASE WHEN SeverityID = 1 THEN 1 ELSE 0 END) as LowCount,
            SUM(CASE WHEN SeverityID = 2 THEN 1 ELSE 0 END) as MediumCount,
            SUM(CASE WHEN SeverityID = 3 THEN 1 ELSE 0 END) as HighCount,
            SUM(CASE WHEN DomainID = 1 THEN 1 ELSE 0 END) as ClinicalCount,
            SUM(CASE WHEN DomainID = 2 THEN 1 ELSE 0 END) as ManagementCount,
            SUM(CASE WHEN DomainID = 3 THEN 1 ELSE 0 END) as RelationalCount
        FROM dbo.APP_IncidentCase
        WHERE FeedbackRecievedDate >= ?
          AND FeedbackRecievedDate <= ?
    """, (season.StartDate, season.EndDate))
    
    stats = cursor.fetchone()
    
    print(f"\n📊 INCIDENT COUNTS:")
    print(f"  Total Cases: {stats.TotalCount}")
    print(f"  Severity Breakdown:")
    print(f"    - Low: {stats.LowCount}")
    print(f"    - Medium: {stats.MediumCount}")
    print(f"    - High: {stats.HighCount}")
    print(f"  Domain Breakdown:")
    print(f"    - Clinical: {stats.ClinicalCount}")
    print(f"    - Management: {stats.ManagementCount}")
    print(f"    - Relational: {stats.RelationalCount}")
    
    # Get classification breakdown
    cursor.execute("""
        SELECT 
            ic.ClassificationID,
            ic.DomainID,
            ic.CategoryID,
            ic.SubCategoryID,
            COUNT(*) as Count
        FROM dbo.APP_IncidentCase ic
        WHERE ic.FeedbackRecievedDate >= ?
          AND ic.FeedbackRecievedDate <= ?
        GROUP BY ic.ClassificationID, ic.DomainID, ic.CategoryID, ic.SubCategoryID
        ORDER BY COUNT(*) DESC
    """, (season.StartDate, season.EndDate))
    
    classifications = cursor.fetchall()
    
    print(f"\n📋 CLASSIFICATION BREAKDOWN ({len(classifications)} unique combinations):")
    for i, cls in enumerate(classifications, 1):
        print(f"  {i}. Classification={cls.ClassificationID}, Domain={cls.DomainID}, "
              f"Category={cls.CategoryID}, SubCategory={cls.SubCategoryID}: {cls.Count} cases")
    
    # List all incident IDs
    cursor.execute("""
        SELECT 
            IncidentRequestCaseID,
            FeedbackRecievedDate,
            IssuingOrgUnitID,
            SeverityID,
            DomainID,
            ClassificationID
        FROM dbo.APP_IncidentCase
        WHERE FeedbackRecievedDate >= ?
          AND FeedbackRecievedDate <= ?
        ORDER BY FeedbackRecievedDate
    """, (season.StartDate, season.EndDate))
    
    incidents = cursor.fetchall()
    
    print(f"\n📝 DETAILED INCIDENT LIST:")
    for inc in incidents:
        print(f"  ID={inc.IncidentRequestCaseID}, Date={inc.FeedbackRecievedDate}, "
              f"OrgUnit={inc.IssuingOrgUnitID}, Severity={inc.SeverityID}, "
              f"Domain={inc.DomainID}, Classification={inc.ClassificationID}")
    
    conn.close()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SEASONAL REPORT DATA VERIFICATION")
    print("="*80)
    
    # Verify Q4 2025 (Season ID 4)
    verify_season_data(4, "Q4 2025")
    
    # Verify Q1 2026 (Season ID 5)
    verify_season_data(5, "Q1 2026")
    
    print("\n" + "="*80)
    print("✅ VERIFICATION COMPLETE")
    print("="*80 + "\n")
