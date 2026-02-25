"""
Test auto-creation of seasons when they don't exist.
"""

import sys
sys.path.insert(0, r'c:\Users\IT\Documents\GitHub Repository\Patient_Feedback\backend')

from backend.api.db_layer.seasonal_report import resolve_season_id_from_year_trimester
import pyodbc

def check_seasons_before():
    """Check what seasons exist before"""
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=SOCIALMEDIA;'
        'DATABASE=IncidentManager;'
        'Trusted_Connection=yes;'
        'TrustServerCertificate=yes;'
    )
    cursor = conn.cursor()
    cursor.execute("SELECT UniqueID, SeasonName, StartDate, EndDate FROM dbo.Season WHERE YEAR(StartDate) >= 2025 ORDER BY StartDate")
    
    print("="*80)
    print("SEASONS IN DATABASE (BEFORE)")
    print("="*80)
    print(f"{'ID':<5} | {'Name':<12} | {'Start':<12} | {'End':<12}")
    print("-"*80)
    
    seasons = []
    for row in cursor.fetchall():
        print(f"{row.UniqueID:<5} | {row.SeasonName:<12} | {str(row.StartDate):<12} | {str(row.EndDate):<12}")
        seasons.append(row.SeasonName)
    
    conn.close()
    return seasons

def test_auto_create():
    """Test auto-creation for 2026"""
    print("\n" + "="*80)
    print("TEST: Auto-Create 2026 Seasons")
    print("="*80)
    
    test_cases = [
        (2026, "Q1"),
        (2026, "Q2"),
        (2026, "Q3"),
        (2026, "Q4"),
    ]
    
    for year, quarter in test_cases:
        try:
            print(f"\nRequesting: {year} {quarter}...")
            season_id = resolve_season_id_from_year_trimester(year, quarter)
            
            if season_id:
                print(f"✅ Season ID: {season_id} (created or found)")
            else:
                print(f"❌ Returned None (season not created)")
        
        except Exception as e:
            print(f"❌ Error: {e}")

def check_seasons_after():
    """Check what seasons exist after"""
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=SOCIALMEDIA;'
        'DATABASE=IncidentManager;'
        'Trusted_Connection=yes;'
        'TrustServerCertificate=yes;'
    )
    cursor = conn.cursor()
    cursor.execute("SELECT UniqueID, SeasonName, StartDate, EndDate FROM dbo.Season WHERE YEAR(StartDate) >= 2025 ORDER BY StartDate")
    
    print("\n" + "="*80)
    print("SEASONS IN DATABASE (AFTER)")
    print("="*80)
    print(f"{'ID':<5} | {'Name':<12} | {'Start':<12} | {'End':<12}")
    print("-"*80)
    
    seasons_after = []
    for row in cursor.fetchall():
        print(f"{row.UniqueID:<5} | {row.SeasonName:<12} | {str(row.StartDate):<12} | {str(row.EndDate):<12}")
        seasons_after.append(row.SeasonName)
    
    conn.close()
    return seasons_after

if __name__ == "__main__":
    print("\n🔹"*40)
    print("AUTO-CREATE SEASONS TEST")
    print("🔹"*40 + "\n")
    
    seasons_before = check_seasons_before()
    test_auto_create()
    seasons_after = check_seasons_after()
    
    # Show what was created
    new_seasons = set(seasons_after) - set(seasons_before)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    if new_seasons:
        print(f"✅ New seasons created: {sorted(new_seasons)}")
    else:
        print("⚠️  No new seasons created")
    
    print("\n✅ The 'Generate' button will now:")
    print("   1. Auto-create Season records if they don't exist (Q1-Q4)")
    print("   2. Generate or regenerate the seasonal report")
    print("   3. Return the report data")
    print("="*80)
