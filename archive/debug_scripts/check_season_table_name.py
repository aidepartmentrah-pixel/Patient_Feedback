"""Quick check of Season table name"""
from core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

# Try both table names
try:
    cursor.execute("SELECT TOP 1 * FROM dbo.Season")
    print("✅ dbo.Season exists")
except:
    print("❌ dbo.Season does NOT exist")

try:
    cursor.execute("SELECT TOP 1 * FROM dbo.APP_LOOKUP_SEASON")
    print("✅ dbo.APP_LOOKUP_SEASON exists")
except:
    print("❌ dbo.APP_LOOKUP_SEASON does NOT exist")

conn.close()
