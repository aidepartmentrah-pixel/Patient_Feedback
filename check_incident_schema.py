"""Quick check for APP_IncidentCase table schema"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

import pyodbc
from core.database import get_connection

def check_incident_case_schema():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get schema information
    query = """
    SELECT 
        COLUMN_NAME,
        DATA_TYPE,
        CHARACTER_MAXIMUM_LENGTH,
        IS_NULLABLE
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = 'APP_IncidentCase'
    AND TABLE_SCHEMA = 'dbo'
    ORDER BY ORDINAL_POSITION
    """
    
    cursor.execute(query)
    
    print("=" * 80)
    print("APP_IncidentCase TABLE SCHEMA")
    print("=" * 80)
    print(f"{'Column Name':<40} {'Data Type':<20} {'Length':<10} {'Nullable':<10}")
    print("=" * 80)
    
    for row in cursor.fetchall():
        col_name = row.COLUMN_NAME
        data_type = row.DATA_TYPE
        length = row.CHARACTER_MAXIMUM_LENGTH or ''
        nullable = row.IS_NULLABLE
        print(f"{col_name:<40} {data_type:<20} {str(length):<10} {nullable:<10}")
    
    cursor.close()
    conn.close()
    print("=" * 80)

if __name__ == "__main__":
    check_incident_case_schema()
