#!/usr/bin/env python3
"""Comprehensive database connection test"""

import sys
import json
import pyodbc

# Load configuration
with open('backend/config/db_settings.json') as f:
    config = json.load(f)

db_config = config['database']

# Build connection string
conn_str = (
    f"Driver={{{db_config['driver']}}};"+
    f"Server=tcp:{db_config['host']},{db_config['port']};"+
    f"Database={db_config['database']};"+
    f"UID={db_config['username']};"+
    f"PWD={db_config['password']};"+
    f"Encrypt=yes;"+
    f"TrustServerCertificate=yes;"
)

print("=" * 60)
print("DATABASE CONNECTION TEST")
print("=" * 60)
print(f"\nServer: {db_config['host']}:{db_config['port']}")
print(f"Database: {db_config['database']}")
print(f"Driver: {db_config['driver']}\n")

try:
    print("Connecting...")
    conn = pyodbc.connect(conn_str, timeout=10)
    cursor = conn.cursor()
    
    # Test 1: Count tables
    cursor.execute('''
        SELECT COUNT(*) as table_count FROM information_schema.tables 
        WHERE table_type = 'BASE TABLE'
    ''')
    result = cursor.fetchone()
    table_count = result[0]
    print(f"✓ PASS: Found {table_count} tables")
    
    # Test 2: List sample tables
    cursor.execute('SELECT TOP 10 name FROM sys.tables ORDER BY name')
    tables = cursor.fetchall()
    print(f"✓ PASS: Sample tables:")
    for t in tables:
        print(f"       - {t[0]}")
    
    # Test 3: Check lookups
    try:
        cursor.execute('SELECT COUNT(*) FROM APP_LOOKUP_DOMAIN')
        count = cursor.fetchone()[0]
        print(f"✓ PASS: Lookup domains: {count}")
    except:
        print("⚠ INFO: Lookup table not accessible (may require permissions)")
    
    # Test 4: Get server info
    cursor.execute('SELECT @@SERVERNAME AS ServerName, @@VERSION AS Version')
    row = cursor.fetchone()
    print(f"✓ PASS: Server: {row[0]}")
    
    conn.close()
    
    print("\n" + "=" * 60)
    print("✓✓✓ DATABASE CONNECTION TEST PASSED ✓✓✓")
    print("=" * 60 + "\n")
    sys.exit(0)
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
