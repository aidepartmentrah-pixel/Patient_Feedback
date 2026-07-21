"""
Database Connection Diagnostics
Tests various connection configurations to identify the issue
"""

import pyodbc
import json
from pathlib import Path

def test_connection(config_name, conn_string):
    """Test a connection string and report results"""
    print(f"\n{'='*60}")
    print(f"Testing: {config_name}")
    print(f"{'='*60}")
    print(f"Connection String: {conn_string.replace('PWD=', 'PWD=***')}")
    
    try:
        conn = pyodbc.connect(conn_string, timeout=10)
        cursor = conn.cursor()
        cursor.execute("SELECT @@VERSION")
        version = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        
        print("✅ SUCCESS!")
        print(f"SQL Server Version: {version[:100]}...")
        return True
        
    except Exception as e:
        print("❌ FAILED!")
        print(f"Error: {str(e)}")
        return False

def main():
    # Load current config
    config_path = Path(__file__).parent / "config" / "db_settings.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    db = config["database"]
    server = db["server"]
    database = db["database"]
    username = db["username"]
    password = db["password"]
    
    print("="*60)
    print("DATABASE CONNECTION DIAGNOSTICS")
    print("="*60)
    print(f"Target Server: {server}")
    print(f"Target Database: {database}")
    print(f"Username: {username}")
    
    results = {}
    
    # Test 1: Current config with ODBC Driver 18 + Windows Auth
    conn_str = (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"Trusted_Connection=yes;"
        f"TrustServerCertificate=yes;"
    )
    results["Driver 18 + Windows Auth"] = test_connection("Driver 18 + Windows Auth", conn_str)
    
    # Test 2: ODBC Driver 18 + SQL Auth
    conn_str = (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        f"TrustServerCertificate=yes;"
    )
    results["Driver 18 + SQL Auth"] = test_connection("Driver 18 + SQL Auth", conn_str)
    
    # Test 3: ODBC Driver 17 + SQL Auth (if available)
    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        f"TrustServerCertificate=yes;"
    )
    results["Driver 17 + SQL Auth"] = test_connection("Driver 17 + SQL Auth", conn_str)
    
    # Test 4: ODBC Driver 17 + Windows Auth (if available)
    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"Trusted_Connection=yes;"
        f"TrustServerCertificate=yes;"
    )
    results["Driver 17 + Windows Auth"] = test_connection("Driver 17 + Windows Auth", conn_str)
    
    # Test 5: Driver 18 with Encrypt=no
    conn_str = (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        f"Encrypt=no;"
    )
    results["Driver 18 + SQL Auth + No Encryption"] = test_connection("Driver 18 + SQL Auth + No Encryption", conn_str)
    
    # Test 6: Driver 18 with explicit port
    conn_str = (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={server},1433;"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        f"TrustServerCertificate=yes;"
    )
    results["Driver 18 + SQL Auth + Explicit Port"] = test_connection("Driver 18 + SQL Auth + Explicit Port", conn_str)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    successful = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if not v]
    
    if successful:
        print(f"\n✅ Successful configurations ({len(successful)}):")
        for config in successful:
            print(f"   - {config}")
    
    if failed:
        print(f"\n❌ Failed configurations ({len(failed)}):")
        for config in failed:
            print(f"   - {config}")
    
    if successful:
        print("\n" + "="*60)
        print("RECOMMENDATION")
        print("="*60)
        print(f"Use: {successful[0]}")
        print("\nUpdate your db_settings.json accordingly.")
    else:
        print("\n⚠️  All connection attempts failed!")
        print("Possible issues:")
        print("  - SQL Server not accepting remote connections")
        print("  - Credentials are incorrect")
        print("  - SQL Server authentication not enabled")
        print("  - Named instance needs explicit port or instance name")

if __name__ == "__main__":
    main()
