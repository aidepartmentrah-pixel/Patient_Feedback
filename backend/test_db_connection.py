"""
Database Connection Diagnostic Script

Run this script to test database connectivity from VM to external SQL Server.
Usage: python test_db_connection.py

This will test:
1. Network connectivity (ping)
2. Port 1433 accessibility
3. ODBC Driver availability
4. SQL Server authentication
5. Database access
"""

import socket
import subprocess
import sys

# Add parent directory to path for imports
sys.path.insert(0, '.')

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")

def print_result(test_name, success, message=""):
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"  {status}: {test_name}")
    if message:
        print(f"         {message}")

def test_network_ping(server_ip):
    """Test basic network connectivity"""
    print_header("1. Testing Network Connectivity (Ping)")
    try:
        # Windows ping with 2 packets, 2 second timeout
        result = subprocess.run(
            ["ping", "-n", "2", "-w", "2000", server_ip],
            capture_output=True,
            text=True,
            timeout=10
        )
        success = result.returncode == 0
        if success:
            print_result("Ping to " + server_ip, True, "Server is reachable")
        else:
            print_result("Ping to " + server_ip, False, "Server not responding to ping (may be blocked)")
        return success
    except Exception as e:
        print_result("Ping test", False, str(e))
        return False

def test_port_1433(server_ip, port=1433):
    """Test if SQL Server port is open"""
    print_header(f"2. Testing Port {port} Accessibility")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((server_ip, port))
        sock.close()
        
        if result == 0:
            print_result(f"Port {port} on {server_ip}", True, "Port is OPEN")
            return True
        else:
            print_result(f"Port {port} on {server_ip}", False, 
                        f"Port is CLOSED or filtered (error code: {result})")
            print("         FIX: Enable TCP/IP in SQL Server Configuration Manager")
            print("         FIX: Open port 1433 in Windows Firewall on the laptop")
            return False
    except socket.timeout:
        print_result(f"Port {port}", False, "Connection TIMEOUT")
        print("         FIX: Check firewall and SQL Server is running")
        return False
    except Exception as e:
        print_result(f"Port {port}", False, str(e))
        return False

def test_odbc_driver():
    """Check if ODBC Driver is installed"""
    print_header("3. Checking ODBC Driver")
    try:
        import pyodbc
        drivers = pyodbc.drivers()
        
        print(f"  Installed ODBC drivers:")
        for d in drivers:
            print(f"    - {d}")
        
        if "ODBC Driver 17 for SQL Server" in drivers:
            print_result("ODBC Driver 17", True, "Driver is installed")
            return True
        elif "ODBC Driver 18 for SQL Server" in drivers:
            print_result("ODBC Driver 18", True, "Driver 18 found (may need config update)")
            return True
        else:
            print_result("ODBC Driver 17", False, "Driver NOT found")
            print("         FIX: Download and install from:")
            print("         https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server")
            return False
    except ImportError:
        print_result("pyodbc module", False, "pyodbc not installed")
        print("         FIX: pip install pyodbc")
        return False
    except Exception as e:
        print_result("ODBC check", False, str(e))
        return False

def test_sql_connection():
    """Test actual SQL Server connection"""
    print_header("4. Testing SQL Server Connection")
    
    try:
        from core.deployment_port import (
            DB_SERVER, DB_DATABASE, DB_DRIVER,
            USE_WINDOWS_AUTH, DB_USERNAME, DB_PASSWORD,
            TRUST_SERVER_CERTIFICATE
        )
        
        print(f"  Configuration loaded:")
        print(f"    Server:   {DB_SERVER}")
        print(f"    Database: {DB_DATABASE}")
        print(f"    Driver:   {DB_DRIVER}")
        print(f"    Auth:     {'Windows' if USE_WINDOWS_AUTH else 'SQL Server'}")
        if not USE_WINDOWS_AUTH:
            print(f"    Username: {DB_USERNAME}")
            print(f"    Password: {'(empty)' if not DB_PASSWORD else '****'}")
        print(f"    TrustCert: {TRUST_SERVER_CERTIFICATE}")
        
    except Exception as e:
        print_result("Config load", False, str(e))
        return False
    
    # Import pyodbc first, separately
    try:
        import pyodbc
    except ImportError as e:
        print_result("Connection", False, "pyodbc module not found")
        print("         FIX: pip install pyodbc")
        return False
    
    try:
        # Build connection string
        conn_parts = [
            f"DRIVER={{{DB_DRIVER}}};",
            f"SERVER={DB_SERVER};",
            f"DATABASE={DB_DATABASE};"
        ]
        
        if USE_WINDOWS_AUTH:
            conn_parts.append("Trusted_Connection=yes;")
        else:
            conn_parts.append(f"UID={DB_USERNAME};")
            conn_parts.append(f"PWD={DB_PASSWORD};")
        
        if TRUST_SERVER_CERTIFICATE:
            conn_parts.append("TrustServerCertificate=yes;")
        
        conn_string = "".join(conn_parts)
        
        print(f"\n  Attempting connection...")
        conn = pyodbc.connect(conn_string, timeout=10)
        
        # Test a simple query
        cursor = conn.cursor()
        cursor.execute("SELECT @@VERSION")
        version = cursor.fetchone()[0]
        
        print_result("SQL Server Connection", True)
        print(f"\n  Server Version:")
        print(f"    {version[:80]}...")
        
        # Test database access
        cursor.execute("SELECT DB_NAME()")
        db_name = cursor.fetchone()[0]
        print_result(f"Connected to database: {db_name}", True)
        
        conn.close()
        return True
        
    except pyodbc.InterfaceError as e:
        print_result("Connection", False, "DRIVER ERROR")
        print(f"         {e}")
        print("         FIX: Install ODBC Driver 17 for SQL Server")
        return False
    except pyodbc.OperationalError as e:
        print_result("Connection", False, "OPERATIONAL ERROR")
        print(f"         {e}")
        error_str = str(e).lower()
        if "timeout" in error_str:
            print("         DIAGNOSIS: Connection timeout")
            print("         FIX: Check firewall and SQL Server TCP/IP")
        elif "cannot open" in error_str or "connection refused" in error_str:
            print("         DIAGNOSIS: Connection refused")
            print("         FIX: Enable remote connections in SQL Server")
        return False
    except pyodbc.ProgrammingError as e:
        print_result("Connection", False, "LOGIN/ACCESS ERROR")
        print(f"         {e}")
        error_str = str(e).lower()
        if "login failed" in error_str:
            print("         DIAGNOSIS: Authentication failed")
            print("         FIX: Check username/password or SQL Server auth mode")
        return False
    except Exception as e:
        print_result("Connection", False, f"{type(e).__name__}: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("  DATABASE CONNECTION DIAGNOSTIC")
    print("  VM → External SQL Server")
    print("="*60)
    
    # Get server IP from config
    try:
        from core.deployment_port import DB_SERVER
        server_ip = DB_SERVER
    except:
        server_ip = "170.70.32.36"  # fallback
    
    print(f"\n  Target Server: {server_ip}")
    
    # Run tests
    results = []
    
    results.append(("Network Ping", test_network_ping(server_ip)))
    results.append(("Port 1433", test_port_1433(server_ip)))
    results.append(("ODBC Driver", test_odbc_driver()))
    results.append(("SQL Connection", test_sql_connection()))
    
    # Summary
    print_header("SUMMARY")
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
    
    print(f"\n  Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All tests passed! Database connection is working.")
    else:
        print("\n  ⚠️  Some tests failed. Fix the issues above before running the backend.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
