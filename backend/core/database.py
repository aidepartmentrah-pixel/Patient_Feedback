"""
Core Database Connection Module

⚠️ CANONICAL SOURCE — DO NOT DUPLICATE ⚠️

This is the ONLY module that should create SQL Server database connections.
All backend code must import get_connection from here.

Connection parameters (DB_SERVER, DB_DATABASE, etc.) are imported by value
from core/deployment_port.py, which freezes them from config/db_settings.json
(+ env var overrides) once at process import time. Editing db_settings.json
or saving new values via /api/config/save does NOT change these values for
an already-running process — restart the backend to pick up new settings.
See core/deployment_port.get_active_snapshot() to inspect the values this
process is actually using right now, and core/bootstrap.check_database_connection
to test them.
"""

import pyodbc
import logging

# Configure logging for database connections
logger = logging.getLogger(__name__)

from .deployment_port import (
    DB_SERVER,
    DB_DATABASE,
    DB_DRIVER,
    USE_WINDOWS_AUTH,
    TRUST_SERVER_CERTIFICATE,
    DB_USERNAME,
    DB_PASSWORD,
)


def get_connection():
    """
    Get SQL Server database connection for IncidentManager.
    
    This is the canonical connection factory for the entire backend.
    All db_layer, services, and routers should import this function.
    
    Returns:
        pyodbc.Connection: Active database connection
        
    Connection Parameters:
        Frozen at process startup from core/deployment_port.py:
        - SERVER: {DB_SERVER}
        - DATABASE: {DB_DATABASE}
        - DRIVER: {DB_DRIVER}
        - AUTH: Windows Trusted Connection (if USE_WINDOWS_AUTH=True)
                SQL Server Auth with UID/PWD (if USE_WINDOWS_AUTH=False)
        - SECURITY: TrustServerCertificate (if TRUST_SERVER_CERTIFICATE=True)

    Note:
        To change these for a running process, edit config/db_settings.json
        (or the env vars in core/config_loader.py) and RESTART the backend.
        Saving via /api/config/save alone is not sufficient.
    """
    # Build connection string from config
    conn_parts = [
        f"DRIVER={{{DB_DRIVER}}};",
        f"SERVER={DB_SERVER};",
        f"DATABASE={DB_DATABASE};"
    ]
    
    if USE_WINDOWS_AUTH:
        conn_parts.append("Trusted_Connection=yes;")
        auth_mode = "Windows Authentication"
    else:
        # SQL Server Authentication
        conn_parts.append(f"UID={DB_USERNAME};")
        conn_parts.append(f"PWD={DB_PASSWORD};")
        auth_mode = f"SQL Server Authentication (User: {DB_USERNAME})"
    
    if TRUST_SERVER_CERTIFICATE:
        conn_parts.append("TrustServerCertificate=yes;")
    
    conn_string = "".join(conn_parts)
    
    # Log connection attempt (without password)
    logger.info(f"Attempting database connection...")
    logger.info(f"  Server: {DB_SERVER}")
    logger.info(f"  Database: {DB_DATABASE}")
    logger.info(f"  Driver: {DB_DRIVER}")
    logger.info(f"  Auth: {auth_mode}")
    logger.info(f"  TrustServerCertificate: {TRUST_SERVER_CERTIFICATE}")
    
    try:
        conn = pyodbc.connect(conn_string, timeout=10)
        logger.info(f"✓ Database connection successful to {DB_SERVER}/{DB_DATABASE}")
        return conn
    except pyodbc.InterfaceError as e:
        # Driver not found or interface issue. NOTE: every ODBC error from a
        # driver that DID load successfully is itself prefixed with that
        # driver's own name, so a naive `"driver" in error_msg` check alone
        # would misfire on unrelated interface errors too — verify against
        # the actually-installed driver list instead of guessing from text.
        error_msg = str(e)
        logger.error(f"✗ DATABASE INTERFACE ERROR")
        logger.error(f"  Error: {error_msg}")
        try:
            installed_drivers = pyodbc.drivers()
        except Exception:
            installed_drivers = []
        if DB_DRIVER not in installed_drivers:
            logger.error(f"  DIAGNOSIS: ODBC Driver '{DB_DRIVER}' is not installed on this server")
            logger.error(f"  FIX: Install '{DB_DRIVER}' on this server (installed here: {installed_drivers or 'none detected'})")
        else:
            logger.error(f"  DIAGNOSIS: Driver '{DB_DRIVER}' is installed, but the interface call still failed — see raw error above")
        raise ConnectionError(f"Database driver error: {error_msg}") from e
    except pyodbc.OperationalError as e:
        # Connection refused, network unreachable, timeout
        error_msg = str(e)
        logger.error(f"✗ DATABASE OPERATIONAL ERROR")
        logger.error(f"  Server: {DB_SERVER}")
        logger.error(f"  Error: {error_msg}")
        if "timeout" in error_msg.lower():
            logger.error(f"  DIAGNOSIS: Connection timeout - server may be unreachable")
            logger.error(f"  FIX: Check if SQL Server is running and port 1433 is open")
        elif "connection refused" in error_msg.lower() or "cannot open" in error_msg.lower():
            logger.error(f"  DIAGNOSIS: Connection refused by server")
            logger.error(f"  FIX: Check firewall, SQL Server TCP/IP settings, and port 1433")
        elif "network" in error_msg.lower():
            logger.error(f"  DIAGNOSIS: Network unreachable")
            logger.error(f"  FIX: Check network connectivity to {DB_SERVER}")
        raise ConnectionError(f"Cannot connect to database server {DB_SERVER}: {error_msg}") from e
    except pyodbc.ProgrammingError as e:
        # Login failed, database not found
        error_msg = str(e)
        logger.error(f"✗ DATABASE PROGRAMMING ERROR")
        logger.error(f"  Error: {error_msg}")
        if "login failed" in error_msg.lower():
            logger.error(f"  DIAGNOSIS: Authentication failed")
            logger.error(f"  FIX: Check DB_USERNAME and DB_PASSWORD credentials")
            if USE_WINDOWS_AUTH:
                logger.error(f"  NOTE: Windows Auth may not work across machines")
        elif "database" in error_msg.lower() and "not" in error_msg.lower():
            logger.error(f"  DIAGNOSIS: Database '{DB_DATABASE}' not found")
            logger.error(f"  FIX: Verify database name exists on server")
        raise ConnectionError(f"Database login/access error: {error_msg}") from e
    except pyodbc.Error as e:
        # Generic pyodbc error
        error_msg = str(e)
        logger.error(f"✗ DATABASE ERROR")
        logger.error(f"  Error: {error_msg}")
        logger.error(f"  Connection string (sanitized): DRIVER={DB_DRIVER};SERVER={DB_SERVER};DATABASE={DB_DATABASE};AUTH={auth_mode}")
        raise ConnectionError(f"Database error: {error_msg}") from e
    except Exception as e:
        # Unexpected error
        error_msg = str(e)
        logger.error(f"✗ UNEXPECTED ERROR during database connection")
        logger.error(f"  Type: {type(e).__name__}")
        logger.error(f"  Error: {error_msg}")
        raise ConnectionError(f"Unexpected database error: {error_msg}") from e


