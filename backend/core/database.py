"""
Core Database Connection Module

⚠️ CANONICAL SOURCE — DO NOT DUPLICATE ⚠️

This is the ONLY module that should create SQL Server database connections.
All backend code must import get_connection from here.

For offline deployment, modify connection parameters in db_config.py.
"""

import pyodbc
import logging

# Configure logging for database connections
logger = logging.getLogger(__name__)

from .db_config import (
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
        Loaded from db_config.py:
        - SERVER: {DB_SERVER}
        - DATABASE: {DB_DATABASE}
        - DRIVER: {DB_DRIVER}
        - AUTH: Windows Trusted Connection (if USE_WINDOWS_AUTH=True)
                SQL Server Auth with UID/PWD (if USE_WINDOWS_AUTH=False)
        - SECURITY: TrustServerCertificate (if TRUST_SERVER_CERTIFICATE=True)
        
    Note:
        For offline deployment, modify core/db_config.py only.
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
        # Driver not found or interface issue
        error_msg = str(e)
        logger.error(f"✗ DATABASE INTERFACE ERROR")
        logger.error(f"  Error: {error_msg}")
        if "driver" in error_msg.lower():
            logger.error(f"  DIAGNOSIS: ODBC Driver '{DB_DRIVER}' may not be installed")
            logger.error(f"  FIX: Install 'ODBC Driver 17 for SQL Server' on this machine")
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


