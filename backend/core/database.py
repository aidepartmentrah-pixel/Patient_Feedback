"""
Core Database Connection Module

⚠️ CANONICAL SOURCE — DO NOT DUPLICATE ⚠️

This is the ONLY module that should create SQL Server database connections.
All backend code must import get_connection from here.

For offline deployment, modify connection parameters in config/db_settings.json.

Connection Protocol:
    This module uses EXPLICIT TCP connections (tcp:<host>,<port>) to avoid
    protocol ambiguity. The ODBC driver will NOT attempt Named Pipes, which
    eliminates intermittent connection failures caused by Named Pipes timeouts.

Retry Logic:
    A lightweight retry mechanism (2 retries, 0.5s delay) handles transient
    network hiccups without masking persistent configuration errors.
"""

import pyodbc
import logging
import time

# Configure logging for database connections
logger = logging.getLogger(__name__)

# Retry configuration for transient connection failures
CONNECTION_MAX_RETRIES = 2
CONNECTION_RETRY_DELAY_SECONDS = 0.5

from .db_config import (
    DB_HOST,
    DB_PORT,
    DB_SERVER,
    DB_DATABASE,
    DB_DRIVER,
    USE_WINDOWS_AUTH,
    TRUST_SERVER_CERTIFICATE,
    DB_USERNAME,
    DB_PASSWORD,
)


def _create_connection(conn_string: str, auth_mode: str):
    """
    Internal function to create a single database connection.
    
    Args:
        conn_string: ODBC connection string
        auth_mode: Description of auth mode for logging
        
    Returns:
        pyodbc.Connection: Active database connection
        
    Raises:
        Various pyodbc exceptions on failure
    """
    try:
        conn = pyodbc.connect(conn_string, timeout=10)
        logger.info(f"Database connection successful to {DB_SERVER}/{DB_DATABASE}")
        return conn
    except pyodbc.InterfaceError as e:
        # Driver not found or interface issue - NOT retryable
        error_msg = str(e)
        logger.error(f"DATABASE INTERFACE ERROR")
        logger.error(f"  Error: {error_msg}")
        if "driver" in error_msg.lower():
            logger.error(f"  DIAGNOSIS: ODBC Driver '{DB_DRIVER}' may not be installed")
            logger.error(f"  FIX: Install 'ODBC Driver 17 for SQL Server' on this machine")
        raise ConnectionError(f"Database driver error: {error_msg}") from e
    except pyodbc.ProgrammingError as e:
        # Login failed, database not found - NOT retryable (config problem)
        error_msg = str(e)
        logger.error(f"DATABASE PROGRAMMING ERROR")
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
    except pyodbc.OperationalError as e:
        # Connection refused, network unreachable, timeout - MAY be retryable
        error_msg = str(e)
        logger.warning(f"DATABASE OPERATIONAL ERROR (may retry)")
        logger.warning(f"  Server: {DB_SERVER}")
        logger.warning(f"  Error: {error_msg}")
        if "timeout" in error_msg.lower():
            logger.warning(f"  DIAGNOSIS: Connection timeout - server may be temporarily unreachable")
        elif "connection refused" in error_msg.lower() or "cannot open" in error_msg.lower():
            logger.warning(f"  DIAGNOSIS: Connection refused by server")
        elif "network" in error_msg.lower():
            logger.warning(f"  DIAGNOSIS: Network unreachable")
        # Re-raise for retry logic to handle
        raise
    except pyodbc.Error as e:
        # Generic pyodbc error
        error_msg = str(e)
        logger.error(f"DATABASE ERROR")
        logger.error(f"  Error: {error_msg}")
        logger.error(f"  Connection string (sanitized): DRIVER={DB_DRIVER};SERVER={DB_SERVER};DATABASE={DB_DATABASE};AUTH={auth_mode}")
        raise ConnectionError(f"Database error: {error_msg}") from e


def get_connection():
    """
    Get SQL Server database connection for IncidentManager.
    
    This is the canonical connection factory for the entire backend.
    All db_layer, services, and routers should import this function.
    
    Connection Protocol:
        Uses explicit TCP connections (tcp:<host>,<port>) when configured
        with host/port fields. This eliminates Named Pipes protocol ambiguity
        and ensures deterministic connection routing.
    
    Retry Logic:
        Operational errors (timeouts, transient network issues) are retried
        up to CONNECTION_MAX_RETRIES times with CONNECTION_RETRY_DELAY_SECONDS
        delay between attempts. Configuration errors (bad credentials, missing
        database) are NOT retried.
    
    Returns:
        pyodbc.Connection: Active database connection
        
    Connection Parameters (from config/db_settings.json):
        - host: Database server hostname/IP (preferred)
        - port: SQL Server port (default 1433)
        - server: Legacy field (used if host not specified)
        - database: Database name
        - driver: ODBC driver name
        - use_windows_auth: Use Windows Integrated Auth
        - username/password: SQL Server Auth credentials
        - trust_server_certificate: Skip certificate validation
        
    Raises:
        ConnectionError: If connection fails after all retry attempts
    """
    # Build connection string from config
    conn_parts = [
        f"DRIVER={{{DB_DRIVER}}};",
        f"SERVER={DB_SERVER};",  # DB_SERVER is now built as tcp:<host>,<port> when host is configured
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
    # Only log on first attempt to avoid log spam during retries
    logger.info(f"Attempting database connection...")
    logger.info(f"  Target: {DB_SERVER}")
    if DB_HOST:
        logger.info(f"  Protocol: Explicit TCP (host={DB_HOST}, port={DB_PORT})")
    else:
        logger.info(f"  Protocol: Default (legacy 'server' field - may use Named Pipes)")
    logger.info(f"  Database: {DB_DATABASE}")
    logger.info(f"  Driver: {DB_DRIVER}")
    logger.info(f"  Auth: {auth_mode}")
    
    # Retry loop for transient operational errors
    last_exception = None
    for attempt in range(CONNECTION_MAX_RETRIES + 1):
        try:
            return _create_connection(conn_string, auth_mode)
        except (pyodbc.OperationalError, pyodbc.Error) as e:
            last_exception = e
            if attempt < CONNECTION_MAX_RETRIES:
                logger.warning(f"Connection attempt {attempt + 1} failed, retrying in {CONNECTION_RETRY_DELAY_SECONDS}s...")
                time.sleep(CONNECTION_RETRY_DELAY_SECONDS)
            else:
                logger.error(f"All {CONNECTION_MAX_RETRIES + 1} connection attempts failed")
                logger.error(f"  Server: {DB_SERVER}")
                logger.error(f"  FIX: Check if SQL Server is running, TCP/IP is enabled, port {DB_PORT} is open")
        except ConnectionError:
            # Non-retryable errors (driver, auth, config problems)
            raise
    
    # All retries exhausted
    error_msg = str(last_exception) if last_exception else "Unknown connection error"
    raise ConnectionError(f"Cannot connect to database server {DB_SERVER} after {CONNECTION_MAX_RETRIES + 1} attempts: {error_msg}")


