"""
Core Database Connection Module

⚠️ CANONICAL SOURCE — DO NOT DUPLICATE ⚠️

This is the ONLY module that should create SQL Server database connections.
All backend code must import get_connection from here.

For offline deployment, modify connection parameters in db_config.py.
"""

import pyodbc
from .db_config import (
    DB_SERVER,
    DB_DATABASE,
    DB_DRIVER,
    USE_WINDOWS_AUTH,
    TRUST_SERVER_CERTIFICATE
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
    
    if TRUST_SERVER_CERTIFICATE:
        conn_parts.append("TrustServerCertificate=yes;")
    
    conn_string = "".join(conn_parts)
    conn = pyodbc.connect(conn_string)
    return conn


