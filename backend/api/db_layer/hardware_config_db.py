"""
Hardware Configuration Database Layer
Database operations for hardware/deployment configuration.
"""

from typing import Dict, List, Any, Optional
from core.database import get_connection


def get_all_configs() -> List[Dict[str, Any]]:
    """
    Fetch all hardware configuration entries.
    
    Returns:
        List of configuration dictionaries
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT 
                ConfigID,
                ConfigKey,
                ConfigValue,
                ConfigType,
                ConfigGroup,
                DisplayName,
                DisplayNameAr,
                Description,
                IsEncrypted,
                IsEditable,
                DisplayOrder,
                UpdatedAt,
                UpdatedByUserID
            FROM APP_HardwareConfig
            ORDER BY ConfigGroup, DisplayOrder, ConfigKey
        """)
        
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
        
        configs = []
        for row in rows:
            config = dict(zip(columns, row))
            # Mask encrypted values
            if config['IsEncrypted'] and config['ConfigValue']:
                config['ConfigValue'] = '********'
            configs.append(config)
        
        return configs
    finally:
        cursor.close()
        conn.close()


def get_configs_by_group(group: str) -> List[Dict[str, Any]]:
    """
    Fetch configurations by group.
    
    Args:
        group: Configuration group (database, views, network, email, system)
    
    Returns:
        List of configuration dictionaries in that group
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT 
                ConfigID,
                ConfigKey,
                ConfigValue,
                ConfigType,
                ConfigGroup,
                DisplayName,
                DisplayNameAr,
                Description,
                IsEncrypted,
                IsEditable,
                DisplayOrder
            FROM APP_HardwareConfig
            WHERE ConfigGroup = ?
            ORDER BY DisplayOrder, ConfigKey
        """, (group,))
        
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
        
        configs = []
        for row in rows:
            config = dict(zip(columns, row))
            if config['IsEncrypted'] and config['ConfigValue']:
                config['ConfigValue'] = '********'
            configs.append(config)
        
        return configs
    finally:
        cursor.close()
        conn.close()


def get_config_value(key: str, decrypt: bool = False) -> Optional[str]:
    """
    Get a single configuration value by key.
    
    Args:
        key: Configuration key
        decrypt: If True, return actual value for encrypted fields
    
    Returns:
        Configuration value or None if not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT ConfigValue, IsEncrypted
            FROM APP_HardwareConfig
            WHERE ConfigKey = ?
        """, (key,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        value, is_encrypted = row
        
        # For encrypted values, only return if decrypt=True
        if is_encrypted and not decrypt:
            return '********' if value else None
        
        return value
    finally:
        cursor.close()
        conn.close()


def get_config_with_metadata(key: str) -> Optional[Dict[str, Any]]:
    """
    Get a single configuration with full metadata.
    
    Args:
        key: Configuration key
    
    Returns:
        Configuration dictionary or None
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT 
                ConfigID,
                ConfigKey,
                ConfigValue,
                ConfigType,
                ConfigGroup,
                DisplayName,
                DisplayNameAr,
                Description,
                IsEncrypted,
                IsEditable,
                DisplayOrder,
                UpdatedAt,
                UpdatedByUserID
            FROM APP_HardwareConfig
            WHERE ConfigKey = ?
        """, (key,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        columns = [col[0] for col in cursor.description]
        config = dict(zip(columns, row))
        
        if config['IsEncrypted'] and config['ConfigValue']:
            config['ConfigValue'] = '********'
        
        return config
    finally:
        cursor.close()
        conn.close()


def update_config(key: str, value: str, user_id: int) -> bool:
    """
    Update a configuration value.
    
    Args:
        key: Configuration key
        value: New value
        user_id: ID of user making the change
    
    Returns:
        True if successful, False if key not found or not editable
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Check if editable
        cursor.execute("""
            SELECT IsEditable FROM APP_HardwareConfig WHERE ConfigKey = ?
        """, (key,))
        
        row = cursor.fetchone()
        if not row:
            return False
        
        if not row[0]:  # Not editable
            return False
        
        # Update value
        cursor.execute("""
            UPDATE APP_HardwareConfig
            SET ConfigValue = ?, UpdatedAt = GETDATE(), UpdatedByUserID = ?
            WHERE ConfigKey = ?
        """, (value, user_id, key))
        
        conn.commit()
        return cursor.rowcount > 0
    finally:
        cursor.close()
        conn.close()


def update_configs_batch(updates: List[Dict[str, str]], user_id: int) -> Dict[str, Any]:
    """
    Update multiple configuration values in a batch.
    
    Args:
        updates: List of {key: str, value: str} dictionaries
        user_id: ID of user making changes
    
    Returns:
        Dictionary with success count and errors
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    results = {
        "success_count": 0,
        "error_count": 0,
        "errors": []
    }
    
    try:
        for update in updates:
            key = update.get('key')
            value = update.get('value')
            
            if not key:
                results["errors"].append({"key": key, "error": "Key is required"})
                results["error_count"] += 1
                continue
            
            # Check if exists and editable
            cursor.execute("""
                SELECT IsEditable FROM APP_HardwareConfig WHERE ConfigKey = ?
            """, (key,))
            
            row = cursor.fetchone()
            if not row:
                results["errors"].append({"key": key, "error": "Configuration not found"})
                results["error_count"] += 1
                continue
            
            if not row[0]:
                results["errors"].append({"key": key, "error": "Configuration is not editable"})
                results["error_count"] += 1
                continue
            
            # Update
            cursor.execute("""
                UPDATE APP_HardwareConfig
                SET ConfigValue = ?, UpdatedAt = GETDATE(), UpdatedByUserID = ?
                WHERE ConfigKey = ?
            """, (value, user_id, key))
            
            results["success_count"] += 1
        
        conn.commit()
        return results
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()


def get_groups() -> List[str]:
    """Get all distinct configuration groups."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT DISTINCT ConfigGroup 
            FROM APP_HardwareConfig 
            ORDER BY ConfigGroup
        """)
        return [row[0] for row in cursor.fetchall()]
    finally:
        cursor.close()
        conn.close()


def test_database_connection(server: str, database: str, driver: str, 
                              use_windows_auth: bool, username: str = None, 
                              password: str = None) -> Dict[str, Any]:
    """
    Test a database connection with provided parameters.
    
    Returns:
        Dictionary with success status and message
    """
    import pyodbc
    
    try:
        conn_parts = [
            f"DRIVER={{{driver}}};",
            f"SERVER={server};",
            f"DATABASE={database};"
        ]
        
        if use_windows_auth:
            conn_parts.append("Trusted_Connection=yes;")
        else:
            if username and password:
                conn_parts.append(f"UID={username};")
                conn_parts.append(f"PWD={password};")
        
        conn_parts.append("TrustServerCertificate=yes;")
        conn_parts.append("Connection Timeout=5;")
        
        conn_string = "".join(conn_parts)
        
        test_conn = pyodbc.connect(conn_string)
        test_cursor = test_conn.cursor()
        test_cursor.execute("SELECT 1")
        test_cursor.close()
        test_conn.close()
        
        return {
            "success": True,
            "message": f"Successfully connected to {database} on {server}"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Connection failed: {str(e)}"
        }


def test_smtp_connection(host: str, port: int, use_tls: bool = False, 
                          use_ssl: bool = False, username: str = None, 
                          password: str = None) -> Dict[str, Any]:
    """
    Test an SMTP connection with provided parameters.
    
    Returns:
        Dictionary with success status and message
    """
    import smtplib
    import socket
    
    try:
        socket.setdefaulttimeout(5)
        
        if use_ssl:
            server = smtplib.SMTP_SSL(host, port, timeout=5)
        else:
            server = smtplib.SMTP(host, port, timeout=5)
            if use_tls:
                server.starttls()
        
        if username and password:
            server.login(username, password)
        
        server.quit()
        
        return {
            "success": True,
            "message": f"Successfully connected to SMTP server {host}:{port}"
        }
    except socket.timeout:
        return {
            "success": False,
            "message": f"Connection timeout - cannot reach {host}:{port}"
        }
    except smtplib.SMTPAuthenticationError:
        return {
            "success": False,
            "message": "SMTP authentication failed - check username/password"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"SMTP connection failed: {str(e)}"
        }
