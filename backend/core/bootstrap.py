"""
Bootstrap Mode Detection Module

Detects whether the database is reachable at startup.
If not, the system enters BOOTSTRAP_MODE where only configuration
endpoints are available.

Usage:
    from core.bootstrap import BOOTSTRAP_MODE, check_database_connection, get_system_status
"""

import logging
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Global flag — set once at startup, updated on config reload
BOOTSTRAP_MODE: bool = False


def check_database_connection(timeout: int = 5) -> Dict[str, Any]:
    """
    Test the current database connection using settings from config_loader.

    Args:
        timeout: Connection timeout in seconds.

    Returns:
        Dict with keys: connected (bool), message (str), duration_ms (float)
    """
    start = time.time()

    try:
        import pyodbc
        from .config_loader import get_config

        config = get_config()
        db = config.get("database", {})

        # Build connection string
        conn_parts = [
            f"DRIVER={{{db.get('driver', 'ODBC Driver 17 for SQL Server')}}};"
            f"SERVER={db.get('server', 'localhost')};"
            f"DATABASE={db.get('database', 'IncidentManager')};"
        ]

        if db.get("use_windows_auth", True):
            conn_parts.append("Trusted_Connection=yes;")
        else:
            conn_parts.append(f"UID={db.get('username', '')};")
            conn_parts.append(f"PWD={db.get('password', '')};")

        if db.get("trust_server_certificate", True):
            conn_parts.append("TrustServerCertificate=yes;")

        conn_string = "".join(conn_parts)

        conn = pyodbc.connect(conn_string, timeout=timeout)
        conn.close()

        duration_ms = round((time.time() - start) * 1000, 1)
        return {
            "connected": True,
            "message": f"Connected to {db.get('server')}/{db.get('database')} in {duration_ms}ms",
            "duration_ms": duration_ms,
        }

    except ImportError:
        duration_ms = round((time.time() - start) * 1000, 1)
        return {
            "connected": False,
            "message": "pyodbc is not installed",
            "duration_ms": duration_ms,
        }
    except Exception as e:
        duration_ms = round((time.time() - start) * 1000, 1)
        error_msg = str(e)

        # Provide diagnostic hints
        if "driver" in error_msg.lower():
            hint = " — ODBC driver may not be installed"
        elif "timeout" in error_msg.lower():
            hint = " — server may be unreachable"
        elif "login failed" in error_msg.lower():
            hint = " — check username/password"
        elif "network" in error_msg.lower() or "cannot open" in error_msg.lower():
            hint = " — check network/firewall"
        else:
            hint = ""

        return {
            "connected": False,
            "message": f"Connection failed: {error_msg}{hint}",
            "duration_ms": duration_ms,
        }


def run_bootstrap_check() -> bool:
    """
    Run the bootstrap check and update BOOTSTRAP_MODE.

    Returns:
        True if database is reachable (normal mode),
        False if database is unreachable (bootstrap mode).
    """
    global BOOTSTRAP_MODE

    logger.info("=" * 60)
    logger.info("BOOTSTRAP CHECK: Testing database connection...")
    logger.info("=" * 60)

    result = check_database_connection(timeout=5)

    if result["connected"]:
        BOOTSTRAP_MODE = False
        logger.info(f"DATABASE OK: {result['message']}")
        logger.info("MODE: Normal — all endpoints active")
    else:
        BOOTSTRAP_MODE = True
        logger.warning(f"DATABASE UNREACHABLE: {result['message']}")
        logger.warning("MODE: Bootstrap — only /api/config/* and /api/status available")
        logger.warning("Configure database via /config page or edit config/db_settings.json")

    logger.info("=" * 60)
    return result["connected"]


def get_system_status() -> Dict[str, Any]:
    """
    Get current system status for the /api/status endpoint.

    Returns:
        Status dictionary.
    """
    from .config_loader import get_config, get_config_file_path

    config = get_config()
    db = config.get("database", {})

    return {
        "bootstrap_mode": BOOTSTRAP_MODE,
        "database": {
            "server": db.get("server", ""),
            "database": db.get("database", ""),
            "driver": db.get("driver", ""),
            "auth_mode": "Windows" if db.get("use_windows_auth") else "SQL Server",
        },
        "config_file": get_config_file_path(),
        "deployment_mode": config.get("deployment_mode", "unknown"),
    }


def exit_bootstrap_mode():
    """Force-exit bootstrap mode (called after successful config reload)."""
    global BOOTSTRAP_MODE
    BOOTSTRAP_MODE = False
    logger.info("Exited bootstrap mode — all endpoints now active")


def enter_bootstrap_mode():
    """Force-enter bootstrap mode."""
    global BOOTSTRAP_MODE
    BOOTSTRAP_MODE = True
    logger.warning("Entered bootstrap mode — only config endpoints available")
