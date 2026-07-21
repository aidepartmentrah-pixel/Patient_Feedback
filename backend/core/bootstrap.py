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


def check_database_connection(timeout: int = 5, use_saved: bool = False) -> Dict[str, Any]:
    """
    Test a database connection.

    Args:
        timeout: Connection timeout in seconds.
        use_saved: If True, test the settings currently SAVED in
            db_settings.json (config_loader.get_config()). If False
            (default), test the settings currently ACTIVE in this running
            process (core.deployment_port — frozen at process import time).
            The active settings are what core.database.get_connection()
            actually uses for live query traffic; the saved settings are
            only a preview of what would be used after a backend restart.

    Returns:
        Dict with keys: connected (bool), message (str), duration_ms (float), source (str)
    """
    start = time.time()
    source = "saved" if use_saved else "active"

    try:
        import pyodbc

        if use_saved:
            from .config_loader import get_config
            config = get_config()
            db = config.get("database", {})
        else:
            from . import deployment_port as dp
            db = dp.get_active_snapshot()["database"]
            # get_active_snapshot() omits the password by design; pull it
            # back in here only, for the purpose of actually connecting.
            db = {**db, "password": dp.DB_PASSWORD}

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
            "source": source,
        }

    except ImportError:
        duration_ms = round((time.time() - start) * 1000, 1)
        return {
            "connected": False,
            "message": "pyodbc is not installed",
            "duration_ms": duration_ms,
            "source": source,
        }
    except Exception as e:
        duration_ms = round((time.time() - start) * 1000, 1)
        error_msg = str(e)
        error_lower = error_msg.lower()

        # Provide diagnostic hints. NOTE: every ODBC error from a driver that
        # DID load is itself prefixed with that driver's own name (e.g.
        # "[Microsoft][ODBC Driver 18 for SQL Server]..."), so a naive
        # `"driver" in error_msg` check matches almost any failure — check
        # the actually-installed driver list first instead of guessing.
        try:
            import pyodbc as _pyodbc
            installed_drivers = _pyodbc.drivers()
        except Exception:
            installed_drivers = []
        requested_driver = db.get("driver", "")

        if requested_driver and requested_driver not in installed_drivers:
            hint = f" — ODBC driver '{requested_driver}' is not installed on this server"
        elif "timeout" in error_lower or "timed out" in error_lower:
            hint = " — server may be unreachable"
        elif "login failed" in error_lower:
            hint = " — check username/password"
        elif "network" in error_lower or "cannot open" in error_lower or "refused" in error_lower:
            hint = " — check network/firewall"
        else:
            hint = ""

        return {
            "connected": False,
            "message": f"Connection failed: {error_msg}{hint}",
            "duration_ms": duration_ms,
            "source": source,
        }


def run_bootstrap_check() -> bool:
    """
    Run the bootstrap check and update BOOTSTRAP_MODE.

    Deliberately tests the ACTIVE (frozen-at-startup) database settings, not
    whatever happens to be saved on disk at the moment this runs. BOOTSTRAP_MODE
    gates real API traffic, and real API traffic always goes through the
    active connection (core.database.get_connection()) — so the gate must be
    driven by the same settings, or a bad save could either falsely lock the
    app out of a working DB, or (worse) falsely let traffic through onto a
    connection that doesn't actually work. See /api/config/reload for testing
    saved-but-not-yet-active settings without affecting this flag.

    Returns:
        True if the active database is reachable (normal mode),
        False if the active database is unreachable (bootstrap mode).
    """
    global BOOTSTRAP_MODE

    logger.info("=" * 60)
    logger.info("BOOTSTRAP CHECK: Testing ACTIVE database connection...")
    logger.info("=" * 60)

    result = check_database_connection(timeout=5, use_saved=False)

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

    Reports the ACTIVE database settings (what live traffic actually uses),
    not the saved-on-disk config — those can differ as soon as someone saves
    new settings without restarting. `config_in_sync` / `restart_required`
    tell the caller whether a pending restart is needed to pick up a save.

    Returns:
        Status dictionary.
    """
    from .config_loader import get_config, get_config_file_path
    from . import deployment_port as dp

    config = get_config()
    saved_db = config.get("database", {})
    active_db = dp.get_active_snapshot()["database"]

    config_in_sync = (
        saved_db.get("server") == active_db["server"]
        and saved_db.get("database") == active_db["database"]
        and saved_db.get("driver") == active_db["driver"]
        and bool(saved_db.get("use_windows_auth", True)) == bool(active_db["use_windows_auth"])
        and saved_db.get("username", "") == active_db["username"]
        and bool(saved_db.get("trust_server_certificate", True)) == bool(active_db["trust_server_certificate"])
    )

    return {
        "bootstrap_mode": BOOTSTRAP_MODE,
        "database": {
            "server": active_db["server"],
            "database": active_db["database"],
            "driver": active_db["driver"],
            "auth_mode": "Windows" if active_db["use_windows_auth"] else "SQL Server",
            # BOOTSTRAP_MODE is derived from testing this exact active
            # connection (see run_bootstrap_check), so its inverse is an
            # accurate, cheap proxy for "is the active DB reachable" without
            # re-testing on every /api/status poll.
            "connected": not BOOTSTRAP_MODE,
        },
        "config_in_sync": config_in_sync,
        "restart_required": not config_in_sync,
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
