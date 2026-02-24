"""
Configuration Router
Bootstrap-safe endpoints for managing system configuration.

These endpoints are protected by a static CONFIG_PASSWORD (from db_settings.json),
NOT by database authentication. This solves the chicken-and-egg problem: you can
configure the database before you can log in.

Endpoints:
    POST /api/config/verify-password   — Verify config password
    GET  /api/config/settings          — Get current settings (password masked)
    POST /api/config/test-connection   — Test a database connection
    POST /api/config/save              — Save settings to JSON
    POST /api/config/reload            — Reload config and re-test DB
    GET  /api/config/drivers           — List installed ODBC drivers
    GET  /api/status                   — System status (no password required)
"""

import logging
import time
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core.config_loader import get_config, save_config, load_config, get_config_password
from core.bootstrap import (
    check_database_connection,
    run_bootstrap_check,
    get_system_status,
    BOOTSTRAP_MODE,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Configuration"])

# Rate limiting for password attempts (simple in-memory)
_password_attempts: dict = {}  # ip -> (count, last_attempt_time)
MAX_ATTEMPTS = 10
LOCKOUT_SECONDS = 60


# ==================== REQUEST/RESPONSE MODELS ====================

class PasswordVerifyRequest(BaseModel):
    password: str


class DatabaseTestRequest(BaseModel):
    server: str
    database: str
    driver: str = "ODBC Driver 17 for SQL Server"
    use_windows_auth: bool = True
    username: Optional[str] = None
    password: Optional[str] = None
    trust_server_certificate: bool = True


class ConfigSaveRequest(BaseModel):
    deployment_mode: Optional[str] = None
    database: Optional[dict] = None
    views: Optional[dict] = None
    network: Optional[dict] = None
    email: Optional[dict] = None


# ==================== HELPERS ====================

def _verify_config_password(request: Request) -> bool:
    """Check X-Config-Password header against stored password."""
    provided = request.headers.get("X-Config-Password", "")
    expected = get_config_password()

    if not provided:
        return False

    # Simple rate limiting
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    if client_ip in _password_attempts:
        count, last_time = _password_attempts[client_ip]
        if now - last_time < LOCKOUT_SECONDS and count >= MAX_ATTEMPTS:
            raise HTTPException(
                status_code=429,
                detail="Too many password attempts. Try again later."
            )
        if now - last_time >= LOCKOUT_SECONDS:
            _password_attempts[client_ip] = (0, now)

    if provided != expected:
        # Track failed attempt
        if client_ip in _password_attempts:
            count, last_time = _password_attempts[client_ip]
            _password_attempts[client_ip] = (count + 1, now)
        else:
            _password_attempts[client_ip] = (1, now)
        return False

    # Successful — reset counter
    _password_attempts.pop(client_ip, None)
    return True


def _require_config_password(request: Request):
    """Raise 401 if config password is invalid."""
    if not _verify_config_password(request):
        raise HTTPException(
            status_code=401,
            detail="Invalid configuration password"
        )


def _mask_password(value) -> str:
    """Mask a password string for safe display."""
    if not value or value is None:
        return ""
    s = str(value)
    if len(s) <= 2:
        return "**"
    return s[0] + "*" * (len(s) - 2) + s[-1]


# ==================== ENDPOINTS ====================

@router.post("/api/config/verify-password")
async def verify_password(body: PasswordVerifyRequest, request: Request):
    """
    Verify the configuration password.

    Body: {"password": "..."}
    Returns: {"valid": true/false}
    """
    expected = get_config_password()

    # Rate limiting
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    if client_ip in _password_attempts:
        count, last_time = _password_attempts[client_ip]
        if now - last_time < LOCKOUT_SECONDS and count >= MAX_ATTEMPTS:
            raise HTTPException(
                status_code=429,
                detail="Too many password attempts. Try again later."
            )
        if now - last_time >= LOCKOUT_SECONDS:
            _password_attempts[client_ip] = (0, now)

    if body.password == expected:
        _password_attempts.pop(client_ip, None)
        return {"valid": True}
    else:
        if client_ip in _password_attempts:
            count, _ = _password_attempts[client_ip]
            _password_attempts[client_ip] = (count + 1, now)
        else:
            _password_attempts[client_ip] = (1, now)
        return {"valid": False}


@router.get("/api/config/settings")
async def get_settings(request: Request):
    """
    Get current configuration settings with passwords masked.

    Header: X-Config-Password required.
    """
    _require_config_password(request)

    config = get_config()

    # Mask sensitive fields
    safe_config = {
        "deployment_mode": config.get("deployment_mode", "offline"),
        "database": {
            "server": config.get("database", {}).get("server", ""),
            "database": config.get("database", {}).get("database", ""),
            "driver": config.get("database", {}).get("driver", ""),
            "use_windows_auth": config.get("database", {}).get("use_windows_auth", True),
            "username": config.get("database", {}).get("username", ""),
            "password": _mask_password(config.get("database", {}).get("password", "")),
            "trust_server_certificate": config.get("database", {}).get("trust_server_certificate", True),
        },
        "views": config.get("views", {}),
        "network": config.get("network", {}),
        "email": {
            **config.get("email", {}),
            "smtp_password": _mask_password(config.get("email", {}).get("smtp_password")),
        },
    }

    return safe_config


@router.post("/api/config/test-connection")
async def test_connection(body: DatabaseTestRequest, request: Request):
    """
    Test a database connection with provided settings (without saving).

    Header: X-Config-Password required.
    Body: Database connection parameters.
    """
    _require_config_password(request)

    import pyodbc

    start = time.time()
    try:
        conn_parts = [
            f"DRIVER={{{body.driver}}};",
            f"SERVER={body.server};",
            f"DATABASE={body.database};",
        ]

        if body.use_windows_auth:
            conn_parts.append("Trusted_Connection=yes;")
        else:
            if body.username:
                conn_parts.append(f"UID={body.username};")
            if body.password:
                conn_parts.append(f"PWD={body.password};")

        if body.trust_server_certificate:
            conn_parts.append("TrustServerCertificate=yes;")

        conn_string = "".join(conn_parts)
        conn = pyodbc.connect(conn_string, timeout=10)

        # Get server version for extra info
        cursor = conn.cursor()
        cursor.execute("SELECT @@VERSION")
        version = cursor.fetchone()[0].split("\n")[0] if cursor.rowcount != 0 else "Unknown"
        conn.close()

        duration_ms = round((time.time() - start) * 1000, 1)
        return {
            "success": True,
            "message": f"Connected successfully to {body.server}/{body.database}",
            "duration_ms": duration_ms,
            "server_version": version,
        }

    except Exception as e:
        duration_ms = round((time.time() - start) * 1000, 1)
        error_msg = str(e)

        # Provide diagnostic message
        if "driver" in error_msg.lower():
            diagnosis = f"ODBC driver '{body.driver}' may not be installed on this machine."
        elif "login failed" in error_msg.lower():
            diagnosis = "Authentication failed. Check username and password."
        elif "timeout" in error_msg.lower():
            diagnosis = f"Connection to {body.server} timed out. Check if server is reachable."
        elif "network" in error_msg.lower() or "cannot open" in error_msg.lower():
            diagnosis = f"Cannot reach {body.server}. Check network/firewall settings."
        else:
            diagnosis = error_msg

        return {
            "success": False,
            "message": diagnosis,
            "duration_ms": duration_ms,
            "raw_error": error_msg[:500],
        }


@router.post("/api/config/save")
async def save_settings(body: ConfigSaveRequest, request: Request):
    """
    Save configuration to db_settings.json.

    Header: X-Config-Password required.
    Body: Partial or full config to merge.
    """
    _require_config_password(request)

    # Load current config
    current = load_config(force=True)

    # Merge provided fields
    if body.deployment_mode is not None:
        current["deployment_mode"] = body.deployment_mode

    if body.database is not None:
        db = current.get("database", {})
        for key, value in body.database.items():
            if value is not None:
                db[key] = value
        current["database"] = db

    if body.views is not None:
        views = current.get("views", {})
        for key, value in body.views.items():
            if value is not None:
                views[key] = value
        current["views"] = views

    if body.network is not None:
        net = current.get("network", {})
        for key, value in body.network.items():
            if value is not None:
                net[key] = value
        current["network"] = net

    if body.email is not None:
        email = current.get("email", {})
        for key, value in body.email.items():
            if value is not None:
                email[key] = value
        current["email"] = email

    # Save
    success = save_config(current)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to write configuration file")

    logger.info("Configuration saved via /api/config/save")

    return {
        "saved": True,
        "message": "Configuration saved. Use /api/config/reload to apply changes.",
        "restart_required": False,
    }


@router.post("/api/config/reload")
async def reload_config(request: Request):
    """
    Reload configuration from disk and re-test database connection.

    Header: X-Config-Password required.
    """
    _require_config_password(request)

    # Force reload from disk
    config = load_config(force=True)
    logger.info("Configuration reloaded from disk")

    # Re-test database connection
    db_ok = run_bootstrap_check()

    return {
        "reloaded": True,
        "database_connected": db_ok,
        "bootstrap_mode": not db_ok,
        "message": "Normal mode — all endpoints active" if db_ok
                   else "Bootstrap mode — database unreachable, only config endpoints available",
    }


@router.get("/api/config/drivers")
async def list_drivers(request: Request):
    """
    List installed ODBC drivers on this machine.

    Header: X-Config-Password required.
    """
    _require_config_password(request)

    try:
        import pyodbc
        drivers = pyodbc.drivers()
        sql_drivers = [d for d in drivers if "sql" in d.lower()]
        return {
            "drivers": drivers,
            "sql_server_drivers": sql_drivers,
            "recommended": sql_drivers[0] if sql_drivers else None,
        }
    except ImportError:
        return {
            "drivers": [],
            "sql_server_drivers": [],
            "recommended": None,
            "error": "pyodbc is not installed",
        }


@router.get("/api/status")
async def system_status():
    """
    Get system status. NO password required.

    Returns bootstrap_mode flag and basic database info.
    """
    return get_system_status()
