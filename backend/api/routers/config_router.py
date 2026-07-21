"""
Configuration Router
Bootstrap-safe endpoints for managing system configuration.

These endpoints are protected by a static CONFIG_PASSWORD (from db_settings.json),
NOT by database authentication. This solves the chicken-and-egg problem: you can
configure the database before you can log in.

Endpoints:
    POST /api/config/verify-password           — Verify config password
    GET  /api/config/settings                  — Get current settings (password masked)
    POST /api/config/test-connection           — Test a database connection
    POST /api/config/save                      — Save settings to JSON
    POST /api/config/reload                    — Reload config and re-test DB
    GET  /api/config/drivers                   — List installed ODBC drivers
    GET  /api/status                           — System status (no password required)
    GET  /api/config/external-api              — Get Hospital Directory API settings (key masked)
    POST /api/config/external-api/save         — Save Hospital Directory API settings (applies immediately, no restart)
    POST /api/config/external-api/test-connection — Test GET {base_url}/health
    POST /api/config/database/reveal-password  — Return the real, unmasked DB password (explicit admin request)
    POST /api/config/external-api/reveal-key   — Return the real, decrypted API key (explicit admin request)
"""

import logging
import time
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core.config_loader import get_config, save_config, load_config, get_config_password
from core.bootstrap import (
    check_database_connection,
    get_system_status,
)
# Imported as a module (not `from core.bootstrap import BOOTSTRAP_MODE`) so
# that reading core_bootstrap.BOOTSTRAP_MODE below always reflects the
# current value — a `from X import NAME` binding of this module-level bool
# would snapshot it at import time and never see later reassignments,
# exactly the stale-constant bug this task exists to eliminate.
import core.bootstrap as core_bootstrap
from core.deployment_port import get_active_snapshot
from core import hospital_directory_client
from core.settings_encryption import encrypt_value, decrypt_value, has_encryption_key
from api.db_layer import external_api_settings_db

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


class ExternalApiSaveRequest(BaseModel):
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout_seconds: Optional[int] = None
    verify_tls: Optional[bool] = None
    enabled: Optional[bool] = None


class ExternalApiTestRequest(BaseModel):
    # All optional: an empty body tests the currently SAVED settings. Any
    # field provided overrides just that field for this one test, without
    # saving anything — same "test before you save" UX as the database tab.
    # api_key IS needed now: the test runs a real authenticated call
    # (GET /doctors) to actually verify the key, not just /health (which
    # per the contract needs no auth and can't validate a key at all).
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout_seconds: Optional[int] = None
    verify_tls: Optional[bool] = None


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


def _is_masked_password(new_value, original_value) -> bool:
    """
    Check if new_value is the masked version of original_value.
    If so, we should keep the original instead of saving asterisks.
    """
    if not new_value or not original_value:
        return False
    return _mask_password(original_value) == new_value


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

    Returns both the SAVED configuration (top-level database/views/network/
    email — what's in db_settings.json right now, i.e. the editable form
    values) and the ACTIVE configuration (what this running backend process
    is actually using for live traffic — see core.deployment_port). They can
    differ after a save until the backend is restarted; `config_in_sync` /
    `restart_required` summarize that for the database section, since that's
    what actually gates whether the app can talk to SQL Server.

    Header: X-Config-Password required.
    """
    _require_config_password(request)

    config = get_config()
    saved_db = config.get("database", {})

    # Mask sensitive fields
    safe_config = {
        "deployment_mode": config.get("deployment_mode", "offline"),
        "database": {
            "server": saved_db.get("server", ""),
            "database": saved_db.get("database", ""),
            "driver": saved_db.get("driver", ""),
            "use_windows_auth": saved_db.get("use_windows_auth", True),
            "username": saved_db.get("username", ""),
            "password": _mask_password(saved_db.get("password", "")),
            "trust_server_certificate": saved_db.get("trust_server_certificate", True),
        },
        "views": config.get("views", {}),
        "network": config.get("network", {}),
        "email": {
            **config.get("email", {}),
            "smtp_password": _mask_password(config.get("email", {}).get("smtp_password")),
        },
    }

    active = get_active_snapshot()
    safe_config["active"] = active  # active.database has no password key by design

    safe_config["config_in_sync"] = (
        active["database"]["server"] == saved_db.get("server")
        and active["database"]["database"] == saved_db.get("database")
        and active["database"]["driver"] == saved_db.get("driver")
        and active["database"]["use_windows_auth"] == bool(saved_db.get("use_windows_auth", True))
        and active["database"]["username"] == saved_db.get("username", "")
        and active["database"]["trust_server_certificate"] == bool(saved_db.get("trust_server_certificate", True))
    )
    safe_config["restart_required"] = not safe_config["config_in_sync"]

    return safe_config


@router.post("/api/config/test-connection")
async def test_connection(body: DatabaseTestRequest, request: Request):
    """
    Test a database connection with provided settings (without saving).

    If `password` is exactly the masked placeholder for the currently SAVED
    database password (e.g. the form was loaded and the user never retyped
    it), the real saved password is substituted before connecting. Without
    this, testing with an unchanged password field would literally try to
    authenticate with a string of asterisks and always fail with a
    misleading "Authentication failed" — the password field always arrives
    masked from GET /api/config/settings, so this is the common case, not an
    edge case.

    Header: X-Config-Password required.
    Body: Database connection parameters.
    """
    _require_config_password(request)

    import pyodbc

    saved_password = get_config().get("database", {}).get("password", "")
    password = body.password
    if _is_masked_password(password, saved_password):
        password = saved_password

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
            if password:
                conn_parts.append(f"PWD={password};")

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
        error_lower = error_msg.lower()

        # Provide diagnostic message.
        #
        # IMPORTANT: every ODBC error from a driver that DID load is itself
        # prefixed with that driver's own name, e.g.
        #   "[Microsoft][ODBC Driver 18 for SQL Server]TCP Provider: ..."
        # so a naive `"driver" in error_msg` check matches almost any
        # failure (timeout, wrong port, bad credentials, refused connection)
        # and falsely reports "driver not installed" for all of them. Check
        # the actually-installed driver list first — only claim a missing
        # driver if it's genuinely absent from it.
        try:
            installed_drivers = pyodbc.drivers()
        except Exception:
            installed_drivers = []

        if body.driver not in installed_drivers:
            diagnosis = (
                f"ODBC driver '{body.driver}' is not installed on the backend SERVER "
                f"(this VM) — it only needs to be installed there, never on the "
                f"browser/client machine running this test. Drivers currently "
                f"installed on the server: {', '.join(installed_drivers) or 'none detected'}."
            )
        elif "login failed" in error_lower:
            diagnosis = "Authentication failed. Check username and password."
        elif "timeout" in error_lower or "timed out" in error_lower:
            diagnosis = f"Connection to {body.server} timed out. Check if server is reachable."
        elif "data source name not found" in error_lower or "im002" in error_lower:
            diagnosis = (
                f"Driver '{body.driver}' is installed, but the driver name string in "
                f"this request didn't resolve to it (check for typos/extra characters)."
            )
        elif "actively refused" in error_lower or "connection refused" in error_lower:
            diagnosis = f"Connection to {body.server} was refused. Check the server address/port and firewall."
        elif "network" in error_lower or "cannot open" in error_lower or "not accessible" in error_lower:
            diagnosis = f"Cannot reach {body.server}. Check network/firewall settings and that SQL Server allows remote connections."
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

    # Track which sections had an ACTUAL value change (not just "were sent
    # in the request") so sections_changed/restart_required stay accurate —
    # e.g. resubmitting an unchanged masked password, or attempting to
    # change the immutable database name, must not falsely claim a restart
    # is now needed.
    database_changed = False
    views_changed = False
    network_changed = False
    email_changed = False

    if body.database is not None:
        db = current.get("database", {})
        original_db_password = db.get("password", "")
        for key, value in body.database.items():
            if value is not None:
                # Don't save masked password - keep original
                if key == "password" and _is_masked_password(value, original_db_password):
                    continue
                # The database NAME is not a legitimate per-deployment
                # variable for this app — the entire schema (APP_IncidentCase,
                # APP_SystemSettings, etc.) is hardcoded by name throughout
                # the codebase, so pointing at a different database would
                # just break the app. Deliberately immutable via this API;
                # only DB_DATABASE env var / direct file edit can change it.
                if key == "database" and value != db.get("database"):
                    logger.warning(
                        f"Ignored attempt to change database name via /api/config/save "
                        f"(requested: {value!r}, kept: {db.get('database')!r})"
                    )
                    continue
                if db.get(key) != value:
                    database_changed = True
                db[key] = value
        current["database"] = db

    if body.views is not None:
        views = current.get("views", {})
        for key, value in body.views.items():
            if value is not None:
                if views.get(key) != value:
                    views_changed = True
                views[key] = value
        current["views"] = views

    if body.network is not None:
        net = current.get("network", {})
        for key, value in body.network.items():
            if value is not None:
                if net.get(key) != value:
                    network_changed = True
                net[key] = value
        current["network"] = net

    if body.email is not None:
        email = current.get("email", {})
        original_smtp_password = email.get("smtp_password", "")
        for key, value in body.email.items():
            if value is not None:
                # Don't save masked password - keep original
                if key == "smtp_password" and _is_masked_password(value, original_smtp_password):
                    continue
                if email.get(key) != value:
                    email_changed = True
                email[key] = value
        current["email"] = email

    # Save
    success = save_config(current)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to write configuration file")

    # database/views/network/email are all frozen into module-level constants
    # in core.deployment_port at process import time (see get_connection(),
    # the CORS middleware, core.table_config, core.notification_config) — so
    # a change to ANY of these sections requires a backend restart before it
    # affects live traffic. /api/config/reload only refreshes the on-disk
    # cache and previews connectivity; it does not make these changes active.
    sections_changed = [
        name for name, changed in (
            ("database", database_changed),
            ("views", views_changed),
            ("network", network_changed),
            ("email", email_changed),
        )
        if changed
    ]
    restart_required = len(sections_changed) > 0

    logger.info(f"Configuration saved via /api/config/save (sections changed: {sections_changed or 'none'})")

    return {
        "saved": True,
        "sections_changed": sections_changed,
        "restart_required": restart_required,
        "message": (
            "Configuration saved to db_settings.json. A backend restart is "
            "required before these changes take effect — live database, "
            "view, network, and email traffic will keep using the previous "
            "settings until the backend process is restarted."
            if restart_required else
            "Configuration saved. No settings sections were changed."
        ),
    }


@router.post("/api/config/database/reveal-password")
async def reveal_database_password(request: Request):
    """
    Return the real, unmasked database password on explicit request.

    Unlike every other endpoint in this router, this deliberately sends a
    real secret to the browser — an accepted, explicit exception to the
    "never return decrypted/plaintext secrets to the client" rule, requested
    by the admin. The act of revealing (not the value) is logged for audit.

    Note: the database password is already stored in plaintext in
    db_settings.json (a pre-existing condition, not introduced by this
    endpoint) — this doesn't weaken storage, it just also exposes on demand
    what's already unencrypted on disk.

    Header: X-Config-Password required.
    """
    _require_config_password(request)

    config = get_config()
    password = config.get("database", {}).get("password", "")

    client_ip = request.client.host if request.client else "unknown"
    logger.warning(f"AUDIT: database password revealed via /config (client={client_ip})")

    return {"password": password}


@router.post("/api/config/reload")
async def reload_config(request: Request):
    """
    Reload the SAVED configuration from disk and test whether it is
    reachable — a preview of what would happen after a restart.

    IMPORTANT: this does NOT change what the running backend actually uses
    for live traffic, and it does NOT touch BOOTSTRAP_MODE. Live traffic goes
    through core.database.get_connection(), which uses settings frozen in
    core.deployment_port at process startup; those only change on an actual
    backend restart. An earlier version of this endpoint re-tested the saved
    settings and used the result to flip BOOTSTRAP_MODE — which meant a
    corrected save could falsely unblock the bootstrap gate while live
    queries kept failing against the still-broken active connection (or,
    symmetrically, a bad save could falsely re-lock a backend whose active
    connection was fine). Neither is safe, so this endpoint now only reports;
    it never mutates BOOTSTRAP_MODE.

    Header: X-Config-Password required.
    """
    _require_config_password(request)

    # Force reload from disk (refreshes config_loader's cache so the next
    # GET /api/config/settings reflects the latest saved values)
    config = load_config(force=True)
    logger.info("Configuration reloaded from disk (saved-settings cache refreshed)")

    saved_test = check_database_connection(timeout=5, use_saved=True)
    active = get_active_snapshot()
    saved_db = config.get("database", {})

    config_in_sync = (
        active["database"]["server"] == saved_db.get("server")
        and active["database"]["database"] == saved_db.get("database")
        and active["database"]["driver"] == saved_db.get("driver")
        and active["database"]["use_windows_auth"] == bool(saved_db.get("use_windows_auth", True))
        and active["database"]["username"] == saved_db.get("username", "")
        and active["database"]["trust_server_certificate"] == bool(saved_db.get("trust_server_certificate", True))
    )

    if saved_test["connected"] and config_in_sync:
        message = "Saved database settings are reachable and already match the active connection — no restart needed."
    elif saved_test["connected"] and not config_in_sync:
        message = (
            f"Saved database settings are reachable ({saved_test['message']}), but they differ from the "
            "connection currently active in this process. Live traffic is still using the previous "
            "settings — restart the backend to apply the saved settings."
        )
    else:
        message = (
            f"Saved database settings are NOT reachable: {saved_test['message']}. "
            "The currently active connection (used by live traffic) is unaffected by this."
        )

    return {
        "reloaded": True,
        "saved_settings_reachable": saved_test["connected"],
        "saved_settings_test": saved_test,
        "config_in_sync": config_in_sync,
        "restart_required": not config_in_sync,
        "bootstrap_mode": core_bootstrap.BOOTSTRAP_MODE,
        "message": message,
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


# ==================== HOSPITAL DIRECTORY API (EXTERNAL) ====================
#
# Runtime integration configuration for the Hospital Directory API. Stored
# in SQL Server (APP_ExternalApiSettings), not db_settings.json — this is
# deliberately separate from bootstrap database connection config: it's read
# fresh on every request, so saving here takes effect immediately, with NO
# backend restart required. See core/hospital_directory_client.py.

def _get_external_api_row_or_404() -> dict:
    row = external_api_settings_db.get_settings_row()
    if row is None:
        raise HTTPException(
            status_code=500,
            detail="APP_ExternalApiSettings has no row for 'hospital_directory' — "
                   "run sql_scripts/create_external_api_settings_table.sql.",
        )
    return row


@router.get("/api/config/external-api")
async def get_external_api_settings(request: Request):
    """
    Get current Hospital Directory API settings, API key masked.

    Header: X-Config-Password required.
    """
    _require_config_password(request)

    row = _get_external_api_row_or_404()

    decrypted_key = ""
    if row["api_key_encrypted"]:
        try:
            decrypted_key = decrypt_value(row["api_key_encrypted"])
        except ValueError as e:
            # Key exists but can't be decrypted with the current
            # SETTINGS_ENCRYPTION_KEY — surface this clearly rather than 500ing.
            logger.error(f"Could not decrypt stored Hospital Directory API key: {e}")
            return {
                "base_url": row["base_url"] or "",
                "api_key": "",
                "has_api_key": True,
                "api_key_error": str(e),
                "timeout_seconds": row["timeout_seconds"],
                "verify_tls": row["verify_tls"],
                "enabled": row["enabled"],
                "last_test_status": row["last_test_status"],
                "last_test_message": row["last_test_message"],
                "last_test_at": row["last_test_at"].isoformat() if row["last_test_at"] else None,
                "encryption_key_configured": has_encryption_key(),
            }

    return {
        "base_url": row["base_url"] or "",
        "api_key": _mask_password(decrypted_key),
        "has_api_key": bool(decrypted_key),
        "timeout_seconds": row["timeout_seconds"],
        "verify_tls": row["verify_tls"],
        "enabled": row["enabled"],
        "last_test_status": row["last_test_status"],
        "last_test_message": row["last_test_message"],
        "last_test_at": row["last_test_at"].isoformat() if row["last_test_at"] else None,
        "encryption_key_configured": has_encryption_key(),
    }


@router.post("/api/config/external-api/save")
async def save_external_api_settings(body: ExternalApiSaveRequest, request: Request):
    """
    Save Hospital Directory API settings. Applies immediately — no backend
    restart required, unlike database/views/network/email settings.

    Header: X-Config-Password required.
    """
    _require_config_password(request)

    row = _get_external_api_row_or_404()
    updates = {}

    if body.base_url is not None:
        normalized, error = hospital_directory_client.normalize_base_url(body.base_url)
        if error:
            raise HTTPException(status_code=400, detail=error)
        updates["base_url"] = normalized or None

    if body.api_key is not None:
        current_decrypted = ""
        if row["api_key_encrypted"]:
            try:
                current_decrypted = decrypt_value(row["api_key_encrypted"])
            except ValueError:
                current_decrypted = ""  # can't compare against an undecryptable key; treat any input as new
        if _is_masked_password(body.api_key, current_decrypted):
            pass  # unchanged placeholder — keep the stored key as-is
        elif body.api_key == "":
            updates["api_key_encrypted"] = None  # explicit clear
        else:
            if not has_encryption_key():
                raise HTTPException(
                    status_code=500,
                    detail="SETTINGS_ENCRYPTION_KEY is not configured on this server — cannot save an API key.",
                )
            updates["api_key_encrypted"] = encrypt_value(body.api_key)

    if body.timeout_seconds is not None:
        if not (1 <= body.timeout_seconds <= 120):
            raise HTTPException(status_code=400, detail="Timeout must be between 1 and 120 seconds.")
        updates["timeout_seconds"] = body.timeout_seconds

    if body.verify_tls is not None:
        updates["verify_tls"] = body.verify_tls

    if body.enabled is not None:
        updates["enabled"] = body.enabled

    if updates:
        external_api_settings_db.save_settings(**updates)
        logger.info(f"Hospital Directory API settings saved (fields changed: {list(updates.keys())})")

    return {
        "saved": True,
        "restart_required": False,
        "message": "Hospital Directory API settings saved and active immediately.",
    }


@router.post("/api/config/external-api/reveal-key")
async def reveal_external_api_key(request: Request):
    """
    Return the real, decrypted Hospital Directory API key on explicit
    request. Same accepted exception to the "never expose secrets to the
    browser" rule as /api/config/database/reveal-password — see that
    endpoint's docstring. The act of revealing (not the value) is logged.

    Header: X-Config-Password required.
    """
    _require_config_password(request)

    row = _get_external_api_row_or_404()
    if not row["api_key_encrypted"]:
        return {"api_key": ""}

    try:
        api_key = decrypt_value(row["api_key_encrypted"])
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    client_ip = request.client.host if request.client else "unknown"
    logger.warning(f"AUDIT: Hospital Directory API key revealed via /config (client={client_ip})")

    return {"api_key": api_key}


@router.post("/api/config/external-api/test-connection")
async def test_external_api_connection(body: ExternalApiTestRequest, request: Request):
    """
    Run the real connection test: GET /health (server reachable?) AND, only
    if that succeeds, GET /doctors with the API key (does the key actually
    authenticate?). /health alone can't answer the second question — the
    API's own contract says it needs no auth — so relying on it alone
    produced a misleading "SUCCESS" even with an invalid key. This endpoint
    fixes that by verifying both and reporting them distinctly.

    api_key resolution: if the submitted value is exactly the masked
    placeholder for the currently saved key, substitute the real saved key
    (same pattern as /api/config/external-api/save) so re-testing without
    retyping the key doesn't send literal asterisks. An empty/omitted key
    tests "no key" explicitly.

    The combined result is always persisted as the integration's "last
    test" state, visible on the settings page even after a reload.

    Header: X-Config-Password required.
    """
    _require_config_password(request)

    row = _get_external_api_row_or_404()

    api_key = body.api_key
    if api_key is not None:
        current_decrypted = ""
        if row["api_key_encrypted"]:
            try:
                current_decrypted = decrypt_value(row["api_key_encrypted"])
            except ValueError:
                current_decrypted = ""
        if _is_masked_password(api_key, current_decrypted):
            api_key = current_decrypted

    result = hospital_directory_client.verify_integration(
        base_url=body.base_url,
        api_key=api_key,
        timeout_seconds=body.timeout_seconds,
        verify_tls=body.verify_tls,
    )

    status = "SUCCESS" if result["success"] else "FAILED"
    external_api_settings_db.record_test_result(status, result["message"])

    from datetime import datetime, timezone
    return {**result, "tested_at": datetime.now(timezone.utc).isoformat()}
