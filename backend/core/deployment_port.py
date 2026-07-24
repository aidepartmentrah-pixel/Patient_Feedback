"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         DEPLOYMENT PORT CONFIGURATION                          ║
║═══════════════════════════════════════════════════════════════════════════════║
║  ⚠️  SINGLE POINT OF CONFIGURATION FOR OFFLINE/ONLINE DEPLOYMENT  ⚠️          ║
║                                                                                 ║
║  Configuration is loaded from config/db_settings.json via config_loader.py.    ║
║  Environment variables still override JSON values.                              ║
║  Edit the JSON via the /config page (password protected) or manually.          ║
║                                                                                 ║
║  Contains:                                                                      ║
║    1. Database Connection Settings (Server, Database, Authentication)          ║
║    2. External System View/Table Names (HR, HIS, Doctors)                       ║
║    3. Network & API Settings (Backend URL, CORS)                                ║
║    4. SMTP Email Server Settings (Outlook integration)                          ║
║                                                                                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import socket
from .config_loader import get_config


# =============================================================================
# AUTO-DETECT LOCAL IP ADDRESS
# =============================================================================
def _get_local_ip() -> str:
    """
    Auto-detect this machine's IP address.
    
    This allows the backend to automatically configure CORS for the current
    VM IP, so clients can connect without manual config changes when IP changes.
    """
    try:
        # Create a socket and connect to an external address
        # This doesn't actually send data, just determines the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(2)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        # Fallback: try hostname resolution
        try:
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            if ip and not ip.startswith("127."):
                return ip
        except Exception:
            pass
        return "127.0.0.1"


# Auto-detected IP (computed once at module load)
AUTO_DETECTED_IP = _get_local_ip()

# Load configuration from JSON + env overrides
_cfg = get_config()
_db = _cfg.get("database", {})
_net = _cfg.get("network", {})
_views = _cfg.get("views", {})
_email = _cfg.get("email", {})

# =============================================================================
# DEPLOYMENT MODE
# =============================================================================
DEPLOYMENT_MODE = _cfg.get("deployment_mode", "offline")


# =============================================================================
# 1. DATABASE CONNECTION SETTINGS
# =============================================================================
DB_SERVER = _db.get("server", "localhost")
DB_DATABASE = _db.get("database", "IncidentManager")
DB_DRIVER = _db.get("driver", "ODBC Driver 17 for SQL Server")
USE_WINDOWS_AUTH = _db.get("use_windows_auth", True)
DB_USERNAME = _db.get("username", "")
DB_PASSWORD = _db.get("password", "")
TRUST_SERVER_CERTIFICATE = _db.get("trust_server_certificate", True)


# =============================================================================
# 2. EXTERNAL SYSTEM VIEW/TABLE NAMES
# =============================================================================
HR_EMPLOYEES_VIEW = _views.get("hr_employees", "VW_HrEmployeeProfileView")
PATIENT_ADMISSION_VIEW = _views.get("patient_admission", "VW_PatientAdmission")
# VW_Doctors is an obsolete view intentionally NOT created on fresh installs
# (see database/sqlserver/install/002_create_schema.sql) -- APP_LOOKUP_DOCTOR
# is the real table that IS created, mirroring HR_EMPLOYEES_VIEW's pattern of
# pointing at an object that actually exists on a fresh database.
DOCTORS_VIEW = _views.get("doctors", "APP_LOOKUP_DOCTOR")


# =============================================================================
# 3. NETWORK & API SETTINGS
# =============================================================================
BACKEND_PORT = _net.get("backend_port", 8000)
# Always bind to 0.0.0.0 to accept connections from any IP (not just localhost)
BACKEND_HOST = "0.0.0.0"
# Auto-generate backend URL using detected IP
BACKEND_API_URL = f"http://{AUTO_DETECTED_IP}:{BACKEND_PORT}"

# Build CORS origins: start with config values, then add auto-detected IP
_configured_cors = _net.get("cors_origins", [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
])

# Auto-add origins for the detected IP — both HTTP and HTTPS, with and without port
_auto_cors = [
    f"http://{AUTO_DETECTED_IP}",
    f"http://{AUTO_DETECTED_IP}:80",
    f"http://{AUTO_DETECTED_IP}:3000",
    f"http://{AUTO_DETECTED_IP}:5173",
    f"https://{AUTO_DETECTED_IP}",
    f"https://{AUTO_DETECTED_IP}:443",
    "http://localhost",
    "http://localhost:80",
    "https://localhost",
    "https://localhost:443",
]

# Also auto-add the machine hostname variants
try:
    _hostname = socket.gethostname().lower()
    _auto_cors += [
        f"http://{_hostname}",
        f"https://{_hostname}",
    ]
except Exception:
    pass

# Combine configured + auto-detected, removing duplicates
CORS_ORIGINS = list(dict.fromkeys(_configured_cors + _auto_cors))


# =============================================================================
# 4. SMTP EMAIL SERVER SETTINGS (Outlook / Exchange)
# =============================================================================
NOTIFICATION_MODE = _email.get("notification_mode", "mock")
SMTP_HOST = _email.get("smtp_host", "smtp.hospital.local")
SMTP_PORT = _email.get("smtp_port", 25)
SMTP_USE_TLS = _email.get("smtp_use_tls", False)
SMTP_USE_SSL = _email.get("smtp_use_ssl", False)
SMTP_USERNAME = _email.get("smtp_username", None)
SMTP_PASSWORD = _email.get("smtp_password", None)
SENDER_EMAIL = _email.get("sender_email", "complaint-system@hospital.local")
SENDER_NAME = _email.get("sender_name", "Hospital Complaint System")


# =============================================================================
# 5. LEGACY COMPATIBILITY EXPORTS
# =============================================================================
# These aliases maintain backward compatibility with existing imports
# from db_config.py and table_config.py

# Legacy table_config.py names (for backward compatibility)
HR_EMPLOYEES_TABLE = HR_EMPLOYEES_VIEW
PATIENT_ADMISSION_TABLE = PATIENT_ADMISSION_VIEW
DOCTORS_TABLE = DOCTORS_VIEW


# =============================================================================
# OFFLINE/ONLINE QUICK SWITCH REFERENCE
# =============================================================================
"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                        QUICK SWITCH: OFFLINE → ONLINE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Change DEPLOYMENT_MODE = "online"                                       │
│                                                                             │
│  2. Database:                                                               │
│     DB_SERVER = "192.168.X.X"  (hospital server IP)                         │
│     DB_DATABASE = "IncidentManager"                                         │
│                                                                             │
│  3. Views (usually no change needed if offline tables match):               │
│     HR_EMPLOYEES_VIEW = "VW_HrEmployeeProfileView"                          │
│     PATIENT_ADMISSION_VIEW = "VW_PatientAdmission"                          │
│     DOCTORS_VIEW = "APP_LOOKUP_DOCTOR"                                      │
│                                                                             │
│  4. Network:                                                                │
│     BACKEND_API_URL = "http://192.168.X.X:8000"                             │
│     BACKEND_HOST = "0.0.0.0"                                                │
│     CORS_ORIGINS = [..., "http://192.168.X.X:3000"]                         │
│                                                                             │
│  5. Email (optional):                                                       │
│     NOTIFICATION_MODE = "smtp"                                              │
│     SMTP_HOST = "mail.hospital.local"                                       │
│                                                                             │
│  6. Restart backend: uvicorn main:app --host 0.0.0.0 --port 8000            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_connection_string() -> str:
    """Build SQL Server connection string from config."""
    conn_parts = [
        f"DRIVER={{{DB_DRIVER}}};",
        f"SERVER={DB_SERVER};",
        f"DATABASE={DB_DATABASE};"
    ]
    
    if USE_WINDOWS_AUTH:
        conn_parts.append("Trusted_Connection=yes;")
    else:
        if DB_USERNAME and DB_PASSWORD:
            conn_parts.append(f"UID={DB_USERNAME};")
            conn_parts.append(f"PWD={DB_PASSWORD};")
    
    if TRUST_SERVER_CERTIFICATE:
        conn_parts.append("TrustServerCertificate=yes;")
    
    return "".join(conn_parts)


def is_online_mode() -> bool:
    """Check if running in online/production mode."""
    return DEPLOYMENT_MODE == "online"


def get_cors_origins() -> list:
    """Get CORS origins for the current deployment mode."""
    return CORS_ORIGINS.copy()


def get_active_snapshot() -> dict:
    """
    Return the configuration values currently ACTIVE in this running process.

    These are the module-level constants above, frozen once at process import
    time — the same values core.database.get_connection(), the CORS
    middleware, core.table_config, and core.notification_config actually use
    for live traffic right now. They will NOT reflect changes saved via
    /api/config/save until the backend process is restarted (deployment_port
    is only ever imported once per process).

    This is deliberately distinct from core.config_loader.get_config(), which
    returns the currently SAVED configuration on disk and can differ from
    this snapshot as soon as someone saves new settings without restarting.

    The database password is intentionally omitted — active credentials are
    never returned by any API response.
    """
    return {
        "database": {
            "server": DB_SERVER,
            "database": DB_DATABASE,
            "driver": DB_DRIVER,
            "use_windows_auth": USE_WINDOWS_AUTH,
            "username": DB_USERNAME,
            "trust_server_certificate": TRUST_SERVER_CERTIFICATE,
        },
        "views": {
            "hr_employees": HR_EMPLOYEES_VIEW,
            "patient_admission": PATIENT_ADMISSION_VIEW,
            "doctors": DOCTORS_VIEW,
        },
        "network": {
            "backend_port": BACKEND_PORT,
            "backend_host": BACKEND_HOST,
            "cors_origins": CORS_ORIGINS,
        },
        "email": {
            "notification_mode": NOTIFICATION_MODE,
            "smtp_host": SMTP_HOST,
            "smtp_port": SMTP_PORT,
            "smtp_use_tls": SMTP_USE_TLS,
            "smtp_use_ssl": SMTP_USE_SSL,
            "smtp_username": SMTP_USERNAME,
            "sender_email": SENDER_EMAIL,
            "sender_name": SENDER_NAME,
        },
    }
