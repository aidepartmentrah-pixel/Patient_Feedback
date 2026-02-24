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
from .config_loader import get_config

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
DOCTORS_VIEW = _views.get("doctors", "VW_Doctors")


# =============================================================================
# 3. NETWORK & API SETTINGS
# =============================================================================
BACKEND_API_URL = _net.get("backend_api_url", "http://localhost:8000")
BACKEND_PORT = _net.get("backend_port", 8000)
BACKEND_HOST = _net.get("backend_host", "127.0.0.1")
CORS_ORIGINS = _net.get("cors_origins", [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
])


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
│     DOCTORS_VIEW = "VW_Doctors"                                             │
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
