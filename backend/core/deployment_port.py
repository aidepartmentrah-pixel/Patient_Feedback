"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         DEPLOYMENT PORT CONFIGURATION                          ║
║═══════════════════════════════════════════════════════════════════════════════║
║  ⚠️  SINGLE POINT OF CONFIGURATION FOR OFFLINE/ONLINE DEPLOYMENT  ⚠️          ║
║                                                                                 ║
║  This is the UNIFIED configuration file for deploying the system.              ║
║  Modify ONLY this file when switching between offline and online modes.        ║
║                                                                                 ║
║  Contains:                                                                      ║
║    1. Database Connection Settings (Server, Database, Authentication)          ║
║    2. External System View/Table Names (HR, HIS, Doctors)                       ║
║    3. Network & API Settings (Backend URL, CORS)                                ║
║    4. SMTP Email Server Settings (Outlook integration)                          ║
║                                                                                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

# =============================================================================
# DEPLOYMENT MODE
# =============================================================================
# "offline"  - Local development with local database and mock tables
# "online"   - Production with hospital network and real views

DEPLOYMENT_MODE = "offline"


# =============================================================================
# 1. DATABASE CONNECTION SETTINGS
# =============================================================================

# Database server hostname or IP address
# Offline:  "SOCIALMEDIA" or "localhost" or ".\SQLEXPRESS"
# Online:   Hospital SQL Server IP (e.g., "192.168.1.100" or "HOSPITAL-DB")
DB_SERVER = "SOCIALMEDIA"

# Database name
# Offline:  "IncidentManager"
# Online:   "IncidentManager" or hospital-specified name
DB_DATABASE = "IncidentManager"

# ODBC Driver (usually no need to change)
DB_DRIVER = "ODBC Driver 17 for SQL Server"

# Authentication method
# True:  Windows domain authentication (Trusted_Connection=yes) - common in hospitals
# False: SQL Server authentication (requires DB_USERNAME and DB_PASSWORD)
USE_WINDOWS_AUTH = True

# SQL Server credentials (only used if USE_WINDOWS_AUTH = False)
DB_USERNAME = None
DB_PASSWORD = None

# Trust server certificate (required for self-signed certificates)
TRUST_SERVER_CERTIFICATE = True


# =============================================================================
# 2. EXTERNAL SYSTEM VIEW/TABLE NAMES
# =============================================================================
# These are the external hospital system views (HIS, HR).
# The system checks for both VIEWS and TABLES with these names.
# 
# Offline (development): Using local tables with production-matching names
# Online (production):   Using real hospital database views

# HR Employee System - Employee profiles from HR department
# Offline: "VW_HrEmployeeProfileView" (local table matching production name)
# Online:  "VW_HrEmployeeProfileView" (real HR view)
HR_EMPLOYEES_VIEW = "VW_HrEmployeeProfileView"

# Patient Admission System (HIS) - Patient records
# Offline: "VW_PatientAdmission" (local table matching production name)
# Online:  "dbo.VW_PatientAdmission" (real HIS view)
PATIENT_ADMISSION_VIEW = "VW_PatientAdmission"

# Doctor Registry (HIS) - Doctor information
# Offline: "VW_Doctors" (local table matching production name)
# Online:  "dbo.VW_Doctors" (real HIS view)
DOCTORS_VIEW = "VW_Doctors"


# =============================================================================
# 3. NETWORK & API SETTINGS
# =============================================================================

# Backend API URL (used by frontend to connect)
# Offline:  "http://localhost:8000"
# Online:   "http://<server-ip>:8000" or custom URL
BACKEND_API_URL = "http://localhost:8000"

# Backend API Port
BACKEND_PORT = 8000

# Backend Host (0.0.0.0 allows external connections)
# Offline:  "127.0.0.1" (localhost only)
# Online:   "0.0.0.0" (accept connections from network)
BACKEND_HOST = "127.0.0.1"

# CORS Origins (frontend URLs allowed to access API)
# Offline:  ["http://localhost:3000", "http://localhost:5173"]
# Online:   Add production frontend URLs
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]


# =============================================================================
# 4. SMTP EMAIL SERVER SETTINGS (Outlook / Exchange)
# =============================================================================

# Email notification mode
# "mock": Log emails only (development)
# "smtp": Send real emails
NOTIFICATION_MODE = "mock"

# SMTP Server Configuration (for Outlook/Exchange)
# Typical Outlook settings:
#   - Internal Exchange: "mail.hospital.local" or IP address
#   - Office 365: "smtp.office365.com"
SMTP_HOST = "smtp.hospital.local"
SMTP_PORT = 25  # Common: 25 (no auth), 587 (TLS), 465 (SSL)

# SMTP Security
SMTP_USE_TLS = False   # True for Office 365 or secure connections
SMTP_USE_SSL = False   # True for port 465

# SMTP Authentication (None for internal relay without auth)
SMTP_USERNAME = None   # e.g., "complaint-system@hospital.org"
SMTP_PASSWORD = None   # App password or email password

# Sender Information
SENDER_EMAIL = "complaint-system@hospital.local"
SENDER_NAME = "Hospital Complaint System"


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
