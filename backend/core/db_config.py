"""
Database Configuration

⚠️ OFFLINE DEPLOYMENT — MODIFY THIS FILE ONLY ⚠️

For offline/on-premise deployment, change SERVER and DATABASE values here.
No other files need to be modified.

This is a simple Python config — no environment variables, no external dependencies.
Safe for hospital servers with restricted environments.
"""

# =============================================================================
# SQL SERVER CONNECTION PARAMETERS
# =============================================================================

# Database server hostname or IP
# Default: SOCIALMEDIA (online/development server)
# Offline: Change to "localhost" or local server name
DB_SERVER = "SOCIALMEDIA"

# Database name
# Default: IncidentManager (online/development database)
# Offline: Change to "IncidentManager_Offline" or custom name
DB_DATABASE = "IncidentManager"

# ODBC Driver (usually no need to change)
DB_DRIVER = "ODBC Driver 17 for SQL Server"

# Authentication method
# True: Use Windows domain authentication (Trusted_Connection=yes)
# False: Use SQL Server authentication (requires username/password)
USE_WINDOWS_AUTH = True

# Trust server certificate (required for self-signed certificates)
TRUST_SERVER_CERTIFICATE = True


# =============================================================================
# OFFLINE DEPLOYMENT QUICK REFERENCE
# =============================================================================
#
# To switch to offline mode:
# 1. Change DB_SERVER to your local server (e.g., "localhost" or "OFFLINE_SERVER")
# 2. Change DB_DATABASE if using different database name
# 3. Save this file
# 4. Restart backend application
#
# Example Offline Configuration:
# ----------------------------
# DB_SERVER = "localhost"
# DB_DATABASE = "IncidentManager_Offline"
#
# =============================================================================
