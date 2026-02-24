"""
Database Configuration

⚠️ DEPRECATED — Use deployment_port.py instead ⚠️

This file now imports from deployment_port.py for backward compatibility.
All configuration should be done in deployment_port.py.
"""

# Import from unified deployment port (single source of truth)
from .deployment_port import (
    DB_SERVER,
    DB_DATABASE,
    DB_DRIVER,
    USE_WINDOWS_AUTH,
    TRUST_SERVER_CERTIFICATE,
    DB_USERNAME,
    DB_PASSWORD,
)
