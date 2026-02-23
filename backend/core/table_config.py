"""
External System Table Name Configuration

⚠️ DEPRECATED — Use deployment_port.py instead ⚠️

This file now imports from deployment_port.py for backward compatibility.
All configuration should be done in deployment_port.py.
"""

# Import from unified deployment port (single source of truth)
from .deployment_port import (
    HR_EMPLOYEES_VIEW as HR_EMPLOYEES_TABLE,
    PATIENT_ADMISSION_VIEW as PATIENT_ADMISSION_TABLE,
    DOCTORS_VIEW as DOCTORS_TABLE,
)
