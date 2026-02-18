"""
External System Table Name Configuration

⚠️ OFFLINE/ONLINE SWITCH — MODIFY THIS FILE ONLY ⚠️

These 3 tables imitate external hospital system views (HIS, HR).
When switching from offline (development) to online (production),
change the table names here to match the real view names.

No other files need to be modified.

This is the ONLY file that maps external system table names.
See also: db_config.py for connection parameters.
"""

# =============================================================================
# EXTERNAL SYSTEM TABLE NAMES
# =============================================================================

# HR Employee System
# Offline (development): "APP_VIEWTABLE_HR_EMPLOYEES"
# Online (production):   Change to real HR view name (e.g. "VW_HR_EMPLOYEES")
HR_EMPLOYEES_TABLE = "APP_VIEWTABLE_HR_EMPLOYEES"

# Patient Admission System (HIS)
# Offline (development): "APP_VIEWTABLE_PATIENT_ADMISSION"
# Online (production):   Change to real HIS view name (e.g. "VW_PATIENT_ADMISSION")
PATIENT_ADMISSION_TABLE = "APP_VIEWTABLE_PATIENT_ADMISSION"

# Doctor Registry (HIS)
# Offline (development): "APP_VIEWTABLE_VW_DOCTORS"
# Online (production):   Change to real HIS view name (e.g. "VW_DOCTORS")
DOCTORS_TABLE = "APP_VIEWTABLE_VW_DOCTORS"


# =============================================================================
# OFFLINE/ONLINE SWITCH QUICK REFERENCE
# =============================================================================
#
# To switch to online mode:
# 1. Change HR_EMPLOYEES_TABLE to real HR view name
# 2. Change PATIENT_ADMISSION_TABLE to real HIS patient view name
# 3. Change DOCTORS_TABLE to real HIS doctor view name
# 4. Also update db_config.py (DB_SERVER, DB_DATABASE)
# 5. Restart backend application
#
# To switch back to offline mode:
# 1. Restore values to APP_VIEWTABLE_* names
# 2. Restore db_config.py
# 3. Restart backend application
#
# =============================================================================
