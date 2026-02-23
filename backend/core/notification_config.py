"""
Notification Configuration
Settings for email notification system.

⚠️ SMTP settings now from deployment_port.py ⚠️
Email templates remain in this file.
"""

# Import SMTP settings from unified deployment port
from .deployment_port import (
    NOTIFICATION_MODE,
    SMTP_HOST,
    SMTP_PORT,
    SMTP_USE_TLS,
    SMTP_USE_SSL,
    SMTP_USERNAME,
    SMTP_PASSWORD,
    SENDER_EMAIL,
    SENDER_NAME,
)


# ============================================================
# EMAIL TEMPLATES
# ============================================================

SUBCASE_ASSIGNMENT_SUBJECT = "New Case Assigned - Action Required"

SUBCASE_ASSIGNMENT_BODY_TEMPLATE = """
Dear Section Administrator,

A new case has been assigned to your department for review.

Case Reference: {case_id}
Assigned At: {assigned_at}

Please login to the Complaint Management System to review and respond.

---
This is an automated message. Please do not reply directly to this email.
Hospital Complaint Management System
"""


# ============================================================
# SAFETY SETTINGS
# ============================================================
# NEVER include patient data, complaint text, or medical info in emails

INCLUDE_PATIENT_DATA = False  # DO NOT CHANGE - compliance requirement
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
