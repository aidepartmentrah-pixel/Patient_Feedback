"""
Notification Configuration
Settings for email notification system.

This module controls email notification behavior.
Change NOTIFICATION_MODE to "smtp" when ready to send real emails.
"""

# ============================================================
# NOTIFICATION MODE
# ============================================================
# Options:
#   "mock"  - Log notifications only (for development/testing)
#   "smtp"  - Send real emails via SMTP
#
NOTIFICATION_MODE = "mock"


# ============================================================
# SENDER CONFIGURATION
# ============================================================
# The email address and display name used when sending notifications

SENDER_EMAIL = "complaint-system@hospital.local"
SENDER_NAME = "Hospital Complaint System"


# ============================================================
# SMTP SERVER CONFIGURATION
# ============================================================
# Only used when NOTIFICATION_MODE = "smtp"

SMTP_HOST = "smtp.hospital.local"
SMTP_PORT = 25
SMTP_USE_TLS = False  # Set True if your server requires TLS
SMTP_USERNAME = None  # None for unauthenticated relay
SMTP_PASSWORD = None  # None for unauthenticated relay


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
