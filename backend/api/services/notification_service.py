"""
Notification Service
Handles email notifications for the complaint management system.

Supports two modes:
- "mock": Logs what would be sent (for development/testing)
- "smtp": Sends real emails via SMTP server

Configuration via: backend/config/notification_config.py
"""

import smtplib
import logging
from email.message import EmailMessage
from datetime import datetime
from typing import Optional
import threading

# Import configuration
from core.notification_config import (
    NOTIFICATION_MODE,
    SENDER_EMAIL,
    SENDER_NAME,
    SMTP_HOST,
    SMTP_PORT,
    SMTP_USE_TLS,
    SMTP_USERNAME,
    SMTP_PASSWORD,
    SUBCASE_ASSIGNMENT_SUBJECT,
    SUBCASE_ASSIGNMENT_BODY_TEMPLATE,
    PUBLICATION_SUMMARY_SUBJECT,
    PUBLICATION_SUMMARY_BODY_TEMPLATE,
    MAX_RETRIES,
    RETRY_DELAY_SECONDS,
)

# Configure logger
logger = logging.getLogger(__name__)


def send_notification(
    to_email: str,
    subject: str,
    body: str,
    run_async: bool = True
) -> bool:
    """
    Send an email notification.
    
    Args:
        to_email: Recipient email address
        subject: Email subject line
        body: Email body text
        run_async: If True, send in background thread (default: True)
        
    Returns:
        bool: True if sent/queued successfully, False otherwise
        
    Note:
        In mock mode, always returns True and logs what would be sent.
        In smtp mode, returns actual send result.
    """
    if not to_email:
        logger.warning("NOTIFICATION: Cannot send - no recipient email provided")
        return False
    
    if NOTIFICATION_MODE == "mock":
        # Mock mode - log only
        logger.info(f"[MOCK] Would send email:")
        logger.info(f"[MOCK]   To: {to_email}")
        logger.info(f"[MOCK]   Subject: {subject}")
        logger.info(f"[MOCK]   Body preview: {body[:100]}...")
        return True
    
    elif NOTIFICATION_MODE == "smtp":
        if run_async:
            # Run in background thread to not block request
            thread = threading.Thread(
                target=_send_via_smtp_with_retry,
                args=(to_email, subject, body),
                daemon=True
            )
            thread.start()
            logger.info(f"NOTIFICATION: Queued email to {to_email} for async delivery")
            return True
        else:
            return _send_via_smtp_with_retry(to_email, subject, body)
    
    else:
        logger.error(f"NOTIFICATION: Unknown mode '{NOTIFICATION_MODE}'")
        return False


def _send_via_smtp_with_retry(
    to_email: str,
    subject: str,
    body: str
) -> bool:
    """
    Send email via SMTP with retry logic.
    
    Args:
        to_email: Recipient email address
        subject: Email subject line
        body: Email body text
        
    Returns:
        bool: True if sent successfully, False otherwise
    """
    import time
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = _send_via_smtp(to_email, subject, body)
            if result:
                logger.info(f"NOTIFICATION: Email sent to {to_email} on attempt {attempt}")
                return True
        except Exception as e:
            logger.warning(
                f"NOTIFICATION: Attempt {attempt}/{MAX_RETRIES} failed for {to_email}: {str(e)}"
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
    
    logger.error(f"NOTIFICATION: Failed to send email to {to_email} after {MAX_RETRIES} attempts")
    return False


def _send_via_smtp(
    to_email: str,
    subject: str,
    body: str
) -> bool:
    """
    Send email via SMTP server (internal function).
    
    Args:
        to_email: Recipient email address
        subject: Email subject line
        body: Email body text
        
    Returns:
        bool: True if sent successfully
        
    Raises:
        Exception: If SMTP connection or send fails
    """
    msg = EmailMessage()
    msg["From"] = f"{SENDER_NAME} <{SENDER_EMAIL}>"
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)
    
    try:
        if SMTP_USE_TLS:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                server.starttls()
                if SMTP_USERNAME and SMTP_PASSWORD:
                    server.login(SMTP_USERNAME, SMTP_PASSWORD)
                server.send_message(msg)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                if SMTP_USERNAME and SMTP_PASSWORD:
                    server.login(SMTP_USERNAME, SMTP_PASSWORD)
                server.send_message(msg)
        
        return True
        
    except Exception as e:
        logger.error(f"NOTIFICATION: SMTP error sending to {to_email}: {str(e)}")
        raise


def send_subcase_assignment_notification(
    to_email: str,
    case_id: int,
    assigned_at: Optional[datetime] = None
) -> bool:
    """
    Send notification when a subcase is assigned to a department.
    
    Args:
        to_email: Recipient email address (section admin)
        case_id: The subcase ID or reference
        assigned_at: When the case was assigned (defaults to now)
        
    Returns:
        bool: True if sent/queued successfully, False otherwise
        
    Note:
        This function uses the template from config.
        Does NOT include patient data or complaint text (compliance).
    """
    if not to_email:
        logger.info(f"NOTIFICATION: No email for case {case_id} - skipping notification")
        return False
    
    assigned_at = assigned_at or datetime.now()
    
    subject = SUBCASE_ASSIGNMENT_SUBJECT
    body = SUBCASE_ASSIGNMENT_BODY_TEMPLATE.format(
        case_id=case_id,
        assigned_at=assigned_at.strftime("%Y-%m-%d %H:%M")
    )
    
    return send_notification(to_email, subject, body)


def send_publication_summary_notifications(
    subcases: list
) -> None:
    """
    Send one summary email per unique admin after a publication batch.

    Groups the created subcases by target_org_unit_id, resolves the admin
    email for each org unit, consolidates by email address (two org units
    may share the same admin), and sends one email per unique recipient
    with the total count of cases directed at them.

    If subcases is empty (publication created nothing), nothing is sent.

    Args:
        subcases: list of {"subcase_id": int, "target_org_unit_id": int}
                  as returned by create_subcases_for_incident().
    """
    if not subcases:
        logger.debug("PUBLICATION NOTIFY: no subcases created, skipping notification")
        return

    # Map org_unit_id → count
    count_by_org: dict = {}
    for sc in subcases:
        oid = sc.get("target_org_unit_id")
        if oid is not None:
            count_by_org[oid] = count_by_org.get(oid, 0) + 1

    # Resolve admin email per org unit, consolidate by email address
    count_by_email: dict = {}
    for org_unit_id, count in count_by_org.items():
        email = get_section_admin_email(org_unit_id)
        if email:
            count_by_email[email] = count_by_email.get(email, 0) + count
        else:
            logger.debug(
                f"PUBLICATION NOTIFY: no admin email for org_unit {org_unit_id}, skipping"
            )

    if not count_by_email:
        logger.info("PUBLICATION NOTIFY: no admin emails found for any org unit, nothing sent")
        return

    for email, count in count_by_email.items():
        subject = PUBLICATION_SUMMARY_SUBJECT
        body = PUBLICATION_SUMMARY_BODY_TEMPLATE.format(count=count)
        send_notification(email, subject, body)
        logger.info(
            f"PUBLICATION NOTIFY: queued summary email to {email} — {count} case(s)"
        )


def get_section_admin_email(org_unit_id: int) -> Optional[str]:
    """
    Look up email address for the admin of a given org unit.
    
    Args:
        org_unit_id: The organization unit ID (section/department)
        
    Returns:
        Email address if found, None otherwise
        
    Note:
        Queries APP_Users + APP_UserRoleScope to find admin assigned to this unit.
    """
    from core.database import get_connection
    
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Find user with SECTION_ADMIN role for this org unit
        query = """
            SELECT TOP 1 u.Email
            FROM dbo.APP_Users u
            INNER JOIN dbo.APP_UserRoleScope urs ON u.UserID = urs.UserID
            INNER JOIN dbo.APP_Roles r ON urs.RoleID = r.RoleID
            WHERE urs.OrgUnitID = ?
              AND r.RoleCode = 'SECTION_ADMIN'
              AND u.Email IS NOT NULL
              AND u.Email != ''
              AND u.IsActive = 1
        """
        
        cursor.execute(query, (org_unit_id,))
        row = cursor.fetchone()
        
        if row and row[0]:
            return row[0]
        
        return None
        
    except Exception as e:
        logger.error(f"NOTIFICATION: Failed to get admin email for org_unit {org_unit_id}: {str(e)}")
        return None
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
