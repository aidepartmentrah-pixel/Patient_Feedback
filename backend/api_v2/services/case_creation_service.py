"""
Case Creation Service (API V2)
Orchestrates subcase creation for incidents and seasonal reports.

Pure orchestration layer - no business logic, validation, or permission checks.
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from core.database import get_connection
from api_v2.db_layer import administrative_subcase_db
from api_v2.db_layer import seasonal_report_db
from api.services.notification_service import (
    send_subcase_assignment_notification,
    get_section_admin_email
)

# Configure logger
logger = logging.getLogger(__name__)

# Org unit type codes (AdminsrationUnit.Type)
_ORG_TYPE_SECTION = 324
_ORG_TYPE_DEPARTMENT = 325
_ORG_TYPE_ADMINISTRATION = 323

# APP_IncidentCase.RecordTypeID value identifying a "Notice" record
_RECORD_TYPE_NOTICE = 2

# Fallback deadline windows (days) used only if the APP_SystemSettings rows
# are missing or contain a non-numeric value
_DEFAULT_DEADLINE_DAYS = {
    "section": 10,
    "department": 7,
    "administration": 7,
}


def _get_deadline_days_settings(cursor) -> dict:
    """
    Read the workflow deadline windows (in days) from APP_SystemSettings.

    Reads section_deadline_days, department_deadline_days and
    administration_deadline_days using the caller's open cursor. Falls back
    to _DEFAULT_DEADLINE_DAYS for any key that is missing or not a valid
    integer.
    """
    keys_by_setting = {
        "section_deadline_days": "section",
        "department_deadline_days": "department",
        "administration_deadline_days": "administration",
    }

    result = dict(_DEFAULT_DEADLINE_DAYS)

    cursor.execute(
        "SELECT SettingKey, SettingValue FROM dbo.APP_SystemSettings WHERE SettingKey IN (?, ?, ?)",
        tuple(keys_by_setting.keys())
    )
    for row in cursor.fetchall():
        level = keys_by_setting.get(row.SettingKey)
        if level is None:
            continue
        try:
            result[level] = int(row.SettingValue)
        except (TypeError, ValueError):
            pass

    return result


def get_initial_subcase_status_for_target_org_type(org_type: int) -> str:
    """
    Return the correct initial APP_AdministrativeSubcase.Status for a given
    target org unit type.

    Section (324)        → SUBMITTED_TO_SECTION          (enters section inbox)
    Department (325)     → SECTION_ACCEPTED_PENDING_DEPT (enters department inbox directly)
    Administration (323) → DEPT_ACCEPTED_PENDING_ADMIN   (enters admin inbox directly)
    """
    if org_type == _ORG_TYPE_SECTION:
        return "SUBMITTED_TO_SECTION"
    if org_type == _ORG_TYPE_DEPARTMENT:
        return "SECTION_ACCEPTED_PENDING_DEPT"
    if org_type == _ORG_TYPE_ADMINISTRATION:
        return "DEPT_ACCEPTED_PENDING_ADMIN"
    raise ValueError(f"Unsupported TargetOrgUnit type for workflow routing: {org_type}")


def _create_subcase(
    case_type: str,
    incident_id: Optional[int],
    seasonal_report_id: Optional[int],
    target_org_unit_id: int,
    created_by_user_id: int,
    initial_status: str,
    section_deadline_at: Optional[datetime] = None,
    department_deadline_at: Optional[datetime] = None,
    administration_deadline_at: Optional[datetime] = None
) -> Optional[int]:
    """
    Internal helper to create a subcase.
    Thin adapter to DB layer.

    After creation, sends notification to section admin if they have an email.
    """
    subcase_id = administrative_subcase_db.create_subcase(
        case_type=case_type,
        incident_id=incident_id,
        seasonal_report_id=seasonal_report_id,
        target_org_unit_id=target_org_unit_id,
        created_by_user_id=created_by_user_id,
        initial_status=initial_status,
        section_deadline_at=section_deadline_at,
        department_deadline_at=department_deadline_at,
        administration_deadline_at=administration_deadline_at
    )
    
    # Send notification (async, non-blocking)
    if subcase_id:
        try:
            admin_email = get_section_admin_email(target_org_unit_id)
            if admin_email:
                send_subcase_assignment_notification(
                    to_email=admin_email,
                    case_id=subcase_id
                )
                logger.info(f"NOTIFICATION: Queued email for subcase {subcase_id} to {admin_email}")
            else:
                logger.debug(f"NOTIFICATION: No admin email for org_unit {target_org_unit_id}")
        except Exception as e:
            # Log but don't fail - notification is non-critical
            logger.warning(f"NOTIFICATION: Failed to send for subcase {subcase_id}: {str(e)}")
    
    return subcase_id


def create_subcases_for_incident(incident_id: int, current_user) -> List[Dict[str, Any]]:
    """
    Create subcases for an incident.
    Idempotent - does nothing if subcases already exist.

    Args:
        incident_id: The incident ID
        current_user: Current user object (must have user_id attribute) or None for system user

    Returns:
        List of {"subcase_id": int, "target_org_unit_id": int} for each subcase
        created during this call. Empty list if subcases already existed
        (idempotent no-op) or no target departments were found.
    """
    existing = administrative_subcase_db.get_subcases_by_incident(incident_id)
    if existing:
        return []

    # Handle None current_user (legacy adapter calls)
    user_id = current_user.user_id if current_user else 1  # Default to system user

    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Determine deadline-exclusion flags for this incident:
        # Notice records (RecordTypeID=2) and Morbidity-related cases
        # (IsMorbidity=1) never receive workflow deadlines.
        cursor.execute(
            "SELECT RecordTypeID, IsMorbidity FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?",
            (incident_id,)
        )
        case_row = cursor.fetchone()
        is_notice = bool(case_row and case_row.RecordTypeID == _RECORD_TYPE_NOTICE)
        is_morbidity = bool(case_row and case_row.IsMorbidity)
        excluded_from_deadlines = is_notice or is_morbidity

        deadline_days = None
        publish_time = None
        if not excluded_from_deadlines:
            deadline_days = _get_deadline_days_settings(cursor)
            publish_time = datetime.now()

        query = """
            SELECT td.DepartmentID, au.Type AS OrgUnitType, au.Name AS OrgUnitName
            FROM dbo.APP_IncidentCaseTargetDepartment td
            LEFT JOIN dbo.AdminsrationUnit au ON td.DepartmentID = au.UniqueID
            WHERE td.IncidentRequestCaseID = ?
        """

        cursor.execute(query, (incident_id,))
        rows = cursor.fetchall()

        created_subcases: List[Dict[str, Any]] = []

        for row in rows:
            org_type = row.OrgUnitType
            org_name = row.OrgUnitName
            dept_id = row.DepartmentID

            initial_status = get_initial_subcase_status_for_target_org_type(org_type)
            logger.info(
                "[ROUTING] incident_id=%d → OrgUnit %d '%s' (Type=%d) → initial_status=%s",
                incident_id, dept_id, org_name or '?', org_type, initial_status
            )

            # Initialize ONLY the deadline for the starting workflow level,
            # and only for non-Notice, non-Morbidity cases.
            section_deadline_at = None
            department_deadline_at = None
            administration_deadline_at = None

            if not excluded_from_deadlines:
                if org_type == _ORG_TYPE_SECTION:
                    section_deadline_at = publish_time + timedelta(days=deadline_days["section"])
                elif org_type == _ORG_TYPE_DEPARTMENT:
                    department_deadline_at = publish_time + timedelta(days=deadline_days["department"])
                elif org_type == _ORG_TYPE_ADMINISTRATION:
                    administration_deadline_at = publish_time + timedelta(days=deadline_days["administration"])

            subcase_id = _create_subcase(
                case_type='INCIDENT_RESPONSE',
                incident_id=incident_id,
                seasonal_report_id=None,
                target_org_unit_id=dept_id,
                created_by_user_id=user_id,
                initial_status=initial_status,
                section_deadline_at=section_deadline_at,
                department_deadline_at=department_deadline_at,
                administration_deadline_at=administration_deadline_at
            )

            created_subcases.append({"subcase_id": subcase_id, "target_org_unit_id": dept_id})

        return created_subcases

    finally:
        cursor.close()
        conn.close()


def create_subcases_for_seasonal_report(seasonal_report_id: int, current_user) -> None:
    """
    Create subcases for a seasonal report.
    Idempotent - does nothing if subcases already exist.
    
    Args:
        seasonal_report_id: The seasonal report ID
        current_user: Current user object (must have user_id attribute) or None for system user
    """
    existing = administrative_subcase_db.get_subcases_by_seasonal_report(seasonal_report_id)
    if existing:
        return
    
    # Handle None current_user (legacy adapter calls)
    user_id = current_user.user_id if current_user else 1  # Default to system user
    
    org_units = seasonal_report_db.get_target_orgunits_for_seasonal_report(seasonal_report_id)
    
    for org_unit_id in org_units:
        _create_subcase(
            case_type='SEASONAL_REPORT_RESPONSE',
            incident_id=None,
            seasonal_report_id=seasonal_report_id,
            target_org_unit_id=org_unit_id,
            created_by_user_id=user_id,
            initial_status='SUBMITTED_TO_SECTION'
        )
