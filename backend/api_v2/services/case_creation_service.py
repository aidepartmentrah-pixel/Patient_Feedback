"""
Case Creation Service (API V2)
Orchestrates subcase creation for incidents and seasonal reports.

Pure orchestration layer - no business logic, validation, or permission checks.
"""

import logging
from typing import Optional
from core.database import get_connection
from api_v2.db_layer import administrative_subcase_db
from api_v2.db_layer import seasonal_report_db
from api.services.notification_service import (
    send_subcase_assignment_notification,
    get_section_admin_email
)

# Configure logger
logger = logging.getLogger(__name__)


def _create_subcase(
    case_type: str,
    incident_id: Optional[int],
    seasonal_report_id: Optional[int],
    target_org_unit_id: int,
    created_by_user_id: int,
    initial_status: str
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
        initial_status=initial_status
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


def create_subcases_for_incident(incident_id: int, current_user) -> None:
    """
    Create subcases for an incident.
    Idempotent - does nothing if subcases already exist.
    
    Args:
        incident_id: The incident ID
        current_user: Current user object (must have user_id attribute) or None for system user
    """
    existing = administrative_subcase_db.get_subcases_by_incident(incident_id)
    if existing:
        return
    
    # Handle None current_user (legacy adapter calls)
    user_id = current_user.user_id if current_user else 1  # Default to system user
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            SELECT td.DepartmentID, au.Type AS OrgUnitType, au.Name AS OrgUnitName
            FROM dbo.APP_IncidentCaseTargetDepartment td
            LEFT JOIN dbo.AdminsrationUnit au ON td.DepartmentID = au.UniqueID
            WHERE td.IncidentRequestCaseID = ?
        """

        cursor.execute(query, (incident_id,))
        rows = cursor.fetchall()

        for row in rows:
            org_type = row.OrgUnitType
            org_name = row.OrgUnitName
            dept_id = row.DepartmentID

            # Guard: routing target should always be a Section (Type=324).
            # Type=325 (Department) or Type=323 (Administration) targets mean the
            # case was routed to the wrong level — the workflow starts at
            # SUBMITTED_TO_SECTION regardless, so Fill Data will show Section as
            # empty and Insight will group the case under the wrong org-type bucket.
            if org_type != 324:
                type_label = {323: 'ADMINISTRATION', 325: 'DEPARTMENT'}.get(org_type, f'UNKNOWN(type={org_type})')
                logger.warning(
                    "[ROUTING_LEVEL_MISMATCH] incident_id=%d routed to OrgUnit %d '%s' "
                    "(Type=%s) — expected Type=324 (SECTION). "
                    "Subcase will start at SUBMITTED_TO_SECTION but the target is not a Section. "
                    "This may cause Insight grouping and Fill Data hierarchy to be inconsistent.",
                    incident_id, dept_id, org_name or '?', type_label
                )

            _create_subcase(
                case_type='INCIDENT_RESPONSE',
                incident_id=incident_id,
                seasonal_report_id=None,
                target_org_unit_id=dept_id,
                created_by_user_id=user_id,
                initial_status='SUBMITTED_TO_SECTION'
            )

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
