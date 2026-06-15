"""
Automatic Force Close Service (API V2)
HCAT Automatic Force Close Policy - Session 4: Automatic Force Close Engine.

Scans APP_AdministrativeSubcase for subcases whose per-level deadline
(SectionDeadlineAt / DepartmentDeadlineAt / AdministrationDeadlineAt) has
expired while that level is still responsible/pending, and marks the
responsible level force-closed.

Automatic Force Close does NOT remove, delete, or hide a case from the
workflow. It marks the current level as force-closed, sets that level's
LateReply flag permanently, and the case escalates to (or remains visible
for) the next responsible level.

Notice records (APP_IncidentCase.RecordTypeID = 2) and Morbidity-related
cases (APP_IncidentCase.IsMorbidity = 1) are never force-closed by this scan,
matching the deadline-initialization exclusions from Session 3.
"""

import logging
from core.database import get_connection
from api_v2.db_layer import administrative_subcase_db

logger = logging.getLogger(__name__)

# APP_IncidentCase.RecordTypeID value identifying a "Notice" record
_RECORD_TYPE_NOTICE = 2

# System user ID used as the actor for automatic database changes
_SYSTEM_USER_ID = 1

# Statuses meaning "this level is currently responsible / pending" for the
# corresponding deadline column. A subcase whose Status is NOT in the
# relevant set is either not yet at this level, or has already been
# force-closed at this level (idempotency via the status filter itself).
_SECTION_PENDING_STATUSES = (
    "SUBMITTED_TO_SECTION",
    "RETURNED_TO_SECTION_FOR_REVISION",
)
_DEPARTMENT_PENDING_STATUSES = (
    "SECTION_ACCEPTED_PENDING_DEPT",
    "RETURNED_TO_DEPT_FOR_REVISION",
    "FORCE_CLOSED_AT_SECTION",
)
_ADMINISTRATION_PENDING_STATUSES = (
    "DEPT_ACCEPTED_PENDING_ADMIN",
    "FORCE_CLOSED_AT_DEPARTMENT",
)

# Automatic explanation text inserted only when the responsible level has not
# entered a real answer yet. Tagged with EntryMode='AUTO_FORCE_CLOSE' so it
# can be identified and replaced later by the Give More Time workflow
# (Session 5) without overwriting a real user's answer.
_SECTION_AUTO_TEXT = "تم الإغلاق قسريا بسبب انتهاء المهلة دون إدخال رد من القسم."
_DEPARTMENT_AUTO_TEXT = "تم الإغلاق قسريا بسبب انتهاء المهلة دون إدخال رد من الإدارة."
_ADMINISTRATION_AUTO_TEXT = "تم الإغلاق قسريا بسبب انتهاء المهلة دون إدخال رد من المديرية."


def _is_policy_enabled(cursor) -> bool:
    """
    Read APP_SystemSettings.automatic_force_close_enabled directly via the
    caller's cursor.

    Avoids backend.api.db_layer.system_settings_db, whose helper functions
    reference a non-existent 'UpdatedByUserID' column (real column is
    'UpdatedBy') and would raise on any call.
    """
    cursor.execute(
        "SELECT SettingValue FROM dbo.APP_SystemSettings WHERE SettingKey = ?",
        ("automatic_force_close_enabled",)
    )
    row = cursor.fetchone()
    if not row or row.SettingValue is None:
        return False
    return str(row.SettingValue).strip().lower() in ("true", "1", "yes")


def _process_level(candidates, force_close_fn, auto_text, result, count_key):
    """
    Apply Notice/Morbidity exclusion to a list of overdue candidates for one
    level, force-closing each remaining candidate and updating result counters
    in place.
    """
    for candidate in candidates:
        if candidate["RecordTypeID"] == _RECORD_TYPE_NOTICE:
            result["skipped_notice_count"] += 1
            continue
        if candidate["IsMorbidity"]:
            result["skipped_morbidity_count"] += 1
            continue

        if force_close_fn(candidate["SubcaseID"], auto_text, _SYSTEM_USER_ID):
            result[count_key] += 1
            logger.info(
                "[AUTO FORCE CLOSE] Subcase %s -> %s (deadline expired)",
                candidate["SubcaseID"], count_key
            )


def run_automatic_force_close_scan(triggered_by_user_id=None) -> dict:
    """
    Run one pass of the automatic force-close scan.

    No-ops entirely (returns skipped_disabled_policy=True) if
    automatic_force_close_enabled is not 'true'.

    Safe to call repeatedly: every level transition is guarded both by the
    Status-based candidate query (a subcase already force-closed at a level
    no longer matches that level's pending statuses) and by a
    WHERE <Level>ForceClosedAt IS NULL guard inside the DB-layer force_close_*
    functions.

    Returns a summary dict:
        success, enabled, scanned_count,
        section_force_closed_count, department_force_closed_count,
        administration_force_closed_count,
        skipped_notice_count, skipped_morbidity_count, skipped_disabled_policy
    """
    result = {
        "success": True,
        "enabled": False,
        "scanned_count": 0,
        "section_force_closed_count": 0,
        "department_force_closed_count": 0,
        "administration_force_closed_count": 0,
        "skipped_notice_count": 0,
        "skipped_morbidity_count": 0,
        "skipped_disabled_policy": False,
    }

    conn = get_connection()
    cursor = conn.cursor()
    try:
        if not _is_policy_enabled(cursor):
            result["skipped_disabled_policy"] = True
            logger.info(
                "[AUTO FORCE CLOSE] automatic_force_close_enabled is not 'true' "
                "- scan skipped, no changes made"
            )
            return result
        result["enabled"] = True
    finally:
        cursor.close()
        conn.close()

    # SECTION: SectionDeadlineAt expired while Section is responsible/pending
    section_candidates = administrative_subcase_db.get_overdue_subcases_for_level(
        "SectionDeadlineAt", _SECTION_PENDING_STATUSES
    )
    result["scanned_count"] += len(section_candidates)
    _process_level(
        section_candidates,
        administrative_subcase_db.force_close_section,
        _SECTION_AUTO_TEXT,
        result,
        "section_force_closed_count",
    )

    # DEPARTMENT: DepartmentDeadlineAt expired while Department is
    # responsible/pending (includes cases just escalated from Section above)
    department_candidates = administrative_subcase_db.get_overdue_subcases_for_level(
        "DepartmentDeadlineAt", _DEPARTMENT_PENDING_STATUSES
    )
    result["scanned_count"] += len(department_candidates)
    _process_level(
        department_candidates,
        administrative_subcase_db.force_close_department,
        _DEPARTMENT_AUTO_TEXT,
        result,
        "department_force_closed_count",
    )

    # ADMINISTRATION: AdministrationDeadlineAt expired while Administration is
    # responsible/pending (includes cases just escalated from Department above)
    administration_candidates = administrative_subcase_db.get_overdue_subcases_for_level(
        "AdministrationDeadlineAt", _ADMINISTRATION_PENDING_STATUSES
    )
    result["scanned_count"] += len(administration_candidates)
    _process_level(
        administration_candidates,
        administrative_subcase_db.force_close_administration,
        _ADMINISTRATION_AUTO_TEXT,
        result,
        "administration_force_closed_count",
    )

    logger.info(
        "[AUTO FORCE CLOSE] Scan complete (triggered_by=%s): scanned=%d "
        "section=%d department=%d administration=%d "
        "notice_skipped=%d morbidity_skipped=%d",
        triggered_by_user_id,
        result["scanned_count"],
        result["section_force_closed_count"],
        result["department_force_closed_count"],
        result["administration_force_closed_count"],
        result["skipped_notice_count"],
        result["skipped_morbidity_count"],
    )
    return result
