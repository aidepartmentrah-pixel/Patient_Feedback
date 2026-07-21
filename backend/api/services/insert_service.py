"""
Insert Service
Handles business logic for creating new incident/feedback records.
"""

from datetime import datetime
from typing import Dict, Any
from backend.core.database import get_connection
from backend.api.db_layer.incident_case import create_incident_case
from backend.api.db_layer.incident_case_target_department import add_target_department
from backend.api.db_layer.incident_case_doctor import add_doctor_to_case
from backend.api.db_layer.incident_case_employee import add_employee_to_case
from backend.api.db_layer.incident_parent import create_incident_parent, assign_case_to_incident
from backend.api.constants.case_statuses import DRAFT_STATUS_ID, READY_TO_SEND_STATUS_ID
from backend.api.db_layer.admin_units import get_admin_unit_type

_ALLOWED_TARGET_ORG_TYPES = {323, 324, 325}  # Administration, Section, Department


def create_record(data: Dict[str, Any], save_mode: str = 'workflow') -> Dict[str, Any]:
    """
    save_mode:
      'workflow'  — existing behaviour (FSM + subcase creation)
      'draft'     — skip validation, status=Draft, no subcase
      'complete'  — validate required fields, status=ReadyToSend, no subcase

    Thin wrapper over the centralized case-creation service (Stage 3 of the
    ML architecture consolidation — see ML_ARCHITECTURE_DECISION_RECORD.md).
    All validation/insert/related-table logic now lives in
    backend.api.services.case_service.create_case(); this function exists
    only to preserve the public signature every existing caller/test relies
    on, with context='ManualInsert'.
    """
    from backend.api.services.case_service import create_case
    return create_case(data, context='ManualInsert', save_mode=save_mode)


def update_record(record_id: int, data: Dict[str, Any], save_mode: str = 'workflow') -> Dict[str, Any]:
    """
    save_mode:
      'workflow'  — existing behaviour (full validation + FSM)
      'draft'     — skip validation, keep/set status=Draft
      'complete'  — validate, promote to Ready to Send

    Thin wrapper over the centralized case-creation service (Stage 5 of the
    ML architecture consolidation — see ML_ARCHITECTURE_DECISION_RECORD.md).
    All validation/update/related-table/ML logic now lives in
    backend.api.services.case_service.update_case(); this function exists
    only to preserve the public signature every existing caller/test relies
    on, with context='ManualInsert'.
    """
    from backend.api.services.case_service import update_case
    return update_case(record_id, data, context='ManualInsert', save_mode=save_mode)


def create_incident_with_cases(payload: Dict[str, Any], save_mode: str = 'workflow') -> Dict[str, Any]:
    """
    Create one parent incident with one or more operational cases.
    save_mode: 'workflow' | 'draft' | 'complete'
    """
    common = payload.get("common", {}) or {}
    cases = payload.get("cases", []) or []

    if not cases:
        return {
            "success": False,
            "error": "VALIDATION_ERROR",
            "message": "At least one case is required",
            "message_ar": "يجب توفير حالة واحدة على الأقل",
            "field": "cases",
        }

    if save_mode != 'draft':
        for idx, case_data in enumerate(cases):
            if not case_data.get("feedback_intent_type_id"):
                return {
                    "success": False,
                    "error": "VALIDATION_ERROR",
                    "message": f"Case #{idx + 1}: Feedback intent type is required",
                    "message_ar": f"الحالة رقم {idx + 1}: نوع نية الملاحظة مطلوب",
                    "field": "feedback_intent_type_id",
                }

    primary_intent = common.get("feedback_intent_type_id") or (cases[0].get("feedback_intent_type_id") if cases else None)

    incident_id = create_incident_parent(
        {
            "patient_name": common.get("patient_name"),
            "primary_doctor_name": common.get("primary_doctor_name"),
            "primary_worker_name": common.get("primary_worker_name"),
            "feedback_intent_type_id": primary_intent,
            "issuing_org_unit_id": common.get("issuing_department_id"),
            "complaint_summary": common.get("complaint_text"),
            "building_id": common.get("building_id"),
            "is_inpatient": 1 if common.get("is_inpatient", True) else 0,
            "created_by_user_id": 1,
        }
    )

    created_cases: list[dict] = []
    for idx, case_data in enumerate(cases):
        target_ids = case_data.get("target_department_ids", [])

        # Validate each target org unit exists and has an allowed type
        for dept_id in target_ids:
            unit_type = get_admin_unit_type(dept_id)
            if unit_type is None:
                return {
                    "success": False,
                    "error": "VALIDATION_ERROR",
                    "message": f"Case #{idx + 1}: target org unit {dept_id} does not exist",
                    "message_ar": f"الحالة رقم {idx + 1}: الوحدة التنظيمية المستهدفة غير موجودة",
                    "field": "target_department_ids",
                }
            if unit_type not in _ALLOWED_TARGET_ORG_TYPES:
                return {
                    "success": False,
                    "error": "VALIDATION_ERROR",
                    "message": (
                        f"Case #{idx + 1}: target org unit {dept_id} has type {unit_type} "
                        "which is not a valid target (allowed: Administration=323, Department=325, Section=324)"
                    ),
                    "message_ar": (
                        f"الحالة رقم {idx + 1}: الوحدة المستهدفة يجب أن تكون إدارة أو قسم أو شعبة"
                    ),
                    "field": "target_department_ids",
                }

        if save_mode == 'workflow' and len(target_ids) != 1:
            return {
                "success": False,
                "error": "VALIDATION_ERROR",
                "message": f"Case #{idx + 1} must target exactly one section/department",
                "message_ar": f"الحالة رقم {idx + 1} يجب أن تستهدف قسماً واحداً فقط",
                "field": "target_department_ids",
            }

        case_payload = {
            **case_data,
            "complaint_text": case_data.get("complaint_text") or common.get("complaint_text") or "",
            "feedback_received_date": case_data.get("feedback_received_date") or common.get("feedback_received_date"),
            "incident_date": case_data.get("incident_date") or common.get("incident_date"),
            "issuing_department_id": case_data.get("issuing_department_id") or common.get("issuing_department_id"),
            "feedback_intent_type_id": case_data.get("feedback_intent_type_id") or common.get("feedback_intent_type_id"),
            "patient_name": common.get("patient_name") or case_data.get("patient_name") or "",
            "is_inpatient": common.get("is_inpatient", True),
            "is_morbidity": case_data.get("is_morbidity", common.get("is_morbidity", False)),
            "source_id": case_data.get("source_id") or common.get("source_id"),
            "building_id": case_data.get("building_id") or common.get("building_id"),
            "doctors": case_data.get("doctors") or common.get("doctors"),
            "employees": case_data.get("employees") or common.get("employees"),
        }

        created = create_record(case_payload, save_mode=save_mode)
        if not created.get("success"):
            # For 'complete' mode with validation failure: save as draft instead
            if save_mode == 'complete' and created.get("save_as_draft"):
                created = create_record(case_payload, save_mode='draft')
                if not created.get("success"):
                    return created
                created["demoted_to_draft"] = True
                created["validation_message"] = "Saved as Draft — some required fields were missing"
            else:
                return created

        case_id = int(created["id"])
        assign_case_to_incident(case_id, incident_id)
        created_cases.append(
            {
                "case_id": case_id,
                "record_id": created.get("record_id"),
            }
        )

    return {
        "success": True,
        "incident_id": incident_id,
        "cases": created_cases,
        "count": len(created_cases),
    }
