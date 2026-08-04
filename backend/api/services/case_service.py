"""
Central Case-Creation Service

Single point of truth for creating an operational case, replacing the
duplicated validation/hierarchy/FSM logic previously spread across
insert_service.create_record(), migration_insert_service.create_record_migrated(),
import_db.insert_case(), and table_view_service._insert_historical_record().

Stage 3 of the ML architecture consolidation (see ML_ARCHITECTURE_DECISION_RECORD.md
and the approved execution plan) moves manual-insert's create_record() logic here
verbatim and adds one new guarantee: every case creation now registers an
ml.EmbeddingProcessingJob row in the SAME transaction as the case insert itself.

This is deliberately NOT the same guarantee as "embeddings are computed
synchronously" — job *registration* is a trivial same-schema INSERT with no
external I/O, so it's safe to make atomic with the case insert. Actual
embedding computation (model inference) remains a separate, non-blocking,
asynchronous concern handled by the Stage 6 worker.

`context` distinguishes the caller (ManualInsert today; BulkImport/
HistoricalImport/LegacyMigration are threaded through for later stages —
see Stage 4+) but does not yet branch behavior. Only the ManualInsert path
is exercised/tested as of Stage 3.
"""

from datetime import datetime
from typing import Dict, Any
from backend.core.database import get_connection
from backend.api.db_layer.incident_case import create_incident_case
from backend.api.db_layer.incident_case_target_department import add_target_department
from backend.api.db_layer.incident_case_doctor import add_doctor_to_case
from backend.api.db_layer.incident_case_employee import add_employee_to_case
from backend.api.services.staff_directory_service import materialize_doctor_id, materialize_employee_id
from backend.api.db_layer.incident_parent import create_incident_parent, assign_case_to_incident
from backend.api.db_layer import ml_embedding_job_db
from backend.api.db_layer import ml_case_training_db
from backend.api.constants.case_statuses import DRAFT_STATUS_ID, READY_TO_SEND_STATUS_ID, CLOSED_STATUS_ID

# Maps insert/update payload keys -> ml.CaseTrainingRecord columns. Kept in
# one place since both create_case() and update_case() need the same mapping.
_ML_TEXT_FIELD_MAP = {
    'complaint_text': 'ComplaintText',
    'immediate_action': 'ImmediateActionText',
    'taken_action': 'TakenActionText',
}
_ML_LABEL_FIELD_MAP = {
    'domain_id': 'DomainID',
    'category_id': 'CategoryID',
    'subcategory_id': 'SubCategoryID',
    'classification_id': 'ClassificationID',
    'severity_id': 'SeverityLevelID',
    'stage_id': 'StageID',
    'harm_id': 'HarmLevelID',
}


def _build_ml_training_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the subset of `data` relevant to ml.CaseTrainingRecord."""
    fields = {}
    for src_key, ml_col in {**_ML_TEXT_FIELD_MAP, **_ML_LABEL_FIELD_MAP}.items():
        if data.get(src_key) is not None:
            fields[ml_col] = data.get(src_key)
    return fields


def create_case(data: Dict[str, Any], context: str = 'ManualInsert', save_mode: str = 'workflow') -> Dict[str, Any]:
    """
    Create an operational case plus its ml.EmbeddingProcessingJob registration.

    context: 'ManualInsert' | 'BulkImport' | 'HistoricalImport' | 'LegacyMigration'
             (only 'ManualInsert' behavior is implemented/tested as of Stage 3)
    save_mode:
      'workflow'  — existing behaviour (FSM + subcase creation)
      'draft'     — skip validation, status=Draft, no subcase
      'complete'  — validate required fields, status=ReadyToSend, no subcase
    """
    conn = None
    cursor = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        # --- Draft: skip all validation, save immediately ---
        if save_mode == 'draft':
            pass  # fall through directly to insert below

        elif save_mode in ('workflow', 'complete'):
            # -----------------------------
            # Required fields validation
            # -----------------------------
            _record_type_id = data.get('record_type_id', 1)

            if _record_type_id == 2:  # Notice — classification not required, but description is
                required_fields = [
                    'complaint_text',
                    'feedback_intent_type_id',
                    'feedback_received_date',
                    'incident_date',
                    'issuing_department_id',
                    'patient_name',
                    'source_id',
                ]
            elif context == 'BulkImport':
                # The Excel import template only marks these columns
                # mandatory (see import_service.TEMPLATE_COLUMNS) — Severity,
                # Stage, Harm Level, Feedback Risk Type, and Immediate Action
                # are optional there, unlike the manual Insert page. This is
                # a deliberate per-context policy, not a gap: bulk-imported
                # historical rows legitimately may not have every assessment
                # field filled in.
                required_fields = [
                    'complaint_text',
                    'feedback_received_date',
                    'incident_date',
                    'issuing_department_id',
                    'domain_id',
                    'category_id',
                    'subcategory_id',
                    'classification_id',
                    'feedback_intent_type_id',
                    'patient_name',
                    'source_id',
                ]
            else:  # Complaint via ManualInsert — full existing list unchanged
                required_fields = [
                    'complaint_text',
                    'feedback_received_date',
                    'incident_date',
                    'issuing_department_id',
                    'domain_id',
                    'category_id',
                    'subcategory_id',
                    'classification_id',
                    'severity_id',
                    'stage_id',
                    'harm_id',
                    'requires_explanation',
                    'clinical_risk_type_id',
                    'feedback_intent_type_id',
                    'immediate_action',
                    'patient_name',
                    'is_inpatient',
                    'source_id',
                ]

            for field in required_fields:
                if field not in data or data[field] is None or data[field] == '':
                    if save_mode == 'complete':
                        # Return error but caller will save as Draft instead
                        return {
                            "success": False,
                            "error": "VALIDATION_ERROR",
                            "message": f"{field.replace('_', ' ').title()} is required",
                            "message_ar": f"حقل {field} مطلوب",
                            "field": field,
                            "save_as_draft": True,
                        }
                    return {
                        "success": False,
                        "error": "VALIDATION_ERROR",
                        "message": f"{field.replace('_', ' ').title()} is required",
                        "message_ar": f"حقل {field} مطلوب",
                        "field": field
                }

        # Incident Date cannot be after Received Date — you can't receive a
        # complaint before the incident that prompted it actually happened.
        if data.get('incident_date') and data.get('feedback_received_date'):
            if data['incident_date'] > data['feedback_received_date']:
                return {
                    "success": False,
                    "error": "VALIDATION_ERROR",
                    "message": "Incident Date cannot be after Received Date",
                    "message_ar": "لا يمكن أن يكون تاريخ الحادثة بعد تاريخ الاستلام",
                    "field": "incident_date"
                }

        # Building: Require either building_id or building_code (skip for draft and notices)
        if save_mode != 'draft' and data.get('record_type_id', 1) != 2 and not data.get('building_id') and not data.get('building_code'):
            return {
                "success": False,
                "error": "VALIDATION_ERROR",
                "message": "Either building_id or building_code is required",
                "message_ar": "يجب توفير رقم المبنى أو رمز المبنى",
                "field": "building_id"
            }
        # -----------------------------
        # Validate foreign keys
        # -----------------------------
        # Use actual APP_LOOKUP_* tables
        validations = [
            ('domain_id', 'Domain', 'dbo.APP_LOOKUP_DOMAIN', 'DomainID', 'القطاع'),
            ('category_id', 'Category', 'dbo.APP_LOOKUP_CATEGORY', 'CategoryID', 'الفئة'),
            ('severity_id', 'Severity', 'dbo.APP_LOOKUP_SEVERITY', 'SeverityID', 'مستوى الخطورة')
        ]

        for field_name, english_name, table_name, id_column, arabic_name in validations:
            if data.get(field_name):
                try:
                    cursor.execute(
                        f"SELECT COUNT(*) FROM {table_name} WHERE {id_column} = ?",
                        (data[field_name],)
                    )
                    if cursor.fetchone()[0] == 0:
                        return {
                            "success": False,
                            "error": "INVALID_REFERENCE",
                            "message": f"{english_name} ID {data[field_name]} does not exist",
                            "message_ar": f"{arabic_name} رقم {data[field_name]} غير موجود",
                            "field": field_name
                        }
                except Exception:
                    pass

        # -----------------------------
        # Optional FK validations
        # -----------------------------
        optional_validations = [
            ('subcategory_id', 'Subcategory', 'dbo.APP_LOOKUP_SUBCATEGORY', 'SubCategoryID', 'الفئة الفرعية'),
            ('classification_id', 'Classification', 'dbo.APP_LOOKUP_CLASSIFICATION', 'ClassificationID', 'التصنيف'),
            ('stage_id', 'Stage', 'dbo.APP_LOOKUP_CASE_STAGE', 'StageID', 'المرحلة'),
            ('harm_id', 'Harm Level', 'dbo.APP_LOOKUP_HARM_LEVEL', 'HarmID', 'مستوى الضرر'),
            ('building_id', 'Building', 'dbo.APP_LOOKUP_BUILDING', 'BuildingID', 'المبنى'),
            ('explanation_status_id', 'Explanation Status', 'dbo.APP_LOOKUP_EXPLANATION_STATUS', 'StatusID', 'حالة الشرح'),
            # Issuing department/source validation omitted due to unknown tables; keep resilient
        ]

        for field_name, english_name, table_name, id_column, arabic_name in optional_validations:
            if data.get(field_name):
                try:
                    cursor.execute(
                        f"SELECT COUNT(*) FROM {table_name} WHERE {id_column} = ?",
                        (data[field_name],)
                    )
                    if cursor.fetchone()[0] == 0:
                        return {
                            "success": False,
                            "error": "INVALID_REFERENCE",
                            "message": f"{english_name} ID {data[field_name]} does not exist",
                            "message_ar": f"{arabic_name} رقم {data[field_name]} غير موجود",
                            "field": field_name
                        }
                except Exception:
                    pass

        # -----------------------------
        # Hierarchy validation
        # -----------------------------
        try:
            if data.get('category_id') and data.get('domain_id'):
                cursor.execute(
                    "SELECT COUNT(*) FROM dbo.APP_LOOKUP_CATEGORY WHERE CategoryID = ? AND DomainID = ?",
                    (data['category_id'], data['domain_id'])
                )
                if cursor.fetchone()[0] == 0:
                    return {
                        "success": False,
                        "error": "VALIDATION_ERROR",
                        "message": "Selected category does not belong to the selected domain",
                        "message_ar": "الفئة المختارة لا تنتمي للقطاع المختار",
                        "field": "category_id"
                    }
        except Exception:
            pass

        try:
            if data.get('subcategory_id') and data.get('category_id'):
                cursor.execute(
                    "SELECT COUNT(*) FROM dbo.APP_LOOKUP_SUBCATEGORY WHERE SubCategoryID = ? AND CategoryID = ?",
                    (data['subcategory_id'], data['category_id'])
                )
                if cursor.fetchone()[0] == 0:
                    return {
                        "success": False,
                        "error": "VALIDATION_ERROR",
                        "message": "Selected subcategory does not belong to the selected category",
                        "message_ar": "الفئة الفرعية المختارة لا تنتمي للفئة المختارة",
                        "field": "subcategory_id"
                    }
        except Exception:
            pass

        try:
            if data.get('classification_id') and data.get('subcategory_id'):
                cursor.execute(
                    "SELECT COUNT(*) FROM dbo.APP_LOOKUP_CLASSIFICATION WHERE ClassificationID = ? AND SubCategoryID = ?",
                    (data['classification_id'], data['subcategory_id'])
                )
                if cursor.fetchone()[0] == 0:
                    return {
                        "success": False,
                        "error": "VALIDATION_ERROR",
                        "message": "Selected classification does not belong to the selected subcategory",
                        "message_ar": "التصنيف المختار لا ينتمي للفئة الفرعية المختارة",
                        "field": "classification_id"
                    }
        except Exception:
            pass

        # -----------------------------
        # Insert main record via db_layer
        # -----------------------------
        # Safe, strict conversion for is_inpatient (MUST be boolean)
        raw = data.get("is_inpatient", True)
        if isinstance(raw, bool):
            is_inpatient_val = 1 if raw else 0
        elif isinstance(raw, str):
            is_inpatient_val = 1 if raw.upper() in ("IN", "TRUE", "1") else 0
        elif isinstance(raw, int):
            is_inpatient_val = 1 if raw == 1 else 0
        else:
            is_inpatient_val = 1

        # Safe, strict conversion for is_morbidity (MUST be boolean, default 0)
        raw_morbidity = data.get("is_morbidity", False)
        if isinstance(raw_morbidity, bool):
            is_morbidity_val = 1 if raw_morbidity else 0
        elif isinstance(raw_morbidity, str):
            is_morbidity_val = 1 if raw_morbidity.lower() in ("yes", "true", "1") else 0
        elif isinstance(raw_morbidity, int):
            is_morbidity_val = 1 if raw_morbidity == 1 else 0
        else:
            is_morbidity_val = 0

        clinical_risk_type_id = data.get('clinical_risk_type_id')
        feedback_intent_type_id = data.get('feedback_intent_type_id')

        # Validate doctors exist, resolving each to a real reserve int id.
        #
        # SESSION C2: doc_id may now be a reserve int OR an opaque Hospital
        # Directory API id (see staff_directory_service) — the old raw
        # `SELECT ... WHERE DoctorID = ?` existence check against
        # APP_LOOKUP_DOCTOR/APP_RESERVE_DOCTOR fails outright on a string id
        # (SQL int conversion error). materialize_doctor_id() replaces it:
        # it finds-or-creates a real reserve row for an external id (which
        # IS the existence check — the id came from a real API search
        # result), or passes a reserve int through unchanged. The resolved
        # int is written back into the doc dict so every later use in this
        # function (the actual APP_IncidentCaseDoctor insert further below)
        # sees an already-resolved id instead of re-decoding it.
        if data.get('doctors'):
            for doc in data['doctors']:
                doc_id = doc.get('doctor_id')
                if not doc_id:
                    continue
                resolved = materialize_doctor_id(doc_id, doc.get('doctor_name', ''))
                if not resolved:
                    return {
                        "success": False,
                        "error": "INVALID_REFERENCE",
                        "message": f"Doctor ID {doc_id} does not exist",
                        "message_ar": f"رقم الطبيب {doc_id} غير موجود",
                        "field": "doctors"
                    }
                doc['doctor_id'] = resolved

        # Resolve building: prefer provided BuildingID, else map BuildingCode via DB
        building_id = data.get('building_id')
        if not building_id and data.get('building_code'):
            code = str(data.get('building_code')).strip().upper()
            try:
                cursor.execute(
                    """
                    SELECT TOP 1 BuildingID
                    FROM dbo.APP_LOOKUP_BUILDING
                    WHERE UPPER(BuildingCode) = ?
                    ORDER BY BuildingID
                    """,
                    (code,)
                )
                row = cursor.fetchone()
                if row:
                    building_id = row.BuildingID
            except Exception:
                building_id = None

        # -----------------------------
        # STATUS OVERRIDE for draft/complete
        # Bypass FSM entirely — subcase creation also skipped below
        # -----------------------------
        if save_mode == 'draft':
            case_status_id = DRAFT_STATUS_ID
            explanation_status_id = 4  # No Explanation Needed (safe default)
            requires_explanation_bit = 0
        elif save_mode == 'complete':
            case_status_id = READY_TO_SEND_STATUS_ID
            explanation_status_id = 4
            requires_explanation_bit = 0
        elif save_mode == 'import_closed':
            # Bulk-import rule: every imported case lands Closed and bypasses
            # the live messaging/inbox workflow, regardless of clinical risk
            # type -- unlike 'workflow' below, a red flag / never event does
            # NOT reopen it here. ClinicalRiskTypeID itself is still stored
            # as given, so the case remains identifiable for reporting; any
            # human-visibility need for red-flag/never-event imports is
            # handled by the import review screen/report, not by reopening
            # the case's workflow.
            case_status_id = CLOSED_STATUS_ID
            explanation_status_id = 4
            requires_explanation_bit = 0
        else:
            # -----------------------------
            # FSM LOGIC: Explanation Workflow
            # -----------------------------
            is_red_flag = clinical_risk_type_id == 2
            is_never_event = clinical_risk_type_id == 3
            requires_explanation = data.get('requires_explanation')

            if isinstance(requires_explanation, bool):
                requires_explanation_bit = 1 if requires_explanation else 0
            elif isinstance(requires_explanation, str):
                requires_explanation_bit = 1 if requires_explanation.lower() in ('true', '1', 'yes') else 0
            elif isinstance(requires_explanation, int):
                requires_explanation_bit = 1 if requires_explanation == 1 else 0
            else:
                requires_explanation_bit = 0

            if is_red_flag or is_never_event or requires_explanation_bit:
                case_status_id = 1  # Open
                explanation_status_id = 1  # Waiting
            else:
                case_status_id = 3  # Closed
                explanation_status_id = 4  # No Explanation Needed

        payload = {
            "ComplaintText": data.get('complaint_text'),
            "ImmediateAction": data.get('immediate_action'),
            "TakenAction": data.get('taken_action'),
            "FeedbackRecievedDate": data.get('feedback_received_date'),
            "IncidentDate": data.get('incident_date') or data.get('feedback_received_date'),
            "PatientName": data.get('patient_name'),
            "IssuingOrgUnitID": data.get('issuing_department_id'),
            "CreatedByUserID": 1,
            "isINPatient": is_inpatient_val,
            "IsMorbidity": is_morbidity_val,
            "ClinicalRiskTypeID": clinical_risk_type_id,
            "FeedbackIntentTypeID": feedback_intent_type_id,
            "BuildingID": data.get('building_id') or building_id,
            "DomainID": data.get('domain_id'),
            "CategoryID": data.get('category_id'),
            "SubCategoryID": data.get('subcategory_id'),
            "ClassificationID": data.get('classification_id'),
            "SeverityID": data.get('severity_id'),
            "StageID": data.get('stage_id'),
            "HarmLevelID": data.get('harm_id'),
            "CaseStatusID": case_status_id,
            "SourceID": data.get('source_id'),
            "ExplanationStatusID": explanation_status_id,
            "RequiresExplanation": requires_explanation_bit,
            "RecordTypeID": data.get('record_type_id', 1),
        }

        # -----------------------------
        # Case insert + ML job registration — ONE shared transaction.
        # Job registration is a trivial same-schema INSERT (FK to the row
        # we just inserted, in the same transaction, so it cannot fail for
        # data reasons) — making it atomic with the case insert is safe and
        # is what guarantees every case has a known ML job, per
        # ML_ARCHITECTURE_DECISION_RECORD.md principle 2. This is NOT the
        # same as making embedding *computation* synchronous — that remains
        # a separate, non-blocking worker concern (Stage 6).
        # -----------------------------
        new_id = create_incident_case(payload, cursor=cursor)
        ml_case_training_db.upsert_case_training_record(cursor, new_id, _build_ml_training_fields(data))
        ml_embedding_job_db.insert_embedding_job(cursor, new_id, 'Create')
        conn.commit()

        # -----------------------------
        # Related tables
        # -----------------------------
        if data.get('target_department_ids'):
            for idx, dept_id in enumerate(data['target_department_ids']):
                add_target_department(
                    incident_id=new_id,
                    department_id=dept_id,
                    assigned_by_user_id=1,
                    is_primary=(idx == 0)
                )

        if data.get('doctors'):
            primary_assigned = False
            for doc in data['doctors']:
                # doctor_id was already resolved to a real reserve int by
                # the "Validate doctors exist" block above (materialize_doctor_id
                # mutates the dict in place) — no need to resolve it again here.
                doc_id = doc.get('doctor_id')
                if not doc_id:
                    continue
                add_doctor_to_case(
                    incident_id=new_id,
                    doctor_id=doc_id,
                    assigned_by_user_id=1,
                    doctor_name=doc.get('doctor_name', ''),
                    is_primary=(not primary_assigned)
                )
                primary_assigned = True

        # -----------------------------
        # Employee Linkage (Phase Fix)
        # -----------------------------
        # Link employees to this incident via APP_IncidentCaseEmployee
        if data.get('employees'):
            primary_assigned = False
            for emp in data['employees']:
                emp_id = emp.get('employee_id')
                if not emp_id:
                    continue
                try:
                    # SESSION C3: emp_id may be a reserve int or an opaque
                    # Hospital Directory API id — APP_IncidentCaseEmployee.
                    # EmployeeID is a real int FK, so resolve/materialize
                    # first (no earlier validation block exists for
                    # employees, unlike doctors above).
                    resolved_employee_id = materialize_employee_id(emp_id, emp.get('full_name', ''))
                    if not resolved_employee_id:
                        continue
                    add_employee_to_case(
                        incident_id=new_id,
                        employee_id=resolved_employee_id,
                        assigned_by_user_id=1,
                        full_name=emp.get('full_name', ''),
                        is_primary=(not primary_assigned)
                    )
                    primary_assigned = True
                except Exception as e:
                    print(f"[WARN] Failed to link employee {emp_id}: {str(e)}")

        # db_layer functions commit internally; nothing to commit here

        # -------------------------------------------
        # API V2 ADAPTER HOOK — create workflow subcase
        # SKIPPED for draft/complete: subcase only created on publish
        # -------------------------------------------
        if save_mode == 'workflow':
            try:
                from backend.api_v2.services.case_creation_service import create_subcases_for_incident
                create_subcases_for_incident(new_id, current_user=None)
            except Exception as e:
                print(f"[API V2 ADAPTER WARNING] Failed to create subcases for incident {new_id}: {str(e)}")
                import traceback
                traceback.print_exc()

        record_id = f"REC-{datetime.now().year}-{str(new_id).zfill(4)}"

        return {
            "success": True,
            "message": "Record created successfully",
            "message_ar": "تم إنشاء السجل بنجاح",
            "record_id": record_id,
            "id": new_id,
            "incident_id": new_id,
            "status_id": case_status_id,
            "save_mode": save_mode,
            "created_at": datetime.now().isoformat()
        }

    except Exception as e:
        if conn:
            conn.rollback()
        return {
            "success": False,
            "error": "DATABASE_ERROR",
            "message": f"Failed to create record: {str(e)}",
            "message_ar": f"فشل في إنشاء السجل: {str(e)}"
        }

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def update_case(record_id: int, data: Dict[str, Any], context: str = 'ManualInsert', save_mode: str = 'workflow') -> Dict[str, Any]:
    """
    save_mode:
      'workflow'  — existing behaviour (full validation + FSM)
      'draft'     — skip validation, keep/set status=Draft
      'complete'  — validate, promote to Ready to Send

    Stage 5 of the ML architecture consolidation: replaces the legacy
    ml_insert_adapter.add_corrected_record_to_ml() append-only SQLite write
    (which created a new, unlinked ML row on every single edit) with an
    upsert against the one current ml.CaseTrainingRecord row for this case,
    plus a job registration so the worker knows what kind of reprocessing is
    needed — TextChanged (embeddings must be recomputed) or LabelsChanged
    (labels only, no embedding recompute needed). Both are done in the same
    transaction as the main record update.
    """
    conn = None
    cursor = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Load current FSM state + record type + current text (the text
        # columns are needed here, in addition to the FSM columns already
        # fetched by the pre-Stage-5 version of this query, to detect
        # whether this edit actually changes complaint/action text).
        cursor.execute(
            "SELECT CaseStatusID, ExplanationStatusID, ClinicalRiskTypeID, RecordTypeID, BuildingID, "
            "ComplaintText, ImmediateAction, TakenAction "
            "FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?",
            (record_id,)
        )
        row = cursor.fetchone()
        if not row:
            return {"success": False, "error": "NOT_FOUND", "message": "Record not found"}

        current_case_status_id = row.CaseStatusID
        current_explanation_status_id = row.ExplanationStatusID
        current_clinical_risk_type_id = row.ClinicalRiskTypeID
        current_record_type_id = row.RecordTypeID or 1
        existing_building_id = row.BuildingID
        current_complaint_text = row.ComplaintText
        current_immediate_action = row.ImmediateAction
        current_taken_action = row.TakenAction

        # --- Draft: skip all validation ---
        if save_mode == 'draft':
            pass  # fall through to update

        elif save_mode in ('workflow', 'complete'):
            if current_record_type_id == 2:  # Notice — classification not required, but description is
                required_fields = [
                    'complaint_text',
                    'feedback_intent_type_id',
                    'feedback_received_date',
                    'incident_date',
                    'issuing_department_id',
                    'patient_name',
                    'source_id',
                ]
            else:
                required_fields = [
                    'complaint_text', 'feedback_received_date', 'incident_date', 'issuing_department_id',
                    'domain_id', 'category_id', 'subcategory_id', 'classification_id',
                    'severity_id', 'stage_id', 'harm_id', 'requires_explanation',
                    'clinical_risk_type_id', 'feedback_intent_type_id',
                    'immediate_action', 'patient_name', 'is_inpatient', 'source_id'
                ]

            for field in required_fields:
                if field not in data or data[field] is None or data[field] == "":
                    if save_mode == 'complete':
                        return {
                            "success": False,
                            "error": "VALIDATION_ERROR",
                            "message": f"{field.replace('_', ' ').title()} is required",
                            "message_ar": f"حقل {field} مطلوب",
                            "field": field,
                            "save_as_draft": True,
                        }
                    return {
                        "success": False,
                        "error": "VALIDATION_ERROR",
                        "message": f"{field.replace('_', ' ').title()} is required",
                        "message_ar": f"حقل {field} مطلوب",
                        "field": field
                    }

            # Building required for complaints only (Notices are exempt)
            if current_record_type_id != 2:
                if not data.get('building_id') and not data.get('building_code') and not existing_building_id:
                    return {"success": False, "error": "VALIDATION_ERROR", "message": "Either building_id or building_code is required", "field": "building_id"}

            if data.get('clinical_risk_type_id') is not None and int(data.get('clinical_risk_type_id')) != int(current_clinical_risk_type_id):
                return {"success": False, "error": "IMMUTABLE_FIELD", "message": "Clinical Risk Type cannot be changed after creation", "field": "clinical_risk_type_id"}

        # Incident Date cannot be after Received Date — you can't receive a
        # complaint before the incident that prompted it actually happened.
        if data.get('incident_date') and data.get('feedback_received_date'):
            if data['incident_date'] > data['feedback_received_date']:
                return {
                    "success": False,
                    "error": "VALIDATION_ERROR",
                    "message": "Incident Date cannot be after Received Date",
                    "message_ar": "لا يمكن أن يكون تاريخ الحادثة بعد تاريخ الاستلام",
                    "field": "incident_date"
                }

        # Validate foreign keys (same as create_case)
        validations = [
            ('domain_id', 'Domain', 'dbo.APP_LOOKUP_DOMAIN', 'DomainID', 'القطاع'),
            ('category_id', 'Category', 'dbo.APP_LOOKUP_CATEGORY', 'CategoryID', 'الفئة'),
            ('severity_id', 'Severity', 'dbo.APP_LOOKUP_SEVERITY', 'SeverityID', 'مستوى الخطورة')
        ]

        # -----------------------------
        # FSM TRANSITION LOGIC
        # -----------------------------
        # Status override for draft/complete save modes
        # -----------------------------
        if save_mode == 'draft':
            new_case_status_id = DRAFT_STATUS_ID
            new_explanation_status_id = current_explanation_status_id
            command = None
        elif save_mode == 'complete':
            new_case_status_id = READY_TO_SEND_STATUS_ID
            new_explanation_status_id = current_explanation_status_id
            command = None
        else:
            new_case_status_id = current_case_status_id
            new_explanation_status_id = current_explanation_status_id
            command = data.get("fsm_command")  # "submit_explanation" | "complete_actions" | "force_close"

        # Protect terminal states - block FSM transitions on closed cases
        if current_case_status_id == 3 and command:  # CaseStatusID = 3 is Closed
            return {
                "success": False,
                "error": "TERMINAL_STATE",
                "message": "Cannot perform FSM transitions on closed cases",
                "message_ar": "لا يمكن تنفيذ تغييرات الحالة على القضايا المغلقة",
                "field": "fsm_command"
            }

        # S0 → S1
        if command == "submit_explanation":
            if not (current_case_status_id == 1 and current_explanation_status_id == 1):
                return {"success": False, "error": "INVALID_STATE", "message": "Cannot submit explanation in this state"}

            new_case_status_id = 2  # In Progress
            new_explanation_status_id = 2  # Responded

        # S1 → S3
        elif command == "complete_actions":
            if not (current_case_status_id == 2 and current_explanation_status_id == 2):
                return {"success": False, "error": "INVALID_STATE", "message": "Cannot complete actions in this state"}

            # TODO: Check all action items completed here

            new_case_status_id = 3  # Closed
            new_explanation_status_id = 2  # Responded

        # S0 → S2
        elif command == "force_close":
            if not (current_case_status_id == 1 and current_explanation_status_id == 1):
                return {"success": False, "error": "INVALID_STATE", "message": "Cannot force close in this state"}

            new_case_status_id = 3  # Closed
            new_explanation_status_id = 3  # Forcibly Closed

        for field_name, english_name, table_name, id_column, arabic_name in validations:
            if data.get(field_name):
                try:
                    cursor.execute(
                        f"SELECT COUNT(*) FROM {table_name} WHERE {id_column} = ?",
                        (data[field_name],)
                    )
                    if cursor.fetchone()[0] == 0:
                        return {
                            "success": False,
                            "error": "INVALID_REFERENCE",
                            "message": f"{english_name} ID {data[field_name]} does not exist",
                            "message_ar": f"{arabic_name} رقم {data[field_name]} غير موجود",
                            "field": field_name
                        }
                except Exception:
                    pass

        # Hierarchy validation
        try:
            if data.get('category_id') and data.get('domain_id'):
                cursor.execute(
                    "SELECT COUNT(*) FROM dbo.APP_LOOKUP_CATEGORY WHERE CategoryID = ? AND DomainID = ?",
                    (data['category_id'], data['domain_id'])
                )
                if cursor.fetchone()[0] == 0:
                    return {
                        "success": False,
                        "error": "VALIDATION_ERROR",
                        "message": "Selected category does not belong to the selected domain",
                        "message_ar": "الفئة المختارة لا تنتمي للقطاع المختار",
                        "field": "category_id"
                    }
        except Exception:
            pass

        try:
            if data.get('subcategory_id') and data.get('category_id'):
                cursor.execute(
                    "SELECT COUNT(*) FROM dbo.APP_LOOKUP_SUBCATEGORY WHERE SubCategoryID = ? AND CategoryID = ?",
                    (data['subcategory_id'], data['category_id'])
                )
                if cursor.fetchone()[0] == 0:
                    return {
                        "success": False,
                        "error": "VALIDATION_ERROR",
                        "message": "Selected subcategory does not belong to the selected category",
                        "message_ar": "الفئة الفرعية المختارة لا تنتمي للفئة المختارة",
                        "field": "subcategory_id"
                    }
        except Exception:
            pass

        try:
            if data.get('classification_id') and data.get('subcategory_id'):
                cursor.execute(
                    "SELECT COUNT(*) FROM dbo.APP_LOOKUP_CLASSIFICATION WHERE ClassificationID = ? AND SubCategoryID = ?",
                    (data['classification_id'], data['subcategory_id'])
                )
                if cursor.fetchone()[0] == 0:
                    return {
                        "success": False,
                        "error": "VALIDATION_ERROR",
                        "message": "Selected classification does not belong to the selected subcategory",
                        "message_ar": "التصنيف المختار لا ينتمي للفئة الفرعية المختارة",
                        "field": "classification_id"
                    }
        except Exception:
            pass

        # Update main record
        # Safe, strict conversion for is_inpatient (MUST be boolean)
        raw = data.get("is_inpatient", True)
        if isinstance(raw, bool):
            is_inpatient_val = 1 if raw else 0
        elif isinstance(raw, str):
            is_inpatient_val = 1 if raw.upper() in ("IN", "TRUE", "1") else 0
        elif isinstance(raw, int):
            is_inpatient_val = 1 if raw == 1 else 0
        else:
            is_inpatient_val = 1

        clinical_risk_type_id = data.get('clinical_risk_type_id')
        feedback_intent_type_id = data.get('feedback_intent_type_id')

        # -----------------------------
        # BUILDING RESOLUTION WITH FALLBACK
        # -----------------------------
        building_id = None
        if data.get('building_id'):
            building_id = data.get('building_id')
        elif data.get('building_code'):
            code = str(data.get('building_code')).strip().upper()
            try:
                cursor.execute(
                    """
                    SELECT TOP 1 BuildingID
                    FROM dbo.APP_LOOKUP_BUILDING
                    WHERE UPPER(BuildingCode) = ?
                    ORDER BY BuildingID
                    """,
                    (code,)
                )
                row2 = cursor.fetchone()
                if row2:
                    building_id = row2.BuildingID
            except Exception:
                building_id = None
        if building_id is None:
            building_id = existing_building_id

        update_query = """
            UPDATE dbo.APP_IncidentCase
            SET
                ComplaintText = ?,
                ImmediateAction = ?,
                TakenAction = ?,
                FeedbackRecievedDate = ?,
                IncidentDate = ?,
                PatientName = ?,
                IssuingOrgUnitID = ?,
                isINPatient = ?,
                ClinicalRiskTypeID = ?,
                FeedbackIntentTypeID = ?,
                BuildingID = ?,
                DomainID = ?,
                CategoryID = ?,
                SubCategoryID = ?,
                ClassificationID = ?,
                SeverityID = ?,
                StageID = ?,
                HarmLevelID = ?,
                SourceID = ?,
                CaseStatusID = ?,
                ExplanationStatusID = ?,
                UpdatedAt = GETDATE()
            WHERE IncidentRequestCaseID = ?
        """
        # -----------------------------
        # FSM SAFETY CHECK
        # -----------------------------
        # Only validate FSM state when an explicit FSM command was given.
        # Simple data edits (no command) should NOT be blocked by state checks.
        if command:
            allowed = {
                (3,4),  # Closed + No Explanation Needed
                (1,1),  # Open + Waiting
                (2,2),  # In Progress + Responded
                (3,2),  # Closed + Responded
                (3,3),  # Closed + Forcibly Closed
            }

            if (new_case_status_id, new_explanation_status_id) not in allowed:
                conn.rollback()
                return {"success": False, "error": "FSM_VIOLATION", "message": "Illegal state combination"}

        cursor.execute(update_query, (
            data.get('complaint_text'),
            data.get('immediate_action'),
            data.get('taken_action'),
            data.get('feedback_received_date'),
            data.get('incident_date') or data.get('feedback_received_date'),
            data.get('patient_name'),
            data.get('issuing_department_id'),
            is_inpatient_val,
            clinical_risk_type_id,
            feedback_intent_type_id,
            building_id,
            data.get('domain_id'),
            data.get('category_id'),
            data.get('subcategory_id'),
            data.get('classification_id'),
            data.get('severity_id'),
            data.get('stage_id'),
            data.get('harm_id'),
            data.get('source_id'),
            new_case_status_id,
            new_explanation_status_id,
            record_id
        ))

        # Fix: Handle Target Departments update (NO nested connection calls!)
        if 'target_department_ids' in data:
            if not data['target_department_ids'] or len(data['target_department_ids']) == 0:
                pass  # preserve existing departments
            else:
                cursor.execute(
                    "DELETE FROM dbo.APP_IncidentCaseTargetDepartment WHERE IncidentRequestCaseID = ?",
                    (record_id,)
                )
                for idx, dept_id in enumerate(data['target_department_ids']):
                    is_primary = 1 if idx == 0 else 0
                    cursor.execute(
                        """
                        INSERT INTO dbo.APP_IncidentCaseTargetDepartment (
                            IncidentRequestCaseID, DepartmentID, IsPrimary, AssignedByUserID
                        )
                        VALUES (?, ?, ?, ?)
                        """,
                        (record_id, dept_id, is_primary, 1)
                    )

        # -----------------------------
        # Employee Linkage Update (Supervisor / Worker)
        # -----------------------------
        if 'employees' in data and data['employees'] is not None:
            emp_list = data['employees']
            if isinstance(emp_list, list):
                cursor.execute(
                    "DELETE FROM dbo.APP_IncidentCaseEmployee WHERE IncidentRequestCaseID = ?",
                    (record_id,)
                )
                primary_assigned = False
                for emp in emp_list:
                    emp_id = emp.get('employee_id')
                    if not emp_id:
                        continue
                    # SESSION C3: resolve an opaque external id to a real
                    # reserve int before it goes into this int FK column —
                    # see materialize_employee_id's docstring.
                    resolved_employee_id = materialize_employee_id(emp_id, emp.get('full_name', ''))
                    if not resolved_employee_id:
                        continue
                    is_primary = 1 if not primary_assigned else 0
                    cursor.execute(
                        """
                        INSERT INTO dbo.APP_IncidentCaseEmployee (
                            EmployeeID, IncidentRequestCaseID, IsPrimary, FullName, AssignedByUserID, AssignedAt
                        )
                        VALUES (?, ?, ?, ?, ?, GETDATE())
                        """,
                        (resolved_employee_id, record_id, is_primary, emp.get('full_name', ''), 1)
                    )
                    primary_assigned = True

        # -----------------------------
        # Doctor Linkage Update
        # -----------------------------
        if 'doctors' in data and data['doctors'] is not None:
            doc_list = data['doctors']
            if isinstance(doc_list, list) and len(doc_list) > 0:
                cursor.execute(
                    "DELETE FROM dbo.APP_IncidentCaseDoctor WHERE IncidentRequestCaseID = ?",
                    (record_id,)
                )
                primary_assigned = False
                for doc in doc_list:
                    doc_id = doc.get('doctor_id')
                    if not doc_id:
                        continue
                    # SESSION C2: resolve an opaque external id to a real
                    # reserve int before it goes into this int FK column —
                    # see materialize_doctor_id's docstring.
                    resolved_doctor_id = materialize_doctor_id(doc_id, doc.get('doctor_name', ''))
                    if not resolved_doctor_id:
                        continue
                    is_primary = 1 if not primary_assigned else 0
                    cursor.execute(
                        """
                        INSERT INTO dbo.APP_IncidentCaseDoctor (
                            DoctorID, IncidentRequestCaseID, IsPrimary, DoctorName, AssignedByUserID, AssignedAt
                        )
                        VALUES (?, ?, ?, ?, ?, GETDATE())
                        """,
                        (resolved_doctor_id, record_id, is_primary, doc.get('doctor_name', ''), 1)
                    )
                    primary_assigned = True

        # -----------------------------
        # Sync parent APP_Incident
        # -----------------------------
        cursor.execute(
            "SELECT incident_id FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?",
            (record_id,)
        )
        inc_row = cursor.fetchone()
        if inc_row and inc_row.incident_id:
            cursor.execute(
                """
                UPDATE dbo.APP_Incident
                SET
                    patient_name        = ?,
                    feedback_intent_type_id = ?,
                    issuing_org_unit_id = ?,
                    complaint_summary   = ?,
                    building_id         = ?,
                    is_inpatient        = ?
                WHERE incident_id = ?
                """,
                (
                    data.get('patient_name'),
                    data.get('feedback_intent_type_id'),
                    data.get('issuing_department_id'),
                    data.get('complaint_text'),
                    building_id,
                    is_inpatient_val,
                    inc_row.incident_id,
                )
            )

        # -----------------------------------------------------------
        # ML: upsert the one current ml.CaseTrainingRecord row (replacing
        # the legacy SQLite append-only write) + register a job so the
        # worker knows whether embeddings need recomputing. Same
        # transaction as the case update, for the same reason as
        # create_case()'s job registration — this is a trivial write
        # against data we already hold, not the (separate, non-blocking)
        # embedding computation itself.
        # -----------------------------------------------------------
        text_changed = (
            (data.get('complaint_text') or '') != (current_complaint_text or '') or
            (data.get('immediate_action') or '') != (current_immediate_action or '') or
            (data.get('taken_action') or '') != (current_taken_action or '')
        )
        ml_fields = _build_ml_training_fields(data)
        if ml_fields:
            ml_case_training_db.upsert_case_training_record(cursor, record_id, ml_fields)
            job_type = 'TextChanged' if text_changed else 'LabelsChanged'
            ml_embedding_job_db.insert_embedding_job(cursor, record_id, job_type)

        conn.commit()

        return {
            "success": True,
            "message": "Record updated successfully",
            "message_ar": "تم تحديث السجل بنجاح",
            "record_id": record_id,
            "id": record_id,
            "updated_at": datetime.now().isoformat()
        }

    except Exception as e:
        if conn:
            conn.rollback()
        return {
            "success": False,
            "error": "DATABASE_ERROR",
            "message": f"Failed to update record: {str(e)}",
            "message_ar": f"فشل في تحديث السجل: {str(e)}"
        }

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
