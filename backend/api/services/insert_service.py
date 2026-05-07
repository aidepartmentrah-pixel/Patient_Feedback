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
        

def create_record(data: Dict[str, Any]) -> Dict[str, Any]:
    conn = None
    cursor = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        # -----------------------------
        # Required fields validation
        # -----------------------------
        required_fields = [
            'complaint_text',
            'feedback_received_date',
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
            'taken_action',
            'patient_name',
            'is_inpatient',
            'source_id'
        ]

        for field in required_fields:
            if field not in data or data[field] is None or data[field] == '':
                return {
                    "success": False,
                    "error": "VALIDATION_ERROR",
                    "message": f"{field.replace('_', ' ').title()} is required",
                    "message_ar": f"حقل {field} مطلوب",
                    "field": field
                }        
        # Building: Require either building_id or building_code
        if not data.get('building_id') and not data.get('building_code'):
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
        
        print(f"[DEBUG] is_inpatient raw: {data.get('is_inpatient')} (type: {type(raw).__name__}) -> stored: {is_inpatient_val}")
        
        clinical_risk_type_id = data.get('clinical_risk_type_id')
        feedback_intent_type_id = data.get('feedback_intent_type_id')

        # Validate doctors exist (check both hospital and reserve tables)
        if data.get('doctors'):
            for doc in data['doctors']:
                doc_id = doc.get('doctor_id')
                if not doc_id:
                    continue
                # Use UNION to check both hospital (APP_LOOKUP_DOCTOR) and reserve (APP_RESERVE_DOCTOR) tables
                cursor.execute("""
                    SELECT COUNT(*) FROM (
                        SELECT DoctorID FROM dbo.APP_LOOKUP_DOCTOR WHERE DoctorID = ?
                        UNION ALL
                        SELECT DoctorID FROM dbo.APP_RESERVE_DOCTOR WHERE DoctorID = ?
                    ) AS combined
                """, (doc_id, doc_id))
                if cursor.fetchone()[0] == 0:
                    return {
                        "success": False,
                        "error": "INVALID_REFERENCE",
                        "message": f"Doctor ID {doc_id} does not exist",
                        "message_ar": f"رقم الطبيب {doc_id} غير موجود",
                        "field": "doctors"
                    }

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
        # FSM LOGIC: Explanation Workflow
        # -----------------------------
        # Three paths:
        # 1. Red Flag (2) OR Never Event (3) -> ALWAYS needs explanation
        # 2. Ordinary with RequiresExplanation=True -> needs explanation
        # 3. Ordinary with RequiresExplanation=False -> no explanation needed
        
        is_red_flag = clinical_risk_type_id == 2
        is_never_event = clinical_risk_type_id == 3
        requires_explanation = data.get('requires_explanation')
        
        print(f"[DEBUG] requires_explanation from frontend: {requires_explanation} (type: {type(requires_explanation).__name__})")
        
        # Convert to BIT value (0 or 1)
        if isinstance(requires_explanation, bool):
            requires_explanation_bit = 1 if requires_explanation else 0
        elif isinstance(requires_explanation, str):
            requires_explanation_bit = 1 if requires_explanation.lower() in ('true', '1', 'yes') else 0
        elif isinstance(requires_explanation, int):
            requires_explanation_bit = 1 if requires_explanation == 1 else 0
        else:
            requires_explanation_bit = 0
        
        print(f"[DEBUG] requires_explanation_bit to be stored: {requires_explanation_bit}")
        
        if is_red_flag or is_never_event or requires_explanation_bit:
            # Path 1 & 2: Open + Waiting (S0)
            case_status_id = 1  # Open
            explanation_status_id = 1  # Waiting
            print(f"[FSM] Case requires explanation: RedFlag={is_red_flag}, NeverEvent={is_never_event}, RequiresExplanation={requires_explanation_bit}")
        else:
            # Path 3: Closed + No Explanation Needed (S4)
            case_status_id = 3  # Closed
            explanation_status_id = 4  # No Explanation Needed
            print(f"[FSM] Case does not require explanation")

        payload = {
            "ComplaintText": data.get('complaint_text'),
            "ImmediateAction": data.get('immediate_action'),
            "TakenAction": data.get('taken_action'),
            "FeedbackRecievedDate": data.get('feedback_received_date'),
            "PatientName": data.get('patient_name'),
            "IssuingOrgUnitID": data.get('issuing_department_id'),
            "CreatedByUserID": 1,
            "isINPatient": is_inpatient_val,
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
        }

        new_id = create_incident_case(payload)

        # -------------------------------------------
        # ML INSERT HOOK (SAFE / NON-BLOCKING)
        # With Embedding Generation via Wrapper
        # -------------------------------------------
        try:
            # Call the ML wrapper function with full data dict
            from backend.ml_mapping import add_corrected_record_to_ml
            add_corrected_record_to_ml(data)
        except Exception as e:
            # Log only — never interrupt main flow
            print(f"[ML INSERT WARNING] {str(e)}")
            import traceback
            traceback.print_exc()

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
                    add_employee_to_case(
                        incident_id=new_id,
                        employee_id=emp_id,
                        assigned_by_user_id=1,
                        full_name=emp.get('full_name', ''),
                        is_primary=(not primary_assigned)
                    )
                    primary_assigned = True
                except Exception as e:
                    print(f"[WARN] Failed to link employee {emp_id}: {str(e)}")

        # db_layer functions commit internally; nothing to commit here

        # -------------------------------------------
        # API V2 ADAPTER HOOK (SAFE / NON-BLOCKING)
        # Automatically create subcases for this incident
        # -------------------------------------------
        try:
            from backend.api_v2.services.case_creation_service import create_subcases_for_incident
            # Note: current_user is not available in this legacy code context
            # We'll pass None and the service will handle it gracefully
            create_subcases_for_incident(new_id, current_user=None)
        except Exception as e:
            # Log only — never interrupt main flow
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
            "incident_id": new_id,  # For compatibility
            "status_id": 3,
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


def update_record(record_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
    conn = None
    cursor = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        # DEBUG: Log incoming payload
        print(f"\n{'='*60}")
        print(f"[DEBUG] UPDATE RECORD {record_id}")
        print(f"[DEBUG] Incoming payload keys: {list(data.keys())}")
        print(f"[DEBUG] building = {data.get('building')}")
        print(f"[DEBUG] building_id = {data.get('building_id')}")
        print(f"[DEBUG] building_code = {data.get('building_code')}")
        print(f"{'='*60}\n")

        # Validation
        required_fields = [
            'complaint_text',
            'feedback_received_date',
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
            'taken_action',
            'patient_name',
            'is_inpatient',
            'source_id'
        ]
        
        # Load current FSM state + ClinicalRiskTypeID
        cursor.execute(
            "SELECT CaseStatusID, ExplanationStatusID, ClinicalRiskTypeID FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?",
            (record_id,)
        )
        row = cursor.fetchone()
        if not row:
            return {"success": False, "error": "NOT_FOUND", "message": "Record not found"}
        
        current_case_status_id = row.CaseStatusID
        current_explanation_status_id = row.ExplanationStatusID
        current_clinical_risk_type_id = row.ClinicalRiskTypeID

        # Fix: Allow False/0 values, only reject None/empty string
        for field in required_fields:
            if field not in data or data[field] is None or data[field] == "":
                return {
                    "success": False,
                    "error": "VALIDATION_ERROR",
                    "message": f"{field.replace('_', ' ').title()} is required",
                    "message_ar": f"حقل {field} مطلوب",
                    "field": field
                }
        
        # Building: Require either building_id, building_code, or existing building in DB
        # Fetch existing building as fallback before validation
        existing_building_id_prefetch = None
        try:
            cursor.execute(
                "SELECT BuildingID FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?",
                (record_id,)
            )
            brow = cursor.fetchone()
            if brow:
                existing_building_id_prefetch = brow.BuildingID
        except Exception:
            pass

        if not data.get('building_id') and not data.get('building_code') and not existing_building_id_prefetch:
            return {
                "success": False,
                "error": "VALIDATION_ERROR",
                "message": "Either building_id or building_code is required",
                "message_ar": "يجب توفير رقم المبنى أو رمز المبنى",
                "field": "building_id"
            }
        
        # Block changing Clinical Risk Type
        if data.get('clinical_risk_type_id') is not None and int(data.get('clinical_risk_type_id')) != int(current_clinical_risk_type_id):
            return {
                "success": False,
                "error": "IMMUTABLE_FIELD",
                "message": "Clinical Risk Type cannot be changed after creation",
                "message_ar": "لا يمكن تغيير نوع المخاطر السريرية بعد الإنشاء",
                "field": "clinical_risk_type_id"
            }

        # Validate foreign keys (same as create_record)
        validations = [
            ('domain_id', 'Domain', 'dbo.APP_LOOKUP_DOMAIN', 'DomainID', 'القطاع'),
            ('category_id', 'Category', 'dbo.APP_LOOKUP_CATEGORY', 'CategoryID', 'الفئة'),
            ('severity_id', 'Severity', 'dbo.APP_LOOKUP_SEVERITY', 'SeverityID', 'مستوى الخطورة')
        ]

        # -----------------------------
        # FSM TRANSITION LOGIC
        # -----------------------------
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
        
        print(f"[DEBUG] is_inpatient raw: {data.get('is_inpatient')} (type: {type(raw).__name__}) -> stored: {is_inpatient_val}")
        
        clinical_risk_type_id = data.get('clinical_risk_type_id')
        feedback_intent_type_id = data.get('feedback_intent_type_id')

        # -----------------------------
        # BUILDING RESOLUTION WITH FALLBACK
        # -----------------------------
        # Step 1: Get existing BuildingID from database
        cursor.execute(
            "SELECT BuildingID FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?",
            (record_id,)
        )
        existing_row = cursor.fetchone()
        existing_building_id = existing_row.BuildingID if existing_row else None
        print(f"[DEBUG] Building: Existing BuildingID in DB = {existing_building_id}")

        # DEBUG: Show all buildings in lookup table
        try:
            cursor.execute("SELECT BuildingID, BuildingCode FROM dbo.APP_LOOKUP_BUILDING ORDER BY BuildingID")
            all_buildings = cursor.fetchall()
            print(f"[DEBUG] Building: Available codes in lookup table:")
            for b in all_buildings:
                print(f"  - BuildingID={b.BuildingID}, BuildingCode='{b.BuildingCode}'")
        except Exception as e:
            print(f"[DEBUG] Building: Could not list lookup table: {e}")

        # Step 2: Try to resolve new BuildingID from payload
        building_id = None
        
        # Option A: Direct BuildingID provided
        if data.get('building_id'):
            building_id = data.get('building_id')
            print(f"[DEBUG] Building: Direct BuildingID provided = {building_id}")
        
        # Option B: BuildingCode provided - lookup in database
        elif data.get('building_code'):
            code = str(data.get('building_code')).strip().upper()
            print(f"[DEBUG] Building: Incoming BuildingCode = '{code}'")
            
            try:
                cursor.execute(
                    """
                    SELECT TOP 1 BuildingID, BuildingCode
                    FROM dbo.APP_LOOKUP_BUILDING
                    WHERE UPPER(BuildingCode) = ?
                    ORDER BY BuildingID
                    """,
                    (code,)
                )
                row = cursor.fetchone()
                if row:
                    building_id = row.BuildingID
                    print(f"[DEBUG] Building: Resolved '{code}' → BuildingID = {building_id} (Actual code in DB: '{row.BuildingCode}')")
                else:
                    print(f"[DEBUG] Building: Code '{code}' NOT FOUND in APP_LOOKUP_BUILDING")
            except Exception as e:
                print(f"[DEBUG] Building: Lookup query failed: {e}")
        
        # Step 3: Fallback to existing BuildingID if resolution failed
        if building_id is None:
            building_id = existing_building_id
            print(f"[DEBUG] Building: Using fallback to existing BuildingID = {building_id}")
        
        print(f"[DEBUG] Building: FINAL BuildingID for update = {building_id}")

        update_query = """
            UPDATE dbo.APP_IncidentCase
            SET
                ComplaintText = ?,
                ImmediateAction = ?,
                TakenAction = ?,
                FeedbackRecievedDate = ?,
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
                ExplanationStatusID = ?
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
            # If empty list provided, skip update (preserve existing departments)
            if not data['target_department_ids'] or len(data['target_department_ids']) == 0:
                print(f"[DEBUG] target_department_ids is empty, preserving existing departments")
            else:
                # Remove existing target departments (same cursor - no deadlock)
                cursor.execute(
                    "DELETE FROM dbo.APP_IncidentCaseTargetDepartment WHERE IncidentRequestCaseID = ?",
                    (record_id,)
                )
                
                # Add new target departments (same cursor - no deadlock)
                for idx, dept_id in enumerate(data['target_department_ids']):
                    is_primary = 1 if idx == 0 else 0
                    cursor.execute(
                        """
                        INSERT INTO dbo.APP_IncidentCaseTargetDepartment (
                            IncidentRequestCaseID,
                            DepartmentID,
                            IsPrimary,
                            AssignedByUserID
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
                # Remove existing employee links
                cursor.execute(
                    "DELETE FROM dbo.APP_IncidentCaseEmployee WHERE IncidentRequestCaseID = ?",
                    (record_id,)
                )
                
                # Add new employee links
                primary_assigned = False
                for emp in emp_list:
                    emp_id = emp.get('employee_id')
                    if not emp_id:
                        continue
                    is_primary = 1 if not primary_assigned else 0
                    cursor.execute(
                        """
                        INSERT INTO dbo.APP_IncidentCaseEmployee (
                            EmployeeID,
                            IncidentRequestCaseID,
                            IsPrimary,
                            FullName,
                            AssignedByUserID,
                            AssignedAt
                        )
                        VALUES (?, ?, ?, ?, ?, GETDATE())
                        """,
                        (emp_id, record_id, is_primary, emp.get('full_name', ''), 1)
                    )
                    primary_assigned = True
                print(f"[DEBUG] Updated {len(emp_list)} employees for record {record_id}")

        # -----------------------------
        # Doctor Linkage Update
        # -----------------------------
        if 'doctors' in data and data['doctors'] is not None:
            doc_list = data['doctors']
            if isinstance(doc_list, list) and len(doc_list) > 0:
                # Remove existing doctor links
                cursor.execute(
                    "DELETE FROM dbo.APP_IncidentCaseDoctor WHERE IncidentRequestCaseID = ?",
                    (record_id,)
                )
                
                # Add new doctor links
                primary_assigned = False
                for doc in doc_list:
                    doc_id = doc.get('doctor_id')
                    if not doc_id:
                        continue
                    is_primary = 1 if not primary_assigned else 0
                    cursor.execute(
                        """
                        INSERT INTO dbo.APP_IncidentCaseDoctor (
                            DoctorID,
                            IncidentRequestCaseID,
                            IsPrimary,
                            DoctorName,
                            AssignedByUserID,
                            AssignedAt
                        )
                        VALUES (?, ?, ?, ?, ?, GETDATE())
                        """,
                        (doc_id, record_id, is_primary, doc.get('doctor_name', ''), 1)
                    )
                    primary_assigned = True
                print(f"[DEBUG] Updated {len(doc_list)} doctors for record {record_id}")

        conn.commit()

        # ML INSERT HOOK (SAFE / NON-BLOCKING)
        try:
            from backend.ml_mapping import add_corrected_record_to_ml
            add_corrected_record_to_ml(data)
        except Exception as e:
            print(f"[ML UPDATE WARNING] {str(e)}")
            import traceback
            traceback.print_exc()

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


def create_incident_with_cases(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create one parent incident with one or more operational cases.
    Compatibility-first: each case is still created through legacy create_record flow.
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

    if not common.get("feedback_intent_type_id"):
        return {
            "success": False,
            "error": "VALIDATION_ERROR",
            "message": "Feedback intent type is required",
            "message_ar": "نوع نية الملاحظة مطلوب",
            "field": "feedback_intent_type_id",
        }

    incident_id = create_incident_parent(
        {
            "patient_name": common.get("patient_name"),
            "primary_doctor_name": common.get("primary_doctor_name"),
            "primary_worker_name": common.get("primary_worker_name"),
            "feedback_intent_type_id": common.get("feedback_intent_type_id"),
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
        if len(target_ids) != 1:
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
            "issuing_department_id": case_data.get("issuing_department_id") or common.get("issuing_department_id"),
            "feedback_intent_type_id": common.get("feedback_intent_type_id"),
            "patient_name": common.get("patient_name") or case_data.get("patient_name") or "",
            "is_inpatient": common.get("is_inpatient", True),
            "source_id": case_data.get("source_id") or common.get("source_id"),
            "building_id": case_data.get("building_id") or common.get("building_id"),
            "doctors": case_data.get("doctors") or common.get("doctors"),
            "employees": case_data.get("employees") or common.get("employees"),
        }

        created = create_record(case_payload)
        if not created.get("success"):
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
