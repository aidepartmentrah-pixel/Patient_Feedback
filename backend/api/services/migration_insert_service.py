"""
PHASE K — Migration Insert Service

Migration-safe variant of create_record for migrating legacy cases.

This service:
- Inserts new APP_IncidentCase from migration payload
- Forces closed/no-explanation FSM state
- Does NOT create subcases
- Records mapping in APP_DataMigration_Map
- Keeps ML hook (non-blocking)
"""

from datetime import datetime
from typing import Dict, Any
from core.database import get_connection
from api.db_layer.incident_case import create_incident_case
from api.db_layer.incident_case_target_department import add_target_department
from api.db_layer.incident_case_doctor import add_doctor_to_case


def create_record_migrated(
    data: Dict[str, Any],
    legacy_case_id: int,
    migrated_by_user_id: int
) -> Dict[str, Any]:
    """
    Create new APP_IncidentCase from migration payload.
    
    This is a migration-specific variant of create_record that:
    - Forces FSM state to closed/no-explanation
    - Does NOT create subcases (legacy cases are historical)
    - Records migration mapping (prevents duplicates)
    - Keeps ML hook for model improvement
    
    Args:
        data: Case payload (same shape as create_record)
            Required fields: complaint_text, feedback_received_date,
                           issuing_department_id, domain_id, category_id,
                           subcategory_id, classification_id, severity_id,
                           stage_id, harm_id, requires_explanation,
                           clinical_risk_type_id, feedback_intent_type_id,
                           immediate_action, taken_action, patient_name,
                           is_inpatient, source_id
        
        legacy_case_id: UniqueID from IncidentRequestCase (legacy table)
        migrated_by_user_id: UserID performing migration
    
    Returns:
        Dict with:
            success: bool
            message: str
            record_id: str (formatted)
            id: int (new case ID)
            legacy_case_id: int
            migration: bool (always True)
            
            OR on error:
            success: False
            error: str (error code)
            message: str (error description)
    
    Safety:
        - Does NOT modify legacy tables
        - Does NOT create subcases
        - Validates foreign keys
        - ML hook is non-blocking
    """
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
        # Validate doctors exist
        # -----------------------------
        if data.get('doctors'):
            for doc in data['doctors']:
                doc_id = doc.get('doctor_id')
                if not doc_id:
                    continue
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

        # -----------------------------
        # Building resolution
        # -----------------------------
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
        # FSM OVERRIDE FOR MIGRATION
        # Force closed/no-explanation state
        # Migration cases are historical - already resolved
        # -----------------------------
        case_status_id = 3  # Closed
        explanation_status_id = 4  # No Explanation Required
        requires_explanation_bit = 0  # False

        print(f"[MIGRATION] Forcing FSM: CaseStatus=3 (Closed), ExplanationStatus=4 (No Explanation), RequiresExplanation=0")

        # -----------------------------
        # Convert is_inpatient to BIT
        # -----------------------------
        raw = data.get("is_inpatient", True)
        if isinstance(raw, bool):
            is_inpatient_val = 1 if raw else 0
        elif isinstance(raw, str):
            is_inpatient_val = 1 if raw.upper() in ("IN", "TRUE", "1") else 0
        elif isinstance(raw, int):
            is_inpatient_val = 1 if raw == 1 else 0
        else:
            is_inpatient_val = 1

        # -----------------------------
        # Insert main record via db_layer
        # -----------------------------
        payload = {
            "ComplaintText": data.get('complaint_text'),
            "ImmediateAction": data.get('immediate_action'),
            "TakenAction": data.get('taken_action'),
            "FeedbackRecievedDate": data.get('feedback_received_date'),
            "PatientName": data.get('patient_name'),
            "IssuingOrgUnitID": data.get('issuing_department_id'),
            "CreatedByUserID": migrated_by_user_id,
            "isINPatient": is_inpatient_val,
            "ClinicalRiskTypeID": data.get('clinical_risk_type_id'),
            "FeedbackIntentTypeID": data.get('feedback_intent_type_id'),
            "BuildingID": building_id,
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
        print(f"[MIGRATION] Created case ID: {new_id}")

        # -----------------------------
        # ML INSERT HOOK (SAFE / NON-BLOCKING)
        # Keep for model improvement
        # -----------------------------
        try:
            from ml_mapping import add_corrected_record_to_ml
            add_corrected_record_to_ml(data)
            print(f"[MIGRATION] ML hook executed for case {new_id}")
        except Exception as e:
            print(f"[MIGRATION ML WARNING] {str(e)}")
            import traceback
            traceback.print_exc()

        # -----------------------------
        # NO MAPPING INSERT HERE
        # Mapping is handled by migration orchestrator (K-SVC-6A)
        # This keeps create_record_migrated focused on case creation only
        # -----------------------------

        # -----------------------------
        # Related tables (doctors, target departments)
        # -----------------------------
        if data.get('target_department_ids'):
            for idx, dept_id in enumerate(data['target_department_ids']):
                add_target_department(
                    incident_id=new_id,
                    department_id=dept_id,
                    assigned_by_user_id=migrated_by_user_id,
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
                    assigned_by_user_id=migrated_by_user_id,
                    doctor_name=doc.get('doctor_name', ''),
                    is_primary=(not primary_assigned)
                )
                primary_assigned = True

        # -----------------------------
        # NO SUBCASE CREATION FOR MIGRATION
        # Legacy cases are historical - no active workflows
        # -----------------------------
        # REMOVED: create_subcases_for_incident(new_id)
        print(f"[MIGRATION] Skipping subcase creation (historical case)")

        record_id = f"REC-{datetime.now().year}-{str(new_id).zfill(4)}"

        return {
            "success": True,
            "message": "Migration completed successfully",
            "message_ar": "تم الترحيل بنجاح",
            "record_id": record_id,
            "id": new_id,
            "legacy_case_id": legacy_case_id,
            "migration": True,
            "created_at": datetime.now().isoformat()
        }

    except Exception as e:
        if conn:
            conn.rollback()
        return {
            "success": False,
            "error": "DATABASE_ERROR",
            "message": f"Failed to migrate record: {str(e)}",
            "message_ar": f"فشل في ترحيل السجل: {str(e)}"
        }

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
