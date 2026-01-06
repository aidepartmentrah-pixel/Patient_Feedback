"""
Insert Service
Handles business logic for creating new incident/feedback records.
"""

from datetime import datetime
from typing import Dict, Any
from core.database import get_connection
from backend.api.db_layer.incident_case import create_incident_case
from backend.api.db_layer.incident_case_target_department import add_target_department
from backend.api.db_layer.incident_case_doctor import add_doctor_to_case
        

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
            'harm_id'
        ]

        for field in required_fields:
            if field not in data or not data[field]:
                return {
                    "success": False,
                    "error": "VALIDATION_ERROR",
                    "message": f"{field.replace('_', ' ').title()} is required",
                    "message_ar": f"حقل {field} مطلوب",
                    "field": field
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
        is_inpatient_val = 1 if bool(data.get('is_inpatient', True)) else 0
        clinical_risk_type_id = data.get('clinical_risk_type_id') or 1
        feedback_intent_type_id = data.get('feedback_intent_type_id') or 1

        # Validate doctors exist (skip entries without IDs)
        if data.get('doctors'):
            for doc in data['doctors']:
                doc_id = doc.get('doctor_id')
                if not doc_id:
                    continue
                cursor.execute("SELECT COUNT(*) FROM APP_VIEWTABLE_VW_DOCTORS WHERE DoctorID = ?", (doc_id,))
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
        if clinical_risk_type_id in (2, 3):  # Red Flag or Never Event
            case_status_id = 1  # Open
        else:
            case_status_id = 3  # Closed
        if clinical_risk_type_id in (2, 3):  # Red Flag or Never Event
            explanation_status_id = 1  # Waiting
        else:
            explanation_status_id = 4  # No Explanation Needed

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
            "CaseStatusID": 3,
            "SourceID": data.get('source_id'),
            "ExplanationStatusID": explanation_status_id,
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

        # Employees linkage not implemented (no case link in schema provided)

        # db_layer functions commit internally; nothing to commit here

        record_id = f"REC-{datetime.now().year}-{str(new_id).zfill(4)}"

        return {
            "success": True,
            "message": "Record created successfully",
            "message_ar": "تم إنشاء السجل بنجاح",
            "record_id": record_id,
            "id": new_id,
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
    """Update an existing incident case record."""
    conn = None
    cursor = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

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
            'clinical_risk_type_id',
            'feedback_intent_type_id'
        ]

        for field in required_fields:
            if field not in data or not data[field]:
                return {
                    "success": False,
                    "error": "VALIDATION_ERROR",
                    "message": f"{field.replace('_', ' ').title()} is required",
                    "message_ar": f"حقل {field} مطلوب",
                    "field": field
                }

        # Validate foreign keys (same as create_record)
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

        # Update main record
        is_inpatient_val = 1 if bool(data.get('is_inpatient', True)) else 0
        clinical_risk_type_id = data.get('clinical_risk_type_id') or 1
        feedback_intent_type_id = data.get('feedback_intent_type_id') or 1

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
                ExplanationStatusID = ?
            WHERE IncidentRequestCaseID = ?
        """

        explanation_status_id = 1 if clinical_risk_type_id in (2, 3) else 4

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
            data.get('building_id'),
            data.get('domain_id'),
            data.get('category_id'),
            data.get('subcategory_id'),
            data.get('classification_id'),
            data.get('severity_id'),
            data.get('stage_id'),
            data.get('harm_id'),
            data.get('source_id'),
            explanation_status_id,
            record_id
        ))

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
