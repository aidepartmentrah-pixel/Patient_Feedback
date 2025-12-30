"""
Insert Service
Handles business logic for creating new incident/feedback records.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from core.database import get_connection


def create_record(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new incident/feedback record in the database.
    
    Args:
        data: Dictionary containing record fields
        
    Returns:
        Dictionary with success status, record_id, and created record details
    """
    
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Validate required fields
        required_fields = ['complaint_text', 'feedback_received_date', 'domain_id', 
                          'category_id', 'severity_id']
        
        for field in required_fields:
            if field not in data or not data[field]:
                return {
                    "success": False,
                    "error": "VALIDATION_ERROR",
                    "message": f"{field.replace('_', ' ').title()} is required",
                    "message_ar": f"حقل {field} مطلوب",
                    "field": field
                }
        
        # Validate foreign key references exist
        validations = [
            ('domain_id', 'Domain', 'Domain_Lookup', 'القطاع'),
            ('category_id', 'Category', 'Category_Lookup', 'الفئة'),
            ('severity_id', 'Severity', 'Severity_Level_Lookup', 'مستوى الخطورة')
        ]
        
        for field_name, english_name, table_name, arabic_name in validations:
            if field_name in data and data[field_name]:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE Id = ?", (data[field_name],))
                if cursor.fetchone()[0] == 0:
                    return {
                        "success": False,
                        "error": "INVALID_REFERENCE",
                        "message": f"{english_name} ID {data[field_name]} does not exist",
                        "message_ar": f"{arabic_name} رقم {data[field_name]} غير موجود",
                        "field": field_name
                    }
        
        # Validate optional foreign keys if provided
        optional_validations = [
            ('subcategory_id', 'Subcategory', 'Subcategory_Lookup', 'الفئة الفرعية'),
            ('classification_id', 'Classification', 'Classification_Lookup', 'التصنيف'),
            ('stage_id', 'Stage', 'Stage_Lookup', 'المرحلة'),
            ('harm_id', 'Harm Level', 'Harm_Level_Lookup', 'مستوى الضرر'),
            ('issuing_department_id', 'Issuing Department', 'Department', 'القسم المصدر'),
            ('target_department_id', 'Target Department', 'Department', 'القسم المستهدف'),
            ('source_id', 'Source', 'Source_Lookup', 'المصدر')
        ]
        
        for field_name, english_name, table_name, arabic_name in optional_validations:
            if field_name in data and data[field_name]:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE Id = ?", (data[field_name],))
                if cursor.fetchone()[0] == 0:
                    return {
                        "success": False,
                        "error": "INVALID_REFERENCE",
                        "message": f"{english_name} ID {data[field_name]} does not exist",
                        "message_ar": f"{arabic_name} رقم {data[field_name]} غير موجود",
                        "field": field_name
                    }
        
        # Validate hierarchical relationships
        # Check if category belongs to selected domain
        if 'category_id' in data and 'domain_id' in data:
            cursor.execute(
                "SELECT COUNT(*) FROM Category_Lookup WHERE Id = ? AND Domain_Id = ?",
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
        
        # Check if subcategory belongs to selected category (if provided)
        if 'subcategory_id' in data and data['subcategory_id'] and 'category_id' in data:
            cursor.execute(
                "SELECT COUNT(*) FROM Subcategory_Lookup WHERE Id = ? AND Category_Id = ?",
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
        
        # Check if classification belongs to selected subcategory (if provided)
        if 'classification_id' in data and data['classification_id'] and 'subcategory_id' in data and data['subcategory_id']:
            cursor.execute(
                "SELECT COUNT(*) FROM Classification_Lookup WHERE Id = ? AND Subcategory_Id = ?",
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
        
        # Build INSERT statement dynamically
        fields = [
            'Patient_Feedback',
            'Immediate_Action_Taken',
            'Action_Taken',
            'Feedback_Received_Date',
            'Issuing_Department_Id',
            'Target_Department_Id',
            'Source_Id',
            'isINPatient',
            'Worker_Type',
            'Patient_Name',
            'Domain_Id',
            'Category_Id',
            'Subcategory_Id',
            'Classification_Id',
            'Severity_Level_Id',
            'Stage_Id',
            'Harm_Level_Id',
            'Improvement_Opportunity_Type',
            'Status_Id',
            'Created_At'
        ]
        
        # Map input data to database fields
        field_mapping = {
            'complaint_text': 'Patient_Feedback',
            'immediate_action': 'Immediate_Action_Taken',
            'taken_action': 'Action_Taken',
            'feedback_received_date': 'Feedback_Received_Date',
            'issuing_department_id': 'Issuing_Department_Id',
            'target_department_id': 'Target_Department_Id',
            'source_id': 'Source_Id',
            'is_inpatient': 'isINPatient',
            'worker_type': 'Worker_Type',
            'patient_name': 'Patient_Name',
            'domain_id': 'Domain_Id',
            'category_id': 'Category_Id',
            'subcategory_id': 'Subcategory_Id',
            'classification_id': 'Classification_Id',
            'severity_id': 'Severity_Level_Id',
            'stage_id': 'Stage_Id',
            'harm_id': 'Harm_Level_Id',
            'improvement_type': 'Improvement_Opportunity_Type'
        }
        
        # Prepare values for insertion
        values = []
        insert_fields = []
        
        for input_key, db_field in field_mapping.items():
            if input_key in data and data[input_key] is not None and data[input_key] != '':
                insert_fields.append(db_field)
                values.append(data[input_key])
        
        # Add isINPatient field (default to True/1 if not provided)
        if 'isINPatient' not in insert_fields:
            insert_fields.append('isINPatient')
            values.append(1)  # 1 = True (inpatient)
        
        # Add system fields
        insert_fields = []
        
        for input_key, db_field in field_mapping.items():
            if input_key in data and data[input_key] is not None and data[input_key] != '':
                insert_fields.append(db_field)
                values.append(data[input_key])
        
        # Add system fields
        insert_fields.extend(['Status_Id', 'Created_At'])
        values.extend([3, datetime.now()])  # Status_Id = 3 (In Progress)
        
        # Build and execute INSERT query
        placeholders = ','.join(['?' for _ in values])
        fields_str = ','.join(insert_fields)
        
        insert_query = f"""
            INSERT INTO Patient_Feedback_Encoded ({fields_str})
            OUTPUT INSERTED.Id
            VALUES ({placeholders})
        """
        
        cursor.execute(insert_query, values)
        new_id = cursor.fetchone()[0]
        conn.commit()
        
        # Generate record_id (e.g., REC-2024-0156)
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
