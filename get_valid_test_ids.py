"""
Get Valid Test Data for Employee Linkage Test
Finds valid IDs from lookup tables
"""
from backend.core.database import get_connection


def get_valid_ids():
    """Retrieve valid IDs from lookup tables"""
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        print("=" * 60)
        print("Finding Valid Test Data")
        print("=" * 60)
        
        # Get valid domain
        cursor.execute("SELECT TOP 1 DomainID FROM dbo.APP_LOOKUP_DOMAIN ORDER BY DomainID")
        domain = cursor.fetchone()
        domain_id = domain.DomainID if domain else None
        print(f"\n✅ Domain ID: {domain_id}")
        
        # Get valid category for that domain
        cursor.execute(
            "SELECT TOP 1 CategoryID FROM dbo.APP_LOOKUP_CATEGORY WHERE DomainID = ? ORDER BY CategoryID",
            (domain_id,)
        )
        category = cursor.fetchone()
        category_id = category.CategoryID if category else None
        print(f"✅ Category ID: {category_id}")
        
        # Get valid subcategory for that category
        cursor.execute(
            "SELECT TOP 1 SubCategoryID FROM dbo.APP_LOOKUP_SUBCATEGORY WHERE CategoryID = ? ORDER BY SubCategoryID",
            (category_id,)
        )
        subcategory = cursor.fetchone()
        subcategory_id = subcategory.SubCategoryID if subcategory else None
        print(f"✅ Subcategory ID: {subcategory_id}")
        
        # Get valid classification for that subcategory
        cursor.execute(
            "SELECT TOP 1 ClassificationID FROM dbo.APP_LOOKUP_CLASSIFICATION WHERE SubCategoryID = ? ORDER BY ClassificationID",
            (subcategory_id,)
        )
        classification = cursor.fetchone()
        classification_id = classification.ClassificationID if classification else None
        print(f"✅ Classification ID: {classification_id}")
        
        # Get valid severity
        cursor.execute("SELECT TOP 1 SeverityID FROM dbo.APP_LOOKUP_SEVERITY ORDER BY SeverityID")
        severity = cursor.fetchone()
        severity_id = severity.SeverityID if severity else None
        print(f"✅ Severity ID: {severity_id}")
        
        # Get valid stage
        cursor.execute("SELECT TOP 1 StageID FROM dbo.APP_LOOKUP_CASE_STAGE ORDER BY StageID")
        stage = cursor.fetchone()
        stage_id = stage.StageID if stage else None
        print(f"✅ Stage ID: {stage_id}")
        
        # Get valid harm level
        cursor.execute("SELECT TOP 1 HarmID FROM dbo.APP_LOOKUP_HARM_LEVEL ORDER BY HarmID")
        harm = cursor.fetchone()
        harm_id = harm.HarmID if harm else None
        print(f"✅ Harm ID: {harm_id}")
        
        # Get valid building
        cursor.execute("SELECT TOP 1 BuildingID FROM dbo.APP_LOOKUP_BUILDING ORDER BY BuildingID")
        building = cursor.fetchone()
        building_id = building.BuildingID if building else None
        print(f"✅ Building ID: {building_id}")
        
        # Get valid source
        cursor.execute("SELECT TOP 1 SourceID FROM dbo.APP_LOOKUP_SOURCE ORDER BY SourceID")
        source = cursor.fetchone()
        source_id = source.SourceID if source else None
        print(f"✅ Source ID: {source_id}")
        
        # Get valid issuing department (org unit)
        cursor.execute("SELECT TOP 1 OrgUnitID FROM dbo.APP_LOOKUP_ORGUNIT ORDER BY OrgUnitID")
        org_unit = cursor.fetchone()
        org_unit_id = org_unit.OrgUnitID if org_unit else None
        print(f"✅ Issuing Department ID: {org_unit_id}")
        
        # Get valid target department
        cursor.execute("SELECT TOP 1 OrgUnitID FROM dbo.APP_LOOKUP_ORGUNIT WHERE OrgUnitID <> ? ORDER BY OrgUnitID", (org_unit_id,))
        target_dept = cursor.fetchone()
        target_dept_id = target_dept.OrgUnitID if target_dept else org_unit_id
        print(f"✅ Target Department ID: {target_dept_id}")
        
        print("\n" + "=" * 60)
        print("Test Payload Template:")
        print("=" * 60)
        print(f"""
payload = {{
    "complaint_text": "Test incident with employee linkage",
    "feedback_received_date": "2026-02-12",
    "issuing_department_id": {org_unit_id},
    "domain_id": {domain_id},
    "category_id": {category_id},
    "subcategory_id": {subcategory_id},
    "classification_id": {classification_id},
    "severity_id": {severity_id},
    "stage_id": {stage_id},
    "harm_id": {harm_id},
    "requires_explanation": True,
    "clinical_risk_type_id": 1,
    "feedback_intent_type_id": 1,
    "immediate_action": "Test immediate action",
    "taken_action": "Test taken action",
    "patient_name": "Test Patient",
    "is_inpatient": True,
    "source_id": {source_id},
    "building_id": {building_id},
    "target_department_ids": [{target_dept_id}],
    "employees": [
        {{"employee_id": 101, "full_name": "Ahmed Mohamed"}},
        {{"employee_id": 102, "full_name": "Sara Ahmed"}}
    ]
}}
        """)
        
        return {
            "domain_id": domain_id,
            "category_id": category_id,
            "subcategory_id": subcategory_id,
            "classification_id": classification_id,
            "severity_id": severity_id,
            "stage_id": stage_id,
            "harm_id": harm_id,
            "building_id": building_id,
            "source_id": source_id,
            "issuing_department_id": org_unit_id,
            "target_department_id": target_dept_id
        }
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


if __name__ == "__main__":
    get_valid_ids()
