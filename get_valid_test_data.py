"""
Get valid test data IDs from the database
"""
import sys
import os

backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

print("\n" + "="*80)
print("FINDING VALID TEST DATA")
print("="*80)

# Get valid classification ID
cursor.execute("SELECT TOP 1 ClassificationID FROM dbo.APP_LOOKUP_CLASSIFICATION ORDER BY ClassificationID")
row = cursor.fetchone()
classification_id = row.ClassificationID if row else None
print(f"\nValid ClassificationID: {classification_id}")

# Get valid domain ID  
cursor.execute("SELECT TOP 1 DomainID FROM dbo.APP_LOOKUP_DOMAIN ORDER BY DomainID")
row = cursor.fetchone()
domain_id = row.DomainID if row else None
print(f"Valid DomainID: {domain_id}")

# Get valid category ID
cursor.execute("SELECT TOP 1 CategoryID FROM dbo.APP_LOOKUP_CATEGORY ORDER BY CategoryID")
row = cursor.fetchone()
category_id = row.CategoryID if row else None
print(f"Valid CategoryID: {category_id}")

# Get valid department IDs for target_department_ids (use simple values for now)
target_dept_ids = [1, 2, 3]
print(f"Valid Target Department IDs: {target_dept_ids} (using defaults)")

# Show the test data block
print("\n" + "="*80)
print("USE THIS TEST DATA:")
print("="*80)
print(f"""
test_data = {{
    'complaint_text': 'Test incident for adapter integration',
    'feedback_received_date': datetime.now().strftime('%Y-%m-%d'),
    'issuing_department_id': {target_dept_ids[0] if target_dept_ids else 1},
    'domain_id': {domain_id},
    'category_id': {category_id},
    'subcategory_id': 1,
    'classification_id': {classification_id},
    'severity_id': 1,
    'stage_id': 1,
    'harm_id': 1,
    'requires_explanation': False,
    'clinical_risk_type_id': 1,
    'feedback_intent_type_id': 1,
    'immediate_action': 'Immediate action taken',
    'taken_action': 'Action taken',
    'patient_name': 'Test Patient',
    'is_inpatient': True,
    'source_id': 1,
    'building_id': 1,
    'target_department_ids': {target_dept_ids[1:3] if len(target_dept_ids) >= 3 else [2, 3]}
}}
""")

cursor.close()
conn.close()
