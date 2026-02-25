from backend.api.services.insert_service import create_record
from backend.api.services.explanation_service import get_case_explanation_details, submit_explanation
from backend.api.db_layer.incident_case import hard_delete_incident_case

# Create Red Flag case
r = create_record({
    'complaint_text': 'Test',
    'feedback_received_date': '2026-01-19',
    'issuing_department_id': 43,
    'domain_id': 2,
    'category_id': 5,
    'subcategory_id': 13,
    'classification_id': 106,
    'severity_id': 2,
    'stage_id': 2,
    'harm_id': 2,
    'building_id': 2,
    'source_id': 4,
    'clinical_risk_type_id': 2  # Red Flag
})

cid = r['incident_id']
print(f'Case {cid} created')

v = get_case_explanation_details(cid)
print(f'ClinicalRiskTypeID: {v["case"].get("ClinicalRiskTypeID")}')
print(f'requires_explanation (calculated): {v["validation"]["requires_explanation"]}')
print(f'is_red_flag_or_never_event: {v["validation"].get("is_red_flag_or_never_event")}')
print(f'can_submit: {v["validation"]["can_submit_explanation"]}')
print(f'ExplanationStatusName: {v["case"].get("ExplanationStatusName")}')
print(f'CaseStatusName: {v["case"].get("CaseStatusName")}')

print('\nAttempting to submit explanation...')
result = submit_explanation(
    case_id=cid,
    explanation_text="This is a test explanation that is long enough to pass validation",
    user_id=1
)

print(f'Result: {result}')

hard_delete_incident_case(cid)
print(f'Cleaned up case {cid}')
