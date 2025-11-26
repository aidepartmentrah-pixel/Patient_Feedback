from datetime import datetime
from api_layer import add_incident, edit_incident, fetch_incident_records, get_incident, delete_incident
from api.universal_object import UniversalIncidentRecord

# ------------------ 1. Add sample incidents ------------------
incident1_id = add_incident(
    feedback_received_date=datetime.now(),
    record_id="R001",
    patient_full_name="John Doe",
    issuing_department="Cardiology",
    target_department=101,
    source_1=1,
    feedback_type=1,
    domain="Patient Care",
    category="Service",
    sub_category="Delay",
    classification_ar="تأخير",
    classification_en="Delay",
    complaint_text="Patient experienced delay in service.",
    immediate_action="Apologized",
    taken_action="Reviewed workflow",
    severity_level=2,
    stage="Admission",
    harm_level="Low",
    status=1,
    improvement_opportunity_type=True
)

incident2_id = add_incident(
    feedback_received_date=datetime.now(),
    record_id="R002",
    patient_full_name="Jane Smith",
    issuing_department="Radiology",
    target_department=102,
    source_1=2,
    feedback_type=2,
    domain="Facilities",
    category="Cleanliness",
    sub_category="Hygiene",
    classification_ar="نظافة",
    classification_en="Cleanliness",
    complaint_text="Patient complained about hygiene in waiting area.",
    immediate_action="Cleaned area",
    taken_action="Scheduled regular cleaning",
    severity_level=3,
    stage="Care",
    harm_level="Medium",
    status=1,
    improvement_opportunity_type=False
)

print(f"Added incident IDs: {incident1_id}, {incident2_id}")

# ------------------ 2. Fetch all incidents ------------------
all_incidents = fetch_incident_records()
print(f"\nAll incidents ({len(all_incidents)}):")
for inc in all_incidents:
    print(inc["record_id"], inc["patient_full_name"])

# ------------------ 3. Edit the first incident ------------------
edited_incident = edit_incident(
    unique_id=incident1_id,
    complaint_text="Updated complaint text.",
    severity_level=1,
    status=2
)
print(f"\nEdited incident ID {incident1_id}:")
print(edited_incident)

# ------------------ 4. Fetch a single incident by record_id ------------------
single_incident = get_incident(record_id="R002")
print(f"\nFetched single incident by record_id 'R002':")
print(single_incident)

# ------------------ 5. Delete the first incident ------------------
delete_success = delete_incident(incident1_id)
print(f"\nDelete incident ID {incident1_id} success: {delete_success}")

# ------------------ 6. Verify deletion ------------------
post_delete_incidents = fetch_incident_records()
print(f"\nIncidents after deletion ({len(post_delete_incidents)}):")
for inc in post_delete_incidents:
    print(inc["record_id"], inc["patient_full_name"])
