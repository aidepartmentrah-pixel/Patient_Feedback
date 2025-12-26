from backend.api.db_layer.incident_case import create_incident_case
from backend.api.db_layer.incident_case_feedback import (
    create_incident_case_feedback,
    get_incident_case_feedback,
    update_incident_case_feedback,
)

# -----------------------------
# BASE INCIDENT
# -----------------------------

BASE_INCIDENT_DATA = {
    "ComplaintText": "Feedback test",
    "ImmediateAction": None,
    "TakenAction": None,
    "FeedbackRecievedDate": None,
    "PatientName": "Test Patient",
    "IssuingOrgUnitID": 1,
    "CreatedByUserID": 1,
    "InOut": "IN",

    "ClinicalRiskTypeID": 1,
    "FeedbackIntentTypeID": 1,
    "DomainID": 1,
    "CategoryID": 1,
    "SubCategoryID": 1,
    "ClassificationID": 1,
    "StageID": 1,
    "HarmLevelID": 1,
    "CaseStatusID": 1,
}


# -----------------------------
# TESTS
# -----------------------------

def test_create_and_get_feedback():
    incident_id = create_incident_case(BASE_INCIDENT_DATA)

    create_incident_case_feedback(
        incident_id,
        {
            "Cause_Staff_Training": 1,
            "Cause_Staff_OtherText": "Lack of refresher courses",
            "DepartmentExplanationStatusID": 1,
        },
        created_by_user_id=1,
    )

    feedback = get_incident_case_feedback(incident_id)
    assert feedback is not None
    assert feedback["Cause_Staff_Training"] == 1

test_create_and_get_feedback()


def test_update_feedback():
    incident_id = create_incident_case(BASE_INCIDENT_DATA)

    create_incident_case_feedback(
        incident_id,
        {"DepartmentExplanationStatusID": 1},
        created_by_user_id=1,
    )

    update_incident_case_feedback(
        incident_id,
        {
            "Preventive_TrainingPrograms": 1,
            "Preventive_OtherText": "Quarterly workshops",
        },
    )

    feedback = get_incident_case_feedback(incident_id)
    assert feedback["Preventive_TrainingPrograms"] == 1

test_update_feedback()
