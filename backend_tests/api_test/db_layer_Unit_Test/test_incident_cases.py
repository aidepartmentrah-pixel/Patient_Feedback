from backend.api.db_layer.incident_case import (
    create_incident_case,
    get_incident_case_by_id,
    list_incident_cases,
    update_incident_case,
    soft_delete_incident_case,
)

# -----------------------------
# TEST DATA
# -----------------------------

BASE_INCIDENT_DATA = {
    "ComplaintText": "Test complaint from pytest",
    "ImmediateAction": "Immediate test action",
    "TakenAction": "Taken test action",
    "FeedbackRecievedDate": None,
    "PatientName": "Test Patient",
    "IssuingOrgUnitID": 1,
    "CreatedByUserID": 1,
    "InOut": "IN",

    # LOOKUPS (must exist)
    "ClinicalRiskTypeID": 1,
    "FeedbackIntentTypeID": 1,
    "DomainID": 1,
    "CategoryID": 1,
    "SubCategoryID": 1,
    "ClassificationID": 1,
    "StageID": 1,
    "HarmLevelID": 1,
    "CaseStatusID": 1,

    # OPTIONAL
    "SeverityID": None,
    "BuildingID": None,
}


# -----------------------------
# TESTS
# -----------------------------

def test_create_and_get_incident_case():
    incident_id = create_incident_case(BASE_INCIDENT_DATA)
    assert incident_id is not None

    incident = get_incident_case_by_id(incident_id)
    assert incident["ComplaintText"] == BASE_INCIDENT_DATA["ComplaintText"]

test_create_and_get_incident_case()


def test_list_incident_cases():
    incidents = list_incident_cases()
    assert isinstance(incidents, list)

test_list_incident_cases()


def test_update_incident_case():
    incident_id = create_incident_case(BASE_INCIDENT_DATA)

    update_incident_case(
        incident_id,
        {"TakenAction": "Updated taken action"},
    )

    updated = get_incident_case_by_id(incident_id)
    assert updated["TakenAction"] == "Updated taken action"

test_update_incident_case()


def test_soft_delete_incident_case():
    incident_id = create_incident_case(BASE_INCIDENT_DATA)

    CLOSED_STATUS_ID = 2  # must exist in APP_LOOKUP_CASE_STATUS

    soft_delete_incident_case(
        incident_id,
        closed_status_id=CLOSED_STATUS_ID,
    )

    incident = get_incident_case_by_id(incident_id)
    assert incident["CaseStatusID"] == CLOSED_STATUS_ID

test_soft_delete_incident_case(1)
