# backend/backend_tests/api_test/db_layer_Unit_Test/test_action_items.py

from datetime import date
from backend.api.db_layer.incident_case import create_incident_case
from backend.api.db_layer.action_items import (
    create_action_item,
    get_action_item_by_id,
    list_action_items_for_incident,
    update_action_item,
    mark_action_item_done,
)

# -----------------------------
# BASE INCIDENT
# -----------------------------

BASE_INCIDENT_DATA = {
    "ComplaintText": "Action item test",
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

def test_create_and_get_action_item_for_incident():
    incident_id = create_incident_case(BASE_INCIDENT_DATA)

    action_id = create_action_item(
        incident_case_id=incident_id,
        action_title="Follow up with department",
        action_description="Request explanation letter",
        due_date=date.today(),
        created_by_user_id=1,
    )

    action = get_action_item_by_id(action_id)
    assert action is not None
    assert action["IncidentRequestCaseID"] == incident_id
    assert action["IsDone"] == 0

test_create_and_get_action_item_for_incident()


def test_list_action_items_for_incident():
    incident_id = create_incident_case(BASE_INCIDENT_DATA)

    create_action_item(
        incident_case_id=incident_id,
        action_title="Second task",
        created_by_user_id=1,
    )

    items = list_action_items_for_incident(incident_id)
    assert isinstance(items, list)
    assert len(items) >= 1

test_list_action_items_for_incident()


def test_update_action_item():
    incident_id = create_incident_case(BASE_INCIDENT_DATA)

    action_id = create_action_item(
        incident_case_id=incident_id,
        action_title="Initial title",
        created_by_user_id=1,
    )

    update_action_item(
        action_id,
        {"ActionTitle": "Updated title"},
    )

    updated = get_action_item_by_id(action_id)
    assert updated["ActionTitle"] == "Updated title"

test_update_action_item()


def test_mark_action_item_done():
    incident_id = create_incident_case(BASE_INCIDENT_DATA)

    action_id = create_action_item(
        incident_case_id=incident_id,
        action_title="Complete task",
        created_by_user_id=1,
    )

    mark_action_item_done(action_id)

    done_item = get_action_item_by_id(action_id)
    assert done_item["IsDone"] == 1
    assert done_item["DateSubmitted"] is not None

test_mark_action_item_done()
