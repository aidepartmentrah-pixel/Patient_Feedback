from backend.api.db_layer.incident_case import create_incident_case
from backend.api.db_layer.incident_case_target_department import (
    add_target_department,
    list_target_departments,
    remove_target_department,
    set_primary_department,
)

# -----------------------------
# BASE INCIDENT DATA
# -----------------------------

BASE_INCIDENT_DATA = {
    "ComplaintText": "Target department assignment test",
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

    "SeverityID": None,
    "BuildingID": None,
}


# -----------------------------
# TESTS
# -----------------------------

def test_add_and_list_target_departments():
    incident_id = create_incident_case(BASE_INCIDENT_DATA)

    add_target_department(
        incident_id=incident_id,
        department_id=1,
        assigned_by_user_id=1,
        is_primary=True,
    )

    targets = list_target_departments(incident_id)
    assert len(targets) == 1
    assert targets[0]["DepartmentID"] == 1
    assert targets[0]["IsPrimary"] == 1

test_add_and_list_target_departments()


def test_set_primary_department():
    incident_id = create_incident_case(BASE_INCIDENT_DATA)

    t1 = add_target_department(
        incident_id,
        department_id=1,
        assigned_by_user_id=1,
        is_primary=True,
    )

    t2 = add_target_department(
        incident_id,
        department_id=2,
        assigned_by_user_id=1,
        is_primary=False,
    )

    set_primary_department(
        incident_id=incident_id,
        target_id=t2,
    )

    targets = list_target_departments(incident_id)

    primary = [t for t in targets if t["IsPrimary"] == 1]
    assert len(primary) == 1
    assert primary[0]["DepartmentID"] == 2

test_set_primary_department()


def test_remove_target_department():
    incident_id = create_incident_case(BASE_INCIDENT_DATA)

    t_id = add_target_department(
        incident_id,
        department_id=1,
        assigned_by_user_id=1,
        is_primary=False,
    )

    remove_target_department(t_id)

    targets = list_target_departments(incident_id)
    assert len(targets) == 0

test_remove_target_department()
