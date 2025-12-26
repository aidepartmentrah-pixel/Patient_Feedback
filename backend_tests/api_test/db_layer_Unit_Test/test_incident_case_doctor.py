from backend.api.db_layer.incident_case import create_incident_case
from backend.api.db_layer.incident_case_doctor import (
    add_doctor_to_case,
    list_case_doctors,
    remove_doctor_from_case,
    set_primary_doctor,
)


# -----------------------------
# BASE INCIDENT DATA
# -----------------------------

BASE_INCIDENT_DATA = {
    "ComplaintText": "Doctor assignment test",
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

def test_add_and_list_doctors():
    incident_id = create_incident_case(BASE_INCIDENT_DATA)

    add_doctor_to_case(
        incident_id=incident_id,
        doctor_id=1,
        assigned_by_user_id=1,
        is_primary=True,
    )

    doctors = list_case_doctors(incident_id)
    assert len(doctors) == 1
    assert doctors[0]["DoctorID"] == 1
    assert doctors[0]["IsPrimary"] == 1

test_add_and_list_doctors()


def test_set_primary_doctor():
    incident_id = create_incident_case(BASE_INCIDENT_DATA)

    d1 = add_doctor_to_case(
        incident_id,
        doctor_id=1,
        assigned_by_user_id=1,
        is_primary=True,
    )

    d2 = add_doctor_to_case(
        incident_id,
        doctor_id=2,
        assigned_by_user_id=1,
        is_primary=False,
    )

    set_primary_doctor(
        incident_id=incident_id,
        incident_case_doctor_id=d2,
    )

    doctors = list_case_doctors(incident_id)

    primary = [d for d in doctors if d["IsPrimary"] == 1]
    assert len(primary) == 1
    assert primary[0]["DoctorID"] == 2

test_set_primary_doctor()


def test_remove_doctor():
    incident_id = create_incident_case(BASE_INCIDENT_DATA)

    d_id = add_doctor_to_case(
        incident_id,
        doctor_id=1,
        assigned_by_user_id=1,
        is_primary=False,
    )

    remove_doctor_from_case(d_id)

    doctors = list_case_doctors(incident_id)
    assert len(doctors) == 0

test_remove_doctor()
