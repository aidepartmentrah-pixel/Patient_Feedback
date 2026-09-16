"""
Regression tests for the Notice-vs-Complaint required-field exemption in
the bulk Excel import validator (_validate_group in import_service.py).

Root cause fixed: Severity/Stage/Harm Level/Feedback Risk Type were
mandatory for every Record Type, so a Notice row missing them could never
leave "Need Attention" during import review -- even though the manual
Insert/Edit form and case_service.py have always exempted Notice from
these same fields. Building was deliberately left mandatory for every
Record Type (not exempted).

Talks directly to _validate_group() with a hand-built `maps` dict, so it
runs with no DB/network dependency (the DB-backed patient-ambiguity check
inside _validate_group is monkeypatched out below).

Run from the repository root (not from inside backend/ -- import_service.py
pulls in modules that do `from backend.api...` absolute imports, which only
resolve when the repo root itself is on sys.path):
    python -m pytest backend/tests/test_import_notice_validation.py -v
"""

import os
import sys
from datetime import date

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from api.services import import_service
from api.services.import_service import _validate_group, ROW_NUMBER_KEY


# ---- fake lookup maps, keyed the same way _load_lookups() would build them
# (category -> {lowercased name: id}) ----
MAPS = {
    "org_units": {"cardiac 1": 10},
    "sources": {"box": 5},
    "classifications": {"test classification": 7},
    "classification_chains": {7: {"domain_id": 1, "category_id": 2, "subcategory_id": 3}},
    "severities": {"high": 1},
    "stages": {"initial": 1},
    "harm_levels": {"none": 1},
    "risk_types": {"ordinary": 1},
    "buildings": {"main": 1},
    "doctors": {},
    "workers": {},
}

# A row that's fully valid for every field EXCEPT whatever the test
# overrides -- lets each test isolate exactly the field(s) under test.
BASE_ROW = {
    "Incident Date": date(2026, 1, 1),
    "Received Date": date(2026, 1, 2),
    "Incident Number (Old System) (الرقم)": "TEST-001",
    "Patient Name": "Test Patient",
    "Issuing Dept (قسم الصادر)": "Cardiac 1",
    "Target Dept": "Cardiac 1",
    "Source (المصدر)": "Box",
    "Record Type": "Complaint",
    "Classification (Arabic)": "Test Classification",
    "Classification (English)": None,
    "Complaint Text": "Some complaint text",
    "Immediate Action": None,
    "Taken Action (الإجراءات المتخذة)": None,
    "Severity": "High",
    "Stage": "Initial",
    "Harm Level": "None",
    "Feedback Risk Type": "Ordinary",
    "Building": "Main",
    "Is Inpatient": "No",
    "Doctor 1": None, "Doctor 2": None, "Doctor 3": None,
    "Worker 1 (Full Name)": None, "Worker 2 (Full Name)": None, "Worker 3 (Full Name)": None,
    ROW_NUMBER_KEY: 1,
}


def make_row(**overrides):
    row = dict(BASE_ROW)
    row.update(overrides)
    return row


def error_fields(result):
    return {e["field"] for r in result["rows"] for e in r["errors"]}


def _stub_out_patient_directory(monkeypatch):
    """_validate_group calls _count_patient_matches() (a real Hospital
    Directory/DB lookup) once a group is otherwise error-free. Stub it so
    these tests stay pure/offline -- always "new patient", never ambiguous."""
    monkeypatch.setattr(import_service, "_count_patient_matches", lambda name: 0)


def test_notice_row_missing_severity_stage_harm_risk_is_valid(monkeypatch):
    _stub_out_patient_directory(monkeypatch)
    row = make_row(**{
        "Record Type": "Notice",
        "Severity": None,
        "Stage": None,
        "Harm Level": None,
        "Feedback Risk Type": None,
    })

    result = _validate_group("TEST-001", [row], MAPS)

    fields = error_fields(result)
    assert "Severity" not in fields, result["rows"][0]["errors"]
    assert "Stage" not in fields, result["rows"][0]["errors"]
    assert "Harm Level" not in fields, result["rows"][0]["errors"]
    assert "Feedback Risk Type" not in fields, result["rows"][0]["errors"]
    assert result["valid"] is True, result["rows"][0]["errors"]


def test_notice_row_with_invalid_severity_value_is_still_rejected(monkeypatch):
    _stub_out_patient_directory(monkeypatch)
    row = make_row(**{
        "Record Type": "Notice",
        "Severity": "Not A Real Severity Value",
    })

    result = _validate_group("TEST-001", [row], MAPS)

    assert "Severity" in error_fields(result)
    assert result["valid"] is False


def test_notice_row_missing_building_is_still_rejected(monkeypatch):
    _stub_out_patient_directory(monkeypatch)
    row = make_row(**{
        "Record Type": "Notice",
        "Severity": None,
        "Stage": None,
        "Harm Level": None,
        "Feedback Risk Type": None,
        "Building": None,
    })

    result = _validate_group("TEST-001", [row], MAPS)

    assert "Building" in error_fields(result)
    assert result["valid"] is False


def test_complaint_row_missing_severity_stage_harm_risk_is_rejected(monkeypatch):
    _stub_out_patient_directory(monkeypatch)
    row = make_row(**{
        "Record Type": "Complaint",
        "Severity": None,
        "Stage": None,
        "Harm Level": None,
        "Feedback Risk Type": None,
    })

    result = _validate_group("TEST-001", [row], MAPS)

    fields = error_fields(result)
    assert {"Severity", "Stage", "Harm Level", "Feedback Risk Type"} <= fields
    assert result["valid"] is False


def test_complaint_row_fully_filled_is_valid(monkeypatch):
    _stub_out_patient_directory(monkeypatch)
    row = make_row()  # BASE_ROW is already a fully-valid Complaint row

    result = _validate_group("TEST-001", [row], MAPS)

    assert result["rows"][0]["errors"] == []
    assert result["valid"] is True
