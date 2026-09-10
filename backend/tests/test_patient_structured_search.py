"""
Regression tests for the 3-field (first/middle/last) patient search added
on top of the free-text search in patients_db.search_patients() and the
new patient_directory_service.search_patients_structured().

Run: pytest backend/tests/test_patient_structured_search.py -v
"""

import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from api.db_layer import patients_db
from api.services import patient_directory_service
from api.services import middle_name_sets_service


class _FakeCursor:
    def __init__(self):
        self.executed_sql = None
        self.executed_params = None
        self.description = [(c,) for c in (
            "patient_id", "mrn", "full_name", "first_name", "middle_name",
            "last_name", "date_of_birth", "age", "gender", "phone", "source",
        )]

    def execute(self, sql, params):
        self.executed_sql = sql
        self.executed_params = params

    def fetchall(self):
        return []

    def close(self):
        pass


class _FakeConnection:
    def __init__(self):
        self.cursor_obj = _FakeCursor()

    def cursor(self):
        return self.cursor_obj

    def close(self):
        pass


def test_search_patients_structured_where_builds_per_field_conditions(monkeypatch):
    created = {}

    def fake_get_connection():
        conn = _FakeConnection()
        created["conn"] = conn
        return conn

    monkeypatch.setattr(patients_db, "get_connection", fake_get_connection)

    patients_db.search_patients(first_name="Ahmed", middle_name="Mohamed", last_name="Ali", limit=10)

    cursor = created["conn"].cursor_obj
    assert "FirstName LIKE ?" in cursor.executed_sql
    assert "MiddleName LIKE ?" in cursor.executed_sql
    assert "LastName LIKE ?" in cursor.executed_sql
    assert "%Ahmed%" in cursor.executed_params
    assert "%Mohamed%" in cursor.executed_params
    assert "%Ali%" in cursor.executed_params


def test_search_patients_dedupes_repeat_admissions_before_the_limit(monkeypatch):
    """
    APP_RESERVE_PATIENT is admission-centric -- the same real patient with
    N past admissions produces N identical-looking rows. The dedup must
    happen inside the SQL (before TOP (?)), not after in Python, or real
    distinct patients would already have been dropped to make room for the
    duplicates.
    """
    created = {}

    def fake_get_connection():
        conn = _FakeConnection()
        created["conn"] = conn
        return conn

    monkeypatch.setattr(patients_db, "get_connection", fake_get_connection)

    patients_db.search_patients(last_name="Zahreddine", limit=10)

    cursor = created["conn"].cursor_obj
    assert "ROW_NUMBER()" in cursor.executed_sql
    assert "PARTITION BY COALESCE(NULLIF(LTRIM(RTRIM(DocumentNumber)), ''), FullName)" in cursor.executed_sql
    assert "WHERE rn = 1" in cursor.executed_sql
    # The limit is still the last param -- TOP (?) is in the outer query,
    # after the CTE's own WHERE-clause params.
    assert cursor.executed_params[-1] == 10


def test_search_patients_structured_omits_middle_condition_when_blank(monkeypatch):
    created = {}

    def fake_get_connection():
        conn = _FakeConnection()
        created["conn"] = conn
        return conn

    monkeypatch.setattr(patients_db, "get_connection", fake_get_connection)

    patients_db.search_patients(first_name="Ahmed", middle_name=None, last_name="Ali", limit=10)

    cursor = created["conn"].cursor_obj
    assert "MiddleName LIKE ?" not in cursor.executed_sql
    assert "FirstName LIKE ?" in cursor.executed_sql
    assert "LastName LIKE ?" in cursor.executed_sql


def test_search_patients_structured_free_text_query_unchanged(monkeypatch):
    """The pre-existing `query` free-text mode must be untouched by the new params."""
    created = {}

    def fake_get_connection():
        conn = _FakeConnection()
        created["conn"] = conn
        return conn

    monkeypatch.setattr(patients_db, "get_connection", fake_get_connection)

    patients_db.search_patients(query="ahmed ali", limit=10)

    cursor = created["conn"].cursor_obj
    assert "(FullName LIKE ? OR FirstName LIKE ? OR LastName LIKE ?)" in cursor.executed_sql
    # No standalone structured conditions were added -- just the three
    # OR-matched params from the free-text condition, followed by the
    # limit (TOP (?) is now in the outer query, after the CTE's own ?'s).
    assert cursor.executed_params == ["%ahmed ali%", "%ahmed ali%", "%ahmed ali%", 10]


def test_service_search_patients_structured_maps_middle_name_to_father_name(monkeypatch):
    captured = {}

    def fake_reserve_search(**kwargs):
        captured["reserve_kwargs"] = kwargs
        return []

    def fake_external_search(**kwargs):
        captured["external_kwargs"] = kwargs
        return {"status": "ok", "message": None, "items": [], "total": 0}

    monkeypatch.setattr(patient_directory_service.patients_db, "search_patients", fake_reserve_search)
    monkeypatch.setattr(patient_directory_service.directory_client, "search_patients", fake_external_search)

    result = patient_directory_service.search_patients_structured("Ahmed", "Mohamed", "Ali", limit=10)

    assert captured["reserve_kwargs"]["first_name"] == "Ahmed"
    assert captured["reserve_kwargs"]["middle_name"] == "Mohamed"
    assert captured["reserve_kwargs"]["last_name"] == "Ali"

    assert captured["external_kwargs"]["first_name"] == "Ahmed"
    assert captured["external_kwargs"]["father_name"] == "Mohamed"
    assert captured["external_kwargs"]["last_name"] == "Ali"

    assert result["success"] is True
    assert result["external_status"] == "ok"


def test_search_patients_structured_dedupes_external_repeat_admissions(monkeypatch):
    """
    20 external rows for the same real person (identical name + birth date,
    just different patient_ids -- e.g. re-entered on every admission) must
    collapse to one. A row with the same name but a DIFFERENT birth date
    (a genuinely different real person, per the vendor's own 90016/90017
    contract fixtures) must NOT be collapsed away.
    """
    monkeypatch.setattr(patient_directory_service.patients_db, "search_patients", lambda **kwargs: [])

    same_person_repeats = [
        {"patient_id": str(i), "full_name": "Abbas Mohamed Zahreddine", "first_name": "Abbas",
         "last_name": "Zahreddine", "birth_date": "1990-01-01", "sex": "Male"}
        for i in range(20)
    ]
    different_real_person = {
        "patient_id": "999", "full_name": "Abbas Mohamed Zahreddine", "first_name": "Abbas",
        "last_name": "Zahreddine", "birth_date": "1985-05-05", "sex": "Male",
    }

    monkeypatch.setattr(
        patient_directory_service.directory_client, "search_patients",
        lambda **kwargs: {
            "status": "ok", "message": None,
            "items": same_person_repeats + [different_real_person],
            "total": 21,
        },
    )

    result = patient_directory_service.search_patients_structured("Abbas", "Mohamed", "Zahreddine", limit=50)

    assert result["count"] == 2  # 20 repeats collapsed to 1, plus the genuinely different person


def test_service_search_patients_structured_reserve_failure_reported(monkeypatch):
    def fake_reserve_search(**kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(patient_directory_service.patients_db, "search_patients", fake_reserve_search)

    result = patient_directory_service.search_patients_structured("Ahmed", "Mohamed", "Ali", limit=10)

    assert result["success"] is False
    assert "db down" in result["error"]


def test_missing_middle_name_uses_structured_search_not_free_text(monkeypatch):
    """
    Regression test for the bug found via live testing: the candidate loop
    used to join each guess into one free-text string and call
    search_patients_insert_flow, which the v1.1-conformant Hospital
    Directory API mock rejects with 422 on every single try (father_name
    is mandatory once q is no longer accepted). It must call
    search_patients_structured directly instead.
    """
    monkeypatch.setattr(
        middle_name_sets_service, "get_active_set",
        lambda: {"id": "test", "display_name": "test", "names": ["Mohamed", "Khaled"]},
    )

    captured_calls = []

    def fake_structured(first_name, middle_name, last_name, limit=20):
        captured_calls.append((first_name, middle_name, last_name))
        if middle_name == "Mohamed":
            return {
                "success": True,
                "patients": [{"patient_admission_id": "1", "full_name": "Ahmed Mohamed Ali"}],
                "count": 1, "external_status": "ok", "external_message": None,
            }
        return {"success": True, "patients": [], "count": 0, "external_status": "ok", "external_message": None}

    monkeypatch.setattr(patient_directory_service, "search_patients_structured", fake_structured)

    def fail_if_called(*a, **k):
        raise AssertionError("search_patients_insert_flow must not be used by the candidate loop anymore")

    monkeypatch.setattr(patient_directory_service, "search_patients_insert_flow", fail_if_called)

    result = patient_directory_service.search_patients_missing_middle_name("Ahmed", "Ali", limit=10)

    assert captured_calls[0] == ("Ahmed", "Mohamed", "Ali")
    assert result["success"] is True
    assert result["count"] == 1
    # Tries every candidate, even after finding a match -- a first+last
    # pair can legitimately match more than one real person who differ
    # only in father's name (see test_missing_middle_name_finds_two_
    # distinct_people_with_different_father_names below).
    assert result["tried"] == 2


def test_missing_middle_name_finds_two_distinct_people_with_different_father_names(monkeypatch):
    """
    Regression test: stopping at the first matching candidate used to
    silently hide a second real person sharing the same first+last name
    but a different father's name. Both must now be found and combined.
    """
    monkeypatch.setattr(
        middle_name_sets_service, "get_active_set",
        lambda: {"id": "test", "display_name": "test", "names": ["Mohamed", "Khaled", "Hassan"]},
    )

    def fake_structured(first_name, middle_name, last_name, limit=20):
        if middle_name == "Mohamed":
            return {
                "success": True,
                "patients": [{"patient_admission_id": "1", "full_name": "Abbas Mohamed Zahreddine"}],
                "count": 1, "external_status": "ok", "external_message": None,
            }
        if middle_name == "Hassan":
            return {
                "success": True,
                "patients": [{"patient_admission_id": "2", "full_name": "Abbas Hassan Zahreddine"}],
                "count": 1, "external_status": "ok", "external_message": None,
            }
        return {"success": True, "patients": [], "count": 0, "external_status": "ok", "external_message": None}

    monkeypatch.setattr(patient_directory_service, "search_patients_structured", fake_structured)

    result = patient_directory_service.search_patients_missing_middle_name("Abbas", "Zahreddine", limit=10)

    assert result["tried"] == 3
    assert result["count"] == 2
    ids = {p["patient_admission_id"] for p in result["patients"]}
    assert ids == {"1", "2"}


def test_missing_middle_name_tries_all_candidates_when_none_match(monkeypatch):
    monkeypatch.setattr(
        middle_name_sets_service, "get_active_set",
        lambda: {"id": "test", "display_name": "test", "names": ["Mohamed", "Khaled", "Ali"]},
    )
    monkeypatch.setattr(
        patient_directory_service, "search_patients_structured",
        lambda first_name, middle_name, last_name, limit=20: {
            "success": True, "patients": [], "count": 0, "external_status": "ok", "external_message": None,
        },
    )
    monkeypatch.setattr(patient_directory_service.time, "sleep", lambda s: None)

    result = patient_directory_service.search_patients_missing_middle_name("Ahmed", "Ali", limit=10)

    assert result["count"] == 0
    assert result["tried"] == 3


# --- Exploratory coverage: how the API's own substring matching behaves on
# multi-word single-field name components (e.g. "شمس الدين" as one
# last_name value) -- documents ground truth for a future import name-
# splitting design, does not test any HCAT code path. Requires the mock
# Hospital Directory API to be running locally with the Muslim Arabic
# fixtures seeded (seed/seed_muslim_arabic_fixtures.py in the
# RESTful-API-Integration repo) -- skipped automatically if unreachable.
import requests

_MOCK_BASE = "http://localhost:6000/api/directory/v1"
_MOCK_HEADERS = {"X-API-Key": "change_me"}


def _mock_reachable():
    try:
        return requests.get(f"{_MOCK_BASE}/health", timeout=1).status_code == 200
    except Exception:
        return False


import pytest as _pytest

requires_mock = _pytest.mark.skipif(not _mock_reachable(), reason="mock Hospital Directory API not running on localhost:6000")


@requires_mock
def test_mock_matches_two_word_last_name_as_single_value():
    """fixture 90023: كريم / عباس / شمس الدين -- the two-word last name is
    stored and matched as one value, never split on the internal space."""
    r = requests.get(f"{_MOCK_BASE}/patients", headers=_MOCK_HEADERS, params={
        "first_name": "كريم", "father_name": "عباس", "last_name": "شمس الدين",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    assert body["items"][0]["patient_id"] == "90023"


@requires_mock
def test_mock_compound_first_name_with_no_father_name_only_findable_by_id():
    """fixture 90019: أبو بكر (two-word first/kunya name) has no father_name
    on file -- confirms the name-trio search genuinely cannot find it (by
    design, not a bug), only GET /patients/{id} can."""
    r = requests.get(f"{_MOCK_BASE}/patients/90019", headers=_MOCK_HEADERS)
    assert r.status_code == 200
    assert r.json()["first_name"] == "أبو بكر"

    # Guessing a father_name (any of the 32 candidates) must never match --
    # there genuinely isn't one on this record.
    r2 = requests.get(f"{_MOCK_BASE}/patients", headers=_MOCK_HEADERS, params={
        "first_name": "أبو بكر", "father_name": "محمد", "last_name": "خليل",
    })
    assert r2.status_code == 200
    assert r2.json()["total"] == 0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
