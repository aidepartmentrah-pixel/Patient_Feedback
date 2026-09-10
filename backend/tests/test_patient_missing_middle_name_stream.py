"""
Tests for the streaming counterpart of search_patients_missing_middle_name
(see test_patient_structured_search.py for the non-streaming function's own
contract tests -- this file leaves that one untouched).

Run: pytest backend/tests/test_patient_missing_middle_name_stream.py -v
"""

import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from api.services import patient_directory_service
from api.services import middle_name_sets_service


def test_stream_tries_all_candidates_unconditionally(monkeypatch):
    """
    Generator form of the "never stop early" guarantee: every candidate
    gets its own "trying" event and the final "done" event's tried count
    equals the full candidate list length, even when nothing matches.
    """
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

    events = list(patient_directory_service._iter_missing_middle_name_candidates("Ahmed", "Ali", limit=10))

    trying_events = [e for e in events if e["type"] == "trying"]
    assert [e["candidate"] for e in trying_events] == ["Mohamed", "Khaled", "Ali"]
    assert [e["index"] for e in trying_events] == [1, 2, 3]
    assert all(e["total"] == 3 for e in trying_events)

    done_events = [e for e in events if e["type"] == "done"]
    assert len(done_events) == 1
    assert done_events[0]["tried"] == 3
    assert done_events[0]["count"] == 0
    assert events[-1] is done_events[0]


def test_stream_emits_match_event_per_new_distinct_full_name(monkeypatch):
    """
    Two candidates matching genuinely different father's names must each
    produce their own "match" event -- the multi-person regression case,
    in generator form (see test_missing_middle_name_finds_two_distinct_
    people_with_different_father_names in test_patient_structured_search.py
    for the non-streaming version of this same guarantee).
    """
    monkeypatch.setattr(
        middle_name_sets_service, "get_active_set",
        lambda: {"id": "test", "display_name": "test", "names": ["Mohamed", "Khaled", "Hassan"]},
    )

    def fake_structured(first_name, middle_name, last_name, limit=20):
        if middle_name == "Mohamed":
            return {"success": True, "patients": [{"patient_admission_id": "1", "full_name": "Abbas Mohamed Zahreddine"}],
                    "count": 1, "external_status": "ok", "external_message": None}
        if middle_name == "Hassan":
            return {"success": True, "patients": [{"patient_admission_id": "2", "full_name": "Abbas Hassan Zahreddine"}],
                    "count": 1, "external_status": "ok", "external_message": None}
        return {"success": True, "patients": [], "count": 0, "external_status": "ok", "external_message": None}

    monkeypatch.setattr(patient_directory_service, "search_patients_structured", fake_structured)
    monkeypatch.setattr(patient_directory_service.time, "sleep", lambda s: None)

    events = list(patient_directory_service._iter_missing_middle_name_candidates("Abbas", "Zahreddine", limit=10))

    match_events = [e for e in events if e["type"] == "match"]
    assert len(match_events) == 2
    assert {e["patient"]["full_name"] for e in match_events} == {"Abbas Mohamed Zahreddine", "Abbas Hassan Zahreddine"}

    done = [e for e in events if e["type"] == "done"][0]
    assert done["tried"] == 3
    assert done["count"] == 2


def test_stream_collapses_same_full_name_across_records(monkeypatch):
    """
    A single candidate returning many records that share one full_name but
    differ in birth_date (the visits-not-persons case) must emit exactly
    one "match" event -- the streaming form of test_missing_middle_name_
    collapses_same_full_name_across_records in test_patient_structured_
    search.py.
    """
    monkeypatch.setattr(
        middle_name_sets_service, "get_active_set",
        lambda: {"id": "test", "display_name": "test", "names": ["Mohamed"]},
    )

    def fake_structured(first_name, middle_name, last_name, limit=20):
        return {
            "success": True,
            "patients": [
                {"patient_admission_id": f"ext__{i}", "full_name": "Abbas Mohamed Zahreddine", "birth_date": f"19{60 + i}-01-01"}
                for i in range(20)
            ],
            "count": 20, "external_status": "ok", "external_message": None,
        }

    monkeypatch.setattr(patient_directory_service, "search_patients_structured", fake_structured)
    monkeypatch.setattr(patient_directory_service.time, "sleep", lambda s: None)

    events = list(patient_directory_service._iter_missing_middle_name_candidates("Abbas", "Zahreddine", limit=20))

    match_events = [e for e in events if e["type"] == "match"]
    assert len(match_events) == 1

    candidate_done = [e for e in events if e["type"] == "candidate_done"][0]
    assert candidate_done["new_match_count"] == 1


def test_stream_final_done_event_matches_legacy_function_return(monkeypatch):
    """
    Draining the generator to its "done" event must produce the exact same
    count/tried/external_status/external_message as the legacy synchronous
    search_patients_missing_middle_name for the same inputs -- proof the
    refactor (function now just drains this generator) didn't change the
    contract.
    """
    monkeypatch.setattr(
        middle_name_sets_service, "get_active_set",
        lambda: {"id": "test", "display_name": "test", "names": ["Mohamed", "Khaled"]},
    )

    def fake_structured(first_name, middle_name, last_name, limit=20):
        if middle_name == "Mohamed":
            return {"success": True, "patients": [{"patient_admission_id": "1", "full_name": "Ahmed Mohamed Ali"}],
                    "count": 1, "external_status": "ok", "external_message": None}
        return {"success": True, "patients": [], "count": 0, "external_status": "ok", "external_message": None}

    monkeypatch.setattr(patient_directory_service, "search_patients_structured", fake_structured)
    monkeypatch.setattr(patient_directory_service.time, "sleep", lambda s: None)

    legacy = patient_directory_service.search_patients_missing_middle_name("Ahmed", "Ali", limit=10)

    stream_done = None
    for event in patient_directory_service._iter_missing_middle_name_candidates("Ahmed", "Ali", limit=10):
        if event["type"] == "done":
            stream_done = event

    assert stream_done["count"] == legacy["count"]
    assert stream_done["tried"] == legacy["tried"]
    assert stream_done["external_status"] == legacy["external_status"]
    assert stream_done["external_message"] == legacy["external_message"]


def test_legacy_function_still_returns_same_shape(monkeypatch):
    """Belt-and-suspenders: the public function's return keys are unchanged post-refactor."""
    monkeypatch.setattr(
        middle_name_sets_service, "get_active_set",
        lambda: {"id": "test", "display_name": "test", "names": ["Mohamed"]},
    )
    monkeypatch.setattr(
        patient_directory_service, "search_patients_structured",
        lambda first_name, middle_name, last_name, limit=20: {
            "success": True, "patients": [], "count": 0, "external_status": "ok", "external_message": None,
        },
    )
    monkeypatch.setattr(patient_directory_service.time, "sleep", lambda s: None)

    result = patient_directory_service.search_patients_missing_middle_name("Ahmed", "Ali", limit=10)

    assert set(result.keys()) == {"success", "patients", "count", "tried", "external_status", "external_message"}


def test_stream_endpoint_returns_sse_media_type_and_all_events(monkeypatch):
    """
    HTTP-level smoke test for the new /stream route: authenticate via a
    dependency override (this codebase has no existing session-cookie test
    fixture to reuse, and overriding get_current_user is the standard
    FastAPI approach), then confirm the response is a real event-stream
    containing one "done" event whose tried count matches the mocked
    candidate list.
    """
    from fastapi.testclient import TestClient
    from main import app
    from api.dependencies.user_context import get_current_user
    from api.schemas.auth_models import CurrentUser

    monkeypatch.setattr(
        middle_name_sets_service, "get_active_set",
        lambda: {"id": "test", "display_name": "test", "names": ["Mohamed", "Khaled"]},
    )
    monkeypatch.setattr(
        patient_directory_service, "search_patients_structured",
        lambda first_name, middle_name, last_name, limit=20: {
            "success": True, "patients": [], "count": 0, "external_status": "ok", "external_message": None,
        },
    )
    monkeypatch.setattr(patient_directory_service.time, "sleep", lambda s: None)

    fake_user = CurrentUser(user_id=1, username="test_user", is_active=True, scopes=[])
    app.dependency_overrides[get_current_user] = lambda: fake_user
    try:
        client = TestClient(app)
        response = client.get(
            "/api/records/search/patients/missing-middle-name/stream",
            params={"first_name": "Ahmed", "last_name": "Ali"},
        )
    finally:
        app.dependency_overrides.pop(get_current_user, None)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: done" in response.text
    assert '"tried": 2' in response.text
