"""
Regression tests for the Hospital Directory API v1.1 contract-conformance
pass (see hospital_directory_client.py's search_patients/search_workers/_get).

No mocking library is used elsewhere in this repo's tests -- pytest's
built-in monkeypatch fixture is enough here, and pytest is already a
declared dependency (requirements.txt).

Run: pytest backend/tests/test_hospital_directory_client.py -v
"""

import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from core import hospital_directory_client as directory_client


class _FakeResponse:
    def __init__(self, status_code, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        if self._json_data is None:
            raise ValueError("response has no JSON body")
        return self._json_data


def _patch_get(monkeypatch, captured):
    """Replace directory_client._get with a stub that records (path, params)
    and returns a canned 'ok' envelope, so search_patients/search_workers
    can be tested purely on the params they build."""
    def fake_get(path, params):
        captured["path"] = path
        captured["params"] = params
        return {"status": "ok", "message": None, "data": {"items": [], "total": 0}}
    monkeypatch.setattr(directory_client, "_get", fake_get)


def test_search_patients_patient_id_mode(monkeypatch):
    captured = {}
    _patch_get(monkeypatch, captured)
    directory_client.search_patients(patient_id="123")
    params = captured["params"]
    assert params.get("patient_id") == "123"
    assert "q" not in params
    assert "first_name" not in params and "father_name" not in params and "last_name" not in params


def test_search_patients_structured_name_mode(monkeypatch):
    captured = {}
    _patch_get(monkeypatch, captured)
    directory_client.search_patients(first_name="Ahmed", father_name="Mohamed", last_name="Ali")
    params = captured["params"]
    assert params.get("first_name") == "Ahmed"
    assert params.get("father_name") == "Mohamed"
    assert params.get("last_name") == "Ali"
    assert "q" not in params and "patient_id" not in params


def test_search_patients_free_text_q_still_supported(monkeypatch):
    """q is deliberately kept (not part of v1.1's documented shape) since it
    matches the only proven behavior of the real production server today —
    see the "Known consequence" note in the client's search_patients docstring."""
    captured = {}
    _patch_get(monkeypatch, captured)
    directory_client.search_patients(q="ahmed ali")
    params = captured["params"]
    assert params.get("q") == "ahmed ali"
    assert "patient_id" not in params
    assert "first_name" not in params


def test_search_workers_never_sends_q_or_active_only(monkeypatch):
    captured = {}
    _patch_get(monkeypatch, captured)
    directory_client.search_workers()
    params = captured["params"]
    assert "q" not in params
    assert "active_only" not in params
    assert params == {"limit": 10, "offset": 0}


def test_search_workers_default_limit_is_ten(monkeypatch):
    captured = {}
    _patch_get(monkeypatch, captured)
    directory_client.search_workers(offset=20)
    assert captured["params"]["limit"] == 10
    assert captured["params"]["offset"] == 20


def _patch_client_config(monkeypatch):
    monkeypatch.setattr(
        directory_client, "_get_client_config",
        lambda: ({"base_url": "https://fake.example", "api_key": "k", "timeout_seconds": 5, "verify_tls": True}, None),
    )


def test_get_extracts_message_on_401(monkeypatch):
    _patch_client_config(monkeypatch)
    monkeypatch.setattr(
        directory_client.requests, "get",
        lambda *a, **k: _FakeResponse(401, {"error": "UNAUTHORIZED", "message": "bad key"}),
    )
    result = directory_client._get("/patients", {})
    assert result["status"] == "unauthorized"
    assert result["message"] == "bad key"


def test_get_extracts_message_on_404(monkeypatch):
    _patch_client_config(monkeypatch)
    monkeypatch.setattr(
        directory_client.requests, "get",
        lambda *a, **k: _FakeResponse(404, {"error": "PATIENT_NOT_FOUND", "message": "no such patient"}),
    )
    result = directory_client._get("/patients/999", {})
    assert result["status"] == "not_found"
    assert result["message"] == "no such patient"


def test_get_extracts_message_on_500(monkeypatch):
    _patch_client_config(monkeypatch)
    monkeypatch.setattr(
        directory_client.requests, "get",
        lambda *a, **k: _FakeResponse(500, {"error": "SERVER_ERROR", "message": "db down"}),
    )
    result = directory_client._get("/workers", {})
    assert result["status"] == "http_error"
    assert result["message"] == "db down"


def test_get_extracts_message_on_503(monkeypatch):
    _patch_client_config(monkeypatch)
    monkeypatch.setattr(
        directory_client.requests, "get",
        lambda *a, **k: _FakeResponse(503, {"error": "SERVICE_UNAVAILABLE", "message": "HR system unreachable"}),
    )
    result = directory_client._get("/workers", {})
    assert result["status"] == "http_error"
    assert result["message"] == "HR system unreachable"


def test_get_falls_back_to_generic_message_when_body_not_json(monkeypatch):
    _patch_client_config(monkeypatch)
    monkeypatch.setattr(
        directory_client.requests, "get",
        lambda *a, **k: _FakeResponse(500, json_data=None, text="<html>gateway error</html>"),
    )
    result = directory_client._get("/workers", {})
    assert result["status"] == "http_error"
    assert "500" in result["message"]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
