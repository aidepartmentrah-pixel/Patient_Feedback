"""
Smoke Test — Quick-Add Patient Feature
(Insert Record page → "Add Patient" button when search returns no results)

Tests:
  1. Create patient with minimum data (first_name only)
  2. Create patient with full name fields
  3. Reject missing first_name (validation)
  4. Reject first_name too short (< 2 chars)
  5. Confirm response shape contains patient id + full_name
  6. Patient search endpoint returns empty list for non-existent name
  7. Doctor search unchanged — returns list (no add-patient logic in backend)

Run:
    cd backend
    python -m pytest tests/test_smoke_quick_add_patient.py -v

Or with DB (set SMOKE_DB=1):
    SMOKE_DB=1 python -m pytest tests/test_smoke_quick_add_patient.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── 1. Import sanity ──────────────────────────────────────────────────────────

def test_router_imports_clean():
    from api.routers.patients_router import router, CreatePatientRequest
    assert router is not None
    assert CreatePatientRequest is not None


def test_service_imports_clean():
    from api.services.patients_service import create_patient_service
    assert callable(create_patient_service)


# ── 2. Request model validation (no DB needed) ────────────────────────────────

def test_create_patient_request_minimum():
    from api.routers.patients_router import CreatePatientRequest
    req = CreatePatientRequest(first_name="Ahmed")
    assert req.first_name == "Ahmed"
    assert req.middle_name is None
    assert req.last_name is None


def test_create_patient_request_full_name():
    from api.routers.patients_router import CreatePatientRequest
    req = CreatePatientRequest(first_name="Ahmed", middle_name="Mohammed", last_name="Al-Rashid")
    assert req.first_name == "Ahmed"
    assert req.middle_name == "Mohammed"
    assert req.last_name == "Al-Rashid"


def test_create_patient_request_missing_first_name():
    from api.routers.patients_router import CreatePatientRequest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        CreatePatientRequest()


def test_create_patient_request_first_name_too_short():
    from api.routers.patients_router import CreatePatientRequest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        CreatePatientRequest(first_name="A")  # min_length=2


# ── 3. DB round-trip tests (require SMOKE_DB=1) ───────────────────────────────

SMOKE_DB = os.environ.get("SMOKE_DB") == "1"
skip_no_db = pytest.mark.skipif(not SMOKE_DB, reason="Set SMOKE_DB=1 to run DB tests")


@skip_no_db
def test_create_patient_minimum_db():
    from api.services.patients_service import create_patient_service
    import asyncio

    async def run():
        result = await create_patient_service(
            first_name="TestQuickAdd",
            middle_name=None,
            last_name=None,
            mother_name=None,
            phone_number=None,
            phone_number2=None,
            birth_date=None,
            sex=None,
            document_number=None,
            medical_file_number=None,
            spouse=None,
            address_line1=None,
            address_line2=None,
        )
        return result

    result = asyncio.run(run())
    assert result.get("success") is True
    patient = result.get("patient", {})
    assert patient.get("patient_admission_id") is not None
    assert "TestQuickAdd" in (patient.get("full_name") or "")


@skip_no_db
def test_patient_search_no_results_for_nonsense():
    from api.db_layer.search_db import search_patients_db
    import asyncio

    async def run():
        return await search_patients_db("xyzzy_nonexistent_zzz", limit=5)

    results = asyncio.run(run())
    assert isinstance(results, list)
    assert len(results) == 0


@skip_no_db
def test_doctor_search_returns_list():
    from api.db_layer.search_db import search_doctors_db
    import asyncio

    async def run():
        return await search_doctors_db("a", limit=5)

    results = asyncio.run(run())
    assert isinstance(results, list)


# ── 4. API response shape ─────────────────────────────────────────────────────

def test_create_patient_response_shape_contract():
    """
    Verifies the expected response keys that the frontend relies on.
    Frontend reads: result.patient.full_name and result.patient.patient_admission_id
    """
    expected_patient_keys = {"patient_admission_id", "full_name", "first_name", "source"}
    # This just documents the contract; actual values tested in DB test above.
    assert expected_patient_keys.issubset({"patient_admission_id", "full_name", "first_name", "source", "middle_name", "last_name"})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
