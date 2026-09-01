"""
Patient Directory Adapter (Session C1)

Merges HCAT's own reserve patients (APP_RESERVE_PATIENT, via
api.db_layer.patients_db) with external patients from the Hospital
Directory API (via core.hospital_directory_client) into ONE normalized
model, so the rest of the app (search_service, patients_service, drawer
notes) doesn't need to know which source a given patient came from.

Identity model
--------------
Every patient-like record HCAT hands out as "patient_id" is one of:
  - a plain reserve PatientAdmissionID (int, or its string form) — resolves
    against APP_RESERVE_PATIENT.
  - an opaque "ext__{patient_id}" string (see
    hospital_directory_client.encode_external_patient_id) — resolves
    against the Hospital Directory API's GET /patients/{id}.

NOTE: the Hospital Directory API used to model identity as patient+visit
(a PatientVisit schema requiring a visit_id). That concept was dropped from
the API — patients are identified by patient_id alone now (see the current
vendor OpenAPI spec's Patient schema). The shape functions below and the
encode/decode helpers in hospital_directory_client reflect that current
contract.

Both merge functions below always return reserve results even if the
external call fails — external unavailability must never make local data
disappear (requirement: HCAT keeps running if the external API is down).
The failure is instead surfaced via an explicit external_status/
external_message pair so callers can render "external search unavailable"
distinctly from "zero results".
"""

import time
from typing import Any, Dict, List, Optional

from api.db_layer import patients_db
from core import hospital_directory_client as directory_client


class ExternalPatientUnavailableError(Exception):
    """
    Raised when a patient_id resolves to an EXTERNAL identity but the
    Hospital Directory API could not be reached/authenticated/etc. Distinct
    from "not found" (which means the API responded and said 404) — see
    hospital_directory_client's status vocabulary.
    """

    def __init__(self, status: str, message: str):
        self.status = status
        self.message = message
        super().__init__(f"Hospital Directory API unavailable ({status}): {message}")


_SEX_TO_GENDER = {"M": "Male", "F": "Female"}


def _patient_to_history_shape(patient: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize one external Patient into the shape patients_db.search_patients rows use."""
    encoded_id = directory_client.encode_external_patient_id(patient["patient_id"])
    sex = patient.get("sex")
    return {
        "patient_id": encoded_id,
        "mrn": None,  # not part of the Hospital Directory API's Patient schema
        "full_name": patient.get("full_name"),
        "first_name": patient.get("first_name"),
        "last_name": patient.get("last_name"),
        "date_of_birth": patient.get("birth_date"),
        "age": patient.get("age"),
        "gender": _SEX_TO_GENDER.get(sex, sex),
        "phone": None,  # not part of the Hospital Directory API's Patient schema
        "source": "external",
    }


def _patient_to_insert_shape(patient: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize one external Patient into the shape search_service.search_patients rows use."""
    encoded_id = directory_client.encode_external_patient_id(patient["patient_id"])
    return {
        "patient_admission_id": encoded_id,
        "full_name": patient.get("full_name"),
        "first_name": patient.get("first_name"),
        "last_name": patient.get("last_name"),
        "document_number": None,  # not part of the Hospital Directory API's Patient schema
        "phone_number": None,  # not part of the Hospital Directory API's Patient schema
        "birth_date": patient.get("birth_date"),
        "sex": patient.get("sex"),
        "medical_file_number": None,  # not part of the Hospital Directory API's Patient schema
        "admission_date": None,  # not part of the Hospital Directory API's Patient schema
        "source": "external",
    }


def _patient_to_profile_shape(patient: Dict[str, Any], encoded_id: str) -> Dict[str, Any]:
    sex = patient.get("sex")
    return {
        "PatientID": encoded_id,
        "MRN": None,  # not part of the Hospital Directory API's Patient schema
        "PatientName": patient.get("full_name"),
        "PatientNameEnglish": patient.get("first_name"),
        "DateOfBirth": patient.get("birth_date"),
        "Age": patient.get("age"),
        "Gender": _SEX_TO_GENDER.get(sex, sex),
        "Nationality": "",
        "Phone": None,  # not part of the Hospital Directory API's Patient schema
        "Email": "",
        "Address": "",
        "EmergencyContact": "",
        "EmergencyPhone": "",
        "RegistrationDate": None,  # not part of the Hospital Directory API's Patient schema
        "source": "external",
    }


def search_patients_history(
    query: Optional[str] = None,
    mrn: Optional[str] = None,
    phone: Optional[str] = None,
    date_of_birth: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Merged search for the Patient History page (backs patients_db-style
    consumers: /api/patients/search, /api/v2/patients/search).

    Returns:
        {
          "patients": [...],       # reserve items + external items (if any)
          "external_status": "ok" | "not_searched" | "disabled" | "unauthorized" |
                              "timeout" | "connection_error" | "ssl_error" |
                              "http_error" | "invalid_response" | "bad_request",
          "external_message": str | None,
        }
    """
    reserve_items = patients_db.search_patients(
        query=query, mrn=mrn, phone=phone, date_of_birth=date_of_birth, limit=limit
    )

    external_status = "not_searched"
    external_message = (
        "Hospital Directory API only supports searching by name/patient id "
        "— showing local (reserve) results only for phone, date of birth, or MRN searches."
    )
    external_items: List[Dict[str, Any]] = []

    if query:
        result = directory_client.search_patients(q=query, limit=limit)
        external_status = result["status"]
        external_message = result.get("message")
        if result["status"] == "ok":
            external_items = [_patient_to_history_shape(v) for v in result["items"]]

    return {
        "patients": reserve_items + external_items,
        "external_status": external_status,
        "external_message": external_message,
    }


def search_patients_insert_flow(search_text: str, limit: int = 20) -> Dict[str, Any]:
    """
    Merged search for the incident-creation autocomplete (backs
    search_service.search_patients / GET /api/records/search/patients).
    Only supports free-text search (matches the old single `q` parameter
    this endpoint always had).

    Returns the same {success, patients, count} shape as before, plus
    external_status/external_message for callers that want to surface it.
    """
    try:
        reserve_rows = patients_db.search_patients(query=search_text, limit=limit)
        reserve_items = [
            {
                "patient_admission_id": row["patient_id"],
                "full_name": row["full_name"],
                "first_name": row["first_name"],
                "last_name": row["last_name"],
                "document_number": None,  # not selected by patients_db.search_patients's reserve query
                "phone_number": row["phone"],
                "birth_date": row["date_of_birth"],
                "sex": None,
                "medical_file_number": row["mrn"],
                "admission_date": None,
                "source": "reserve",
            }
            for row in reserve_rows
        ]
    except Exception as e:
        return {"success": False, "patients": [], "count": 0, "error": f"Failed to search reserve patients: {str(e)}"}

    result = directory_client.search_patients(q=search_text, limit=limit)
    external_status = result["status"]
    external_items = [_patient_to_insert_shape(v) for v in result["items"]] if result["status"] == "ok" else []

    combined = reserve_items + external_items
    return {
        "success": True,
        "patients": combined,
        "count": len(combined),
        "external_status": external_status,
        "external_message": result.get("message"),
    }


_MIDDLE_NAME_SEARCH_DELAY_SECONDS = 0.4


def search_patients_missing_middle_name(first_name: str, last_name: str, limit: int = 20) -> Dict[str, Any]:
    """
    First name + last name known, middle/father's name unknown: tries names
    from the active middle-name candidate set (see middle_name_sets_service),
    in the set's own stored order (curated most-common-first, e.g. محمد/علي
    lead thirty_names.json), as real, throttled calls to
    search_patients_insert_flow — see HCAT-Middle-Name-Search-Assist-Plan.md.

    Stops at the FIRST candidate that returns a match and returns
    immediately. Originally this ran through all candidates and combined
    every match, on the theory that the same first+last pair could belong
    to more than one real patient with different middle names — correct in
    principle, but in real testing it meant waiting through up to all
    ~30 throttled real calls even for a common name found on the first or
    second try. Given the candidate list is already ordered by how common
    each name actually is, stopping at the first hit makes the common case
    fast and only costs the (already rare) chance of missing a second real
    patient sharing the same first+last name — an accepted trade-off since
    that person can still be found via a manual full-name search.

    This is real load against the live Hospital Directory API — one
    explicit user action (a single "find possible matches" request from the
    frontend), sequential and throttled, never fired automatically per
    keystroke. Backend-owned (not client-side looping) so the throttling/
    sequencing/early-stop is enforced in one place regardless of what UI
    calls it.
    """
    from api.services import middle_name_sets_service

    first_name = (first_name or "").strip()
    last_name = (last_name or "").strip()
    if not first_name or not last_name:
        return {"success": True, "patients": [], "count": 0, "tried": 0, "external_status": "not_searched", "external_message": None}

    try:
        candidates = middle_name_sets_service.get_active_set()["names"]
    except middle_name_sets_service.MiddleNameSetsError:
        candidates = []

    seen_ids = set()
    combined_patients: List[Dict[str, Any]] = []
    any_ok = False
    last_message = None
    tried = 0

    for candidate in candidates:
        tried += 1
        result = search_patients_insert_flow(f"{first_name} {candidate} {last_name}", limit=limit)
        if result.get("success") and result.get("external_status") == "ok":
            any_ok = True
        if result.get("external_message"):
            last_message = result["external_message"]
        for p in result.get("patients", []):
            pid = p.get("patient_admission_id")
            if pid not in seen_ids:
                seen_ids.add(pid)
                combined_patients.append(p)
        if combined_patients:
            break
        if tried < len(candidates):
            time.sleep(_MIDDLE_NAME_SEARCH_DELAY_SECONDS)

    return {
        "success": True,
        "patients": combined_patients,
        "count": len(combined_patients),
        "tried": tried,
        "external_status": "ok" if any_ok else "not_searched",
        "external_message": last_message,
    }


def resolve_patient_profile(patient_id: Any) -> Optional[Dict[str, Any]]:
    """
    Resolve ONE patient's full profile (demographics + incident stats),
    routing to reserve SQL or the external API based on the id shape.

    Returns None if genuinely not found (reserve id doesn't exist, or the
    external API responded 404). Raises ExternalPatientUnavailableError if
    the id is external but the API could not be reached/authenticated —
    this is deliberately NOT the same as "not found" (requirement: a
    network failure must show a clear "external search unavailable" state).
    """
    decoded = directory_client.decode_external_patient_id(str(patient_id))

    if decoded:
        result = directory_client.get_patient(decoded)
        if result["status"] == "not_found":
            return None
        if result["status"] != "ok":
            raise ExternalPatientUnavailableError(result["status"], result["message"])
        profile = _patient_to_profile_shape(result["patient"], str(patient_id))
    else:
        profile = patients_db.get_reserve_patient_profile(int(patient_id))
        if not profile:
            return None

    stats = patients_db.get_incident_stats_for_name(profile.get("PatientName"))
    profile["TotalIncidents"] = stats["TotalIncidents"]
    profile["LastVisitDate"] = stats["LastVisitDate"]
    return profile


def get_patient_by_id_insert_shape(patient_id: Any) -> Dict[str, Any]:
    """
    Backs search_service.get_patient_by_id (GET /api/records/patient/{id}) —
    currently unreachable from the frontend (verified: no caller in
    Front_End_Feedback_Analysis), kept correct for API completeness/parity.
    """
    try:
        profile = resolve_patient_profile(patient_id)
    except ExternalPatientUnavailableError as e:
        return {"success": False, "patient": None, "error": f"external_unavailable: {e.message}"}

    if not profile:
        return {"success": False, "patient": None, "error": "Patient not found"}

    return {
        "success": True,
        "patient": {
            "patient_admission_id": profile["PatientID"],
            "full_name": profile["PatientName"],
            "first_name": profile.get("PatientNameEnglish"),
            "last_name": None,
            "document_number": None,
            "phone_number": profile.get("Phone"),
            "birth_date": profile.get("DateOfBirth"),
            "sex": profile.get("Gender"),
            "medical_file_number": profile.get("MRN"),
            "admission_date": profile.get("RegistrationDate"),
            "source": profile.get("source"),
        },
    }
