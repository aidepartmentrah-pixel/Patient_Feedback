"""
Search Service
Provides search functionality for patients, doctors, and employees.
Used by the insert page to search and select entities from the database.
"""

from typing import List, Dict, Any
from api.services import patient_directory_service
from api.services import staff_directory_service


def search_patients(search_text: str, limit: int = 20) -> Dict[str, Any]:
    """
    Search for patients by name (free-text only — this is the endpoint
    behind the incident-creation autocomplete, GET /api/records/search/patients).

    SESSION C1: merges HCAT's reserve table (APP_RESERVE_PATIENT) with the
    Hospital Directory API instead of the old hospital view — see
    api.services.patient_directory_service for the merge/normalization logic.
    """
    return patient_directory_service.search_patients_insert_flow(search_text, limit)


def search_doctors(search_text: str, limit: int = 20) -> Dict[str, Any]:
    """
    Search for doctors by name (free-text) — behind the incident-creation
    autocomplete, GET /api/records/search/doctors.

    SESSION C2: merges HCAT's reserve table (APP_RESERVE_DOCTOR) with the
    Hospital Directory API instead of the old VW_Doctors view — see
    api.services.staff_directory_service for the merge/normalization logic.
    """
    return staff_directory_service.search_doctors_merged(search_text, limit)


def search_employees(search_text: str, limit: int = 20) -> Dict[str, Any]:
    """
    Search for employees/workers by name (free-text) — behind the
    incident-creation autocomplete, GET /api/records/search/employees.

    SESSION C3: merges HCAT's reserve table (APP_RESERVE_WORKER — newly
    created, no reserve table existed before this) with the Hospital
    Directory API instead of the old VW_HrEmployeeProfileView view — see
    api.services.staff_directory_service for the merge/normalization logic.
    """
    return staff_directory_service.search_workers_merged(search_text, limit)


def get_patient_by_id(patient_admission_id) -> Dict[str, Any]:
    """
    Get a specific patient by id — verifies patient selection. Confirmed
    unreachable from the current frontend (no caller found in
    Front_End_Feedback_Analysis), kept correct for API completeness.

    SESSION C1: patient_admission_id is now either a reserve
    PatientAdmissionID or an opaque external id (see
    hospital_directory_client.encode_external_patient_id) — routes to
    reserve SQL or the Hospital Directory API accordingly.
    """
    return patient_directory_service.get_patient_by_id_insert_shape(patient_admission_id)


def get_doctor_by_id(doctor_id) -> Dict[str, Any]:
    """
    Get a specific doctor by id — verifies doctor selection.

    SESSION C2: doctor_id is now either a reserve DoctorID or an opaque
    external id (see hospital_directory_client.encode_external_id) —
    routes to reserve SQL or the Hospital Directory API accordingly.
    """
    return staff_directory_service.get_doctor_by_id_merged(doctor_id)


def get_employee_by_id(employee_id) -> Dict[str, Any]:
    """
    Get a specific employee by id — verifies employee selection.

    SESSION C3: employee_id is now either a reserve EmployeeID (in the new
    APP_RESERVE_WORKER table) or an opaque external id — routes to reserve
    SQL or the Hospital Directory API accordingly.
    """
    return staff_directory_service.get_employee_by_id_merged(employee_id)
