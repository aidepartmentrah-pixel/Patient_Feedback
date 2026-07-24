"""
Doctor/Worker Directory Adapter (Session C2/C3 — search + incident linkage only)

Merges HCAT's own reserve tables (APP_RESERVE_DOCTOR, APP_RESERVE_WORKER)
with doctors/workers from the Hospital Directory API into ONE normalized
model, same pattern as api.services.patient_directory_service for patients.

Scope: this backs search_service.search_doctors/search_employees and the
doctor/employee linkage step in case_service.py (create + update incident).
It does NOT touch the separate Doctor/Worker History & Reporting pages
(api/db_layer/doctors_db.py, worker_reporting_db.py) — those are still
view-based and out of scope here.

Identity model — DIFFERENT from patients
-----------------------------------------
Patients only ever link to incidents by free-text name (no FK), so an
opaque external id could flow straight through. Doctors and workers link to
incidents via a real int foreign key
(APP_IncidentCaseDoctor.DoctorID / APP_IncidentCaseEmployee.EmployeeID) —
an API-sourced string id cannot go directly into those columns. So:

  - For SEARCH/DISPLAY, external doctors/workers carry an opaque id
    (hospital_directory_client.encode_external_id), exactly like patients.
  - For LINKAGE (attaching a doctor/worker to an incident), the opaque id
    must first be resolved to a real reserve-table int id — see
    materialize_doctor_id()/materialize_employee_id() below. If the same
    external person is selected again later, the existing reserve row
    (matched by ExternalDoctorID/ExternalEmployeeID) is reused, not
    duplicated.
"""

from typing import Any, Dict, List, Optional

from core.database import get_connection
from core import hospital_directory_client as directory_client


# =============================================================================
# DOCTORS
# =============================================================================

def _reserve_search_doctors(search_text: str, limit: int) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        pattern = f"%{search_text}%"
        cursor.execute("""
            SELECT TOP (?) DoctorID, DoctorName, Specialty, IsActive, ExternalDoctorID
            FROM dbo.APP_RESERVE_DOCTOR
            WHERE DoctorName LIKE ? AND IsActive = 1
            ORDER BY DoctorName
        """, (limit, pattern))
        return [
            {
                "doctor_id": row.DoctorID,
                "name": row.DoctorName,
                "speciality_id": None,
                "speciality_name": row.Specialty,
                "is_active": bool(row.IsActive),
                "is_admitted": False,
                "is_clinic": False,
                "source": "reserve",
                "external_id": row.ExternalDoctorID,
            }
            for row in cursor.fetchall()
        ]
    finally:
        cursor.close()
        conn.close()


def _visit_to_doctor_shape(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "doctor_id": directory_client.encode_external_id(doc["doctor_id"]),
        "name": doc.get("full_name"),
        "speciality_id": doc.get("specialty_id"),
        "speciality_name": doc.get("specialty_name"),
        "is_active": doc.get("is_active", True),
        "is_admitted": False,
        "is_clinic": False,
        "source": "external",
    }


def search_doctors_merged(search_text: str, limit: int = 20) -> Dict[str, Any]:
    """Merged reserve + external doctor search, same {success, doctors, count} shape as before."""
    try:
        reserve_items = _reserve_search_doctors(search_text, limit)
    except Exception as e:
        return {"success": False, "doctors": [], "count": 0, "error": f"Failed to search reserve doctors: {str(e)}"}

    # A doctor already materialized into the reserve table (see
    # materialize_doctor_id) is now represented by its reserve row -- the
    # external record for the same person must not also be listed, or the
    # same doctor shows up twice (once as the real int id, once as the
    # still-unmaterialized-looking ext__ id).
    materialized_external_ids = {r["external_id"] for r in reserve_items if r.get("external_id")}

    result = directory_client.search_doctors(q=search_text, limit=limit)
    external_items = [
        _visit_to_doctor_shape(d) for d in result["items"]
        if str(d.get("doctor_id")) not in materialized_external_ids
    ] if result["status"] == "ok" else []

    combined = reserve_items + external_items
    return {
        "success": True,
        "doctors": combined,
        "count": len(combined),
        "external_status": result["status"],
        "external_message": result.get("message"),
    }


def get_doctor_by_id_merged(doctor_id: Any) -> Dict[str, Any]:
    """Exact-lookup, reserve or external, mirrors the old get_doctor_by_id shape."""
    external_id = directory_client.decode_external_id(str(doctor_id))

    if external_id:
        result = directory_client.get_doctor(external_id)
        if result["status"] == "not_found":
            return {"success": False, "doctor": None, "error": "Doctor not found"}
        if result["status"] != "ok":
            return {"success": False, "doctor": None, "error": f"external_unavailable: {result['status']}: {result['message']}"}
        return {"success": True, "doctor": _visit_to_doctor_shape(result["doctor"])}

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT DoctorID, DoctorName, Specialty, IsActive
            FROM dbo.APP_RESERVE_DOCTOR WHERE DoctorID = ?
        """, (doctor_id,))
        row = cursor.fetchone()
        if not row:
            return {"success": False, "doctor": None, "error": "Doctor not found"}
        return {
            "success": True,
            "doctor": {
                "doctor_id": row.DoctorID, "name": row.DoctorName,
                "speciality_id": None, "speciality_name": row.Specialty,
                "is_active": bool(row.IsActive), "is_admitted": False, "is_clinic": False,
                "source": "reserve",
            },
        }
    finally:
        cursor.close()
        conn.close()


def materialize_doctor_id(doctor_id: Any, doctor_name: str = "") -> Optional[int]:
    """
    Resolve a doctor_id (reserve int, or opaque external string) to a real
    APP_RESERVE_DOCTOR.DoctorID int suitable for an
    APP_IncidentCaseDoctor.DoctorID foreign key.

    - Plain reserve int id: returned as-is (assumed valid — same trust level
      as before this change, no new validation added).
    - External id: find-or-create a reserve row (matched by
      ExternalDoctorID), so repeat selection of the same API doctor reuses
      one row. Returns None only if doctor_id is falsy.
    """
    if not doctor_id:
        return None

    external_id = directory_client.decode_external_id(str(doctor_id))
    if not external_id:
        return int(doctor_id)

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT DoctorID FROM dbo.APP_RESERVE_DOCTOR WHERE ExternalDoctorID = ?",
            (external_id,),
        )
        row = cursor.fetchone()
        if row:
            return row.DoctorID

        cursor.execute("""
            INSERT INTO dbo.APP_RESERVE_DOCTOR (
                DoctorName, Specialty, IsActive, SourceSystem, ExternalDoctorID, LastSyncedAt
            )
            OUTPUT INSERTED.DoctorID
            VALUES (?, NULL, 1, 'hospital_directory_api', ?, GETDATE())
        """, (doctor_name or f"Doctor {external_id}", external_id))
        new_id = cursor.fetchone()[0]
        conn.commit()
        return new_id
    finally:
        cursor.close()
        conn.close()


# =============================================================================
# WORKERS
# =============================================================================

def _reserve_search_workers(search_text: str, limit: int) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        pattern = f"%{search_text}%"
        cursor.execute("""
            SELECT TOP (?) EmployeeID, FullName, JobTitle, JobID, DepartmentID, SectionID, AdministrationID, IsManager, IsActive, ExternalEmployeeID
            FROM dbo.APP_RESERVE_WORKER
            WHERE FullName LIKE ? AND IsActive = 1
            ORDER BY FullName
        """, (limit, pattern))
        return [
            {
                "employee_id": row.EmployeeID,
                "full_name": row.FullName,
                "job_title": row.JobTitle,
                "job_id": row.JobID,
                "department_id": row.DepartmentID,
                "section_id": row.SectionID,
                "administration_id": row.AdministrationID,
                "is_manager": bool(row.IsManager) if row.IsManager is not None else False,
                "is_active": bool(row.IsActive),
                "source": "reserve",
                "external_id": row.ExternalEmployeeID,
            }
            for row in cursor.fetchall()
        ]
    finally:
        cursor.close()
        conn.close()


def _visit_to_worker_shape(w: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "employee_id": directory_client.encode_external_id(w["employee_id"]),
        "full_name": w.get("full_name"),
        "job_title": w.get("job_title"),
        "job_id": w.get("job_id"),
        "department_id": w.get("department_id"),
        "section_id": w.get("section_id"),
        "administration_id": w.get("administration_id"),
        "is_manager": w.get("is_manager", False),
        "is_active": w.get("is_active", True),
        "source": "external",
    }


def search_workers_merged(search_text: str, limit: int = 20) -> Dict[str, Any]:
    """Merged reserve + external worker search, same {success, employees, count} shape as before."""
    try:
        reserve_items = _reserve_search_workers(search_text, limit)
    except Exception as e:
        return {"success": False, "employees": [], "count": 0, "error": f"Failed to search reserve workers: {str(e)}"}

    # A worker already materialized into the reserve table (see
    # materialize_employee_id) is now represented by its reserve row -- the
    # external record for the same person must not also be listed, or the
    # same worker shows up twice.
    materialized_external_ids = {r["external_id"] for r in reserve_items if r.get("external_id")}

    result = directory_client.search_workers(q=search_text, limit=limit)
    external_items = [
        _visit_to_worker_shape(w) for w in result["items"]
        if str(w.get("employee_id")) not in materialized_external_ids
    ] if result["status"] == "ok" else []

    combined = reserve_items + external_items
    return {
        "success": True,
        "employees": combined,
        "count": len(combined),
        "external_status": result["status"],
        "external_message": result.get("message"),
    }


def get_employee_by_id_merged(employee_id: Any) -> Dict[str, Any]:
    """Exact-lookup, reserve or external, mirrors the old get_employee_by_id shape."""
    external_id = directory_client.decode_external_id(str(employee_id))

    if external_id:
        result = directory_client.get_worker(external_id)
        if result["status"] == "not_found":
            return {"success": False, "employee": None, "error": "Employee not found"}
        if result["status"] != "ok":
            return {"success": False, "employee": None, "error": f"external_unavailable: {result['status']}: {result['message']}"}
        return {"success": True, "employee": _visit_to_worker_shape(result["worker"])}

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT EmployeeID, FullName, JobTitle, JobID, DepartmentID, SectionID, AdministrationID, IsManager, IsActive
            FROM dbo.APP_RESERVE_WORKER WHERE EmployeeID = ?
        """, (employee_id,))
        row = cursor.fetchone()
        if not row:
            return {"success": False, "employee": None, "error": "Employee not found"}
        return {
            "success": True,
            "employee": {
                "employee_id": row.EmployeeID, "full_name": row.FullName,
                "job_title": row.JobTitle, "job_id": row.JobID,
                "department_id": row.DepartmentID, "section_id": row.SectionID,
                "administration_id": row.AdministrationID,
                "is_manager": bool(row.IsManager) if row.IsManager is not None else False,
                "is_active": bool(row.IsActive), "source": "reserve",
            },
        }
    finally:
        cursor.close()
        conn.close()


def materialize_employee_id(employee_id: Any, full_name: str = "") -> Optional[int]:
    """
    Resolve an employee_id (reserve int, or opaque external string) to a
    real APP_RESERVE_WORKER.EmployeeID int suitable for an
    APP_IncidentCaseEmployee.EmployeeID foreign key. Same find-or-create
    pattern as materialize_doctor_id — see its docstring.
    """
    if not employee_id:
        return None

    external_id = directory_client.decode_external_id(str(employee_id))
    if not external_id:
        return int(employee_id)

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT EmployeeID FROM dbo.APP_RESERVE_WORKER WHERE ExternalEmployeeID = ?",
            (external_id,),
        )
        row = cursor.fetchone()
        if row:
            return row.EmployeeID

        cursor.execute("""
            INSERT INTO dbo.APP_RESERVE_WORKER (
                FullName, IsActive, SourceSystem, ExternalEmployeeID, LastSyncedAt
            )
            OUTPUT INSERTED.EmployeeID
            VALUES (?, 1, 'hospital_directory_api', ?, GETDATE())
        """, (full_name or f"Worker {external_id}", external_id))
        new_id = cursor.fetchone()[0]
        conn.commit()
        return new_id
    finally:
        cursor.close()
        conn.close()
