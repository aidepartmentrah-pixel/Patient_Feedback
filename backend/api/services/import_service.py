"""
Import Service — Hospital Data Intake Pipeline
Handles template generation, Excel parsing, validation, and controlled import.
"""

import base64
import hashlib
from datetime import datetime, date
from io import BytesIO
from typing import Dict, Any, List, Optional, Tuple

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.utils import get_column_letter

from core.database import get_connection
from ..db_layer import import_db
from ..db_layer import ml_import_batch_db
from ..db_layer.incident_parent import assign_case_to_incident
from .case_service import create_case

# Source-system tag for ml.ImportSourceRecordMap — generalizes the exact
# idempotency pattern already proven by dbo.APP_DataMigration_Map for the
# Phase K legacy-migration path (see ML_ARCHITECTURE_DECISION_RECORD.md 4.8).
EXCEL_IMPORT_SOURCE_SYSTEM = "ExcelImportTemplate"

# Clinical risk type ID meaning "Ordinary" (no red flag / never event) —
# used as the default when the optional "Feedback Risk Type" column is blank,
# since case_service.create_case() requires clinical_risk_type_id.
DEFAULT_CLINICAL_RISK_TYPE_ID = 1


# ============================================================
# TEMPLATE COLUMN DEFINITIONS
# ============================================================

# (excel_header, lookup_list_key_or_inline_list, is_mandatory, col_width)
TEMPLATE_COLUMNS = [
    ("Incident Group Key",          None,               True,  22),
    ("Patient Name",                None,               True,  25),
    ("Complaint Date (YYYY-MM-DD)", None,               True,  22),
    ("Feedback Type (النوع)",       "feedback_types",   True,  20),
    ("Source (المصدر)",             "sources",          True,  20),
    ("Issuing Dept (قسم الصادر)",   "org_units",        True,  28),
    ("Domain",                      "domains",          True,  20),
    ("Category",                    "categories",       True,  22),
    ("Subcategory",                 "subcategories",    True,  22),
    ("Classification",              "classifications",  True,  30),
    ("Severity",                    "severities",       False, 16),
    ("Stage",                       "stages",           False, 16),
    ("Harm Level",                  "harm_levels",      False, 16),
    ("Feedback Risk Type",          "risk_types",       False, 20),
    ("Building",                    "buildings",        False, 18),
    ("Is Inpatient",                ["Yes", "No"],      False, 14),
    ("Complaint Text",              None,               True,  50),
    ("Immediate Action",            None,               False, 40),
    ("Taken Action (الإجراءات المتخذة)", None,          False, 40),
    ("Target Dept 1",               "org_units",        False, 28),
    ("Target Dept 2",               "org_units",        False, 28),
    ("Target Dept 3",               "org_units",        False, 28),
    ("Doctor 1",                    "doctors",          False, 25),
    ("Doctor 2",                    "doctors",          False, 25),
    ("Doctor 3",                    "doctors",          False, 25),
    ("Worker 1 (Full Name)",        None,               False, 25),
    ("Worker 2 (Full Name)",        None,               False, 25),
    ("Worker 3 (Full Name)",        None,               False, 25),
]

# Column index constants (0-based)
COL_GROUP_KEY    = 0
COL_PATIENT      = 1
COL_DATE         = 2
COL_FB_TYPE      = 3
COL_SOURCE       = 4
COL_ISSUING_DEPT = 5
COL_DOMAIN       = 6
COL_CATEGORY     = 7
COL_SUBCATEGORY  = 8
COL_CLASS        = 9
COL_SEVERITY     = 10
COL_STAGE        = 11
COL_HARM         = 12
COL_RISK         = 13
COL_BUILDING     = 14
COL_INPATIENT    = 15
COL_COMPLAINT    = 16
COL_IMMEDIATE    = 17
COL_TAKEN        = 18
COL_DEPT1        = 19
COL_DEPT2        = 20
COL_DEPT3        = 21
COL_DOCTOR1      = 22
COL_DOCTOR2      = 23
COL_DOCTOR3      = 24
COL_WORKER1      = 25
COL_WORKER2      = 26
COL_WORKER3      = 27

MAX_DATA_ROWS = 2000  # data validation applies up to this row


# ============================================================
# TEMPLATE GENERATOR
# ============================================================

def generate_template() -> BytesIO:
    """
    Build the Excel import template with live DB dropdowns.
    Returns BytesIO of the .xlsx file.
    """
    data = import_db.load_all_lookups()
    lookup_lists = data["lists"]

    wb = Workbook()

    # ---- Hidden lookups sheet ----
    ls = wb.active
    ls.title = "Lookups"
    ls.sheet_state = "hidden"

    # Each lookup category occupies one column in the Lookups sheet
    # Col order matches the TEMPLATE_COLUMNS lookup keys
    lookup_key_to_col: Dict[str, int] = {}   # key -> 1-based col number in Lookups sheet
    lookup_col = 1
    for _, lookup_key, _, _ in TEMPLATE_COLUMNS:
        if lookup_key is None or isinstance(lookup_key, list):
            continue
        if lookup_key in lookup_key_to_col:
            continue
        lookup_key_to_col[lookup_key] = lookup_col
        ls.cell(row=1, column=lookup_col, value=lookup_key)  # header row
        values = lookup_lists.get(lookup_key, [])
        for row_i, val in enumerate(values, start=2):
            ls.cell(row=row_i, column=lookup_col, value=val)
        lookup_col += 1

    # ---- Template sheet ----
    ts = wb.create_sheet("Import Template", 0)
    ts.sheet_view.rightToLeft = False

    header_fill = PatternFill("solid", fgColor="1F4E79")
    mandatory_fill = PatternFill("solid", fgColor="2E75B6")
    header_font = Font(bold=True, color="FFFFFF", size=11)
    border = Border(
        left=Side(style="thin"), right=Side(style="thin"),
        top=Side(style="thin"), bottom=Side(style="thin")
    )

    for col_i, (header, lookup_key, mandatory, width) in enumerate(TEMPLATE_COLUMNS, start=1):
        cell = ts.cell(row=1, column=col_i, value=header)
        cell.fill = mandatory_fill if mandatory else header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = border
        ts.column_dimensions[get_column_letter(col_i)].width = width

    ts.row_dimensions[1].height = 40
    ts.freeze_panes = "A2"

    # Apply data validation dropdowns
    for col_i, (_, lookup_key, _, _) in enumerate(TEMPLATE_COLUMNS, start=1):
        col_letter = get_column_letter(col_i)
        cell_range = f"{col_letter}2:{col_letter}{MAX_DATA_ROWS}"

        if lookup_key is None:
            continue

        if isinstance(lookup_key, list):
            # Inline small list (e.g. Yes/No)
            formula = '"' + ",".join(lookup_key) + '"'
            dv = DataValidation(type="list", formula1=formula, allow_blank=True)
        else:
            ls_col = lookup_key_to_col.get(lookup_key)
            if ls_col is None:
                continue
            values = lookup_lists.get(lookup_key, [])
            if not values:
                continue
            ls_col_letter = get_column_letter(ls_col)
            last_row = len(values) + 1
            formula = f"'Lookups'!${ls_col_letter}$2:${ls_col_letter}${last_row}"
            dv = DataValidation(type="list", formula1=formula, allow_blank=True)

        dv.sqref = cell_range
        ts.add_data_validation(dv)

    # Instructions row (row 2 is first data row)
    ts.cell(row=2, column=COL_GROUP_KEY + 1).value = "IMP-001"
    ts.cell(row=2, column=COL_PATIENT + 1).value = "Patient Name Here"
    ts.cell(row=2, column=COL_DATE + 1).value = datetime.today().strftime("%Y-%m-%d")
    ts.cell(row=2, column=COL_COMPLAINT + 1).value = "Complaint text here"

    # Style the example row lightly
    example_fill = PatternFill("solid", fgColor="EBF3FB")
    for col_i in range(1, len(TEMPLATE_COLUMNS) + 1):
        cell = ts.cell(row=2, column=col_i)
        cell.fill = example_fill

    buf = BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf


# ============================================================
# UPLOAD PIPELINE
# ============================================================

def process_upload(file_bytes: bytes, created_by_user_id: int = 1) -> Dict[str, Any]:
    """
    Full import pipeline:
      checksum dedup → parse → group → validate → per-group duplicate check
      → import valid groups → report
    Returns a structured report dict including base64-encoded rejected Excel.
    """
    file_checksum = hashlib.sha256(file_bytes).hexdigest()

    # 0. Batch-level duplicate check — has this exact file already been
    # imported successfully? (see ML_ARCHITECTURE_DECISION_RECORD.md 4.8)
    batch_conn = get_connection()
    batch_cursor = batch_conn.cursor()
    try:
        existing_batch = ml_import_batch_db.find_batch_by_checksum(batch_cursor, file_checksum)
        if existing_batch:
            return _empty_report(
                f"This exact file was already imported successfully on "
                f"{existing_batch['UploadedAt']} (batch #{existing_batch['ImportBatchID']}). "
                f"Re-upload blocked to prevent duplicate cases."
            )

        import_batch_id = ml_import_batch_db.create_import_batch(
            batch_cursor,
            original_file_name=None,
            file_checksum=file_checksum,
            template_version="v1",
            uploaded_by_user_id=created_by_user_id,
        )
        batch_conn.commit()
    finally:
        batch_cursor.close()
        batch_conn.close()

    data = import_db.load_all_lookups()
    maps = data["maps"]

    # 1. Parse
    rows = _parse_excel(file_bytes)
    if not rows:
        return _empty_report("No data rows found in the uploaded file.")

    # 2. Group by Incident Group Key
    groups = _group_rows(rows)

    valid_groups: List[Tuple[str, List[dict], dict]] = []
    rejected_groups: List[Dict[str, Any]] = []
    all_warnings: List[Dict[str, str]] = []

    # 3. Validate
    for group_key, group_rows in groups.items():
        result = _validate_group(group_key, group_rows, maps)
        if result["valid"]:
            valid_groups.append((group_key, group_rows, result))
        else:
            rejected_groups.append({
                "group_key": group_key,
                "rows": group_rows,
                "reason": result["error"],
            })
        all_warnings.extend(result.get("warnings", []))

    # 3.5 Record-level duplicate check — has this Incident Group Key already
    # been imported in a previous upload? Checked before any case creation.
    dedup_conn = get_connection()
    dedup_cursor = dedup_conn.cursor()
    try:
        still_valid_groups = []
        for group_key, group_rows, validation in valid_groups:
            already_imported = ml_import_batch_db.find_group_already_imported(
                dedup_cursor, EXCEL_IMPORT_SOURCE_SYSTEM, group_key
            )
            if already_imported:
                rejected_groups.append({
                    "group_key": group_key,
                    "rows": group_rows,
                    "reason": f"Incident Group Key '{group_key}' was already imported previously — duplicate",
                })
            else:
                still_valid_groups.append((group_key, group_rows, validation))
        valid_groups = still_valid_groups
    finally:
        dedup_cursor.close()
        dedup_conn.close()

    # 4. Import valid groups
    imported: List[Dict[str, Any]] = []
    new_patients_count = 0

    for group_key, group_rows, validation in valid_groups:
        try:
            imp_result = _import_group(group_key, validation["validated_rows"],
                                       validation["is_new_patient"], created_by_user_id,
                                       import_batch_id)
            imported.append(imp_result)
            if validation["is_new_patient"]:
                new_patients_count += 1
        except Exception as exc:
            rejected_groups.append({
                "group_key": group_key,
                "rows": group_rows,
                "reason": f"Import error: {exc}",
            })

    # 5. Build rejected Excel
    rejected_b64 = None
    if rejected_groups:
        rej_buf = _generate_rejected_excel(rejected_groups)
        rejected_b64 = base64.b64encode(rej_buf.getvalue()).decode()

    total_imported_rows = sum(r["rows_imported"] for r in imported)
    total_failed_rows_within_groups = sum(r.get("rows_failed", 0) for r in imported)
    total_rejected_rows = sum(len(r["rows"]) for r in rejected_groups) + total_failed_rows_within_groups

    # 6. Update batch summary (best-effort — never blocks the response)
    try:
        summary_conn = get_connection()
        summary_cursor = summary_conn.cursor()
        try:
            ml_import_batch_db.update_batch_summary(
                summary_cursor, import_batch_id,
                total_rows=len(rows),
                accepted_rows=total_imported_rows,
                rejected_rows=total_rejected_rows,
                created_case_count=total_imported_rows,
                status="Completed",
            )
            summary_conn.commit()
        finally:
            summary_cursor.close()
            summary_conn.close()
    except Exception as e:
        print(f"[IMPORT BATCH WARNING] Failed to update batch summary: {e}")

    return {
        "summary": {
            "import_batch_id": import_batch_id,
            "total_groups": len(groups),
            "imported_groups": len(imported),
            "rejected_groups": len(rejected_groups),
            "total_rows": len(rows),
            "imported_rows": total_imported_rows,
            "rejected_rows": total_rejected_rows,
            "new_patients_created": new_patients_count,
            "warnings_count": len(all_warnings),
        },
        "imported": imported,
        "rejected": [
            {"group_key": r["group_key"], "reason": r["reason"]}
            for r in rejected_groups
        ],
        "warnings": all_warnings,
        "rejected_excel_b64": rejected_b64,
    }


# ============================================================
# INTERNAL — PARSE
# ============================================================

def _parse_excel(file_bytes: bytes) -> List[Dict[str, Any]]:
    wb = load_workbook(BytesIO(file_bytes), data_only=True)
    # Prefer "Import Template" sheet, else use first visible sheet
    sheet = None
    for name in wb.sheetnames:
        if name == "Import Template":
            sheet = wb[name]
            break
    if sheet is None:
        sheet = wb.active

    headers = [cell.value for cell in sheet[1]]
    rows = []
    for row in sheet.iter_rows(min_row=2, values_only=True):
        # Skip completely empty rows
        if all(v is None or str(v).strip() == "" for v in row):
            continue
        row_dict = {}
        for i, val in enumerate(row):
            if i < len(headers) and headers[i]:
                row_dict[headers[i]] = val
        rows.append(row_dict)
    return rows


# ============================================================
# INTERNAL — GROUP
# ============================================================

def _group_rows(rows: List[Dict]) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = {}
    for row in rows:
        key_raw = row.get("Incident Group Key")
        key = str(key_raw).strip() if key_raw is not None else ""
        if not key:
            key = "__NO_KEY__"
        groups.setdefault(key, []).append(row)
    return groups


# ============================================================
# INTERNAL — VALIDATE
# ============================================================

def _get(row: Dict, col_index: int) -> Optional[str]:
    """Get cell value by column index, return stripped string or None."""
    header = TEMPLATE_COLUMNS[col_index][0]
    val = row.get(header)
    if val is None:
        return None
    return str(val).strip() or None


def _lookup(maps: Dict, category: str, value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    return maps.get(category, {}).get(value.lower().strip())


def _validate_group(group_key: str, rows: List[Dict], maps: Dict) -> Dict[str, Any]:
    warnings: List[Dict] = []
    validated_rows: List[Dict] = []
    patient_name = None

    for row_num, row in enumerate(rows, start=1):
        errors = []

        # --- Patient name (consistent across group) ---
        pname = _get(row, COL_PATIENT)
        if not pname:
            errors.append("Patient Name is missing")
        else:
            if patient_name is None:
                patient_name = pname
            elif pname.lower().strip() != patient_name.lower().strip():
                errors.append(
                    f"Row {row_num}: Patient Name '{pname}' differs from group patient '{patient_name}'"
                )

        # --- Date ---
        date_raw = _get(row, COL_DATE)
        parsed_date = None
        if not date_raw:
            errors.append("Complaint Date is missing")
        else:
            for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
                try:
                    parsed_date = datetime.strptime(str(date_raw).split(" ")[0], fmt).date()
                    break
                except ValueError:
                    continue
            if parsed_date is None:
                errors.append(f"Complaint Date '{date_raw}' is not a valid date (use YYYY-MM-DD)")

        # --- Complaint Text ---
        complaint = _get(row, COL_COMPLAINT)
        if not complaint:
            errors.append("Complaint Text is missing")

        # --- Feedback Type (mandatory) ---
        fb_type_name = _get(row, COL_FB_TYPE)
        fb_type_id = _lookup(maps, "feedback_types", fb_type_name)
        if not fb_type_name:
            errors.append("Feedback Type is missing")
        elif fb_type_id is None:
            errors.append(f"Feedback Type '{fb_type_name}' not found in database")

        # --- Source (mandatory) ---
        source_name = _get(row, COL_SOURCE)
        source_id = _lookup(maps, "sources", source_name)
        if not source_name:
            errors.append("Source is missing")
        elif source_id is None:
            errors.append(f"Source '{source_name}' not found in database")

        # --- Issuing Dept (mandatory) ---
        issuing_name = _get(row, COL_ISSUING_DEPT)
        issuing_id = _lookup(maps, "org_units", issuing_name)
        if not issuing_name:
            errors.append("Issuing Dept is missing")
        elif issuing_id is None:
            errors.append(f"Issuing Dept '{issuing_name}' not found — reject")

        # --- Classification (mandatory, drives domain/category/subcategory) ---
        class_name = _get(row, COL_CLASS)
        class_id = _lookup(maps, "classifications", class_name)
        domain_id = category_id = subcategory_id = None
        if not class_name:
            errors.append("Classification is missing")
        elif class_id is None:
            errors.append(f"Classification '{class_name}' not found — reject")
        else:
            chain = maps.get("classification_chains", {}).get(class_id, {})
            domain_id = chain.get("domain_id")
            category_id = chain.get("category_id")
            subcategory_id = chain.get("subcategory_id")

        # --- Optional lookups (warn+nullify if not found) ---
        severity_name = _get(row, COL_SEVERITY)
        severity_id = _lookup(maps, "severities", severity_name)
        if severity_name and severity_id is None:
            warnings.append({"group_key": group_key, "message": f"Severity '{severity_name}' not found — imported as empty"})

        stage_name = _get(row, COL_STAGE)
        stage_id = _lookup(maps, "stages", stage_name)
        if stage_name and stage_id is None:
            warnings.append({"group_key": group_key, "message": f"Stage '{stage_name}' not found — imported as empty"})

        harm_name = _get(row, COL_HARM)
        harm_id = _lookup(maps, "harm_levels", harm_name)
        if harm_name and harm_id is None:
            warnings.append({"group_key": group_key, "message": f"Harm Level '{harm_name}' not found — imported as empty"})

        risk_name = _get(row, COL_RISK)
        risk_id = _lookup(maps, "risk_types", risk_name)
        if risk_name and risk_id is None:
            warnings.append({"group_key": group_key, "message": f"Risk Type '{risk_name}' not found — imported as empty"})

        building_name = _get(row, COL_BUILDING)
        building_id = _lookup(maps, "buildings", building_name)
        if building_name and building_id is None:
            warnings.append({"group_key": group_key, "message": f"Building '{building_name}' not found — imported as empty"})

        # --- Is Inpatient ---
        inpatient_val = _get(row, COL_INPATIENT)
        is_inpatient = str(inpatient_val).strip().lower() in ("yes", "نعم", "1", "true") if inpatient_val else False

        # --- Target Departments (reject if specified but not found) ---
        target_dept_ids: List[int] = []
        for dept_col in (COL_DEPT1, COL_DEPT2, COL_DEPT3):
            dept_name = _get(row, dept_col)
            if dept_name:
                dept_id = _lookup(maps, "org_units", dept_name)
                if dept_id is None:
                    errors.append(f"Target Department '{dept_name}' not found — reject")
                else:
                    target_dept_ids.append(dept_id)

        # --- Doctors (warn if not found) ---
        doctor_ids: List[Tuple[int, str]] = []
        for doc_col in (COL_DOCTOR1, COL_DOCTOR2, COL_DOCTOR3):
            doc_name = _get(row, doc_col)
            if doc_name:
                doc_id = _lookup(maps, "doctors", doc_name)
                if doc_id is None:
                    warnings.append({"group_key": group_key, "message": f"Doctor '{doc_name}' not found — imported without doctor linkage"})
                else:
                    doctor_ids.append((doc_id, doc_name))

        # --- Workers (warn if not found) ---
        worker_ids: List[int] = []
        for wrk_col in (COL_WORKER1, COL_WORKER2, COL_WORKER3):
            wrk_name = _get(row, wrk_col)
            if wrk_name:
                wrk_id = import_db.find_worker_by_name(wrk_name)
                if wrk_id is None:
                    warnings.append({"group_key": group_key, "message": f"Worker '{wrk_name}' not found in HR system — skipped"})
                else:
                    worker_ids.append(wrk_id)

        if errors:
            # First error in any row rejects the whole group
            return {
                "valid": False,
                "error": f"Row {row_num}: {errors[0]}",
                "warnings": warnings,
            }

        validated_rows.append({
            "patient_name": pname,
            "feedback_date": parsed_date,
            "feedback_intent_id": fb_type_id,
            "source_id": source_id,
            "issuing_org_unit_id": issuing_id,
            "domain_id": domain_id,
            "category_id": category_id,
            "subcategory_id": subcategory_id,
            "classification_id": class_id,
            "severity_id": severity_id,
            "stage_id": stage_id,
            "harm_id": harm_id,
            "risk_type_id": risk_id,
            "building_id": building_id,
            "is_inpatient": is_inpatient,
            "complaint_text": complaint,
            "immediate_action": _get(row, COL_IMMEDIATE),
            "taken_action": _get(row, COL_TAKEN),
            "target_dept_ids": target_dept_ids,
            "doctor_ids": doctor_ids,
            "worker_ids": worker_ids,
        })

    # --- Patient ambiguity check (done once per group) ---
    is_new_patient = False
    if patient_name:
        conn = get_connection()
        cursor = conn.cursor()
        try:
            match_count = import_db.count_patient_exact(patient_name, cursor)
        finally:
            cursor.close()
            conn.close()

        if match_count > 1:
            return {
                "valid": False,
                "error": f"Patient '{patient_name}' matches {match_count} records — ambiguous, cannot import",
                "warnings": warnings,
            }
        is_new_patient = (match_count == 0)

    return {
        "valid": True,
        "error": None,
        "warnings": warnings,
        "validated_rows": validated_rows,
        "patient_name": patient_name,
        "is_new_patient": is_new_patient,
    }


# ============================================================
# INTERNAL — IMPORT
# ============================================================

def _build_case_service_payload(vrow: Dict, patient_name: str) -> Dict[str, Any]:
    """
    Translate an import_service validated-row dict into the data shape
    case_service.create_case() expects. The Excel template doesn't collect a
    distinct "incident date" or a "requires explanation" checkbox, so both
    get sensible defaults (incident_date falls back to the complaint date,
    matching the same fallback used elsewhere in the app; requires_explanation
    defaults to False — the FSM's own red-flag/never-event check, driven by
    the template's optional Risk Type column, is what actually opens the
    explanation workflow for imported rows that need it).
    """
    return {
        "record_type_id": 1,  # Excel import template is complaint-only, not Notice
        "patient_name": patient_name,
        "feedback_received_date": vrow["feedback_date"],
        "incident_date": vrow["feedback_date"],
        "feedback_intent_type_id": vrow["feedback_intent_id"],
        "source_id": vrow.get("source_id"),
        "issuing_department_id": vrow["issuing_org_unit_id"],
        "domain_id": vrow.get("domain_id"),
        "category_id": vrow.get("category_id"),
        "subcategory_id": vrow.get("subcategory_id"),
        "classification_id": vrow.get("classification_id"),
        "severity_id": vrow.get("severity_id"),
        "stage_id": vrow.get("stage_id"),
        "harm_id": vrow.get("harm_id"),
        "clinical_risk_type_id": vrow.get("risk_type_id") or DEFAULT_CLINICAL_RISK_TYPE_ID,
        "requires_explanation": False,
        "building_id": vrow.get("building_id") or 1,  # default building when blank
        "is_inpatient": vrow.get("is_inpatient", False),
        "is_morbidity": False,
        "complaint_text": vrow["complaint_text"],
        "immediate_action": vrow.get("immediate_action") or "",
        "taken_action": vrow.get("taken_action") or "",
        "target_department_ids": vrow.get("target_dept_ids") or [],
        "doctors": [
            {"doctor_id": doc_id, "doctor_name": doc_name}
            for doc_id, doc_name in vrow.get("doctor_ids", [])
        ],
        "employees": [
            {"employee_id": emp_id, "full_name": ""}
            for emp_id in vrow.get("worker_ids", [])
        ],
    }


def _import_group(
    group_key: str,
    validated_rows: List[Dict],
    is_new_patient: bool,
    created_by_user_id: int,
    import_batch_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Create the shared incident parent for this group, then create each row's
    case through the centralized case_service.create_case() (context=
    'BulkImport') — the same validation/FSM/ML-job-registration logic manual
    insert uses, rather than a separate raw-SQL implementation.

    Each row is its own atomic unit (case + its ml.EmbeddingProcessingJob
    commit together, inside create_case()). This means a later row failing
    within a group does NOT roll back earlier rows already committed in the
    same group — a deliberate trade-off (see ML_ARCHITECTURE_DECISION_RECORD.md
    Stage 4) that favors per-case durability and ML-job guarantees over
    whole-group atomicity. Partial success is reported accurately via
    rows_imported/rows_failed/row_errors rather than silently rolling back
    already-valid rows or misreporting the whole group as rejected.
    """
    first = validated_rows[0]
    patient_name = first["patient_name"]

    conn = get_connection()
    conn.autocommit = False
    cursor = conn.cursor()

    try:
        # Create reserve patient if new
        if is_new_patient:
            import_db.create_reserve_patient(patient_name, created_by_user_id, cursor)

        # Create the shared incident parent for this group
        incident_id = import_db.insert_incident(
            patient_name=patient_name,
            feedback_intent_type_id=first["feedback_intent_id"],
            issuing_org_unit_id=first["issuing_org_unit_id"],
            building_id=first.get("building_id") or 1,
            is_inpatient=first.get("is_inpatient", False),
            created_by_user_id=created_by_user_id,
            cursor=cursor,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        cursor.close()
        conn.close()
        raise
    cursor.close()
    conn.close()

    rows_imported = 0
    row_errors: List[Dict[str, Any]] = []

    for idx, vrow in enumerate(validated_rows, start=1):
        try:
            data = _build_case_service_payload(vrow, patient_name)
            result = create_case(data, context='BulkImport', save_mode='workflow')

            if not result.get("success"):
                row_errors.append({"row_index": idx, "error": result.get("message", "Unknown error")})
                continue

            case_id = int(result["id"])

            # Link this case to the group's shared incident parent
            assign_case_to_incident(case_id, incident_id)

            # Record-level import idempotency mapping (generalizes
            # APP_DataMigration_Map's proven pattern). One group can span
            # multiple rows/cases, so each gets its own '{group_key}#{idx}'
            # external ID — group-level dedup uses a prefix match instead
            # (see find_group_already_imported).
            external_record_id = f"{group_key}#{idx}"
            map_conn = get_connection()
            map_cursor = map_conn.cursor()
            try:
                ml_import_batch_db.record_source_map(
                    map_cursor, import_batch_id, EXCEL_IMPORT_SOURCE_SYSTEM, external_record_id, case_id
                )
                map_conn.commit()
            finally:
                map_cursor.close()
                map_conn.close()

            rows_imported += 1

        except Exception as exc:
            row_errors.append({"row_index": idx, "error": str(exc)})

    return {
        "group_key": group_key,
        "incident_id": incident_id,
        "rows_imported": rows_imported,
        "rows_failed": len(row_errors),
        "row_errors": row_errors,
        "new_patient": is_new_patient,
    }


# ============================================================
# INTERNAL — REJECTED EXCEL
# ============================================================

def _generate_rejected_excel(rejected_groups: List[Dict]) -> BytesIO:
    wb = Workbook()
    ws = wb.active
    ws.title = "Rejected Groups"

    # Build header from template columns + reason column
    headers = [col[0] for col in TEMPLATE_COLUMNS] + ["Rejection Reason"]
    for col_i, h in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col_i, value=h)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="C00000")

    row_i = 2
    for group in rejected_groups:
        reason = group["reason"]
        for row_data in group["rows"]:
            for col_i, (header, _, _, _) in enumerate(TEMPLATE_COLUMNS, start=1):
                ws.cell(row=row_i, column=col_i, value=row_data.get(header))
            ws.cell(row=row_i, column=len(TEMPLATE_COLUMNS) + 1, value=reason)
            row_i += 1

    buf = BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf


def _empty_report(message: str) -> Dict[str, Any]:
    return {
        "summary": {
            "total_groups": 0, "imported_groups": 0, "rejected_groups": 0,
            "total_rows": 0, "imported_rows": 0, "rejected_rows": 0,
            "new_patients_created": 0, "warnings_count": 0,
        },
        "imported": [],
        "rejected": [],
        "warnings": [{"group_key": "-", "message": message}],
        "rejected_excel_b64": None,
    }
