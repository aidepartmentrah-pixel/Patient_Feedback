"""
Import Service — Hospital Data Intake Pipeline
Handles template generation, Excel parsing, validation, and controlled import.
"""

import base64
import hashlib
from datetime import datetime, date
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side, Protection
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.utils import get_column_letter

from core.database import get_connection
from ..db_layer import import_db
from ..db_layer import ml_import_batch_db
from ..db_layer.incident_parent import assign_case_to_incident
from .case_service import create_case
from . import staff_directory_service
from . import patient_directory_service

# Source-system tag for ml.ImportSourceRecordMap — generalizes the exact
# idempotency pattern already proven by dbo.APP_DataMigration_Map for the
# Phase K legacy-migration path (see ML_ARCHITECTURE_DECISION_RECORD.md 4.8).
EXCEL_IMPORT_SOURCE_SYSTEM = "ExcelImportTemplate"

# Injected into each parsed row dict to carry its real Excel sheet row
# number through grouping/validation -- lets the review screen reference a
# specific physical row for inline edits (see the /rows/{row_number} patch
# endpoints). Not a real template header, so it can't collide with one.
ROW_NUMBER_KEY = "__row_number__"

# Grouping key used when Incident Group Key is blank -- a real, presentable
# string (not an internal-looking sentinel) since it's shown directly in
# the review screen as the group's label.
MISSING_GROUP_KEY_LABEL = "(No Group Key)"

# Clinical risk type ID meaning "Ordinary" (no red flag / never event) —
# used as the default when the optional "Feedback Risk Type" column is blank,
# since case_service.create_case() requires clinical_risk_type_id.
DEFAULT_CLINICAL_RISK_TYPE_ID = 1

# Must match case_service.py's is_red_flag/is_never_event checks
# (clinical_risk_type_id == 2 / == 3) -- used here only to badge a row for
# human visibility in the review screen/report, since 'import_closed' save
# mode keeps every imported case Closed regardless of risk type.
RED_FLAG_RISK_TYPE_ID = 2
NEVER_EVENT_RISK_TYPE_ID = 3

# Where staged (uploaded-but-not-yet-confirmed) files live between
# stage_upload() and confirm_import(), keyed by ImportBatchID. Filesystem-
# based rather than a new DB column/table: no schema change needed, and it
# works fine across multiple backend worker processes on one server since
# they share a filesystem.
STAGING_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "import_staging"


def _staged_file_path(import_batch_id: int) -> Path:
    return STAGING_DIR / f"{import_batch_id}.xlsx"


# Generated batch reports are persisted here so the batch history page can
# offer a "download report" link for past batches, not just the one just
# confirmed -- the original uploaded rows for rejected groups aren't stored
# anywhere else (only accepted rows become real case records), so the
# report itself is the only durable record of what a past batch contained.
REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "import_reports"


def _report_file_path(import_batch_id: int) -> Path:
    return REPORTS_DIR / f"{import_batch_id}.xlsx"


def get_report_path(import_batch_id: int) -> Optional[Path]:
    """Path to a past batch's saved report, or None if it doesn't exist."""
    path = _report_file_path(import_batch_id)
    return path if path.exists() else None


# ============================================================
# TEMPLATE COLUMN DEFINITIONS
# ============================================================

# (excel_header, lookup_list_key_or_inline_list, is_mandatory, col_width)
# Record Type is optional (defaults to Complaint if blank) rather than
# mandatory, so templates/sheets filled out before this column existed
# still work without every row being rejected.
# Feedback Type was dropped: dbo.APP_LOOKUP_FEEDBACK_INTENT_TYPE only has 2
# values ("Improvement Opportunity" / "Notice") that map 1:1 onto Record
# Type (Complaint/Notice) -- it's auto-derived instead, see
# _feedback_intent_type_id_for_record_type().
TEMPLATE_COLUMNS = [
    ("Incident Group Key (رقم)",    None,               True,  22),
    ("Patient Name",                None,               True,  25),
    ("Incident Date",               None,               True,  22),
    ("Received Date",               None,               True,  22),
    ("Record Type",                 ["Complaint", "Notice"], False, 16),
    ("Source (المصدر)",             "sources",          True,  20),
    ("Issuing Dept (قسم الصادر)",   "org_units",        True,  28),
    ("Domain",                      "domains",          True,  20),
    ("Category",                    "categories",       True,  22),
    ("Subcategory",                 "subcategories",    True,  22),
    ("Classification",              "classifications",  True,  30),
    ("Severity",                    "severities",       True,  16),
    ("Stage",                       "stages",           True,  16),
    ("Harm Level",                  "harm_levels",      True,  16),
    ("Feedback Risk Type",          "risk_types",       True,  20),
    ("Building",                    "buildings",        True,  18),
    ("Is Inpatient",                ["Yes", "No"],      False, 14),
    ("Complaint Text",              None,               True,  50),
    ("Immediate Action",            None,               False, 40),
    ("Taken Action (الإجراءات المتخذة)", None,          False, 40),
    ("Target Dept",                 "org_units",        True,  28),
    ("Doctor 1",                    "doctors",          False, 25),
    ("Doctor 2",                    "doctors",          False, 25),
    ("Doctor 3",                    "doctors",          False, 25),
    ("Worker 1 (Full Name)",        "workers",          False, 25),
    ("Worker 2 (Full Name)",        "workers",          False, 25),
    ("Worker 3 (Full Name)",        "workers",          False, 25),
]

# Column index constants (0-based)
COL_GROUP_KEY      = 0
COL_PATIENT        = 1
COL_INCIDENT_DATE  = 2
COL_DATE           = 3  # Received Date
COL_RECORD_TYPE    = 4
COL_SOURCE         = 5
COL_ISSUING_DEPT   = 6
COL_DOMAIN         = 7
COL_CATEGORY       = 8
COL_SUBCATEGORY    = 9
COL_CLASS          = 10
COL_SEVERITY       = 11
COL_STAGE          = 12
COL_HARM           = 13
COL_RISK           = 14
COL_BUILDING       = 15
COL_INPATIENT      = 16
COL_COMPLAINT      = 17
COL_IMMEDIATE      = 18
COL_TAKEN          = 19
COL_TARGET_DEPT    = 20
COL_DOCTOR1        = 21
COL_DOCTOR2        = 22
COL_DOCTOR3        = 23
COL_WORKER1        = 24
COL_WORKER2        = 25
COL_WORKER3        = 26

MAX_DATA_ROWS = 2000  # data validation applies up to this row

RECORD_TYPE_IDS = {"complaint": 1, "notice": 2}
DEFAULT_RECORD_TYPE_ID = 1  # Complaint

# dbo.APP_LOOKUP_FEEDBACK_INTENT_TYPE: 1=Improvement Opportunity (used for
# Complaint records), 2=Notice (used for Notice records) -- confirmed
# 1:1 with Record Type, see TEMPLATE_COLUMNS comment above.
FEEDBACK_INTENT_TYPE_BY_RECORD_TYPE = {1: 1, 2: 2}


def _feedback_intent_type_id_for_record_type(record_type_id: int) -> int:
    return FEEDBACK_INTENT_TYPE_BY_RECORD_TYPE.get(record_type_id, 1)

DIRECTORY_LOOKUP_LIMIT = 500  # matches the Hospital Directory API's per-call max (see hospital_directory_client)


# ============================================================
# DIRECTORY LOOKUPS (doctors / workers / patients)
# ============================================================

def _load_directory_lookups() -> Dict[str, Dict[str, Any]]:
    """
    Merged doctor/worker lookups (reserve + Hospital Directory API),
    replacing the old APP_LOOKUP_DOCTOR/HR_EMPLOYEES_TABLE-only source.
    IDs may be a plain reserve int or an opaque external id string --
    case_service.create_case() already resolves either one via
    materialize_doctor_id()/materialize_employee_id(), so no extra
    resolution is needed here or at commit time.

    Capped at DIRECTORY_LOOKUP_LIMIT reserve + DIRECTORY_LOOKUP_LIMIT
    external entries per category (a single API page) -- a hospital with
    more active doctors/workers than that would need pagination added
    here; not attempted since it's well past what a dropdown list is
    usable for anyway.
    """
    maps: Dict[str, Any] = {}
    lists: Dict[str, Any] = {}

    doc_result = staff_directory_service.search_doctors_merged("", limit=DIRECTORY_LOOKUP_LIMIT)
    doctors = doc_result.get("doctors", []) if doc_result.get("success") else []
    maps["doctors"] = {(d["name"] or "").lower().strip(): d["doctor_id"] for d in doctors if d.get("name")}
    lists["doctors"] = [d["name"] for d in doctors if d.get("name")]

    wrk_result = staff_directory_service.search_workers_merged("", limit=DIRECTORY_LOOKUP_LIMIT)
    workers = wrk_result.get("employees", []) if wrk_result.get("success") else []
    maps["workers"] = {(w["full_name"] or "").lower().strip(): w["employee_id"] for w in workers if w.get("full_name")}
    lists["workers"] = [w["full_name"] for w in workers if w.get("full_name")]

    return {"maps": maps, "lists": lists}


def _load_lookups() -> Dict[str, Dict[str, Any]]:
    """import_db.load_all_lookups() plus the directory-sourced doctor/worker lists."""
    data = import_db.load_all_lookups()
    directory_data = _load_directory_lookups()
    data["maps"].update(directory_data["maps"])
    data["lists"].update(directory_data["lists"])
    return data


def _count_patient_matches(full_name: str) -> int:
    """
    Exact-name match count across reserve + Hospital Directory patients.
    Replaces the old count against the retired dbo.VW_PatientAdmission view.
    """
    result = patient_directory_service.search_patients_insert_flow(full_name, limit=50)
    if not result.get("success"):
        raise Exception(result.get("error") or "Patient search failed")
    target = full_name.strip().lower()
    return sum(
        1 for p in result["patients"]
        if (p.get("full_name") or "").strip().lower() == target
    )


# ============================================================
# TEMPLATE GENERATOR
# ============================================================

_INSTRUCTIONS_TITLE = "تعليمات استخدام قالب استيراد الحالات"

# Case, Incident, and Closed are kept in English inside the Arabic text on
# purpose -- staff were trained on the app using these English terms, so
# translating them would introduce unfamiliar vocabulary instead of helping.
_INSTRUCTIONS_TEXT = [
    "كل صف في هذا الملف يمثل Case واحد وليس Incident. الصفوف التي لها نفس "
    "رقم Incident Group Key سيتم دمجها معًا تحت Incident واحد في النظام.",
    "رقم Incident Group Key يجب أن يكون رقمًا فقط (مثال: 1)، بدون حروف أو رموز.",
    "الأعمدة التي تحتوي على قائمة منسدلة (مثل التصنيف، القسم، الطبيب) يجب اختيار "
    "القيمة منها فقط. لا تكتب نصًا غير موجود في القائمة، وإلا سيتم رفض الصف مع توضيح السبب.",
    "لا تقم بتغيير ترتيب الأعمدة أو أسماء العناوين أو حذف/إضافة أعمدة. "
    "الملف محمي لمنع ذلك، ويمكنك فقط تعبئة الخلايا المخصصة للبيانات.",
    "إذا مر وقت طويل منذ تحميل هذا الملف، يفضل تحميل نسخة جديدة قبل التعبئة "
    "لضمان أن القوائم المنسدلة (الأطباء، الأقسام، وغيرها) محدثة.",
    "بعد رفع الملف، سيظهر لك تقرير يوضح كل Case تمت إضافتها بنجاح (مع رقمها "
    "الجديد في النظام) وكل Case تم رفضها مع سبب الرفض بوضوح.",
    "جميع الـ Cases التي يتم استيرادها تدخل النظام بحالة Closed مباشرة، "
    "ولا تمر عبر نظام الرسائل أو صندوق الوارد.",
    "إذا كان الطبيب أو الموظف غير موجود في القائمة، يمكنك ترك الخانة فارغة "
    "والمتابعة، أو إضافته أولاً من صفحة الإعدادات > الأطباء ثم إعادة تحميل القالب.",
]


def _add_instructions_sheet(wb: Workbook) -> None:
    """
    Visible, RTL instructions sheet, first tab in the file (but not the
    active one on open -- see generate_template's wb.active -- so the file
    opens straight onto the fillable grid; Instructions is one click away).
    Numbered-card layout: badge column + wrapped text column, bordered rows.
    """
    ins = wb.create_sheet("تعليمات - Instructions", 0)
    ins.sheet_view.rightToLeft = True
    ins.sheet_view.showGridLines = False
    ins.column_dimensions["A"].width = 6
    ins.column_dimensions["B"].width = 100

    title_fill = PatternFill("solid", fgColor="1F4E79")
    badge_fill = PatternFill("solid", fgColor="2E75B6")
    card_fill = PatternFill("solid", fgColor="F2F7FC")
    thin = Side(style="thin", color="D0DCE8")
    card_border = Border(left=thin, right=thin, top=thin, bottom=thin)

    # Title banner
    ins.merge_cells("A1:B1")
    title_cell = ins.cell(row=1, column=1, value=_INSTRUCTIONS_TITLE)
    title_cell.font = Font(bold=True, size=16, color="FFFFFF")
    title_cell.fill = title_fill
    title_cell.alignment = Alignment(horizontal="center", vertical="center")
    ins.row_dimensions[1].height = 36

    row_i = 3
    for i, line in enumerate(_INSTRUCTIONS_TEXT, start=1):
        badge = ins.cell(row=row_i, column=1, value=i)
        badge.font = Font(bold=True, size=13, color="FFFFFF")
        badge.fill = badge_fill
        badge.alignment = Alignment(horizontal="center", vertical="center")
        badge.border = card_border

        text = ins.cell(row=row_i, column=2, value=line)
        text.font = Font(size=12)
        text.fill = card_fill
        text.alignment = Alignment(horizontal="right", vertical="center", wrap_text=True)
        text.border = card_border

        # Rough auto-height: ~45 chars per line at this column width/font size
        ins.row_dimensions[row_i].height = max(30, 15 * (len(line) // 45 + 1))
        ins.row_dimensions[row_i + 1].height = 8  # spacer between cards
        row_i += 2

    ins.protection.sheet = True
    ins.protection.formatCells = True
    ins.protection.formatColumns = True
    ins.protection.insertColumns = True
    ins.protection.deleteColumns = True


def generate_template() -> BytesIO:
    """
    Build the Excel import template with live DB dropdowns.
    Returns BytesIO of the .xlsx file.
    """
    data = _load_lookups()
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
    ts = wb.create_sheet("Import Template", 1)
    ts.sheet_view.rightToLeft = True

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

    # Apply data validation dropdowns (+ a couple of special-cased columns
    # that aren't list-based lookups: date type-checking, numeric-only Group Key)
    for col_i, (_, lookup_key, _, _) in enumerate(TEMPLATE_COLUMNS, start=1):
        col_letter = get_column_letter(col_i)
        cell_range = f"{col_letter}2:{col_letter}{MAX_DATA_ROWS}"
        col_index0 = col_i - 1

        if col_index0 in (COL_INCIDENT_DATE, COL_DATE):
            dv = DataValidation(
                type="date", operator="greaterThan", formula1="1900-01-01",
                allow_blank=True,
            )
            dv.error = "Please enter a valid date (use the cell's date picker)."
            dv.errorTitle = "Invalid date"
            dv.sqref = cell_range
            ts.add_data_validation(dv)
            for row_i in range(2, MAX_DATA_ROWS + 1):
                ts.cell(row=row_i, column=col_i).number_format = "yyyy-mm-dd"
            continue

        if col_index0 == COL_GROUP_KEY:
            dv = DataValidation(
                type="whole", operator="greaterThan", formula1="0",
                allow_blank=True,
            )
            dv.error = "Incident Group Key must be a number only (e.g. 1), no letters."
            dv.errorTitle = "Numbers only"
            dv.sqref = cell_range
            ts.add_data_validation(dv)
            continue

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
    ts.cell(row=2, column=COL_GROUP_KEY + 1).value = 1
    ts.cell(row=2, column=COL_PATIENT + 1).value = "Patient Name Here"
    ts.cell(row=2, column=COL_INCIDENT_DATE + 1).value = datetime.today().date()
    ts.cell(row=2, column=COL_INCIDENT_DATE + 1).number_format = "yyyy-mm-dd"
    ts.cell(row=2, column=COL_DATE + 1).value = datetime.today().date()
    ts.cell(row=2, column=COL_DATE + 1).number_format = "yyyy-mm-dd"
    ts.cell(row=2, column=COL_COMPLAINT + 1).value = "Complaint text here"

    # Style the example row lightly
    example_fill = PatternFill("solid", fgColor="EBF3FB")
    for col_i in range(1, len(TEMPLATE_COLUMNS) + 1):
        cell = ts.cell(row=2, column=col_i)
        cell.fill = example_fill

    # ---- Lock the structure: header + column layout can't be touched,
    # only the data-entry cells below it. Matches the decision that the
    # template must be "fully locked structure" given the original problem
    # was a customer editing it without knowing they shouldn't.
    for row in ts.iter_rows(min_row=2, max_row=MAX_DATA_ROWS, max_col=len(TEMPLATE_COLUMNS)):
        for cell in row:
            cell.protection = Protection(locked=False)
    ts.protection.sheet = True
    ts.protection.formatCells = True
    ts.protection.formatColumns = True
    ts.protection.formatRows = True
    ts.protection.insertColumns = True
    ts.protection.insertRows = True
    ts.protection.deleteColumns = True
    ts.protection.deleteRows = True
    ts.protection.sort = True
    ts.protection.autoFilter = True
    ts.protection.selectLockedCells = False
    ts.protection.selectUnlockedCells = False

    _add_instructions_sheet(wb)
    # Instructions is the first tab (visible, one click away) but the file
    # should open straight onto the fillable grid, not a wall of text.
    wb.active = wb.sheetnames.index(ts.title)

    buf = BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf


# ============================================================
# UPLOAD PIPELINE
# ============================================================

def _is_flagged_risk(risk_type_id: Optional[int]) -> bool:
    """Red flag or never event — see RED_FLAG_RISK_TYPE_ID/NEVER_EVENT_RISK_TYPE_ID."""
    return risk_type_id in (RED_FLAG_RISK_TYPE_ID, NEVER_EVENT_RISK_TYPE_ID)


def _summarize_row_errors(row_results: List[Dict[str, Any]]) -> str:
    """One-line summary of a group's per-row errors, for contexts that only
    show a single reason string (the report Excel, legacy display)."""
    parts = []
    for r in row_results:
        if r["errors"]:
            messages = "; ".join(e["message"] for e in r["errors"])
            parts.append(f"Row {r['row_number']}: {messages}")
    return " | ".join(parts) if parts else "Unknown validation error"


def _validate_and_group(file_bytes: bytes) -> Dict[str, Any]:
    """
    Parse -> group by Incident Group Key -> validate each group -> check
    per-group duplicate status. Pure computation, no case/incident writes —
    shared by stage_upload() (preview) and confirm_import() (re-validated
    fresh right before commit, so anything that changed in the lookups
    between preview and confirm — e.g. a doctor added in Settings — is
    picked up automatically).
    """
    data = _load_lookups()
    maps = data["maps"]

    rows = _parse_excel(file_bytes)

    groups = _group_rows(rows)
    valid_groups: List[Tuple[str, List[dict], dict]] = []
    rejected_groups: List[Dict[str, Any]] = []
    all_warnings: List[Dict[str, str]] = []

    for group_key, group_rows in groups.items():
        result = _validate_group(group_key, group_rows, maps)
        if result["valid"]:
            valid_groups.append((group_key, group_rows, result))
        else:
            rejected_groups.append({
                "group_key": group_key,
                "rows": group_rows,
                "reason": _summarize_row_errors(result["rows"]),
                "row_results": result["rows"],
                "status": "red",
            })
        all_warnings.extend(result.get("warnings", []))

    # Record-level duplicate check — has this Incident Group Key already
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
                    "row_results": validation["rows"],
                    "status": "duplicate",
                })
            else:
                still_valid_groups.append((group_key, group_rows, validation))
        valid_groups = still_valid_groups
    finally:
        dedup_cursor.close()
        dedup_conn.close()

    return {
        "rows": rows,
        "groups": groups,
        "valid_groups": valid_groups,
        "rejected_groups": rejected_groups,
        "warnings": all_warnings,
    }


def _json_safe(value: Any) -> Any:
    """Make a raw cell value JSON-serializable (dates/datetimes -> ISO strings)."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def _row_preview(row_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Shape one row for the review grid using the CURRENT template's column
    set, not whatever headers happened to be in the uploaded file. A file
    uploaded before a template change (e.g. missing the newer Incident Date
    column, or still carrying a since-removed one like the old Feedback
    Type) would otherwise leak stale, unpatchable field names into the grid
    -- editing them would silently do nothing, since patch_staged_rows()
    only recognizes today's TEMPLATE_COLUMNS headers. Columns the file
    never had simply show up blank here, ready to be filled in through the
    grid even though the original upload didn't have them.
    """
    raw = row_result["raw"]
    return {
        "row_number": row_result["row_number"],
        "errors": row_result["errors"],
        "fields": {col[0]: _json_safe(raw.get(col[0])) for col in TEMPLATE_COLUMNS},
        "derived_hierarchy": row_result.get("derived_hierarchy") or {"domain": None, "category": None, "subcategory": None},
    }


def _build_preview_groups(valid_groups: List[Tuple[str, List[dict], dict]],
                           rejected_groups: List[Dict[str, Any]],
                           order: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Shape validated/rejected groups for the review screen: one entry per
    Incident Group Key, colored per the decided taxonomy (green/yellow ready,
    red/duplicate blocked), with a red-flag/never-event badge that's purely
    informational -- it never blocks the group or changes its color. Each
    group carries its individual rows (row_number, per-row errors, editable
    field values) so the review grid can show and fix a specific row.

    `order` is the Incident Group Key sequence as it first appeared in the
    uploaded file (see _group_rows -- a plain dict, insertion-ordered). Valid
    and rejected groups are validated as two separate passes below, which
    would otherwise bunch every ready group first and every rejected group
    after; sorting by `order` at the end restores the file's natural
    row-by-row order so a reviewer scanning the screen sees groups in the
    same sequence they appear in Excel, statuses interleaved.
    """
    preview: List[Dict[str, Any]] = []

    for group_key, group_rows, validation in valid_groups:
        validated_rows = [r["data"] for r in validation["rows"]]
        preview.append({
            "group_key": group_key,
            "status": "yellow" if validation["is_new_patient"] else "green",
            "is_new_patient": validation["is_new_patient"],
            "patient_name": validation["patient_name"],
            "row_count": len(validated_rows),
            "has_flagged_risk": any(_is_flagged_risk(r.get("risk_type_id")) for r in validated_rows),
            "reason": None,
            "rows": [_row_preview(r) for r in validation["rows"]],
        })

    for rejected in rejected_groups:
        preview.append({
            "group_key": rejected["group_key"],
            "status": rejected["status"],  # "red" or "duplicate"
            "is_new_patient": False,
            "patient_name": None,
            "row_count": len(rejected["rows"]),
            "has_flagged_risk": False,
            "reason": rejected["reason"],
            "rows": [_row_preview(r) for r in rejected.get("row_results", [])],
        })

    if order is not None:
        order_index = {key: i for i, key in enumerate(order)}
        preview.sort(key=lambda g: order_index.get(g["group_key"], len(order_index)))

    return preview


def stage_upload(file_bytes: bytes, uploaded_by_user_id: int = 1) -> Dict[str, Any]:
    """
    Phase 1: checksum dedup -> parse -> group -> validate -> per-group
    duplicate check -> build a review-screen preview grouped by incident.
    Nothing is written to APP_Incident/APP_IncidentCase here — the staged
    file is persisted so confirm_import(import_batch_id) can commit it later.
    """
    file_checksum = hashlib.sha256(file_bytes).hexdigest()

    # Batch-level duplicate check — has this exact file already been
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
            uploaded_by_user_id=uploaded_by_user_id,
        )
        batch_conn.commit()
    finally:
        batch_cursor.close()
        batch_conn.close()

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    _staged_file_path(import_batch_id).write_bytes(file_bytes)

    validated = _validate_and_group(file_bytes)
    if not validated["rows"]:
        return _empty_report("No data rows found in the uploaded file.")

    preview_groups = _build_preview_groups(validated["valid_groups"], validated["rejected_groups"], order=list(validated["groups"].keys()))

    summary_conn = get_connection()
    summary_cursor = summary_conn.cursor()
    try:
        ml_import_batch_db.update_batch_summary(
            summary_cursor, import_batch_id,
            total_rows=len(validated["rows"]),
            status="PendingReview",
        )
        summary_conn.commit()
    finally:
        summary_cursor.close()
        summary_conn.close()

    return {
        "import_batch_id": import_batch_id,
        "status": "PendingReview",
        "total_rows": len(validated["rows"]),
        "groups": preview_groups,
        "warnings": validated["warnings"],
    }


def _load_pending_batch(import_batch_id: int) -> Dict[str, Any]:
    """Fetch a batch and assert it's still PendingReview, or raise ValueError."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        batch = ml_import_batch_db.get_batch(cursor, import_batch_id)
    finally:
        cursor.close()
        conn.close()

    if batch is None:
        raise ValueError(f"Import batch {import_batch_id} not found.")
    if batch["Status"] != "PendingReview":
        raise ValueError(
            f"Import batch {import_batch_id} is not awaiting review (status: {batch['Status']})."
        )
    return batch


def patch_staged_rows(import_batch_id: int, patches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Apply field edits directly onto the staged Excel file for a still-
    PendingReview batch, then re-validate. patches: [{"row_number": int,
    "fields": {header: value}}, ...] -- the caller (PATCH /rows/{n}) always
    passes a single-row list, but this stays list-shaped since the file
    open/save is one cycle regardless of how many rows it covers. No other
    code path needs to change: resume_preview()/confirm_import() already
    just re-read whatever's currently in the staged file.

    Incident Group Key is deliberately not patchable here -- changing it
    would move a row into a different incident group entirely, which needs
    a full re-group, not a single-cell edit. Fix that one in Excel and
    re-upload, same as before this feature existed.
    """
    _load_pending_batch(import_batch_id)  # raises ValueError if not found/not PendingReview

    staged_path = _staged_file_path(import_batch_id)
    if not staged_path.exists():
        raise ValueError(
            f"Staged file for batch {import_batch_id} is no longer available — please re-upload."
        )

    header_to_col = {col[0]: i + 1 for i, col in enumerate(TEMPLATE_COLUMNS)}
    group_key_header = TEMPLATE_COLUMNS[COL_GROUP_KEY][0]
    date_headers = {TEMPLATE_COLUMNS[COL_INCIDENT_DATE][0], TEMPLATE_COLUMNS[COL_DATE][0]}

    wb = load_workbook(staged_path)
    ws = wb["Import Template"] if "Import Template" in wb.sheetnames else wb.active

    for patch in patches:
        row_number = patch.get("row_number")
        if not isinstance(row_number, int) or row_number < 2:
            continue
        for header, value in (patch.get("fields") or {}).items():
            if header == group_key_header:
                continue  # not patchable, see docstring
            col_i = header_to_col.get(header)
            if col_i is None:
                continue  # unknown field name, ignore rather than error the whole batch
            cell = ws.cell(row=row_number, column=col_i)
            if header in date_headers and value:
                cell.value = datetime.strptime(str(value), "%Y-%m-%d").date()
                cell.number_format = "yyyy-mm-dd"
            else:
                cell.value = value if value not in ("", None) else None

    wb.save(staged_path)

    validated = _validate_and_group(staged_path.read_bytes())
    preview_groups = _build_preview_groups(validated["valid_groups"], validated["rejected_groups"], order=list(validated["groups"].keys()))
    return {
        "import_batch_id": import_batch_id,
        "status": "PendingReview",
        "total_rows": len(validated["rows"]),
        "groups": preview_groups,
        "warnings": validated["warnings"],
    }


def get_editable_lookups() -> Dict[str, List[str]]:
    """
    Plain internal lookup lists for the review grid's lookup-field editors
    (Classification, Department, Domain, etc.) -- small, fetched once,
    filtered client-side. Doctor/Worker are deliberately excluded: those
    use live search instead (GET /api/records/search/doctors|employees),
    same directory the manual Insert Record form already searches.
    """
    data = _load_lookups()
    lists = dict(data["lists"])
    lists.pop("doctors", None)
    lists.pop("workers", None)
    return lists


def discard_batch(import_batch_id: int) -> None:
    """Discard a still-PendingReview batch: delete its DB row and staged file."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        batch = ml_import_batch_db.get_batch(cursor, import_batch_id)
        if batch is None:
            raise ValueError(f"Import batch {import_batch_id} not found.")
        if batch["Status"] == "Completed":
            raise ValueError(
                f"Import batch {import_batch_id} has already been confirmed and can't be discarded."
            )
        ml_import_batch_db.delete_batch(cursor, import_batch_id)
        conn.commit()
    finally:
        cursor.close()
        conn.close()

    staged_path = _staged_file_path(import_batch_id)
    if staged_path.exists():
        staged_path.unlink()


def resume_preview(import_batch_id: int) -> Dict[str, Any]:
    """
    Re-open the review screen for a batch that's still PendingReview --
    e.g. the user uploaded, then refreshed/navigated away before clicking
    Add, losing the in-browser preview state. Re-reads the still-staged
    file and re-validates fresh, same as stage_upload()'s first pass, just
    without re-creating the batch row or re-checking the file checksum.
    """
    _load_pending_batch(import_batch_id)

    staged_path = _staged_file_path(import_batch_id)
    if not staged_path.exists():
        raise ValueError(
            f"Staged file for batch {import_batch_id} is no longer available — please re-upload."
        )
    file_bytes = staged_path.read_bytes()

    validated = _validate_and_group(file_bytes)
    preview_groups = _build_preview_groups(validated["valid_groups"], validated["rejected_groups"], order=list(validated["groups"].keys()))

    return {
        "import_batch_id": import_batch_id,
        "status": "PendingReview",
        "total_rows": len(validated["rows"]),
        "groups": preview_groups,
        "warnings": validated["warnings"],
    }


def confirm_import(import_batch_id: int, confirmed_by_user_id: int = 1) -> Dict[str, Any]:
    """
    Phase 2: re-validate the staged file fresh (catches anything that
    changed in the lookups since staging) and actually commit every fully
    valid incident group. Call only after the user has reviewed
    stage_upload()'s preview and clicked Add.
    """
    _load_pending_batch(import_batch_id)

    staged_path = _staged_file_path(import_batch_id)
    if not staged_path.exists():
        raise ValueError(
            f"Staged file for batch {import_batch_id} is no longer available — please re-upload."
        )
    file_bytes = staged_path.read_bytes()

    validated = _validate_and_group(file_bytes)
    rows = validated["rows"]
    groups = validated["groups"]
    valid_groups = validated["valid_groups"]
    rejected_groups = validated["rejected_groups"]
    all_warnings = validated["warnings"]

    # Import valid groups
    imported: List[Dict[str, Any]] = []
    imported_with_rows: List[Tuple[Dict[str, Any], List[Dict]]] = []
    new_patients_count = 0

    for group_key, group_rows, validation in valid_groups:
        try:
            validated_rows = [r["data"] for r in validation["rows"]]
            imp_result = _import_group(group_key, validated_rows,
                                       validation["is_new_patient"], confirmed_by_user_id,
                                       import_batch_id)
            imp_result["has_flagged_risk"] = any(
                _is_flagged_risk(r.get("risk_type_id")) for r in validated_rows
            )
            imported.append(imp_result)
            imported_with_rows.append((imp_result, group_rows))
            if validation["is_new_patient"]:
                new_patients_count += 1
        except Exception as exc:
            rejected_groups.append({
                "group_key": group_key,
                "rows": group_rows,
                "reason": f"Import error: {exc}",
                "status": "red",
            })

    # Full report Excel: green (imported, with new Incident Number) +
    # red/blue (rejected/duplicate, with reason) -- always generated, not
    # just when something was rejected, since it's the receipt of the batch.
    report_buf = _generate_import_report_excel(imported_with_rows, rejected_groups)
    report_bytes = report_buf.getvalue()
    rejected_b64 = base64.b64encode(report_bytes).decode()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _report_file_path(import_batch_id).write_bytes(report_bytes)

    total_imported_rows = sum(r["rows_imported"] for r in imported)
    total_failed_rows_within_groups = sum(r.get("rows_failed", 0) for r in imported)
    total_rejected_rows = sum(len(r["rows"]) for r in rejected_groups) + total_failed_rows_within_groups

    # Update batch summary (best-effort — never blocks the response)
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

    # The staged file has served its purpose — nothing left to re-confirm.
    try:
        staged_path.unlink(missing_ok=True)
    except OSError:
        pass

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
            {"group_key": r["group_key"], "reason": r["reason"], "status": r.get("status", "red")}
            for r in rejected_groups
        ],
        "warnings": all_warnings,
        "rejected_excel_b64": rejected_b64,
    }


def process_upload(file_bytes: bytes, created_by_user_id: int = 1) -> Dict[str, Any]:
    """
    One-shot convenience wrapper: stage then immediately confirm, preserving
    the pre-review-screen behavior (used by existing tests and any caller
    that doesn't need the review step). New UI flows should call
    stage_upload() and confirm_import() separately instead.
    """
    staged = stage_upload(file_bytes, uploaded_by_user_id=created_by_user_id)
    if "import_batch_id" not in staged:
        return staged  # blocked before staging (e.g. duplicate file, empty file)
    return confirm_import(staged["import_batch_id"], confirmed_by_user_id=created_by_user_id)


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
    for row_number, row in enumerate(sheet.iter_rows(min_row=2, values_only=True), start=2):
        # Skip completely empty rows
        if all(v is None or str(v).strip() == "" for v in row):
            continue
        row_dict = {ROW_NUMBER_KEY: row_number}
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
    header = TEMPLATE_COLUMNS[COL_GROUP_KEY][0]
    for row in rows:
        key_raw = row.get(header)
        if isinstance(key_raw, (int, float)) and float(key_raw).is_integer():
            # Excel hands back a real number for the numeric-only Group Key
            # column -- normalize 1.0/1 to the same "1" grouping key.
            key = str(int(key_raw))
        else:
            key = str(key_raw).strip() if key_raw is not None else ""
        if not key:
            key = MISSING_GROUP_KEY_LABEL
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


def _parse_date_cell(raw: Any, field_label: str) -> Tuple[Optional[date], Optional[str]]:
    """Parse a date-typed template cell, returning (parsed_date, error_message)."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None, f"{field_label} is missing"
    if isinstance(raw, datetime):
        return raw.date(), None
    if isinstance(raw, date):
        return raw, None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(str(raw).strip().split(" ")[0], fmt).date(), None
        except ValueError:
            continue
    return None, f"{field_label} '{raw}' is not a valid date (use YYYY-MM-DD)"


def _validate_group(group_key: str, rows: List[Dict], maps: Dict) -> Dict[str, Any]:
    """
    Validates every row in a group -- every row is checked regardless of
    earlier rows' errors (unlike the old version, which stopped at the
    first bad row), so the review screen can show and let the user fix
    each broken row individually. A group is valid only if every one of
    its rows ends up with zero errors.
    """
    warnings: List[Dict] = []
    row_results: List[Dict[str, Any]] = []
    patient_name = None

    for row in rows:
        row_number = row.get(ROW_NUMBER_KEY)
        errors: List[Dict[str, str]] = []

        def add_error(field: str, message: str) -> None:
            errors.append({"field": field, "message": message})

        # --- Patient name (consistent across group) ---
        pname = _get(row, COL_PATIENT)
        if not pname:
            add_error("Patient Name", "Patient Name is missing")
        else:
            if patient_name is None:
                patient_name = pname
            elif pname.lower().strip() != patient_name.lower().strip():
                add_error("Patient Name", f"Patient Name '{pname}' differs from group patient '{patient_name}'")

        # --- Incident Date / Received Date ---
        # Both are Excel date-typed, so openpyxl usually hands back a real
        # datetime/date object -- handled directly rather than round-tripped
        # through a string. Plain text is still accepted as a fallback for
        # legacy sheets copy-pasted from before these columns had a date type.
        incident_date_raw = row.get(TEMPLATE_COLUMNS[COL_INCIDENT_DATE][0])
        parsed_incident_date, incident_date_error = _parse_date_cell(incident_date_raw, "Incident Date")
        if incident_date_error:
            add_error("Incident Date", incident_date_error)

        date_raw = row.get(TEMPLATE_COLUMNS[COL_DATE][0])
        parsed_date, received_date_error = _parse_date_cell(date_raw, "Received Date")
        if received_date_error:
            add_error("Received Date", received_date_error)

        # You can't receive a complaint before the incident that prompted it
        # actually happened -- same rule case_service.create_case() enforces
        # for manual entry, checked here too so it shows as a clear preview
        # rejection instead of a raw error at confirm time.
        if parsed_incident_date and parsed_date and parsed_incident_date > parsed_date:
            add_error("Incident Date", "Incident Date cannot be after Received Date")

        # --- Complaint Text ---
        complaint = _get(row, COL_COMPLAINT)
        if not complaint:
            add_error("Complaint Text", "Complaint Text is missing")

        # --- Source (mandatory) ---
        source_name = _get(row, COL_SOURCE)
        source_id = _lookup(maps, "sources", source_name)
        if not source_name:
            add_error("Source", "Source is missing")
        elif source_id is None:
            add_error("Source", f"Source '{source_name}' not found in database")

        # --- Issuing Dept (mandatory) ---
        issuing_name = _get(row, COL_ISSUING_DEPT)
        issuing_id = _lookup(maps, "org_units", issuing_name)
        if not issuing_name:
            add_error("Issuing Dept", "Issuing Dept is missing")
        elif issuing_id is None:
            add_error("Issuing Dept", f"Issuing Dept '{issuing_name}' not found — reject")

        # --- Record Type (optional, defaults to Complaint) ---
        record_type_name = _get(row, COL_RECORD_TYPE)
        if record_type_name:
            record_type_id = RECORD_TYPE_IDS.get(record_type_name.lower().strip())
            if record_type_id is None:
                add_error("Record Type", f"Record Type '{record_type_name}' not recognized — use Complaint or Notice")
        else:
            record_type_id = DEFAULT_RECORD_TYPE_ID

        # --- Classification (mandatory for Complaint rows; Notice rows
        # don't require it, mirroring case_service's own Notice exception —
        # drives domain/category/subcategory when present) ---
        class_name = _get(row, COL_CLASS)
        class_id = _lookup(maps, "classifications", class_name)
        domain_id = category_id = subcategory_id = None
        if not class_name:
            if record_type_id != RECORD_TYPE_IDS["notice"]:
                add_error("Classification", "Classification is missing")
        elif class_id is None:
            add_error("Classification", f"Classification '{class_name}' not found — reject")
        else:
            chain = maps.get("classification_chains", {}).get(class_id, {})
            domain_id = chain.get("domain_id")
            category_id = chain.get("category_id")
            subcategory_id = chain.get("subcategory_id")

        # Display-only breadcrumb, computed whenever Classification resolved
        # -- independent of `errors`/`row_data` below, which stay gated on
        # the row being fully valid. An otherwise-invalid row (e.g. a bad
        # Severity) still has a real, resolved Classification chain worth
        # showing the reviewer, not just a blank "derived" placeholder.
        derived_hierarchy = {
            "domain": maps.get("domain_names", {}).get(domain_id),
            "category": maps.get("category_names", {}).get(category_id),
            "subcategory": maps.get("subcategory_names", {}).get(subcategory_id),
        }

        # --- Severity / Stage / Harm Level / Feedback Risk Type / Building
        # (all mandatory now -- was warn-and-continue, per explicit decision
        # to require every field with no exceptions) ---
        severity_name = _get(row, COL_SEVERITY)
        severity_id = _lookup(maps, "severities", severity_name)
        if not severity_name:
            add_error("Severity", "Severity is missing")
        elif severity_id is None:
            add_error("Severity", f"Severity '{severity_name}' not found — reject")

        stage_name = _get(row, COL_STAGE)
        stage_id = _lookup(maps, "stages", stage_name)
        if not stage_name:
            add_error("Stage", "Stage is missing")
        elif stage_id is None:
            add_error("Stage", f"Stage '{stage_name}' not found — reject")

        harm_name = _get(row, COL_HARM)
        harm_id = _lookup(maps, "harm_levels", harm_name)
        if not harm_name:
            add_error("Harm Level", "Harm Level is missing")
        elif harm_id is None:
            add_error("Harm Level", f"Harm Level '{harm_name}' not found — reject")

        risk_name = _get(row, COL_RISK)
        risk_id = _lookup(maps, "risk_types", risk_name)
        if not risk_name:
            add_error("Feedback Risk Type", "Feedback Risk Type is missing")
        elif risk_id is None:
            add_error("Feedback Risk Type", f"Risk Type '{risk_name}' not found — reject")

        building_name = _get(row, COL_BUILDING)
        building_id = _lookup(maps, "buildings", building_name)
        if not building_name:
            add_error("Building", "Building is missing")
        elif building_id is None:
            add_error("Building", f"Building '{building_name}' not found — reject")

        # --- Is Inpatient ---
        inpatient_val = _get(row, COL_INPATIENT)
        is_inpatient = str(inpatient_val).strip().lower() in ("yes", "نعم", "1", "true") if inpatient_val else False

        # --- Target Department (mandatory) ---
        # One column, not three: a row represents a case, not an incident,
        # and a case has exactly one target department.
        target_dept_ids: List[int] = []
        dept_name = _get(row, COL_TARGET_DEPT)
        if not dept_name:
            add_error("Target Dept", "Target Dept is missing")
        else:
            dept_id = _lookup(maps, "org_units", dept_name)
            if dept_id is None:
                add_error("Target Dept", f"Target Department '{dept_name}' not found — reject")
            else:
                target_dept_ids.append(dept_id)

        # --- Doctors: all three slots optional -- warn, don't block, if
        # given but not found; silent if blank. ---
        doctor_ids: List[Tuple[int, str]] = []
        for doc_col in (COL_DOCTOR1, COL_DOCTOR2, COL_DOCTOR3):
            doc_name = _get(row, doc_col)
            if doc_name:
                doc_id = _lookup(maps, "doctors", doc_name)
                if doc_id is None:
                    warnings.append({"group_key": group_key, "row_number": row_number, "message": f"Doctor '{doc_name}' not found — imported without doctor linkage"})
                else:
                    doctor_ids.append((doc_id, doc_name))

        # --- Workers: same as doctors, all three slots optional. ---
        worker_ids: List[Tuple[Any, str]] = []
        for wrk_col in (COL_WORKER1, COL_WORKER2, COL_WORKER3):
            wrk_name = _get(row, wrk_col)
            if wrk_name:
                wrk_id = _lookup(maps, "workers", wrk_name)
                if wrk_id is None:
                    warnings.append({"group_key": group_key, "row_number": row_number, "message": f"Worker '{wrk_name}' not found — skipped"})
                else:
                    worker_ids.append((wrk_id, wrk_name))

        row_data = None
        if not errors:
            row_data = {
                "patient_name": pname,
                "feedback_date": parsed_date,
                "incident_date": parsed_incident_date,
                "record_type_id": record_type_id,
                "feedback_intent_id": _feedback_intent_type_id_for_record_type(record_type_id),
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
            }

        row_results.append({
            "row_number": row_number,
            "errors": errors,
            "data": row_data,
            "raw": {k: v for k, v in row.items() if k != ROW_NUMBER_KEY},
            "derived_hierarchy": derived_hierarchy,
        })

    # --- Patient ambiguity check (group-level, once per group) ---
    # Only worth the directory-search round trip if every row is otherwise
    # clean -- no point checking a group that's already going to be
    # rejected for unrelated reasons. If ambiguous, every row's blocked on
    # it, since it's the shared patient the whole group was validated against.
    is_new_patient = False
    if patient_name and all(not r["errors"] for r in row_results):
        match_count = _count_patient_matches(patient_name)
        if match_count > 1:
            ambiguous_msg = {"field": "Patient Name", "message": f"Patient '{patient_name}' matches {match_count} records — ambiguous, cannot import"}
            for r in row_results:
                r["errors"].append(ambiguous_msg)
                r["data"] = None
        is_new_patient = (match_count == 0)

    return {
        "group_key": group_key,
        "valid": all(not r["errors"] for r in row_results),
        "rows": row_results,
        "warnings": warnings,
        "patient_name": patient_name,
        "is_new_patient": is_new_patient,
    }


# ============================================================
# INTERNAL — IMPORT
# ============================================================

def _build_case_service_payload(vrow: Dict, patient_name: str) -> Dict[str, Any]:
    """
    Translate an import_service validated-row dict into the data shape
    case_service.create_case() expects. Incident Date and Received Date are
    now distinct template columns (both mandatory, validated against each
    other in _validate_group). requires_explanation defaults to False — the
    FSM's own red-flag/never-event check, driven by the template's optional
    Risk Type column, is what actually opens the explanation workflow for
    imported rows that need it (moot anyway under save_mode='import_closed',
    which keeps every imported case Closed regardless).
    """
    return {
        "record_type_id": vrow.get("record_type_id", DEFAULT_RECORD_TYPE_ID),
        "patient_name": patient_name,
        "feedback_received_date": vrow["feedback_date"],
        "incident_date": vrow["incident_date"],
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
            {"employee_id": emp_id, "full_name": emp_name}
            for emp_id, emp_name in vrow.get("worker_ids", [])
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
            result = create_case(data, context='BulkImport', save_mode='import_closed')

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

def _generate_import_report_excel(
    imported_with_rows: List[Tuple[Dict[str, Any], List[Dict]]],
    rejected_groups: List[Dict[str, Any]],
) -> BytesIO:
    """
    Full receipt of a confirmed batch: every original row, green if imported
    (with its new system Incident Number) or red/blue if rejected (with the
    reason) -- matches the original ask (green=imported+new ID, red=failure
    +reason), extended with a distinct color for duplicates so that reason
    isn't confused with a plain validation failure.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Import Report"
    ws.sheet_view.rightToLeft = True

    headers = [col[0] for col in TEMPLATE_COLUMNS] + ["Result", "System ID / Reason"]
    for col_i, h in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col_i, value=h)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E79")

    green_fill = PatternFill("solid", fgColor="C6EFCE")
    red_fill = PatternFill("solid", fgColor="FFC7CE")
    duplicate_fill = PatternFill("solid", fgColor="BDD7EE")

    row_i = 2
    for imp_result, group_rows in imported_with_rows:
        incident_number = f"INC-{imp_result['incident_id']:06d}"
        for row_data in group_rows:
            for col_i, (header, _, _, _) in enumerate(TEMPLATE_COLUMNS, start=1):
                cell = ws.cell(row=row_i, column=col_i, value=row_data.get(header))
                cell.fill = green_fill
            ws.cell(row=row_i, column=len(TEMPLATE_COLUMNS) + 1, value="Imported").fill = green_fill
            ws.cell(row=row_i, column=len(TEMPLATE_COLUMNS) + 2, value=incident_number).fill = green_fill
            row_i += 1

    for group in rejected_groups:
        fill = duplicate_fill if group.get("status") == "duplicate" else red_fill
        result_label = "Duplicate" if group.get("status") == "duplicate" else "Rejected"
        for row_data in group["rows"]:
            for col_i, (header, _, _, _) in enumerate(TEMPLATE_COLUMNS, start=1):
                cell = ws.cell(row=row_i, column=col_i, value=row_data.get(header))
                cell.fill = fill
            ws.cell(row=row_i, column=len(TEMPLATE_COLUMNS) + 1, value=result_label).fill = fill
            ws.cell(row=row_i, column=len(TEMPLATE_COLUMNS) + 2, value=group["reason"]).fill = fill
            row_i += 1

    for col_i in range(1, len(headers) + 1):
        ws.column_dimensions[get_column_letter(col_i)].width = 20
    ws.freeze_panes = "A2"

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


def list_import_batches(limit: int = 50) -> List[Dict[str, Any]]:
    """Batch history for the review UI — mirrors the ML Training Run Artifacts table layout."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        batches = ml_import_batch_db.list_batches(cursor, limit=limit)
    finally:
        cursor.close()
        conn.close()

    for b in batches:
        for key in ("UploadedAt", "CompletedAt"):
            if b.get(key) is not None:
                b[key] = b[key].isoformat()
    return batches
