"""
Smoke Test — Workflow Activity Report

Verifies:
1. DB layer import is clean (no syntax errors)
2. Service layer import is clean
3. Word formatter import is clean
4. Formatter produces a non-empty bytes blob for an empty dataset (no DB needed)
5. (Optional) Full DB round-trip if SMOKE_DB=1 env var is set

Run:
    cd backend
    python -m pytest tests/test_smoke_workflow_activity.py -v

Or without pytest:
    python tests/test_smoke_workflow_activity.py
"""

import sys
import os
import importlib
from datetime import date

# ---------------------------------------------------------------------------
# PATH SETUP — allow running from backend/ directory
# ---------------------------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


# ---------------------------------------------------------------------------
# TEST 1 — Import DB layer (no DB connection made)
# ---------------------------------------------------------------------------
def test_import_db_layer():
    mod = importlib.import_module("api.db_layer.workflow_activity_db")
    assert hasattr(mod, "get_workflow_activity_cases"), "Missing get_workflow_activity_cases"
    assert hasattr(mod, "expand_scope_to_unit_ids"), "Missing expand_scope_to_unit_ids"
    print("[OK] DB layer imports cleanly")


# ---------------------------------------------------------------------------
# TEST 2 — Import service layer
# ---------------------------------------------------------------------------
def test_import_service():
    mod = importlib.import_module("api.services.workflow_activity_report_service")
    assert hasattr(mod, "build_workflow_activity_report"), "Missing build_workflow_activity_report"
    print("[OK] Service layer imports cleanly")


# ---------------------------------------------------------------------------
# TEST 3 — Import Word formatter
# ---------------------------------------------------------------------------
def test_import_formatter():
    mod = importlib.import_module("api.services.workflow_activity_word_formatter")
    assert hasattr(mod, "generate_workflow_activity_word"), "Missing generate_workflow_activity_word"
    print("[OK] Word formatter imports cleanly")


# ---------------------------------------------------------------------------
# TEST 4 — Formatter with empty dataset (no DB required)
# ---------------------------------------------------------------------------
def test_formatter_empty_dataset():
    from api.services.workflow_activity_word_formatter import generate_workflow_activity_word

    empty_report = {
        "meta": {
            "start_date": date(2026, 1, 1),
            "end_date": date(2026, 3, 31),
            "scope": "hospital",
            "generated_at": date(2026, 5, 18),
            "generated_by": "Smoke Test",
            "total_cases": 0,
            "total_subcases": 0,
            "total_action_items": 0,
        },
        "cases": [],
    }

    result = generate_workflow_activity_word(empty_report)
    assert isinstance(result, bytes), "Expected bytes output"
    assert len(result) > 1000, f"Expected a real .docx blob, got {len(result)} bytes"
    # .docx files start with PK (ZIP magic bytes)
    assert result[:2] == b'PK', "Expected .docx (ZIP) magic bytes at start"
    print(f"[OK] Formatter produced {len(result):,} bytes for empty dataset")


# ---------------------------------------------------------------------------
# TEST 5 — Formatter with synthetic data (no DB required)
# ---------------------------------------------------------------------------
def test_formatter_synthetic_data():
    from api.services.workflow_activity_word_formatter import generate_workflow_activity_word

    synthetic_report = {
        "meta": {
            "start_date": date(2026, 1, 1),
            "end_date": date(2026, 3, 31),
            "scope": "section",
            "generated_at": date(2026, 5, 18),
            "generated_by": "Smoke Test",
            "total_cases": 1,
            "total_subcases": 1,
            "total_action_items": 2,
        },
        "cases": [
            {
                "case_id": 999,
                "patient_name": "مريض تجريبي",
                "feedback_date": date(2026, 2, 15),
                "complaint_text": "نص شكوى تجريبية لغرض اختبار الدخان",
                "subcases": [
                    {
                        "subcase_id": 888,
                        "status": "ADMIN_APPROVED",
                        "target_org_unit_id": 10,
                        "target_org_unit_name": "قسم تجريبي",
                        "section_explanation": "توضيح القسم التجريبي",
                        "department_explanation": None,
                        "administration_explanation": None,
                        "action_items": [
                            {
                                "action_item_id": 1,
                                "title": "إجراء تجريبي منجز",
                                "description": "وصف الإجراء",
                                "due_date": date(2026, 2, 28),
                                "status": "DONE",
                                "completed_at": date(2026, 2, 27),
                                "is_overdue": False,
                                "days_overdue": None,
                            },
                            {
                                "action_item_id": 2,
                                "title": "إجراء تجريبي متأخر",
                                "description": None,
                                "due_date": date(2026, 1, 15),
                                "status": "IN_PROGRESS",
                                "completed_at": None,
                                "is_overdue": True,
                                "days_overdue": 30,
                            },
                        ],
                    }
                ],
            }
        ],
    }

    result = generate_workflow_activity_word(synthetic_report)
    assert isinstance(result, bytes)
    assert result[:2] == b'PK'
    assert len(result) > 5000, f"Expected a larger .docx with data, got {len(result)} bytes"
    print(f"[OK] Formatter produced {len(result):,} bytes for synthetic dataset")


# ---------------------------------------------------------------------------
# TEST 6 — Full DB round-trip (only if SMOKE_DB=1)
# ---------------------------------------------------------------------------
def test_full_db_roundtrip():
    if os.environ.get("SMOKE_DB") != "1":
        print("[SKIP] Full DB round-trip skipped (set SMOKE_DB=1 to enable)")
        return

    from api.services.workflow_activity_report_service import build_workflow_activity_report
    from api.services.workflow_activity_word_formatter import generate_workflow_activity_word

    report_data = build_workflow_activity_report(
        start_date=date(2025, 1, 1),
        end_date=date(2026, 12, 31),
        scope="hospital",
        administration_ids=None,
        department_ids=None,
        section_ids=None,
        hospital_id=1,
        generated_by="Smoke Test DB",
    )

    total_cases = report_data["meta"]["total_cases"]
    print(f"[DB] Retrieved {total_cases} case(s) from the database")

    docx_bytes = generate_workflow_activity_word(report_data)
    assert docx_bytes[:2] == b'PK'
    print(f"[OK] Full DB round-trip produced {len(docx_bytes):,} bytes")


# ---------------------------------------------------------------------------
# RUNNER
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_import_db_layer,
        test_import_service,
        test_import_formatter,
        test_formatter_empty_dataset,
        test_formatter_synthetic_data,
        test_full_db_roundtrip,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {t.__name__}: {exc}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Smoke test result: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
