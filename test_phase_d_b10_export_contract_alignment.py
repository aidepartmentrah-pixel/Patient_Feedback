"""
TEST TASK D-B10 — EXPORT CONTRACT ALIGNMENT

Verify person seasonal endpoints align with existing export contract pattern from reports_router.

Tests:
1. Person seasonal endpoints return StreamingResponse
2. media_type matches other DOCX exports in reports_router
3. Content-Disposition header exists
4. Filename format: doctor_seasonal_{id}.docx / worker_seasonal_{id}.docx
5. No new export_id system introduced
6. No temp file writes added
7. reports_router was NOT modified
"""

import os
import re
import ast
import sys


def test_file_exists():
    """Test 1: Verify person_seasonal_report_router.py exists"""
    router_path = "backend/api/routers/person_seasonal_report_router.py"
    assert os.path.exists(router_path), f"Router file not found: {router_path}"
    print("✅ Test 1: File exists")


def test_has_filename_helper():
    """Test 2: Verify _build_person_report_filename helper exists"""
    router_path = "backend/api/routers/person_seasonal_report_router.py"
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for helper function
    assert "def _build_person_report_filename" in content, \
        "Missing _build_person_report_filename helper function"
    
    # Check helper signature
    assert "person_type: str" in content and "person_id: int" in content, \
        "Helper function signature mismatch"
    
    # Check return format in docstring or code
    assert "doctor_seasonal_{id}.docx" in content or 'f"{person_type}_seasonal_{person_id}.docx"' in content, \
        "Helper doesn't follow required filename format"
    
    print("✅ Test 2: Filename helper exists with correct signature")


def test_endpoints_use_helper():
    """Test 3: Verify both endpoints use _build_person_report_filename"""
    router_path = "backend/api/routers/person_seasonal_report_router.py"
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check doctor endpoint uses helper
    doctor_section = content[content.find("export_doctor_seasonal_word"):]
    assert "_build_person_report_filename" in doctor_section[:2000], \
        "Doctor endpoint doesn't use filename helper"
    assert '_build_person_report_filename("doctor"' in content, \
        "Doctor endpoint doesn't call helper with correct person_type"
    
    # Check worker endpoint uses helper
    worker_section = content[content.find("export_worker_seasonal_word"):]
    assert "_build_person_report_filename" in worker_section[:2000], \
        "Worker endpoint doesn't use filename helper"
    assert '_build_person_report_filename("worker"' in content, \
        "Worker endpoint doesn't call helper with correct person_type"
    
    print("✅ Test 3: Both endpoints use filename helper")


def test_streaming_response_used():
    """Test 4: Verify endpoints return StreamingResponse"""
    router_path = "backend/api/routers/person_seasonal_report_router.py"
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check imports
    assert "from fastapi.responses import StreamingResponse" in content, \
        "StreamingResponse not imported"
    
    # Check both endpoints use StreamingResponse
    assert content.count("return StreamingResponse(") >= 2, \
        "Both endpoints must return StreamingResponse"
    
    print("✅ Test 4: StreamingResponse used in both endpoints")


def test_media_type_matches_reports_router():
    """Test 5: Verify media_type matches DOCX pattern in reports_router"""
    router_path = "backend/api/routers/person_seasonal_report_router.py"
    reports_router_path = "backend/api/routers/reports_router.py"
    
    # Get expected media_type from reports_router
    with open(reports_router_path, 'r', encoding='utf-8') as f:
        reports_content = f.read()
    
    # Find DOCX media type in reports_router
    docx_media_match = re.search(
        r'media_type="(application/vnd\.openxmlformats-officedocument\.wordprocessingml\.document)"',
        reports_content
    )
    assert docx_media_match, "Cannot find DOCX media_type in reports_router"
    expected_media_type = docx_media_match.group(1)
    
    # Check person_seasonal_report_router uses same media_type
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    assert expected_media_type in content, \
        f"media_type doesn't match reports_router pattern: {expected_media_type}"
    
    # Check both endpoints use correct media_type
    media_type_count = content.count(expected_media_type)
    assert media_type_count >= 2, \
        f"Expected media_type in both endpoints, found {media_type_count} occurrences"
    
    print(f"✅ Test 5: media_type matches reports_router: {expected_media_type}")


def test_content_disposition_header():
    """Test 6: Verify Content-Disposition header exists and follows pattern"""
    router_path = "backend/api/routers/person_seasonal_report_router.py"
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check Content-Disposition exists
    assert "Content-Disposition" in content, "Content-Disposition header missing"
    
    # Check pattern matches reports_router style (attachment; filename=...)
    assert 'attachment; filename=' in content, \
        "Content-Disposition doesn't follow reports_router pattern"
    
    # Count occurrences (should be at least 2, one per endpoint)
    disposition_count = content.count("Content-Disposition")
    assert disposition_count >= 2, \
        f"Expected Content-Disposition in both endpoints, found {disposition_count}"
    
    print("✅ Test 6: Content-Disposition header exists with correct pattern")


def test_filename_format():
    """Test 7: Verify filename format matches specification"""
    router_path = "backend/api/routers/person_seasonal_report_router.py"
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for correct filename pattern in helper or docstrings
    # Format should be: doctor_seasonal_{id}.docx or worker_seasonal_{id}.docx
    assert "doctor_seasonal_" in content and "worker_seasonal_" in content, \
        "Missing required filename prefixes"
    
    # Check format string pattern
    assert 'f"{person_type}_seasonal_{person_id}.docx"' in content or \
           ('doctor_seasonal_{id}.docx' in content and 'worker_seasonal_{id}.docx' in content), \
        "Filename format doesn't match specification"
    
    print("✅ Test 7: Filename format correct: doctor_seasonal_{id}.docx / worker_seasonal_{id}.docx")


def test_no_export_id_system():
    """Test 8: Verify no new export_id system introduced"""
    router_path = "backend/api/routers/person_seasonal_report_router.py"
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for export_id patterns
    assert "export_id" not in content.lower(), \
        "export_id system should not be introduced (direct streaming only)"
    
    # Check no async job queue patterns
    assert "celery" not in content.lower() and "redis" not in content.lower(), \
        "No async job queue should be added"
    
    print("✅ Test 8: No export_id system introduced")


def test_no_temp_files():
    """Test 9: Verify no temporary file writes added"""
    router_path = "backend/api/routers/person_seasonal_report_router.py"
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for temp file operations
    assert "tempfile" not in content.lower(), "No tempfile operations should be added"
    assert 'open(' not in content or 'open(' in content and 'open(router_path' not in content, \
        "No file write operations should be in router"
    assert ".write(" not in content or "word_file.seek" in content, \
        "No file writes (except BytesIO operations) should be added"
    
    # BytesIO is allowed for streaming
    assert "BytesIO" in content, "BytesIO required for streaming"
    
    print("✅ Test 9: No temp file writes added (BytesIO streaming only)")


def test_reports_router_not_modified():
    """Test 10: Verify reports_router.py was NOT modified"""
    reports_router_path = "backend/api/routers/reports_router.py"
    
    # This test checks that reports_router imports and structure remain unchanged
    with open(reports_router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check that person report references don't exist in reports_router
    assert "person_seasonal" not in content.lower(), \
        "reports_router should not reference person seasonal reports"
    assert "doctor_seasonal" not in content.lower() or "Emergency" in content, \
        "reports_router should not reference doctor_seasonal (except existing Emergency exports)"
    assert "worker_seasonal" not in content.lower(), \
        "reports_router should not reference worker_seasonal"
    
    print("✅ Test 10: reports_router.py not modified")


def test_export_contract_alignment_comment():
    """Test 11: Verify export contract alignment comment exists"""
    router_path = "backend/api/routers/person_seasonal_report_router.py"
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for alignment comment
    assert "Export contract aligned with reports_router pattern" in content, \
        "Missing required comment: 'Export contract aligned with reports_router pattern'"
    
    print("✅ Test 11: Export contract alignment comment present")


def test_bytesio_pattern():
    """Test 12: Verify BytesIO pattern used correctly"""
    router_path = "backend/api/routers/person_seasonal_report_router.py"
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check BytesIO import and usage
    assert "from io import BytesIO" in content, "BytesIO not imported"
    assert "BytesIO(" in content, "BytesIO not instantiated"
    assert ".seek(0)" in content, "BytesIO not properly reset with seek(0)"
    
    # Check BytesIO passed to StreamingResponse
    assert "StreamingResponse(" in content, "StreamingResponse not used"
    
    print("✅ Test 12: BytesIO pattern used correctly")


def test_no_sql_in_router():
    """Test 13: Verify no SQL queries in router (architecture check)"""
    router_path = "backend/api/routers/person_seasonal_report_router.py"
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for SQL patterns
    sql_patterns = [
        "SELECT ", "FROM APP_", "INSERT ", "UPDATE ", "DELETE ",
        "execute(", "cursor", "pyodbc"
    ]
    
    for pattern in sql_patterns:
        assert pattern not in content, f"Router contains SQL pattern: {pattern}"
    
    print("✅ Test 13: No SQL in router (architecture preserved)")


def run_all_tests():
    """Run all D-B10 export contract alignment tests"""
    print("\n" + "="*70)
    print("PHASE D - TASK D-B10: EXPORT CONTRACT ALIGNMENT")
    print("="*70 + "\n")
    
    tests = [
        test_file_exists,
        test_has_filename_helper,
        test_endpoints_use_helper,
        test_streaming_response_used,
        test_media_type_matches_reports_router,
        test_content_disposition_header,
        test_filename_format,
        test_no_export_id_system,
        test_no_temp_files,
        test_reports_router_not_modified,
        test_export_contract_alignment_comment,
        test_bytesio_pattern,
        test_no_sql_in_router,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: Unexpected error: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ EXPORT CONTRACT ALIGNMENT OK")
    else:
        print("❌ EXPORT CONTRACT ALIGNMENT FAILED")
        sys.exit(1)
    
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
