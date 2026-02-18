"""
TEST TASK D-B9 — SEASONAL PERSON REPORT ENDPOINT

Verify:
1. File exists: person_seasonal_report_router.py
2. Endpoints exist: 
   GET /api/person-reports/doctor/{id}/seasonal-word
   GET /api/person-reports/worker/{id}/seasonal-word
3. Uses: Depends(get_current_user)
4. Calls: doctor_seasonal_reporting_service, worker_seasonal_reporting_service, person_report_word_adapter
5. Returns StreamingResponse with docx content type
6. No SQL in router
7. Correct filename headers set

Report: PERSON SEASONAL ENDPOINT OK or list issues.
"""

import pytest
import os


class TestPersonSeasonalReportEndpoint:
    """Test suite for D-B9 Seasonal Person Report Endpoint"""
    
    def test_file_exists(self):
        """1. Verify file exists: person_seasonal_report_router.py"""
        file_path = "backend/api/routers/person_seasonal_report_router.py"
        assert os.path.exists(file_path), f"File not found: {file_path}"
    
    def test_endpoints_exist(self):
        """2. Verify endpoints exist"""
        from backend.api.routers.person_seasonal_report_router import router
        
        # Get all routes
        routes = [route.path for route in router.routes]
        
        # Check doctor endpoint exists
        assert any('/doctor/{doctor_id}/seasonal-word' in route for route in routes), \
            "Missing endpoint: /doctor/{doctor_id}/seasonal-word"
        
        # Check worker endpoint exists
        assert any('/worker/{employee_id}/seasonal-word' in route for route in routes), \
            "Missing endpoint: /worker/{employee_id}/seasonal-word"
        
        # Verify they are GET methods
        get_methods = [route.methods for route in router.routes if hasattr(route, 'methods')]
        assert any('GET' in methods for methods in get_methods), \
            "Endpoints should be GET methods"
    
    def test_uses_get_current_user_dependency(self):
        """3. Verify uses Depends(get_current_user)"""
        import backend.api.routers.person_seasonal_report_router as module
        import inspect
        
        # Get source code
        source = inspect.getsource(module)
        
        # Check that it uses get_current_user dependency
        assert 'get_current_user' in source, \
            "Must use get_current_user dependency"
        assert 'Depends(get_current_user)' in source, \
            "Must use Depends(get_current_user) pattern"
    
    def test_calls_required_services(self):
        """4. Verify calls required services"""
        import backend.api.routers.person_seasonal_report_router as module
        import inspect
        
        # Get source code
        source = inspect.getsource(module)
        
        # Check doctor seasonal reporting service
        assert 'DoctorSeasonalReportingService' in source, \
            "Must import DoctorSeasonalReportingService"
        assert 'build_doctor_seasonal_report_data' in source, \
            "Must call build_doctor_seasonal_report_data"
        
        # Check worker seasonal reporting service
        assert 'WorkerSeasonalReportingService' in source, \
            "Must import WorkerSeasonalReportingService"
        assert 'build_worker_seasonal_report_data' in source, \
            "Must call build_worker_seasonal_report_data"
        
        # Check word adapter
        assert 'generate_person_seasonal_word_report' in source, \
            "Must import generate_person_seasonal_word_report"
        assert 'person_type="doctor"' in source or "person_type='doctor'" in source, \
            "Must call adapter with person_type='doctor'"
        assert 'person_type="worker"' in source or "person_type='worker'" in source, \
            "Must call adapter with person_type='worker'"
    
    def test_returns_streaming_response_docx(self):
        """5. Verify returns StreamingResponse with docx content type"""
        import backend.api.routers.person_seasonal_report_router as module
        import inspect
        
        # Get source code
        source = inspect.getsource(module)
        
        # Check StreamingResponse is imported and used
        assert 'StreamingResponse' in source, \
            "Must import StreamingResponse"
        assert 'return StreamingResponse' in source, \
            "Must return StreamingResponse"
        
        # Check docx media type is used
        docx_media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        assert docx_media_type in source, \
            f"Must use docx media type: {docx_media_type}"
    
    def test_no_sql_in_router(self):
        """6. Verify no SQL in router"""
        import backend.api.routers.person_seasonal_report_router as module
        import inspect
        
        # Get source code
        source = inspect.getsource(module)
        
        # Check for SQL patterns (actual SQL, not just keywords in comments)
        sql_patterns = ['SELECT *', 'SELECT ', 'FROM APP_', 'JOIN APP_', 'INSERT INTO', 'UPDATE APP_', 'DELETE FROM']
        for pattern in sql_patterns:
            assert pattern not in source, \
                f"Router should not contain SQL pattern: {pattern}"
        
        # Check no pyodbc import
        assert 'import pyodbc' not in source, \
            "Router should not import pyodbc"
    
    def test_correct_filename_headers(self):
        """7. Verify correct filename headers set"""
        import backend.api.routers.person_seasonal_report_router as module
        import inspect
        
        # Get source code
        source = inspect.getsource(module)
        
        # Check doctor filename pattern
        assert 'doctor_' in source and '_seasonal.docx' in source, \
            "Doctor endpoint must set filename: doctor_{id}_seasonal.docx"
        
        # Check worker filename pattern
        assert 'worker_' in source and '_seasonal.docx' in source, \
            "Worker endpoint must set filename: worker_{id}_seasonal.docx"
        
        # Check Content-Disposition header is set
        assert 'Content-Disposition' in source, \
            "Must set Content-Disposition header"
        assert 'attachment' in source, \
            "Content-Disposition must include 'attachment'"
    
    def test_router_prefix_and_tags(self):
        """Verify router has correct prefix and tags"""
        from backend.api.routers.person_seasonal_report_router import router
        
        # Check prefix
        assert router.prefix == "/api/person-reports", \
            f"Expected prefix '/api/person-reports', got '{router.prefix}'"
        
        # Check tags
        assert "Person Reports" in router.tags or "person reports" in [t.lower() for t in router.tags], \
            f"Expected tag 'Person Reports', got {router.tags}"
    
    def test_uses_bytesio_pattern(self):
        """Verify uses BytesIO to wrap bytes for StreamingResponse"""
        import backend.api.routers.person_seasonal_report_router as module
        import inspect
        
        # Get source code
        source = inspect.getsource(module)
        
        # Check BytesIO is imported and used
        assert 'BytesIO' in source, \
            "Must import BytesIO"
        assert 'BytesIO(' in source, \
            "Must use BytesIO to wrap bytes"
    
    def test_router_registered_in_main(self):
        """Verify router is registered in main.py"""
        import os
        
        # Read main.py directly
        main_path = "backend/main.py"
        assert os.path.exists(main_path), "main.py not found"
        
        with open(main_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Check router is imported
        assert 'person_seasonal_report_router' in source, \
            "Router must be imported in main.py"
        
        # Check router is included
        assert 'include_router(person_seasonal_report_router)' in source, \
            "Router must be registered with app.include_router()"
    
    def test_query_params_configured(self):
        """Verify query parameters are properly configured"""
        import backend.api.routers.person_seasonal_report_router as module
        import inspect
        
        # Get source code
        source = inspect.getsource(module)
        
        # Check season_start and season_end are query parameters
        assert 'season_start' in source, \
            "Must have season_start parameter"
        assert 'season_end' in source, \
            "Must have season_end parameter"
        assert 'Query(' in source, \
            "Must use Query() for query parameters"
        assert 'date' in source, \
            "Parameters should be date type"
    
    def test_path_params_configured(self):
        """Verify path parameters are properly configured"""
        import backend.api.routers.person_seasonal_report_router as module
        import inspect
        
        # Get source code
        source = inspect.getsource(module)
        
        # Check doctor_id and employee_id are path parameters
        assert 'doctor_id' in source, \
            "Must have doctor_id path parameter"
        assert 'employee_id' in source, \
            "Must have employee_id path parameter"
        assert 'Path(' in source, \
            "Must use Path() for path parameters"


if __name__ == "__main__":
    print("=" * 70)
    print("TEST TASK D-B9 — SEASONAL PERSON REPORT ENDPOINT")
    print("=" * 70)
    print()
    
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    
    print()
    print("=" * 70)
    if exit_code == 0:
        print("✅ PERSON SEASONAL ENDPOINT OK")
    else:
        print("❌ ISSUES FOUND - See output above")
    print("=" * 70)
    
    exit(exit_code)
