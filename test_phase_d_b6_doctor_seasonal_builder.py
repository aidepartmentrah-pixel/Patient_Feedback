"""
TEST TASK D-B6 — DOCTOR SEASONAL REPORT BUILDER

Verify:
1. File exists: doctor_seasonal_reporting_service.py
2. Function exists: build_doctor_seasonal_report_data
3. Uses doctors_service — not DB layer directly
4. Returns dict with keys: doctor_identity, period, metrics, incidents_summary
5. No Word/docx imports yet
6. No router imports

Report: DOCTOR SEASONAL BUILDER OK or list problems.
"""

import pytest
import os
from datetime import date
from collections import Counter


class TestDoctorSeasonalBuilder:
    """Test suite for D-B6 Doctor Seasonal Report Builder"""
    
    def test_file_exists(self):
        """1. Verify file exists: doctor_seasonal_reporting_service.py"""
        file_path = "backend/api/services/doctor_seasonal_reporting_service.py"
        assert os.path.exists(file_path), f"File not found: {file_path}"
    
    def test_function_exists(self):
        """2. Verify function exists: build_doctor_seasonal_report_data"""
        from backend.api.services.doctor_seasonal_reporting_service import DoctorSeasonalReportingService
        
        assert hasattr(DoctorSeasonalReportingService, 'build_doctor_seasonal_report_data'), \
            "Function build_doctor_seasonal_report_data does not exist"
        
        # Verify it's callable
        assert callable(DoctorSeasonalReportingService.build_doctor_seasonal_report_data), \
            "build_doctor_seasonal_report_data is not callable"
    
    def test_uses_doctors_service_not_db(self):
        """3. Verify uses doctors_service — not DB layer directly"""
        import inspect
        from backend.api.services.doctor_seasonal_reporting_service import DoctorSeasonalReportingService
        
        # Get source code of the function
        source = inspect.getsource(DoctorSeasonalReportingService.build_doctor_seasonal_report_data)
        
        # Check that it uses DoctorService
        assert 'DoctorService' in source, "Function does not use DoctorService"
        assert 'get_doctor_profile' in source, "Function does not call get_doctor_profile"
        assert 'get_doctor_statistics' in source, "Function does not call get_doctor_statistics"
        assert 'get_doctor_incidents' in source, "Function does not call get_doctor_incidents"
        
        # Check that it does NOT import or use DB layer directly
        assert 'doctors_db' not in source.lower(), "Function should not use DB layer directly"
        assert 'DoctorsDB' not in source, "Function should not use DoctorsDB class"
    
    def test_returns_correct_structure(self):
        """4. Verify returns dict with required keys"""
        from backend.api.services.doctor_seasonal_reporting_service import DoctorSeasonalReportingService
        
        # Call with real data (using existing doctor ID 1)
        try:
            result = DoctorSeasonalReportingService.build_doctor_seasonal_report_data(
                doctor_id=1,
                season_start=date(2025, 1, 1),
                season_end=date(2025, 12, 31)
            )
            
            # Check it returns a dict
            assert isinstance(result, dict), "Function must return a dict"
            
            # Check required top-level keys
            assert 'doctor_identity' in result, "Missing key: doctor_identity"
            assert 'period' in result, "Missing key: period"
            assert 'metrics' in result, "Missing key: metrics"
            assert 'incidents_summary' in result, "Missing key: incidents_summary"
            
            # Check doctor_identity structure
            assert 'id' in result['doctor_identity'], "doctor_identity missing 'id'"
            assert 'name' in result['doctor_identity'], "doctor_identity missing 'name'"
            assert 'specialty' in result['doctor_identity'], "doctor_identity missing 'specialty'"
            
            # Check period structure
            assert 'start' in result['period'], "period missing 'start'"
            assert 'end' in result['period'], "period missing 'end'"
            
            # Check metrics structure
            assert 'total_incidents' in result['metrics'], "metrics missing 'total_incidents'"
            assert 'high_severity' in result['metrics'], "metrics missing 'high_severity'"
            assert 'medium_severity' in result['metrics'], "metrics missing 'medium_severity'"
            assert 'low_severity' in result['metrics'], "metrics missing 'low_severity'"
            assert 'red_flags' in result['metrics'], "metrics missing 'red_flags'"
            
            # Check incidents_summary structure
            assert 'count' in result['incidents_summary'], "incidents_summary missing 'count'"
            assert 'top_categories' in result['incidents_summary'], "incidents_summary missing 'top_categories'"
            
            # Verify top_categories is a list (top 5)
            assert isinstance(result['incidents_summary']['top_categories'], list), \
                "top_categories must be a list"
            
        except Exception as e:
            pytest.fail(f"Function call failed: {str(e)}")
    
    def test_no_word_imports(self):
        """5. Verify no Word/docx imports yet"""
        import backend.api.services.doctor_seasonal_reporting_service as module
        
        # Check module-level imports
        import_names = [name for name in dir(module) if not name.startswith('_')]
        
        # Should NOT have docx-related imports
        docx_related = ['Document', 'docx', 'Mm', 'Pt', 'RGBColor', 'Inches']
        for name in docx_related:
            assert name not in import_names, f"Should not import {name} yet (Word generation comes later)"
        
        # Check source code
        import inspect
        source = inspect.getsource(module)
        assert 'from docx import' not in source, "Should not import from docx module"
        assert 'import docx' not in source, "Should not import docx module"
    
    def test_no_router_imports(self):
        """6. Verify no router imports"""
        import backend.api.services.doctor_seasonal_reporting_service as module
        import inspect
        
        # Check source code
        source = inspect.getsource(module)
        
        # Should NOT have router imports
        assert 'APIRouter' not in source, "Should not import APIRouter"
        assert 'from ..routers' not in source, "Should not import from routers"
        assert 'import routers' not in source, "Should not import routers"
    
    def test_computes_top_categories(self):
        """Verify top_categories are computed from incidents list"""
        from backend.api.services.doctor_seasonal_reporting_service import DoctorSeasonalReportingService
        import inspect
        
        # Check source code contains Counter logic
        source = inspect.getsource(DoctorSeasonalReportingService.build_doctor_seasonal_report_data)
        
        # Should use Counter to compute top categories
        assert 'Counter' in source, "Should use Counter to compute top categories"
        assert 'most_common' in source, "Should use most_common() to get top 5"
    
    def test_function_signature(self):
        """Verify function signature matches specification"""
        from backend.api.services.doctor_seasonal_reporting_service import DoctorSeasonalReportingService
        import inspect
        
        sig = inspect.signature(DoctorSeasonalReportingService.build_doctor_seasonal_report_data)
        params = list(sig.parameters.keys())
        
        # Check parameters (excluding 'self' if present)
        expected_params = ['doctor_id', 'season_start', 'season_end']
        actual_params = [p for p in params if p != 'self']
        
        assert actual_params == expected_params, \
            f"Expected params {expected_params}, got {actual_params}"


if __name__ == "__main__":
    print("=" * 70)
    print("TEST TASK D-B6 — DOCTOR SEASONAL REPORT BUILDER")
    print("=" * 70)
    print()
    
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    
    print()
    print("=" * 70)
    if exit_code == 0:
        print("✅ DOCTOR SEASONAL BUILDER OK")
    else:
        print("❌ PROBLEMS FOUND - See output above")
    print("=" * 70)
    
    exit(exit_code)
