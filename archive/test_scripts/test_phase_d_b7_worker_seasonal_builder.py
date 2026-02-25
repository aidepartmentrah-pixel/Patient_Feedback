"""
TEST TASK D-B7 — WORKER SEASONAL REPORT BUILDER

Verify:
1. File exists: worker_seasonal_reporting_service.py
2. Function exists: build_worker_seasonal_report_data
3. Calls: worker_reporting_service.get_worker_profile
4. Calls: performance_scoring.compute_performance_score
5. No SQL in file. No DB imports.
6. Return dict contains: worker_identity, period, metrics, performance
7. No Word/docx imports.

Report: WORKER SEASONAL BUILDER OK or list issues.
"""

import pytest
import os
from datetime import date


class TestWorkerSeasonalBuilder:
    """Test suite for D-B7 Worker Seasonal Report Builder"""
    
    def test_file_exists(self):
        """1. Verify file exists: worker_seasonal_reporting_service.py"""
        file_path = "backend/api/services/worker_seasonal_reporting_service.py"
        assert os.path.exists(file_path), f"File not found: {file_path}"
    
    def test_function_exists(self):
        """2. Verify function exists: build_worker_seasonal_report_data"""
        from backend.api.services.worker_seasonal_reporting_service import WorkerSeasonalReportingService
        
        assert hasattr(WorkerSeasonalReportingService, 'build_worker_seasonal_report_data'), \
            "Function build_worker_seasonal_report_data does not exist"
        
        # Verify it's callable
        assert callable(WorkerSeasonalReportingService.build_worker_seasonal_report_data), \
            "build_worker_seasonal_report_data is not callable"
    
    def test_calls_worker_reporting_service(self):
        """3. Verify calls worker_reporting_service.get_worker_profile"""
        import inspect
        from backend.api.services.worker_seasonal_reporting_service import WorkerSeasonalReportingService
        
        # Get source code of the function
        source = inspect.getsource(WorkerSeasonalReportingService.build_worker_seasonal_report_data)
        
        # Check that it uses WorkerReportingService
        assert 'WorkerReportingService' in source, "Function does not use WorkerReportingService"
        assert 'get_worker_profile' in source, "Function does not call get_worker_profile"
    
    def test_calls_performance_scoring(self):
        """4. Verify calls performance_scoring.compute_performance_score"""
        import inspect
        from backend.api.services.worker_seasonal_reporting_service import WorkerSeasonalReportingService
        
        # Get source code of the function
        source = inspect.getsource(WorkerSeasonalReportingService.build_worker_seasonal_report_data)
        
        # Check that it calls compute_performance_score
        assert 'compute_performance_score' in source, \
            "Function does not call compute_performance_score"
    
    def test_no_sql_no_db_imports(self):
        """5. Verify no SQL in file, no DB imports"""
        import backend.api.services.worker_seasonal_reporting_service as module
        import inspect
        
        # Get full module source
        source = inspect.getsource(module)
        
        # Should NOT have SQL queries (check actual SQL patterns, not just keywords in comments)
        sql_patterns = ['SELECT *', 'SELECT ', 'FROM APP_', 'JOIN APP_', 'INSERT INTO', 'UPDATE APP_', 'DELETE FROM']
        for pattern in sql_patterns:
            assert pattern not in source, f"Should not contain SQL pattern: {pattern}"
        
        # Should NOT import DB layer
        assert 'worker_reporting_db' not in source, "Should not import worker_reporting_db"
        assert 'from ..db_layer' not in source, "Should not import from db_layer"
        assert 'import pyodbc' not in source, "Should not import pyodbc"
    
    def test_returns_correct_structure(self):
        """6. Verify return dict contains required keys"""
        from backend.api.services.worker_seasonal_reporting_service import WorkerSeasonalReportingService
        
        # Call with real data (using existing employee ID 1)
        try:
            result = WorkerSeasonalReportingService.build_worker_seasonal_report_data(
                employee_id=1,
                season_start=date(2025, 1, 1),
                season_end=date(2025, 12, 31)
            )
            
            # Check it returns a dict
            assert isinstance(result, dict), "Function must return a dict"
            
            # Check required top-level keys
            assert 'worker_identity' in result, "Missing key: worker_identity"
            assert 'period' in result, "Missing key: period"
            assert 'metrics' in result, "Missing key: metrics"
            assert 'performance' in result, "Missing key: performance"
            
            # Check worker_identity structure
            identity = result['worker_identity']
            assert 'employee_id' in identity, "worker_identity missing 'employee_id'"
            assert 'full_name' in identity, "worker_identity missing 'full_name'"
            assert 'job_title' in identity, "worker_identity missing 'job_title'"
            assert 'department_id' in identity, "worker_identity missing 'department_id'"
            assert 'section_id' in identity, "worker_identity missing 'section_id'"
            assert 'administration_id' in identity, "worker_identity missing 'administration_id'"
            
            # Check period structure
            assert 'start' in result['period'], "period missing 'start'"
            assert 'end' in result['period'], "period missing 'end'"
            
            # Check metrics structure
            metrics = result['metrics']
            assert 'total_incidents' in metrics, "metrics missing 'total_incidents'"
            assert 'total_action_items' in metrics, "metrics missing 'total_action_items'"
            assert 'completed_action_items' in metrics, "metrics missing 'completed_action_items'"
            assert 'overdue_action_items' in metrics, "metrics missing 'overdue_action_items'"
            assert 'explanation_accepted_count' in metrics, "metrics missing 'explanation_accepted_count'"
            assert 'explanation_rejected_count' in metrics, "metrics missing 'explanation_rejected_count'"
            
            # Check performance structure
            performance = result['performance']
            assert 'score' in performance, "performance missing 'score'"
            assert 'praise_level' in performance, "performance missing 'praise_level'"
            assert 'risk_level' in performance, "performance missing 'risk_level'"
            assert 'flags' in performance, "performance missing 'flags'"
            
        except Exception as e:
            pytest.fail(f"Function call failed: {str(e)}")
    
    def test_no_word_imports(self):
        """7. Verify no Word/docx imports"""
        import backend.api.services.worker_seasonal_reporting_service as module
        import inspect
        
        # Check module-level imports
        import_names = [name for name in dir(module) if not name.startswith('_')]
        
        # Should NOT have docx-related imports
        docx_related = ['Document', 'docx', 'Mm', 'Pt', 'RGBColor', 'Inches']
        for name in docx_related:
            assert name not in import_names, f"Should not import {name} yet (Word generation comes later)"
        
        # Check source code
        source = inspect.getsource(module)
        assert 'from docx import' not in source, "Should not import from docx module"
        assert 'import docx' not in source, "Should not import docx module"
    
    def test_function_signature(self):
        """Verify function signature matches specification"""
        from backend.api.services.worker_seasonal_reporting_service import WorkerSeasonalReportingService
        import inspect
        
        sig = inspect.signature(WorkerSeasonalReportingService.build_worker_seasonal_report_data)
        params = list(sig.parameters.keys())
        
        # Check parameters (excluding 'self' if present)
        expected_params = ['employee_id', 'season_start', 'season_end']
        actual_params = [p for p in params if p != 'self']
        
        assert actual_params == expected_params, \
            f"Expected params {expected_params}, got {actual_params}"
    
    def test_no_router_imports(self):
        """Verify no router imports"""
        import backend.api.services.worker_seasonal_reporting_service as module
        import inspect
        
        # Check source code
        source = inspect.getsource(module)
        
        # Should NOT have router imports
        assert 'APIRouter' not in source, "Should not import APIRouter"
        assert 'from ..routers' not in source, "Should not import from routers"
        assert 'import routers' not in source, "Should not import routers"


if __name__ == "__main__":
    print("=" * 70)
    print("TEST TASK D-B7 — WORKER SEASONAL REPORT BUILDER")
    print("=" * 70)
    print()
    
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    
    print()
    print("=" * 70)
    if exit_code == 0:
        print("✅ WORKER SEASONAL BUILDER OK")
    else:
        print("❌ ISSUES FOUND - See output above")
    print("=" * 70)
    
    exit(exit_code)
