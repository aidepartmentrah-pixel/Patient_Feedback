"""
TEST TASK D-B8 — WORD GENERATOR ADAPTER

Verify:
1. File exists: person_report_word_adapter.py
2. Function exists: generate_person_seasonal_word_report
3. Imports: seasonal_report_formatter
4. Does NOT duplicate formatter logic
5. Returns bytes
6. No DB imports. No router imports.
7. Arabic titles set based on person_type

Report: WORD ADAPTER OK or list problems.
"""

import pytest
import os
from datetime import date


class TestWordGeneratorAdapter:
    """Test suite for D-B8 Word Generator Reuse Adapter"""
    
    def test_file_exists(self):
        """1. Verify file exists: person_report_word_adapter.py"""
        file_path = "backend/api/services/person_report_word_adapter.py"
        assert os.path.exists(file_path), f"File not found: {file_path}"
    
    def test_function_exists(self):
        """2. Verify function exists: generate_person_seasonal_word_report"""
        from backend.api.services.person_report_word_adapter import generate_person_seasonal_word_report
        
        assert callable(generate_person_seasonal_word_report), \
            "generate_person_seasonal_word_report is not callable"
    
    def test_imports_seasonal_formatter(self):
        """3. Verify imports seasonal_report_formatter"""
        import backend.api.services.person_report_word_adapter as module
        import inspect
        
        # Get source code
        source = inspect.getsource(module)
        
        # Check that it imports seasonal_report_formatter
        assert 'seasonal_report_formatter' in source, \
            "Must import seasonal_report_formatter"
        assert 'generate_seasonal_word_report' in source, \
            "Must import generate_seasonal_word_report function"
    
    def test_not_duplicate_formatter_logic(self):
        """4. Verify does NOT duplicate formatter logic"""
        import backend.api.services.person_report_word_adapter as module
        import inspect
        
        # Get source code
        source = inspect.getsource(module)
        
        # Should NOT have Document creation (that's in the formatter)
        assert 'Document()' not in source, \
            "Should not create Document (use formatter instead)"
        
        # Should NOT have section setup
        assert 'section.page_height' not in source, \
            "Should not set page dimensions (use formatter)"
        assert 'WD_ORIENT' not in source, \
            "Should not set page orientation (use formatter)"
        
        # Should NOT have paragraph/table creation logic
        assert 'add_paragraph' not in source or source.count('add_paragraph') == 0, \
            "Should not add paragraphs directly (use formatter)"
        assert 'add_table' not in source or source.count('add_table') == 0, \
            "Should not add tables directly (use formatter)"
        
        # MUST call the formatter function
        assert 'generate_seasonal_word_report(' in source, \
            "Must call generate_seasonal_word_report function"
    
    def test_returns_bytes(self):
        """5. Verify returns bytes"""
        from backend.api.services.person_report_word_adapter import generate_person_seasonal_word_report
        
        # Create minimal test payload for doctor
        doctor_payload = {
            'doctor_identity': {
                'id': 1,
                'name': 'Test Doctor',
                'specialty': 'Cardiology'
            },
            'period': {
                'start': date(2025, 1, 1),
                'end': date(2025, 12, 31)
            },
            'metrics': {
                'total_incidents': 5,
                'high_severity': 1,
                'medium_severity': 2,
                'low_severity': 2
            },
            'performance': {
                'score': 85,
                'praise_level': 'Good',
                'risk_level': 'Low'
            }
        }
        
        try:
            result = generate_person_seasonal_word_report("doctor", doctor_payload)
            
            # Check it returns bytes
            assert isinstance(result, bytes), \
                f"Function must return bytes, got {type(result)}"
            
            # Check bytes are not empty
            assert len(result) > 0, "Returned bytes should not be empty"
            
            # Check it looks like a Word document (starts with PK for zip format)
            assert result[:2] == b'PK', \
                "Returned bytes should be a valid Word document (zip format)"
        
        except Exception as e:
            pytest.fail(f"Function call failed: {str(e)}")
    
    def test_no_db_imports(self):
        """6a. Verify no DB imports"""
        import backend.api.services.person_report_word_adapter as module
        import inspect
        
        # Get source code
        source = inspect.getsource(module)
        
        # Should NOT import DB layer
        assert 'from ..db_layer' not in source, "Should not import from db_layer"
        assert 'import pyodbc' not in source, "Should not import pyodbc"
        assert '_db.py' not in source, "Should not import any DB modules"
    
    def test_no_router_imports(self):
        """6b. Verify no router imports"""
        import backend.api.services.person_report_word_adapter as module
        import inspect
        
        # Get source code
        source = inspect.getsource(module)
        
        # Should NOT import router
        assert 'APIRouter' not in source, "Should not import APIRouter"
        assert 'from ..routers' not in source, "Should not import from routers"
    
    def test_arabic_titles_by_person_type(self):
        """7. Verify Arabic titles set based on person_type"""
        import backend.api.services.person_report_word_adapter as module
        import inspect
        
        # Get source code
        source = inspect.getsource(module)
        
        # Check for doctor Arabic title
        assert 'التقرير الموسمي للطبيب' in source, \
            "Missing Arabic title for doctor: 'التقرير الموسمي للطبيب'"
        
        # Check for worker Arabic title
        assert 'التقرير الموسمي للموظف' in source, \
            "Missing Arabic title for worker: 'التقرير الموسمي للموظف'"
        
        # Check person_type is used to set title
        assert 'person_type' in source, "Must use person_type parameter"
    
    def test_worker_report_generation(self):
        """Verify worker report generation works"""
        from backend.api.services.person_report_word_adapter import generate_person_seasonal_word_report
        
        # Create minimal test payload for worker
        worker_payload = {
            'worker_identity': {
                'employee_id': 1,
                'full_name': 'Test Worker',
                'job_title': 'Quality Specialist'
            },
            'period': {
                'start': date(2025, 1, 1),
                'end': date(2025, 12, 31)
            },
            'metrics': {
                'total_incidents': 3,
                'high_severity': 0,
                'medium_severity': 1,
                'low_severity': 2
            },
            'performance': {
                'score': 90,
                'praise_level': 'Excellent',
                'risk_level': 'Low'
            }
        }
        
        try:
            result = generate_person_seasonal_word_report("worker", worker_payload)
            
            # Check it returns bytes
            assert isinstance(result, bytes), "Function must return bytes"
            assert len(result) > 0, "Returned bytes should not be empty"
        
        except Exception as e:
            pytest.fail(f"Worker report generation failed: {str(e)}")
    
    def test_function_signature(self):
        """Verify function signature matches specification"""
        from backend.api.services.person_report_word_adapter import generate_person_seasonal_word_report
        import inspect
        
        sig = inspect.signature(generate_person_seasonal_word_report)
        params = list(sig.parameters.keys())
        
        # Check parameters
        expected_params = ['person_type', 'payload']
        assert params == expected_params, \
            f"Expected params {expected_params}, got {params}"


if __name__ == "__main__":
    print("=" * 70)
    print("TEST TASK D-B8 — WORD GENERATOR ADAPTER")
    print("=" * 70)
    print()
    
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    
    print()
    print("=" * 70)
    if exit_code == 0:
        print("✅ WORD ADAPTER OK")
    else:
        print("❌ PROBLEMS FOUND - See output above")
    print("=" * 70)
    
    exit(exit_code)
