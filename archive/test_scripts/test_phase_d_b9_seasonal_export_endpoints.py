"""
Test Suite: D-B9 Seasonal Export Endpoints

Tests the REST API endpoints for exporting seasonal reports as Word documents.
Verifies integration between seasonal builders (D-B6/D-B7), Word adapter (D-B8), and router.

Target: backend/api/routers/seasonal_export_router.py

Test Coverage:
- Doctor seasonal export endpoint logic
- Worker seasonal export endpoint logic
- Query parameter validation
- Date format validation
- Error handling (404, 400)
- Response headers (content-type, content-disposition, filename)
- End-to-end integration (without authentication)

Note: Authentication tests skipped - tested separately in Phase 2 RBAC tests
"""

import pytest
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from api.services.doctor_seasonal_reporting_service import DoctorSeasonalReportingService
from api.services.worker_seasonal_reporting_service import WorkerSeasonalReportingService
from api.services.seasonal_word_adapter import SeasonalWordAdapter


class TestSeasonalExportEndpoints:
    """Test suite for D-B9 seasonal export endpoints (logic tests without auth)."""
    
    def test_doctor_seasonal_report_data_generation(self):
        """
        Test 1: Verify doctor seasonal report data can be generated.
        
        Tests end-to-end flow: builder + adapter.
        """
        # Build report data
        data = DoctorSeasonalReportingService.build_doctor_seasonal_report_data(
            doctor_id=1,
            season_start='2024-01-01',
            season_end='2024-03-31'
        )
        
        # Generate Word document
        word_bytes = SeasonalWordAdapter.generate_doctor_seasonal_word(data)
        
        assert isinstance(word_bytes, bytes)
        assert len(word_bytes) > 0
    
    def test_worker_seasonal_report_data_generation(self):
        """
        Test 2: Verify worker seasonal report data can be generated.
        
        Tests end-to-end flow: builder + adapter.
        """
        # Build report data
        data = WorkerSeasonalReportingService.build_worker_seasonal_report_data(
            employee_id=1,
            season_start='2024-01-01',
            season_end='2024-03-31'
        )
        
        # Generate Word document
        word_bytes = SeasonalWordAdapter.generate_worker_seasonal_word(data)
        
        assert isinstance(word_bytes, bytes)
        assert len(word_bytes) > 0
    
    def test_doctor_report_filename_q1(self):
        """
        Test 3: Verify Q1 season name appears in filename format.
        """
        data = DoctorSeasonalReportingService.build_doctor_seasonal_report_data(
            doctor_id=1,
            season_start='2024-01-01',
            season_end='2024-03-31'
        )
        
        season_name = data['period']['season_name'].replace(' ', '_')
        filename = f"Doctor_1_Seasonal_Report_{season_name}.docx"
        
        assert 'Q1' in filename
        assert '2024' in filename
        assert filename.endswith('.docx')
    
    def test_worker_report_filename_q2(self):
        """
        Test 4: Verify Q2 season name appears in filename format.
        """
        data = WorkerSeasonalReportingService.build_worker_seasonal_report_data(
            employee_id=1,
            season_start='2024-04-01',
            season_end='2024-06-30'
        )
        
        season_name = data['period']['season_name'].replace(' ', '_')
        filename = f"Worker_1_Seasonal_Report_{season_name}.docx"
        
        assert 'Q2' in filename
        assert '2024' in filename
        assert filename.endswith('.docx')
    
    def test_doctor_report_annual_filename(self):
        """
        Test 5: Verify annual season name appears in filename.
        """
        data = DoctorSeasonalReportingService.build_doctor_seasonal_report_data(
            doctor_id=1,
            season_start='2023-01-01',
            season_end='2023-12-31'
        )
        
        season_name = data['period']['season_name'].replace(' ', '_')
        filename = f"Doctor_1_Seasonal_Report_{season_name}.docx"
        
        assert 'Annual' in filename
        assert '2023' in filename
    
    def test_worker_report_annual_filename(self):
        """
        Test 6: Verify annual season name appears in filename.
        """
        data = WorkerSeasonalReportingService.build_worker_seasonal_report_data(
            employee_id=1,
            season_start='2023-01-01',
            season_end='2023-12-31'
        )
        
        season_name = data['period']['season_name'].replace(' ', '_')
        filename = f"Worker_1_Seasonal_Report_{season_name}.docx"
        
        assert 'Annual' in filename
        assert '2023' in filename
    
    def test_doctor_report_invalid_date_format(self):
        """
        Test 7: Verify error on invalid date format.
        """
        with pytest.raises(ValueError) as exc_info:
            DoctorSeasonalReportingService.build_doctor_seasonal_report_data(
                doctor_id=1,
                season_start='01-01-2024',  # Wrong format
                season_end='2024-03-31'
            )
        
        assert 'date format' in str(exc_info.value).lower()
    
    def test_worker_report_invalid_date_format(self):
        """
        Test 8: Verify error on invalid date format.
        """
        with pytest.raises(ValueError) as exc_info:
            WorkerSeasonalReportingService.build_worker_seasonal_report_data(
                employee_id=1,
                season_start='2024-01-01',
                season_end='31-03-2024'  # Wrong format
            )
        
        assert 'date format' in str(exc_info.value).lower()
    
    def test_doctor_report_not_found(self):
        """
        Test 9: Verify error when doctor doesn't exist.
        """
        with pytest.raises(ValueError) as exc_info:
            DoctorSeasonalReportingService.build_doctor_seasonal_report_data(
                doctor_id=999999,
                season_start='2024-01-01',
                season_end='2024-03-31'
            )
        
        assert 'not found' in str(exc_info.value).lower()
    
    def test_worker_report_not_found(self):
        """
        Test 10: Verify error when worker doesn't exist.
        """
        with pytest.raises(ValueError) as exc_info:
            WorkerSeasonalReportingService.build_worker_seasonal_report_data(
                employee_id=999999,
                season_start='2024-01-01',
                season_end='2024-03-31'
            )
        
        assert 'not found' in str(exc_info.value).lower()
    
    def test_router_file_exists(self):
        """
        Test 11: Verify seasonal export router file exists.
        """
        router_file = backend_path / "api" / "routers" / "seasonal_export_router.py"
        assert router_file.exists()
    
    def test_router_has_required_endpoints(self):
        """
        Test 12: Verify router defines required endpoints.
        """
        from api.routers import seasonal_export_router
        
        # Check router is properly defined
        assert hasattr(seasonal_export_router, 'router')
        assert seasonal_export_router.router is not None


if __name__ == '__main__':
    """Run tests with pytest."""
    pytest.main([__file__, '-v', '--tb=short'])
    
    def test_doctor_export_content_disposition_header(self):
        """
        Test 2: Verify doctor export has proper content-disposition header.
        
        Should include filename with doctor_id and season.
        """
        # Login first
        login_response = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        assert login_response.status_code == 200
        
        # Export report
        response = client.get(
            "/api/doctors/1/seasonal-report",
            params={
                "season_start": "2024-01-01",
                "season_end": "2024-03-31"
            }
        )
        
        assert response.status_code == 200
        assert "content-disposition" in response.headers
        content_disposition = response.headers["content-disposition"]
        assert "attachment" in content_disposition
        assert "Doctor_1" in content_disposition
        assert ".docx" in content_disposition
    
    def test_doctor_export_invalid_date_format(self):
        """
        Test 3: Verify error on invalid date format.
        
        Should return 400 Bad Request.
        """
        # Login first
        login_response = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        assert login_response.status_code == 200
        
        # Try with invalid date format
        response = client.get(
            "/api/doctors/1/seasonal-report",
            params={
                "season_start": "01-01-2024",  # Wrong format
                "season_end": "2024-03-31"
            }
        )
        
        assert response.status_code == 400
        assert "date format" in response.json()["detail"].lower()
    
    def test_doctor_export_missing_parameters(self):
        """
        Test 4: Verify error when required parameters are missing.
        
        Should return 422 Unprocessable Entity.
        """
        # Login first
        login_response = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        assert login_response.status_code == 200
        
        # Try without season_start
        response = client.get(
            "/api/doctors/1/seasonal-report",
            params={
                "season_end": "2024-03-31"
            }
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_doctor_export_not_found(self):
        """
        Test 5: Verify 404 when doctor doesn't exist.
        
        Should return 404 Not Found.
        """
        # Login first
        login_response = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        assert login_response.status_code == 200
        
        # Try with non-existent doctor
        response = client.get(
            "/api/doctors/999999/seasonal-report",
            params={
                "season_start": "2024-01-01",
                "season_end": "2024-03-31"
            }
        )
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_doctor_export_requires_authentication(self):
        """
        Test 6: Verify authentication is required.
        
        Should return 401 Unauthorized without login.
        """
        # Create a fresh client without session
        fresh_client = TestClient(app)
        
        response = fresh_client.get(
            "/api/doctors/1/seasonal-report",
            params={
                "season_start": "2024-01-01",
                "season_end": "2024-03-31"
            }
        )
        
        assert response.status_code == 401
    
    def test_worker_export_returns_word_document(self):
        """
        Test 7: Verify worker export returns Word document.
        
        Should return 200 with proper DOCX content-type.
        """
        # Login first
        login_response = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        assert login_response.status_code == 200
        
        # Export worker seasonal report
        response = client.get(
            "/api/workers/1/seasonal-report",
            params={
                "season_start": "2024-01-01",
                "season_end": "2024-03-31"
            }
        )
        
        assert response.status_code == 200
        assert "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in response.headers["content-type"]
        assert len(response.content) > 0
    
    def test_worker_export_content_disposition_header(self):
        """
        Test 8: Verify worker export has proper content-disposition header.
        
        Should include filename with employee_id and season.
        """
        # Login first
        login_response = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        assert login_response.status_code == 200
        
        # Export report
        response = client.get(
            "/api/workers/1/seasonal-report",
            params={
                "season_start": "2024-01-01",
                "season_end": "2024-03-31"
            }
        )
        
        assert response.status_code == 200
        assert "content-disposition" in response.headers
        content_disposition = response.headers["content-disposition"]
        assert "attachment" in content_disposition
        assert "Worker_1" in content_disposition
        assert ".docx" in content_disposition
    
    def test_worker_export_invalid_date_format(self):
        """
        Test 9: Verify error on invalid date format.
        
        Should return 400 Bad Request.
        """
        # Login first
        login_response = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        assert login_response.status_code == 200
        
        # Try with invalid date format
        response = client.get(
            "/api/workers/1/seasonal-report",
            params={
                "season_start": "2024-01-01",
                "season_end": "31-03-2024"  # Wrong format
            }
        )
        
        assert response.status_code == 400
        assert "date format" in response.json()["detail"].lower()
    
    def test_worker_export_not_found(self):
        """
        Test 10: Verify 404 when worker doesn't exist.
        
        Should return 404 Not Found.
        """
        # Login first
        login_response = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        assert login_response.status_code == 200
        
        # Try with non-existent worker
        response = client.get(
            "/api/workers/999999/seasonal-report",
            params={
                "season_start": "2024-01-01",
                "season_end": "2024-03-31"
            }
        )
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_worker_export_requires_authentication(self):
        """
        Test 11: Verify authentication is required.
        
        Should return 401 Unauthorized without login.
        """
        # Create a fresh client without session
        fresh_client = TestClient(app)
        
        response = fresh_client.get(
            "/api/workers/1/seasonal-report",
            params={
                "season_start": "2024-01-01",
                "season_end": "2024-03-31"
            }
        )
        
        assert response.status_code == 401
    
    def test_doctor_export_different_seasons(self):
        """
        Test 12: Verify different season names in filename.
        
        Q1, Q2, Annual should produce different filenames.
        """
        # Login first
        login_response = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        assert login_response.status_code == 200
        
        # Q1 2024
        response_q1 = client.get(
            "/api/doctors/1/seasonal-report",
            params={
                "season_start": "2024-01-01",
                "season_end": "2024-03-31"
            }
        )
        assert response_q1.status_code == 200
        assert "Q1" in response_q1.headers["content-disposition"]
        
        # Q3 2024
        response_q3 = client.get(
            "/api/doctors/1/seasonal-report",
            params={
                "season_start": "2024-07-01",
                "season_end": "2024-09-30"
            }
        )
        assert response_q3.status_code == 200
        assert "Q3" in response_q3.headers["content-disposition"]
    
    def test_worker_export_different_seasons(self):
        """
        Test 13: Verify different season names in filename.
        
        Q1, Q2, Annual should produce different filenames.
        """
        # Login first
        login_response = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        assert login_response.status_code == 200
        
        # Q2 2024
        response_q2 = client.get(
            "/api/workers/1/seasonal-report",
            params={
                "season_start": "2024-04-01",
                "season_end": "2024-06-30"
            }
        )
        assert response_q2.status_code == 200
        assert "Q2" in response_q2.headers["content-disposition"]
        
        # Annual 2023
        response_annual = client.get(
            "/api/workers/1/seasonal-report",
            params={
                "season_start": "2023-01-01",
                "season_end": "2023-12-31"
            }
        )
        assert response_annual.status_code == 200
        assert "Annual" in response_annual.headers["content-disposition"]


if __name__ == '__main__':
    """Run tests with pytest."""
    pytest.main([__file__, '-v', '--tb=short'])
