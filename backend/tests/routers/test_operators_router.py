"""
Test Suite: Operators Router

Integration tests for the Distribution Operator endpoint.
"""

import sys
import os
# Add backend directory to path for imports
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from main import app
from api.schemas.operators.base import DimensionType, TimeMode


# Create test client
client = TestClient(app)


# ============================================================================
# TEST SUCCESSFUL REQUESTS (200)
# ============================================================================

class TestSuccessfulRequests:
    """Test successful distribution operator requests"""
    
    @patch('api.routers.operators_router.DistributionService')
    def test_single_mode_year_request(self, mock_service_class):
        """Test successful single mode with year"""
        # Mock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        # Mock response
        mock_service.execute.return_value = Mock(
            dimension="severity",
            time_mode="single",
            buckets=[
                Mock(
                    time_label="2025",
                    total=100,
                    status=None,
                    values=[Mock(key="High", count=100, percent=1.0)]
                )
            ]
        )
        
        # Make request
        response = client.post("/api/operators/distribution", json={
            "dimension": "severity",
            "time_mode": "single",
            "time_window": {
                "type": "year",
                "value": 2025
            }
        })
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["dimension"] == "severity"
        assert data["time_mode"] == "single"
        assert len(data["buckets"]) == 1
    
    @patch('api.routers.operators_router.DistributionService')
    def test_single_mode_season_request(self, mock_service_class):
        """Test successful single mode with season (quarter)"""
        # Mock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        mock_service.execute.return_value = Mock(
            dimension="domain",
            time_mode="single",
            buckets=[
                Mock(
                    time_label="2024-Q1",
                    total=200,
                    status=None,
                    values=[
                        Mock(key="Clinical", count=150, percent=0.75),
                        Mock(key="Administrative", count=50, percent=0.25)
                    ]
                )
            ]
        )
        
        # Make request
        response = client.post("/api/operators/distribution", json={
            "dimension": "domain",
            "time_mode": "single",
            "time_window": {
                "type": "season",
                "value": "2024-Q1"
            }
        })
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["dimension"] == "domain"
        assert len(data["buckets"]) == 1
        assert data["buckets"][0]["time_label"] == "2024-Q1"
    
    @patch('api.routers.operators_router.DistributionService')
    def test_multi_mode_request(self, mock_service_class):
        """Test successful multi mode request"""
        # Mock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        from api.schemas.operators.distribution import DistributionResponse, DistributionBucket, DistributionValue
        
        mock_service.execute.return_value = DistributionResponse(
            dimension="severity",
            time_mode="multi",
            buckets=[
                DistributionBucket(time_label="2023", total=0, status="NO_DATA", values=[]),
                DistributionBucket(time_label="2024", total=0, status="NO_DATA", values=[]),
                DistributionBucket(time_label="2025", total=0, status="NO_DATA", values=[])
            ]
        )
        
        # Make request
        response = client.post("/api/operators/distribution", json={
            "dimension": "severity",
            "time_mode": "multi",
            "time_windows": [
                {"type": "year", "value": 2023},
                {"type": "year", "value": 2024},
                {"type": "year", "value": 2025}
            ]
        })
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["time_mode"] == "multi"
        assert len(data["buckets"]) == 3
    
    @patch('api.routers.operators_router.DistributionService')
    def test_binary_split_request(self, mock_service_class):
        """Test successful binary split mode request"""
        # Mock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        from api.schemas.operators.distribution import DistributionResponse, DistributionBucket, DistributionValue
        
        mock_service.execute.return_value = DistributionResponse(
            dimension="stage",
            time_mode="binary_split",
            buckets=[
                DistributionBucket(time_label="Before", total=0, status="NO_DATA", values=[]),
                DistributionBucket(time_label="After", total=0, status="NO_DATA", values=[])
            ]
        )
        
        # Make request
        response = client.post("/api/operators/distribution", json={
            "dimension": "stage",
            "time_mode": "binary_split",
            "split_date": "2023-06-01"
        })
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["time_mode"] == "binary_split"
        assert len(data["buckets"]) == 2
        assert data["buckets"][0]["time_label"] == "Before"
        assert data["buckets"][1]["time_label"] == "After"
    
    @patch('api.routers.operators_router.DistributionService')
    def test_request_with_filters(self, mock_service_class):
        """Test request with organizational and dimensional filters"""
        # Mock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        from api.schemas.operators.distribution import DistributionResponse, DistributionBucket, DistributionValue
        
        mock_service.execute.return_value = DistributionResponse(
            dimension="severity",
            time_mode="single",
            buckets=[DistributionBucket(time_label="2025", total=0, status="NO_DATA", values=[])]
        )
        
        # Make request with filters
        response = client.post("/api/operators/distribution", json={
            "dimension": "severity",
            "time_mode": "single",
            "time_window": {"type": "year", "value": 2025},
            "filters": {
                "department_id": 42,
                "severity": "High",
                "domain": "Clinical"
            }
        })
        
        # Verify response
        assert response.status_code == 200
        
        # Verify filters were passed to service
        call_args = mock_service.execute.call_args[0][0]
        assert call_args.filters is not None
        assert call_args.filters.department_id == 42
        assert call_args.filters.severity == "High"
        assert call_args.filters.domain == "Clinical"


# ============================================================================
# TEST VALIDATION ERRORS (422)
# ============================================================================

class TestValidationErrors:
    """Test Pydantic validation errors"""
    
    def test_missing_dimension(self):
        """Test error when dimension is missing"""
        response = client.post("/api/operators/distribution", json={
            "time_mode": "single",
            "time_window": {"type": "year", "value": 2025}
        })
        
        assert response.status_code == 422
        error = response.json()
        assert "detail" in error
    
    def test_invalid_dimension_value(self):
        """Test error when dimension has invalid value"""
        response = client.post("/api/operators/distribution", json={
            "dimension": "invalid_dimension",
            "time_mode": "single",
            "time_window": {"type": "year", "value": 2025}
        })
        
        assert response.status_code == 422
    
    def test_missing_time_mode(self):
        """Test error when time_mode is missing"""
        response = client.post("/api/operators/distribution", json={
            "dimension": "severity",
            "time_window": {"type": "year", "value": 2025}
        })
        
        assert response.status_code == 422
    
    def test_invalid_time_mode(self):
        """Test error when time_mode has invalid value"""
        response = client.post("/api/operators/distribution", json={
            "dimension": "severity",
            "time_mode": "invalid_mode",
            "time_window": {"type": "year", "value": 2025}
        })
        
        assert response.status_code == 422
    
    def test_single_mode_without_time_window(self):
        """Test error when single mode missing time_window"""
        response = client.post("/api/operators/distribution", json={
            "dimension": "severity",
            "time_mode": "single"
        })
        
        assert response.status_code == 422
    
    def test_single_mode_with_time_windows_array(self):
        """Test error when single mode has time_windows instead of time_window"""
        response = client.post("/api/operators/distribution", json={
            "dimension": "severity",
            "time_mode": "single",
            "time_windows": [{"type": "year", "value": 2025}]
        })
        
        assert response.status_code == 422
    
    def test_multi_mode_without_time_windows(self):
        """Test error when multi mode missing time_windows"""
        response = client.post("/api/operators/distribution", json={
            "dimension": "severity",
            "time_mode": "multi"
        })
        
        assert response.status_code == 422
    
    def test_multi_mode_with_single_time_window(self):
        """Test error when multi mode has time_window instead of time_windows"""
        response = client.post("/api/operators/distribution", json={
            "dimension": "severity",
            "time_mode": "multi",
            "time_window": {"type": "year", "value": 2025}
        })
        
        assert response.status_code == 422
    
    def test_binary_split_without_split_date(self):
        """Test error when binary_split missing split_date"""
        response = client.post("/api/operators/distribution", json={
            "dimension": "severity",
            "time_mode": "binary_split"
        })
        
        assert response.status_code == 422
    
    def test_invalid_year_value(self):
        """Test error when year is invalid (too old)"""
        response = client.post("/api/operators/distribution", json={
            "dimension": "severity",
            "time_mode": "single",
            "time_window": {"type": "year", "value": 1999}
        })
        
        assert response.status_code == 422
    
    def test_invalid_year_value_future(self):
        """Test error when year is too far in future"""
        response = client.post("/api/operators/distribution", json={
            "dimension": "severity",
            "time_mode": "single",
            "time_window": {"type": "year", "value": 2101}  # Above max of 2100
        })
        
        assert response.status_code == 422
    
    def test_invalid_season_format(self):
        """Test error when season format is invalid"""
        response = client.post("/api/operators/distribution", json={
            "dimension": "severity",
            "time_mode": "single",
            "time_window": {"type": "season", "value": "2025-Quarter1"}
        })
        
        assert response.status_code == 422
    
    def test_invalid_month_format(self):
        """Test error when month format is invalid"""
        response = client.post("/api/operators/distribution", json={
            "dimension": "severity",
            "time_mode": "single",
            "time_window": {"type": "month", "value": "2025/13"}
        })
        
        assert response.status_code == 422
    
    def test_missing_time_window_type(self):
        """Test error when time_window missing 'type' discriminator"""
        response = client.post("/api/operators/distribution", json={
            "dimension": "severity",
            "time_mode": "single",
            "time_window": {"value": 2025}
        })
        
        assert response.status_code == 422


# ============================================================================
# TEST BUSINESS LOGIC ERRORS (400)
# ============================================================================

class TestBusinessLogicErrors:
    """Test business logic errors from service layer"""
    
    @patch('api.routers.operators_router.DistributionService')
    def test_service_raises_value_error(self, mock_service_class):
        """Test that ValueError from service returns 400"""
        # Mock service to raise ValueError
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.execute.side_effect = ValueError("Invalid configuration")
        
        # Make request
        response = client.post("/api/operators/distribution", json={
            "dimension": "severity",
            "time_mode": "single",
            "time_window": {"type": "year", "value": 2025}
        })
        
        # Verify 400 error
        assert response.status_code == 400
        error = response.json()
        assert "detail" in error
        assert "Invalid configuration" in error["detail"]


# ============================================================================
# TEST INTERNAL ERRORS (500)
# ============================================================================

class TestInternalErrors:
    """Test internal server errors"""
    
    @patch('api.routers.operators_router.DistributionService')
    def test_service_raises_unexpected_error(self, mock_service_class):
        """Test that unexpected errors return 500"""
        # Mock service to raise unexpected error
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.execute.side_effect = RuntimeError("Database connection failed")
        
        # Make request
        response = client.post("/api/operators/distribution", json={
            "dimension": "severity",
            "time_mode": "single",
            "time_window": {"type": "year", "value": 2025}
        })
        
        # Verify 500 error
        assert response.status_code == 500
        error = response.json()
        assert "detail" in error
        assert "internal server error" in error["detail"].lower()


# ============================================================================
# TEST ALL DIMENSION TYPES
# ============================================================================

class TestAllDimensionTypes:
    """Test that all dimension types are accepted"""
    
    @patch('api.routers.operators_router.DistributionService')
    def test_domain_dimension(self, mock_service_class):
        """Test domain dimension"""
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.execute.return_value = Mock(
            dimension="domain",
            time_mode="single",
            buckets=[Mock(time_label="2025", total=0, status="NO_DATA", values=[])]
        )
        
        response = client.post("/api/operators/distribution", json={
            "dimension": "domain",
            "time_mode": "single",
            "time_window": {"type": "year", "value": 2025}
        })
        
        assert response.status_code == 200
    
    @patch('api.routers.operators_router.DistributionService')
    def test_category_dimension(self, mock_service_class):
        """Test category dimension"""
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.execute.return_value = Mock(
            dimension="category",
            time_mode="single",
            buckets=[Mock(time_label="2025", total=0, status="NO_DATA", values=[])]
        )
        
        response = client.post("/api/operators/distribution", json={
            "dimension": "category",
            "time_mode": "single",
            "time_window": {"type": "year", "value": 2025}
        })
        
        assert response.status_code == 200
    
    @patch('api.routers.operators_router.DistributionService')
    def test_severity_dimension(self, mock_service_class):
        """Test severity dimension"""
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.execute.return_value = Mock(
            dimension="severity",
            time_mode="single",
            buckets=[Mock(time_label="2025", total=0, status="NO_DATA", values=[])]
        )
        
        response = client.post("/api/operators/distribution", json={
            "dimension": "severity",
            "time_mode": "single",
            "time_window": {"type": "year", "value": 2025}
        })
        
        assert response.status_code == 200


# ============================================================================
# TEST EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    @patch('api.routers.operators_router.DistributionService')
    def test_empty_request_body(self, mock_service_class):
        """Test error with empty request body"""
        response = client.post("/api/operators/distribution", json={})
        
        assert response.status_code == 422
    
    @patch('api.routers.operators_router.DistributionService')
    def test_request_with_extra_fields(self, mock_service_class):
        """Test that extra fields cause validation error (Pydantic default is forbid)"""
        # Pydantic v2 forbids extra fields by default
        # Request with extra field
        response = client.post("/api/operators/distribution", json={
            "dimension": "severity",
            "time_mode": "single",
            "time_window": {"type": "year", "value": 2025},
            "extra_field": "should_be_rejected"
        })
        
        # Should fail with 422 (extra fields forbidden)
        assert response.status_code == 422
    
    @patch('api.routers.operators_router.DistributionService')
    def test_multi_mode_with_single_window(self, mock_service_class):
        """Test multi mode with only one window in array fails validation"""
        # Distribution schema requires min_length=2 for time_windows in multi mode
        response = client.post("/api/operators/distribution", json={
            "dimension": "severity",
            "time_mode": "multi",
            "time_windows": [{"type": "year", "value": 2025}]
        })
        
        # Should fail with 422 (min 2 windows required for multi mode)
        assert response.status_code == 422
    
    @patch('api.routers.operators_router.DistributionService')
    def test_filters_with_null_values(self, mock_service_class):
        """Test that null filter values are accepted"""
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.execute.return_value = Mock(
            dimension="severity",
            time_mode="single",
            buckets=[Mock(time_label="2025", total=0, status="NO_DATA", values=[])]
        )
        
        response = client.post("/api/operators/distribution", json={
            "dimension": "severity",
            "time_mode": "single",
            "time_window": {"type": "year", "value": 2025},
            "filters": {
                "department_id": None,
                "severity": None
            }
        })
        
        assert response.status_code == 200
