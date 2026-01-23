"""
Test Suite: Distribution Service

Tests for business logic layer of the Distribution Operator.
"""

import sys
import os
# Add backend directory to path for imports
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import date

from api.services.operators.distribution_service import DistributionService
from api.schemas.operators.distribution import (
    DistributionRequest,
    DistributionResponse,
    DistributionBucket,
    DistributionValue
)
from api.schemas.operators.base import (
    DimensionType,
    TimeMode,
    TimeWindowYear,
    TimeWindowSeason,
    TimeWindowMonth,
    TimeWindowRange,
    OperatorFilters
)


# ============================================================================
# TEST DISTRIBUTION SERVICE - SINGLE MODE
# ============================================================================

class TestDistributionServiceSingleMode:
    """Test DistributionService with SINGLE time mode"""
    
    def test_execute_single_mode_with_data(self):
        """Test single mode with actual data"""
        # Mock DB
        mock_db = Mock()
        mock_db.query_single_window.return_value = [
            {"dimension_value": "High", "count": 150},
            {"dimension_value": "Medium", "count": 300},
            {"dimension_value": "Low", "count": 50}
        ]
        
        # Create service
        service = DistributionService(db=mock_db)
        
        # Create request
        request = DistributionRequest(
            dimension=DimensionType.SEVERITY,
            time_mode=TimeMode.SINGLE,
            time_window=TimeWindowYear(value=2025)
        )
        
        # Execute
        response = service.execute(request)
        
        # Verify response structure
        assert isinstance(response, DistributionResponse)
        assert response.dimension == "severity"
        assert response.time_mode == "single"
        assert len(response.buckets) == 1
        
        # Verify bucket
        bucket = response.buckets[0]
        assert bucket.time_label == "2025"
        assert bucket.total == 500
        assert len(bucket.values) == 3
        assert bucket.status is None
        
        # Verify values and percentages
        assert bucket.values[0].key == "High"
        assert bucket.values[0].count == 150
        assert bucket.values[0].percent == 0.3
        
        assert bucket.values[1].key == "Medium"
        assert bucket.values[1].count == 300
        assert bucket.values[1].percent == 0.6
        
        assert bucket.values[2].key == "Low"
        assert bucket.values[2].count == 50
        assert bucket.values[2].percent == 0.1
    
    def test_execute_single_mode_with_no_data(self):
        """Test single mode when no data exists"""
        # Mock DB returns empty
        mock_db = Mock()
        mock_db.query_single_window.return_value = []
        
        # Create service
        service = DistributionService(db=mock_db)
        
        # Create request
        request = DistributionRequest(
            dimension=DimensionType.SEVERITY,
            time_mode=TimeMode.SINGLE,
            time_window=TimeWindowYear(value=2025)
        )
        
        # Execute
        response = service.execute(request)
        
        # Verify NO_DATA status
        assert len(response.buckets) == 1
        bucket = response.buckets[0]
        assert bucket.total == 0
        assert len(bucket.values) == 0
        assert bucket.status == "NO_DATA"
    
    def test_execute_single_mode_with_season(self):
        """Test single mode with season time window"""
        # Mock DB
        mock_db = Mock()
        mock_db.query_single_window.return_value = [
            {"dimension_value": "Clinical", "count": 200}
        ]
        
        # Create service
        service = DistributionService(db=mock_db)
        
        # Create request with season
        request = DistributionRequest(
            dimension=DimensionType.DOMAIN,
            time_mode=TimeMode.SINGLE,
            time_window=TimeWindowSeason(value="2025-Q1")
        )
        
        # Execute
        response = service.execute(request)
        
        # Verify season label
        assert response.buckets[0].time_label == "2025-Q1"


# ============================================================================
# TEST DISTRIBUTION SERVICE - MULTI MODE
# ============================================================================

class TestDistributionServiceMultiMode:
    """Test DistributionService with MULTI time mode"""
    
    def test_execute_multi_mode_with_years(self):
        """Test multi mode with multiple years"""
        # Mock DB - different data for each year
        mock_db = Mock()
        mock_db.query_single_window.side_effect = [
            # 2023 data
            [{"dimension_value": "High", "count": 100}],
            # 2024 data
            [{"dimension_value": "High", "count": 150}],
            # 2025 data
            [{"dimension_value": "High", "count": 200}]
        ]
        
        # Create service
        service = DistributionService(db=mock_db)
        
        # Create request
        request = DistributionRequest(
            dimension=DimensionType.SEVERITY,
            time_mode=TimeMode.MULTI,
            time_windows=[
                TimeWindowYear(value=2023),
                TimeWindowYear(value=2024),
                TimeWindowYear(value=2025)
            ]
        )
        
        # Execute
        response = service.execute(request)
        
        # Verify 3 buckets
        assert len(response.buckets) == 3
        
        # Verify labels
        assert response.buckets[0].time_label == "2023"
        assert response.buckets[1].time_label == "2024"
        assert response.buckets[2].time_label == "2025"
        
        # Verify totals
        assert response.buckets[0].total == 100
        assert response.buckets[1].total == 150
        assert response.buckets[2].total == 200
        
        # Verify DB was called 3 times
        assert mock_db.query_single_window.call_count == 3
    
    def test_execute_multi_mode_with_quarters(self):
        """Test multi mode with quarters"""
        # Mock DB
        mock_db = Mock()
        mock_db.query_single_window.side_effect = [
            [{"dimension_value": "Stage 1", "count": 50}],
            [{"dimension_value": "Stage 1", "count": 60}],
            [{"dimension_value": "Stage 1", "count": 70}],
            [{"dimension_value": "Stage 1", "count": 80}]
        ]
        
        # Create service
        service = DistributionService(db=mock_db)
        
        # Create request
        request = DistributionRequest(
            dimension=DimensionType.STAGE,
            time_mode=TimeMode.MULTI,
            time_windows=[
                TimeWindowSeason(value="2024-Q1"),
                TimeWindowSeason(value="2024-Q2"),
                TimeWindowSeason(value="2024-Q3"),
                TimeWindowSeason(value="2024-Q4")
            ]
        )
        
        # Execute
        response = service.execute(request)
        
        # Verify 4 buckets
        assert len(response.buckets) == 4
        
        # Verify labels
        assert response.buckets[0].time_label == "2024-Q1"
        assert response.buckets[1].time_label == "2024-Q2"
        assert response.buckets[2].time_label == "2024-Q3"
        assert response.buckets[3].time_label == "2024-Q4"
    
    def test_execute_multi_mode_mixed_empty_and_data(self):
        """Test multi mode where some periods have no data"""
        # Mock DB - some empty, some with data
        mock_db = Mock()
        mock_db.query_single_window.side_effect = [
            [],  # 2023: no data
            [{"dimension_value": "Medium", "count": 100}],  # 2024: has data
            []   # 2025: no data
        ]
        
        # Create service
        service = DistributionService(db=mock_db)
        
        # Create request
        request = DistributionRequest(
            dimension=DimensionType.SEVERITY,
            time_mode=TimeMode.MULTI,
            time_windows=[
                TimeWindowYear(value=2023),
                TimeWindowYear(value=2024),
                TimeWindowYear(value=2025)
            ]
        )
        
        # Execute
        response = service.execute(request)
        
        # Verify all 3 buckets exist
        assert len(response.buckets) == 3
        
        # Verify first bucket is empty
        assert response.buckets[0].total == 0
        assert response.buckets[0].status == "NO_DATA"
        
        # Verify second bucket has data
        assert response.buckets[1].total == 100
        assert response.buckets[1].status is None
        
        # Verify third bucket is empty
        assert response.buckets[2].total == 0
        assert response.buckets[2].status == "NO_DATA"


# ============================================================================
# TEST DISTRIBUTION SERVICE - BINARY_SPLIT MODE
# ============================================================================

class TestDistributionServiceBinarySplitMode:
    """Test DistributionService with BINARY_SPLIT time mode"""
    
    def test_execute_binary_split(self):
        """Test binary split mode"""
        # Mock DB
        mock_db = Mock()
        mock_db.query_date_range.side_effect = [
            # Before data
            [{"dimension_value": "Old", "count": 50}],
            # After data
            [{"dimension_value": "New", "count": 150}]
        ]
        
        # Create service
        service = DistributionService(db=mock_db)
        
        # Create request
        request = DistributionRequest(
            dimension=DimensionType.STAGE,
            time_mode=TimeMode.BINARY_SPLIT,
            split_date="2023-06-01"
        )
        
        # Execute
        response = service.execute(request)
        
        # Verify 2 buckets
        assert len(response.buckets) == 2
        
        # Verify labels
        assert response.buckets[0].time_label == "Before"
        assert response.buckets[1].time_label == "After"
        
        # Verify data
        assert response.buckets[0].total == 50
        assert response.buckets[1].total == 150
        
        # Verify DB was called correctly
        assert mock_db.query_date_range.call_count == 2
        
        # Verify first call (before)
        call1_args = mock_db.query_date_range.call_args_list[0]
        assert call1_args[1]['from_date'] is None
        assert call1_args[1]['to_date'] == date(2023, 6, 1)
        
        # Verify second call (after)
        call2_args = mock_db.query_date_range.call_args_list[1]
        assert call2_args[1]['from_date'] == date(2023, 6, 2)  # Day after split
        assert call2_args[1]['to_date'] is None
    
    def test_execute_binary_split_both_empty(self):
        """Test binary split when both periods are empty"""
        # Mock DB returns empty for both
        mock_db = Mock()
        mock_db.query_date_range.side_effect = [[], []]
        
        # Create service
        service = DistributionService(db=mock_db)
        
        # Create request
        request = DistributionRequest(
            dimension=DimensionType.SEVERITY,
            time_mode=TimeMode.BINARY_SPLIT,
            split_date="2020-01-01"
        )
        
        # Execute
        response = service.execute(request)
        
        # Verify both buckets are empty
        assert response.buckets[0].status == "NO_DATA"
        assert response.buckets[1].status == "NO_DATA"


# ============================================================================
# TEST DATA TRANSFORMATION
# ============================================================================

class TestDataTransformation:
    """Test _raw_data_to_bucket method"""
    
    def test_percentage_calculation(self):
        """Test that percentages are calculated correctly"""
        service = DistributionService()
        
        raw_data = [
            {"dimension_value": "A", "count": 100},
            {"dimension_value": "B", "count": 200},
            {"dimension_value": "C", "count": 700}
        ]
        
        bucket = service._raw_data_to_bucket("Test", raw_data)
        
        # Verify total
        assert bucket.total == 1000
        
        # Verify percentages
        assert bucket.values[0].percent == 0.1
        assert bucket.values[1].percent == 0.2
        assert bucket.values[2].percent == 0.7
        
        # Verify sum of percentages = 1.0
        sum_percent = sum(v.percent for v in bucket.values)
        assert abs(sum_percent - 1.0) < 0.0001
    
    def test_handles_none_dimension_value(self):
        """Test that None dimension values are converted to 'Unknown'"""
        service = DistributionService()
        
        raw_data = [
            {"dimension_value": None, "count": 50},
            {"dimension_value": "Known", "count": 100}
        ]
        
        bucket = service._raw_data_to_bucket("Test", raw_data)
        
        # Verify None is converted to "Unknown"
        assert bucket.values[0].key == "Unknown"
        assert bucket.values[1].key == "Known"
    
    def test_single_value_gets_100_percent(self):
        """Test that a single value gets 100% (1.0)"""
        service = DistributionService()
        
        raw_data = [
            {"dimension_value": "Only", "count": 456}
        ]
        
        bucket = service._raw_data_to_bucket("Test", raw_data)
        
        assert bucket.total == 456
        assert len(bucket.values) == 1
        assert bucket.values[0].percent == 1.0
    
    def test_empty_data_returns_no_data_status(self):
        """Test that empty data returns NO_DATA status"""
        service = DistributionService()
        
        raw_data = []
        
        bucket = service._raw_data_to_bucket("Empty Period", raw_data)
        
        assert bucket.total == 0
        assert len(bucket.values) == 0
        assert bucket.status == "NO_DATA"
        assert bucket.time_label == "Empty Period"


# ============================================================================
# TEST ERROR HANDLING
# ============================================================================

class TestErrorHandling:
    """Test error handling in service"""
    
    def test_invalid_time_mode_raises_error(self):
        """Test that invalid time mode raises ValueError"""
        service = DistributionService()
        
        # Create invalid request (manually construct to bypass validation)
        request = Mock()
        request.time_mode = "invalid_mode"
        request.dimension = DimensionType.SEVERITY
        
        with pytest.raises(ValueError, match="Unknown time mode"):
            service.execute(request)
    
    def test_service_works_without_explicit_db(self):
        """Test that service creates DB instance if not provided"""
        # Create service without DB
        service = DistributionService()
        
        # Verify DB exists
        assert service.db is not None
        assert hasattr(service.db, 'query_single_window')
    
    def test_service_uses_provided_db(self):
        """Test that service uses provided DB instance"""
        # Create mock DB
        mock_db = Mock()
        
        # Create service with mock DB
        service = DistributionService(db=mock_db)
        
        # Verify service uses the mock
        assert service.db is mock_db


# ============================================================================
# TEST WITH FILTERS
# ============================================================================

class TestServiceWithFilters:
    """Test that filters are passed through to DB layer"""
    
    def test_filters_passed_to_db(self):
        """Test that filters in request are passed to DB"""
        # Mock DB
        mock_db = Mock()
        mock_db.query_single_window.return_value = [
            {"dimension_value": "Test", "count": 100}
        ]
        
        # Create service
        service = DistributionService(db=mock_db)
        
        # Create request with filters
        filters = OperatorFilters(
            department_id=42,
            severity="High"
        )
        
        request = DistributionRequest(
            dimension=DimensionType.DOMAIN,
            time_mode=TimeMode.SINGLE,
            time_window=TimeWindowYear(value=2025),
            filters=filters
        )
        
        # Execute
        service.execute(request)
        
        # Verify filters were passed to DB
        call_args = mock_db.query_single_window.call_args
        assert call_args[1]['filters'] == filters
        assert call_args[1]['filters'].department_id == 42
        assert call_args[1]['filters'].severity == "High"
