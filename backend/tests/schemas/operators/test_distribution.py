"""
Test Suite: Distribution Operator Schema

Comprehensive tests for the Distribution Operator (DIST_1D_TIME_PARTITIONED).
Tests all validation rules, edge cases, and error conditions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import pytest
from datetime import date
from pydantic import ValidationError

from api.schemas.operators.distribution import (
    DistributionRequest,
    DistributionResponse,
    DistributionValue,
    DistributionBucket
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
# TEST DISTRIBUTION REQUEST - SINGLE MODE
# ============================================================================

class TestDistributionRequestSingleMode:
    """Test DistributionRequest with SINGLE time mode"""
    
    def test_valid_single_mode_year(self):
        """Test valid single mode with year window"""
        request = DistributionRequest(
            dimension=DimensionType.SEVERITY,
            time_mode=TimeMode.SINGLE,
            time_window=TimeWindowYear(value=2025)
        )
        assert request.dimension == DimensionType.SEVERITY
        assert request.time_mode == TimeMode.SINGLE
        assert request.time_window.value == 2025
    
    def test_valid_single_mode_season(self):
        """Test valid single mode with season window"""
        request = DistributionRequest(
            dimension=DimensionType.DOMAIN,
            time_mode=TimeMode.SINGLE,
            time_window=TimeWindowSeason(value="2025-Q1")
        )
        assert request.time_window.value == "2025-Q1"
    
    def test_valid_single_mode_month(self):
        """Test valid single mode with month window"""
        request = DistributionRequest(
            dimension=DimensionType.STAGE,
            time_mode=TimeMode.SINGLE,
            time_window=TimeWindowMonth(value="2025-03")
        )
        assert request.time_window.value == "2025-03"
    
    def test_valid_single_mode_range(self):
        """Test valid single mode with range window"""
        request = DistributionRequest(
            dimension=DimensionType.HARM,
            time_mode=TimeMode.SINGLE,
            time_window=TimeWindowRange(
                from_date=date(2025, 1, 1),
                to_date=date(2025, 6, 30)
            )
        )
        assert request.time_window.from_date == date(2025, 1, 1)
        assert request.time_window.to_date == date(2025, 6, 30)
    
    def test_single_mode_missing_time_window(self):
        """Test that SINGLE mode requires time_window"""
        with pytest.raises(ValidationError) as exc_info:
            DistributionRequest(
                dimension=DimensionType.SEVERITY,
                time_mode=TimeMode.SINGLE
                # Missing time_window
            )
        error_msg = str(exc_info.value)
        assert "time_window" in error_msg.lower()
        # Our custom validation provides the message
        assert "single mode" in error_msg.lower()
    
    def test_single_mode_rejects_multiple_configs(self):
        """Test that SINGLE mode rejects multiple time configs"""
        with pytest.raises(ValidationError) as exc_info:
            DistributionRequest(
                dimension=DimensionType.SEVERITY,
                time_mode=TimeMode.SINGLE,
                time_window=TimeWindowYear(value=2025),
                split_date="2025-01-01"  # Should not be present
            )
        error_msg = str(exc_info.value)
        assert "only" in error_msg.lower()
    
    def test_single_mode_with_filters(self):
        """Test single mode with filters"""
        request = DistributionRequest(
            dimension=DimensionType.SEVERITY,
            time_mode=TimeMode.SINGLE,
            time_window=TimeWindowYear(value=2025),
            filters=OperatorFilters(
                department_id=42,
                severity="High"
            )
        )
        assert request.filters.department_id == 42
        assert request.filters.severity == "High"


# ============================================================================
# TEST DISTRIBUTION REQUEST - MULTI MODE
# ============================================================================

class TestDistributionRequestMultiMode:
    """Test DistributionRequest with MULTI time mode"""
    
    def test_valid_multi_mode_years(self):
        """Test valid multi mode with multiple years"""
        request = DistributionRequest(
            dimension=DimensionType.DOMAIN,
            time_mode=TimeMode.MULTI,
            time_windows=[
                TimeWindowYear(value=2023),
                TimeWindowYear(value=2024),
                TimeWindowYear(value=2025)
            ]
        )
        assert len(request.time_windows) == 3
        assert request.time_windows[0].value == 2023
        assert request.time_windows[2].value == 2025
    
    def test_valid_multi_mode_seasons(self):
        """Test valid multi mode with multiple seasons"""
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
        assert len(request.time_windows) == 4
    
    def test_valid_multi_mode_months(self):
        """Test valid multi mode with multiple months"""
        request = DistributionRequest(
            dimension=DimensionType.SEVERITY,
            time_mode=TimeMode.MULTI,
            time_windows=[
                TimeWindowMonth(value="2025-01"),
                TimeWindowMonth(value="2025-02"),
                TimeWindowMonth(value="2025-03")
            ]
        )
        assert len(request.time_windows) == 3
    
    def test_multi_mode_minimum_windows(self):
        """Test that MULTI mode requires at least 2 windows"""
        # Exactly 2 windows should work
        request = DistributionRequest(
            dimension=DimensionType.DOMAIN,
            time_mode=TimeMode.MULTI,
            time_windows=[
                TimeWindowYear(value=2024),
                TimeWindowYear(value=2025)
            ]
        )
        assert len(request.time_windows) == 2
        
        # Only 1 window should fail
        with pytest.raises(ValidationError) as exc_info:
            DistributionRequest(
                dimension=DimensionType.DOMAIN,
                time_mode=TimeMode.MULTI,
                time_windows=[TimeWindowYear(value=2025)]
            )
        error_msg = str(exc_info.value)
        assert "at least 2" in error_msg.lower()
    
    def test_multi_mode_missing_time_windows(self):
        """Test that MULTI mode requires time_windows"""
        with pytest.raises(ValidationError) as exc_info:
            DistributionRequest(
                dimension=DimensionType.SEVERITY,
                time_mode=TimeMode.MULTI
                # Missing time_windows
            )
        error_msg = str(exc_info.value)
        assert "time_windows" in error_msg.lower()
        # Our custom validation provides the message
        assert "multi mode" in error_msg.lower()
    
    def test_multi_mode_rejects_mixed_window_types(self):
        """Test that MULTI mode rejects mixing different window types"""
        with pytest.raises(ValidationError) as exc_info:
            DistributionRequest(
                dimension=DimensionType.DOMAIN,
                time_mode=TimeMode.MULTI,
                time_windows=[
                    TimeWindowYear(value=2024),
                    TimeWindowSeason(value="2025-Q1")  # Different type!
                ]
            )
        error_msg = str(exc_info.value)
        assert "same type" in error_msg.lower()
    
    def test_multi_mode_rejects_multiple_configs(self):
        """Test that MULTI mode rejects other time configs"""
        with pytest.raises(ValidationError) as exc_info:
            DistributionRequest(
                dimension=DimensionType.SEVERITY,
                time_mode=TimeMode.MULTI,
                time_windows=[
                    TimeWindowYear(value=2024),
                    TimeWindowYear(value=2025)
                ],
                split_date="2025-01-01"  # Should not be present
            )
        error_msg = str(exc_info.value)
        assert "only" in error_msg.lower()


# ============================================================================
# TEST DISTRIBUTION REQUEST - BINARY_SPLIT MODE
# ============================================================================

class TestDistributionRequestBinarySplitMode:
    """Test DistributionRequest with BINARY_SPLIT time mode"""
    
    def test_valid_binary_split(self):
        """Test valid binary split mode"""
        request = DistributionRequest(
            dimension=DimensionType.STAGE,
            time_mode=TimeMode.BINARY_SPLIT,
            split_date="2023-06-01"
        )
        assert request.split_date == "2023-06-01"
    
    def test_binary_split_missing_split_date(self):
        """Test that BINARY_SPLIT mode requires split_date"""
        with pytest.raises(ValidationError) as exc_info:
            DistributionRequest(
                dimension=DimensionType.SEVERITY,
                time_mode=TimeMode.BINARY_SPLIT
                # Missing split_date
            )
        error_msg = str(exc_info.value)
        assert "split_date" in error_msg.lower()
        # Our custom validation provides the message
        assert "binary_split mode" in error_msg.lower()
    
    def test_binary_split_invalid_date_format(self):
        """Test that split_date must be in YYYY-MM-DD format"""
        # These dates fail the pattern regex
        invalid_dates = [
            "2023/06/01",  # Wrong separator
            "01-06-2023",  # Wrong order
            "2023-6-1",    # Missing leading zeros
            "not-a-date"   # Not a date
        ]
        
        for invalid_date in invalid_dates:
            try:
                DistributionRequest(
                    dimension=DimensionType.SEVERITY,
                    time_mode=TimeMode.BINARY_SPLIT,
                    split_date=invalid_date
                )
                # If we reach here, validation did not raise an error
                pytest.fail(f"Expected ValidationError for date '{invalid_date}' but none was raised")
            except ValidationError:
                # This is expected
                pass
    
    def test_binary_split_semantically_invalid_dates(self):
        """Test that dates with correct format but invalid semantics are handled"""
        # These dates pass the regex pattern but are semantically invalid
        # Note: Pydantic's pattern validation doesn't validate semantic correctness
        # These would need additional validation in the service layer if needed
        potentially_invalid_dates = [
            "2023-13-01",  # Month 13 (passes regex)
            "2023-06-32",  # Day 32 (passes regex)
        ]
        
        for test_date in potentially_invalid_dates:
            # The pattern allows these through (they match YYYY-MM-DD)
            # Semantic validation would happen in service/business logic layer
            request = DistributionRequest(
                dimension=DimensionType.SEVERITY,
                time_mode=TimeMode.BINARY_SPLIT,
                split_date=test_date
            )
            assert request.split_date == test_date
    
    def test_binary_split_rejects_multiple_configs(self):
        """Test that BINARY_SPLIT mode rejects other time configs"""
        with pytest.raises(ValidationError) as exc_info:
            DistributionRequest(
                dimension=DimensionType.SEVERITY,
                time_mode=TimeMode.BINARY_SPLIT,
                split_date="2023-06-01",
                time_window=TimeWindowYear(value=2025)  # Should not be present
            )
        error_msg = str(exc_info.value)
        assert "only" in error_msg.lower()


# ============================================================================
# TEST DISTRIBUTION REQUEST - GENERAL
# ============================================================================

class TestDistributionRequestGeneral:
    """General tests for DistributionRequest"""
    
    def test_all_dimension_types_supported(self):
        """Test that all dimension types work"""
        for dimension in DimensionType:
            request = DistributionRequest(
                dimension=dimension,
                time_mode=TimeMode.SINGLE,
                time_window=TimeWindowYear(value=2025)
            )
            assert request.dimension == dimension
    
    def test_reject_extra_fields(self):
        """Test that extra fields are rejected"""
        with pytest.raises(ValidationError) as exc_info:
            DistributionRequest(
                dimension=DimensionType.SEVERITY,
                time_mode=TimeMode.SINGLE,
                time_window=TimeWindowYear(value=2025),
                extra_field="should_fail"
            )
        assert "Extra inputs are not permitted" in str(exc_info.value)
    
    def test_default_empty_filters(self):
        """Test that filters default to empty"""
        request = DistributionRequest(
            dimension=DimensionType.SEVERITY,
            time_mode=TimeMode.SINGLE,
            time_window=TimeWindowYear(value=2025)
        )
        assert isinstance(request.filters, OperatorFilters)
        assert request.filters.department_id is None


# ============================================================================
# TEST DISTRIBUTION VALUE
# ============================================================================

class TestDistributionValue:
    """Test DistributionValue model"""
    
    def test_valid_value(self):
        """Test valid distribution value"""
        value = DistributionValue(
            key="High",
            count=234,
            percent=0.19
        )
        assert value.key == "High"
        assert value.count == 234
        assert value.percent == 0.19
    
    def test_percent_boundaries(self):
        """Test percent boundaries (0.0 to 1.0)"""
        # Valid boundaries
        DistributionValue(key="A", count=0, percent=0.0)
        DistributionValue(key="B", count=100, percent=1.0)
        
        # Invalid boundaries
        with pytest.raises(ValidationError) as exc_info:
            DistributionValue(key="C", count=10, percent=-0.1)
        # Pydantic v2 uses 'greater_than_equal'
        assert "greater than or equal" in str(exc_info.value).lower() or "percent must be between" in str(exc_info.value).lower()
        
        with pytest.raises(ValidationError) as exc_info:
            DistributionValue(key="D", count=10, percent=1.1)
        # Pydantic v2 uses 'less_than_equal'
        assert "less than or equal" in str(exc_info.value).lower() or "percent must be between" in str(exc_info.value).lower()
    
    def test_percent_precision(self):
        """Test that percent is rounded to 6 decimal places"""
        value = DistributionValue(
            key="Test",
            count=1,
            percent=0.123456789
        )
        assert value.percent == 0.123457  # Rounded to 6 decimals
    
    def test_negative_count_rejected(self):
        """Test that negative counts are rejected"""
        with pytest.raises(ValidationError) as exc_info:
            DistributionValue(key="Test", count=-1, percent=0.5)
        assert exc_info.value


# ============================================================================
# TEST DISTRIBUTION BUCKET
# ============================================================================

class TestDistributionBucket:
    """Test DistributionBucket model"""
    
    def test_valid_bucket_with_data(self):
        """Test valid bucket with data"""
        bucket = DistributionBucket(
            time_label="2025",
            total=1000,
            values=[
                DistributionValue(key="Low", count=300, percent=0.3),
                DistributionValue(key="Medium", count=500, percent=0.5),
                DistributionValue(key="High", count=200, percent=0.2)
            ]
        )
        assert bucket.time_label == "2025"
        assert bucket.total == 1000
        assert len(bucket.values) == 3
        assert bucket.status is None
    
    def test_valid_bucket_zero_with_no_data_status(self):
        """Test bucket with total=0 and NO_DATA status"""
        bucket = DistributionBucket(
            time_label="2020-Q1",
            total=0,
            values=[],
            status="NO_DATA"
        )
        assert bucket.total == 0
        assert len(bucket.values) == 0
        assert bucket.status == "NO_DATA"
    
    def test_valid_bucket_zero_without_status(self):
        """Test bucket with total=0 but no status (real zero)"""
        bucket = DistributionBucket(
            time_label="2025",
            total=0,
            values=[],
            status=None
        )
        assert bucket.total == 0
        assert bucket.status is None
    
    def test_bucket_rejects_total_mismatch(self):
        """Test that sum of counts must equal total"""
        with pytest.raises(ValidationError) as exc_info:
            DistributionBucket(
                time_label="2025",
                total=1000,  # Says 1000
                values=[
                    DistributionValue(key="Low", count=300, percent=0.3),
                    DistributionValue(key="High", count=500, percent=0.5)
                    # Sum = 800, not 1000!
                ]
            )
        error_msg = str(exc_info.value)
        assert "sum of counts" in error_msg.lower()
        assert "does not equal total" in error_msg.lower()
    
    def test_bucket_rejects_percent_mismatch(self):
        """Test that sum of percents must equal 1.0"""
        with pytest.raises(ValidationError) as exc_info:
            DistributionBucket(
                time_label="2025",
                total=1000,
                values=[
                    DistributionValue(key="Low", count=300, percent=0.3),
                    DistributionValue(key="High", count=700, percent=0.6)
                    # Sum = 0.9, not 1.0!
                ]
            )
        error_msg = str(exc_info.value)
        assert "sum of percents" in error_msg.lower()
        assert "does not equal 1.0" in error_msg.lower()
    
    def test_bucket_allows_small_floating_point_error(self):
        """Test that small floating point errors are tolerated"""
        # This should pass (0.999999 is within 0.001 tolerance)
        bucket = DistributionBucket(
            time_label="2025",
            total=3,
            values=[
                DistributionValue(key="A", count=1, percent=0.333333),
                DistributionValue(key="B", count=1, percent=0.333333),
                DistributionValue(key="C", count=1, percent=0.333333)
                # Sum = 0.999999, which is close enough
            ]
        )
        assert bucket.total == 3
    
    def test_bucket_rejects_empty_values_when_total_positive(self):
        """Test that values cannot be empty when total > 0"""
        with pytest.raises(ValidationError) as exc_info:
            DistributionBucket(
                time_label="2025",
                total=100,
                values=[]  # Empty but total > 0!
            )
        error_msg = str(exc_info.value)
        assert "cannot be empty when total > 0" in error_msg.lower()
    
    def test_bucket_rejects_values_when_total_zero(self):
        """Test that values must be empty when total = 0"""
        with pytest.raises(ValidationError) as exc_info:
            DistributionBucket(
                time_label="2025",
                total=0,
                values=[
                    DistributionValue(key="Low", count=0, percent=0.0)
                ]  # Has values but total = 0!
            )
        error_msg = str(exc_info.value)
        assert "must be empty when total=0" in error_msg.lower()
    
    def test_bucket_rejects_invalid_status(self):
        """Test that only 'NO_DATA' or None are valid status values"""
        with pytest.raises(ValidationError) as exc_info:
            DistributionBucket(
                time_label="2025",
                total=0,
                values=[],
                status="INVALID_STATUS"
            )
        error_msg = str(exc_info.value)
        # Our custom validator provides the message (case-insensitive)
        assert "no_data" in error_msg.lower() and "none" in error_msg.lower()


# ============================================================================
# TEST DISTRIBUTION RESPONSE
# ============================================================================

class TestDistributionResponse:
    """Test DistributionResponse model"""
    
    def test_valid_single_bucket_response(self):
        """Test valid response with single bucket"""
        response = DistributionResponse(
            dimension="severity",
            time_mode="single",
            buckets=[
                DistributionBucket(
                    time_label="2025",
                    total=1000,
                    values=[
                        DistributionValue(key="Low", count=300, percent=0.3),
                        DistributionValue(key="Medium", count=500, percent=0.5),
                        DistributionValue(key="High", count=200, percent=0.2)
                    ]
                )
            ]
        )
        assert response.dimension == "severity"
        assert response.time_mode == "single"
        assert len(response.buckets) == 1
    
    def test_valid_multi_bucket_response(self):
        """Test valid response with multiple buckets"""
        response = DistributionResponse(
            dimension="domain",
            time_mode="multi",
            buckets=[
                DistributionBucket(
                    time_label="2024",
                    total=500,
                    values=[
                        DistributionValue(key="Clinical", count=250, percent=0.5),
                        DistributionValue(key="Admin", count=250, percent=0.5)
                    ]
                ),
                DistributionBucket(
                    time_label="2025",
                    total=800,
                    values=[
                        DistributionValue(key="Clinical", count=400, percent=0.5),
                        DistributionValue(key="Admin", count=400, percent=0.5)
                    ]
                )
            ]
        )
        assert len(response.buckets) == 2
    
    def test_valid_binary_split_response(self):
        """Test valid response for binary split"""
        response = DistributionResponse(
            dimension="stage",
            time_mode="binary_split",
            buckets=[
                DistributionBucket(
                    time_label="Before",
                    total=0,
                    values=[],
                    status="NO_DATA"
                ),
                DistributionBucket(
                    time_label="After",
                    total=856,
                    values=[
                        DistributionValue(key="Stage 1", count=400, percent=0.467290),
                        DistributionValue(key="Stage 2", count=456, percent=0.532710)
                    ]
                )
            ]
        )
        assert len(response.buckets) == 2
        assert response.buckets[0].status == "NO_DATA"
        assert response.buckets[1].status is None
    
    def test_response_rejects_empty_buckets(self):
        """Test that response must have at least one bucket"""
        with pytest.raises(ValidationError) as exc_info:
            DistributionResponse(
                dimension="severity",
                time_mode="single",
                buckets=[]  # Empty!
            )
        error_msg = str(exc_info.value)
        # Pydantic v2 uses 'at least 1 item' or our custom validator
        assert ("at least" in error_msg.lower() and ("1" in error_msg or "one" in error_msg)) or "bucket" in error_msg.lower()
