"""
Test Suite: Base Operator Schemas

Tests for shared enums, time models, and filters.
Ensures type safety and validation rules work correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import pytest
from datetime import date
from pydantic import ValidationError

from api.schemas.operators.base import (
    DimensionType,
    TimeMode,
    TimeWindowType,
    TimeWindowYear,
    TimeWindowSeason,
    TimeWindowMonth,
    TimeWindowRange,
    OperatorFilters
)


# ============================================================================
# TEST ENUMS
# ============================================================================

class TestDimensionType:
    """Test DimensionType enum"""
    
    def test_all_dimension_values(self):
        """Test all valid dimension types"""
        assert DimensionType.DOMAIN.value == "domain"
        assert DimensionType.CATEGORY.value == "category"
        assert DimensionType.SUBCATEGORY.value == "subcategory"
        assert DimensionType.CLASSIFICATION.value == "classification"
        assert DimensionType.STAGE.value == "stage"
        assert DimensionType.SEVERITY.value == "severity"
        assert DimensionType.HARM.value == "harm"
    
    def test_dimension_count(self):
        """Test we have exactly 7 dimensions"""
        assert len(DimensionType) == 7


class TestTimeMode:
    """Test TimeMode enum"""
    
    def test_all_time_modes(self):
        """Test all valid time modes"""
        assert TimeMode.SINGLE.value == "single"
        assert TimeMode.MULTI.value == "multi"
        assert TimeMode.BINARY_SPLIT.value == "binary_split"
    
    def test_time_mode_count(self):
        """Test we have exactly 3 time modes"""
        assert len(TimeMode) == 3


class TestTimeWindowType:
    """Test TimeWindowType enum"""
    
    def test_all_window_types(self):
        """Test all valid window types"""
        assert TimeWindowType.YEAR.value == "year"
        assert TimeWindowType.SEASON.value == "season"
        assert TimeWindowType.MONTH.value == "month"
        assert TimeWindowType.RANGE.value == "range"
    
    def test_window_type_count(self):
        """Test we have exactly 4 window types"""
        assert len(TimeWindowType) == 4


# ============================================================================
# TEST TIME WINDOW MODELS
# ============================================================================

class TestTimeWindowYear:
    """Test TimeWindowYear model"""
    
    def test_valid_year(self):
        """Test valid year creation"""
        window = TimeWindowYear(value=2025)
        assert window.type == TimeWindowType.YEAR
        assert window.value == 2025
        assert window.get_label() == "2025"
    
    def test_year_boundaries(self):
        """Test year boundary values"""
        # Valid boundaries
        TimeWindowYear(value=2000)
        TimeWindowYear(value=2100)
        
        # Invalid boundaries
        with pytest.raises(ValidationError) as exc_info:
            TimeWindowYear(value=1999)
        # Pydantic 2.x uses 'greater_than_equal' error
        assert "greater than or equal to 2000" in str(exc_info.value) or "Year must be between" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            TimeWindowYear(value=2101)
        # Pydantic 2.x uses 'less_than_equal' error
        assert "less than or equal to 2100" in str(exc_info.value) or "Year must be between" in str(exc_info.value)


class TestTimeWindowSeason:
    """Test TimeWindowSeason model"""
    
    def test_valid_quarters(self):
        """Test valid quarter seasons"""
        for quarter in ["Q1", "Q2", "Q3", "Q4"]:
            window = TimeWindowSeason(value=f"2025-{quarter}")
            assert window.type == TimeWindowType.SEASON
            assert window.value == f"2025-{quarter}"
            assert window.get_label() == f"2025-{quarter}"
    
    def test_valid_trimesters(self):
        """Test valid trimester seasons"""
        for trimester in ["T1", "T2", "T3"]:
            window = TimeWindowSeason(value=f"2025-{trimester}")
            assert window.type == TimeWindowType.SEASON
            assert window.value == f"2025-{trimester}"
            assert window.get_label() == f"2025-{trimester}"
    
    def test_invalid_season_formats(self):
        """Test invalid season formats"""
        invalid_formats = [
            "2025-Q5",  # Invalid quarter
            "2025-Q0",  # Invalid quarter
            "2025-T4",  # Invalid trimester
            "2025-T0",  # Invalid trimester
            "2025Q1",   # Missing dash
            "Q1-2025",  # Wrong order
            "2025-1",   # Just number
            "2025",     # Missing season
        ]
        
        for invalid_format in invalid_formats:
            with pytest.raises(ValidationError) as exc_info:
                TimeWindowSeason(value=invalid_format)
            # Pydantic 2.x uses 'string_pattern_mismatch' error
            assert "pattern" in str(exc_info.value).lower() or "season must be in format" in str(exc_info.value).lower()


class TestTimeWindowMonth:
    """Test TimeWindowMonth model"""
    
    def test_valid_months(self):
        """Test valid month formats"""
        for month in range(1, 13):
            month_str = f"{month:02d}"
            window = TimeWindowMonth(value=f"2025-{month_str}")
            assert window.type == TimeWindowType.MONTH
            assert window.value == f"2025-{month_str}"
            assert window.get_label() == f"2025-{month_str}"
    
    def test_invalid_month_formats(self):
        """Test invalid month formats"""
        invalid_formats = [
            "2025-13",     # Invalid month
            "2025-00",     # Invalid month
            "2025-1",      # Missing leading zero
            "202501",      # Missing dash
            "01-2025",     # Wrong order
            "2025-JAN",    # Text month
        ]
        
        for invalid_format in invalid_formats:
            with pytest.raises(ValidationError) as exc_info:
                TimeWindowMonth(value=invalid_format)
            # Pydantic 2.x uses 'string_pattern_mismatch' error
            assert "pattern" in str(exc_info.value).lower() or "month must be in format" in str(exc_info.value).lower()


class TestTimeWindowRange:
    """Test TimeWindowRange model"""
    
    def test_valid_range(self):
        """Test valid date range"""
        window = TimeWindowRange(
            from_date=date(2025, 1, 1),
            to_date=date(2025, 12, 31)
        )
        assert window.type == TimeWindowType.RANGE
        assert window.from_date == date(2025, 1, 1)
        assert window.to_date == date(2025, 12, 31)
        assert window.get_label() == "2025-01-01 to 2025-12-31"
    
    def test_same_date_range(self):
        """Test range with same from and to date"""
        window = TimeWindowRange(
            from_date=date(2025, 6, 15),
            to_date=date(2025, 6, 15)
        )
        assert window.from_date == window.to_date
    
    def test_invalid_range_reversed(self):
        """Test that from_date must be before to_date"""
        with pytest.raises(ValidationError) as exc_info:
            TimeWindowRange(
                from_date=date(2025, 12, 31),
                to_date=date(2025, 1, 1)
            )
        assert "from_date must be before or equal to to_date" in str(exc_info.value)


# ============================================================================
# TEST OPERATOR FILTERS
# ============================================================================

class TestOperatorFilters:
    """Test OperatorFilters model"""
    
    def test_empty_filters(self):
        """Test creating empty filters"""
        filters = OperatorFilters()
        assert filters.org_unit_id is None
        assert filters.department_id is None
        assert filters.domain is None
    
    def test_organizational_filters(self):
        """Test organizational unit filters"""
        filters = OperatorFilters(
            org_unit_id=1,
            administration_id=2,
            department_id=3,
            section_id=4
        )
        assert filters.org_unit_id == 1
        assert filters.administration_id == 2
        assert filters.department_id == 3
        assert filters.section_id == 4
    
    def test_dimension_filters(self):
        """Test dimension-based filters"""
        filters = OperatorFilters(
            domain="Clinical",
            category="Medication",
            severity="High"
        )
        assert filters.domain == "Clinical"
        assert filters.category == "Medication"
        assert filters.severity == "High"
    
    def test_mixed_filters(self):
        """Test combination of organizational and dimension filters"""
        filters = OperatorFilters(
            department_id=42,
            severity="High",
            harm="Moderate"
        )
        assert filters.department_id == 42
        assert filters.severity == "High"
        assert filters.harm == "Moderate"
    
    def test_reject_extra_fields(self):
        """Test that extra fields are rejected"""
        with pytest.raises(ValidationError) as exc_info:
            OperatorFilters(
                department_id=1,
                invalid_field="should_fail"
            )
        assert "Extra inputs are not permitted" in str(exc_info.value)
    
    def test_all_filter_fields_optional(self):
        """Test that all fields are optional"""
        filters = OperatorFilters()
        
        # Check all fields exist and are None
        assert hasattr(filters, 'org_unit_id')
        assert hasattr(filters, 'administration_id')
        assert hasattr(filters, 'department_id')
        assert hasattr(filters, 'section_id')
        assert hasattr(filters, 'domain')
        assert hasattr(filters, 'category')
        assert hasattr(filters, 'subcategory')
        assert hasattr(filters, 'classification')
        assert hasattr(filters, 'stage')
        assert hasattr(filters, 'severity')
        assert hasattr(filters, 'harm')
