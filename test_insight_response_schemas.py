"""
Test B-I19: Insight Response Schemas

Verifies:
1. All 6 Pydantic models are correctly defined
2. Models accept correct data types
3. Models reject invalid data
4. datetime handling works correctly
5. Union types (str | int) work correctly
6. Nested models work correctly
7. list fields work correctly
8. No validators or computed fields (pure shape definitions)
"""

import pytest
from datetime import datetime
from pydantic import ValidationError, BaseModel
import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.api_v2.schemas.insight_schemas import (
    KPIStatusCount,
    KPIActionItemSummary,
    KPISummaryResponse,
    DistributionItem,
    TrendItem,
    StuckItem
)


class TestKPIStatusCount:
    """Test KPIStatusCount model."""
    
    def test_valid_kpi_status_count(self):
        """Test 1: Valid KPIStatusCount creation."""
        item = KPIStatusCount(status="Open", count=42)
        assert item.status == "Open"
        assert item.count == 42
    
    def test_kpi_status_count_fields_exist(self):
        """Test 2: KPIStatusCount has required fields."""
        item = KPIStatusCount(status="Closed", count=10)
        assert hasattr(item, 'status')
        assert hasattr(item, 'count')
    
    def test_kpi_status_count_rejects_missing_status(self):
        """Test 3: KPIStatusCount rejects missing status."""
        with pytest.raises(ValidationError):
            KPIStatusCount(count=10)
    
    def test_kpi_status_count_rejects_missing_count(self):
        """Test 4: KPIStatusCount rejects missing count."""
        with pytest.raises(ValidationError):
            KPIStatusCount(status="Open")
    
    def test_kpi_status_count_rejects_invalid_count(self):
        """Test 5: KPIStatusCount rejects non-integer count."""
        with pytest.raises(ValidationError):
            KPIStatusCount(status="Open", count="not a number")


class TestKPIActionItemSummary:
    """Test KPIActionItemSummary model."""
    
    def test_valid_action_item_summary(self):
        """Test 6: Valid KPIActionItemSummary creation."""
        summary = KPIActionItemSummary(
            total=100,
            open=50,
            completed=30,
            overdue=20
        )
        assert summary.total == 100
        assert summary.open == 50
        assert summary.completed == 30
        assert summary.overdue == 20
    
    def test_action_item_summary_all_fields_exist(self):
        """Test 7: KPIActionItemSummary has all required fields."""
        summary = KPIActionItemSummary(total=10, open=5, completed=3, overdue=2)
        assert hasattr(summary, 'total')
        assert hasattr(summary, 'open')
        assert hasattr(summary, 'completed')
        assert hasattr(summary, 'overdue')
    
    def test_action_item_summary_rejects_missing_field(self):
        """Test 8: KPIActionItemSummary rejects missing fields."""
        with pytest.raises(ValidationError):
            KPIActionItemSummary(total=10, open=5, completed=3)  # Missing overdue
    
    def test_action_item_summary_rejects_invalid_types(self):
        """Test 9: KPIActionItemSummary rejects invalid types."""
        with pytest.raises(ValidationError):
            KPIActionItemSummary(
                total=[],  # Invalid type (list, not int)
                open=50,
                completed=30,
                overdue=20
            )


class TestKPISummaryResponse:
    """Test KPISummaryResponse model."""
    
    def test_valid_kpi_summary_response(self):
        """Test 10: Valid KPISummaryResponse creation."""
        response = KPISummaryResponse(
            total_subcases=150,
            by_status=[
                KPIStatusCount(status="Open", count=50),
                KPIStatusCount(status="Closed", count=100)
            ],
            action_items=KPIActionItemSummary(
                total=200,
                open=80,
                completed=100,
                overdue=20
            )
        )
        assert response.total_subcases == 150
        assert len(response.by_status) == 2
        assert response.action_items.total == 200
    
    def test_kpi_summary_response_nested_models(self):
        """Test 11: KPISummaryResponse correctly nests models."""
        response = KPISummaryResponse(
            total_subcases=100,
            by_status=[KPIStatusCount(status="Open", count=100)],
            action_items=KPIActionItemSummary(total=50, open=25, completed=20, overdue=5)
        )
        assert isinstance(response.by_status[0], KPIStatusCount)
        assert isinstance(response.action_items, KPIActionItemSummary)
    
    def test_kpi_summary_response_empty_status_list(self):
        """Test 12: KPISummaryResponse accepts empty status list."""
        response = KPISummaryResponse(
            total_subcases=0,
            by_status=[],
            action_items=KPIActionItemSummary(total=0, open=0, completed=0, overdue=0)
        )
        assert response.by_status == []
    
    def test_kpi_summary_response_rejects_missing_field(self):
        """Test 13: KPISummaryResponse rejects missing fields."""
        with pytest.raises(ValidationError):
            KPISummaryResponse(
                total_subcases=100,
                by_status=[]
                # Missing action_items
            )
    
    def test_kpi_summary_response_accepts_dict_input(self):
        """Test 14: KPISummaryResponse accepts dict for nested models."""
        response = KPISummaryResponse(
            total_subcases=100,
            by_status=[{"status": "Open", "count": 50}],
            action_items={"total": 20, "open": 10, "completed": 8, "overdue": 2}
        )
        assert isinstance(response.by_status[0], KPIStatusCount)
        assert isinstance(response.action_items, KPIActionItemSummary)


class TestDistributionItem:
    """Test DistributionItem model."""
    
    def test_valid_distribution_item_with_string_key(self):
        """Test 15: Valid DistributionItem with string key."""
        item = DistributionItem(key="Dr. Smith", count=25)
        assert item.key == "Dr. Smith"
        assert item.count == 25
    
    def test_valid_distribution_item_with_int_key(self):
        """Test 16: Valid DistributionItem with int key."""
        item = DistributionItem(key=101, count=30)
        assert item.key == 101
        assert item.count == 30
    
    def test_distribution_item_union_type_str(self):
        """Test 17: DistributionItem key accepts str (union type)."""
        item = DistributionItem(key="Category A", count=10)
        assert isinstance(item.key, str)
    
    def test_distribution_item_union_type_int(self):
        """Test 18: DistributionItem key accepts int (union type)."""
        item = DistributionItem(key=42, count=10)
        assert isinstance(item.key, int)
    
    def test_distribution_item_rejects_invalid_count(self):
        """Test 19: DistributionItem rejects invalid count."""
        with pytest.raises(ValidationError):
            DistributionItem(key="Test", count="not_int")


class TestTrendItem:
    """Test TrendItem model."""
    
    def test_valid_trend_item(self):
        """Test 20: Valid TrendItem creation."""
        item = TrendItem(bucket="2026-01", count=45)
        assert item.bucket == "2026-01"
        assert item.count == 45
    
    def test_trend_item_fields_exist(self):
        """Test 21: TrendItem has required fields."""
        item = TrendItem(bucket="2026-W05", count=12)
        assert hasattr(item, 'bucket')
        assert hasattr(item, 'count')
    
    def test_trend_item_accepts_day_bucket(self):
        """Test 22: TrendItem accepts day bucket format."""
        item = TrendItem(bucket="2026-02-03", count=5)
        assert item.bucket == "2026-02-03"
    
    def test_trend_item_accepts_week_bucket(self):
        """Test 23: TrendItem accepts week bucket format."""
        item = TrendItem(bucket="2026-W05", count=10)
        assert item.bucket == "2026-W05"
    
    def test_trend_item_accepts_month_bucket(self):
        """Test 24: TrendItem accepts month bucket format."""
        item = TrendItem(bucket="2026-02", count=15)
        assert item.bucket == "2026-02"
    
    def test_trend_item_rejects_missing_bucket(self):
        """Test 25: TrendItem rejects missing bucket."""
        with pytest.raises(ValidationError):
            TrendItem(count=10)


class TestStuckItem:
    """Test StuckItem model."""
    
    def test_valid_stuck_item(self):
        """Test 26: Valid StuckItem creation."""
        item = StuckItem(
            subcase_id=12345,
            status="Open",
            target_org_unit_id=101,
            updated_at=datetime(2026, 1, 1, 12, 0, 0),
            days_in_stage=45
        )
        assert item.subcase_id == 12345
        assert item.status == "Open"
        assert item.target_org_unit_id == 101
        assert item.updated_at == datetime(2026, 1, 1, 12, 0, 0)
        assert item.days_in_stage == 45
    
    def test_stuck_item_all_fields_exist(self):
        """Test 27: StuckItem has all required fields."""
        item = StuckItem(
            subcase_id=1,
            status="Pending",
            target_org_unit_id=2,
            updated_at=datetime.now(),
            days_in_stage=30
        )
        assert hasattr(item, 'subcase_id')
        assert hasattr(item, 'status')
        assert hasattr(item, 'target_org_unit_id')
        assert hasattr(item, 'updated_at')
        assert hasattr(item, 'days_in_stage')
    
    def test_stuck_item_datetime_field(self):
        """Test 28: StuckItem updated_at is datetime."""
        dt = datetime(2026, 2, 3, 10, 30, 0)
        item = StuckItem(
            subcase_id=1,
            status="Open",
            target_org_unit_id=2,
            updated_at=dt,
            days_in_stage=10
        )
        assert isinstance(item.updated_at, datetime)
        assert item.updated_at == dt
    
    def test_stuck_item_rejects_invalid_datetime(self):
        """Test 29: StuckItem rejects invalid datetime."""
        with pytest.raises(ValidationError):
            StuckItem(
                subcase_id=1,
                status="Open",
                target_org_unit_id=2,
                updated_at="not a datetime",  # Invalid
                days_in_stage=10
            )
    
    def test_stuck_item_rejects_missing_field(self):
        """Test 30: StuckItem rejects missing fields."""
        with pytest.raises(ValidationError):
            StuckItem(
                subcase_id=1,
                status="Open",
                target_org_unit_id=2,
                updated_at=datetime.now()
                # Missing days_in_stage
            )


class TestSchemaIntegration:
    """Test schema integration and overall design."""
    
    def test_all_models_importable(self):
        """Test 31: All models are importable."""
        # If imports worked at top of file, this passes
        assert KPIStatusCount is not None
        assert KPIActionItemSummary is not None
        assert KPISummaryResponse is not None
        assert DistributionItem is not None
        assert TrendItem is not None
        assert StuckItem is not None
    
    def test_models_are_pydantic_base_models(self):
        """Test 32: All models inherit from BaseModel."""
        assert issubclass(KPIStatusCount, BaseModel)
        assert issubclass(KPIActionItemSummary, BaseModel)
        assert issubclass(KPISummaryResponse, BaseModel)
        assert issubclass(DistributionItem, BaseModel)
        assert issubclass(TrendItem, BaseModel)
        assert issubclass(StuckItem, BaseModel)
    
    def test_kpi_summary_response_json_serialization(self):
        """Test 33: KPISummaryResponse can be serialized to JSON."""
        response = KPISummaryResponse(
            total_subcases=100,
            by_status=[
                KPIStatusCount(status="Open", count=50),
                KPIStatusCount(status="Closed", count=50)
            ],
            action_items=KPIActionItemSummary(total=75, open=30, completed=40, overdue=5)
        )
        json_data = response.model_dump()
        assert json_data['total_subcases'] == 100
        assert len(json_data['by_status']) == 2
        assert json_data['action_items']['total'] == 75
    
    def test_stuck_item_json_serialization_with_datetime(self):
        """Test 34: StuckItem can be serialized to JSON with datetime."""
        dt = datetime(2026, 2, 3, 12, 0, 0)
        item = StuckItem(
            subcase_id=123,
            status="Open",
            target_org_unit_id=10,
            updated_at=dt,
            days_in_stage=30
        )
        json_data = item.model_dump()
        assert json_data['subcase_id'] == 123
        assert json_data['updated_at'] == dt
    
    def test_distribution_list_serialization(self):
        """Test 35: List of DistributionItem can be serialized."""
        items = [
            DistributionItem(key="A", count=10),
            DistributionItem(key="B", count=20),
            DistributionItem(key=123, count=30)
        ]
        json_data = [item.model_dump() for item in items]
        assert len(json_data) == 3
        assert json_data[0]['key'] == "A"
        assert json_data[2]['key'] == 123
    
    def test_trend_list_serialization(self):
        """Test 36: List of TrendItem can be serialized."""
        items = [
            TrendItem(bucket="2026-01", count=100),
            TrendItem(bucket="2026-02", count=120)
        ]
        json_data = [item.model_dump() for item in items]
        assert len(json_data) == 2
        assert json_data[0]['bucket'] == "2026-01"
    
    def test_no_validators_on_models(self):
        """Test 37: Models have no custom validators (pure shape)."""
        # Check that models accept any valid data without extra validation
        # This is a design verification - models should be pure shapes
        
        # KPIStatusCount accepts any status string
        item1 = KPIStatusCount(status="InvalidStatus", count=0)
        assert item1.status == "InvalidStatus"
        
        # TrendItem accepts any bucket string
        item2 = TrendItem(bucket="invalid-format", count=0)
        assert item2.bucket == "invalid-format"
        
        # No ValueError should be raised for "invalid" but type-correct data
        assert True  # If we got here, no validators are blocking
    
    def test_kpi_summary_response_with_many_statuses(self):
        """Test 38: KPISummaryResponse handles multiple status items."""
        statuses = [
            KPIStatusCount(status="Open", count=10),
            KPIStatusCount(status="In Progress", count=20),
            KPIStatusCount(status="Pending", count=15),
            KPIStatusCount(status="Closed", count=100),
        ]
        response = KPISummaryResponse(
            total_subcases=145,
            by_status=statuses,
            action_items=KPIActionItemSummary(total=50, open=20, completed=25, overdue=5)
        )
        assert len(response.by_status) == 4
    
    def test_distribution_item_key_coercion(self):
        """Test 39: DistributionItem key type is preserved."""
        item_str = DistributionItem(key="test", count=1)
        item_int = DistributionItem(key=42, count=2)
        
        assert type(item_str.key) == str
        assert type(item_int.key) == int
    
    def test_stuck_item_accepts_iso_datetime_string(self):
        """Test 40: StuckItem accepts ISO datetime string."""
        item = StuckItem(
            subcase_id=1,
            status="Open",
            target_org_unit_id=2,
            updated_at="2026-02-03T12:00:00",  # ISO format
            days_in_stage=10
        )
        assert isinstance(item.updated_at, datetime)
        assert item.updated_at.year == 2026
        assert item.updated_at.month == 2
        assert item.updated_at.day == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
