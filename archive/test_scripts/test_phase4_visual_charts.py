"""
Phase 4: Visual Charts - Comprehensive Test Suite
Tests chart data formatting functions for frontend visualization
"""

import sys
import os
from pathlib import Path

# Add workspace root to path
workspace_root = Path(__file__).resolve().parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from backend.api.services.training_service import (
    get_db_growth_chart_data,
    get_performance_trends_chart_data,
    get_training_timeline_chart_data,
    get_family_comparison_chart_data
)

# Test counter
_test_count = 0
_passed_count = 0
_failed_count = 0


def test_case(description):
    """Decorator for test cases."""
    def decorator(func):
        def wrapper():
            global _test_count, _passed_count, _failed_count
            _test_count += 1
            print(f"\n[TEST {_test_count}] {description}")
            try:
                func()
                _passed_count += 1
                print(f"✅ PASSED: {description}")
            except AssertionError as e:
                _failed_count += 1
                print(f"❌ FAILED: {description}")
                print(f"   Error: {str(e)}")
            except Exception as e:
                _failed_count += 1
                print(f"❌ FAILED: {description}")
                print(f"   Unexpected error: {str(e)}")
                import traceback
                traceback.print_exc()
        return wrapper
    return decorator


print("=" * 80)
print("PHASE 4: VISUAL CHARTS TEST SUITE")
print("=" * 80)
print()
print("=" * 80)
print("RUNNING TEST SUITE")
print("=" * 80)


# ==================== DB GROWTH CHART TESTS ====================

@test_case("DB growth chart - Structure validation")
def test_db_growth_structure():
    """Test that DB growth chart has correct structure."""
    chart_data = get_db_growth_chart_data(days=30)
    
    # Check top-level keys
    assert "labels" in chart_data, "Should have 'labels'"
    assert "datasets" in chart_data, "Should have 'datasets'"
    assert "metadata" in chart_data, "Should have 'metadata'"
    
    # Check datasets structure
    datasets = chart_data["datasets"]
    assert isinstance(datasets, list), "Datasets should be a list"
    assert len(datasets) > 0, "Should have at least one dataset"
    
    first_dataset = datasets[0]
    required_dataset_keys = ["label", "label_ar", "data", "backgroundColor", "borderColor"]
    for key in required_dataset_keys:
        assert key in first_dataset, f"Dataset should have '{key}'"
    
    # Check metadata structure
    metadata = chart_data["metadata"]
    assert "total_points" in metadata, "Metadata should have total_points"
    assert "date_range" in metadata, "Metadata should have date_range"
    assert "growth" in metadata, "Metadata should have growth"
    
    print(f"   ✓ Chart structure valid")
    print(f"     Labels: {len(chart_data['labels'])}")
    print(f"     Datasets: {len(datasets)}")


@test_case("DB growth chart - Data types")
def test_db_growth_data_types():
    """Test that all data types are correct."""
    chart_data = get_db_growth_chart_data(days=30)
    
    assert isinstance(chart_data["labels"], list), "Labels should be list"
    assert isinstance(chart_data["datasets"], list), "Datasets should be list"
    assert isinstance(chart_data["metadata"], dict), "Metadata should be dict"
    
    # Check data array types
    if len(chart_data["datasets"]) > 0:
        data_array = chart_data["datasets"][0]["data"]
        assert isinstance(data_array, list), "Data should be list"
        if len(data_array) > 0:
            assert isinstance(data_array[0], (int, float)), "Data points should be numeric"
    
    print(f"   ✓ All data types correct")


@test_case("DB growth chart - Bilingual support")
def test_db_growth_bilingual():
    """Test that bilingual labels are present."""
    chart_data = get_db_growth_chart_data(days=30)
    
    if len(chart_data["datasets"]) > 0:
        first_dataset = chart_data["datasets"][0]
        assert "label" in first_dataset, "Should have English label"
        assert "label_ar" in first_dataset, "Should have Arabic label"
        assert len(first_dataset["label_ar"]) > 0, "Arabic label should not be empty"
        
        print(f"   ✓ Bilingual support confirmed")
        print(f"     English: {first_dataset['label']}")
        print(f"     Arabic: {first_dataset['label_ar']}")


@test_case("DB growth chart - Growth calculation")
def test_db_growth_calculation():
    """Test that growth metrics are calculated correctly."""
    chart_data = get_db_growth_chart_data(days=30)
    
    growth = chart_data["metadata"]["growth"]
    
    assert "total" in growth, "Growth should have total"
    assert "percentage" in growth, "Growth should have percentage"
    
    if "first_count" in growth and "last_count" in growth:
        first = growth["first_count"]
        last = growth["last_count"]
        expected_total = last - first
        
        assert growth["total"] == expected_total, f"Total growth should be {expected_total}"
        
        print(f"   ✓ Growth calculation correct")
        print(f"     Total: {growth['total']}")
        print(f"     Percentage: {growth['percentage']}%")


# ==================== PERFORMANCE TRENDS CHART TESTS ====================

@test_case("Performance trends chart - Structure validation")
def test_performance_trends_structure():
    """Test that performance trends chart has correct structure."""
    chart_data = get_performance_trends_chart_data()
    
    assert "labels" in chart_data, "Should have 'labels'"
    assert "datasets" in chart_data, "Should have 'datasets'"
    assert "metadata" in chart_data, "Should have 'metadata'"
    
    # Check metadata
    metadata = chart_data["metadata"]
    assert "total_runs" in metadata, "Metadata should have total_runs"
    assert "families" in metadata, "Metadata should have families"
    
    print(f"   ✓ Chart structure valid")
    print(f"     Total runs: {metadata['total_runs']}")
    print(f"     Families: {metadata['families']}")


@test_case("Performance trends chart - Family datasets")
def test_performance_trends_families():
    """Test that each family has its own dataset."""
    chart_data = get_performance_trends_chart_data()
    
    datasets = chart_data["datasets"]
    
    if len(datasets) > 0:
        # Check that we have datasets for families
        expected_families = ["Hierarchical Models", "Harm Assessment", "Classification Models", "Severity Model"]
        dataset_labels = [d["label"] for d in datasets]
        
        for family in expected_families:
            assert family in dataset_labels, f"Should have dataset for {family}"
        
        # Check dataset structure
        first_dataset = datasets[0]
        assert "label" in first_dataset, "Dataset should have label"
        assert "label_ar" in first_dataset, "Dataset should have Arabic label"
        assert "data" in first_dataset, "Dataset should have data array"
        assert "borderColor" in first_dataset, "Dataset should have border color"
        
        print(f"   ✓ Found {len(datasets)} family datasets")
        print(f"     Families: {', '.join(dataset_labels)}")


@test_case("Performance trends chart - Data consistency")
def test_performance_trends_consistency():
    """Test that all datasets have same length."""
    chart_data = get_performance_trends_chart_data()
    
    labels_count = len(chart_data["labels"])
    
    if len(chart_data["datasets"]) > 0:
        for dataset in chart_data["datasets"]:
            data_count = len(dataset["data"])
            assert data_count == labels_count, f"Dataset data length ({data_count}) should match labels ({labels_count})"
        
        print(f"   ✓ All datasets have consistent length: {labels_count}")


# ==================== TRAINING TIMELINE CHART TESTS ====================

@test_case("Training timeline chart - Structure validation")
def test_training_timeline_structure():
    """Test that training timeline chart has correct structure."""
    chart_data = get_training_timeline_chart_data(limit=20)
    
    assert "labels" in chart_data, "Should have 'labels'"
    assert "datasets" in chart_data, "Should have 'datasets'"
    assert "metadata" in chart_data, "Should have 'metadata'"
    
    # Check datasets
    datasets = chart_data["datasets"]
    assert len(datasets) > 0, "Should have at least one dataset"
    
    first_dataset = datasets[0]
    assert "label" in first_dataset, "Dataset should have label"
    assert "data" in first_dataset, "Dataset should have data"
    assert "backgroundColor" in first_dataset, "Dataset should have background colors"
    
    print(f"   ✓ Chart structure valid")


@test_case("Training timeline chart - Metadata validation")
def test_training_timeline_metadata():
    """Test that metadata contains expected fields."""
    chart_data = get_training_timeline_chart_data(limit=20)
    
    metadata = chart_data["metadata"]
    
    required_fields = ["total_runs", "avg_duration", "success_rate", "successful_runs", "failed_runs"]
    for field in required_fields:
        assert field in metadata, f"Metadata should have '{field}'"
    
    # Validate numbers
    assert isinstance(metadata["avg_duration"], (int, float)), "avg_duration should be numeric"
    assert isinstance(metadata["success_rate"], (int, float)), "success_rate should be numeric"
    assert metadata["success_rate"] >= 0, "success_rate should be non-negative"
    assert metadata["success_rate"] <= 100, "success_rate should not exceed 100"
    
    print(f"   ✓ Metadata valid")
    print(f"     Avg Duration: {metadata['avg_duration']}s")
    print(f"     Success Rate: {metadata['success_rate']}%")


@test_case("Training timeline chart - Duration data")
def test_training_timeline_durations():
    """Test that duration data is valid."""
    chart_data = get_training_timeline_chart_data(limit=20)
    
    if len(chart_data["datasets"]) > 0:
        durations = chart_data["datasets"][0]["data"]
        
        # All durations should be non-negative
        for duration in durations:
            assert duration >= 0, f"Duration should be non-negative, got {duration}"
        
        print(f"   ✓ All durations valid")
        if len(durations) > 0:
            print(f"     Min: {min(durations)}s, Max: {max(durations)}s")


# ==================== FAMILY COMPARISON CHART TESTS ====================

@test_case("Family comparison chart - Structure validation")
def test_family_comparison_structure():
    """Test that family comparison chart has correct structure."""
    chart_data = get_family_comparison_chart_data()
    
    assert "labels" in chart_data, "Should have 'labels'"
    assert "labels_ar" in chart_data, "Should have 'labels_ar'"
    assert "datasets" in chart_data, "Should have 'datasets'"
    assert "metadata" in chart_data, "Should have 'metadata'"
    
    # Check labels match
    assert len(chart_data["labels"]) == len(chart_data["labels_ar"]), "Labels and labels_ar should have same length"
    
    print(f"   ✓ Chart structure valid")
    print(f"     Families: {len(chart_data['labels'])}")


@test_case("Family comparison chart - Metric datasets")
def test_family_comparison_metrics():
    """Test that all metrics have datasets."""
    chart_data = get_family_comparison_chart_data()
    
    datasets = chart_data["datasets"]
    
    expected_metrics = ["F1 Score", "Accuracy", "Precision", "Recall"]
    dataset_labels = [d["label"] for d in datasets]
    
    for metric in expected_metrics:
        assert metric in dataset_labels, f"Should have dataset for {metric}"
    
    # Check each dataset has bilingual labels
    for dataset in datasets:
        assert "label" in dataset, "Dataset should have English label"
        assert "label_ar" in dataset, "Dataset should have Arabic label"
        assert len(dataset["label_ar"]) > 0, "Arabic label should not be empty"
    
    print(f"   ✓ All {len(expected_metrics)} metrics present")
    print(f"     Metrics: {', '.join(expected_metrics)}")


@test_case("Family comparison chart - Data consistency")
def test_family_comparison_consistency():
    """Test that all metric datasets have same length."""
    chart_data = get_family_comparison_chart_data()
    
    family_count = len(chart_data["labels"])
    
    for dataset in chart_data["datasets"]:
        data_count = len(dataset["data"])
        assert data_count == family_count, f"Dataset should have {family_count} data points, got {data_count}"
    
    print(f"   ✓ All datasets have {family_count} data points")


@test_case("Family comparison chart - Metric value ranges")
def test_family_comparison_ranges():
    """Test that metric values are in valid ranges."""
    chart_data = get_family_comparison_chart_data()
    
    for dataset in chart_data["datasets"]:
        metric_name = dataset["label"]
        data = dataset["data"]
        
        for value in data:
            assert 0 <= value <= 1, f"{metric_name} value should be between 0 and 1, got {value}"
    
    print(f"   ✓ All metric values in valid range [0, 1]")


@test_case("Family comparison chart - Metadata")
def test_family_comparison_metadata():
    """Test that metadata is complete."""
    chart_data = get_family_comparison_chart_data()
    
    metadata = chart_data["metadata"]
    
    assert "total_families" in metadata, "Metadata should have total_families"
    assert "metrics_shown" in metadata, "Metadata should have metrics_shown"
    assert "record_counts" in metadata, "Metadata should have record_counts"
    
    # Check record counts match families
    family_count = len(chart_data["labels"])
    record_count_len = len(metadata["record_counts"])
    assert record_count_len == family_count, f"Record counts should match families ({family_count}), got {record_count_len}"
    
    print(f"   ✓ Metadata complete")
    print(f"     Total Families: {metadata['total_families']}")
    print(f"     Metrics Shown: {len(metadata['metrics_shown'])}")


# ==================== CHART STYLING TESTS ====================

@test_case("Chart styling - Color consistency")
def test_chart_colors():
    """Test that charts have proper color styling."""
    
    # Test DB growth chart
    db_chart = get_db_growth_chart_data(days=30)
    if len(db_chart["datasets"]) > 0:
        dataset = db_chart["datasets"][0]
        assert "backgroundColor" in dataset, "Should have background color"
        assert "borderColor" in dataset, "Should have border color"
        assert dataset["backgroundColor"].startswith("rgba("), "Background color should be RGBA"
    
    # Test family comparison chart
    family_chart = get_family_comparison_chart_data()
    for dataset in family_chart["datasets"]:
        assert "backgroundColor" in dataset, "Should have background color"
        assert "borderColor" in dataset, "Should have border color"
    
    print(f"   ✓ All charts have proper color styling")


@test_case("Chart styling - Border widths")
def test_chart_borders():
    """Test that charts have border width defined."""
    
    # Test all chart types
    charts = [
        get_db_growth_chart_data(days=30),
        get_performance_trends_chart_data(),
        get_training_timeline_chart_data(limit=20),
        get_family_comparison_chart_data()
    ]
    
    for chart_data in charts:
        for dataset in chart_data["datasets"]:
            assert "borderWidth" in dataset, "Dataset should have borderWidth"
            assert isinstance(dataset["borderWidth"], int), "borderWidth should be integer"
    
    print(f"   ✓ All datasets have border width defined")


# ==================== INTEGRATION TESTS ====================

@test_case("Integration - All charts return data")
def test_all_charts_return_data():
    """Test that all chart endpoints return data."""
    
    charts = {
        "DB Growth": get_db_growth_chart_data(days=30),
        "Performance Trends": get_performance_trends_chart_data(),
        "Training Timeline": get_training_timeline_chart_data(limit=20),
        "Family Comparison": get_family_comparison_chart_data()
    }
    
    for name, chart_data in charts.items():
        assert chart_data is not None, f"{name} should return data"
        assert isinstance(chart_data, dict), f"{name} should return dict"
        assert "labels" in chart_data, f"{name} should have labels"
        assert "datasets" in chart_data, f"{name} should have datasets"
    
    print(f"   ✓ All {len(charts)} chart endpoints return valid data")


@test_case("Integration - Chart data is JSON serializable")
def test_json_serializable():
    """Test that all chart data can be serialized to JSON."""
    import json
    
    charts = [
        get_db_growth_chart_data(days=30),
        get_performance_trends_chart_data(),
        get_training_timeline_chart_data(limit=20),
        get_family_comparison_chart_data()
    ]
    
    for chart_data in charts:
        try:
            json_str = json.dumps(chart_data, ensure_ascii=False)
            assert len(json_str) > 0, "JSON string should not be empty"
        except (TypeError, ValueError) as e:
            raise AssertionError(f"Chart data is not JSON serializable: {e}")
    
    print(f"   ✓ All charts are JSON serializable")


# Run tests
test_db_growth_structure()
test_db_growth_data_types()
test_db_growth_bilingual()
test_db_growth_calculation()

test_performance_trends_structure()
test_performance_trends_families()
test_performance_trends_consistency()

test_training_timeline_structure()
test_training_timeline_metadata()
test_training_timeline_durations()

test_family_comparison_structure()
test_family_comparison_metrics()
test_family_comparison_consistency()
test_family_comparison_ranges()
test_family_comparison_metadata()

test_chart_colors()
test_chart_borders()

test_all_charts_return_data()
test_json_serializable()


# Print summary
print()
print("=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"Total Tests: {_test_count}")
print(f"✅ Passed: {_passed_count}")
print(f"❌ Failed: {_failed_count}")
print(f"Success Rate: {(_passed_count / _test_count * 100):.1f}%")
print("=" * 80)

if _failed_count > 0:
    print()
    print("⚠️ Some tests failed. Please review and fix.")
    sys.exit(1)
else:
    print()
    print("🎉 ALL TESTS PASSED!")
    sys.exit(0)
