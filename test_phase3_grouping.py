"""
TEST PHASE 3: MODEL GROUPING & AGGREGATION
===========================================
Comprehensive test suite for model family grouping, aggregated metrics,
and performance alert detection.

Tests:
1. Model family definitions
2. Metric calculation and aggregation
3. Performance alert detection
4. Grouped status generation
5. API endpoint response structure
"""

import sys
from pathlib import Path

workspace_root = Path(__file__).resolve().parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from backend.api.services.training_service import (
    MODEL_FAMILIES,
    _calculate_family_metrics,
    _detect_performance_alerts,
    get_grouped_training_status
)

print("\n" + "="*80)
print("PHASE 3: MODEL GROUPING & AGGREGATION TEST SUITE")
print("="*80)

# Test counters
tests_passed = 0
tests_failed = 0
total_tests = 0


def test_case(name: str):
    """Decorator for test cases."""
    def decorator(func):
        def wrapper():
            global tests_passed, tests_failed, total_tests
            total_tests += 1
            print(f"\n[TEST {total_tests}] {name}")
            try:
                func()
                print(f"✅ PASSED: {name}")
                tests_passed += 1
                return True
            except AssertionError as e:
                print(f"❌ FAILED: {name}")
                print(f"   Error: {e}")
                tests_failed += 1
                return False
            except Exception as e:
                print(f"❌ ERROR: {name}")
                print(f"   Exception: {e}")
                tests_failed += 1
                return False
        return wrapper
    return decorator


# ==================== TEST SUITE ====================

@test_case("Model family definitions exist")
def test_family_definitions():
    """Test that all model families are defined."""
    expected_families = ["hierarchical", "harm_assessment", "classification", "severity"]
    
    for family_key in expected_families:
        assert family_key in MODEL_FAMILIES, f"Missing family: {family_key}"
    
    print(f"   ✓ All {len(expected_families)} families defined")


@test_case("Family structure validation")
def test_family_structure():
    """Test that each family has required fields."""
    required_fields = ["name", "name_ar", "models"]
    
    for family_key, family_info in MODEL_FAMILIES.items():
        for field in required_fields:
            assert field in family_info, f"Family {family_key} missing field: {field}"
        
        assert isinstance(family_info["models"], list), \
            f"Family {family_key} models should be a list"
        assert len(family_info["models"]) > 0, \
            f"Family {family_key} should have at least one model"
    
    print(f"   ✓ All families have valid structure")


@test_case("Model count validation")
def test_model_count():
    """Test that all 18 models are accounted for."""
    all_models = []
    for family_info in MODEL_FAMILIES.values():
        all_models.extend(family_info["models"])
    
    # Should have 18 unique models
    assert len(all_models) == 18, f"Expected 18 models, got {len(all_models)}"
    assert len(set(all_models)) == 18, f"Duplicate models found"
    
    print(f"   ✓ All 18 models accounted for across families")


@test_case("Calculate family metrics - Empty list")
def test_metrics_empty():
    """Test metric calculation with empty model list."""
    metrics = _calculate_family_metrics([])
    
    assert metrics["avg_f1"] == 0.0, "Empty list should have 0 avg_f1"
    assert metrics["avg_accuracy"] == 0.0, "Empty list should have 0 avg_accuracy"
    assert metrics["total_records"] == 0, "Empty list should have 0 total_records"
    
    print(f"   ✓ Empty list returns zero metrics")


@test_case("Calculate family metrics - Single model")
def test_metrics_single():
    """Test metric calculation with single model."""
    models = [
        {
            "model_name": "Test_Model",
            "num_records": 100,
            "accuracy": 0.85,
            "precision": 0.83,
            "recall": 0.84,
            "f1": 0.835
        }
    ]
    
    metrics = _calculate_family_metrics(models)
    
    assert metrics["avg_f1"] == 0.835, f"Expected 0.835, got {metrics['avg_f1']}"
    assert metrics["avg_accuracy"] == 0.85, f"Expected 0.85, got {metrics['avg_accuracy']}"
    assert metrics["total_records"] == 100, f"Expected 100, got {metrics['total_records']}"
    
    print(f"   ✓ Single model metrics calculated correctly")


@test_case("Calculate family metrics - Multiple models")
def test_metrics_multiple():
    """Test metric calculation with multiple models."""
    models = [
        {"model_name": "Model_A", "num_records": 100, "f1": 0.8, "accuracy": 0.85, "precision": 0.82, "recall": 0.83},
        {"model_name": "Model_B", "num_records": 200, "f1": 0.6, "accuracy": 0.65, "precision": 0.62, "recall": 0.63},
        {"model_name": "Model_C", "num_records": 150, "f1": 0.7, "accuracy": 0.75, "precision": 0.72, "recall": 0.73}
    ]
    
    metrics = _calculate_family_metrics(models)
    
    # Average F1: (0.8 + 0.6 + 0.7) / 3 = 0.7
    expected_f1 = round((0.8 + 0.6 + 0.7) / 3, 4)
    assert metrics["avg_f1"] == expected_f1, f"Expected {expected_f1}, got {metrics['avg_f1']}"
    
    # Total records: 100 + 200 + 150 = 450
    assert metrics["total_records"] == 450, f"Expected 450, got {metrics['total_records']}"
    
    print(f"   ✓ Multiple model metrics aggregated correctly")
    print(f"      Avg F1: {metrics['avg_f1']}, Total Records: {metrics['total_records']}")


@test_case("Detect alerts - Critical F1 score")
def test_alert_critical_f1():
    """Test detection of critical F1 score."""
    models = [
        {"model_name": "Bad_Model", "num_records": 100, "f1": 0.15, "accuracy": 0.20}
    ]
    
    alerts = _detect_performance_alerts(models)
    
    # Should have 1 critical alert
    critical_alerts = [a for a in alerts if a["severity"] == "critical"]
    assert len(critical_alerts) == 1, f"Expected 1 critical alert, got {len(critical_alerts)}"
    
    alert = critical_alerts[0]
    assert alert["model_name"] == "Bad_Model", "Alert should reference Bad_Model"
    assert alert["metric"] == "f1_score", "Metric should be f1_score"
    assert alert["value"] == 0.15, "Value should be 0.15"
    
    print(f"   ✓ Critical F1 score detected: {alert['message']}")


@test_case("Detect alerts - Warning F1 score")
def test_alert_warning_f1():
    """Test detection of warning-level F1 score."""
    models = [
        {"model_name": "Mediocre_Model", "num_records": 100, "f1": 0.35, "accuracy": 0.40}
    ]
    
    alerts = _detect_performance_alerts(models)
    
    # Should have 1 warning alert
    warning_alerts = [a for a in alerts if a["severity"] == "warning"]
    assert len(warning_alerts) == 1, f"Expected 1 warning alert, got {len(warning_alerts)}"
    
    alert = warning_alerts[0]
    assert alert["model_name"] == "Mediocre_Model", "Alert should reference Mediocre_Model"
    assert alert["metric"] == "f1_score", "Metric should be f1_score"
    
    print(f"   ✓ Warning F1 score detected: {alert['message']}")


@test_case("Detect alerts - Insufficient training data")
def test_alert_insufficient_data():
    """Test detection of insufficient training data."""
    models = [
        {"model_name": "Sparse_Model", "num_records": 5, "f1": 0.60, "accuracy": 0.65}
    ]
    
    alerts = _detect_performance_alerts(models)
    
    # Should have 1 warning for insufficient data
    data_alerts = [a for a in alerts if a["metric"] == "training_data"]
    assert len(data_alerts) == 1, f"Expected 1 data alert, got {len(data_alerts)}"
    
    alert = data_alerts[0]
    assert alert["severity"] == "warning", "Insufficient data should be warning"
    assert alert["value"] == 5, "Value should be 5"
    
    print(f"   ✓ Insufficient data detected: {alert['message']}")


@test_case("Detect alerts - No training data")
def test_alert_no_data():
    """Test detection of zero training data."""
    models = [
        {"model_name": "Empty_Model", "num_records": 0, "f1": 0.0, "accuracy": 0.0}
    ]
    
    alerts = _detect_performance_alerts(models)
    
    # Should have 1 info alert for no data
    no_data_alerts = [a for a in alerts if a["metric"] == "training_data" and a["value"] == 0]
    assert len(no_data_alerts) == 1, f"Expected 1 no-data alert, got {len(no_data_alerts)}"
    
    alert = no_data_alerts[0]
    assert alert["severity"] == "info", "No data should be info severity"
    
    print(f"   ✓ No training data detected: {alert['message']}")


@test_case("Detect alerts - Multiple issues")
def test_alert_multiple():
    """Test detection of multiple alerts."""
    models = [
        {"model_name": "Bad_Model", "num_records": 100, "f1": 0.15, "accuracy": 0.20},
        {"model_name": "Sparse_Model", "num_records": 5, "f1": 0.60, "accuracy": 0.65},
        {"model_name": "Empty_Model", "num_records": 0, "f1": 0.0, "accuracy": 0.0},
        {"model_name": "Good_Model", "num_records": 200, "f1": 0.85, "accuracy": 0.90}
    ]
    
    alerts = _detect_performance_alerts(models)
    
    # Should have multiple alerts
    assert len(alerts) > 0, "Should have at least one alert"
    
    # Count by severity
    critical_count = sum(1 for a in alerts if a["severity"] == "critical")
    warning_count = sum(1 for a in alerts if a["severity"] == "warning")
    info_count = sum(1 for a in alerts if a["severity"] == "info")
    
    print(f"   ✓ Multiple alerts detected:")
    print(f"      Critical: {critical_count}, Warning: {warning_count}, Info: {info_count}")


@test_case("Alert sorting by severity")
def test_alert_sorting():
    """Test that alerts are sorted by severity."""
    models = [
        {"model_name": "Info_Model", "num_records": 0, "f1": 0.0, "accuracy": 0.0},
        {"model_name": "Critical_Model", "num_records": 100, "f1": 0.10, "accuracy": 0.15},
        {"model_name": "Warning_Model", "num_records": 5, "f1": 0.60, "accuracy": 0.65}
    ]
    
    alerts = _detect_performance_alerts(models)
    
    # Alerts should be sorted: critical, warning, info
    if len(alerts) >= 2:
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        for i in range(len(alerts) - 1):
            current_severity = severity_order[alerts[i]["severity"]]
            next_severity = severity_order[alerts[i+1]["severity"]]
            assert current_severity <= next_severity, \
                f"Alerts not properly sorted: {alerts[i]['severity']} before {alerts[i+1]['severity']}"
    
    print(f"   ✓ Alerts properly sorted by severity")


@test_case("Get grouped status - Structure")
def test_grouped_status_structure():
    """Test grouped status response structure."""
    result = get_grouped_training_status()
    
    # Check top-level fields
    required_fields = ["last_run", "status", "model_families", "alerts", "summary"]
    for field in required_fields:
        assert field in result, f"Missing field: {field}"
    
    # Check summary fields
    summary_fields = ["total_models", "total_families", "overall_avg_f1", "critical_alerts", "warning_alerts"]
    for field in summary_fields:
        assert field in result["summary"], f"Missing summary field: {field}"
    
    print(f"   ✓ Grouped status has all required fields")


@test_case("Get grouped status - Model families")
def test_grouped_status_families():
    """Test that model families are properly structured."""
    result = get_grouped_training_status()
    
    families = result["model_families"]
    
    if len(families) > 0:
        # Check first family structure
        family = families[0]
        required_fields = [
            "family_key", "family_name", "family_name_ar",
            "model_count", "avg_f1", "avg_accuracy", "avg_precision", "avg_recall",
            "total_records", "models"
        ]
        
        for field in required_fields:
            assert field in family, f"Family missing field: {field}"
        
        assert isinstance(family["models"], list), "Models should be a list"
        assert family["model_count"] == len(family["models"]), \
            "Model count should match models list length"
        
        print(f"   ✓ Found {len(families)} model families")
        print(f"      First family: {family['family_name']} ({family['model_count']} models)")
    else:
        print(f"   ⚠️  No families found (OK if no training data)")


@test_case("Get grouped status - Alerts")
def test_grouped_status_alerts():
    """Test that alerts are included in grouped status."""
    result = get_grouped_training_status()
    
    alerts = result["alerts"]
    assert isinstance(alerts, list), "Alerts should be a list"
    
    if len(alerts) > 0:
        # Check first alert structure
        alert = alerts[0]
        required_fields = [
            "severity", "model_name", "metric", "value",
            "message", "message_ar", "recommendation", "recommendation_ar"
        ]
        
        for field in required_fields:
            assert field in alert, f"Alert missing field: {field}"
        
        # Check severity is valid
        assert alert["severity"] in ["critical", "warning", "info"], \
            f"Invalid severity: {alert['severity']}"
        
        print(f"   ✓ Found {len(alerts)} alerts")
        print(f"      First alert: {alert['severity']} - {alert['message']}")
    else:
        print(f"   ✓ No alerts (all models performing well)")


@test_case("Get grouped status - Summary statistics")
def test_grouped_status_summary():
    """Test summary statistics calculation."""
    result = get_grouped_training_status()
    
    summary = result["summary"]
    
    # Check types
    assert isinstance(summary["total_models"], int), "total_models should be int"
    assert isinstance(summary["total_families"], int), "total_families should be int"
    assert isinstance(summary["overall_avg_f1"], float), "overall_avg_f1 should be float"
    assert isinstance(summary["critical_alerts"], int), "critical_alerts should be int"
    assert isinstance(summary["warning_alerts"], int), "warning_alerts should be int"
    
    # Check values are reasonable
    assert summary["total_models"] >= 0, "total_models should be non-negative"
    assert summary["total_families"] >= 0, "total_families should be non-negative"
    assert 0.0 <= summary["overall_avg_f1"] <= 1.0, "overall_avg_f1 should be between 0 and 1"
    
    print(f"   ✓ Summary statistics valid:")
    print(f"      Models: {summary['total_models']}, Families: {summary['total_families']}")
    print(f"      Avg F1: {summary['overall_avg_f1']:.4f}")
    print(f"      Alerts: {summary['critical_alerts']} critical, {summary['warning_alerts']} warning")


@test_case("Bilingual support")
def test_bilingual():
    """Test that bilingual labels are present."""
    result = get_grouped_training_status()
    
    # Check families have bilingual names
    for family in result["model_families"]:
        assert "family_name" in family, "Missing English name"
        assert "family_name_ar" in family, "Missing Arabic name"
        assert isinstance(family["family_name"], str), "English name should be string"
        assert isinstance(family["family_name_ar"], str), "Arabic name should be string"
    
    # Check alerts have bilingual messages
    for alert in result["alerts"]:
        assert "message" in alert, "Missing English message"
        assert "message_ar" in alert, "Missing Arabic message"
        assert "recommendation" in alert, "Missing English recommendation"
        assert "recommendation_ar" in alert, "Missing Arabic recommendation"
    
    print(f"   ✓ Bilingual support present (English/Arabic)")


# ==================== RUN ALL TESTS ====================

print("\n" + "="*80)
print("RUNNING TEST SUITE")
print("="*80)

# Run all tests
test_family_definitions()
test_family_structure()
test_model_count()
test_metrics_empty()
test_metrics_single()
test_metrics_multiple()
test_alert_critical_f1()
test_alert_warning_f1()
test_alert_insufficient_data()
test_alert_no_data()
test_alert_multiple()
test_alert_sorting()
test_grouped_status_structure()
test_grouped_status_families()
test_grouped_status_alerts()
test_grouped_status_summary()
test_bilingual()

# Print summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print(f"Total Tests: {total_tests}")
print(f"✅ Passed: {tests_passed}")
print(f"❌ Failed: {tests_failed}")
print(f"Success Rate: {(tests_passed/total_tests)*100:.1f}%")
print("="*80)

if tests_failed == 0:
    print("\n🎉 ALL TESTS PASSED! Phase 3 implementation is complete and verified.")
    sys.exit(0)
else:
    print(f"\n⚠️ {tests_failed} test(s) failed. Please review and fix.")
    sys.exit(1)
