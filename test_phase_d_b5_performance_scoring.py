"""
TEST TASK D-B5 — PERFORMANCE SCORING MODULE

Verifies performance scoring logic implementation.
"""

import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))


def test_file_exists():
    """Verify scoring module file exists at correct location."""
    scoring_path = backend_path / "api" / "services" / "performance_scoring.py"
    assert scoring_path.exists(), f"❌ Scoring file not found at: {scoring_path}"
    print("✅ performance_scoring.py exists")
    return True


def test_function_exists():
    """Verify compute_performance_score function exists."""
    try:
        from api.services.performance_scoring import compute_performance_score
        
        assert callable(compute_performance_score), "❌ compute_performance_score is not callable"
        
        print("✅ compute_performance_score function exists")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_result_model_exists():
    """Verify PerformanceScoreResult model exists."""
    try:
        from api.services.performance_scoring import PerformanceScoreResult
        from pydantic import BaseModel
        
        assert issubclass(PerformanceScoreResult, BaseModel), "❌ PerformanceScoreResult should be a Pydantic model"
        
        # Check for required fields
        model_fields = PerformanceScoreResult.model_fields
        required_fields = ['score', 'praise_level', 'risk_level', 'flags']
        
        for field_name in required_fields:
            assert field_name in model_fields, f"❌ Missing field: {field_name}"
        
        print("✅ PerformanceScoreResult model exists with correct fields")
        return True
        
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        return False


def test_no_db_imports():
    """Verify no database layer imports."""
    scoring_path = backend_path / "api" / "services" / "performance_scoring.py"
    
    with open(scoring_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    forbidden_imports = [
        'from ..db_layer',
        'from api.db_layer',
        'import db_layer',
        'import pyodbc',
        'from pyodbc'
    ]
    
    for forbidden in forbidden_imports:
        if forbidden in content:
            print(f"❌ Forbidden database import found: {forbidden}")
            return False
    
    print("✅ No database imports (pure logic)")
    return True


def test_no_router_imports():
    """Verify no router or FastAPI dependency imports."""
    scoring_path = backend_path / "api" / "services" / "performance_scoring.py"
    
    with open(scoring_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Allow FastAPI for Pydantic models but not router/dependency stuff
    forbidden_imports = [
        'from ..routers',
        'from api.routers',
        'APIRouter',
        'Depends(',
        'HTTPException',
        'from fastapi import'
    ]
    
    for forbidden in forbidden_imports:
        if forbidden in content:
            # Allow 'from pydantic' since we need BaseModel
            if 'pydantic' in content and forbidden == 'from fastapi import':
                continue
            print(f"❌ Forbidden router/FastAPI import found: {forbidden}")
            return False
    
    print("✅ No router or FastAPI dependency imports (pure logic)")
    return True


def test_score_clamping():
    """Verify score is clamped between 0 and 100."""
    try:
        from api.services.performance_scoring import compute_performance_score
        
        # Test upper bound - lots of completed actions
        result_upper = compute_performance_score(
            total_incidents=0,
            completed_actions=1000,
            overdue_actions=0,
            rejected_explanations=0
        )
        assert result_upper.score <= 100, f"❌ Score exceeded 100: {result_upper.score}"
        assert result_upper.score >= 0, f"❌ Score below 0: {result_upper.score}"
        
        # Test lower bound - lots of penalties
        result_lower = compute_performance_score(
            total_incidents=100,
            completed_actions=0,
            overdue_actions=50,
            rejected_explanations=50
        )
        assert result_lower.score >= 0, f"❌ Score below 0: {result_lower.score}"
        assert result_lower.score <= 100, f"❌ Score exceeded 100: {result_lower.score}"
        
        print(f"✅ Score properly clamped (upper test: {result_upper.score}, lower test: {result_lower.score})")
        return True
        
    except Exception as e:
        print(f"❌ Score clamping test failed: {e}")
        return False


def test_praise_levels():
    """Verify praise level labels are correct."""
    try:
        from api.services.performance_scoring import compute_performance_score
        
        # Test excellent (>= 85)
        result_excellent = compute_performance_score(
            total_incidents=0,
            completed_actions=50,
            overdue_actions=0,
            rejected_explanations=0
        )
        assert result_excellent.praise_level == "excellent", \
            f"❌ Expected 'excellent', got '{result_excellent.praise_level}' for score {result_excellent.score}"
        
        # Test good (>= 70 and < 85)
        # Calculation: 100 - 20*1 - 3*5 - 1*7 + 10*2 = 100 - 20 - 15 - 7 + 20 = 78
        result_good = compute_performance_score(
            total_incidents=20,
            completed_actions=10,
            overdue_actions=3,
            rejected_explanations=1
        )
        assert result_good.score >= 70 and result_good.score < 85, \
            f"❌ Score {result_good.score} not in good range (70-84)"
        assert result_good.praise_level == "good", \
            f"❌ Expected 'good', got '{result_good.praise_level}' for score {result_good.score}"
        
        # Test watch (>= 50 and < 70)
        # Calculation: 100 - 25*1 - 4*5 - 1*7 + 10*2 = 100 - 25 - 20 - 7 + 20 = 68
        result_watch = compute_performance_score(
            total_incidents=25,
            completed_actions=10,
            overdue_actions=4,
            rejected_explanations=1
        )
        assert result_watch.score >= 50 and result_watch.score < 70, \
            f"❌ Score {result_watch.score} not in watch range (50-69)"
        assert result_watch.praise_level == "watch", \
            f"❌ Expected 'watch', got '{result_watch.praise_level}' for score {result_watch.score}"
        
        # Test critical (< 50)
        result_critical = compute_performance_score(
            total_incidents=30,
            completed_actions=5,
            overdue_actions=10,
            rejected_explanations=5
        )
        assert result_critical.score < 50, f"❌ Score {result_critical.score} not in critical range"
        assert result_critical.praise_level == "critical", \
            f"❌ Expected 'critical', got '{result_critical.praise_level}' for score {result_critical.score}"
        
        print("✅ All praise levels correct (excellent, good, watch, critical)")
        return True
        
    except Exception as e:
        print(f"❌ Praise level test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_risk_levels():
    """Verify risk level labels are correct."""
    try:
        from api.services.performance_scoring import compute_performance_score
        
        # Test high risk (overdue > 5 OR rejected > 3)
        result_high_overdue = compute_performance_score(
            total_incidents=5,
            completed_actions=10,
            overdue_actions=6,  # > 5
            rejected_explanations=0
        )
        assert result_high_overdue.risk_level == "high", \
            f"❌ Expected 'high' risk for overdue=6, got '{result_high_overdue.risk_level}'"
        
        result_high_rejected = compute_performance_score(
            total_incidents=5,
            completed_actions=10,
            overdue_actions=0,
            rejected_explanations=4  # > 3
        )
        assert result_high_rejected.risk_level == "high", \
            f"❌ Expected 'high' risk for rejected=4, got '{result_high_rejected.risk_level}'"
        
        # Test medium risk (overdue > 2)
        result_medium = compute_performance_score(
            total_incidents=5,
            completed_actions=10,
            overdue_actions=3,  # > 2 but <= 5
            rejected_explanations=0  # <= 3
        )
        assert result_medium.risk_level == "medium", \
            f"❌ Expected 'medium' risk for overdue=3, got '{result_medium.risk_level}'"
        
        # Test low risk
        result_low = compute_performance_score(
            total_incidents=5,
            completed_actions=10,
            overdue_actions=2,  # <= 2
            rejected_explanations=0  # <= 3
        )
        assert result_low.risk_level == "low", \
            f"❌ Expected 'low' risk for overdue=2, got '{result_low.risk_level}'"
        
        print("✅ All risk levels correct (high, medium, low)")
        return True
        
    except Exception as e:
        print(f"❌ Risk level test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_flags():
    """Verify warning flags are set correctly."""
    try:
        from api.services.performance_scoring import compute_performance_score
        
        # Test many_overdue flag
        result_overdue = compute_performance_score(
            total_incidents=5,
            completed_actions=10,
            overdue_actions=6,  # > 5
            rejected_explanations=0
        )
        assert "many_overdue" in result_overdue.flags, \
            f"❌ Expected 'many_overdue' flag for overdue=6, got {result_overdue.flags}"
        
        # Test many_rejections flag
        result_rejected = compute_performance_score(
            total_incidents=5,
            completed_actions=10,
            overdue_actions=0,
            rejected_explanations=4  # > 3
        )
        assert "many_rejections" in result_rejected.flags, \
            f"❌ Expected 'many_rejections' flag for rejected=4, got {result_rejected.flags}"
        
        # Test both flags
        result_both = compute_performance_score(
            total_incidents=5,
            completed_actions=10,
            overdue_actions=6,  # > 5
            rejected_explanations=4  # > 3
        )
        assert "many_overdue" in result_both.flags, \
            f"❌ Expected 'many_overdue' flag, got {result_both.flags}"
        assert "many_rejections" in result_both.flags, \
            f"❌ Expected 'many_rejections' flag, got {result_both.flags}"
        
        # Test no flags
        result_none = compute_performance_score(
            total_incidents=5,
            completed_actions=10,
            overdue_actions=2,  # <= 5
            rejected_explanations=1  # <= 3
        )
        assert len(result_none.flags) == 0, \
            f"❌ Expected no flags, got {result_none.flags}"
        
        print("✅ All flags work correctly (many_overdue, many_rejections)")
        return True
        
    except Exception as e:
        print(f"❌ Flags test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scoring_formula():
    """Verify exact scoring formula implementation."""
    try:
        from api.services.performance_scoring import compute_performance_score
        
        # Test known calculation
        # Base: 100
        # - incidents: 10 * 1 = -10
        # - overdue: 3 * 5 = -15
        # - rejected: 2 * 7 = -14
        # + completed: 20 * 2 = +40
        # Expected: 100 - 10 - 15 - 14 + 40 = 101 → clamped to 100
        
        result = compute_performance_score(
            total_incidents=10,
            completed_actions=20,
            overdue_actions=3,
            rejected_explanations=2
        )
        
        # Raw would be 101, should be clamped to 100
        assert result.score == 100, \
            f"❌ Expected score 100 (clamped), got {result.score}"
        
        # Test another known calculation
        # Base: 100
        # - incidents: 5 * 1 = -5
        # - overdue: 0 * 5 = 0
        # - rejected: 0 * 7 = 0
        # + completed: 0 * 2 = 0
        # Expected: 100 - 5 = 95
        
        result2 = compute_performance_score(
            total_incidents=5,
            completed_actions=0,
            overdue_actions=0,
            rejected_explanations=0
        )
        
        assert result2.score == 95, \
            f"❌ Expected score 95, got {result2.score}"
        
        print("✅ Scoring formula is correct (verified with known calculations)")
        return True
        
    except Exception as e:
        print(f"❌ Scoring formula test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_deterministic():
    """Verify scoring is deterministic (same inputs = same outputs)."""
    try:
        from api.services.performance_scoring import compute_performance_score
        
        # Call with same inputs multiple times
        inputs = {
            "total_incidents": 12,
            "completed_actions": 15,
            "overdue_actions": 4,
            "rejected_explanations": 2
        }
        
        results = []
        for _ in range(5):
            result = compute_performance_score(**inputs)
            results.append({
                "score": result.score,
                "praise_level": result.praise_level,
                "risk_level": result.risk_level,
                "flags": sorted(result.flags)
            })
        
        # Check all results are identical
        first_result = results[0]
        for i, result in enumerate(results[1:], 2):
            assert result == first_result, \
                f"❌ Non-deterministic: call {i} differs from call 1"
        
        print(f"✅ Scoring is deterministic (5 identical calls produced same result)")
        return True
        
    except Exception as e:
        print(f"❌ Determinism test failed: {e}")
        return False


def test_high_penalty_scenario():
    """Test scenario with high overdue and high rejected (as specified in prompt)."""
    try:
        from api.services.performance_scoring import compute_performance_score
        
        # High overdue + high rejected scenario
        result = compute_performance_score(
            total_incidents=20,
            completed_actions=5,
            overdue_actions=10,  # High overdue
            rejected_explanations=8  # High rejected
        )
        
        # Should have:
        # - Low score (lots of penalties)
        # - Both flags
        # - High risk level
        # - Critical or watch praise level
        
        assert result.score < 50, \
            f"❌ Expected low score (<50) for high penalties, got {result.score}"
        
        assert "many_overdue" in result.flags, \
            f"❌ Expected 'many_overdue' flag, got {result.flags}"
        
        assert "many_rejections" in result.flags, \
            f"❌ Expected 'many_rejections' flag, got {result.flags}"
        
        assert result.risk_level == "high", \
            f"❌ Expected 'high' risk level, got '{result.risk_level}'"
        
        assert result.praise_level in ["critical", "watch"], \
            f"❌ Expected 'critical' or 'watch' praise level, got '{result.praise_level}'"
        
        print(f"✅ High penalty scenario works correctly:")
        print(f"   Score: {result.score}")
        print(f"   Praise: {result.praise_level}")
        print(f"   Risk: {result.risk_level}")
        print(f"   Flags: {result.flags}")
        
        return True
        
    except Exception as e:
        print(f"❌ High penalty scenario test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_zero_inputs():
    """Test with all zero inputs (baseline)."""
    try:
        from api.services.performance_scoring import compute_performance_score
        
        result = compute_performance_score(
            total_incidents=0,
            completed_actions=0,
            overdue_actions=0,
            rejected_explanations=0
        )
        
        # Should be perfect baseline score
        assert result.score == 100, \
            f"❌ Expected perfect score 100 for zero inputs, got {result.score}"
        
        assert result.praise_level == "excellent", \
            f"❌ Expected 'excellent' for perfect score, got '{result.praise_level}'"
        
        assert result.risk_level == "low", \
            f"❌ Expected 'low' risk for zero inputs, got '{result.risk_level}'"
        
        assert len(result.flags) == 0, \
            f"❌ Expected no flags for zero inputs, got {result.flags}"
        
        print("✅ Zero inputs produce perfect baseline score (100, excellent, low risk)")
        return True
        
    except Exception as e:
        print(f"❌ Zero inputs test failed: {e}")
        return False


def run_all_tests():
    """Run all verification tests."""
    print("=" * 70)
    print("TEST TASK D-B5 — PERFORMANCE SCORING MODULE")
    print("=" * 70)
    print()
    
    tests = [
        ("File Exists", test_file_exists),
        ("Function Exists", test_function_exists),
        ("Result Model Exists", test_result_model_exists),
        ("No Database Imports", test_no_db_imports),
        ("No Router Imports", test_no_router_imports),
        ("Score Clamping (0-100)", test_score_clamping),
        ("Praise Levels", test_praise_levels),
        ("Risk Levels", test_risk_levels),
        ("Warning Flags", test_flags),
        ("Scoring Formula", test_scoring_formula),
        ("Deterministic Behavior", test_deterministic),
        ("High Penalty Scenario", test_high_penalty_scenario),
        ("Zero Inputs Baseline", test_zero_inputs),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 70)
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total:  {passed + failed}")
    print()
    
    if failed == 0:
        print("🎉 SCORING MODULE OK — ALL TESTS PASSED")
        return 0
    else:
        print("⚠️  SCORING MODULE HAS ISSUES — REVIEW FAILURES ABOVE")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
