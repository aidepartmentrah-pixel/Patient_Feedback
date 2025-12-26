from datetime import date, timedelta
import json

from backend.api.services.dashboard_service import (
    get_dashboard_stats,
    get_dashboard_hierarchy,
)


def pretty(title, data):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    print("=" * 80)


def test_dashboard_service_integration_real_database():
    """
    INTEGRATION TEST

    This test:
    - Hits the REAL database
    - Uses REAL db_layer
    - Prints REAL hierarchy
    - Prints REAL dashboard stats
    - Makes only minimal safety assertions

    Purpose:
    - Verify DB connectivity
    - Verify db_layer + service wiring
    - Allow human inspection of real data
    """

    # =========================================================
    # ACT: HIERARCHY (REAL DB)
    # =========================================================
    hierarchy = get_dashboard_hierarchy()
    pretty("REAL DATABASE — DASHBOARD HIERARCHY", hierarchy)

    # Minimal structural assertions
    assert isinstance(hierarchy, dict)
    assert "Administration" in hierarchy
    assert "Department" in hierarchy
    assert "Section" in hierarchy

    # =========================================================
    # ACT: DASHBOARD STATS (REAL DB)
    # =========================================================
    end_date = date.today()
    start_date = end_date - timedelta(days=30)

    stats = get_dashboard_stats(
        scope="hospital",
        administration_id=None,
        department_id=None,
        section_id=None,
        start_date=start_date,
        end_date=end_date,
    )

    # =========================================================
    # PRINT EVERYTHING (REAL DATA)
    # =========================================================
    pretty("REAL DATABASE — METRICS", stats.get("metrics"))
    pretty("REAL DATABASE — CHARTS", stats.get("charts"))
    pretty("REAL DATABASE — RECENT ACTIVITY", stats.get("recentActivity"))
    pretty("REAL DATABASE — TRENDS", stats.get("trends"))

    # =========================================================
    # Minimal safety assertions only
    # =========================================================
    assert "metrics" in stats
    assert "charts" in stats
    assert "recentActivity" in stats
    assert "trends" in stats
