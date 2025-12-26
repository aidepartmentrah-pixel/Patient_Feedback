from datetime import date, datetime, timedelta
import json

from backend.api.services.dashboard_service import (
    get_dashboard_stats,
    get_dashboard_hierarchy,
)


def pretty(title, data):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print("=" * 80)


def test_dashboard_service_full_with_prints(monkeypatch):
    """
    Human-readable service test.
    This test:
    - Prints hierarchy
    - Prints metrics
    - Prints charts
    - Prints recent activity
    - Prints trends
    """

    # =========================================================
    # MOCK: ORGANIZATIONAL HIERARCHY
    # =========================================================
    monkeypatch.setattr(
        "backend.api.db_layer_Unit_Test.admin_units.get_admin_unit_tree",
        lambda: [
            {"UniqueID": 1, "ParentID": None, "Type": 1, "Name": "Administration A"},
            {"UniqueID": 2, "ParentID": 1, "Type": 2, "Name": "Department A"},
            {"UniqueID": 3, "ParentID": 2, "Type": 3, "Name": "Section A"},
        ],
    )

    # =========================================================
    # MOCK: INCIDENT DATA
    # =========================================================
    now = datetime.now()

    monkeypatch.setattr(
        "backend.api.db_layer_Unit_Test.incident_case.list_incident_cases",
        lambda: [
            {
                "IssuingOrgUnitID": 3,
                "CreatedAt": now - timedelta(days=2),
                "PatientName": "Patient One",
                "CaseStatusID": 1,        # Open
                "SeverityID": 3,          # High
                "DomainID": 1,            # Clinical
                "ClassificationID": 101,
                "StageID": 1,
                "ComplaintText": "High severity clinical incident",
            },
            {
                "IssuingOrgUnitID": 3,
                "CreatedAt": now - timedelta(days=1),
                "PatientName": "Patient Two",
                "CaseStatusID": 2,        # Closed
                "SeverityID": 2,          # Medium
                "DomainID": 2,            # Management
                "ClassificationID": 202,
                "StageID": 2,
                "ComplaintText": "Medium severity management issue",
            },
        ],
    )

    # =========================================================
    # ACT: HIERARCHY
    # =========================================================
    hierarchy = get_dashboard_hierarchy()
    pretty("HIERARCHY OUTPUT", hierarchy)

    # ---- Assertions (structure + meaning)
    assert hierarchy["Administration"][0]["nameEn"] == "Administration A"
    assert hierarchy["Department"][1][0]["nameEn"] == "Department A"
    assert hierarchy["Section"][2][0]["nameEn"] == "Section A"

    # =========================================================
    # ACT: DASHBOARD STATS
    # =========================================================
    stats = get_dashboard_stats(
        scope="hospital",
        administration_id=None,
        department_id=None,
        section_id=None,
        start_date=date.today() - timedelta(days=7),
        end_date=date.today(),
    )

    # =========================================================
    # PRINT EVERYTHING
    # =========================================================
    pretty("METRICS", stats["metrics"])
    pretty("CHARTS", stats["charts"])
    pretty("RECENT ACTIVITY", stats["recentActivity"])
    pretty("TRENDS", stats["trends"])

    # =========================================================
    # ASSERT METRICS (semantic)
    # =========================================================
    metrics = stats["metrics"]

    assert metrics["totalIncidents"] == 2
    assert metrics["uniquePatients"] == 2
    assert metrics["openClosed"]["open"] == 1
    assert metrics["openClosed"]["closed"] == 1
    assert metrics["severityBreakdown"]["high"] == 1
    assert metrics["severityBreakdown"]["medium"] == 1
    assert metrics["domainBreakdown"]["clinical"] == 1
    assert metrics["domainBreakdown"]["management"] == 1
    assert metrics["redFlags"] == 1

    # =========================================================
    # ASSERT CHARTS
    # =========================================================
    assert len(stats["charts"]["top5Classification"]) == 2
    assert len(stats["charts"]["stageHistogram"]) == 2
    assert "issuingDept" in stats["charts"]

    # =========================================================
    # ASSERT RECENT ACTIVITY
    # =========================================================
    assert len(stats["recentActivity"]) == 2
    assert stats["recentActivity"][0]["description"] != ""

    # =========================================================
    # ASSERT TRENDS EXIST
    # =========================================================
    assert "incidentsPatients" in stats["trends"]
