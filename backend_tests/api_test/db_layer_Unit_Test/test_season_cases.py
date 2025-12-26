from backend.api.db_layer.season_cases import (
    create_season_case,
    get_season_case_by_id,
    list_season_cases,
    update_season_case,
)


def test_create_and_get_season_case():
    season_case_id = create_season_case(
        season_id=1,
        department_id=1,
        season_case_status_id=1,
        created_by_user_id=1,
        seasonal_report_text="Initial seasonal report",
    )

    case = get_season_case_by_id(season_case_id)
    assert case is not None
    assert case["SeasonID"] == 1
    assert case["DepartmentID"] == 1


def test_list_season_cases():
    cases = list_season_cases()
    assert isinstance(cases, list)


def test_update_season_case():
    season_case_id = create_season_case(
        season_id=1,
        department_id=1,
        season_case_status_id=1,
        created_by_user_id=1,
    )

    update_season_case(
        season_case_id,
        {
            "SeasonalReportDepartmentFeedback": "Department response text",
            "SeasonCaseStatusID": 2,
        },
    )

    updated = get_season_case_by_id(season_case_id)
    assert updated["SeasonCaseStatusID"] == 2
