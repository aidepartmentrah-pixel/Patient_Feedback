from backend.api.db_layer.lookups import (
    get_case_stages,
    get_case_statuses,
    get_domains,
    get_categories,
    get_subcategories,
    get_classifications,
    get_clinical_risk_types,
    get_feedback_intent_types,
    get_harm_levels,
    get_explanation_statuses,
    get_doctors,
)


def test_case_lookups():
    assert isinstance(get_case_stages(), list)
    assert isinstance(get_case_statuses(), list)


def test_classification_lookups():
    assert isinstance(get_domains(), list)
    assert isinstance(get_categories(), list)
    assert isinstance(get_subcategories(), list)
    assert isinstance(get_classifications(), list)


def test_other_lookups():
    assert isinstance(get_clinical_risk_types(), list)
    assert isinstance(get_feedback_intent_types(), list)
    assert isinstance(get_harm_levels(), list)
    assert isinstance(get_explanation_statuses(), list)
    assert isinstance(get_doctors(), list)
