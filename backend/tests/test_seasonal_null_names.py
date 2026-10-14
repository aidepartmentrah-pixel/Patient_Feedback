"""
Regression: a classification whose English name is NULL in the database made EVERY seasonal Word report
fail ("TypeError: 'NoneType' object is not iterable" at python-docx cell.text = None), because
dict.get(key, 'N/A') returns None - not 'N/A' - when the key is present with a None value.
"""
from backend.api.services.seasonal_report_formatter import _build_hierarchy


def _stat(**over):
    base = {
        "classification_id": 1, "total_count": 3, "low_count": 1, "medium_count": 1, "high_count": 1,
        "preventive_yes_count": 0, "preventive_no_count": 3,
        "domain_name": "Clinical", "category_name": "Care", "subcategory_name": "Monitoring",
        "classification_name": "اسم", "classification_name_en": "Name",
    }
    base.update(over)
    return base


def _all_strings(h):
    for domain, d in h.items():
        assert isinstance(domain, str)
        for cat, c in d["categories"].items():
            assert isinstance(cat, str)
            for sub, s in c["subcategories"].items():
                assert isinstance(sub, str)
                for cl in s["classifications"]:
                    assert isinstance(cl["classification_name"], str) and cl["classification_name"]
                    assert isinstance(cl["classification_name_en"], str) and cl["classification_name_en"]


def test_null_english_name_falls_back_to_arabic():
    h = _build_hierarchy([_stat(classification_name_en=None)])
    _all_strings(h)
    cl = h["Clinical"]["categories"]["Care"]["subcategories"]["Monitoring"]["classifications"][0]
    assert cl["classification_name_en"] == "اسم"


def test_every_name_null_still_builds_strings():
    h = _build_hierarchy([_stat(domain_name=None, category_name=None, subcategory_name=None,
                                classification_name=None, classification_name_en=None, total_count=None)])
    _all_strings(h)
    assert "Unknown Domain" in h


def test_does_not_mutate_the_input():
    s = _stat(classification_name_en=None)
    _build_hierarchy([s])
    assert s["classification_name_en"] is None
