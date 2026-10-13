"""
Regression test for get_dashboard_hierarchy()'s Type-aware bucketing.

Covers the org-unit mislabeling bug: a unit parented directly under an
Administration (no Department in between) must be classified by its own
Type, not by "where it sits in ParentID". Before the fix, Step 2 swept any
direct child of an Administration into the Department bucket regardless of
Type, which is exactly what happened to "الخدمات البيئية" and "الهندسة
الطبية" once their AdminsrationUnit.Type was corrected from 325 to 324.

No mocking library is used elsewhere in this repo's tests -- pytest's
built-in monkeypatch fixture is enough here, and pytest is already a
declared dependency (requirements.txt).

Run: pytest backend/tests/test_dashboard_hierarchy.py -v
"""

import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from api.services import dashboard_service
from api.db_layer import admin_units
from api.schemas.auth_models import CurrentUser, UserScope


# Fixture hierarchy:
#   1  Administration "الإدارة العامة"                    Type=323
#     5    Department "دائرة الهندسة"  (parent=1)          Type=325
#       50     Section "قسم الصيانة"   (parent=5)          Type=324  <- normal, nested
#     64   Section "الخدمات البيئية"   (parent=1, direct)  Type=324  <- skip-level, the bug case
_FIXTURE_UNITS = [
    {"UniqueID": 1, "ParentID": 1, "Type": 323, "Name": "الإدارة العامة", "Frozen": False},
    {"UniqueID": 5, "ParentID": 1, "Type": 325, "Name": "دائرة الهندسة", "Frozen": False},
    {"UniqueID": 50, "ParentID": 5, "Type": 324, "Name": "قسم الصيانة", "Frozen": False},
    {"UniqueID": 64, "ParentID": 1, "Type": 324, "Name": "الخدمات البيئية", "Frozen": False},
]


def _make_current_user():
    # Two scopes so the single-scope "widen to ancestors" branch (lines
    # ~370-378) doesn't engage -- allowed_unit_ids already covers everything
    # in the fixture directly, which is all that branch would achieve anyway.
    return CurrentUser(
        user_id=1,
        username="test_admin",
        is_active=True,
        scopes=[
            UserScope(role_code="SOFTWARE_ADMIN", org_unit_id=1, org_unit_type="ADMINISTRATION"),
            UserScope(role_code="SOFTWARE_ADMIN", org_unit_id=5, org_unit_type="DEPARTMENT"),
        ],
        allowed_unit_ids={1, 5, 50, 64},
    )


def test_department_bucket_excludes_direct_child_sections(monkeypatch):
    """A Type=324 unit parented directly under an Administration must NOT
    appear in the Department bucket, even though it sits exactly where a
    Department would (ParentID == admin_id)."""
    monkeypatch.setattr(admin_units, "get_admin_unit_tree", lambda: _FIXTURE_UNITS)

    result = dashboard_service.get_dashboard_hierarchy(_make_current_user())

    dept_ids_under_admin_1 = {d["id"] for d in result["Department"].get(1, [])}
    assert dept_ids_under_admin_1 == {5}, (
        f"Expected only the real Department (5) under Administration 1, got {dept_ids_under_admin_1}"
    )


def test_skip_level_section_lands_in_section_bucket(monkeypatch):
    """The same unit (64) must show up in the Section bucket, keyed by its
    Administration's id since it has no Department parent."""
    monkeypatch.setattr(admin_units, "get_admin_unit_tree", lambda: _FIXTURE_UNITS)

    result = dashboard_service.get_dashboard_hierarchy(_make_current_user())

    section_ids_under_admin_1 = {s["id"] for s in result["Section"].get(1, [])}
    assert 64 in section_ids_under_admin_1, (
        f"Expected skip-level section 64 under Administration 1's Section bucket, got {section_ids_under_admin_1}"
    )


def test_normal_nested_section_unaffected(monkeypatch):
    """A regular Section under a real Department must still resolve exactly
    as before -- this fix must not disturb the common 3-level case."""
    monkeypatch.setattr(admin_units, "get_admin_unit_tree", lambda: _FIXTURE_UNITS)

    result = dashboard_service.get_dashboard_hierarchy(_make_current_user())

    section_ids_under_dept_5 = {s["id"] for s in result["Section"].get(5, [])}
    assert section_ids_under_dept_5 == {50}


def test_administration_bucket_unaffected(monkeypatch):
    monkeypatch.setattr(admin_units, "get_admin_unit_tree", lambda: _FIXTURE_UNITS)

    result = dashboard_service.get_dashboard_hierarchy(_make_current_user())

    admin_ids = {a["id"] for a in result["Administration"]}
    assert admin_ids == {1}
