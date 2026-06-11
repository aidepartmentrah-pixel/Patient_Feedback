"""
Policy Service — 4-card organizational unit policy management.

UI loading rule: one representative row per level.
Save rule: overwrite ALL rows at that level (standardized).
Missing rows are created automatically on save.

Section semantics (NEW):
    LowSeverityLimit    = total incidents per HCAT classification
    MediumSeverityLimit = medium incidents per HCAT classification
    HighSeverityLimit   = high incidents per HCAT classification

Department / Administration semantics (UNCHANGED):
    ClinicalDomainLimit / ManagementDomainLimit / RelationalDomainLimit
"""

from ..constants.org_unit_types import (
    ORG_TYPE_ADMINISTRATION,
    ORG_TYPE_DEPARTMENT,
    ORG_TYPE_SECTION,
)
from ..db_layer.org_unit_policy import (
    get_representative_policy_for_type,
    get_hospital_policy,
    upsert_all_policies_for_type,
    upsert_hospital_policy,
)


# ──────────────────────────────────────────────────────────────
# UI Helpers
# ──────────────────────────────────────────────────────────────

def _section_ui(row: dict | None) -> dict:
    """Map DB row → section card fields (NEW semantics)."""
    if not row:
        return {"all_limit": 10, "medium_limit": 5, "high_limit": 3}
    return {
        "all_limit":    row.get("LowSeverityLimit", 10),
        "medium_limit": row.get("MediumSeverityLimit", 5),
        "high_limit":   row.get("HighSeverityLimit", 3),
    }


def _domain_ui(row: dict | None) -> dict:
    """Map DB row → department/administration card fields."""
    if not row:
        return {
            "clinical_domain_limit":    10,
            "management_domain_limit":  10,
            "relational_domain_limit":  10,
        }
    return {
        "clinical_domain_limit":   row.get("ClinicalDomainLimit", 10),
        "management_domain_limit": row.get("ManagementDomainLimit", 10),
        "relational_domain_limit": row.get("RelationalDomainLimit", 10),
    }


def _hospital_ui(row: dict | None) -> dict:
    """Map DB row → hospital card fields."""
    if not row:
        return {
            "low_severity_limit":       10,
            "medium_severity_limit":    5,
            "high_severity_limit":      3,
            "clinical_domain_limit":    10,
            "management_domain_limit":  10,
            "relational_domain_limit":  10,
        }
    return {
        "low_severity_limit":      row.get("LowSeverityLimit", 10),
        "medium_severity_limit":   row.get("MediumSeverityLimit", 5),
        "high_severity_limit":     row.get("HighSeverityLimit", 3),
        "clinical_domain_limit":   row.get("ClinicalDomainLimit", 10),
        "management_domain_limit": row.get("ManagementDomainLimit", 10),
        "relational_domain_limit": row.get("RelationalDomainLimit", 10),
    }


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def get_org_level_policies() -> dict:
    """
    Return current policy values for all 4 Settings cards.
    Reads one representative active row per level.
    """
    return {
        "hospital":        _hospital_ui(get_hospital_policy()),
        "sections":        _section_ui(get_representative_policy_for_type(ORG_TYPE_SECTION)),
        "departments":     _domain_ui(get_representative_policy_for_type(ORG_TYPE_DEPARTMENT)),
        "administrations": _domain_ui(get_representative_policy_for_type(ORG_TYPE_ADMINISTRATION)),
    }


def save_sections_policy(
    all_limit: int,
    medium_limit: int,
    high_limit: int,
    user_id: int,
) -> int:
    """
    Overwrite ALL section policy rows with the new standardized values.
    LowSeverityLimit  → all_limit   (total incidents per classification)
    MediumSeverityLimit → medium_limit
    HighSeverityLimit   → high_limit
    Domain limits are zeroed (not applicable to sections).
    Returns number of sections processed.
    """
    fields = {
        "low_severity_limit":      all_limit,
        "medium_severity_limit":   medium_limit,
        "high_severity_limit":     high_limit,
        "clinical_domain_limit":   0,
        "management_domain_limit": 0,
        "relational_domain_limit": 0,
        "enable_low_rule":         True,
        "enable_medium_rule":      True,
    }
    return upsert_all_policies_for_type(ORG_TYPE_SECTION, fields, user_id)


def save_departments_policy(
    clinical_limit: int,
    management_limit: int,
    relational_limit: int,
    user_id: int,
) -> int:
    """
    Overwrite ALL department policy rows with the new standardized domain limits.
    Severity limits are zeroed (not applicable to departments).
    Returns number of departments processed.
    """
    fields = {
        "low_severity_limit":      0,
        "medium_severity_limit":   0,
        "high_severity_limit":     0,
        "clinical_domain_limit":   clinical_limit,
        "management_domain_limit": management_limit,
        "relational_domain_limit": relational_limit,
        "enable_low_rule":         False,
        "enable_medium_rule":      False,
    }
    return upsert_all_policies_for_type(ORG_TYPE_DEPARTMENT, fields, user_id)


def save_administrations_policy(
    clinical_limit: int,
    management_limit: int,
    relational_limit: int,
    user_id: int,
) -> int:
    """
    Overwrite ALL administration policy rows with the new standardized domain limits.
    Returns number of administrations processed.
    """
    fields = {
        "low_severity_limit":      0,
        "medium_severity_limit":   0,
        "high_severity_limit":     0,
        "clinical_domain_limit":   clinical_limit,
        "management_domain_limit": management_limit,
        "relational_domain_limit": relational_limit,
        "enable_low_rule":         False,
        "enable_medium_rule":      False,
    }
    return upsert_all_policies_for_type(ORG_TYPE_ADMINISTRATION, fields, user_id)


def save_hospital_policy(
    low_severity_limit: int,
    medium_severity_limit: int,
    high_severity_limit: int,
    clinical_domain_limit: int,
    management_domain_limit: int,
    relational_domain_limit: int,
    user_id: int,
) -> None:
    """Update or create the single hospital-level policy row."""
    fields = {
        "low_severity_limit":      low_severity_limit,
        "medium_severity_limit":   medium_severity_limit,
        "high_severity_limit":     high_severity_limit,
        "clinical_domain_limit":   clinical_domain_limit,
        "management_domain_limit": management_domain_limit,
        "relational_domain_limit": relational_domain_limit,
        "enable_low_rule":         True,
        "enable_medium_rule":      True,
    }
    upsert_hospital_policy(fields, user_id)
