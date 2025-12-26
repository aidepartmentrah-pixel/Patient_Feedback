from backend.api.db_layer.org_unit_policy import (
    get_policy_by_unit_id,
    get_section_policy,
    update_policy_for_unit,
    update_policy_for_unit_with_descendants,
)

# ---------------------------------------
# CONFIG (CHANGE IDS TO REAL VALUES)
# ---------------------------------------

ADMIN_ID = 1       # administration
DEPARTMENT_ID = 10 # department
SECTION_ID = 25    # section
USER_ID = 1

POLICY_SAMPLE = {
    "low_severity_limit": 5,
    "medium_severity_limit": 3,
    "high_severity_limit": 10,
    "clinical_domain_limit": 15,
    "management_domain_limit": 10,
    "relational_domain_limit": 12,
    "enable_low_rule": True,
    "enable_medium_rule": True,
    "enable_high_percentage_rule": True,
    "enable_high_percentage_by_domain_rule": True,
}


# ---------------------------------------
# TESTS
# ---------------------------------------

def test_get_policy():
    policy = get_policy_by_unit_id(SECTION_ID)
    assert policy is not None
    print("✔ get_policy_by_unit_id OK")


def test_update_single_unit():
    update_policy_for_unit(
        SECTION_ID,
        updated_by_user_id=USER_ID,
        **POLICY_SAMPLE,
    )

    policy = get_section_policy(SECTION_ID)
    assert policy["LowSeverityLimit"] == POLICY_SAMPLE["low_severity_limit"]
    print("✔ update_policy_for_unit OK")


def test_update_department_with_sections():
    update_policy_for_unit_with_descendants(
        DEPARTMENT_ID,
        policy_data=POLICY_SAMPLE,
        updated_by_user_id=USER_ID,
    )

    dept_policy = get_policy_by_unit_id(DEPARTMENT_ID)
    assert dept_policy["MediumSeverityLimit"] == POLICY_SAMPLE["medium_severity_limit"]

    print("✔ update_policy_for_unit_with_descendants OK")


# ---------------------------------------
# RUN
# ---------------------------------------

if __name__ == "__main__":
    test_get_policy()
    test_update_single_unit()
    test_update_department_with_sections()
    print("\nALL POLICY TESTS PASSED ✔")
