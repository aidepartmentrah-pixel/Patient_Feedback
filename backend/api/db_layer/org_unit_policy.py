from typing import List, Dict, Any
from core.database import get_connection

# ============================================================
# READ FUNCTIONS
# ============================================================

def get_policy_by_unit_id(org_unit_id: int) -> dict | None:
    """
    Fetch policy row for any organizational unit
    (Administration / Department / Section).
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                OrgUnitID,
                OrgUnitType,
                LowSeverityLimit,
                MediumSeverityLimit,
                HighSeverityLimit,
                ClinicalDomainLimit,
                ManagementDomainLimit,
                RelationalDomainLimit,
                EnableLowSeverityRepetitionRule,
                EnableMediumSeverityRepetitionRule,
                EnableHighSeverityPercentageRule,
                EnableHighSeverityPercentageByDomainRule,
                IsActive
            FROM dbo.APP_OrgUnitPolicy
            WHERE OrgUnitID = ?
            """,
            org_unit_id,
        )
        row = cursor.fetchone()
        if not row:
            return None

        columns = [col[0] for col in cursor.description]
        return dict(zip(columns, row))


def get_administration_policy(administration_id: int) -> dict | None:
    """
    Fetch policy for an administration unit.
    """
    return get_policy_by_unit_id(administration_id)


def get_department_policy(department_id: int) -> dict | None:
    """
    Fetch policy for a department unit.
    """
    return get_policy_by_unit_id(department_id)


def get_section_policy(section_id: int) -> dict | None:
    """
    Fetch policy for a section unit.
    """
    return get_policy_by_unit_id(section_id)


# ============================================================
# WRITE FUNCTIONS
# ============================================================

def update_policy_for_unit(
    org_unit_id: int,
    *,
    low_severity_limit: int,
    medium_severity_limit: int,
    high_severity_limit: int,
    clinical_domain_limit: int,
    management_domain_limit: int,
    relational_domain_limit: int,
    enable_low_rule: bool,
    enable_medium_rule: bool,
    enable_high_percentage_rule: bool,
    enable_high_percentage_by_domain_rule: bool,
    updated_by_user_id: int,
) -> None:
    """
    Update policy values for a single organizational unit only.
    No cascading.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE dbo.APP_OrgUnitPolicy
            SET
                LowSeverityLimit = ?,
                MediumSeverityLimit = ?,
                HighSeverityLimit = ?,
                ClinicalDomainLimit = ?,
                ManagementDomainLimit = ?,
                RelationalDomainLimit = ?,
                EnableLowSeverityRepetitionRule = ?,
                EnableMediumSeverityRepetitionRule = ?,
                EnableHighSeverityPercentageRule = ?,
                EnableHighSeverityPercentageByDomainRule = ?,
                CreatedByUserID = ?
            WHERE OrgUnitID = ?
            """,
            (
                low_severity_limit,
                medium_severity_limit,
                high_severity_limit,
                clinical_domain_limit,
                management_domain_limit,
                relational_domain_limit,
                int(enable_low_rule),
                int(enable_medium_rule),
                int(enable_high_percentage_rule),
                int(enable_high_percentage_by_domain_rule),
                updated_by_user_id,
                org_unit_id,
            ),
        )
        conn.commit()


def update_policy_for_unit_with_descendants(
    org_unit_id: int,
    *,
    policy_data: dict,
    updated_by_user_id: int,
) -> None:
    """
    Update policy for a unit and ALL its descendants safely.
    Uses iterative traversal (no recursion).
    """

    visited = set()
    queue = [org_unit_id]

    with get_connection() as conn:
        cursor = conn.cursor()

        while queue:
            current_id = queue.pop(0)

            # 🛑 Safety guard
            if current_id in visited:
                continue

            visited.add(current_id)

            # 1️⃣ Update current unit
            cursor.execute(
                """
                UPDATE dbo.APP_OrgUnitPolicy
                SET
                    LowSeverityLimit = ?,
                    MediumSeverityLimit = ?,
                    HighSeverityLimit = ?,
                    ClinicalDomainLimit = ?,
                    ManagementDomainLimit = ?,
                    RelationalDomainLimit = ?,
                    EnableLowSeverityRepetitionRule = ?,
                    EnableMediumSeverityRepetitionRule = ?,
                    EnableHighSeverityPercentageRule = ?,
                    EnableHighSeverityPercentageByDomainRule = ?,
                    CreatedByUserID = ?
                WHERE OrgUnitID = ?
                """,
                (
                    policy_data["low_severity_limit"],
                    policy_data["medium_severity_limit"],
                    policy_data["high_severity_limit"],
                    policy_data["clinical_domain_limit"],
                    policy_data["management_domain_limit"],
                    policy_data["relational_domain_limit"],
                    int(policy_data["enable_low_rule"]),
                    int(policy_data["enable_medium_rule"]),
                    int(policy_data["enable_high_percentage_rule"]),
                    int(policy_data["enable_high_percentage_by_domain_rule"]),
                    updated_by_user_id,
                    current_id,
                ),
            )

            # 2️⃣ Fetch children
            cursor.execute(
                """
                SELECT UniqueID
                FROM dbo.AdminsrationUnit
                WHERE ParentID = ?
                """,
                current_id,
            )

            children = cursor.fetchall()
            for (child_id,) in children:
                if child_id not in visited:
                    queue.append(child_id)

        conn.commit()


# ============================================================
# VALIDATION FUNCTIONS
# ============================================================

def get_all_policy_rows() -> Dict[int, Dict[str, Any]]:
    """Return all policy rows keyed by OrgUnitID — single bulk query."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                OrgUnitID, IsActive,
                EnableLowSeverityRepetitionRule,
                EnableMediumSeverityRepetitionRule,
                EnableHighSeverityPercentageRule,
                EnableHighSeverityPercentageByDomainRule,
                LowSeverityLimit, MediumSeverityLimit, HighSeverityLimit,
                ClinicalDomainLimit, ManagementDomainLimit, RelationalDomainLimit
            FROM dbo.APP_OrgUnitPolicy
            """
        )
        rows: Dict[int, Dict[str, Any]] = {}
        for row in cursor.fetchall():
            cols = [c[0] for c in cursor.description]
            d = dict(zip(cols, row))
            rows[d["OrgUnitID"]] = d
        return rows


def validate_policy_coverage(unit_type_filter: int | None = None) -> Dict[str, Any]:
    """
    Scan all active org units and classify their policy status.

    Status values:
        ok               – active policy row with at least one rule enabled
        no_rules_enabled – row exists and active, but all Enable* flags = 0
        inactive         – row exists but IsActive = 0
        missing          – no row in APP_OrgUnitPolicy at all

    Args:
        unit_type_filter: Restrict to one type code (323/325/324). None = all.

    Returns:
        { "summary": {...}, "units": [{id, name, unit_type, type_label, status, issues}] }
    """
    from ..constants.org_unit_types import (
        ORG_TYPE_ADMINISTRATION, ORG_TYPE_DEPARTMENT, ORG_TYPE_SECTION,
        ORG_TYPE_NAME_MAP,
    )
    from .admin_units import get_units_by_type

    type_label_ar = {
        ORG_TYPE_ADMINISTRATION: "إدارة",
        ORG_TYPE_DEPARTMENT:     "دائرة",
        ORG_TYPE_SECTION:        "قسم",
    }

    types_to_check = (
        [unit_type_filter] if unit_type_filter is not None
        else [ORG_TYPE_ADMINISTRATION, ORG_TYPE_DEPARTMENT, ORG_TYPE_SECTION]
    )

    policy_map = get_all_policy_rows()
    results: List[Dict[str, Any]] = []
    summary = {"total": 0, "ok": 0, "missing": 0, "inactive": 0, "no_rules_enabled": 0}

    for unit_type in types_to_check:
        for unit in get_units_by_type(unit_type):
            uid  = unit["id"]
            summary["total"] += 1
            issues: List[str] = []
            policy = policy_map.get(uid)

            if policy is None:
                status = "missing"
                issues.append("لا يوجد سجل سياسة (No policy record in APP_OrgUnitPolicy)")
            elif not policy["IsActive"]:
                status = "inactive"
                issues.append("السياسة غير مفعّلة (IsActive = 0)")
            elif not any([
                bool(policy["EnableLowSeverityRepetitionRule"]),
                bool(policy["EnableMediumSeverityRepetitionRule"]),
                bool(policy["EnableHighSeverityPercentageRule"]),
                bool(policy["EnableHighSeverityPercentageByDomainRule"]),
            ]):
                status = "no_rules_enabled"
                issues.append("جميع قواعد الامتثال معطّلة (All compliance rules disabled)")
            else:
                status = "ok"

            summary[status] += 1
            results.append({
                "id":         uid,
                "name":       unit["name"],
                "unit_type":  unit_type,
                "type_label": type_label_ar.get(unit_type, str(unit_type)),
                "type_en":    ORG_TYPE_NAME_MAP.get(unit_type, "UNKNOWN"),
                "status":     status,
                "issues":     issues,
            })

    status_order = {"missing": 0, "inactive": 1, "no_rules_enabled": 2, "ok": 3}
    results.sort(key=lambda x: (status_order[x["status"]], x["name"]))
    return {"summary": summary, "units": results}


# ============================================================
# BULK / LEVEL OPERATIONS (Policy Metrics Refactor)
# ============================================================

HOSPITAL_POLICY_UNIT_ID: int = 1
HOSPITAL_POLICY_UNIT_TYPE: int = 0


def get_representative_policy_for_type(unit_type: int) -> dict | None:
    """
    Return one representative active policy row for the given unit type.
    Only considers non-frozen org units so the row reflects the values
    actually set by the standardized bulk save.
    Used by the Settings page to populate the 4-card UI with current values.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT TOP 1
                p.OrgUnitID, p.OrgUnitType,
                p.LowSeverityLimit, p.MediumSeverityLimit, p.HighSeverityLimit,
                p.ClinicalDomainLimit, p.ManagementDomainLimit, p.RelationalDomainLimit,
                p.EnableLowSeverityRepetitionRule, p.EnableMediumSeverityRepetitionRule,
                p.EnableHighSeverityPercentageRule, p.EnableHighSeverityPercentageByDomainRule,
                p.IsActive
            FROM dbo.APP_OrgUnitPolicy p
            JOIN dbo.AdminsrationUnit a ON p.OrgUnitID = a.UniqueID
            WHERE p.OrgUnitType = ?
              AND p.IsActive = 1
              AND a.Frozen = 0
              AND a.Type IS NOT NULL
            ORDER BY p.OrgUnitID
            """,
            unit_type,
        )
        row = cursor.fetchone()
        if not row:
            return None
        columns = [col[0] for col in cursor.description]
        return dict(zip(columns, row))


def get_hospital_policy() -> dict | None:
    """Return the hospital-level policy row (OrgUnitID=1)."""
    return get_policy_by_unit_id(HOSPITAL_POLICY_UNIT_ID)


def upsert_all_policies_for_type(
    unit_type: int,
    policy_fields: dict,
    updated_by: int,
) -> int:
    """
    Update ALL active org units of the given type to identical policy values.
    Creates missing rows automatically.  Returns count of units processed.

    policy_fields keys:
        low_severity_limit, medium_severity_limit, high_severity_limit,
        clinical_domain_limit, management_domain_limit, relational_domain_limit,
        enable_low_rule (bool), enable_medium_rule (bool)
    """
    from .admin_units import get_units_by_type

    units = get_units_by_type(unit_type)
    if not units:
        return 0

    unit_ids = [u["id"] for u in units]

    low    = policy_fields["low_severity_limit"]
    medium = policy_fields["medium_severity_limit"]
    high   = policy_fields["high_severity_limit"]
    clin   = policy_fields["clinical_domain_limit"]
    mgmt   = policy_fields["management_domain_limit"]
    rel    = policy_fields["relational_domain_limit"]
    en_low = int(policy_fields.get("enable_low_rule", True))
    en_med = int(policy_fields.get("enable_medium_rule", True))

    with get_connection() as conn:
        cursor = conn.cursor()

        # Fetch which IDs already have a policy row (any type, any state)
        placeholders = ",".join("?" * len(unit_ids))
        cursor.execute(
            f"SELECT OrgUnitID FROM dbo.APP_OrgUnitPolicy WHERE OrgUnitID IN ({placeholders})",
            unit_ids,
        )
        existing_ids = {row[0] for row in cursor.fetchall()}

        for unit in units:
            uid = unit["id"]

            if uid in existing_ids:
                cursor.execute(
                    """
                    UPDATE dbo.APP_OrgUnitPolicy
                    SET
                        LowSeverityLimit = ?,
                        MediumSeverityLimit = ?,
                        HighSeverityLimit = ?,
                        ClinicalDomainLimit = ?,
                        ManagementDomainLimit = ?,
                        RelationalDomainLimit = ?,
                        EnableLowSeverityRepetitionRule = ?,
                        EnableMediumSeverityRepetitionRule = ?,
                        EnableHighSeverityPercentageRule = 0,
                        EnableHighSeverityPercentageByDomainRule = 0,
                        IsActive = 1,
                        UpdatedAt = GETDATE(),
                        UpdatedByUserID = ?
                    WHERE OrgUnitID = ?
                    """,
                    (low, medium, high, clin, mgmt, rel, en_low, en_med, updated_by, uid),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO dbo.APP_OrgUnitPolicy (
                        OrgUnitID, OrgUnitType,
                        LowSeverityLimit, MediumSeverityLimit, HighSeverityLimit,
                        ClinicalDomainLimit, ManagementDomainLimit, RelationalDomainLimit,
                        EnableLowSeverityRepetitionRule, EnableMediumSeverityRepetitionRule,
                        EnableHighSeverityPercentageRule, EnableHighSeverityPercentageByDomainRule,
                        IsActive, CreatedAt, CreatedByUserID
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 1, GETDATE(), ?)
                    """,
                    (uid, unit_type, low, medium, high, clin, mgmt, rel, en_low, en_med, updated_by),
                )

        conn.commit()

    return len(units)


def upsert_hospital_policy(policy_fields: dict, updated_by: int) -> None:
    """
    Update or create the hospital-level policy row (OrgUnitID=1, OrgUnitType=0).
    """
    low    = policy_fields["low_severity_limit"]
    medium = policy_fields["medium_severity_limit"]
    high   = policy_fields["high_severity_limit"]
    clin   = policy_fields["clinical_domain_limit"]
    mgmt   = policy_fields["management_domain_limit"]
    rel    = policy_fields["relational_domain_limit"]
    en_low = int(policy_fields.get("enable_low_rule", True))
    en_med = int(policy_fields.get("enable_medium_rule", True))

    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            "SELECT OrgUnitID FROM dbo.APP_OrgUnitPolicy WHERE OrgUnitID = ?",
            HOSPITAL_POLICY_UNIT_ID,
        )
        exists = cursor.fetchone() is not None

        if exists:
            cursor.execute(
                """
                UPDATE dbo.APP_OrgUnitPolicy
                SET
                    LowSeverityLimit = ?,
                    MediumSeverityLimit = ?,
                    HighSeverityLimit = ?,
                    ClinicalDomainLimit = ?,
                    ManagementDomainLimit = ?,
                    RelationalDomainLimit = ?,
                    EnableLowSeverityRepetitionRule = ?,
                    EnableMediumSeverityRepetitionRule = ?,
                    EnableHighSeverityPercentageRule = 0,
                    EnableHighSeverityPercentageByDomainRule = 0,
                    IsActive = 1,
                    UpdatedAt = GETDATE(),
                    UpdatedByUserID = ?
                WHERE OrgUnitID = ?
                """,
                (low, medium, high, clin, mgmt, rel, en_low, en_med, updated_by, HOSPITAL_POLICY_UNIT_ID),
            )
        else:
            cursor.execute(
                """
                INSERT INTO dbo.APP_OrgUnitPolicy (
                    OrgUnitID, OrgUnitType,
                    LowSeverityLimit, MediumSeverityLimit, HighSeverityLimit,
                    ClinicalDomainLimit, ManagementDomainLimit, RelationalDomainLimit,
                    EnableLowSeverityRepetitionRule, EnableMediumSeverityRepetitionRule,
                    EnableHighSeverityPercentageRule, EnableHighSeverityPercentageByDomainRule,
                    IsActive, CreatedAt, CreatedByUserID
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 1, GETDATE(), ?)
                """,
                (
                    HOSPITAL_POLICY_UNIT_ID, HOSPITAL_POLICY_UNIT_TYPE,
                    low, medium, high, clin, mgmt, rel,
                    en_low, en_med, updated_by,
                ),
            )

        conn.commit()
