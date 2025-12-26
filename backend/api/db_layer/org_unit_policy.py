import pyodbc

def get_connection():
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=SOCIALMEDIA;"
        "DATABASE=IncidentManager;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )
    return conn

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
