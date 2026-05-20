from core.database import get_connection

# -----------------------------
# GENERIC HELPER
# -----------------------------

def _fetch_all(query: str, params: tuple = ()) -> list[dict]:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(query, params)
    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]

    conn.close()
    return [dict(zip(columns, row)) for row in rows]


# -----------------------------
# CASE LOOKUPS
# -----------------------------

def get_case_stages() -> list[dict]:
    return _fetch_all(
        """
        SELECT StageID, StageName, StageOrder
        FROM dbo.APP_LOOKUP_CASE_STAGE
        ORDER BY StageOrder
        """
    )


def get_case_statuses(active_only: bool = True) -> list[dict]:
    query = """
        SELECT CaseStatusID, Code, Name, IsFinal, IsActive, DisplayOrder
        FROM dbo.APP_LOOKUP_CASE_STATUS
    """
    if active_only:
        query += " WHERE IsActive = 1"

    query += " ORDER BY DisplayOrder"

    return _fetch_all(query)


# -----------------------------
# CLASSIFICATION LOOKUPS
# -----------------------------

def get_domains() -> list[dict]:
    return _fetch_all(
        """
        SELECT DomainID, DomainCode, DomainName, DomainOrder
        FROM dbo.APP_LOOKUP_DOMAIN
        ORDER BY DomainOrder
        """
    )


def get_categories(domain_id: int | None = None) -> list[dict]:
    if domain_id:
        return _fetch_all(
            """
            SELECT CategoryID, DomainID, CategoryName, CategoryOrder
            FROM dbo.APP_LOOKUP_CATEGORY
            WHERE DomainID = ?
            ORDER BY CategoryOrder
            """,
            (domain_id,),
        )

    return _fetch_all(
        """
        SELECT CategoryID, DomainID, CategoryName, CategoryOrder
        FROM dbo.APP_LOOKUP_CATEGORY
        ORDER BY CategoryOrder
        """
    )


def get_subcategories(category_id: int | None = None) -> list[dict]:
    if category_id:
        return _fetch_all(
            """
            SELECT SubCategoryID, CategoryID, SubCategoryName
            FROM dbo.APP_LOOKUP_SUBCATEGORY
            WHERE CategoryID = ?
            ORDER BY SubCategoryName
            """,
            (category_id,),
        )

    return _fetch_all(
        """
        SELECT SubCategoryID, CategoryID, SubCategoryName
        FROM dbo.APP_LOOKUP_SUBCATEGORY
        ORDER BY SubCategoryName
        """
    )


def get_classifications(subcategory_id: int | None = None) -> list[dict]:
    if subcategory_id:
        return _fetch_all(
            "SELECT ClassificationID, SubCategoryID, Classification_AR, Classification_EN "
            "FROM dbo.APP_LOOKUP_CLASSIFICATION "
            "WHERE SubCategoryID = ? AND IsActive = 1 "
            "ORDER BY Classification_AR",
            (subcategory_id,),
        )

    return _fetch_all(
        "SELECT ClassificationID, SubCategoryID, Classification_AR, Classification_EN "
        "FROM dbo.APP_LOOKUP_CLASSIFICATION "
        "WHERE IsActive = 1 "
        "ORDER BY Classification_AR"
    )


# -----------------------------
# OTHER LOOKUPS
# -----------------------------

def get_clinical_risk_types(active_only: bool = True) -> list[dict]:
    query = """
        SELECT ClinicalRiskTypeID, Code, Name, IsActive, DisplayOrder
        FROM dbo.APP_LOOKUP_CLINICAL_RISK_TYPE
    """
    if active_only:
        query += " WHERE IsActive = 1"

    query += " ORDER BY DisplayOrder"

    return _fetch_all(query)


def get_feedback_intent_types(active_only: bool = True) -> list[dict]:
    query = """
        SELECT FeedbackIntentTypeID, Code, NameAr, NameEn, IsActive, DisplayOrder
        FROM dbo.APP_LOOKUP_FEEDBACK_INTENT_TYPE
    """
    if active_only:
        query += " WHERE IsActive = 1"

    query += " ORDER BY DisplayOrder"

    return _fetch_all(query)


def get_harm_levels() -> list[dict]:
    return _fetch_all(
        """
        SELECT HarmID, HarmLevel, SeverityOrder
        FROM dbo.APP_LOOKUP_HARM_LEVEL
        ORDER BY SeverityOrder
        """
    )


def get_explanation_statuses() -> list[dict]:
    return _fetch_all(
        """
        SELECT StatusID, StatusName
        FROM dbo.APP_LOOKUP_EXPLANATION_STATUS
        ORDER BY StatusName
        """
    )


# -----------------------------
# BUILDINGS LOOKUP
# -----------------------------

def get_buildings() -> list[dict]:
    """Return list of buildings (BuildingID, BuildingCode, BuildingName)."""
    return _fetch_all(
        """
        SELECT BuildingID, BuildingCode, BuildingName
        FROM dbo.APP_LOOKUP_BUILDING
        ORDER BY BuildingCode
        """
    )


def get_doctors(active_only: bool = True) -> list[dict]:
    query = """
        SELECT DoctorID, DoctorName, Specialty, IsActive
        FROM dbo.APP_LOOKUP_DOCTOR
    """
    if active_only:
        query += " WHERE IsActive = 1"

    query += " ORDER BY DoctorName"

    return _fetch_all(query)
