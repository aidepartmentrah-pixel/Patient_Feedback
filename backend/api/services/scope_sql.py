"""
SQL fragments for organizational scoping that are shared by the Critical Issues services
(red flags and never events).

A user sees a case when it is TARGETED at one of their units (APP_IncidentCaseTargetDepartment),
which is the same rule the dashboard and Table View use. The unit that merely ISSUED the case is
not what scopes it. A case that has no target row at all would otherwise disappear from every
view, so for those cases only the issuing unit decides visibility.
"""

from typing import Iterable, Optional, Tuple, List

# Names of the units a case is targeted at, joined into one display string. Falls back to the
# issuing unit's name (pass it as {issuing}) when the case has no target rows.
TARGET_UNIT_NAMES_SQL = """(
    SELECT STRING_AGG(tu.Name, N' / ')
    FROM dbo.APP_IncidentCaseTargetDepartment tn_td
    JOIN dbo.AdminsrationUnit tu ON tu.UniqueID = tn_td.DepartmentID
    WHERE tn_td.IncidentRequestCaseID = c.IncidentRequestCaseID
)"""


def _int_list(unit_ids: Optional[Iterable[int]]) -> List[int]:
    """Unit ids as plain ints. They are written straight into the SQL text rather than passed as
    parameters: a hospital-wide user has ~175 units, the scope condition appears several times per
    query and some queries run it up to 6 times, which exceeds SQL Server's 2100-parameter limit.
    Casting to int makes that safe (these ids come from the server's own scope resolver anyway)."""
    return sorted({int(i) for i in (unit_ids or [])})


def target_scope_condition(allowed_unit_ids: Optional[Iterable[int]], alias: str = "c") -> Tuple[str, List[int]]:
    """
    Return (sql_condition, params) restricting `alias`.IncidentRequestCaseID rows to cases the user
    may see. `alias` must be a real table alias of APP_IncidentCase (the correlated sub-queries
    need a qualified column; an unqualified one would bind to the inner table).

    No allowed units means no access: ("1=0", []).
    The ids are inlined, so the returned params list is empty (kept so callers can still extend with it).
    """
    ids = _int_list(allowed_unit_ids)
    if not ids:
        return "1=0", []

    placeholders = ",".join(str(i) for i in ids)
    condition = f"""(
        EXISTS (
            SELECT 1 FROM dbo.APP_IncidentCaseTargetDepartment sc_td
            WHERE sc_td.IncidentRequestCaseID = {alias}.IncidentRequestCaseID
              AND sc_td.DepartmentID IN ({placeholders})
        )
        OR (
            NOT EXISTS (
                SELECT 1 FROM dbo.APP_IncidentCaseTargetDepartment sc_td2
                WHERE sc_td2.IncidentRequestCaseID = {alias}.IncidentRequestCaseID
            )
            AND {alias}.IssuingOrgUnitID IN ({placeholders})
        )
    )"""
    return condition, []


def target_unit_filter(unit_ids: Iterable[int], alias: str = "c") -> Tuple[str, List[int]]:
    """
    Return (sql_condition, params) for "targeted at one of these units" (a unit plus everything
    beneath it, resolved by the caller). An empty set matches nothing.
    """
    ids = _int_list(unit_ids)
    if not ids:
        return "1=0", []
    placeholders = ",".join(str(i) for i in ids)
    condition = f"""EXISTS (
        SELECT 1 FROM dbo.APP_IncidentCaseTargetDepartment uf_td
        WHERE uf_td.IncidentRequestCaseID = {alias}.IncidentRequestCaseID
          AND uf_td.DepartmentID IN ({placeholders})
    )"""
    return condition, []
