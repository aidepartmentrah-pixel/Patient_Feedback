-- ============================================================================
-- READ-ONLY. Modifies nothing. Run this BEFORE fix_org_unit_type_mismatch.sql
-- and review the full output with a domain expert before applying the fix.
--
-- Background: dbo.AdminsrationUnit.Type is the org hierarchy level for each
-- unit (323 = Administration, 324 = Section, 325 = Department). Some units
-- are typed 325 (Department) even though their dedicated SECTION_ADMIN
-- account was created with APP_UserRoleScope.OrgUnitType = 'SECTION' for the
-- same OrgUnitID -- e.g. "الخدمات البيئية" (id 64) and "الهندسة الطبية"
-- (id 73). That disagreement is what makes these units print/display as
-- "Department" everywhere (report filters, Word reports, target-unit
-- selector) when they were set up and staffed as Sections attached directly
-- to an Administration.
--
-- This does NOT touch the 6 already-known, unrelated orphaned units (22, 138,
-- 167, 168, 172, 177), which have NULL/unresolvable ParentID and NULL Type --
-- a different, already-catalogued issue. Their IDs are excluded inline below.
--
-- Each section below is a fully self-contained statement -- no DECLARE'd
-- variable is shared across statements -- because SQL client tools that run
-- a script "statement by statement" (DBeaver's default script execution,
-- among others) send each one to SQL Server as its own separate batch, and
-- T-SQL local/table variables do NOT persist across batch boundaries (this
-- is standard SQL Server behavior, not a client bug -- the same thing
-- happens in SSMS across a `GO`). An earlier version of this script used a
-- single DECLARE'd table variable shared across sections and failed in
-- DBeaver with "Must declare the table variable "@known_orphans"" for
-- exactly this reason.
--
-- Reports:
--   1. Whether you're connected to the right database.
--   2. Every unit typed 325 (Department) whose associated SECTION_ADMIN
--      scope says 'SECTION' -- the candidate list for the Type correction,
--      with parent unit context so a human can sanity-check each one.
--   3. The reverse checks (units typed 324/323 with a scope that disagrees)
--      for completeness, so nothing else is hiding in the same pattern.
--   4. A summary count to compare against the ~59 units found during
--      investigation.
-- ============================================================================
SET NOCOUNT ON;

-- --- 1. Session state ---
-- Confirm this is the database your backend actually uses -- different
-- deployments of this app have used different DB names (this dev stack's
-- backend container is configured with DB_DATABASE=pfms, NOT the
-- 'IncidentManager' name that appears as the fallback default in
-- backend/config/db_settings.json), so check your backend's actual
-- DB_DATABASE env var / config rather than assuming a specific name here.
SELECT DB_NAME() AS current_database;

-- --- 2. Units typed 325 (Department) but staffed/scoped as SECTION (primary candidate list) ---
SELECT
    au.UniqueID       AS OrgUnitID,
    au.Name            AS OrgUnitName,
    au.Type            AS CurrentType,
    'DEPARTMENT (325)' AS CurrentTypeLabel,
    au.Frozen,
    parent.UniqueID    AS ParentID,
    parent.Name        AS ParentName,
    parent.Type        AS ParentType,
    u.Username         AS SectionAdminUsername,
    urs.OrgUnitType    AS ScopeSaysType,
    r.RoleCode
FROM dbo.AdminsrationUnit au
INNER JOIN dbo.APP_UserRoleScope urs
    ON urs.OrgUnitID = au.UniqueID AND urs.OrgUnitType = 'SECTION'
INNER JOIN dbo.APP_Roles r
    ON r.RoleID = urs.RoleID AND r.RoleCode = 'SECTION_ADMIN'
INNER JOIN dbo.APP_Users u
    ON u.UserID = urs.UserID
LEFT JOIN dbo.AdminsrationUnit parent
    ON parent.UniqueID = au.ParentID
WHERE au.Type = 325
  AND au.UniqueID NOT IN (22, 138, 167, 168, 172, 177)
ORDER BY au.Name;

-- --- 3a. Reverse check: units typed 324 (Section) but scoped as DEPARTMENT or ADMINISTRATION ---
SELECT
    au.UniqueID AS OrgUnitID, au.Name AS OrgUnitName, au.Type AS CurrentType,
    u.Username AS AdminUsername, urs.OrgUnitType AS ScopeSaysType, r.RoleCode
FROM dbo.AdminsrationUnit au
INNER JOIN dbo.APP_UserRoleScope urs ON urs.OrgUnitID = au.UniqueID AND urs.OrgUnitType IN ('DEPARTMENT', 'ADMINISTRATION')
INNER JOIN dbo.APP_Roles r ON r.RoleID = urs.RoleID
INNER JOIN dbo.APP_Users u ON u.UserID = urs.UserID
WHERE au.Type = 324
  AND au.UniqueID NOT IN (22, 138, 167, 168, 172, 177)
ORDER BY au.Name;

-- --- 3b. Reverse check: units typed 323 (Administration) but scoped as DEPARTMENT or SECTION ---
SELECT
    au.UniqueID AS OrgUnitID, au.Name AS OrgUnitName, au.Type AS CurrentType,
    u.Username AS AdminUsername, urs.OrgUnitType AS ScopeSaysType, r.RoleCode
FROM dbo.AdminsrationUnit au
INNER JOIN dbo.APP_UserRoleScope urs ON urs.OrgUnitID = au.UniqueID AND urs.OrgUnitType IN ('DEPARTMENT', 'SECTION')
INNER JOIN dbo.APP_Roles r ON r.RoleID = urs.RoleID
INNER JOIN dbo.APP_Users u ON u.UserID = urs.UserID
WHERE au.Type = 323
  AND au.UniqueID NOT IN (22, 138, 167, 168, 172, 177)
ORDER BY au.Name;

-- --- 4. Summary counts ---
SELECT
    (SELECT COUNT(DISTINCT au.UniqueID)
     FROM dbo.AdminsrationUnit au
     INNER JOIN dbo.APP_UserRoleScope urs ON urs.OrgUnitID = au.UniqueID AND urs.OrgUnitType = 'SECTION'
     INNER JOIN dbo.APP_Roles r ON r.RoleID = urs.RoleID AND r.RoleCode = 'SECTION_ADMIN'
     WHERE au.Type = 325 AND au.UniqueID NOT IN (22, 138, 167, 168, 172, 177)
    ) AS candidates_325_to_324,
    (SELECT COUNT(DISTINCT au.UniqueID)
     FROM dbo.AdminsrationUnit au
     INNER JOIN dbo.APP_UserRoleScope urs ON urs.OrgUnitID = au.UniqueID AND urs.OrgUnitType IN ('DEPARTMENT', 'ADMINISTRATION')
     WHERE au.Type = 324 AND au.UniqueID NOT IN (22, 138, 167, 168, 172, 177)
    ) AS reverse_mismatches_324,
    (SELECT COUNT(DISTINCT au.UniqueID)
     FROM dbo.AdminsrationUnit au
     INNER JOIN dbo.APP_UserRoleScope urs ON urs.OrgUnitID = au.UniqueID AND urs.OrgUnitType IN ('DEPARTMENT', 'SECTION')
     WHERE au.Type = 323 AND au.UniqueID NOT IN (22, 138, 167, 168, 172, 177)
    ) AS reverse_mismatches_323;
