-- ============================================================================
-- Retypes org units from 325 (Department) to 324 (Section) where the unit's
-- own SECTION_ADMIN account was set up with APP_UserRoleScope.OrgUnitType =
-- 'SECTION' -- e.g. "الخدمات البيئية" (id 64), "الهندسة الطبية" (id 73), and
-- ~54 others with the same pattern (56 total, confirmed against the live
-- dev DB via diagnose_org_unit_type_mismatch.sql on 2026-09-17).
--
-- Does NOT touch the 6 known orphaned units (22, 138, 167, 168, 172, 177) --
-- unrelated issue, excluded explicitly below as a second safety net.
--
-- This file is meant to be run top-to-bottom AS-IS, no editing required --
-- earlier versions used DECLARE'd variables (a table variable, then a
-- @ConfirmFix flag) to gate/coordinate the fix, and a version after that
-- required manually uncommenting a block; both were fragile or error-prone
-- under DBeaver, which sends each top-level statement (even ones nested
-- inside a bare BEGIN...END) to SQL Server as its own separate batch, so
-- nothing here depends on state shared between statements. It's also safe
-- to re-run: every statement's WHERE clause only ever matches rows still
-- typed 325, so once a unit is fixed, later runs simply skip it.
--
-- Review diagnose_org_unit_type_mismatch.sql section 2 before running this
-- (Step 1 below reprints the same preview so you see it again right here).
--
-- Rollback if needed (uses the FixedAt timestamp this run's INSERT wrote --
-- check dbo.AdminsrationUnit_TypeFix_Backup for the exact value):
--   UPDATE au SET au.Type = b.OldType
--   FROM dbo.AdminsrationUnit au
--   JOIN dbo.AdminsrationUnit_TypeFix_Backup b ON b.OrgUnitID = au.UniqueID
--   WHERE b.FixedAt >= '<timestamp>';
-- ============================================================================
SET NOCOUNT ON;

-- --- Sanity check: confirm this is the database your backend actually uses.
-- Different deployments of this app have used different DB names (this dev
-- stack's backend container is configured with DB_DATABASE=pfms, NOT the
-- 'IncidentManager' name that appears as the fallback default in
-- backend/config/db_settings.json) -- check your backend's actual DB_DATABASE
-- env var / config rather than assuming a specific name here. ---
SELECT DB_NAME() AS current_database;

-- --- Step 1: preview exactly what's about to change ---
SELECT au.UniqueID, au.Name, au.Type AS CurrentType
FROM dbo.AdminsrationUnit au
WHERE au.Type = 325
  AND au.UniqueID NOT IN (22, 138, 167, 168, 172, 177)
  AND EXISTS (
        SELECT 1
        FROM dbo.APP_UserRoleScope urs
        INNER JOIN dbo.APP_Roles r ON r.RoleID = urs.RoleID AND r.RoleCode = 'SECTION_ADMIN'
        WHERE urs.OrgUnitID = au.UniqueID AND urs.OrgUnitType = 'SECTION'
  )
ORDER BY au.Name;

-- --- Step 2: ensure the backup table exists (idempotent) ---
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'AdminsrationUnit_TypeFix_Backup' AND schema_id = SCHEMA_ID('dbo'))
BEGIN
    CREATE TABLE dbo.AdminsrationUnit_TypeFix_Backup (
        BackupID    INT IDENTITY(1,1) PRIMARY KEY,
        OrgUnitID   INT NOT NULL,
        OrgUnitName NVARCHAR(50) NULL,
        OldType     INT NULL,
        NewType     INT NOT NULL,
        FixedAt     DATETIME NOT NULL DEFAULT GETDATE()
    );
END

-- --- Step 3: back up the rows about to change ---
INSERT INTO dbo.AdminsrationUnit_TypeFix_Backup (OrgUnitID, OrgUnitName, OldType, NewType)
SELECT au.UniqueID, au.Name, au.Type, 324
FROM dbo.AdminsrationUnit au
WHERE au.Type = 325
  AND au.UniqueID NOT IN (22, 138, 167, 168, 172, 177)
  AND EXISTS (
        SELECT 1
        FROM dbo.APP_UserRoleScope urs
        INNER JOIN dbo.APP_Roles r ON r.RoleID = urs.RoleID AND r.RoleCode = 'SECTION_ADMIN'
        WHERE urs.OrgUnitID = au.UniqueID AND urs.OrgUnitType = 'SECTION'
  );

-- --- Step 4: the actual fix ---
UPDATE au
SET au.Type = 324,
    au.UpdateDate = GETDATE()
FROM dbo.AdminsrationUnit au
WHERE au.Type = 325
  AND au.UniqueID NOT IN (22, 138, 167, 168, 172, 177)
  AND EXISTS (
        SELECT 1
        FROM dbo.APP_UserRoleScope urs
        INNER JOIN dbo.APP_Roles r ON r.RoleID = urs.RoleID AND r.RoleCode = 'SECTION_ADMIN'
        WHERE urs.OrgUnitID = au.UniqueID AND urs.OrgUnitType = 'SECTION'
  );

-- --- Step 5: verify -- should return 0 ---
SELECT COUNT(*) AS remaining_325_mismatches
FROM dbo.AdminsrationUnit au
WHERE au.Type = 325
  AND au.UniqueID NOT IN (22, 138, 167, 168, 172, 177)
  AND EXISTS (
        SELECT 1
        FROM dbo.APP_UserRoleScope urs
        INNER JOIN dbo.APP_Roles r ON r.RoleID = urs.RoleID AND r.RoleCode = 'SECTION_ADMIN'
        WHERE urs.OrgUnitID = au.UniqueID AND urs.OrgUnitType = 'SECTION'
  );
