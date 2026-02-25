-- ============================================================================
-- MODULE 5.2 — Bulk Admin User Generator
-- ============================================================================
-- Purpose: Create one admin user per organizational unit (Type 323/324/325)
-- Safe to run multiple times (idempotent)
-- ============================================================================

SET NOCOUNT ON;

BEGIN TRANSACTION;

BEGIN TRY

    -- ========================================================================
    -- STEP 1: Build derived org dataset with computed usernames and roles
    -- ========================================================================
    
    WITH OrgUnitsToProcess AS (
        SELECT 
            UniqueID AS org_unit_id,
            Type AS org_unit_type,
            Name AS org_unit_name,
            CASE 
                WHEN Type = 323 THEN 'adm_' + CAST(UniqueID AS NVARCHAR(20)) + '_admin'
                WHEN Type = 325 THEN 'dept_' + CAST(UniqueID AS NVARCHAR(20)) + '_admin'
                WHEN Type = 324 THEN 'sec_' + CAST(UniqueID AS NVARCHAR(20)) + '_admin'
            END AS username,
            CASE 
                WHEN Type = 323 THEN 'ADMINISTRATION_ADMIN'
                WHEN Type = 325 THEN 'DEPARTMENT_ADMIN'
                WHEN Type = 324 THEN 'SECTION_ADMIN'
            END AS role_code,
            CASE 
                WHEN Type = 323 THEN 'ADMINISTRATION'
                WHEN Type = 325 THEN 'DEPARTMENT'
                WHEN Type = 324 THEN 'SECTION'
            END AS org_unit_type_label
        FROM dbo.AdminsrationUnit
        WHERE Type IN (323, 324, 325)
          AND Frozen = 0
    )
    
    -- ========================================================================
    -- STEP 2: Insert missing users only (NOT EXISTS check)
    -- ========================================================================
    
    INSERT INTO dbo.APP_Users (Username, PasswordHash, IsActive, CreatedAt)
    SELECT 
        org.username,
        'TEMP_HASH_Hospital2026!' AS PasswordHash,
        1 AS IsActive,
        GETDATE() AS CreatedAt
    FROM OrgUnitsToProcess org
    WHERE NOT EXISTS (
        SELECT 1 
        FROM dbo.APP_Users u 
        WHERE u.Username = org.username
    );
    
    -- ========================================================================
    -- STEP 3: Resolve UserID and RoleID for scope insertion
    -- ========================================================================
    
    WITH OrgUnitsToProcess AS (
        SELECT 
            UniqueID AS org_unit_id,
            Type AS org_unit_type,
            CASE 
                WHEN Type = 323 THEN 'adm_' + CAST(UniqueID AS NVARCHAR(20)) + '_admin'
                WHEN Type = 325 THEN 'dept_' + CAST(UniqueID AS NVARCHAR(20)) + '_admin'
                WHEN Type = 324 THEN 'sec_' + CAST(UniqueID AS NVARCHAR(20)) + '_admin'
            END AS username,
            CASE 
                WHEN Type = 323 THEN 'ADMINISTRATION_ADMIN'
                WHEN Type = 325 THEN 'DEPARTMENT_ADMIN'
                WHEN Type = 324 THEN 'SECTION_ADMIN'
            END AS role_code,
            CASE 
                WHEN Type = 323 THEN 'ADMINISTRATION'
                WHEN Type = 325 THEN 'DEPARTMENT'
                WHEN Type = 324 THEN 'SECTION'
            END AS org_unit_type_label
        FROM dbo.AdminsrationUnit
        WHERE Type IN (323, 324, 325)
          AND Frozen = 0
    ),
    UsersWithRoles AS (
        SELECT 
            org.org_unit_id,
            org.org_unit_type,
            org.org_unit_type_label,
            u.UserID,
            r.RoleID
        FROM OrgUnitsToProcess org
        INNER JOIN dbo.APP_Users u ON u.Username = org.username
        INNER JOIN dbo.APP_Roles r ON r.RoleCode = org.role_code
    )
    
    -- ========================================================================
    -- STEP 4: Insert missing scopes only (NOT EXISTS check)
    -- ========================================================================
    
    INSERT INTO dbo.APP_UserRoleScope (UserID, RoleID, OrgUnitID, OrgUnitType)
    SELECT 
        uwr.UserID,
        uwr.RoleID,
        uwr.org_unit_id,
        uwr.org_unit_type_label
    FROM UsersWithRoles uwr
    WHERE NOT EXISTS (
        SELECT 1 
        FROM dbo.APP_UserRoleScope urs
        WHERE urs.UserID = uwr.UserID
          AND urs.RoleID = uwr.RoleID
          AND urs.OrgUnitID = uwr.org_unit_id
    );

    -- ========================================================================
    -- Success summary
    -- ========================================================================
    
    DECLARE @UserCount INT = (SELECT COUNT(*) FROM dbo.APP_Users WHERE Username LIKE 'adm_%_admin' OR Username LIKE 'dept_%_admin' OR Username LIKE 'sec_%_admin');
    DECLARE @ScopeCount INT = (SELECT COUNT(*) FROM dbo.APP_UserRoleScope urs INNER JOIN dbo.APP_Users u ON urs.UserID = u.UserID WHERE u.Username LIKE 'adm_%_admin' OR u.Username LIKE 'dept_%_admin' OR u.Username LIKE 'sec_%_admin');
    
    PRINT '============================================================================';
    PRINT 'MODULE 5.2 — Bulk Admin User Generation COMPLETE';
    PRINT '============================================================================';
    PRINT 'Total admin users: ' + CAST(@UserCount AS NVARCHAR(10));
    PRINT 'Total role scopes: ' + CAST(@ScopeCount AS NVARCHAR(10));
    PRINT 'Password: TEMP_HASH_Hospital2026!';
    PRINT 'All accounts active: IsActive = 1';
    PRINT '============================================================================';

    COMMIT TRANSACTION;
    
END TRY
BEGIN CATCH
    
    ROLLBACK TRANSACTION;
    
    DECLARE @ErrorMessage NVARCHAR(4000) = ERROR_MESSAGE();
    DECLARE @ErrorSeverity INT = ERROR_SEVERITY();
    DECLARE @ErrorState INT = ERROR_STATE();
    
    PRINT '============================================================================';
    PRINT 'ERROR: Script failed';
    PRINT '============================================================================';
    PRINT 'Message: ' + @ErrorMessage;
    PRINT '============================================================================';
    
    RAISERROR(@ErrorMessage, @ErrorSeverity, @ErrorState);
    
END CATCH;

SET NOCOUNT OFF;
