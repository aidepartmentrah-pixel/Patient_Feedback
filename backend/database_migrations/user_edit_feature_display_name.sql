/*
================================================================================
USER EDIT FEATURE: Display Name Column
================================================================================
Purpose: Ensure display_name column exists in APP_Users table for user editing
         feature in Settings page.

Author: System
Date: 2026-02-11
Version: 1.0

SAFE TO RUN MULTIPLE TIMES (idempotent checks included)

Note: This migration is provided for reference. The DisplayName column was
      already added in phase_a_step1_extend_user_table.sql. This file documents
      its usage for the user editing feature.

Changes:
- Verify DisplayName column exists in APP_Users table
- Set default display names for existing users without one

Related Files:
- backend/api/routers/admin_user_management_router.py (PUT /api/admin/users/{user_id})
- backend/api/services/user_management_service.py (update_user_service)
- backend/api/db_layer/user_management_db.py (update_user_credentials)
================================================================================
*/

BEGIN TRANSACTION;

BEGIN TRY

    PRINT '========================================';
    PRINT 'User Edit Feature: Display Name Column';
    PRINT '========================================';
    PRINT '';

    -- ========================================================================
    -- Verify DisplayName column exists
    -- ========================================================================
    
    IF EXISTS (
        SELECT * FROM sys.columns 
        WHERE object_id = OBJECT_ID('dbo.APP_Users') 
        AND name = 'DisplayName'
    )
    BEGIN
        PRINT '✓ DisplayName column already exists';
        PRINT '';
    END
    ELSE
    BEGIN
        PRINT 'Adding column: DisplayName...';
        
        ALTER TABLE dbo.APP_Users
        ADD DisplayName NVARCHAR(150) NULL;
        
        PRINT '✓ DisplayName column added successfully';
        PRINT '';
    END

    -- ========================================================================
    -- Set default display names for existing users
    -- ========================================================================
    
    PRINT 'Setting default display names for users without one...';
    
    UPDATE u
    SET DisplayName = CASE 
        WHEN r.RoleCode = 'SOFTWARE_ADMIN' THEN 'System Administrator'
        WHEN a.Name IS NOT NULL THEN a.Name + ' Admin'
        ELSE u.Username
    END
    FROM dbo.APP_Users u
    LEFT JOIN dbo.APP_UserRoleScope urs ON u.UserID = urs.UserID
    LEFT JOIN dbo.APP_Roles r ON urs.RoleID = r.RoleID
    LEFT JOIN dbo.AdminsrationUnit a ON urs.OrgUnitID = a.UniqueID
    WHERE u.DisplayName IS NULL OR u.DisplayName = '';
    
    PRINT '✓ Default display names set';
    PRINT '';

    -- ========================================================================
    -- Commit Transaction
    -- ========================================================================
    
    COMMIT TRANSACTION;
    
    PRINT '';
    PRINT '========================================';
    PRINT '✓ User Edit Feature migration completed';
    PRINT '========================================';
    PRINT '';

END TRY
BEGIN CATCH
    
    IF @@TRANCOUNT > 0
        ROLLBACK TRANSACTION;
    
    PRINT '';
    PRINT '========================================';
    PRINT '✗ ERROR: Migration failed';
    PRINT '========================================';
    PRINT 'Error: ' + ERROR_MESSAGE();
    PRINT 'Line: ' + CAST(ERROR_LINE() AS NVARCHAR(10));
    PRINT '';
    
    THROW;
    
END CATCH;

GO
