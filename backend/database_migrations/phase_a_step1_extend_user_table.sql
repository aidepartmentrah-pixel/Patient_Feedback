/*
================================================================================
PHASE A — STEP 1: EXTEND USER TABLE SCHEMA
================================================================================
Purpose: Add display name fields to APP_Users table for person greeting 
         and department display in UI.

Author: System
Date: 2026-02-05
Version: 1.0

SAFE TO RUN MULTIPLE TIMES (idempotent checks included)

Changes:
- Add DisplayName column to APP_Users
- Add DepartmentDisplayName column to APP_Users

Both columns are nullable for backward compatibility.
================================================================================
*/

BEGIN TRANSACTION;

BEGIN TRY

    PRINT '========================================';
    PRINT 'Phase A - Step 1: Extend User Table Schema';
    PRINT '========================================';
    PRINT '';

    -- ========================================================================
    -- Add DisplayName column
    -- ========================================================================
    
    IF NOT EXISTS (
        SELECT * FROM sys.columns 
        WHERE object_id = OBJECT_ID('dbo.APP_Users') 
        AND name = 'DisplayName'
    )
    BEGIN
        PRINT 'Adding column: DisplayName...';
        
        ALTER TABLE dbo.APP_Users
        ADD DisplayName NVARCHAR(150) NULL;
        
        PRINT '✓ DisplayName column added successfully';
        PRINT '';
    END
    ELSE
    BEGIN
        PRINT '✓ DisplayName column already exists';
        PRINT '';
    END

    -- ========================================================================
    -- Add DepartmentDisplayName column
    -- ========================================================================
    
    IF NOT EXISTS (
        SELECT * FROM sys.columns 
        WHERE object_id = OBJECT_ID('dbo.APP_Users') 
        AND name = 'DepartmentDisplayName'
    )
    BEGIN
        PRINT 'Adding column: DepartmentDisplayName...';
        
        ALTER TABLE dbo.APP_Users
        ADD DepartmentDisplayName NVARCHAR(150) NULL;
        
        PRINT '✓ DepartmentDisplayName column added successfully';
        PRINT '';
    END
    ELSE
    BEGIN
        PRINT '✓ DepartmentDisplayName column already exists';
        PRINT '';
    END

    -- ========================================================================
    -- Commit Transaction
    -- ========================================================================
    
    COMMIT TRANSACTION;
    
    PRINT '';
    PRINT '========================================';
    PRINT '✓ Phase A Step 1 completed successfully';
    PRINT '========================================';
    PRINT '';

END TRY
BEGIN CATCH
    
    IF @@TRANCOUNT > 0
        ROLLBACK TRANSACTION;
    
    PRINT '';
    PRINT '========================================';
    PRINT '✗ ERROR: Phase A Step 1 failed';
    PRINT '========================================';
    PRINT 'Error: ' + ERROR_MESSAGE();
    PRINT 'Line: ' + CAST(ERROR_LINE() AS NVARCHAR(10));
    PRINT '';
    
    THROW;
    
END CATCH;

GO
