/*
================================================================================
EMAIL NOTIFICATION FEATURE — Add Email Column to APP_Users
================================================================================
Purpose: Add Email field to APP_Users table for notification system.

Author: System
Date: 2026-02-18
Version: 1.0

SAFE TO RUN MULTIPLE TIMES (idempotent checks included)

Changes:
- Add Email column to APP_Users (NVARCHAR(255), nullable)

Note: No unique constraint yet - will be added later if needed.
================================================================================
*/

BEGIN TRANSACTION;

BEGIN TRY

    PRINT '========================================';
    PRINT 'Email Notification Feature: Add Email Column';
    PRINT '========================================';
    PRINT '';

    -- ========================================================================
    -- Add Email column
    -- ========================================================================
    
    IF NOT EXISTS (
        SELECT * FROM sys.columns 
        WHERE object_id = OBJECT_ID('dbo.APP_Users') 
        AND name = 'Email'
    )
    BEGIN
        PRINT 'Adding column: Email...';
        
        ALTER TABLE dbo.APP_Users
        ADD Email NVARCHAR(255) NULL;
        
        PRINT '✓ Email column added successfully';
        PRINT '';
    END
    ELSE
    BEGIN
        PRINT '✓ Email column already exists';
        PRINT '';
    END

    -- ========================================================================
    -- Commit Transaction
    -- ========================================================================
    
    COMMIT TRANSACTION;
    
    PRINT '';
    PRINT '========================================';
    PRINT '✓ Email column migration completed successfully';
    PRINT '========================================';
    PRINT '';

END TRY
BEGIN CATCH
    
    IF @@TRANCOUNT > 0
        ROLLBACK TRANSACTION;
    
    PRINT '';
    PRINT '========================================';
    PRINT '✗ ERROR: Email column migration failed';
    PRINT '========================================';
    PRINT '';
    PRINT 'Error Number: ' + CAST(ERROR_NUMBER() AS NVARCHAR(10));
    PRINT 'Error Message: ' + ERROR_MESSAGE();
    PRINT 'Error Line: ' + CAST(ERROR_LINE() AS NVARCHAR(10));
    PRINT '';

END CATCH
