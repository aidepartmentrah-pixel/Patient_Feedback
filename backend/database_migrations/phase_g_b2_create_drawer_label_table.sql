/*
================================================================================
Phase G-B2: Drawer Labels - Create Drawer_Label Table
================================================================================
Purpose: Create the APP_DrawerLabel table for multi-label tagging system
Author: Phase G Implementation
Date: 2026-02-07
Version: 1.0

IMPORTANT: Safe to run multiple times (uses IF NOT EXISTS checks)

Table Created:
- APP_DrawerLabel - Stores global reusable labels for drawer notes

Design Decisions:
- Labels are global (not org-scoped)
- Flat labels (no hierarchy)
- Labels are reusable across all notes
- label_name must be UNIQUE
- Labels can be disabled via is_active flag
- No optional metadata columns

Access Control:
- SOFTWARE_ADMIN
- COMPLAINT_SUPERVISOR
- WORKER

================================================================================
*/

BEGIN TRANSACTION;

BEGIN TRY

    PRINT '========================================';
    PRINT 'Phase G-B2: Creating Drawer Label Table';
    PRINT '========================================';
    PRINT '';

    -- ========================================================================
    -- 1) APP_DrawerLabel Table
    -- ========================================================================
    -- Stores reusable labels for tagging drawer notes
    
    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'APP_DrawerLabel' AND schema_id = SCHEMA_ID('dbo'))
    BEGIN
        PRINT 'Creating table: APP_DrawerLabel...';
        
        CREATE TABLE dbo.APP_DrawerLabel (
            -- Primary Key
            LabelID INT IDENTITY(1,1) PRIMARY KEY,
            
            -- Label Name (must be unique)
            LabelName NVARCHAR(100) NOT NULL,
            
            -- Active Flag
            IsActive BIT NOT NULL DEFAULT 1,
            
            -- Audit Field
            CreatedAt DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME(),
            
            -- Unique constraint on label name (prevent duplicates)
            CONSTRAINT UQ_DrawerLabel_LabelName UNIQUE (LabelName),
            
            -- Index for filtering active labels
            INDEX IX_DrawerLabel_IsActive NONCLUSTERED (IsActive)
        );
        
        PRINT '✓ APP_DrawerLabel table created successfully';
        PRINT '';
    END
    ELSE
    BEGIN
        PRINT '✓ APP_DrawerLabel table already exists';
        PRINT '';
    END

    -- ========================================================================
    -- Verification Query
    -- ========================================================================
    PRINT 'Verifying table structure...';
    PRINT '';
    
    SELECT 
        COLUMN_NAME,
        DATA_TYPE,
        CHARACTER_MAXIMUM_LENGTH,
        IS_NULLABLE,
        COLUMN_DEFAULT
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = 'APP_DrawerLabel'
    ORDER BY ORDINAL_POSITION;
    
    PRINT '';
    PRINT 'Verifying constraints...';
    PRINT '';
    
    SELECT 
        CONSTRAINT_NAME,
        CONSTRAINT_TYPE
    FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS
    WHERE TABLE_NAME = 'APP_DrawerLabel';
    
    PRINT '';
    PRINT '========================================';
    PRINT 'Phase G-B2: Completed Successfully';
    PRINT '========================================';
    
    COMMIT TRANSACTION;
    PRINT 'Transaction committed.';

END TRY
BEGIN CATCH
    
    ROLLBACK TRANSACTION;
    
    PRINT '';
    PRINT '========================================';
    PRINT 'ERROR: Transaction rolled back';
    PRINT '========================================';
    PRINT 'Error Message: ' + ERROR_MESSAGE();
    PRINT 'Error Line: ' + CAST(ERROR_LINE() AS NVARCHAR(10));
    PRINT '';
    
    THROW;
    
END CATCH;

GO

/*
================================================================================
ROLLBACK SCRIPT (if needed)
================================================================================
-- Uncomment and run if you need to remove the table

-- DROP TABLE IF EXISTS dbo.APP_DrawerLabel;
-- PRINT 'APP_DrawerLabel table dropped';

================================================================================
*/
